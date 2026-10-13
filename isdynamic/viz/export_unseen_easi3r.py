from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path("/home/xiaoang/FF_Reconstruction/isDynamicProbe")
REPO_ROOT = Path("/home/xiaoang/FF_Reconstruction/Easi3R")
SAM2_ROOT = REPO_ROOT / "third_party" / "sam2"
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from isdynamic.viz.unseen_examples_common import (
    EXAMPLE_SPECS,
    resolve_sequence_image_paths,
    save_binary_mask_png,
    save_heatmap_overlay_png,
    select_frame_window,
)

sys.path.remove(str(PROJECT_ROOT))
for _path in (str(REPO_ROOT), str(SAM2_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
sys.modules.pop("dust3r", None)

importlib.import_module("dust3r")
_cloud_opt = importlib.import_module("dust3r.cloud_opt")
_image_pairs = importlib.import_module("dust3r.image_pairs")
_inference_mod = importlib.import_module("dust3r.inference")
_model_mod = importlib.import_module("dust3r.model")
_image_utils = importlib.import_module("dust3r.utils.image")
GlobalAlignerMode = _cloud_opt.GlobalAlignerMode
global_aligner = _cloud_opt.global_aligner
make_pairs = _image_pairs.make_pairs
inference = _inference_mod.inference
AsymmetricCroCo3DStereo = _model_mod.AsymmetricCroCo3DStereo
crop_img = _image_utils.crop_img
load_images = _image_utils.load_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export minimal unseen-class outputs with Easi3R.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/isDynamicProbe/unseen_classes/easi3r",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS",
    )
    parser.add_argument(
        "--segtrack_root",
        type=str,
        default="/home/xiaoang/Datasets/SegTrackv2",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1)
    parser.add_argument("--flow_loss_threshold", type=float, default=25.0)
    return parser.parse_args()


def _scene_graph() -> str:
    return "swinstride-5-noncyclic"


def _prepare_filelist_for_easi3r(image_paths: list[Path], stem: str) -> tuple[list[str], contextlib.AbstractContextManager]:
    supported = {".jpg", ".jpeg", ".png"}
    if all(path.suffix.lower() in supported for path in image_paths):
        return [str(path) for path in image_paths], contextlib.nullcontext()

    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"easi3r_{stem}_")
    tmp_dir = Path(tmp_ctx.__enter__())
    for idx, src in enumerate(image_paths):
        dst = tmp_dir / f"{idx:05d}.png"
        Image.open(src).convert("RGB").save(dst)
    return [str(path) for path in sorted(tmp_dir.glob("*.png"))], tmp_ctx


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    device = args.device
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(device)
    model.eval()

    for spec in EXAMPLE_SPECS:
        if args.stem is not None and str(spec["stem"]) != args.stem:
            continue
        image_paths = resolve_sequence_image_paths(
            dataset=str(spec["dataset"]),
            sequence=str(spec["sequence"]),
            davis_root=args.davis_root,
            segtrack_root=args.segtrack_root,
        )
        frame_idx = int(spec["frame_idx"])
        if frame_idx < 0 or frame_idx >= len(image_paths):
            raise IndexError(f"frame_idx={frame_idx} out of range for {spec['sequence']} ({len(image_paths)} frames)")
        image_paths, frame_idx = select_frame_window(image_paths, frame_idx, args.window_size)

        filelist, file_ctx = _prepare_filelist_for_easi3r(image_paths, str(spec["stem"]))
        try:
            imgs, _width, _height, _video_fps = load_images(
                filelist,
                size=args.image_size,
                verbose=False,
                dynamic_mask_root=None,
                fps=0,
                num_frames=len(filelist),
                return_img_size=True,
            )
        finally:
            if not isinstance(file_ctx, contextlib.nullcontext):
                file_ctx.__exit__(None, None, None)
        if len(imgs) == 1:
            imgs = [imgs[0], dict(imgs[0])]
            imgs[1]["idx"] = 1

        pairs = make_pairs(imgs, scene_graph=_scene_graph(), prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=False)

        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=False,
            shared_focal=True,
            temporal_smoothing_weight=0.0,
            translation_weight=1.0,
            flow_loss_weight=0.0,
            flow_loss_start_epoch=args.flow_loss_start_iter,
            flow_loss_thre=args.flow_loss_threshold,
            use_self_mask=True,
            num_total_iter=args.niter,
            empty_cache=len(filelist) > 72,
            batchify=True,
            use_atten_mask=True,
            sam2_mask_refine=False,
        )
        atten_masks = scene.dynamic_masks
        del pairs
        del output
        del scene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, img in enumerate(imgs):
            img["atten_mask"] = atten_masks[i].cpu().unsqueeze(0)

        pairs = make_pairs(imgs, scene_graph=_scene_graph(), prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=False)
        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=False,
            shared_focal=True,
            temporal_smoothing_weight=0.0,
            translation_weight=1.0,
            flow_loss_weight=args.flow_loss_weight,
            flow_loss_start_epoch=args.flow_loss_start_iter,
            flow_loss_thre=args.flow_loss_threshold,
            use_self_mask=True,
            num_total_iter=args.niter,
            empty_cache=len(filelist) > 72,
            batchify=True,
            use_atten_mask=True,
            sam2_mask_refine=False,
        )
        stem = str(spec["stem"])
        attns = scene.get_atts()
        upsampled_attns = F.interpolate(
            attns.unsqueeze(1),
            size=scene.imshape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        with tempfile.TemporaryDirectory(prefix=f"easi3r_native_{stem}_") as native_tmp:
            native_dir = Path(native_tmp)
            scene.save_dynamic_masks(str(native_dir))
            native_mask = native_dir / f"dynamic_mask_{frame_idx}.png"
            if not native_mask.exists():
                raise FileNotFoundError(f"Missing native Easi3R mask frame: {native_mask}")

            save_binary_mask_png(np.array(Image.open(native_mask).convert("L")) > 0, out_dir / f"binary_mask_{stem}.png")
        save_heatmap_overlay_png(
            image_np=np.asarray(crop_img(Image.open(image_paths[frame_idx]).convert("RGB"), size=args.image_size, square_ok=False, crop=True), dtype=np.uint8),
            score_map=upsampled_attns[frame_idx].detach().cpu().numpy().astype(np.float32),
            out_path=out_dir / f"avg_score_upsampled_overlay_{stem}.png",
        )

        del pairs
        del output
        del scene
        del attns
        del upsampled_attns
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
