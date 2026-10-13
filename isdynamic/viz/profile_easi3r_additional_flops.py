from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/xiaoang/FF_Reconstruction/isDynamicProbe")
REPO_ROOT = Path("/home/xiaoang/FF_Reconstruction/Easi3R")
SAM2_ROOT = REPO_ROOT / "third_party" / "sam2"

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from isdynamic.viz.unseen_examples_common import resolve_sequence_image_paths, select_frame_window  # noqa: E402
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
load_images = _image_utils.load_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Easi3R additional motion FLOPs per frame.")
    parser.add_argument("--dataset", type=str, choices=["davis", "segtrackv2"], required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--weights", type=str, default="/home/xiaoang/FF_Reconstruction/Easi3R/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--davis_root", type=str, default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS")
    parser.add_argument("--segtrack_root", type=str, default="/home/xiaoang/Datasets/SegTrackv2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1)
    parser.add_argument("--flow_loss_threshold", type=float, default=25.0)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def _prepare_filelist(image_paths: list[Path], stem: str) -> tuple[list[str], contextlib.AbstractContextManager]:
    supported = {".jpg", ".jpeg", ".png"}
    if all(path.suffix.lower() in supported for path in image_paths):
        return [str(path) for path in image_paths], contextlib.nullcontext()

    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"easi3r_prof_{stem}_")
    tmp_dir = Path(tmp_ctx.__enter__())
    from PIL import Image
    for idx, src in enumerate(image_paths):
        dst = tmp_dir / f"{idx:05d}.png"
        Image.open(src).convert("RGB").save(dst)
    return [str(path) for path in sorted(tmp_dir.glob("*.png"))], tmp_ctx


def _scene_graph() -> str:
    return "swinstride-5-noncyclic"


def _sum_profile_flops(prof: torch.profiler.profile) -> int:
    total = 0
    for evt in prof.key_averages():
        total += int(getattr(evt, "flops", 0) or 0)
    return total


def main() -> None:
    args = parse_args()
    device = args.device
    if device != "cuda":
        raise RuntimeError("This profiler currently expects CUDA to be available.")

    image_paths = resolve_sequence_image_paths(
        dataset=args.dataset,
        sequence=args.sequence,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
    )
    image_paths, frame_idx = select_frame_window(image_paths, args.frame_idx, args.window_size)
    num_frames = len(image_paths)

    filelist, file_ctx = _prepare_filelist(image_paths, f"{args.sequence}_{frame_idx}")
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

    torch.serialization.add_safe_globals([argparse.Namespace])
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(device)
    model.eval()

    pairs = make_pairs(imgs, scene_graph=_scene_graph(), prefilter=None, symmetrize=True)

    # Base backbone pass; excluded from additional FLOPs.
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
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
            del scene
            for i, img in enumerate(imgs):
                img["atten_mask"] = atten_masks[i].cpu().unsqueeze(0)

            pairs2 = make_pairs(imgs, scene_graph=_scene_graph(), prefilter=None, symmetrize=True)
            output2 = inference(pairs2, model, device, batch_size=1, verbose=False)

            scene2 = global_aligner(
                output2,
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
            _dynamic_masks = scene2.dynamic_masks
            del scene2
            torch.cuda.synchronize()

    total_flops = _sum_profile_flops(prof)
    per_frame_flops = total_flops / max(num_frames, 1)
    peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))

    result = {
        "method": "Easi3R",
        "dataset": args.dataset,
        "sequence": args.sequence,
        "frame_idx": args.frame_idx,
        "window_size": args.window_size,
        "num_frames": num_frames,
        "pair_count": len(pairs),
        "additional_flops_total": total_flops,
        "additional_flops_per_frame": per_frame_flops,
        "peak_cuda_mem_bytes": peak_mem_bytes,
        "note": "Additional-only FLOPs exclude the first inference and include first global_aligner attention-mask pass, second inference with atten_mask, and second global_aligner. torch.profiler only counts supported ops.",
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
