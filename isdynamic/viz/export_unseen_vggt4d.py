from __future__ import annotations

import argparse
import contextlib
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/home/xiaoang/FF_Reconstruction/VGGT4D")
PROJECT_ROOT = Path("/home/xiaoang/FF_Reconstruction/isDynamicProbe")
for _path in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

# Force the local VGGT package tree to load before VGGT4D imports it indirectly.
importlib.import_module("vggt")
importlib.import_module("vggt.layers")
importlib.import_module("vggt.layers.attention")

import torch
import torch.nn.functional as F
from einops import rearrange

from isdynamic.viz.unseen_examples_common import (
    EXAMPLE_SPECS,
    resolve_sequence_image_paths,
    save_heatmap_overlay_png,
    select_frame_window,
)

from vggt4d.masks.dynamic_mask import adaptive_multiotsu_variance, cluster_attention_maps, extract_dyn_map  # noqa: E402
from vggt4d.models.vggt4d import VGGTFor4D  # noqa: E402
from vggt4d.utils.store import save_dynamic_masks  # noqa: E402
from vggt4d.utils.model_utils import inference, organize_qk_dict  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export minimal unseen-class outputs with VGGT4D.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/isDynamicProbe/unseen_classes/vggt4d",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/VGGT4D/ckpts/model_tracker_fixed_e20.pt",
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
    return parser.parse_args()


def _tensor_to_uint8_image(image_t: torch.Tensor) -> torch.Tensor:
    return (image_t.detach().clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = VGGTFor4D()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    model = model.to(device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode():
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

            images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
            with amp_ctx:
                predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
            del predictions1
            del agg_tokens_list

            qk_dict = organize_qk_dict(qk_dict, images.shape[0])
            dyn_maps = extract_dyn_map(qk_dict, images)

            n_img, _c, h_img, w_img = images.shape
            h_tok, w_tok = h_img // 14, w_img // 14
            feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", n_img=n_img, h=h_tok, w=w_tok)
            norm_dyn_map, _labels = cluster_attention_maps(feat_map, dyn_maps)

            upsampled_map = F.interpolate(
                norm_dyn_map.unsqueeze(1),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            threshold = adaptive_multiotsu_variance(upsampled_map.detach().cpu().numpy())
            dyn_masks = upsampled_map > threshold

            image_np = _tensor_to_uint8_image(images[frame_idx]).numpy()
            heat_np = upsampled_map[frame_idx].detach().cpu().numpy().astype("float32")

            stem = str(spec["stem"])
            with tempfile.TemporaryDirectory(prefix=f"vggt4d_native_{stem}_") as native_tmp:
                native_dir = Path(native_tmp)
                save_dynamic_masks(native_dir, dyn_masks)
                native_mask = native_dir / f"dynamic_mask_{frame_idx:04d}.png"
                if not native_mask.exists():
                    raise FileNotFoundError(f"Missing native VGGT4D mask frame: {native_mask}")
                shutil.copy2(native_mask, out_dir / f"binary_mask_{stem}.png")
            save_heatmap_overlay_png(
                image_np=image_np,
                score_map=heat_np,
                out_path=out_dir / f"avg_score_upsampled_overlay_{stem}.png",
            )

            del images
            del qk_dict
            del enc_feat
            del dyn_maps
            del feat_map
            del norm_dyn_map
            del upsampled_map
            del dyn_masks
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
