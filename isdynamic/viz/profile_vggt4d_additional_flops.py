from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

REPO_ROOT = Path("/home/xiaoang/FF_Reconstruction/VGGT4D")
PROJECT_ROOT = Path("/home/xiaoang/FF_Reconstruction/isDynamicProbe")
for _path in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

importlib.import_module("vggt")
importlib.import_module("vggt.layers")
importlib.import_module("vggt.layers.attention")

from isdynamic.viz.unseen_examples_common import resolve_sequence_image_paths, select_frame_window
from vggt4d.masks.dynamic_mask import adaptive_multiotsu_variance, cluster_attention_maps, extract_dyn_map  # noqa: E402
from vggt4d.masks.refine_dyn_mask import RefineDynMask  # noqa: E402
from vggt4d.models.vggt4d import VGGTFor4D  # noqa: E402
from vggt4d.utils.model_utils import inference, organize_qk_dict  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VGGT4D additional motion FLOPs per frame.")
    parser.add_argument("--dataset", type=str, choices=["davis", "segtrackv2"], required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="/home/xiaoang/FF_Reconstruction/VGGT4D/ckpts/model_tracker_fixed_e20.pt")
    parser.add_argument("--davis_root", type=str, default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS")
    parser.add_argument("--segtrack_root", type=str, default="/home/xiaoang/Datasets/SegTrackv2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def _sum_profile_flops(prof: torch.profiler.profile) -> int:
    total = 0
    for evt in prof.key_averages():
        total += int(getattr(evt, "flops", 0) or 0)
    return total


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This profiler currently expects CUDA to be available.")

    image_paths = resolve_sequence_image_paths(
        dataset=args.dataset,
        sequence=args.sequence,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
    )
    image_paths, frame_idx = select_frame_window(image_paths, args.frame_idx, args.window_size)
    num_frames = len(image_paths)

    model = VGGTFor4D()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)

    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

    # Base backbone pass to materialize features; excluded from additional FLOPs.
    with torch.inference_mode():
        with amp_ctx:
            predictions1, qk_dict, enc_feat, _agg_tokens_list = inference(model, images)
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with amp_ctx:
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

                predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

                pred_intrinsic = predictions1["intrinsic"]
                pred_cam2world = predictions2["cam2world"]
                pred_depths = predictions1["depth"]
                pred_conf = predictions1["depth_conf"]
                _ = pred_conf

                refiner = RefineDynMask(
                    images,
                    torch.tensor(pred_depths).to(device),
                    dyn_masks.to(device),
                    torch.tensor(pred_cam2world).float().to(device),
                    torch.tensor(pred_intrinsic).to(device),
                    device,
                )
                _refined_mask = refiner.refine_masks()
                del refiner
            torch.cuda.synchronize(device)

    total_flops = _sum_profile_flops(prof)
    per_frame_flops = total_flops / max(num_frames, 1)
    peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))

    result = {
        "method": "VGGT4D",
        "dataset": args.dataset,
        "sequence": args.sequence,
        "frame_idx": args.frame_idx,
        "window_size": args.window_size,
        "num_frames": num_frames,
        "additional_flops_total": total_flops,
        "additional_flops_per_frame": per_frame_flops,
        "peak_cuda_mem_bytes": peak_mem_bytes,
        "note": "Additional-only FLOPs exclude the first backbone inference and include dynamic-map extraction, thresholding, second masked inference, and mask refinement. torch.profiler only counts supported ops.",
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
