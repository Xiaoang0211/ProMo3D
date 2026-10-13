from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image

from isdynamic.legacy.pi3.build_davis_isdynamic_npz import compute_resize_dims
from isdynamic.legacy.pi3.visualize_isdynamic import GROUP_TO_SOURCE, discover_run_dirs
from isdynamic.models.pi3 import PI3_PATCH_SIZE, PI3_TARGET_SIZE, extract_feature_map, load_pi3_family_model
from isdynamic.viz.motion_mask_benchmarks import (
    load_davis_sequence_paths,
    resolve_davis_root,
    select_context_indices,
)
from isdynamic.viz.motion_mask_export_common import get_cached_probe, score_single_stream_blocks
from isdynamic.viz.motion_mask_utils import binary_mask_from_score, synchronize_device
from isdynamic.viz.scoremap_utils import crop_to_content_bbox, upsample_patch_map


@dataclass(frozen=True)
class FrameSpec:
    sequence: str
    frame_idx: int
    image_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile batch-1 Pi3X backbone and PMD coarse-mask latency on fixed DAVIS frames."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--davis_root", type=Path, default=Path("/home/xiaoang/Datasets/DAVIS-2017-trainval-480p"))
    parser.add_argument("--sequence", type=str, default="parkour")
    parser.add_argument("--warmup_sequence", type=str, default="cows")
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--context_stride", type=int, default=2)
    parser.add_argument("--group", type=str, default="inter_frame_block")
    parser.add_argument("--score_batch_size", type=int, default=262144)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/experiments/runtime/pi3x"))
    parser.add_argument("--save_masks", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def preprocess_rgb(image_rgb: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(image_rgb)
    width, height = image.size
    new_w, new_h = compute_resize_dims(width, height)
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    pad_w = PI3_TARGET_SIZE - new_w
    pad_h = PI3_TARGET_SIZE - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    image_arr = np.asarray(image, dtype=np.uint8)
    image_arr = np.pad(
        image_arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return torch.from_numpy(image_arr.copy()).float().permute(2, 0, 1) / 255.0


def build_frame_specs(
    *,
    davis_root: Path,
    sequence: str,
    num_frames: int,
    context_size: int,
    context_stride: int,
) -> list[FrameSpec]:
    image_paths, _ = load_davis_sequence_paths(davis_root, sequence)
    if num_frames > len(image_paths):
        raise ValueError(f"Requested {num_frames} frames from {sequence}, but only {len(image_paths)} are available.")
    specs = []
    for frame_idx in range(num_frames):
        context = select_context_indices(
            len(image_paths),
            frame_idx,
            total_frames=context_size,
            stride=context_stride,
        )
        specs.append(
            FrameSpec(
                sequence=sequence,
                frame_idx=frame_idx,
                image_paths=tuple(image_paths[idx] for idx in context),
            )
        )
    return specs


def load_rgb_cache(specs: Sequence[FrameSpec]) -> Dict[Path, np.ndarray]:
    paths = sorted({path for spec in specs for path in spec.image_paths})
    return {path: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy() for path in paths}


def percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean() * 1000.0),
        "median_ms": float(np.median(array) * 1000.0),
        "p95_ms": float(np.percentile(array, 95) * 1000.0),
    }


def save_minimal_mask(
    *,
    output_root: Path,
    spec: FrameSpec,
    target_image: torch.Tensor,
    upsampled_score: np.ndarray,
    threshold: float,
) -> None:
    frame_dir = output_root / spec.sequence / f"frame_{spec.frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_np = np.clip(target_image.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    _, _, cropped_maps = crop_to_content_bbox(image_np, upsampled_score)
    cropped_score = cropped_maps[0].astype(np.float32)
    binary, threshold_value = binary_mask_from_score(cropped_score, threshold)
    np.save(frame_dir / "avg_score_upsampled.npy", cropped_score)
    np.save(frame_dir / "binary_mask_upsampled.npy", binary.astype(np.uint8))
    Image.fromarray((binary * 255).astype(np.uint8)).save(frame_dir / "binary_mask_upsampled.png")
    with (frame_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "image_path": str(spec.image_paths[0]),
                "sequence": spec.sequence,
                "frame_idx": spec.frame_idx,
                "threshold": float(threshold_value),
                "runtime_artifact": True,
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0 or args.warmup_frames < 0:
        raise ValueError("--num_frames must be positive and --warmup_frames must be non-negative.")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The runtime protocol requires an available CUDA device.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    davis_root = resolve_davis_root(args.davis_root)
    measured_specs = build_frame_specs(
        davis_root=davis_root,
        sequence=args.sequence,
        num_frames=args.num_frames,
        context_size=args.context_size,
        context_stride=args.context_stride,
    )
    warmup_specs = build_frame_specs(
        davis_root=davis_root,
        sequence=args.warmup_sequence,
        num_frames=args.warmup_frames,
        context_size=args.context_size,
        context_stride=args.context_stride,
    )
    rgb_cache = load_rgb_cache([*warmup_specs, *measured_specs])

    device = torch.device(args.device)
    model = load_pi3_family_model("pi3x", args.checkpoint, device)
    source = GROUP_TO_SOURCE[args.group]
    run_dir_by_key = dict(discover_run_dirs(args.probe_root, group=args.group))
    usable_keys = sorted(run_dir_by_key, key=lambda key: int(key.rsplit("_", 1)[-1]))
    if not usable_keys:
        raise RuntimeError(f"No probe blocks found under {args.probe_root / args.group}")
    probe_cache: Dict[str, torch.nn.Module] = {}
    for key in usable_keys:
        get_cached_probe(
            key=key,
            run_dir_by_key=run_dir_by_key,
            device=device,
            probe_cache=probe_cache,
        )

    def run_frame(spec: FrameSpec, measured: bool, mask_subdir: str | None = None) -> dict[str, object]:
        if measured:
            torch.cuda.reset_peak_memory_stats(device)
        total_t0 = perf_counter()
        preprocess_t0 = total_t0
        context_tensors = [preprocess_rgb(rgb_cache[path]) for path in spec.image_paths]
        preprocess_seconds = perf_counter() - preprocess_t0

        transfer_t0 = perf_counter()
        images_b = torch.stack(context_tensors, dim=0).unsqueeze(0).to(device)
        synchronize_device(device)
        transfer_seconds = perf_counter() - transfer_t0

        backbone_t0 = perf_counter()
        with torch.inference_mode():
            feature_map = extract_feature_map(
                model_name="pi3x",
                model=model,
                images_b=images_b,
                token_sources=[source],
            )
        synchronize_device(device)
        backbone_forward_seconds = perf_counter() - backbone_t0
        backbone_total_seconds = perf_counter() - total_t0
        backbone_peak_bytes = int(torch.cuda.max_memory_allocated(device)) if measured else 0

        missing = [key for key in usable_keys if key not in feature_map]
        if missing:
            raise RuntimeError(f"Backbone did not return probe features: {missing}")
        target_h, target_w = context_tensors[0].shape[-2:]
        grid_shape = (target_h // PI3_PATCH_SIZE, target_w // PI3_PATCH_SIZE)
        averaged_patch_score, probe_seconds, average_seconds = score_single_stream_blocks(
            feature_map=feature_map,
            usable_keys=usable_keys,
            run_dir_by_key=run_dir_by_key,
            query_idx=0,
            batch_size=args.score_batch_size,
            device=device,
            grid_shape=grid_shape,
            probe_cache=probe_cache,
            invert_query_idx_zero=True,
        )
        postprocess_t0 = perf_counter()
        upsampled_score = upsample_patch_map(averaged_patch_score, (target_h, target_w))
        binary_mask_from_score(upsampled_score, args.threshold)
        postprocess_seconds = perf_counter() - postprocess_t0
        synchronize_device(device)
        coarse_total_seconds = perf_counter() - total_t0
        coarse_peak_bytes = int(torch.cuda.max_memory_allocated(device)) if measured else 0

        if mask_subdir is not None and args.save_masks:
            save_minimal_mask(
                output_root=output_dir / mask_subdir,
                spec=spec,
                target_image=context_tensors[0],
                upsampled_score=upsampled_score,
                threshold=args.threshold,
            )
        del feature_map, images_b
        return {
            "sequence": spec.sequence,
            "frame_idx": spec.frame_idx,
            "context_frame_paths": "|".join(str(path) for path in spec.image_paths),
            "preprocess_seconds": preprocess_seconds,
            "transfer_seconds": transfer_seconds,
            "backbone_forward_seconds": backbone_forward_seconds,
            "backbone_end_to_end_seconds": backbone_total_seconds,
            "probe_scoring_seconds": probe_seconds,
            "block_averaging_seconds": average_seconds,
            "upsample_threshold_seconds": postprocess_seconds,
            "coarse_total_seconds": coarse_total_seconds,
            "backbone_peak_vram_gib": backbone_peak_bytes / (1024.0**3),
            "coarse_peak_vram_gib": coarse_peak_bytes / (1024.0**3),
        }

    print(f"Warm-up: {len(warmup_specs)} frames from DAVIS/{args.warmup_sequence}")
    for spec in warmup_specs:
        run_frame(spec, measured=False, mask_subdir="warmup_masks")
    synchronize_device(device)

    rows = []
    print(f"Measured run: {len(measured_specs)} frames from DAVIS/{args.sequence}")
    for index, spec in enumerate(measured_specs, start=1):
        rows.append(run_frame(spec, measured=True, mask_subdir="coarse_masks"))
        print(f"[{index:03d}/{len(measured_specs):03d}] {spec.sequence}/frame_{spec.frame_idx:05d}")

    per_frame_csv = output_dir / "coarse_per_frame.csv"
    with per_frame_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "precision": "float32",
            "batch_size": 1,
            "context_size": args.context_size,
            "context_stride": args.context_stride,
            "input_size": [PI3_TARGET_SIZE, PI3_TARGET_SIZE],
            "warmup_frames": len(warmup_specs),
            "measured_frames": len(measured_specs),
            "sequence": args.sequence,
            "model_loading_included": False,
            "disk_io_included": False,
            "cpu_preprocessing_included": True,
            "mask_output_saving_included": False,
            "blocks": usable_keys,
        },
        "backbone": {
            **percentile_summary([float(row["backbone_end_to_end_seconds"]) for row in rows]),
            "forward_only": percentile_summary([float(row["backbone_forward_seconds"]) for row in rows]),
            "peak_vram_gib": max(float(row["backbone_peak_vram_gib"]) for row in rows),
        },
        "pmd_coarse": {
            **percentile_summary([float(row["coarse_total_seconds"]) for row in rows]),
            "peak_vram_gib": max(float(row["coarse_peak_vram_gib"]) for row in rows),
        },
    }
    with (output_dir / "coarse_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
