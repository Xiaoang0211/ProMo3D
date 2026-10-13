from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GeoMotion end-to-end inference.")
    parser.add_argument("--geomotion_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--pi3_model_path", type=Path, required=True)
    parser.add_argument("--raft_model_path", type=Path, required=True)
    parser.add_argument("--sam2_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--davis_root", type=Path, required=True)
    parser.add_argument("--sequence", default="parkour")
    parser.add_argument("--warmup_sequence", default="cows")
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/experiments/runtime/geomotion"),
    )
    return parser.parse_args()


def sequence_paths(davis_root: Path, sequence: str, count: int) -> list[Path]:
    image_root = davis_root / "DAVIS" / "JPEGImages" / "480p" / sequence
    paths = sorted(image_root.glob("*.jpg"))
    if count > len(paths):
        raise ValueError(f"Requested {count} frames from {image_root}, found {len(paths)}.")
    return paths[:count]


def load_frames(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def percentile_summary(seconds: list[float]) -> dict[str, float]:
    values_ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "mean_ms_per_frame": float(values_ms.mean()),
        "median_ms_per_frame": float(np.median(values_ms)),
        "p95_ms_per_frame": float(np.percentile(values_ms, 95)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The runtime protocol requires an available CUDA device.")
    if args.num_frames <= 0 or args.warmup_frames < 0 or args.sequence_length <= 0:
        raise ValueError("Frame counts and sequence length must be positive.")

    geomotion_root = args.geomotion_root.resolve()
    sys.path.insert(0, str((geomotion_root / "sam2-main").resolve()))
    sys.path.insert(0, str(geomotion_root))
    import motion_seg_inference as geomotion  # type: ignore[import-not-found]

    warmup_frames = load_frames(
        sequence_paths(args.davis_root.resolve(), args.warmup_sequence, args.warmup_frames)
    )
    measured_frames = load_frames(
        sequence_paths(args.davis_root.resolve(), args.sequence, args.num_frames)
    )

    device = torch.device("cuda")
    inference = geomotion.MotionSegmentationInference(
        model_path=str(args.model_path.resolve()),
        pi3_model_path=str(args.pi3_model_path.resolve()),
        raft_model_path=str(args.raft_model_path.resolve()),
        device=str(device),
    )
    def run_coarse_chunk(
        frames: list[Image.Image],
        sequence: str,
        start_idx: int,
        measured: bool,
    ) -> list[dict[str, object]]:
        if measured:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        coarse_t0 = perf_counter()
        motion_masks = inference.predict_motion_mask(frames)
        torch.cuda.synchronize(device)
        coarse_seconds = perf_counter() - coarse_t0
        coarse_peak = torch.cuda.max_memory_allocated(device) if measured else 0
        chunk_size = len(frames)
        return [
            {
                "sequence": sequence,
                "frame_idx": start_idx + offset,
                "chunk_start": start_idx,
                "chunk_size": chunk_size,
                "coarse_seconds_per_frame_amortized": coarse_seconds / chunk_size,
                "coarse_peak_vram_gib": coarse_peak / (1024.0**3),
            }
            for offset in range(chunk_size)
        ]

    def run_refined_chunk(
        frames: list[Image.Image],
        sequence: str,
        start_idx: int,
        measured: bool,
        predictor: object,
    ) -> list[dict[str, object]]:
        if measured:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        coarse_t0 = perf_counter()
        motion_masks = inference.predict_motion_mask(frames)
        torch.cuda.synchronize(device)
        coarse_seconds = perf_counter() - coarse_t0
        mask_h, mask_w = motion_masks.shape[1:]
        refine_t0 = perf_counter()
        frame_arrays = np.stack(
            [
                np.asarray(frame.resize((mask_w, mask_h), Image.Resampling.BILINEAR))
                .astype(np.float32)
                / 255.0
                for frame in frames
            ],
            axis=0,
        )
        frame_tensors = torch.from_numpy(frame_arrays).permute(0, 3, 1, 2).float()
        mask_list = torch.from_numpy(motion_masks).float()
        refined_masks = torch.zeros_like(mask_list)
        geomotion.refine_sam(
            frame_tensors,
            mask_list,
            refined_masks,
            offset=0,
            predictor=predictor,
        )
        torch.cuda.synchronize(device)
        refine_seconds = perf_counter() - refine_t0
        refined_peak = torch.cuda.max_memory_allocated(device) if measured else 0

        chunk_size = len(frames)
        refined_per_frame = (coarse_seconds + refine_seconds) / chunk_size
        return [
            {
                "sequence": sequence,
                "frame_idx": start_idx + offset,
                "chunk_start": start_idx,
                "chunk_size": chunk_size,
                "refinement_seconds_per_frame_amortized": refine_seconds / chunk_size,
                "refined_total_seconds_per_frame_amortized": refined_per_frame,
                "refined_peak_vram_gib": refined_peak / (1024.0**3),
            }
            for offset in range(chunk_size)
        ]

    print(
        f"GeoMotion warm-up: {len(warmup_frames)} frames "
        f"from DAVIS/{args.warmup_sequence}"
    )
    for start in range(0, len(warmup_frames), args.sequence_length):
        run_coarse_chunk(
            warmup_frames[start : start + args.sequence_length],
            args.warmup_sequence,
            start,
            measured=False,
        )

    rows: list[dict[str, object]] = []
    print(f"GeoMotion measured run: {len(measured_frames)} frames from DAVIS/{args.sequence}")
    for start in range(0, len(measured_frames), args.sequence_length):
        chunk = measured_frames[start : start + args.sequence_length]
        rows.extend(run_coarse_chunk(chunk, args.sequence, start, measured=True))
        print(f"[{start + len(chunk):03d}/{len(measured_frames):03d}]")

    predictor = geomotion.build_sam.build_sam2_video_predictor(
        args.sam2_config,
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
    )
    print(
        f"GeoMotion+SAM2 warm-up: {len(warmup_frames)} frames "
        f"from DAVIS/{args.warmup_sequence}"
    )
    for start in range(0, len(warmup_frames), args.sequence_length):
        run_refined_chunk(
            warmup_frames[start : start + args.sequence_length],
            args.warmup_sequence,
            start,
            measured=False,
            predictor=predictor,
        )

    refined_rows: list[dict[str, object]] = []
    print(
        f"GeoMotion+SAM2 measured run: {len(measured_frames)} frames "
        f"from DAVIS/{args.sequence}"
    )
    for start in range(0, len(measured_frames), args.sequence_length):
        chunk = measured_frames[start : start + args.sequence_length]
        refined_rows.extend(
            run_refined_chunk(
                chunk,
                args.sequence,
                start,
                measured=True,
                predictor=predictor,
            )
        )
        print(f"[{start + len(chunk):03d}/{len(measured_frames):03d}]")

    for coarse_row, refined_row in zip(rows, refined_rows, strict=True):
        if coarse_row["frame_idx"] != refined_row["frame_idx"]:
            raise RuntimeError("Coarse and refined runtime rows are misaligned.")
        coarse_row.update(
            {
                "refinement_seconds_per_frame_amortized": refined_row[
                    "refinement_seconds_per_frame_amortized"
                ],
                "refined_total_seconds_per_frame_amortized": refined_row[
                    "refined_total_seconds_per_frame_amortized"
                ],
                "refined_peak_vram_gib": refined_row["refined_peak_vram_gib"],
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coarse_seconds = [
        float(row["coarse_seconds_per_frame_amortized"]) for row in rows
    ]
    refined_seconds = [
        float(row["refined_total_seconds_per_frame_amortized"]) for row in rows
    ]
    summary = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(device),
            "precision": "float32; SAM2 refinement uses bfloat16 autocast",
            "batch_size": 1,
            "sequence_length": args.sequence_length,
            "input_size": [518, 518],
            "warmup_sequence": args.warmup_sequence,
            "warmup_frames": len(warmup_frames),
            "sequence": args.sequence,
            "measured_frames": len(measured_frames),
            "model_loading_included": False,
            "source_image_disk_io_included": False,
            "mask_writes_included": False,
            "cpu_preprocessing_included": True,
            "host_to_device_transfer_included": True,
            "sam2_temp_frame_io_included": True,
            "sam2_hole_filling_included": False,
        },
        "geomotion_coarse": {
            **percentile_summary(coarse_seconds),
            "peak_vram_gib": max(float(row["coarse_peak_vram_gib"]) for row in rows),
        },
        "geomotion_sam2": {
            **percentile_summary(refined_seconds),
            "peak_vram_gib": max(float(row["refined_peak_vram_gib"]) for row in rows),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
