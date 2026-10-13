from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Pi3X/PMD and SAM2 wall-clock runtime measurements.")
    parser.add_argument("--coarse_csv", type=Path, required=True)
    parser.add_argument("--coarse_summary", type=Path, required=True)
    parser.add_argument("--sam2_meta", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def stats(seconds: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
    }


def main() -> None:
    args = parse_args()
    with args.coarse_csv.open("r", encoding="utf-8", newline="") as handle:
        coarse_rows = list(csv.DictReader(handle))
    with args.coarse_summary.open("r", encoding="utf-8") as handle:
        coarse_summary = json.load(handle)
    with args.sam2_meta.open("r", encoding="utf-8") as handle:
        sam2_meta = json.load(handle)

    runtime_profile = sam2_meta.get("runtime_profile")
    if not isinstance(runtime_profile, dict):
        raise RuntimeError(f"SAM2 runtime profile missing from {args.sam2_meta}")
    sam2_by_frame = runtime_profile.get("frame_latency_seconds_amortized")
    if not isinstance(sam2_by_frame, dict):
        raise RuntimeError("SAM2 per-frame amortized timings are missing.")

    backbone_seconds = [float(row["backbone_end_to_end_seconds"]) for row in coarse_rows]
    coarse_seconds = [float(row["coarse_total_seconds"]) for row in coarse_rows]
    refined_seconds = []
    missing = []
    for row, coarse_value in zip(coarse_rows, coarse_seconds):
        frame_key = f"frame_{int(row['frame_idx']):05d}"
        if frame_key not in sam2_by_frame:
            missing.append(frame_key)
            continue
        refined_seconds.append(coarse_value + float(sam2_by_frame[frame_key]))
    if missing:
        raise RuntimeError(f"SAM2 timings are missing for {len(missing)} measured frames: {missing[:10]}")

    gpu = coarse_summary["protocol"]["gpu"]
    common = {
        "GPU": gpu,
        "Measured frames": len(coarse_rows),
        "Context": (
            f"{coarse_summary['protocol']['context_size']} frames, "
            f"stride {coarse_summary['protocol']['context_stride']}"
        ),
    }
    result_rows = [
        {
            "Method": "Pi3X backbone",
            **common,
            **stats(backbone_seconds),
            "peak_vram_gib": float(coarse_summary["backbone"]["peak_vram_gib"]),
        },
        {
            "Method": "Pi3X + PMD coarse",
            **common,
            **stats(coarse_seconds),
            "peak_vram_gib": float(coarse_summary["pmd_coarse"]["peak_vram_gib"]),
        },
        {
            "Method": "Pi3X + PMD + SAM2",
            **common,
            **stats(refined_seconds),
            "peak_vram_gib": max(
                float(coarse_summary["pmd_coarse"]["peak_vram_gib"]),
                float(runtime_profile["peak_vram_gib"]),
            ),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    output = {
        "protocol": {
            **coarse_summary["protocol"],
            "sam2_timing_scope": runtime_profile["timing_scope"],
            "sam2_sequence_processing": True,
            "sam2_frame_latency_note": (
                "Per-yield propagation time plus sequence initialization/prompting overhead amortized "
                "uniformly over measured output frames."
            ),
        },
        "results": result_rows,
        "sam2": runtime_profile,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
