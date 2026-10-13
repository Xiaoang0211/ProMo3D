from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine staged RAFT, FlowP-SAM, and FlowI-SAM profiles.")
    parser.add_argument("--raft_csv", type=Path, required=True)
    parser.add_argument("--raft_summary", type=Path, required=True)
    parser.add_argument("--flowpsam_csv", type=Path, required=True)
    parser.add_argument("--flowpsam_summary", type=Path, required=True)
    parser.add_argument("--flowisam_csv", type=Path, required=True)
    parser.add_argument("--flowisam_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def stats(seconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    raft_rows = read_csv(args.raft_csv)
    flowpsam_rows = read_csv(args.flowpsam_csv)
    flowisam_rows = read_csv(args.flowisam_csv)
    with args.raft_summary.open("r", encoding="utf-8") as handle:
        raft_summary = json.load(handle)
    with args.flowpsam_summary.open("r", encoding="utf-8") as handle:
        flowpsam_summary = json.load(handle)
    with args.flowisam_summary.open("r", encoding="utf-8") as handle:
        flowisam_summary = json.load(handle)

    raft_by_frame = {
        (row["sequence"], int(row["frame_idx"])): float(row["raft_stage_total_seconds"])
        for row in raft_rows
    }
    flowpsam_by_frame = {
        (row["sequence"], int(row["frame_idx"])): float(row["flowpsam_stage_total_seconds"])
        for row in flowpsam_rows
    }
    flowisam_by_frame = {
        (row["sequence"], int(row["frame_idx"])): float(row["flowisam_stage_total_seconds"])
        for row in flowisam_rows
    }
    if not (raft_by_frame.keys() == flowpsam_by_frame.keys() == flowisam_by_frame.keys()):
        raise RuntimeError("RAFT, FlowP-SAM, and FlowI-SAM profiles do not cover the same frames.")
    ordered_keys = sorted(raft_by_frame)
    raft_seconds = [raft_by_frame[key] for key in ordered_keys]
    flowpsam_seconds = [flowpsam_by_frame[key] for key in ordered_keys]
    flowisam_seconds = [flowisam_by_frame[key] for key in ordered_keys]
    flowp_total_seconds = [
        raft_by_frame[key] + flowpsam_by_frame[key]
        for key in ordered_keys
    ]
    combined_total_seconds = [
        raft_by_frame[key] + flowpsam_by_frame[key] + flowisam_by_frame[key]
        for key in ordered_keys
    ]

    raft_peak = float(raft_summary["raft"]["peak_vram_gib"])
    flowpsam_peak = float(flowpsam_summary["flowpsam"]["peak_vram_gib"])
    flowisam_peak = float(flowisam_summary["flowisam"]["peak_vram_gib"])
    common = {
        "GPU": raft_summary["protocol"]["gpu"],
        "Measured frames": len(ordered_keys),
        "Staged": "yes",
    }
    result_rows = [
        {
            "Method": "RAFT (4 flow gaps)",
            **common,
            **stats(raft_seconds),
            "peak_vram_gib": raft_peak,
        },
        {
            "Method": "FlowP-SAM",
            **common,
            **stats(flowpsam_seconds),
            "peak_vram_gib": flowpsam_peak,
        },
        {
            "Method": "FlowP-SAM (including RAFT)",
            **common,
            **stats(flowp_total_seconds),
            "peak_vram_gib": max(raft_peak, flowpsam_peak),
        },
        {
            "Method": "FlowI-SAM",
            **common,
            **stats(flowisam_seconds),
            "peak_vram_gib": flowisam_peak,
        },
        {
            "Method": "Flow-SAM (FlowP + FlowI, including RAFT)",
            **common,
            **stats(combined_total_seconds),
            "peak_vram_gib": max(raft_peak, flowpsam_peak, flowisam_peak),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    output = {
        "protocol": {
            "gpu": raft_summary["protocol"]["gpu"],
            "sequence": raft_summary["protocol"]["sequence"],
            "measured_frames": len(ordered_keys),
            "staged_pipeline": True,
            "latency_note": "Shared RAFT, FlowP-SAM, and FlowI-SAM per-frame latencies are summed.",
            "peak_vram_note": "The official components run sequentially; peak VRAM is their maximum.",
            "raft": raft_summary["protocol"],
            "flowpsam": flowpsam_summary["protocol"],
            "flowisam": flowisam_summary["protocol"],
        },
        "results": result_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
