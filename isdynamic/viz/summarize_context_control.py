from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare temporal and repeated-target PMD score maps.")
    parser.add_argument("--temporal_mask_root", type=Path, required=True)
    parser.add_argument("--repeated_mask_root", type=Path, required=True)
    parser.add_argument("--model", default="pi3x")
    parser.add_argument("--group", default="inter_frame_block")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _frame_rows(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    repeated_base = args.repeated_mask_root / args.model / args.group
    temporal_base = args.temporal_mask_root / args.model / args.group
    pattern = "*/*/bilinear_upsample/frame_*"

    for repeated_dir in sorted(repeated_base.glob(pattern)):
        relative = repeated_dir.relative_to(repeated_base)
        benchmark, sequence, _variant, frame_name = relative.parts
        temporal_dir = temporal_base / relative

        required = (
            "avg_score_upsampled.npy",
            "binary_mask_upsampled.npy",
            "binary_mask_gt.npy",
        )
        if not all((repeated_dir / name).exists() and (temporal_dir / name).exists() for name in required):
            continue

        temporal_score = np.load(temporal_dir / "avg_score_upsampled.npy").astype(np.float32)
        repeated_score = np.load(repeated_dir / "avg_score_upsampled.npy").astype(np.float32)
        temporal_mask = np.load(temporal_dir / "binary_mask_upsampled.npy") > 0
        repeated_mask = np.load(repeated_dir / "binary_mask_upsampled.npy") > 0
        gt = np.load(repeated_dir / "binary_mask_gt.npy") > 0
        if not (
            temporal_score.shape
            == repeated_score.shape
            == temporal_mask.shape
            == repeated_mask.shape
            == gt.shape
        ):
            continue

        foreground = gt
        background = ~gt
        temporal_fg = _safe_mean(temporal_score[foreground])
        repeated_fg = _safe_mean(repeated_score[foreground])
        temporal_bg = _safe_mean(temporal_score[background])
        repeated_bg = _safe_mean(repeated_score[background])

        yield {
            "benchmark": benchmark,
            "sequence": sequence,
            "frame": frame_name,
            "temporal_dynamic_area": float(temporal_mask.mean()),
            "repeated_dynamic_area": float(repeated_mask.mean()),
            "dynamic_area_suppression": float(temporal_mask.mean() - repeated_mask.mean()),
            "temporal_score_mean": float(temporal_score.mean()),
            "repeated_score_mean": float(repeated_score.mean()),
            "score_suppression": float(temporal_score.mean() - repeated_score.mean()),
            "temporal_foreground_score": temporal_fg,
            "repeated_foreground_score": repeated_fg,
            "foreground_score_suppression": temporal_fg - repeated_fg,
            "temporal_background_score": temporal_bg,
            "repeated_background_score": repeated_bg,
            "background_score_suppression": temporal_bg - repeated_bg,
        }


def _aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_names = [key for key in rows[0] if key not in {"benchmark", "sequence", "frame"}] if rows else []
    aggregates: List[Dict[str, object]] = []
    benchmark_names = sorted({str(row["benchmark"]) for row in rows})
    for benchmark in benchmark_names + ["all"]:
        selected = rows if benchmark == "all" else [row for row in rows if row["benchmark"] == benchmark]
        summary: Dict[str, object] = {"benchmark": benchmark, "num_frames": len(selected)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            summary[f"{metric}_std"] = float(finite.std()) if finite.size else float("nan")
        aggregates.append(summary)
    return aggregates


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = list(_frame_rows(args))
    if not rows:
        raise RuntimeError("No paired temporal/repeated-target frame outputs were found.")
    summary = _aggregate(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_frame.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"per_frame": rows, "summary": summary}, handle, indent=2)
    print(f"Saved context-control summary under: {args.output_dir}")


if __name__ == "__main__":
    main()
