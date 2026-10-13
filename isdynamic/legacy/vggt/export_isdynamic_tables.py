#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.artifacts import artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export IsDynamic blockwise probe results to CSV tables from one or more summary.json files."
        )
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path or comma-separated list of summary.json files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=artifact_path("illustration", "diagram", "tables"),
        help="Directory to store output CSV tables.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
        help="Primary metric used for grouped and per-block tables.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Optional comma-separated group filter.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="",
        help="Optional comma-separated model_type filter.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="isdynamic",
        help="File prefix for output CSVs.",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_summaries(paths: Iterable[Path]) -> Dict:
    model_types: List[str] = []
    results: List[Dict] = []
    seen = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")
        summary = _load_json(path)

        for model_type in summary.get("model_types", []):
            if model_type not in model_types:
                model_types.append(model_type)

        for item in summary.get("results", []):
            key = (
                item.get("group"),
                int(item.get("block_idx", -1)),
                item.get("feature_key"),
                item.get("model_type"),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    return {"model_types": model_types, "results": results}


def _model_label(model_type: str) -> str:
    label = model_type[len("pair_") :] if model_type.startswith("pair_") else model_type
    return label.replace("biased", "initialized")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if not values:
        return float("nan")
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_paths = [Path(p) for p in _parse_csv(args.summary_json)]
    if not summary_paths:
        raise ValueError("--summary_json resolved to empty list.")
    merged = _merge_summaries(summary_paths)
    results = merged.get("results", [])

    group_filter = set(_parse_csv(args.groups))
    model_filter = set(_parse_csv(args.model_types))

    filtered: List[Dict] = []
    for item in results:
        group = str(item.get("group"))
        model_type = str(item.get("model_type"))
        if group_filter and group not in group_filter:
            continue
        if model_filter and model_type not in model_filter:
            continue
        filtered.append(item)

    if not filtered:
        raise RuntimeError("No results left after filtering.")

    # 1) Long table
    metric_keys = set()
    for item in filtered:
        best = item.get("best", {})
        metric_keys.update(best.keys())
    metric_cols = sorted(metric_keys)
    long_header = [
        "group",
        "block_idx",
        "feature_key",
        "model_type",
        "model_label",
        "run_dir",
    ] + metric_cols
    long_rows: List[List[object]] = []
    for item in sorted(filtered, key=lambda x: (str(x.get("group")), int(x.get("block_idx", -1)), str(x.get("model_type")))):
        best = item.get("best", {})
        row = [
            item.get("group"),
            int(item.get("block_idx", -1)),
            item.get("feature_key"),
            item.get("model_type"),
            _model_label(str(item.get("model_type"))),
            item.get("run_dir", ""),
        ]
        for col in metric_cols:
            row.append(best.get(col, ""))
        long_rows.append(row)

    out_dir = Path(args.out_dir)
    long_path = out_dir / f"{args.prefix}_results_long.csv"
    _write_csv(long_path, long_header, long_rows)

    # 2) Group/model summary table for selected metric
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for item in filtered:
        group = str(item.get("group"))
        model_type = str(item.get("model_type"))
        value = item.get("best", {}).get(args.metric, None)
        if value is None:
            continue
        key = (group, model_type)
        grouped.setdefault(key, []).append(float(value))

    summary_header = ["group", "model_type", "model_label", f"{args.metric}_mean", f"{args.metric}_std", "n_blocks"]
    summary_rows: List[List[object]] = []
    for (group, model_type), values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        summary_rows.append(
            [
                group,
                model_type,
                _model_label(model_type),
                _mean(values),
                _std(values),
                len(values),
            ]
        )
    summary_path = out_dir / f"{args.prefix}_summary_{args.metric}.csv"
    _write_csv(summary_path, summary_header, summary_rows)

    # 3) Per-group blockwise table (rows=block_idx, cols=model_label, values=metric)
    groups = sorted({str(x.get("group")) for x in filtered})
    for group in groups:
        items = [x for x in filtered if str(x.get("group")) == group]
        blocks = sorted({int(x.get("block_idx", -1)) for x in items})
        model_types = sorted({str(x.get("model_type")) for x in items})
        label_cols = [_model_label(m) for m in model_types]

        value_map: Dict[Tuple[int, str], float] = {}
        for item in items:
            block = int(item.get("block_idx", -1))
            model_type = str(item.get("model_type"))
            value = item.get("best", {}).get(args.metric, None)
            if value is not None:
                value_map[(block, model_type)] = float(value)

        header = ["block_idx"] + label_cols
        rows: List[List[object]] = []
        for block in blocks:
            row: List[object] = [block]
            for model_type in model_types:
                row.append(value_map.get((block, model_type), ""))
            rows.append(row)

        group_path = out_dir / f"{args.prefix}_{group}_{args.metric}_by_block.csv"
        _write_csv(group_path, header, rows)

    print(f"Saved: {long_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved per-group block tables under: {out_dir}")


if __name__ == "__main__":
    main()
