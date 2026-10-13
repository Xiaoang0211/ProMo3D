#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.artifacts import artifact_path


DEFAULT_MODEL_TYPES = [
    "pair_linear",
    "pair_diag_quadratic",
    "pair_lowrank_quadratic",
    "pair_attention",
    "pair_value",
    "pair_attention_biased_lowrank_quadratic",
    "pair_value_biased_lowrank_quadratic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cross-model IsDynamic comparison tables with blockwise mean/std and "
            "improvement over the trivial all-positive baseline."
        )
    )
    parser.add_argument(
        "--result_roots",
        type=str,
        required=True,
        help="Comma-separated list of probing result roots, one per model.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Optional comma-separated model labels matching --result_roots.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="inter_frame_block,intra_frame_block",
        help="Comma-separated group names; one output table is generated per group.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default=",".join(DEFAULT_MODEL_TYPES),
        help="Comma-separated probe model_types to include.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
        help="Metric to aggregate across blocks.",
    )
    parser.add_argument(
        "--metric_splits",
        type=str,
        default="test_davis,test_segtrackv2",
        help="Comma-separated metric splits to export. Supported: test_davis,test_segtrackv2,val.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=artifact_path("illustration", "diagram", "tables"),
        help="Directory for exported CSV tables.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="isdynamic_model_comparison",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _derive_label(result_root: Path) -> str:
    text = str(result_root).lower()
    name = result_root.name.lower()
    if "depth-anything-3" in text or "da3" in text:
        return "da3"
    if "map-anything" in text or "mapanything" in text:
        return "mapanything"
    if "pi3x" in name or "pi3x" in text:
        return "pi3x"
    if "pi3" in name or "/pi3/" in text:
        return "pi3"
    if "vggt" in text:
        return "vggt"
    return result_root.name


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if not values:
        return float("nan")
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _model_type_label(model_type: str) -> str:
    label = model_type[len("pair_") :] if model_type.startswith("pair_") else model_type
    return label.replace("biased", "initialized")


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _load_summary_meta(result_root: Path) -> Dict:
    summary_path = result_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found under result root: {result_root}")
    return _read_json(summary_path)


def _metric_from_metrics_blob(blob: Dict, metric: str, metric_split: str) -> Optional[float]:
    if metric_split == "val":
        source = blob.get("best_val") or blob.get("best")
    elif metric_split.startswith("test_"):
        source = (blob.get("best_tests") or {}).get(metric_split)
    else:
        raise ValueError(f"Unsupported metric_split: {metric_split}")

    if not source or metric not in source:
        return None
    return float(source[metric])


def _collect_result_records(
    result_root: Path,
    model_label: str,
    groups: List[str],
    model_types: List[str],
    metric: str,
    metric_split: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen = set()

    for metrics_path in sorted(result_root.glob("pair_*/*/*/metrics.json")):
        feature_dir = metrics_path.parent
        group_dir = feature_dir.parent
        model_type_dir = group_dir.parent

        group = group_dir.name
        model_type = model_type_dir.name
        feature_key = feature_dir.name

        if groups and group not in groups:
            continue
        if model_types and model_type not in model_types:
            continue

        try:
            block_idx = int(feature_key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue

        key = (model_label, group, model_type, block_idx)
        if key in seen:
            continue
        seen.add(key)

        blob = _read_json(metrics_path)
        value = _metric_from_metrics_blob(blob, metric=metric, metric_split=metric_split)
        if value is None:
            continue

        records.append(
            {
                "model": model_label,
                "group": group,
                "model_type": model_type,
                "feature_key": feature_key,
                "block_idx": block_idx,
                "metric_value": value,
                "metrics_path": str(metrics_path),
            }
        )

    return records


def _pick_sequence_feature_dir(sequence_dir: Path) -> Optional[Path]:
    for child in sorted(sequence_dir.iterdir()):
        if child.is_dir() and any(child.glob("part_*.npz")):
            return child
    return None


def _compute_trivial_baseline_same_label(
    sharded_root: Path,
    split_name: str,
) -> float:
    split_root = sharded_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    group_pos_counts: Dict[int, int] = {}
    group_total_counts: Dict[int, int] = {}

    for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        feature_dir = _pick_sequence_feature_dir(sequence_dir)
        if feature_dir is None:
            continue

        for part_path in sorted(feature_dir.glob("part_*.npz")):
            with np.load(part_path, allow_pickle=False) as part:
                labels = np.asarray(part["is_dynamic"]).reshape(-1).astype(np.int64, copy=False)
                if "valid_mask" in part:
                    valid = np.asarray(part["valid_mask"]).reshape(-1).astype(bool, copy=False)
                else:
                    valid = np.ones(labels.shape[0], dtype=bool)
                if "group_ids" in part:
                    groups = np.asarray(part["group_ids"]).reshape(-1).astype(np.int64, copy=False)
                else:
                    groups = np.zeros(labels.shape[0], dtype=np.int64)

            labels = labels[valid]
            groups = groups[valid]
            if labels.size == 0:
                continue

            unique_groups, inverse = np.unique(groups, return_inverse=True)
            total_per_group = np.bincount(inverse)
            pos_per_group = np.bincount(inverse, weights=labels.astype(np.int64))

            for idx, group_id in enumerate(unique_groups.tolist()):
                group_total_counts[group_id] = group_total_counts.get(group_id, 0) + int(total_per_group[idx])
                group_pos_counts[group_id] = group_pos_counts.get(group_id, 0) + int(pos_per_group[idx])

    positive_pairs = 0
    total_pairs = 0
    for group_id, total_count in group_total_counts.items():
        if total_count < 2:
            continue
        pos_count = group_pos_counts.get(group_id, 0)
        neg_count = total_count - pos_count
        total_pairs += total_count * (total_count - 1) // 2
        positive_pairs += pos_count * (pos_count - 1) // 2
        positive_pairs += neg_count * (neg_count - 1) // 2

    if total_pairs == 0:
        return float("nan")
    return float(positive_pairs / total_pairs)


def _resolve_baseline_split(summary_meta: Dict, metric_split: str) -> str:
    if metric_split == "val":
        return str(summary_meta.get("val_split", "val"))
    if metric_split.startswith("test_"):
        return metric_split
    raise ValueError(f"Unsupported metric_split: {metric_split}")


def _format_cell(mean_value: float, std_value: float, baseline: float, delta: float) -> str:
    if not np.isfinite(mean_value):
        return ""
    if np.isfinite(std_value) and np.isfinite(delta):
        return f"{mean_value:.6f} +/- {std_value:.6f} ({delta:+.6f})"
    if np.isfinite(std_value):
        return f"{mean_value:.6f} +/- {std_value:.6f}"
    return f"{mean_value:.6f}"


def _export_for_metric_split(
    *,
    result_roots: List[Path],
    labels: List[str],
    groups: List[str],
    model_types: List[str],
    metric: str,
    metric_split: str,
    out_dir: Path,
    prefix: str,
) -> None:
    baseline_cache: Dict[Tuple[str, str], float] = {}
    all_records: List[Dict[str, object]] = []

    for result_root, model_label in zip(result_roots, labels):
        summary_meta = _load_summary_meta(result_root)
        sharded_root = Path(str(summary_meta["sharded_root"]))
        baseline_split = _resolve_baseline_split(summary_meta, metric_split)
        baseline_key = (str(sharded_root), baseline_split)
        if baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = _compute_trivial_baseline_same_label(
                sharded_root=sharded_root,
                split_name=baseline_split,
            )
        baseline_value = baseline_cache[baseline_key]

        records = _collect_result_records(
            result_root=result_root,
            model_label=model_label,
            groups=groups,
            model_types=model_types,
            metric=metric,
            metric_split=metric_split,
        )
        for record in records:
            record["baseline"] = baseline_value
            all_records.append(record)

    if not all_records:
        raise RuntimeError(f"No records found for metric_split={metric_split}.")

    long_rows: List[List[object]] = []
    grouped_records: Dict[Tuple[str, str, str], List[float]] = {}
    baseline_by_model: Dict[str, float] = {}

    for record in sorted(
        all_records,
        key=lambda r: (
            str(r["group"]),
            str(r["model"]),
            str(r["model_type"]),
            int(r["block_idx"]),
        ),
    ):
        group = str(record["group"])
        model_label = str(record["model"])
        model_type = str(record["model_type"])
        metric_value = float(record["metric_value"])
        baseline = float(record["baseline"])
        delta = metric_value - baseline
        long_rows.append(
            [
                group,
                model_label,
                model_type,
                _model_type_label(model_type),
                int(record["block_idx"]),
                metric_value,
                baseline,
                delta,
                record["metrics_path"],
            ]
        )
        grouped_records.setdefault((group, model_label, model_type), []).append(metric_value)
        baseline_by_model[model_label] = baseline

    long_path = out_dir / f"{prefix}_{metric_split}_{metric}_blockwise_long.csv"
    _write_csv(
        long_path,
        [
            "group",
            "model",
            "model_type",
            "model_label",
            "block_idx",
            metric,
            "baseline",
            "delta_vs_baseline",
            "metrics_path",
        ],
        long_rows,
    )

    summary_rows: List[List[object]] = []
    lookup: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for (group, model_label, model_type), values in sorted(grouped_records.items()):
        mean_value = _mean(values)
        std_value = _std(values)
        baseline = baseline_by_model[model_label]
        delta = mean_value - baseline
        row = [
            group,
            model_label,
            model_type,
            _model_type_label(model_type),
            mean_value,
            std_value,
            baseline,
            delta,
            len(values),
        ]
        summary_rows.append(row)
        lookup[(group, model_type, model_label)] = {
            "mean": mean_value,
            "std": std_value,
            "baseline": baseline,
            "delta": delta,
            "n_blocks": float(len(values)),
        }

    summary_path = out_dir / f"{prefix}_{metric_split}_{metric}_summary.csv"
    _write_csv(
        summary_path,
        [
            "group",
            "model",
            "model_type",
            "model_label",
            f"{metric}_mean",
            f"{metric}_std",
            "baseline",
            "delta_vs_baseline",
            "n_blocks",
        ],
        summary_rows,
    )

    for group in groups:
        header = ["probe"] + labels
        rows: List[List[object]] = []
        baseline_row: List[object] = ["baseline"]
        for model_label in labels:
            baseline = baseline_by_model.get(model_label, float("nan"))
            baseline_row.append(f"{baseline:.6f}" if np.isfinite(baseline) else "")
        rows.append(baseline_row)

        for model_type in model_types:
            row: List[object] = [_model_type_label(model_type)]
            for model_label in labels:
                stats = lookup.get((group, model_type, model_label))
                if stats is None:
                    row.append("")
                    continue
                row.append(
                    _format_cell(
                        mean_value=float(stats["mean"]),
                        std_value=float(stats["std"]),
                        baseline=float(stats["baseline"]),
                        delta=float(stats["delta"]),
                    )
                )
            rows.append(row)

        group_path = out_dir / f"{prefix}_{group}_{metric_split}_{metric}.csv"
        _write_csv(group_path, header, rows)

    print(f"Saved: {long_path}")
    print(f"Saved: {summary_path}")
    for group in groups:
        print(f"Saved: {out_dir / f'{prefix}_{group}_{metric_split}_{metric}.csv'}")


def main() -> None:
    args = parse_args()
    result_roots = [Path(p).resolve() for p in _parse_csv(args.result_roots)]
    if not result_roots:
        raise ValueError("--result_roots resolved to empty list.")

    labels = _parse_csv(args.labels)
    if labels and len(labels) != len(result_roots):
        raise ValueError("--labels must match --result_roots in length.")
    if not labels:
        labels = [_derive_label(root) for root in result_roots]

    groups = _parse_csv(args.groups)
    model_types = _parse_csv(args.model_types)
    metric_splits = _parse_csv(args.metric_splits)
    out_dir = Path(args.out_dir)

    for metric_split in metric_splits:
        _export_for_metric_split(
            result_roots=result_roots,
            labels=labels,
            groups=groups,
            model_types=model_types,
            metric=args.metric,
            metric_split=metric_split,
            out_dir=out_dir,
            prefix=args.prefix,
        )


if __name__ == "__main__":
    main()
