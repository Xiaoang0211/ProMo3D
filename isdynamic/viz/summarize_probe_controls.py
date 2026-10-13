from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


VARIANTS = {
    "linear": ("probes/capacity/rank_8/pair_linear", "runs/linear"),
    "diag_quadratic": ("probes/capacity/rank_8/pair_diag_quadratic", "runs/diag_quadratic"),
    "lowrank_k4": ("probes/capacity/rank_4/pair_lowrank_quadratic", "runs/lowrank_k4"),
    "lowrank_k8": ("probes/capacity/rank_8/pair_lowrank_quadratic", "runs/lowrank_k8"),
    "shuffled_lowrank_k8": (
        "probes/controls/within_group_shuffled_labels/rank_8/pair_lowrank_quadratic",
        "runs/shuffled_lowrank_k8",
    ),
}
BENCHMARKS = ("davis_all", "segtrackv2", "fbms59")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Pi3X rebuttal probe controls.")
    parser.add_argument("--control_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_gain(metrics: Dict[str, float]) -> float:
    tp = float(metrics["tp"])
    fn = float(metrics["fn"])
    fp = float(metrics["fp"])
    tn = float(metrics["tn"])
    total = tp + fn + fp + tn
    baseline = (tp + fn) / total if total else 1.0
    denominator = 1.0 - baseline
    return (float(metrics["accuracy"]) - baseline) / denominator if denominator > 0 else float("nan")


def _best_val(probe_root: Path) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    candidates: List[Tuple[float, str, Dict[str, float], Dict[str, float]]] = []
    for metrics_path in sorted(probe_root.glob("inter_frame_block/*/metrics.json")):
        payload = _load_json(metrics_path)
        val = payload.get("best_val") or payload.get("best") or {}
        train = payload.get("best_train") or {}
        if "accuracy" not in val:
            continue
        candidates.append((_normalized_gain(val), metrics_path.parent.name, val, train))
    if not candidates:
        raise FileNotFoundError(f"No probe metrics found under: {probe_root}")
    _ng, block, val, train = max(candidates, key=lambda item: item[0])
    return block, val, train


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for variant, (probe_rel, run_rel) in VARIANTS.items():
        probe_root = args.control_root / probe_rel
        run_root = args.control_root / run_rel
        block, val, train = _best_val(probe_root)
        row: Dict[str, object] = {
            "variant": variant,
            "best_val_block": block,
            "youtubevos_val_accuracy": float(val["accuracy"]),
            "youtubevos_val_normalized_gain": _normalized_gain(val),
            "train_accuracy_at_selected_epoch": float(train.get("accuracy", float("nan"))),
        }
        for benchmark in BENCHMARKS:
            eval_path = (
                run_root
                / "isdynamic_motion_masks_eval"
                / "pi3x"
                / "inter_frame_block"
                / "bilinear_upsample"
                / f"{benchmark}.json"
            )
            payload = _load_json(eval_path)
            row[f"{benchmark}_j_mean"] = float(payload["metrics"]["J"]["M"])
            row[f"{benchmark}_j_recall"] = float(payload["metrics"]["J"]["R"])
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved probe-control table: {args.output_csv}")


if __name__ == "__main__":
    main()
