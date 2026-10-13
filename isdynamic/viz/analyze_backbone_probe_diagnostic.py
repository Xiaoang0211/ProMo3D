from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


BACKBONES = {
    "pi3X": {
        "probe_rel": "model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/pair_lowrank_quadratic",
        "mask_model": "pi3x",
        "mask_group": "inter_frame_block",
    },
    "pi3": {
        "probe_rel": "model_outputs/pi3/rank_8_pi3/isdynamic_blockwise_youtubevos_pi3/pair_lowrank_quadratic",
        "mask_model": "pi3",
        "mask_group": "inter_frame_block",
    },
    "VGGT": {
        "probe_rel": "model_outputs/vggt/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic",
        "mask_model": "vggt",
        "mask_group": "inter_frame_block",
    },
    "DA3": {
        "probe_rel": "model_outputs/depth_anything_3/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic",
        "mask_model": "depth_anything_3",
        "mask_group": "inter_frame_block",
    },
    "MapAnything": {
        "probe_rel": "model_outputs/map_anything/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic",
        "mask_model": "map_anything",
        "mask_group": "inter_frame_block",
    },
    "DUSt3R": {
        "probe_rel": "model_outputs/dust3r/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic",
        "mask_model": "dust3r",
        "mask_group": "dec_view1__head_in_view1",
    },
}
BENCHMARKS = ("davis2017", "segtrackv2", "fbms59")
PRIMARY_SPLIT = "youtubevos_val"
MULTIVIEW_BACKBONES = ("pi3X", "pi3", "VGGT", "DA3", "MapAnything")
FIGURE_ARTIFACTS = {
    "figure_2": "isdynamic_visualizations/comparisons/blockwise/pi3x_da3_dust3r_blockwise_widescreen_panel.pdf",
    "appendix_figure_7": "isdynamic_visualizations/comparisons/blockwise/vggt_pi3_mapanything_blockwise_widescreen_panel.pdf",
    "figure_5a": "isdynamic_visualizations/comparisons/data_starvation/pi3x_da3_dust3r_test_davis_normalized_gain_panel.pdf",
    "appendix_figure_8": "isdynamic_visualizations/comparisons/data_starvation/pi3x_da3_dust3r_test_davis_normalized_gain_k8_stats_panel.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate rank-8 token-level probe decodability to coarse PMD J-M."
    )
    parser.add_argument("--probe_outputs_root", type=Path, required=True)
    parser.add_argument("--probe_paths_json", type=Path, help="Optional map from backbone display name to absolute probe directory.")
    parser.add_argument("--mask_outputs_root", type=Path, required=True)
    parser.add_argument("--mask_eval_root", type=Path, required=True)
    parser.add_argument("--source_val_eval_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_gain(metrics: dict[str, Any]) -> tuple[float, float, float]:
    tp = float(metrics["tp"])
    fp = float(metrics["fp"])
    fn = float(metrics["fn"])
    tn = float(metrics["tn"])
    total = tp + fp + fn + tn
    if total <= 0:
        raise ValueError("Cannot compute Normalized Gain from an empty split")
    positive_rate = (tp + fn) / total
    negative_rate = (tn + fp) / total
    baseline = max(positive_rate, negative_rate)
    denominator = 1.0 - baseline
    if denominator <= 0:
        raise ValueError("Majority pair-label baseline leaves no Normalized Gain denominator")
    return (
        (float(metrics["accuracy"]) - baseline) / denominator,
        baseline,
        positive_rate,
    )


def metric_split(payload: dict[str, Any], split: str) -> dict[str, Any]:
    if split == PRIMARY_SPLIT:
        result = payload.get("best_val")
    elif split == "test_davis":
        result = (payload.get("best_tests") or {}).get("test_davis")
    else:
        raise ValueError(f"Unsupported split: {split}")
    if not isinstance(result, dict):
        raise KeyError(f"Missing metrics for split={split}")
    return result


def find_mask_meta(mask_root: Path, model: str, group: str) -> Path:
    search_root = mask_root / model / group / "davis_all"
    matches = sorted(search_root.glob("*/bilinear_upsample/frame_*/meta.json"))
    if not matches:
        raise FileNotFoundError(f"No coarse-mask metadata found under {search_root}")
    return matches[0]


def group_for_feature(feature_key: str, groups: list[str]) -> str:
    matches = [group for group in groups if feature_key.startswith(f"{group}_")]
    if len(matches) != 1:
        raise ValueError(f"Could not map feature {feature_key!r} to one of {groups}")
    return matches[0]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def probe_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    model = config["model"]
    train = config["train"]
    return {
        "model_type": model["model_type"],
        "rank": int(model["rank"]),
        "input_dim": int(model["input_dim"]),
        "seed": int(train["seed"]),
        "epochs": int(train["epochs"]),
        "learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]),
        "pair_train_samples": int(train["pair_train_samples"]),
        "pair_val_samples": int(train["pair_val_samples"]),
        "pair_test_samples": int(train["pair_test_samples"]),
        "pair_same_group": bool(train["pair_same_group"]),
        "pair_balance": bool(train["pair_balance"]),
        "pair_label_mode": str(train["pair_label_mode"]),
        "pair_tokens_per_group": int(train["pair_tokens_per_group"]),
        "pair_resample_each_epoch": bool(train["pair_resample_each_epoch"]),
        "pair_eval_all_pairs": bool(train["pair_eval_all_pairs"]),
        "selection_metric": str(train["selection_metric"]),
    }


def figure_and_table_sources(args: argparse.Namespace) -> dict[str, Any]:
    viz_root = Path(__file__).resolve().parent
    paper_root = Path(__file__).resolve().parents[2]
    return {
        "figure_2": {
            "plot_script": str((viz_root / "plot_blockwise_comparison_panel.py").resolve()),
            "metric_split": "test_davis (DAVIS-all)",
            "artifact": str((args.probe_outputs_root / FIGURE_ARTIFACTS["figure_2"]).resolve()),
        },
        "appendix_figure_7": {
            "plot_script": str((viz_root / "plot_blockwise_head_comparison_panel.py").resolve()),
            "metric_split": "test_davis (DAVIS-all)",
            "artifact": str((args.probe_outputs_root / FIGURE_ARTIFACTS["appendix_figure_7"]).resolve()),
        },
        "figure_5a_and_appendix_figure_8": {
            "plot_scripts": [
                str((viz_root / "plot_isdynamic_data_starvation.py").resolve()),
                str((viz_root / "plot_isdynamic_data_starvation_panel.py").resolve()),
            ],
            "metric_split": "test_davis (DAVIS-all)",
            "figure_5a_artifact": str(
                (args.probe_outputs_root / FIGURE_ARTIFACTS["figure_5a"]).resolve()
            ),
            "note": (
                "These plots select the maximum per-block score and are provenance context only; "
                "the present analysis averages every PMD-selected block."
            ),
            "appendix_figure_8_artifact": str(
                (args.probe_outputs_root / FIGURE_ARTIFACTS["appendix_figure_8"]).resolve()
            ),
        },
        "table_12": {
            "paper_pdf": str((paper_root / "ProMo3D.pdf").resolve()),
            "evaluation_script": str((viz_root / "evaluate_motion_masks.py").resolve()),
            "variant": "bilinear_upsample (coarse, without SAM2)",
        },
    }


def descending_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for name, _value in ordered[index:end]:
            ranks[name] = rank
        index = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Pearson correlation requires paired values")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank_values(xs), rank_values(ys))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_rank(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[dict[str, Any]] = []
    per_block_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "analysis_script": str(Path(__file__).resolve()),
        "metric_semantics": (
            "Stored probe accuracy and confusion counts classify token pairs as the same "
            "or different motion state (pair_label_mode=same_label)."
        ),
        "normalized_gain_formula": (
            "(pair_accuracy - majority_pair_label_baseline) / "
            "(1 - majority_pair_label_baseline)"
        ),
        "majority_pair_label_baseline": (
            "max((tp + fn), (tn + fp)) / (tp + fp + fn + tn)"
        ),
        "primary_diagnostic_split": PRIMARY_SPLIT,
        "secondary_diagnostic_split": "test_davis (DAVIS-all)",
        "coarse_variant": "bilinear_upsample",
        "coarse_benchmarks": list(BENCHMARKS),
        "source_validation_mask_metric": (
            "YouTube-VOS validation coarse J-M/J-R before SAM2"
            if args.source_val_eval_root is not None
            else None
        ),
        "source_validation_eval_root": (
            str(args.source_val_eval_root.resolve())
            if args.source_val_eval_root is not None
            else None
        ),
        "figure_and_table_sources": figure_and_table_sources(args),
        "backbones": {},
    }

    for backbone, spec in BACKBONES.items():
        probe_paths = load_json(args.probe_paths_json) if args.probe_paths_json else {}
        probe_root = Path(probe_paths[backbone]) if backbone in probe_paths else args.probe_outputs_root / spec["probe_rel"]
        meta_path = find_mask_meta(
            args.mask_outputs_root, spec["mask_model"], spec["mask_group"]
        )
        meta = load_json(meta_path)
        usable_blocks = [str(value) for value in meta["usable_blocks"]]
        selected_groups = list(meta.get("groups") or [meta["group"]])

        block_records: list[dict[str, Any]] = []
        metric_paths: list[str] = []
        protocols: list[dict[str, Any]] = []
        for feature_key in usable_blocks:
            group = group_for_feature(feature_key, selected_groups)
            metrics_path = probe_root / group / feature_key / "metrics.json"
            metrics_payload = load_json(metrics_path)
            metric_paths.append(str(metrics_path.resolve()))
            protocol = probe_protocol(metrics_payload)
            if protocol["model_type"] != "pair_lowrank_quadratic" or protocol["rank"] != 8:
                raise ValueError(f"Expected rank-8 low-rank probe in {metrics_path}")
            protocols.append(protocol)
            record: dict[str, Any] = {
                "backbone": backbone,
                "group": group,
                "feature_key": feature_key,
                "metrics_path": str(metrics_path.resolve()),
            }
            for split in (PRIMARY_SPLIT, "test_davis"):
                metrics = metric_split(metrics_payload, split)
                gain, baseline, positive_rate = normalized_gain(metrics)
                record[f"{split}_normalized_gain"] = gain
                record[f"{split}_pair_accuracy"] = float(metrics["accuracy"])
                record[f"{split}_majority_pair_label_baseline"] = baseline
                record[f"{split}_same_state_pair_rate"] = positive_rate
            block_records.append(record)
            per_block_rows.append(record)
        comparable_protocols = [
            {key: value for key, value in protocol.items() if key != "input_dim"}
            for protocol in protocols
        ]
        if any(
            protocol != comparable_protocols[0]
            for protocol in comparable_protocols[1:]
        ):
            raise ValueError(f"Probe protocol changes across selected blocks for {backbone}")

        eval_paths: dict[str, str] = {}
        coarse_jm: dict[str, float] = {}
        coarse_jr: dict[str, float] = {}
        for benchmark in BENCHMARKS:
            eval_path = (
                args.mask_eval_root
                / spec["mask_model"]
                / spec["mask_group"]
                / "bilinear_upsample"
                / f"{benchmark}.json"
            )
            eval_payload = load_json(eval_path)
            eval_paths[benchmark] = str(eval_path.resolve())
            coarse_jm[benchmark] = float(eval_payload["metrics"]["J"]["M"])
            coarse_jr[benchmark] = float(eval_payload["metrics"]["J"]["R"])

        source_val_eval_path = None
        source_val_jm = None
        source_val_jr = None
        if args.source_val_eval_root is not None:
            source_val_eval_path = (
                args.source_val_eval_root
                / spec["mask_model"]
                / spec["mask_group"]
                / "bilinear_upsample"
                / "davis_all.json"
            )
            source_val_payload = load_json(source_val_eval_path)
            source_val_jm = float(source_val_payload["metrics"]["J"]["M"])
            source_val_jr = float(source_val_payload["metrics"]["J"]["R"])

        val_ng = summarize(
            [row[f"{PRIMARY_SPLIT}_normalized_gain"] for row in block_records]
        )
        val_accuracy = summarize(
            [row[f"{PRIMARY_SPLIT}_pair_accuracy"] for row in block_records]
        )
        davis_ng = summarize(
            [row["test_davis_normalized_gain"] for row in block_records]
        )
        davis_accuracy = summarize(
            [row["test_davis_pair_accuracy"] for row in block_records]
        )
        coarse_mean = statistics.fmean(coarse_jm.values())
        coarse_recall_mean = statistics.fmean(coarse_jr.values())
        aggregate = {
            "backbone": backbone,
            "selected_token_group": " + ".join(selected_groups),
            "selected_block_count": len(block_records),
            "youtubevos_val_ng_mean": val_ng["mean"],
            "youtubevos_val_ng_population_std": val_ng["population_std"],
            "youtubevos_val_ng_min": val_ng["min"],
            "youtubevos_val_ng_max": val_ng["max"],
            "youtubevos_val_accuracy_mean": val_accuracy["mean"],
            "youtubevos_val_accuracy_population_std": val_accuracy["population_std"],
            "youtubevos_val_accuracy_min": val_accuracy["min"],
            "youtubevos_val_accuracy_max": val_accuracy["max"],
            "test_davis_ng_mean": davis_ng["mean"],
            "test_davis_ng_population_std": davis_ng["population_std"],
            "test_davis_ng_min": davis_ng["min"],
            "test_davis_ng_max": davis_ng["max"],
            "test_davis_accuracy_mean": davis_accuracy["mean"],
            "test_davis_accuracy_population_std": davis_accuracy["population_std"],
            "test_davis_accuracy_min": davis_accuracy["min"],
            "test_davis_accuracy_max": davis_accuracy["max"],
            "davis2017_coarse_j_mean": coarse_jm["davis2017"],
            "davis2017_coarse_j_recall": coarse_jr["davis2017"],
            "segtrackv2_coarse_j_mean": coarse_jm["segtrackv2"],
            "segtrackv2_coarse_j_recall": coarse_jr["segtrackv2"],
            "fbms59_coarse_j_mean": coarse_jm["fbms59"],
            "fbms59_coarse_j_recall": coarse_jr["fbms59"],
            "three_benchmark_coarse_j_mean": coarse_mean,
            "three_benchmark_coarse_j_recall": coarse_recall_mean,
            "youtubevos_val_coarse_j_mean": source_val_jm,
            "youtubevos_val_coarse_j_recall": source_val_jr,
        }
        aggregates.append(aggregate)
        provenance["backbones"][backbone] = {
            "probe_root": str(probe_root.resolve()),
            "coarse_mask_meta": str(meta_path.resolve()),
            "selected_groups": selected_groups,
            "usable_blocks": usable_blocks,
            "metrics_files": metric_paths,
            "summary_file": str((probe_root.parent / "summary.json").resolve()),
            "probe_protocol": comparable_protocols[0],
            "input_dim_by_block": {
                record["feature_key"]: protocol["input_dim"]
                for record, protocol in zip(block_records, protocols)
            },
            "coarse_evaluation_files": eval_paths,
            "source_validation_evaluation_file": (
                str(source_val_eval_path.resolve())
                if source_val_eval_path is not None
                else None
            ),
        }

    ng_ranks = descending_ranks(
        {row["backbone"]: row["youtubevos_val_ng_mean"] for row in aggregates}
    )
    target_jm_ranks = descending_ranks(
        {row["backbone"]: row["three_benchmark_coarse_j_mean"] for row in aggregates}
    )
    source_jm_ranks = None
    if args.source_val_eval_root is not None:
        source_jm_ranks = descending_ranks(
            {row["backbone"]: row["youtubevos_val_coarse_j_mean"] for row in aggregates}
        )
    for row in aggregates:
        row["rank_by_youtubevos_val_ng"] = ng_ranks[row["backbone"]]
        row["rank_by_target_coarse_j_mean"] = target_jm_ranks[row["backbone"]]
        row["rank_by_youtubevos_val_coarse_j_mean"] = (
            source_jm_ranks[row["backbone"]] if source_jm_ranks is not None else None
        )

    aggregate_by_name = {row["backbone"]: row for row in aggregates}
    correlation_rows: list[dict[str, Any]] = []

    def append_correlations(
        *,
        diagnostic_label: str,
        diagnostic_split: str,
        diagnostic_key: str,
        outcome_label: str,
        outcome_split: str,
        outcome_key: str,
    ) -> None:
        for included_label, names in (
            ("all 6", tuple(BACKBONES)),
            ("5 multi-view only", MULTIVIEW_BACKBONES),
        ):
            xs = [aggregate_by_name[name][diagnostic_key] for name in names]
            ys = [aggregate_by_name[name][outcome_key] for name in names]
            correlation_rows.append(
                {
                    "diagnostic": diagnostic_label,
                    "diagnostic_split": diagnostic_split,
                    "outcome": outcome_label,
                    "outcome_split": outcome_split,
                    "backbones_included": included_label,
                    "backbone_names": ",".join(names),
                    "spearman": spearman(xs, ys),
                    "pearson": pearson(xs, ys),
                }
            )

    for diagnostic_split, diagnostic_key, diagnostic_label in (
        (PRIMARY_SPLIT, "youtubevos_val_ng_mean", "Mean Normalized Gain"),
        (PRIMARY_SPLIT, "youtubevos_val_accuracy_mean", "Mean pairwise compatibility accuracy"),
        ("test_davis", "test_davis_ng_mean", "Mean Normalized Gain"),
        ("test_davis", "test_davis_accuracy_mean", "Mean pairwise compatibility accuracy"),
    ):
        for outcome_key, outcome_label in (
            ("three_benchmark_coarse_j_mean", "Three-benchmark coarse mean J-M"),
            ("three_benchmark_coarse_j_recall", "Three-benchmark coarse mean J-R"),
        ):
            append_correlations(
                diagnostic_label=diagnostic_label,
                diagnostic_split=diagnostic_split,
                diagnostic_key=diagnostic_key,
                outcome_label=outcome_label,
                outcome_split="DAVIS17/SegTrackv2/FBMS-59",
                outcome_key=outcome_key,
            )

    if args.source_val_eval_root is not None:
        for diagnostic_key, diagnostic_label in (
            ("youtubevos_val_ng_mean", "Mean Normalized Gain"),
            ("youtubevos_val_accuracy_mean", "Mean pairwise compatibility accuracy"),
        ):
            for outcome_key, outcome_label in (
                ("youtubevos_val_coarse_j_mean", "Coarse J-M"),
                ("youtubevos_val_coarse_j_recall", "Coarse J-R"),
            ):
                append_correlations(
                    diagnostic_label=diagnostic_label,
                    diagnostic_split=PRIMARY_SPLIT,
                    diagnostic_key=diagnostic_key,
                    outcome_label=outcome_label,
                    outcome_split="youtubevos_val",
                    outcome_key=outcome_key,
                )
        for diagnostic_key, diagnostic_label in (
            ("youtubevos_val_coarse_j_mean", "Coarse J-M"),
            ("youtubevos_val_coarse_j_recall", "Coarse J-R"),
        ):
            for outcome_key, outcome_label in (
                ("three_benchmark_coarse_j_mean", "Three-benchmark coarse mean J-M"),
                ("three_benchmark_coarse_j_recall", "Three-benchmark coarse mean J-R"),
            ):
                append_correlations(
                    diagnostic_label=diagnostic_label,
                    diagnostic_split="youtubevos_val",
                    diagnostic_key=diagnostic_key,
                    outcome_label=outcome_label,
                    outcome_split="DAVIS17/SegTrackv2/FBMS-59",
                    outcome_key=outcome_key,
                )

    write_csv(args.output_dir / "backbone_probe_diagnostic_summary.csv", aggregates)
    write_csv(args.output_dir / "backbone_probe_diagnostic_blocks.csv", per_block_rows)
    write_csv(args.output_dir / "backbone_probe_diagnostic_correlations.csv", correlation_rows)

    output_json = {
        **provenance,
        "aggregates": aggregates,
        "per_block": per_block_rows,
        "correlations": correlation_rows,
    }
    with (args.output_dir / "backbone_probe_diagnostic.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output_json, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.source_val_eval_root is not None:
        table_lines = [
            "| Backbone | Selected token group | YT-VOS-val NG, mean ± std. | YT-VOS-val coarse J-M/J-R | Target coarse mean J-M/J-R |",
            "|---|---|---:|---:|---:|",
        ]
        sort_key = "rank_by_youtubevos_val_coarse_j_mean"
    else:
        table_lines = [
            "| Backbone | Selected token group | YT-VOS-val NG, mean ± std. | Target coarse mean J-M |",
            "|---|---|---:|---:|",
        ]
        sort_key = "rank_by_youtubevos_val_ng"

    for row in sorted(aggregates, key=lambda value: value[sort_key]):
        common = (
            f"| {row['backbone']} | {row['selected_token_group']} "
            f"({row['selected_block_count']}) | "
            f"{100.0 * row['youtubevos_val_ng_mean']:.2f} ± "
            f"{100.0 * row['youtubevos_val_ng_population_std']:.2f} |"
        )
        if args.source_val_eval_root is not None:
            table_lines.append(
                f"{common} {100.0 * row['youtubevos_val_coarse_j_mean']:.2f}/"
                f"{100.0 * row['youtubevos_val_coarse_j_recall']:.2f} | "
                f"{100.0 * row['three_benchmark_coarse_j_mean']:.2f}/"
                f"{100.0 * row['three_benchmark_coarse_j_recall']:.2f} |"
            )
        else:
            table_lines.append(
                f"{common} {100.0 * row['three_benchmark_coarse_j_mean']:.2f} |"
            )

    correlation_lines = [
        "| Diagnostic | Diagnostic split | Outcome | Outcome split | Backbones | Spearman | Pearson |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in correlation_rows:
        if args.source_val_eval_root is None and row["diagnostic_split"] != PRIMARY_SPLIT:
            continue
        if args.source_val_eval_root is not None and (
            row["diagnostic_split"] != "youtubevos_val"
            or row["diagnostic"] not in {"Mean Normalized Gain", "Coarse J-M", "Coarse J-R"}
        ):
            continue
        correlation_lines.append(
            f"| {row['diagnostic']} | {row['diagnostic_split']} | {row['outcome']} | "
            f"{row['outcome_split']} | {row['backbones_included']} | "
            f"{row['spearman']:.3f} | {row['pearson']:.3f} |"
        )

    markdown = [
        "# Cross-backbone representation diagnostic",
        "",
        "The representation diagnostic and source mask metrics use the same 80-video "
        "YouTube-VOS validation split. No target benchmark is used for backbone selection.",
        "",
        *table_lines,
        "",
        "Normalized Gain is computed as (pair accuracy - majority pair-label baseline) / "
        "(1 - majority pair-label baseline), using pairwise same/different-motion-state "
        "classification. Source-validation coarse J-M is the primary operational selection "
        "metric; J-R is secondary because it thresholds each frame at J > 0.5.",
        "",
        *correlation_lines,
        "",
        "Target coarse mean J-M is retained only to test whether a source-validation "
        "diagnostic transfers to DAVIS17, SegTrackv2, and FBMS-59.",
        "",
    ]
    (args.output_dir / "backbone_probe_diagnostic.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
