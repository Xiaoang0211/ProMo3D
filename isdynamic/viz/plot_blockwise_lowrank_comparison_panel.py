from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .figure_io import save_vector_figure


MODEL_SPECS = {
    "pi3x": {
        "label": r"$\mathbf{\pi^3X}$",
        "group": "inter_frame_block",
        "group_title": "Inter-Frame Block",
        "base_summary": "outputs/model_outputs/pi3/isdynamic_blockwise_youtubevos_pi3x/summary.json",
        "rank8_summary": "outputs/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/summary.json",
    },
    "da3": {
        "label": r"$\mathbf{DA3}$",
        "group": "inter_frame_block",
        "group_title": "Inter-Frame Block",
        "base_summary": "outputs/model_outputs/depth_anything_3/isdynamic_blockwise_youtubevos/summary.json",
        "rank8_summary": "outputs/model_outputs/depth_anything_3/rank_8/isdynamic_blockwise_youtubevos/summary.json",
    },
    "dust3r": {
        "label": r"$\mathbf{DUSt3R}$",
        "group": "dec_view1",
        "group_title": "Decoder View-1",
        "base_summary": "outputs/model_outputs/dust3r/isdynamic_blockwise_youtubevos/summary.json",
        "rank8_summary": "outputs/model_outputs/dust3r/rank_8/isdynamic_blockwise_youtubevos/summary.json",
    },
}


SERIES_STYLE = {
    "k=256": {
        "label": "low-rank quad (k=256)",
        "color": "#d62728",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 4.5,
    },
    "k=8": {
        "label": "low-rank quad (k=8)",
        "color": "#9467bd",
        "linestyle": (0, (5, 1, 1, 1)),
        "marker": "X",
        "linewidth": 2.1,
        "markersize": 4.8,
    },
}

BASELINE_STYLE = {
    "label": "all-positive baseline",
    "color": "gray",
    "linestyle": "--",
    "linewidth": 1.5,
}
ACCURACY_YLABEL = r"$\mathrm{Acc}_{\mathrm{dyn}}$"

NG_SERIES_STYLE = {
    "k=256": {
        "label": "low-rank quad (k=256, NG)",
        "color": "#222222",
        "linestyle": "-.",
        "linewidth": 1.8,
    },
    "k=8": {
        "label": "low-rank quad (k=8, NG)",
        "color": "#7f7f7f",
        "linestyle": ":",
        "linewidth": 2.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a cross-model blockwise panel comparing low-rank quadratic probes at k=256 and k=8."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss", "ng", "norm_gain", "normalized_gain", "both"],
    )
    parser.add_argument("--metric_split", type=str, default="test_davis")
    parser.add_argument("--out_path", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_ng_metric(metric: str) -> bool:
    return metric in {"ng", "norm_gain", "normalized_gain"}


def _default_out_path(metric: str) -> Path:
    base = Path(
        "outputs/isdynamic_visualizations/comparisons/blockwise/lowrank_k256_vs_k8/"
        "pi3x_da3_dust3r_blockwise_lowrank_k256_k8_panel.pdf"
    )
    if _is_ng_metric(metric):
        return base.with_name("pi3x_da3_dust3r_blockwise_lowrank_k256_k8_panel_normalized_gain.pdf")
    if metric == "both":
        return base.with_name("pi3x_da3_dust3r_blockwise_lowrank_k256_k8_panel_acc_and_normalized_gain.pdf")
    return base


def _extract_block_idx(row: Dict) -> int | None:
    block_idx = row.get("block_idx")
    if isinstance(block_idx, int):
        return block_idx
    if isinstance(block_idx, float):
        return int(block_idx)
    feature_key = row.get("feature_key")
    if isinstance(feature_key, str):
        try:
            return int(feature_key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _metric_value(row: Dict, metric: str, metric_split: str) -> float | None:
    source = _split_metrics(row, metric_split)
    if metric in source:
        return float(source[metric])
    if metric in {"ng", "norm_gain", "normalized_gain"}:
        accuracy = source.get("accuracy")
        tp = source.get("tp")
        fp = source.get("fp")
        fn = source.get("fn")
        tn = source.get("tn")
        if None in (accuracy, tp, fp, fn, tn):
            return None
        total = float(tp) + float(fp) + float(fn) + float(tn)
        if total <= 0:
            return None
        baseline = (float(tp) + float(fn)) / total
        denom = 1.0 - baseline
        if denom <= 0:
            return None
        return (float(accuracy) - baseline) / denom
    if metric_split in {"val", "best"}:
        best_val = row.get("best_val")
        return float(best_val) if best_val is not None else None
    return None


def _split_metrics(row: Dict, metric_split: str) -> Dict:
    if metric_split in {"val", "best"}:
        source = row.get("best")
        return source if isinstance(source, dict) else {}

    source = row.get("best_tests")
    if not isinstance(source, dict):
        return {}
    if metric_split in source and isinstance(source[metric_split], dict):
        return source[metric_split]
    for split_name, split_metrics in source.items():
        if split_name == metric_split or str(split_name).startswith(f"{metric_split}/"):
            return split_metrics if isinstance(split_metrics, dict) else {}
    return {}


def _all_positive_baseline(summary_path: Path, *, group: str, metric_split: str) -> float:
    blob = _load_json(summary_path)
    rows = blob.get("results")
    if not isinstance(rows, list) or not rows:
        rows = blob.get("rows", [])

    ratios: List[float] = []
    for row in rows:
        if row.get("group") != group:
            continue
        best = _split_metrics(row, metric_split)
        try:
            tp = float(best.get("tp", 0.0))
            fp = float(best.get("fp", 0.0))
            fn = float(best.get("fn", 0.0))
            tn = float(best.get("tn", 0.0))
        except (TypeError, ValueError):
            continue
        total = tp + fp + fn + tn
        if total > 0:
            ratios.append((tp + fn) / total)
    if not ratios:
        return float("nan")
    return sum(ratios) / len(ratios)


def _extract_series(summary_path: Path, *, group: str, metric: str, metric_split: str) -> Tuple[List[int], List[float]]:
    blob = _load_json(summary_path)
    rows = blob.get("results")
    if not isinstance(rows, list) or not rows:
        rows = blob.get("rows", [])

    points: List[Tuple[int, float]] = []
    for row in rows:
        if row.get("group") != group:
            continue
        if row.get("model_type") != "pair_lowrank_quadratic":
            continue
        block_idx = _extract_block_idx(row)
        metric_value = _metric_value(row, metric, metric_split)
        if block_idx is None or metric_value is None:
            continue
        points.append((block_idx, metric_value))

    if not points:
        raise ValueError(f"No usable low-rank quadratic rows found in {summary_path} for group={group}")

    points.sort(key=lambda item: item[0])
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return xs, ys


def _plot_metric_row(
    axes,
    *,
    metric: str,
    metric_split: str,
    show_titles: bool,
    include_baseline_in_legend: bool = True,
) -> Tuple[List[object], List[str]]:
    from matplotlib.ticker import MaxNLocator

    legend_handles = []
    legend_labels = []
    global_y: List[float] = []

    for ax, model_key in zip(axes, ["pi3x", "da3", "dust3r"]):
        spec = MODEL_SPECS[model_key]
        base_path = Path(spec["base_summary"])
        rank8_path = Path(spec["rank8_summary"])

        k256_x, k256_y = _extract_series(
            base_path,
            group=spec["group"],
            metric=metric,
            metric_split=metric_split,
        )
        k8_x, k8_y = _extract_series(
            rank8_path,
            group=spec["group"],
            metric=metric,
            metric_split=metric_split,
        )
        baseline_y = 0.0 if _is_ng_metric(metric) else _all_positive_baseline(
            base_path,
            group=spec["group"],
            metric_split=metric_split,
        )

        global_y.extend(k256_y)
        global_y.extend(k8_y)
        if baseline_y == baseline_y:
            global_y.append(baseline_y)

        for series_name, xs, ys in [("k=256", k256_x, k256_y), ("k=8", k8_x, k8_y)]:
            style = SERIES_STYLE[series_name]
            line = ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                label=style["label"],
            )[0]
            if style["label"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(style["label"])

        if baseline_y == baseline_y:
            baseline = ax.axhline(
                baseline_y,
                color=BASELINE_STYLE["color"],
                linestyle=BASELINE_STYLE["linestyle"],
                linewidth=BASELINE_STYLE["linewidth"],
                label=BASELINE_STYLE["label"],
            )
            if include_baseline_in_legend and BASELINE_STYLE["label"] not in legend_labels:
                legend_handles.append(baseline)
                legend_labels.append(BASELINE_STYLE["label"])

        if show_titles:
            ax.set_title(f"{spec['label']} | {spec['group_title']}", fontsize=20, pad=8)
        ax.set_xlabel("Block Index", fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.4)

    y_min = min(global_y)
    y_max = max(global_y)
    pad = max(0.01, 0.08 * (y_max - y_min))
    lower = max(0.0, y_min - pad)
    upper = min(1.0, y_max + pad)

    for idx, ax in enumerate(axes):
        ax.set_ylim(lower, upper)
        if idx == 0:
            ylabel = "Normalized Gain" if _is_ng_metric(metric) else ACCURACY_YLABEL
            ax.set_ylabel(ylabel, fontsize=20)
        else:
            ax.set_ylabel("")

    return legend_handles, legend_labels


def main() -> None:
    args = parse_args()
    if args.out_path is None:
        args.out_path = _default_out_path(args.metric)

    import matplotlib.pyplot as plt

    if args.metric == "both":
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 6.4), squeeze=False)
        legend_handles, legend_labels = _plot_metric_row(
            axes[0],
            metric="accuracy",
            metric_split=args.metric_split,
            show_titles=True,
            include_baseline_in_legend=True,
        )
        _plot_metric_row(
            axes[1],
            metric="normalized_gain",
            metric_split=args.metric_split,
            show_titles=False,
            include_baseline_in_legend=False,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=21,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=1.1, w_pad=1.0)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(19.4, 4.8), squeeze=False)
        legend_handles, legend_labels = _plot_metric_row(
            axes[0],
            metric=args.metric,
            metric_split=args.metric_split,
            show_titles=True,
            include_baseline_in_legend=not _is_ng_metric(args.metric),
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=21,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    save_vector_figure(fig, args.out_path, dpi=220, bbox_inches="tight", pad_inches=0.03, suffixes=(".pdf", ".svg"))
    plt.close(fig)

    print(f"Saved comparison panel PDF: {args.out_path.with_suffix('.pdf')}")
    print(f"Saved comparison panel SVG: {args.out_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
