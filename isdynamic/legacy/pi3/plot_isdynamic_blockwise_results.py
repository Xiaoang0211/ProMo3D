#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from isdynamic.viz.figure_io import save_vector_figure

MODEL_COLOR_MAP = {
    "pair_linear": "#1f77b4",
    "pair_diag_quadratic": "#ff7f0e",
    "pair_quadratic": "#2ca02c",
    "pair_lowrank_quadratic": "#d62728",
    "pair_lowrank_quadratic_strict": "#9467bd",
    "pair_attention": "#8c564b",
    "pair_value": "#e377c2",
    "pair_attention_biased_quadratic": "#7f7f7f",
    "pair_value_biased_quadratic": "#bcbd22",
    "pair_attention_biased_lowrank_quadratic": "#17becf",
    "pair_value_biased_lowrank_quadratic": "#aec7e8",
    "pair_attention_biased_lowrank_quadratic_strict": "#ffbb78",
    "pair_value_biased_lowrank_quadratic_strict": "#98df8a",
}

FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot blockwise IsDynamic probe sweep results (5 groups) from summary JSON."
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="outputs/isdynamic_blockwise/summary.json",
        help="Path or comma-separated list of summary.json files to merge before plotting.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
        help="Metric from each result.best to plot on y-axis.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="inter_frame_block,intra_frame_block,cam_head,conf_head,point_head",
        help="Comma-separated group order for subplots.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="",
        help="Optional comma-separated model order; default uses summary model_types.",
    )
    parser.add_argument("--out_png", type=str, default="outputs/isdynamic_blockwise/blockwise_accuracy.png")
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="always_true",
        choices=["always_true", "constant", "none"],
        help=(
            "How to draw baseline: "
            "'always_true' uses positive-pair ratio from metrics (trivial all-positive predictor), "
            "'constant' uses --baseline_value, 'none' disables baseline."
        ),
    )
    parser.add_argument("--baseline_value", type=float, default=0.5)
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_summaries(summary_paths: List[Path]) -> Dict:
    model_types: List[str] = []
    results: List[Dict] = []
    seen = set()

    for path in summary_paths:
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")
        summary = _load_json(path)

        for mt in summary.get("model_types", []):
            if mt not in model_types:
                model_types.append(mt)

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

    return {
        "model_types": model_types,
        "results": results,
    }


def _always_true_baseline(group_items: List[Dict]) -> float:
    ratios: List[float] = []
    for item in group_items:
        best = item.get("best", {})
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


def _legend_label(model_type: str) -> str:
    label = model_type[len("pair_") :] if model_type.startswith("pair_") else model_type
    return label.replace("biased", "initialized")


def _line_color(model_type: str) -> str:
    if model_type in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[model_type]
    digest = hashlib.sha1(model_type.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def _group_title(group: str) -> str:
    return "intra_frame_block" if group == "intra_frame_block" else group


def main() -> None:
    args = parse_args()

    summary_paths = [Path(p) for p in _parse_csv(args.summary_json)]
    if not summary_paths:
        raise ValueError("--summary_json resolved to empty list.")
    summary = _merge_summaries(summary_paths)

    groups = _parse_csv(args.groups)
    model_types = _parse_csv(args.model_types)
    if not model_types:
        model_types = list(summary.get("model_types", []))
    if not model_types:
        raise ValueError("No model_types available from args or summary.")

    results = summary.get("results", [])

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.2 * n_groups, 4.2), squeeze=False)
    axes = axes[0]

    for ax, group in zip(axes, groups):
        group_items = [r for r in results if r.get("group") == group]

        for model_type in model_types:
            items = [r for r in group_items if r.get("model_type") == model_type]
            if not items:
                continue
            items.sort(key=lambda x: int(x.get("block_idx", -1)))
            xs = [int(x["block_idx"]) for x in items]
            ys = [float(x.get("best", {}).get(args.metric, float("nan"))) for x in items]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.7,
                markersize=3.8,
                label=_legend_label(model_type),
                color=_line_color(model_type),
            )

        baseline_value = None
        if args.baseline_mode == "constant":
            baseline_value = float(args.baseline_value)
        elif args.baseline_mode == "always_true":
            baseline_value = _always_true_baseline(group_items)

        if baseline_value is not None:
            ax.axhline(
                baseline_value,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="baseline" if group == groups[0] else None,
            )

        ax.set_title(_group_title(group))
        ax.set_xlabel("Block Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.25)

    ylabel = r"$\mathrm{Acc}_{\mathrm{dyn}}$" if args.metric == "accuracy" else f"IsDynamic {args.metric.title()}"
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = Path(args.out_png)
    saved_paths = save_vector_figure(fig, out_path, dpi=220)
    plt.close(fig)

    print("Saved plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
