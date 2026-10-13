from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .figure_io import save_vector_figure
from .plot_isdynamic_blockwise_results import (
    ACCURACY_YLABEL,
    BASELINE_LEGEND_LABEL,
    _always_true_baseline,
    _group_title,
    _legend_label_with_context,
    _line_alpha,
    _line_color,
    _line_marker,
    _line_markersize,
    _line_style,
    _line_width,
    _line_zorder,
    _merge_summaries,
    _metric_value,
    _ordered_model_types,
)


PANEL_SPECS = (
    (
        "pi3x",
        Path("outputs/model_outputs/pi3/isdynamic_blockwise_youtubevos_pi3x/summary.json"),
        Path("outputs/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/summary.json"),
        ("inter_frame_block", "intra_frame_block", "cam_head"),
    ),
    (
        "depth_anything_3",
        Path("outputs/model_outputs/depth_anything_3/isdynamic_blockwise_youtubevos/summary.json"),
        Path("outputs/model_outputs/depth_anything_3/rank_8/isdynamic_blockwise_youtubevos/summary.json"),
        ("inter_frame_block", "intra_frame_block", "depth_head"),
    ),
    (
        "dust3r",
        Path("outputs/model_outputs/dust3r/isdynamic_blockwise_youtubevos/summary.json"),
        Path("outputs/model_outputs/dust3r/rank_8/isdynamic_blockwise_youtubevos/summary.json"),
        ("dec_view1", "dec_view2", "head_in_view1"),
    ),
)

DEFAULT_MODEL_TYPES = (
    "pair_linear",
    "pair_diag_quadratic",
    "pair_lowrank_quadratic@@k=8",
)

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 27
YLABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a widescreen 3x3 blockwise comparison panel.")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "f1", "precision", "recall", "loss"])
    parser.add_argument("--metric_split", default="test_davis")
    parser.add_argument("--baseline_mode", default="always_true", choices=["always_true", "constant", "none"])
    parser.add_argument("--baseline_value", type=float, default=0.5)
    parser.add_argument(
        "--model_types",
        default=",".join(DEFAULT_MODEL_TYPES),
        help="Comma-separated model types to include, default: linear, diagonal quad, and low-rank quad (k=8).",
    )
    parser.add_argument(
        "--out_path",
        default="outputs/isdynamic_visualizations/comparisons/blockwise/pi3x_da3_dust3r_blockwise_widescreen_panel.pdf",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _model_title(model_name: str) -> str:
    return {
        "pi3x": r"$\mathbf{\pi^3X}$",
        "depth_anything_3": r"$\mathbf{DA3}$",
        "dust3r": r"$\mathbf{DUSt3R}$",
    }.get(model_name, model_name)
def _panel_legend_label(model_type: str) -> str:
    mapping = {
        "pair_linear": "linear",
        "pair_diag_quadratic": "diagonal quad",
        "pair_lowrank_quadratic@@k=8": "low-rank quad",
    }
    return mapping.get(model_type, _legend_label_with_context(model_type, [model_type]).replace(" (k=8)", ""))


def _panel_color(model_type: str, model_types: list[str]) -> str:
    if model_type == "pair_lowrank_quadratic@@k=8":
        return "#d62728"
    return _line_color(model_type, model_types)


def _load_summary(primary_path: Path, rank8_path: Path) -> dict:
    return _merge_summaries([primary_path, rank8_path], ["", "k=8"])


def main() -> None:
    args = parse_args()
    requested_model_types = _parse_csv(args.model_types) or list(DEFAULT_MODEL_TYPES)
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = {
        model: _load_summary(primary_path.resolve(), rank8_path.resolve())
        for model, primary_path, rank8_path, _ in PANEL_SPECS
    }
    baseline_summaries = {
        model: _merge_summaries([primary_path.resolve()], [""])
        for model, primary_path, _rank8_path, _ in PANEL_SPECS
    }

    fig, axes = plt.subplots(3, 3, figsize=(22.0, 9.8), squeeze=False)
    legend_handle_by_label = {}
    row_y_values: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for row_idx, (model_name, _primary_summary_path, _rank8_summary_path, groups) in enumerate(PANEL_SPECS):
        summary = summaries[model_name]
        baseline_summary = baseline_summaries[model_name]
        results = summary.get("results", [])
        baseline_results = baseline_summary.get("results", [])
        model_types = [mt for mt in _ordered_model_types(list(summary.get("model_types", []))) if mt in requested_model_types]
        for col_idx, group in enumerate(groups):
            ax = axes[row_idx][col_idx]
            group_items = [r for r in results if r.get("group") == group]
            baseline_group_items = [r for r in baseline_results if r.get("group") == group]
            for model_type in model_types:
                items = [r for r in group_items if r.get("model_type") == model_type]
                if not items:
                    continue
                items.sort(key=lambda x: int(x.get("block_idx", -1)))
                xs = [int(x["block_idx"]) for x in items]
                ys = [_metric_value(x, args.metric, args.metric_split) for x in items]
                line = ax.plot(
                    xs,
                    ys,
                    marker=_line_marker(model_type, model_types),
                    linewidth=_line_width(model_type, model_types),
                    markersize=_line_markersize(model_type, model_types) + 1.0,
                    linestyle=_line_style(model_type, model_types),
                    label=_panel_legend_label(model_type),
                    color=_panel_color(model_type, model_types),
                    alpha=_line_alpha(model_type, model_types),
                    zorder=_line_zorder(model_type, model_types),
                )[0]
                legend_handle_by_label.setdefault(_panel_legend_label(model_type), line)
                row_y_values[row_idx].extend([y for y in ys if y == y])

            baseline_value = None
            if args.baseline_mode == "constant":
                baseline_value = float(args.baseline_value)
            elif args.baseline_mode == "always_true":
                baseline_value = _always_true_baseline(baseline_group_items, args.metric_split)
            if baseline_value is not None and baseline_value == baseline_value:
                ax.axhline(
                    baseline_value,
                    color="gray",
                    linestyle="--",
                    linewidth=2.6,
                    alpha=0.9,
                    zorder=10.0,
                )
                row_y_values[row_idx].append(float(baseline_value))

            ax.set_title(f"{_model_title(model_name)} | {_group_title(group, model_name)}", fontsize=TITLE_FONTSIZE)
            if row_idx == len(PANEL_SPECS) - 1:
                ax.set_xlabel("Block index", fontsize=LABEL_FONTSIZE)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ylabel = ACCURACY_YLABEL if args.metric == "accuracy" else args.metric.title()
                ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.25)

    for row_idx, row in enumerate(axes):
        if not row_y_values[row_idx]:
            continue
        y_min = min(row_y_values[row_idx])
        y_max = max(row_y_values[row_idx])
        pad = max(1e-3, (y_max - y_min) * 0.08) if y_min != y_max else max(1e-3, abs(y_min) * 0.05)
        for ax in row:
            ax.set_ylim(y_min - pad, y_max + pad)

    label_to_handle = dict(legend_handle_by_label)
    ordered_handles = []
    ordered_text = []
    for model_type in requested_model_types:
        label = _panel_legend_label(model_type)
        handle = label_to_handle.get(label)
        if handle is not None:
            ordered_handles.append(handle)
            ordered_text.append(label)
    if args.baseline_mode != "none":
        ordered_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=2.6,
                alpha=0.9,
                label=BASELINE_LEGEND_LABEL,
            )
        )
        ordered_text.append(BASELINE_LEGEND_LABEL)
    fig.legend(
        ordered_handles,
        ordered_text,
        loc="upper center",
        ncol=max(1, len(ordered_handles)),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 1.005),
        markerscale=1.2,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    save_vector_figure(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved comparison panel to {out_path}")


if __name__ == "__main__":
    main()
