from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .figure_io import save_vector_figure
from .plot_isdynamic_data_starvation import (
    build_data_starvation_legend_handles,
    build_parser,
    collect_data_starvation_series,
    plot_data_starvation_axis,
)


PANEL_SPECS = (
    ("pi3x", "inter_frame_block"),
    ("da3", "inter_frame_block"),
    ("dust3r", "dec_view1"),
)


def parse_args() -> argparse.Namespace:
    parser = build_parser(require_model=False)
    parser.set_defaults(
        out_dir="outputs/isdynamic_visualizations/comparisons/data_starvation",
        seeds_csv="all",
        include_fullrank=True,
        show_std_band=False,
        show_all_positive_baseline=False,
    )
    return parser.parse_args()


def main() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(19.4, 4.8), squeeze=False)
    global_ymin = None
    global_ymax = None
    for ax, (model, group) in zip(axes[0], PANEL_SPECS):
        _, series, all_positive_baselines, ordered_sizes = collect_data_starvation_series(
            model=model,
            outputs_root=args.outputs_root,
            sizes_csv=args.sizes_csv,
            seeds_csv=args.seeds_csv,
            group=group,
            test_split=args.test_split,
            metric=args.metric,
            lowrank_rank=args.lowrank_rank,
            lowrank_ranks_csv=args.lowrank_ranks_csv,
            result_subpath_prefix=args.result_subpath_prefix,
        )
        plot_data_starvation_axis(
            ax,
            model=model,
            group=group,
            test_split=args.test_split,
            metric=args.metric,
            series=series,
            all_positive_baselines=all_positive_baselines,
            ordered_sizes=ordered_sizes,
            show_baseline=args.show_all_positive_baseline,
            include_fullrank=args.include_fullrank,
            show_std_band=args.show_std_band,
            title_mode="panel",
        )
        ymin, ymax = ax.get_ylim()
        global_ymin = ymin if global_ymin is None else min(global_ymin, ymin)
        global_ymax = ymax if global_ymax is None else max(global_ymax, ymax)

    if global_ymin is not None and global_ymax is not None:
        if args.metric in {"ng", "norm_gain", "normalized_gain"}:
            pad = max(0.01, 0.08 * (global_ymax - global_ymin if global_ymax > global_ymin else 0.1))
            global_ymin = max(0.0, global_ymin - pad)
            global_ymax = min(1.0, global_ymax + pad)
            tick_start = 0.4 if args.show_std_band else 0.2
            tick_step = 0.1
            tick_end = max(tick_start, math.floor((global_ymax + 1e-9) / tick_step) * tick_step)
            yticks = [round(tick_start + i * tick_step, 1) for i in range(int(round((tick_end - tick_start) / tick_step)) + 1)]
        else:
            yticks = None
        for idx, ax in enumerate(axes[0]):
            ax.set_ylim(global_ymin, global_ymax)
            if yticks is not None:
                ax.set_yticks(yticks)
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.tick_params(axis="y", labelleft=True)
            ax.tick_params(axis="both", labelsize=22)
            ax.xaxis.label.set_size(22)
            if idx == 0:
                ax.yaxis.label.set_size(22)
            ax.title.set_fontsize(22)

    fig.legend(
        handles=build_data_starvation_legend_handles(
            include_baseline=(args.metric == "accuracy" and args.show_all_positive_baseline),
            include_fullrank=args.include_fullrank,
        ),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=23,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    suffix = "normalized_gain_k8_stats_panel" if args.show_std_band else f"{args.metric}_panel"
    plot_path = out_dir / f"pi3x_da3_dust3r_{args.test_split}_{suffix}.pdf"
    saved_paths = save_vector_figure(fig, plot_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved panel plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
