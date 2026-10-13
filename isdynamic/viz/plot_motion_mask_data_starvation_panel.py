from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .figure_io import save_vector_figure
from .plot_motion_mask_data_starvation import (
    BENCHMARK_DISPLAY_NAMES,
    _collect_series,
    _parse_csv_values,
    _series_points,
    _legend_handles,
    _metric_display_name,
)

DATASET_TITLE_FONTSIZE = 26
DATASET_BOX_FONTSIZE = 23
DEFAULT_BENCHMARKS = ("davis_all", "segtrackv2", "fbms59")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 3-panel motion-mask data-starvation figure for DAVIS-all, SegTrackv2, and FBMS-59."
    )
    parser.add_argument(
        "--data_starvation_eval_root",
        default="outputs/isdynamic_motion_masks_eval_data_starvation_k8",
    )
    parser.add_argument(
        "--full_eval_root",
        default="outputs/isdynamic_motion_masks_eval",
    )
    parser.add_argument("--model", default="pi3x")
    parser.add_argument("--group", default="inter_frame_block")
    parser.add_argument("--benchmarks_csv", default="davis_all,segtrackv2,fbms59")
    parser.add_argument("--metric", default="J", choices=["J", "F", "J_and_F"])
    parser.add_argument("--stat", default="M", choices=["M", "R", "D"])
    parser.add_argument("--sizes_csv", default="160,80,40,20,10")
    parser.add_argument("--seeds_csv", default="42")
    parser.add_argument(
        "--variants_csv",
        default="bilinear_upsample,sam2_prompt_otsu_off0.2_erode2_w30_mean_otsu_off0.3",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/isdynamic_visualizations/comparisons/data_starvation",
    )
    return parser.parse_args()


def main() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    args = parse_args()
    data_starvation_eval_root = Path(args.data_starvation_eval_root).resolve()
    full_eval_root = Path(args.full_eval_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_csv_values(args.sizes_csv, int)
    seeds = _parse_csv_values(args.seeds_csv, int)
    variants = _parse_csv_values(args.variants_csv, str)
    benchmarks = _parse_csv_values(args.benchmarks_csv, str)

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(19.4, 4.8), squeeze=False)
    axes_row = axes[0]
    global_ymin = None
    global_ymax = None

    for ax, benchmark in zip(axes_row, benchmarks):
        rows, series, ordered_sizes = _collect_series(
            data_starvation_eval_root=data_starvation_eval_root,
            full_eval_root=full_eval_root,
            model=args.model,
            group=args.group,
            benchmark=benchmark,
            metric=args.metric,
            stat=args.stat,
            sizes=sizes,
            seeds=seeds,
            variants=variants,
        )
        if not rows:
            raise SystemExit(f"No matching results for benchmark={benchmark}")

        x_values = [0.2 * (idx + 1) for idx in range(len(ordered_sizes))]
        x_positions = {size: x for size, x in zip(ordered_sizes, x_values)}
        for variant in variants:
            xs, ys, _ = _series_points(series, variant, ordered_sizes, x_positions)
            if not xs:
                continue
            style = next((h for h in _legend_handles([variant])), None)
            line_style = {
                "color": style.get_color(),
                "marker": style.get_marker(),
                "linestyle": style.get_linestyle(),
                "linewidth": style.get_linewidth(),
                "markersize": style.get_markersize(),
            }
            ax.plot(xs, ys, **line_style)

        ax.set_xticks(x_values, [str(size) for size in ordered_sizes])
        ax.set_xlim(0.14, max(x_values) + 0.06)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=20)
        ymin, ymax = ax.get_ylim()
        global_ymin = ymin if global_ymin is None else min(global_ymin, ymin)
        global_ymax = ymax if global_ymax is None else max(global_ymax, ymax)

    for idx, ax in enumerate(axes_row):
        if global_ymin is not None and global_ymax is not None:
            ax.set_ylim(0.4, 0.9)
            ax.margins(y=0.02)
        ax.set_xlabel("Number of Training Videos", fontsize=22)
        if idx == 0:
            ax.set_ylabel(_metric_display_name(args.metric, args.stat), fontsize=22)
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.text(
            0.04,
            0.08,
            BENCHMARK_DISPLAY_NAMES.get(benchmarks[idx], benchmarks[idx]),
            transform=ax.transAxes,
            fontsize=DATASET_BOX_FONTSIZE,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.18",
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.98,
            ),
            zorder=5,
        )

    fig.legend(
        handles=_legend_handles(variants),
        frameon=False,
        fontsize=23,
        loc="upper center",
        ncol=len(variants),
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    stem = f"{args.model}_{args.group}_{args.metric.lower()}_{args.stat.lower()}_motion_masks_data_starvation_panel"
    if tuple(benchmarks) != DEFAULT_BENCHMARKS:
        stem += "_" + "_".join(benchmarks)
    plot_path = out_dir / stem
    saved_paths = save_vector_figure(fig, plot_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved panel plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
