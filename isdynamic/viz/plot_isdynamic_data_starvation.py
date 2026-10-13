from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .figure_io import save_vector_figure


MODEL_LAYOUT = {
    "vggt": ("vggt", "isdynamic_blockwise_youtubevos", ""),
    "dust3r": ("dust3r", "isdynamic_blockwise_youtubevos", ""),
    "da3": ("depth_anything_3", "isdynamic_blockwise_youtubevos", ""),
    "mapanything": ("map_anything", "isdynamic_blockwise_youtubevos", ""),
    "pi3x": ("pi3", "isdynamic_blockwise_youtubevos_pi3x", "_pi3x"),
    "pi3": ("pi3", "isdynamic_blockwise_youtubevos_pi3", "_pi3"),
}

VIS_OUTPUT_DIRS = {
    "vggt": "vggt",
    "dust3r": "dust3r",
    "da3": "depth_anything_3",
    "mapanything": "map_anything",
    "pi3x": "pi3x",
    "pi3": "pi3",
}

DISPLAY_NAMES = {
    "pair_lowrank_quadratic_k8": "low-rank quadratic (k=8)",
    "pair_quadratic": "full-rank quadratic",
}

MODEL_DISPLAY_NAMES = {
    "da3": "DA3",
    "dust3r": "DUSt3R",
    "pi3x": r"$\pi^3$X",
}

TEST_SPLIT_DISPLAY_NAMES = {
    "test_davis": "Test: DAVIS-all",
    "test_segtrackv2": "Test: SegTrackv2",
    "test_fbms59": "Test: FBMS-59",
}

GROUP_DISPLAY_NAMES = {
    "inter_frame_block": "inter-frame block",
    "dec_view1": "decoder view-1",
    "dec_view2": "decoder view-2",
    "head_in_view1": "head-in-view-1",
    "head_in_view2": "head-in-view-2",
}

SERIES_STYLES = {
    "pair_lowrank_quadratic_k8": {"color": "#d62728", "marker": "X", "linestyle": "-", "linewidth": 3.8},
    "pair_quadratic": {"color": "#ff00ff", "marker": "P", "linestyle": "--", "linewidth": 3.7},
}

SERIES_ORDER = (
    ("pair_lowrank_quadratic_k8", "pair_lowrank_quadratic", 8),
    ("pair_quadratic", "pair_quadratic", None),
)


def _metric_display_name(metric: str) -> str:
    if metric in {"ng", "norm_gain", "normalized_gain"}:
        return "Normalized Gain"
    if metric == "accuracy":
        return r"$\mathrm{Acc}_{\mathrm{dyn}}$"
    return metric.title()


def _emphasize_model_display_name(label: str) -> str:
    if not label:
        return label
    if label == r"$\pi^3$X":
        return r"$\mathbf{\pi^3X}$"
    if label == r"$\pi^3$":
        return r"$\mathbf{\pi^3}$"
    return rf"$\mathbf{{{label}}}$"


def build_parser(require_model: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and plot the section-4.5 data-starvation experiment.")
    parser.add_argument("--model", required=require_model, choices=sorted(MODEL_LAYOUT.keys()))
    parser.add_argument("--outputs_root", default="outputs/model_outputs")
    parser.add_argument("--sizes_csv", default="160,80,40,20,10,5")
    parser.add_argument("--seeds_csv", default="42")
    parser.add_argument("--group", default="inter_frame_block")
    parser.add_argument("--test_split", default="test_davis")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--lowrank_rank", type=int, default=8)
    parser.add_argument("--lowrank_ranks_csv", default="8")
    parser.add_argument("--result_subpath_prefix", default="data_starvation")
    parser.add_argument("--show_all_positive_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_fullrank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show_std_band", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out_dir", default=None)
    return parser


def _discover_available_seeds(
    *,
    model_root: Path,
    suffix: str,
    result_subpath_prefix: str | None,
) -> list[int]:
    if not model_root.exists():
        return []
    prefix_pattern = ""
    if result_subpath_prefix:
        prefix_pattern = rf"(?:{re.escape(result_subpath_prefix)}_)?"
    pattern = re.compile(rf"^{prefix_pattern}random_\d+_seed_(\d+){re.escape(suffix)}$")
    seeds = set()
    for child in model_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            seeds.add(int(match.group(1)))
    return sorted(seeds)


def _resolve_requested_seeds(
    *,
    seeds_csv: str,
    model_root: Path,
    suffix: str,
    result_subpath_prefix: str | None,
) -> list[int]:
    token = str(seeds_csv).strip().lower()
    if token not in {"", "auto", "all", "*"}:
        return [int(x.strip()) for x in seeds_csv.split(",") if x.strip()]
    discovered = _discover_available_seeds(
        model_root=model_root,
        suffix=suffix,
        result_subpath_prefix=result_subpath_prefix,
    )
    return discovered if discovered else [42]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_from_split_metrics(split_metrics: dict, metric: str) -> float | None:
    if metric in split_metrics:
        return float(split_metrics[metric])

    if metric in {"ng", "norm_gain", "normalized_gain"}:
        accuracy = split_metrics.get("accuracy")
        tp = split_metrics.get("tp")
        fp = split_metrics.get("fp")
        fn = split_metrics.get("fn")
        tn = split_metrics.get("tn")
        if None in (accuracy, tp, fp, fn, tn):
            return None
        total = float(tp) + float(fp) + float(fn) + float(tn)
        if total <= 0:
            return None
        positive_rate = (float(tp) + float(fn)) / total
        negative_rate = (float(tn) + float(fp)) / total
        baseline = max(positive_rate, negative_rate)
        denom = 1.0 - baseline
        if denom <= 0:
            return None
        return (float(accuracy) - baseline) / denom

    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _all_positive_accuracy(split_metrics: dict) -> float | None:
    tp = split_metrics.get("tp")
    fp = split_metrics.get("fp")
    fn = split_metrics.get("fn")
    tn = split_metrics.get("tn")
    if None in (tp, fp, fn, tn):
        return None
    total = float(tp) + float(fp) + float(fn) + float(tn)
    if total <= 0:
        return None
    return (float(tp) + float(fn)) / total


def _score_from_summary(
    summary: dict,
    model_type: str,
    group: str | None,
    test_split: str,
    metric: str,
    expected_rank: int | None,
) -> float | None:
    rows = summary.get("results")
    if not isinstance(rows, list) or not rows:
        rows = summary.get("rows", [])

    scores = []
    for row in rows:
        if row.get("model_type") != model_type:
            continue
        if group is not None and row.get("group") != group:
            continue
        if model_type == "pair_lowrank_quadratic" and expected_rank is not None:
            row_rank = row.get("rank")
            if row_rank is not None and int(row_rank) != int(expected_rank):
                continue
        best_tests = row.get("best_tests") or {}
        split_metrics = best_tests.get(test_split)
        if not isinstance(split_metrics, dict):
            continue
        score = _metric_from_split_metrics(split_metrics, metric)
        if score is not None:
            scores.append(score)
    return max(scores) if scores else None


def _all_positive_baseline_from_summary(
    summary: dict,
    group: str | None,
    test_split: str,
) -> float | None:
    rows = summary.get("results")
    if not isinstance(rows, list) or not rows:
        rows = summary.get("rows", [])

    baselines = []
    for row in rows:
        if group is not None and row.get("group") != group:
            continue
        best_tests = row.get("best_tests") or {}
        split_metrics = best_tests.get(test_split)
        if not isinstance(split_metrics, dict):
            continue
        baseline = _all_positive_accuracy(split_metrics)
        if baseline is not None:
            baselines.append(float(baseline))
    return max(baselines) if baselines else None


def _candidate_summary_paths(
    model_root: Path,
    experiment_name: str,
    size: int,
    seed: int,
    suffix: str,
    result_subpath_prefix: str | None,
) -> list[Path]:
    paths = []
    if size == 160 and seed == 42:
        bases = [f"full_160{suffix}", f"random_{size}_seed_{seed}{suffix}"]
    else:
        bases = [f"random_{size}_seed_{seed}{suffix}"]
    if result_subpath_prefix:
        for base in bases:
            paths.append(model_root / f"{result_subpath_prefix}_{base}" / experiment_name / "summary.json")
    for base in bases:
        paths.append(model_root / base / experiment_name / "summary.json")
    if size == 160 and seed == 42:
        paths.append(model_root / experiment_name / "summary.json")
    return paths


def collect_data_starvation_series(
    *,
    model: str,
    outputs_root: str,
    sizes_csv: str,
    seeds_csv: str,
    group: str,
    test_split: str,
    metric: str,
    lowrank_rank: int,
    lowrank_ranks_csv: str,
    result_subpath_prefix: str | None,
) -> tuple[list[dict], dict, dict, list[int]]:
    model_dir_name, experiment_name, suffix = MODEL_LAYOUT[model]
    outputs_root_path = Path(outputs_root).resolve()
    model_root = outputs_root_path / model_dir_name

    sizes = [int(x.strip()) for x in sizes_csv.split(",") if x.strip()]
    seeds = _resolve_requested_seeds(
        seeds_csv=seeds_csv,
        model_root=model_root,
        suffix=suffix,
        result_subpath_prefix=result_subpath_prefix,
    )
    lowrank_ranks = [int(x.strip()) for x in lowrank_ranks_csv.split(",") if x.strip()]
    if not lowrank_ranks:
        lowrank_ranks = [lowrank_rank]
    rows: list[dict] = []
    series = defaultdict(lambda: defaultdict(list))
    all_positive_baselines = defaultdict(list)

    for size in sizes:
        for seed in seeds:
            summary_path = None
            for candidate in _candidate_summary_paths(
                model_root=model_root,
                experiment_name=experiment_name,
                size=size,
                seed=seed,
                suffix=suffix,
                result_subpath_prefix=result_subpath_prefix,
            ):
                if candidate.exists():
                    summary_path = candidate
                    break
            if summary_path is None:
                continue
            summary = _load_json(summary_path)
            all_positive_baseline = _all_positive_baseline_from_summary(
                summary=summary,
                group=group,
                test_split=test_split,
            )
            if all_positive_baseline is not None:
                all_positive_baselines[size].append(all_positive_baseline)
            requested_series = [
                (f"pair_lowrank_quadratic_k{rank}", "pair_lowrank_quadratic", rank)
                for rank in lowrank_ranks
            ] + [("pair_quadratic", "pair_quadratic", None)]
            for series_key, model_type, expected_rank in requested_series:
                score = _score_from_summary(
                    summary=summary,
                    model_type=model_type,
                    group=group,
                    test_split=test_split,
                    metric=metric,
                    expected_rank=expected_rank,
                )
                if score is None:
                    continue
                series[series_key][size].append(score)
                rows.append(
                    {
                        "model": model,
                        "size": size,
                        "seed": seed,
                        "model_type": series_key,
                        "group": group,
                        "test_split": test_split,
                        "metric": metric,
                        "score": score,
                        "summary_path": str(summary_path),
                    }
                )

    ordered_series_keys = [series_key for series_key, _, _ in SERIES_ORDER if series_key in series]
    ordered_sizes = [size for size in sizes if any(size in series[series_key] for series_key in ordered_series_keys)]
    return rows, series, all_positive_baselines, ordered_sizes


def build_data_starvation_legend_handles(
    *,
    include_baseline: bool = True,
    include_fullrank: bool = True,
) -> list[Line2D]:
    handles = []
    for series_key, _, _ in SERIES_ORDER:
        if not include_fullrank and series_key == "pair_quadratic":
            continue
        style = dict(SERIES_STYLES.get(series_key, {}))
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "C0"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.0),
                markersize=10,
                label=DISPLAY_NAMES.get(series_key, series_key),
            )
        )
    if include_baseline:
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="all-positive baseline",
            )
        )
    return handles


def plot_data_starvation_axis(
    ax,
    *,
    model: str,
    group: str,
    test_split: str,
    metric: str,
    series: dict,
    all_positive_baselines: dict,
    ordered_sizes: list[int],
    show_baseline: bool = True,
    include_fullrank: bool = True,
    show_std_band: bool = False,
    title_mode: str = "full",
) -> list[Line2D]:
    plotted_any = False
    legend_handles: list[Line2D] = []
    x_positions = {size: idx for idx, size in enumerate(ordered_sizes)}
    for series_key, _, _ in SERIES_ORDER:
        if not include_fullrank and series_key == "pair_quadratic":
            continue
        xs = []
        ys = []
        plot_sizes = []
        style = dict(SERIES_STYLES.get(series_key, {}))
        ystd = []
        for size in ordered_sizes:
            if size not in series[series_key]:
                continue
            values = series[series_key][size]
            xs.append(x_positions[size])
            ys.append(_mean(values))
            ystd.append(_std(values))
            plot_sizes.append(size)
        if not xs:
            continue
        plotted_any = True
        line_zorder = 3 if series_key == "pair_quadratic" else 2
        if show_std_band:
            band_lower = []
            band_upper = []
            band_xs = []
            for x, y, std, size in zip(xs, ys, ystd, plot_sizes):
                if len(series[series_key][size]) <= 1:
                    continue
                band_xs.append(x)
                band_lower.append(y - std)
                band_upper.append(y + std)
            if band_xs:
                ax.fill_between(
                    band_xs,
                    band_lower,
                    band_upper,
                    color=style.get("color", "C0"),
                    alpha=0.18,
                    linewidth=0,
                    zorder=max(1, line_zorder - 1),
                )
        ax.plot(xs, ys, markersize=10, zorder=line_zorder, **style)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "C0"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.0),
                markersize=10,
                markerfacecolor=style.get("markerfacecolor", style.get("color", "C0")),
                markeredgewidth=style.get("markeredgewidth", 1.0),
                label=DISPLAY_NAMES.get(series_key, series_key),
            )
        )

    if not plotted_any:
        raise SystemExit("No matching data-starvation results found.")

    if show_baseline and metric == "accuracy":
        baseline_points = [_mean(all_positive_baselines[size]) for size in ordered_sizes if size in all_positive_baselines]
        if baseline_points:
            ax.axhline(
                y=_mean(baseline_points),
                linestyle="--",
                linewidth=1.5,
                color="gray",
                alpha=0.9,
                label="all-positive baseline",
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    label="all-positive baseline",
                )
            )

    metric_display_name = _metric_display_name(metric)
    ax.set_xlabel("Number of Training Videos", fontsize=13)
    ax.set_ylabel(metric_display_name, fontsize=13)
    model_display_name = _emphasize_model_display_name(MODEL_DISPLAY_NAMES.get(model, model))
    group_display_name = GROUP_DISPLAY_NAMES.get(group, group)
    test_split_display_name = TEST_SPLIT_DISPLAY_NAMES.get(test_split, test_split)
    if title_mode == "panel":
        ax.set_title(f"{model_display_name} | {group_display_name}", fontsize=17)
    else:
        ax.set_title(f"{model_display_name} | {group_display_name} | {test_split_display_name}", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(len(ordered_sizes))), [str(size) for size in ordered_sizes])
    if title_mode == "panel":
        ax.set_xlabel("Number of Training Videos", fontsize=15)
        ax.set_ylabel(metric_display_name, fontsize=15)
        ax.tick_params(axis="both", labelsize=14)
    else:
        ax.tick_params(axis="both", labelsize=12)
    return legend_handles


def parse_args() -> argparse.Namespace:
    return build_parser(require_model=True).parse_args()


def main() -> None:
    args = parse_args()
    output_dir_name = VIS_OUTPUT_DIRS.get(args.model, args.model)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path("outputs/isdynamic_visualizations") / output_dir_name / "data_starvation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, series, all_positive_baselines, ordered_sizes = collect_data_starvation_series(
        model=args.model,
        outputs_root=args.outputs_root,
        sizes_csv=args.sizes_csv,
        seeds_csv=args.seeds_csv,
        group=args.group,
        test_split=args.test_split,
        metric=args.metric,
        lowrank_rank=args.lowrank_rank,
        lowrank_ranks_csv=args.lowrank_ranks_csv,
        result_subpath_prefix=args.result_subpath_prefix,
    )

    csv_path = out_dir / f"{args.model}_{args.group}_{args.test_split}_{args.metric}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "size", "seed", "model_type", "group", "test_split", "metric", "score", "summary_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    legend_handles = plot_data_starvation_axis(
        ax,
        model=args.model,
        group=args.group,
        test_split=args.test_split,
        metric=args.metric,
        series=series,
        all_positive_baselines=all_positive_baselines,
        ordered_sizes=ordered_sizes,
        show_baseline=args.show_all_positive_baseline,
        include_fullrank=args.include_fullrank,
        show_std_band=args.show_std_band,
        title_mode="full",
    )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
    )
    fig.tight_layout()
    plot_path = out_dir / f"{args.model}_{args.group}_{args.test_split}_{args.metric}.pdf"
    saved_paths = save_vector_figure(fig, plot_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved CSV: {csv_path}")
    print("Saved plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
