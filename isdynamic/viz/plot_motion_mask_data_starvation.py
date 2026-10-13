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


BENCHMARK_DISPLAY_NAMES = {
    "davis2016": "DAVIS2016",
    "davis2017": "DAVIS2017",
    "davis_all": "DAVIS-all",
    "segtrackv2": "SegTrackv2",
    "fbms59": "FBMS-59",
}

GROUP_DISPLAY_NAMES = {
    "inter_frame_block": "Inter-Frame Block",
    "dec_view1": "Decoder View-1",
    "dec_view2": "Decoder View-2",
    "head_in_view1": "Head-in-View-1",
    "head_in_view2": "Head-in-View-2",
}

MODEL_DISPLAY_NAMES = {
    "pi3": r"$\pi^3$",
    "pi3x": r"$\pi^3$X",
}

VARIANT_STYLES = {
    "bilinear_upsample": {
        "label": "bilinear upsample",
        "color": "#c65a3a",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3.8,
    },
    "sam2_prompt_otsu_off0.2_erode2_w30_mean_otsu_off0.3": {
        "label": "SAM2 refinement",
        "color": "#d62728",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 3.8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot motion-mask data-starvation curves from evaluated J/F JSON summaries."
    )
    parser.add_argument(
        "--data_starvation_eval_root",
        default="outputs/isdynamic_motion_masks_eval_data_starvation_k8",
        help="Root containing random_<vids>_seed_<seed> evaluation directories.",
    )
    parser.add_argument(
        "--full_eval_root",
        default="outputs/isdynamic_motion_masks_eval",
        help="Root containing the full-data (e.g. 160 videos) evaluation directories.",
    )
    parser.add_argument("--model", default="pi3x")
    parser.add_argument("--group", default="inter_frame_block")
    parser.add_argument("--benchmark", default="")
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
        default="outputs/isdynamic_visualizations/pi3x/data_starvation",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _parse_csv_values(text: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def _collect_series(
    *,
    data_starvation_eval_root: Path,
    full_eval_root: Path,
    model: str,
    group: str,
    benchmark: str,
    metric: str,
    stat: str,
    sizes: list[int],
    seeds: list[int],
    variants: list[str],
) -> tuple[list[dict], dict[str, dict[int, list[float]]], list[int]]:
    rows: list[dict] = []
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for size in sizes:
        if size == 160:
            base = full_eval_root / model / group
            for variant in variants:
                summary_path = base / variant / f"{benchmark}.json"
                if not summary_path.exists():
                    continue
                summary = _load_json(summary_path)
                score = summary.get("metrics", {}).get(metric, {}).get(stat)
                if score is None:
                    continue
                score = float(score)
                series[variant][size].append(score)
                rows.append(
                    {
                        "model": model,
                        "group": group,
                        "benchmark": benchmark,
                        "metric": metric,
                        "stat": stat,
                        "variant": variant,
                        "size": size,
                        "seed": "full",
                        "score": score,
                        "summary_path": str(summary_path),
                    }
                )
            continue

        for seed in seeds:
            base = data_starvation_eval_root / f"random_{size}_seed_{seed}" / model / group
            for variant in variants:
                summary_path = base / variant / f"{benchmark}.json"
                if not summary_path.exists():
                    continue
                summary = _load_json(summary_path)
                score = summary.get("metrics", {}).get(metric, {}).get(stat)
                if score is None:
                    continue
                score = float(score)
                series[variant][size].append(score)
                rows.append(
                    {
                        "model": model,
                        "group": group,
                        "benchmark": benchmark,
                        "metric": metric,
                        "stat": stat,
                        "variant": variant,
                        "size": size,
                        "seed": seed,
                        "score": score,
                        "summary_path": str(summary_path),
                    }
                )

    ordered_sizes = [size for size in sizes if any(size in series[variant] for variant in variants)]
    return rows, series, ordered_sizes


def _series_points(
    series: dict[str, dict[int, list[float]]],
    variant: str,
    ordered_sizes: list[int],
    x_positions: dict[int, int],
) -> tuple[list[int], list[float], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    yerr: list[float] = []
    for size in ordered_sizes:
        values = series.get(variant, {}).get(size)
        if not values:
            continue
        xs.append(x_positions[size])
        ys.append(_mean(values))
        yerr.append(_std(values))
    return xs, ys, yerr


def _legend_handles(variants: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for variant in variants:
        style = dict(VARIANT_STYLES.get(variant, {}))
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "C0"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.2),
                markersize=10,
                label=style.get("label", variant),
            )
        )
    return handles


def _emphasize_model_display_name(label: str) -> str:
    if not label:
        return label
    if label == r"$\pi^3$X":
        return r"$\mathbf{\pi^3X}$"
    if label == r"$\pi^3$":
        return r"$\mathbf{\pi^3}$"
def _metric_display_name(metric: str, stat: str) -> str:
    if stat == "M":
        return f"{metric}-Mean"
    if stat == "R":
        return f"{metric}-Recall"
    if stat == "D":
        return f"{metric}-Decay"
    return f"{metric} {stat}"


def _write_rows_csv(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "group",
                "benchmark",
                "metric",
                "stat",
                "variant",
                "size",
                "seed",
                "score",
                "summary_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_starvation_eval_root = Path(args.data_starvation_eval_root).resolve()
    full_eval_root = Path(args.full_eval_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_csv_values(args.sizes_csv, int)
    seeds = _parse_csv_values(args.seeds_csv, int)
    variants = _parse_csv_values(args.variants_csv, str)
    benchmarks = [args.benchmark] if args.benchmark.strip() else _parse_csv_values(args.benchmarks_csv, str)

    for benchmark in benchmarks:
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
            print(f"[skip] no matching results for benchmark={benchmark}")
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        x_positions = {size: idx for idx, size in enumerate(ordered_sizes)}
        for variant in variants:
            xs, ys, yerr = _series_points(series, variant, ordered_sizes, x_positions)
            style = dict(VARIANT_STYLES.get(variant, {}))
            if not xs:
                continue
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                capsize=4,
                color=style.get("color", "C0"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.2),
                markersize=10,
            )

        metric_display = _metric_display_name(args.metric, args.stat)
        model_display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
        group_display = GROUP_DISPLAY_NAMES.get(args.group, args.group)
        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)

        ax.set_xlabel("Number of Training Videos", fontsize=13)
        ax.set_ylabel(metric_display, fontsize=13)
        ax.set_title(f"{model_display} | {group_display} | {benchmark_display}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(range(len(ordered_sizes))), [str(size) for size in ordered_sizes])
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(handles=_legend_handles(variants), frameon=False, fontsize=12, loc="upper right")
        fig.tight_layout()

        stem = (
            f"{args.model}_{args.group}_{_slugify(benchmark)}_"
            f"{_slugify(args.metric)}_{_slugify(args.stat)}_motion_masks_data_starvation"
        )
        csv_path = out_dir / f"{stem}.csv"
        _write_rows_csv(csv_path, rows)

        plot_path = out_dir / f"{stem}.pdf"
        saved_paths = save_vector_figure(fig, plot_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

        print(f"Saved CSV: {csv_path}")
        print("Saved plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
