from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from .figure_io import save_vector_figure

MODEL_COLOR_MAP = {
    "pair_linear": "#4daf4a",
    "pair_diag_quadratic": "#ff7f0e",
    "pair_quadratic": "#17becf",
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
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

CANONICAL_MODEL_TYPE_ORDER = [
    "pair_linear",
    "pair_diag_quadratic",
    "pair_quadratic",
    "pair_lowrank_quadratic",
    "pair_lowrank_quadratic_strict",
    "pair_attention",
    "pair_value",
    "pair_attention_biased_quadratic",
    "pair_value_biased_quadratic",
    "pair_attention_biased_lowrank_quadratic",
    "pair_value_biased_lowrank_quadratic",
    "pair_attention_biased_lowrank_quadratic_strict",
    "pair_value_biased_lowrank_quadratic_strict",
]

MODEL_TYPE_DISPLAY_NAMES = {
    "pair_linear": "linear",
    "pair_diag_quadratic": "diagonal quad",
    "pair_quadratic": "full-rank quad",
    "pair_lowrank_quadratic": "low-rank quad",
    "pair_lowrank_quadratic_strict": "strict low-rank quad",
    "pair_attention": "attention",
    "pair_value": "value",
    "pair_attention_biased_quadratic": "attention-initialized full-rank quad",
    "pair_value_biased_quadratic": "value-initialized full-rank quad",
    "pair_attention_biased_lowrank_quadratic": "attention-initialized low-rank quad",
    "pair_value_biased_lowrank_quadratic": "value-initialized low-rank quad",
    "pair_attention_biased_lowrank_quadratic_strict": "attention-initialized strict low-rank quad",
    "pair_value_biased_lowrank_quadratic_strict": "value-initialized strict low-rank quad",
}

MODEL_DISPLAY_NAMES = {
    "pi3": r"$\mathbf{\pi^3}$",
    "pi3x": r"$\mathbf{\pi^3X}$",
    "vggt": r"$\mathbf{VGGT}$",
    "depth_anything_3": r"$\mathbf{DA3}$",
    "da3": r"$\mathbf{DA3}$",
    "map_anything": r"$\mathbf{MapA}$",
    "dust3r": r"$\mathbf{DUSt3R}$",
}

BLOCKWISE_LEGEND_FONTSIZE = 22
BLOCKWISE_TITLE_FONTSIZE = 18
BLOCKWISE_LABEL_FONTSIZE = 23
BLOCKWISE_YLABEL_FONTSIZE = 25
BLOCKWISE_TICK_FONTSIZE = 17
BLOCKWISE_ANNOT_FONTSIZE = 14
BASELINE_LEGEND_LABEL = "baseline"
ACCURACY_YLABEL = r"$\mathrm{Acc}_{\mathrm{dyn}}$"


def _is_ng_metric(metric: str) -> bool:
    return metric in {"ng", "norm_gain", "normalized_gain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot blockwise IsDynamic probe sweep results from one or more summary.json files.")
    parser.add_argument("--model", type=str, default="", help="Optional model label; accepted for wrapper compatibility.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path or comma-separated list of summary.json files to merge before plotting.")
    parser.add_argument("--summary_labels", type=str, default="", help="Optional comma-separated per-summary labels. Empty entries keep the original series names.")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1", "precision", "recall", "loss"], help="Metric from each result.best to plot on y-axis.")
    parser.add_argument("--metric_split", type=str, default="val", help="Split to plot: 'val' uses result.best; anything else uses result.best_tests[metric_split].")
    parser.add_argument("--groups", type=str, required=True, help="Comma-separated group order for subplots.")
    parser.add_argument("--model_types", type=str, default="", help="Optional comma-separated model order; default uses summary model_types.")
    parser.add_argument("--out_png", type=str, required=True)
    parser.add_argument("--baseline_mode", type=str, default="always_true", choices=["always_true", "constant", "none"])
    parser.add_argument("--baseline_value", type=float, default=0.5)
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_allow_empty(raw: str) -> List[str]:
    if raw == "":
        return []
    return [x.strip() for x in raw.split(",")]


def _split_series_key(model_type: str) -> tuple[str, str]:
    if "@@" not in model_type:
        return model_type, ""
    base_model_type, series_label = model_type.split("@@", 1)
    return base_model_type, series_label


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_existing_summary(summary_paths: List[Path]) -> Dict:
    for path in summary_paths:
        if path.exists():
            return _load_json(path)
    return {}


def _merge_summaries(summary_paths: List[Path], summary_labels: List[str]) -> Dict:
    model_types: List[str] = []
    results: List[Dict] = []
    seen = set()

    labels = list(summary_labels)
    if len(labels) < len(summary_paths):
        labels.extend([""] * (len(summary_paths) - len(labels)))
    elif len(labels) > len(summary_paths):
        labels = labels[: len(summary_paths)]

    for path, summary_label in zip(summary_paths, labels):
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")
        summary = _load_json(path)

        for mt in summary.get("model_types", []):
            series_key = f"{mt}@@{summary_label}" if summary_label else mt
            if series_key not in model_types:
                model_types.append(series_key)

        summary_results = summary.get("results", [])
        if not isinstance(summary_results, list) or not summary_results:
            summary_results = []
            for item in summary.get("rows", []):
                feature_key = item.get("feature_key")
                block_idx = -1
                if isinstance(feature_key, str):
                    try:
                        block_idx = int(feature_key.rsplit("_", 1)[1])
                    except (IndexError, ValueError):
                        block_idx = -1

                metric_name = str(item.get("metric") or summary.get("metric") or "accuracy")
                metrics_path_raw = item.get("metrics_path", "")
                metrics_blob: Dict = {}
                if isinstance(metrics_path_raw, str) and metrics_path_raw:
                    try:
                        metrics_blob = _load_json(Path(metrics_path_raw))
                    except Exception:
                        metrics_blob = {}
                best: Dict[str, float] = {}
                if isinstance(metrics_blob.get("best"), dict):
                    best = dict(metrics_blob["best"])
                elif item.get("best_val") is not None:
                    try:
                        best[metric_name] = float(item["best_val"])
                    except (TypeError, ValueError):
                        pass

                summary_results.append(
                    {
                        "group": item.get("group"),
                        "block_idx": block_idx,
                        "feature_key": feature_key,
                        "model_type": item.get("model_type"),
                        "best": best,
                        "best_tests": metrics_blob.get("best_tests", item.get("best_tests", {})),
                        "metrics_path": metrics_path_raw,
                    }
                )

        for item in summary_results:
            item = dict(item)
            item_model_type = str(item.get("model_type"))
            if summary_label:
                item_model_type = f"{item_model_type}@@{summary_label}"
            item["model_type"] = item_model_type
            key = (item.get("group"), int(item.get("block_idx", -1)), item.get("feature_key"), item.get("model_type"))
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    return {"model_types": model_types, "results": results}


def _split_metrics(item: Dict, metric_split: str) -> Dict:
    if metric_split in {"val", "best"}:
        best = item.get("best", {})
        return best if isinstance(best, dict) else {}

    best_tests = item.get("best_tests", {})
    if not isinstance(best_tests, dict):
        return {}
    if metric_split in best_tests and isinstance(best_tests[metric_split], dict):
        return best_tests[metric_split]
    for split_name, split_metrics in best_tests.items():
        if split_name == metric_split or split_name.startswith(f"{metric_split}/"):
            return split_metrics if isinstance(split_metrics, dict) else {}
    return {}


def _always_true_baseline(group_items: List[Dict], metric_split: str) -> float:
    ratios: List[float] = []
    for item in group_items:
        best = _split_metrics(item, metric_split)
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
    return _legend_label_with_context(model_type, [])


def _has_labeled_variant(base_model_type: str, all_model_types: List[str]) -> bool:
    for candidate in all_model_types:
        candidate_base, candidate_label = _split_series_key(candidate)
        if candidate_base == base_model_type and candidate_label:
            return True
    return False


def _effective_series_label(model_type: str, all_model_types: List[str]) -> str:
    _base_model_type, series_label = _split_series_key(model_type)
    if series_label:
        return series_label
    return ""


def _is_pi3_overlay_label(series_label: str) -> bool:
    return "pi^3" in series_label and "x" not in series_label.lower()


def _legend_label_with_context(model_type: str, all_model_types: List[str]) -> str:
    base_model_type, _ = _split_series_key(model_type)
    series_label = _effective_series_label(model_type, all_model_types)
    label = MODEL_TYPE_DISPLAY_NAMES.get(
        base_model_type,
        base_model_type[len("pair_") :] if base_model_type.startswith("pair_") else base_model_type,
    )
    has_pi3_overlay = any(_is_pi3_overlay_label(_effective_series_label(candidate, all_model_types)) for candidate in all_model_types)
    if _is_pi3_overlay_label(series_label):
        return f"{label} ($\\pi^3$)"
    if has_pi3_overlay and not series_label and base_model_type == "pair_lowrank_quadratic":
        return f"{label} ($\\pi^3X$)"
    if series_label:
        return f"{label} ({series_label})"
    return label


def _variant_color(base_color: str, base_model_type: str, series_label: str) -> str:
    if not series_label or series_label == "k=256":
        return base_color
    if series_label == "k=8" and base_model_type == "pair_lowrank_quadratic":
        return "#6f42c1"
    if "pi^3x" in series_label.lower():
        return base_color
    if series_label.startswith("Pi3"):
        return "#2ca02c"
    if "pi^3" in series_label.lower():
        return "#2ca02c"

    rgb = tuple(int(base_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    hue, sat, val = colorsys.rgb_to_hsv(*rgb)
    shifted_hue = (hue + 0.28) % 1.0
    shifted_sat = min(1.0, sat * 0.9 + 0.1)
    shifted_val = min(1.0, val * 0.92 + 0.02)
    out_rgb = colorsys.hsv_to_rgb(shifted_hue, shifted_sat, shifted_val)
    return "#" + "".join(f"{int(round(channel * 255)):02x}" for channel in out_rgb)


def _line_color(model_type: str, all_model_types: List[str]) -> str:
    base_model_type, _ = _split_series_key(model_type)
    if base_model_type in MODEL_COLOR_MAP:
        base_color = MODEL_COLOR_MAP[base_model_type]
    else:
        digest = hashlib.sha1(base_model_type.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(FALLBACK_COLORS)
        base_color = FALLBACK_COLORS[idx]
    return _variant_color(base_color, base_model_type, _effective_series_label(model_type, all_model_types))


def _line_style(model_type: str, all_model_types: List[str]) -> str:
    base_model_type, _ = _split_series_key(model_type)
    series_label = _effective_series_label(model_type, all_model_types)
    if series_label.startswith("Pi3"):
        return "-."
    if series_label == "k=8":
        return {
            "pair_linear": ":",
            "pair_diag_quadratic": "--",
            "pair_quadratic": "-.",
            "pair_lowrank_quadratic": (0, (5, 1, 1, 1)),
            "pair_attention": (0, (1, 1)),
            "pair_value": (0, (3, 1, 1, 1)),
            "pair_attention_biased_lowrank_quadratic": (0, (5, 1)),
            "pair_value_biased_lowrank_quadratic": (0, (3, 1)),
        }.get(base_model_type, "--")
    return {
        "pair_linear": "-",
        "pair_diag_quadratic": "--",
        "pair_quadratic": "-.",
        "pair_lowrank_quadratic": "-",
        "pair_attention": (0, (5, 1)),
        "pair_value": (0, (3, 1, 1, 1)),
        "pair_attention_biased_lowrank_quadratic": (0, (5, 1, 1, 1)),
        "pair_value_biased_lowrank_quadratic": (0, (3, 1, 1, 1)),
    }.get(base_model_type, "-")


def _line_marker(model_type: str, all_model_types: List[str]) -> str:
    base_model_type, _ = _split_series_key(model_type)
    series_label = _effective_series_label(model_type, all_model_types)
    if series_label.startswith("Pi3"):
        return "D"
    if series_label == "k=8":
        return {
            "pair_linear": "^",
            "pair_diag_quadratic": "s",
            "pair_quadratic": "P",
            "pair_lowrank_quadratic": "X",
            "pair_attention": "v",
            "pair_value": "<",
            "pair_attention_biased_lowrank_quadratic": ">",
            "pair_value_biased_lowrank_quadratic": "h",
        }.get(base_model_type, "s")
    return {
        "pair_linear": "o",
        "pair_diag_quadratic": "s",
        "pair_quadratic": "P",
        "pair_lowrank_quadratic": "D",
        "pair_attention": "^",
        "pair_value": "v",
        "pair_attention_biased_lowrank_quadratic": ">",
        "pair_value_biased_lowrank_quadratic": "<",
    }.get(base_model_type, "o")


def _line_width(model_type: str, all_model_types: List[str]) -> float:
    series_label = _effective_series_label(model_type, all_model_types)
    if series_label.startswith("Pi3"):
        return 2.9
    return 2.7 if series_label == "k=8" else 2.5


def _line_markersize(model_type: str, all_model_types: List[str]) -> float:
    series_label = _effective_series_label(model_type, all_model_types)
    if series_label.startswith("Pi3"):
        return 6.9
    return 6.3 if series_label == "k=8" else 5.9


def _line_zorder(model_type: str, all_model_types: List[str]) -> float:
    series_label = _effective_series_label(model_type, all_model_types)
    if series_label.startswith("Pi3"):
        return 4.0
    return 3.0 if series_label == "k=8" else 2.0


def _line_alpha(model_type: str, all_model_types: List[str]) -> float:
    series_label = _effective_series_label(model_type, all_model_types)
    if "pi^3" in series_label or "Pi3" in series_label:
        return 0.65
    return 0.95


def _group_title(group: str, model_name: str = "") -> str:
    if group in {"depth_head", "dual_dpt_head"}:
        return "depth head"
    mapping = {
        "inter_frame_block": "inter-frame block",
        "intra_frame_block": "intra-frame block",
        "cam_head": "camera head",
        "conf_head": "confidence head",
        "point_head": "point head",
        "dec_view1": "decoder view-1",
        "dec_view2": "decoder view-2",
        "head_in_view1": "head-in-view-1",
        "head_in_view2": "head-in-view-2",
    }
    return mapping.get(group, group.replace("_", " "))


def _model_title(model_name: str) -> str:
    model_name = str(model_name or "").strip().lower()
    if not model_name:
        return ""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def _figure_model_title(model_name: str, model_types: List[str]) -> str:
    model_title = _model_title(model_name)
    if str(model_name or "").strip().lower() == "pi3x":
        has_pi3_overlay = any(
            _is_pi3_overlay_label(_effective_series_label(candidate, model_types))
            for candidate in model_types
        )
        if has_pi3_overlay:
            return r"$\mathbf{\pi^3X}$ & $\mathbf{\pi^3}$"
    return model_title


def _ordered_model_types(model_types: List[str]) -> List[str]:
    rank = {name: idx for idx, name in enumerate(CANONICAL_MODEL_TYPE_ORDER)}

    def _series_rank(name: str) -> tuple[int, str]:
        series_label = _effective_series_label(name, model_types)
        if series_label == "":
            return (0, "")
        if series_label == "k=256":
            return (1, series_label)
        if _is_pi3_overlay_label(series_label):
            return (2, series_label)
        if series_label == "k=8":
            return (3, series_label)
        return (4, series_label)

    return sorted(
        model_types,
        key=lambda name: (
            rank.get(_split_series_key(name)[0], 10**6),
            _series_rank(name)[0],
            _series_rank(name)[1],
            name,
        ),
    )


def _ordered_legend_labels(model_types: List[str], baseline_mode: str, *, include_baseline: bool = True) -> List[str]:
    labels = [_legend_label_with_context(model_type, model_types) for model_type in model_types]
    if baseline_mode != "none" and include_baseline:
        labels.append(BASELINE_LEGEND_LABEL)
    return labels


def _da3_intraframe_separator(summary_meta: Dict) -> float | None:
    sharded_root = summary_meta.get("sharded_root")
    if not isinstance(sharded_root, str) or not sharded_root:
        return None

    manifest_path = Path(sharded_root) / "train" / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return None

    feature_groups = ((manifest.get("meta") or {}).get("feature_groups") or {})
    inter_actual = feature_groups.get("inter_frame_block_actual_blocks")
    intra_actual = feature_groups.get("intra_frame_block_actual_blocks")
    if not isinstance(inter_actual, list):
        inter_actual = feature_groups.get("glob_attn_actual_blocks")
    if not isinstance(intra_actual, list):
        intra_actual = feature_groups.get("perframe_attn_actual_blocks")
    if not isinstance(inter_actual, list) or not isinstance(intra_actual, list) or not inter_actual or not intra_actual:
        return None

    try:
        first_inter_actual = min(int(value) for value in inter_actual)
        leading_intra_count = sum(1 for value in intra_actual if int(value) < first_inter_actual)
    except (TypeError, ValueError):
        return None

    if leading_intra_count <= 0 or leading_intra_count >= len(intra_actual):
        return None
    return float(leading_intra_count) - 0.5


def _metric_value(item: Dict, metric: str, metric_split: str) -> float:
    source = _split_metrics(item, metric_split)
    if metric in source:
        return float(source[metric])
    best_val = item.get("best_val")
    if best_val is not None:
        return float(best_val)
    return float("nan")


def main() -> None:
    args = parse_args()
    summary_paths = [Path(p) for p in _parse_csv(args.summary_json)]
    summary_labels = _parse_csv_allow_empty(args.summary_labels)
    summary_meta = _first_existing_summary(summary_paths)
    summary = _merge_summaries(summary_paths, summary_labels)
    groups = _parse_csv(args.groups)
    model_types = _parse_csv(args.model_types) or _ordered_model_types(list(summary.get("model_types", [])))
    results = summary.get("results", [])
    da3_separator_x = _da3_intraframe_separator(summary_meta) if args.model in {"depth_anything_3", "da3"} else None

    if not groups:
        available_groups = sorted({str(item.get("group")) for item in results if item.get("group") is not None})
        raise SystemExit(
            "No groups were requested for plotting. "
            f"Available groups={available_groups}. "
            "Check your PROBE_GROUPS / ISDYNAMIC_GROUPS filters."
        )

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.2 * n_groups, 4.2), squeeze=False)
    axes = axes[0]
    plotted_any = False
    legend_handle_by_label = {}
    all_y_values = []

    for ax, group in zip(axes, groups):
        group_items = [r for r in results if r.get("group") == group]
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
                markersize=_line_markersize(model_type, model_types),
                linestyle=_line_style(model_type, model_types),
                label=_legend_label_with_context(model_type, model_types),
                color=_line_color(model_type, model_types),
                alpha=_line_alpha(model_type, model_types),
                zorder=_line_zorder(model_type, model_types),
            )[0]
            legend_handle_by_label.setdefault(_legend_label_with_context(model_type, model_types), line)
            all_y_values.extend([y for y in ys if y == y])
            plotted_any = True

        baseline_value = None
        if args.baseline_mode == "constant":
            baseline_value = float(args.baseline_value)
        elif args.baseline_mode == "always_true":
            baseline_value = _always_true_baseline(group_items, args.metric_split)
        if baseline_value is not None and baseline_value == baseline_value:
            baseline = ax.axhline(
                baseline_value,
                color="gray",
                linestyle="--",
                linewidth=3.2,
                label=BASELINE_LEGEND_LABEL if group == groups[0] else None,
            )
            legend_handle_by_label.setdefault(BASELINE_LEGEND_LABEL, baseline)
            all_y_values.append(float(baseline_value))

        model_title = _figure_model_title(args.model, model_types)
        title = f"{model_title} | {_group_title(group, args.model)}" if model_title else _group_title(group, args.model)
        ax.set_title(title, fontsize=BLOCKWISE_TITLE_FONTSIZE)
        ax.set_xlabel("Block Index", fontsize=BLOCKWISE_LABEL_FONTSIZE)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.tick_params(axis="both", labelsize=BLOCKWISE_TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle="--", alpha=0.25)

    if not plotted_any:
        available_groups = sorted({str(item.get("group")) for item in results if item.get("group") is not None})
        available_model_types = sorted({str(item.get("model_type")) for item in results if item.get("model_type") is not None})
        raise SystemExit(
            "No matching blockwise series found for the requested filters. "
            f"Requested groups={groups}, model_types={model_types}. "
            f"Available groups={available_groups}, model_types={available_model_types}."
        )

    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        if y_min == y_max:
            pad = max(1e-3, abs(y_min) * 0.05)
        else:
            pad = max(1e-3, (y_max - y_min) * 0.06)
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)

    ylabel = ACCURACY_YLABEL if args.metric == "accuracy" else f"IsDynamic {args.metric.title()}"
    axes[0].set_ylabel(ylabel, fontsize=BLOCKWISE_YLABEL_FONTSIZE)
    show_baseline_in_legend = not _is_ng_metric(args.metric)
    ordered_labels = [
        label
        for label in _ordered_legend_labels(
            model_types,
            args.baseline_mode,
            include_baseline=show_baseline_in_legend,
        )
        if label in legend_handle_by_label
    ]
    if ordered_labels:
        handles = [legend_handle_by_label[label] for label in ordered_labels]
        fig.legend(
            handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=max(1, len(ordered_labels)),
            frameon=False,
            fontsize=BLOCKWISE_LEGEND_FONTSIZE,
        )
        baseline_handle = legend_handle_by_label.get(BASELINE_LEGEND_LABEL)
        if baseline_handle is not None and hasattr(baseline_handle, "set_linewidth"):
            baseline_handle.set_linewidth(max(float(baseline_handle.get_linewidth()), 2.8))
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    out_path = Path(args.out_png)
    saved_paths = save_vector_figure(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
