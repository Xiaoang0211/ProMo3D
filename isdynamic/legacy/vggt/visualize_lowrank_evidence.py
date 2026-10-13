#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.artifacts import artifact_path


SUPPORTED_MODEL_TYPES = {
    "pair_lowrank_quadratic",
    "pair_lowrank_quadratic_strict",
    "pair_attention_biased_lowrank_quadratic",
    "pair_value_biased_lowrank_quadratic",
    "pair_attention_biased_lowrank_quadratic_strict",
    "pair_value_biased_lowrank_quadratic_strict",
}

TITLE_FONTSIZE = 12
METRIC_TITLE_FONTSIZE = 12
EFFECTIVE_RANK_TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
EFFECTIVE_RANK_LEGEND_FONTSIZE = 14
ANNOT_FONTSIZE = 9
SAVEFIG_PAD_INCHES = 0.03
BLOCK_LINESTYLES = ["-", "--", "-.", ":"]
BLOCK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
ALL_POSITIVE_BASELINE_LABEL = "all-positive baseline"

PROBE_DISPLAY_NAMES = {
    "pair_lowrank_quadratic": "Low-rank Quad",
    "pair_lowrank_quadratic_strict": "Strict Low-rank Quad",
    "pair_attention_biased_lowrank_quadratic": "Attention-initialized Low-rank Quad",
    "pair_value_biased_lowrank_quadratic": "Value-initialized Low-rank Quad",
    "pair_attention_biased_lowrank_quadratic_strict": "Attention-initialized Strict Low-rank Quad",
    "pair_value_biased_lowrank_quadratic_strict": "Value-initialized Strict Low-rank Quad",
}

GROUP_DISPLAY_NAMES = {
    "inter_frame_block": "Inter-Frame Block",
    "intra_frame_block": "Intra-Frame Block",
    "cam_head": "Camera Head",
    "depth_head": "Depth Head",
    "point_head": "Point Head",
}


def _title_with_prefix(prefix: str, title: str) -> str:
    return f"{prefix} | {title}" if prefix else title


def _emphasize_model_display_label(label: str) -> str:
    if not label:
        return label
    return rf"$\mathbf{{{label}}}$"


def _infer_probe_display_label(results_root: Path, out_name: str, model_type: str) -> str:
    display = PROBE_DISPLAY_NAMES.get(model_type, model_type)
    if model_type == "pair_lowrank_quadratic":
        text = f"{results_root} {out_name}".lower()
        rank = None
        for candidate in ("8", "16", "256", "4", "2"):
            if f"rank_{candidate}" in text or f"rank{candidate}" in text:
                rank = candidate
                break
        if rank is None and "isdynamic_blockwise_youtubevos" in text:
            rank = "256"
        if rank is not None:
            return f"{display}, k = {rank}"
    return display


def _group_display_label(group: str) -> str:
    return GROUP_DISPLAY_NAMES.get(group, group.replace("_", " ").title())


def _choose_scatter_legend_loc(x: np.ndarray, y: np.ndarray, baseline_y: float) -> str:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_mask):
        return "upper left"
    x = x[finite_mask]
    y = y[finite_mask]
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    dx = max(xmax - xmin, 1e-8)
    dy = max(ymax - ymin, 1e-8)
    xw = 0.38 * dx
    yh = 0.28 * dy
    candidates = {
        "upper left": (xmin, xmin + xw, ymax - yh, ymax),
        "upper right": (xmax - xw, xmax, ymax - yh, ymax),
        "lower left": (xmin, xmin + xw, ymin, ymin + yh),
        "lower right": (xmax - xw, xmax, ymin, ymin + yh),
    }
    order = ["upper right", "upper left", "lower right", "lower left"]
    best_loc = order[0]
    best_score = None
    for loc in order:
        x0, x1, y0, y1 = candidates[loc]
        point_score = int(np.count_nonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))
        baseline_penalty = 0.5 if np.isfinite(baseline_y) and y0 <= float(baseline_y) <= y1 else 0.0
        score = point_score + baseline_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc
    return best_loc


def _block_curve_style(block_idx: int, style_idx: int) -> Dict[str, object]:
    return {
        "linestyle": BLOCK_LINESTYLES[style_idx % len(BLOCK_LINESTYLES)],
    }


def _save_figure(fig, out_path: Path, dpi: int = 220) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    fig.savefig(
        out_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize low-rank evidence from trained IsDynamic pairwise low-rank probe checkpoints."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=artifact_path("outputs", "isdynamic_blockwise_lowrank_all"),
    )
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(SUPPORTED_MODEL_TYPES))
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        choices=["inter_frame_block", "intra_frame_block", "cam_head", "depth_head", "point_head"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
    )
    parser.add_argument("--metric_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--top_singular", type=int, default=128, help="Max singular values to display in spectrum plots.")
    parser.add_argument("--energy_threshold", type=float, default=0.9, help="Cumulative energy threshold, e.g. 0.9.")
    parser.add_argument(
        "--cum_energy_plot_log_y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log scale on y-axis for plot 02 (cumulative energy).",
    )
    parser.add_argument(
        "--rank_plot_log_y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log scale on y-axis for plot 03 (effective rank vs block).",
    )
    parser.add_argument(
        "--selected_blocks",
        type=str,
        default="auto",
        help="Comma-separated block indices for line plots, or 'auto' (=first,mid,last,best-metric).",
    )
    parser.add_argument("--out_dir", type=str, default=artifact_path("illustration", "diagram", "lowrank"))
    parser.add_argument("--name", type=str, default=None, help="Output name prefix.")
    return parser.parse_args()


def _parse_block_idx(name: str, group: str) -> int:
    prefix = f"{group}_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected run dir '{name}' for group '{group}'")
    return int(name[len(prefix) :])


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint does not contain state dict: {path}")
    return obj


def _matrix_from_state_dict(model_type: str, sd: Dict[str, torch.Tensor]) -> np.ndarray:
    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().float().numpy()

    if model_type == "pair_lowrank_quadratic":
        u = to_np(sd["u"])  # [K,C]
        v = to_np(sd["v"])  # [K,C]
        c = u.shape[1]
        m = 0.5 * (u.T @ v + v.T @ u) / math.sqrt(float(c))
        return m

    if model_type == "pair_lowrank_quadratic_strict":
        u = to_np(sd["u"])  # [K,C]
        sign = to_np(sd["sign"]).reshape(-1)  # [K]
        m = u.T @ (sign[:, None] * u)
        return m

    if model_type in {"pair_attention_biased_lowrank_quadratic", "pair_value_biased_lowrank_quadratic"}:
        u = to_np(sd["u"])  # [K,C]
        v = to_np(sd["v"])  # [K,C]
        fixed = to_np(sd["fixed_matrix"])  # [C,C]
        c = u.shape[1]
        residual = 0.5 * (u.T @ v + v.T @ u) / math.sqrt(float(c))
        return fixed + residual

    if model_type in {
        "pair_attention_biased_lowrank_quadratic_strict",
        "pair_value_biased_lowrank_quadratic_strict",
    }:
        u = to_np(sd["u"])  # [K,C]
        sign = to_np(sd["sign"]).reshape(-1)  # [K]
        fixed = to_np(sd["fixed_matrix"])  # [C,C]
        residual = u.T @ (sign[:, None] * u)
        return fixed + residual

    raise ValueError(f"Unsupported model_type for low-rank matrix extraction: {model_type}")


def _effective_rank_entropy(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / np.maximum(s.sum(), 1e-12)
    h = -(p * np.log(np.maximum(p, 1e-12))).sum()
    return float(np.exp(h))


def _effective_rank_participation(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    num = float((s.sum()) ** 2)
    den = float((s**2).sum())
    return 0.0 if den <= 0.0 else num / den


def _rank_at_energy(cum_energy: np.ndarray, threshold: float) -> int:
    idx = int(np.searchsorted(cum_energy, threshold, side="left"))
    return idx + 1


def _extract_metric(metrics: Dict, split: str, metric: str) -> float:
    split_metrics = _extract_split_metrics(metrics, split=split)
    if metric in split_metrics:
        return float(split_metrics[metric])
    if "best" in metrics and isinstance(metrics["best"], dict) and metric in metrics["best"]:
        return float(metrics["best"][metric])
    raise KeyError(f"Could not find metric '{metric}' in split '{split}' for metrics.json")


def _extract_split_metrics(metrics: Dict, split: str) -> Dict:
    def _aggregate_named_test_metrics(named_tests: Dict) -> Dict:
        valid = [v for v in named_tests.values() if isinstance(v, dict)]
        if not valid:
            raise KeyError("Named test metrics are empty or invalid.")
        out: Dict[str, float] = {}
        for key in ("tp", "fp", "fn", "tn"):
            vals = [float(v[key]) for v in valid if key in v]
            if vals:
                out[key] = float(sum(vals))
        tp = float(out.get("tp", 0.0))
        fp = float(out.get("fp", 0.0))
        fn = float(out.get("fn", 0.0))
        tn = float(out.get("tn", 0.0))
        total = tp + fp + fn + tn
        if total > 0:
            out["accuracy"] = float((tp + tn) / total)
            out["precision"] = float(tp / max(tp + fp, 1e-12))
            out["recall"] = float(tp / max(tp + fn, 1e-12))
            out["f1"] = float(2.0 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-12))
        losses = [float(v["loss"]) for v in valid if "loss" in v]
        if losses:
            out["loss"] = float(np.mean(losses))
        return out

    key_map = {"train": "best_train", "val": "best_val", "best": "best"}
    if split in key_map:
        main_key = key_map[split]
        if main_key in metrics and isinstance(metrics[main_key], dict):
            return metrics[main_key]
        if "best" in metrics and isinstance(metrics["best"], dict):
            return metrics["best"]
        raise KeyError(f"Could not find split metrics for split '{split}' in metrics.json")
    if split == "test":
        if isinstance(metrics.get("best_tests"), dict) and metrics["best_tests"]:
            return _aggregate_named_test_metrics(metrics["best_tests"])
        raise KeyError(f"Could not find split metrics for split '{split}' in metrics.json")
    if isinstance(metrics.get("best_tests"), dict) and isinstance(metrics["best_tests"].get(split), dict):
        return metrics["best_tests"][split]
    if "best" in metrics and isinstance(metrics["best"], dict):
        return metrics["best"]
    raise KeyError(f"Could not find split metrics for split '{split}' in metrics.json")


def _all_positive_metric(metric: str, split_metrics: Dict) -> float:
    # Hard-prediction baseline: predict every pair as positive (1).
    try:
        tp = float(split_metrics["tp"])
        fp = float(split_metrics["fp"])
        fn = float(split_metrics["fn"])
        tn = float(split_metrics["tn"])
    except Exception:
        return float("nan")
    total = tp + fp + fn + tn
    if total <= 0:
        return float("nan")
    pos_ratio = (tp + fn) / total
    if metric == "accuracy":
        return float(pos_ratio)
    if metric == "precision":
        return float(pos_ratio)
    if metric == "recall":
        return 1.0
    if metric == "f1":
        return float(2.0 * pos_ratio / max(1.0 + pos_ratio, 1e-12))
    return float("nan")


def _select_blocks(rows: List[Dict], selected_blocks: str, metric: str) -> List[int]:
    blocks = [int(r["block_idx"]) for r in rows]
    if not blocks:
        return []
    if selected_blocks != "auto":
        want = [int(x.strip()) for x in selected_blocks.split(",") if x.strip()]
        return [b for b in want if b in blocks]

    first = blocks[0]
    last = blocks[-1]
    mid = blocks[len(blocks) // 2]
    if metric == "loss":
        best = min(rows, key=lambda r: float(r["metric"]))["block_idx"]
    else:
        best = max(rows, key=lambda r: float(r["metric"]))["block_idx"]
    out = []
    for b in [first, mid, last, int(best)]:
        if b not in out:
            out.append(b)
    return out


def _plot_spectrum_selected(
    rows: List[Dict],
    selected: List[int],
    out_path: Path,
    top_singular: int,
    matrix_dim: int,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    selected_to_style = {block_idx: style_idx for style_idx, block_idx in enumerate(selected)}
    for r in rows:
        b = int(r["block_idx"])
        if b not in selected:
            continue
        s = np.asarray(r["singular_values"], dtype=np.float64)
        n = min(int(top_singular), int(s.size))
        x = np.arange(1, n + 1)
        ax.plot(x, s[:n], linewidth=1.8, label=f"block {b}", **_block_curve_style(b, selected_to_style[b]))
    # Trivial low-rank baseline: full-rank uniform spectrum with matched average Frobenius norm.
    uniform_levels = []
    for r in rows:
        b = int(r["block_idx"])
        if b not in selected:
            continue
        s = np.asarray(r["singular_values"], dtype=np.float64)
        frob = float(np.linalg.norm(s))
        uniform_levels.append(frob / math.sqrt(max(1, matrix_dim)))
    if uniform_levels:
        ax.axhline(
            float(np.mean(uniform_levels)),
            linestyle="--",
            linewidth=1.5,
            color="#444444",
            label="trivial baseline (uniform full-rank spectrum)",
        )
    ax.set_yscale("log")
    ax.set_title(_title_with_prefix(title_prefix, "Singular Spectrum Analysis"), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Singular Value Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Singular Value (Log Scale)", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=220)
    plt.close(fig)


def _plot_cum_energy_selected(
    rows: List[Dict],
    selected: List[int],
    out_path: Path,
    top_singular: int,
    matrix_dim: int,
    log_y: bool,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    selected_to_style = {block_idx: style_idx for style_idx, block_idx in enumerate(selected)}
    for r in rows:
        b = int(r["block_idx"])
        if b not in selected:
            continue
        c = np.asarray(r["cum_energy"], dtype=np.float64)
        n = min(int(top_singular), int(c.size))
        x = np.arange(1, n + 1)
        y = c[:n]
        if log_y:
            y = np.clip(y, 1e-8, None)
        ax.plot(x, y, linewidth=1.8, label=f"block {b}", **_block_curve_style(b, selected_to_style[b]))
    # Trivial low-rank baseline: uniform full-rank energy => linear cumulative curve k / C.
    n = min(int(top_singular), int(matrix_dim))
    x = np.arange(1, n + 1, dtype=np.float64)
    ax.plot(
        x,
        x / float(matrix_dim),
        linestyle="--",
        linewidth=1.5,
        color="#444444",
        label="trivial baseline (uniform full-rank energy)",
    )
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1.01)
    else:
        ax.set_ylim(0.0, 1.01)
    ax.set_title(_title_with_prefix(title_prefix, "Cumulative Energy"), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Rank K", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative Energy", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=220)
    plt.close(fig)


def _plot_effective_rank_vs_block(
    rows: List[Dict],
    out_path: Path,
    matrix_dim: int,
    energy_threshold: float,
    log_y: bool,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    b = np.array([int(r["block_idx"]) for r in rows], dtype=np.int64)
    er = np.array([float(r["effective_rank_entropy"]) for r in rows], dtype=np.float64)
    pr = np.array([float(r["effective_rank_participation"]) for r in rows], dtype=np.float64)
    r90 = np.array([float(r["rank_at_energy_threshold"]) for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.4))
    if log_y:
        er = np.clip(er, 1e-8, None)
        pr = np.clip(pr, 1e-8, None)
        r90 = np.clip(r90, 1e-8, None)
    ax.plot(b, er, marker="o", linewidth=1.8, label="effective rank (entropy)")
    ax.plot(b, pr, marker="s", linewidth=1.8, label="effective rank (participation)")
    ax.plot(b, r90, marker="^", linewidth=1.6, linestyle="--", label="rank@energy-threshold")
    # Trivial full-rank baselines.
    full_rank_baseline = float(matrix_dim)
    threshold_baseline = float(math.ceil(float(energy_threshold) * float(matrix_dim)))
    if log_y:
        full_rank_baseline = max(full_rank_baseline, 1e-8)
        threshold_baseline = max(threshold_baseline, 1e-8)
    ax.axhline(full_rank_baseline, linestyle="--", linewidth=1.2, color="#666666", label="trivial baseline (full rank)")
    ax.axhline(
        threshold_baseline,
        linestyle=":",
        linewidth=1.2,
        color="#666666",
        label="trivial baseline rank@threshold",
    )
    if log_y:
        ax.set_yscale("log")
    ax.set_title(
        _title_with_prefix(title_prefix, "Effective Rank vs Block"),
        fontsize=EFFECTIVE_RANK_TITLE_FONTSIZE,
    )
    ax.set_xlabel("Block Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Effective Rank", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        fontsize=EFFECTIVE_RANK_LEGEND_FONTSIZE,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.96),
    )
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=220)
    plt.close(fig)


def _plot_metric_vs_effective_rank(
    rows: List[Dict],
    out_path: Path,
    metric: str,
    metric_split: str,
    all_positive_baseline_mean: float,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    er = np.array([float(r["effective_rank_entropy"]) for r in rows], dtype=np.float64)
    met = np.array([float(r["metric"]) for r in rows], dtype=np.float64)
    block = np.array([int(r["block_idx"]) for r in rows], dtype=np.int64)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    sc = ax.scatter(er, met, c=block, cmap="viridis", s=55, alpha=0.9, edgecolors="none")
    for x, y, b in zip(er, met, block):
        ax.text(x, y, str(int(b)), fontsize=ANNOT_FONTSIZE, alpha=0.75)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Block Index", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    if np.isfinite(all_positive_baseline_mean):
        ax.axhline(
            float(all_positive_baseline_mean),
            linestyle="--",
            linewidth=1.4,
            color="#cc0000",
            label=ALL_POSITIVE_BASELINE_LABEL,
        )
    ax.set_title(_title_with_prefix(title_prefix, "Metric vs Effective Rank"), fontsize=METRIC_TITLE_FONTSIZE)
    ax.set_xlabel("Effective Rank (Entropy)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("isDynamic Accuracy", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    if np.isfinite(all_positive_baseline_mean):
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            fontsize=LEGEND_FONTSIZE,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.16),
        )
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.results_root) / args.model_type / args.group
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")

    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"{args.group}_")]
    run_dirs.sort(key=lambda p: _parse_block_idx(p.name, args.group))
    if not run_dirs:
        raise RuntimeError(f"No run dirs found under: {base}")

    rows: List[Dict] = []
    for run_dir in run_dirs:
        block_idx = _parse_block_idx(run_dir.name, args.group)
        metrics_path = run_dir / "metrics.json"
        ckpt_path = run_dir / "best_probe.pt"

        if not metrics_path.exists() or not ckpt_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics_obj = json.load(f)

        metric_val = _extract_metric(metrics_obj, split=args.metric_split, metric=args.metric)
        split_metrics = _extract_split_metrics(metrics_obj, split=args.metric_split)
        all_positive_metric = _all_positive_metric(args.metric, split_metrics)
        sd = _load_state_dict(ckpt_path)
        m = _matrix_from_state_dict(args.model_type, sd)
        s = np.linalg.svd(m, compute_uv=False)
        matrix_dim = int(s.size)
        e = (s**2).astype(np.float64)
        e_sum = float(np.maximum(e.sum(), 1e-12))
        cum = np.cumsum(e) / e_sum

        rows.append(
            {
                "block_idx": int(block_idx),
                "metric": float(metric_val),
                "all_positive_metric": float(all_positive_metric),
                "matrix_dim": int(matrix_dim),
                "effective_rank_entropy": _effective_rank_entropy(s),
                "effective_rank_participation": _effective_rank_participation(s),
                "rank_at_energy_threshold": int(_rank_at_energy(cum, float(args.energy_threshold))),
                "singular_values": s.tolist(),
                "cum_energy": cum.tolist(),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid runs with metrics+checkpoint found under: {base}")

    rows.sort(key=lambda r: int(r["block_idx"]))
    matrix_dim_values = sorted({int(r["matrix_dim"]) for r in rows})
    if len(matrix_dim_values) != 1:
        raise RuntimeError(f"Inconsistent matrix dimensions across blocks: {matrix_dim_values}")
    matrix_dim = int(matrix_dim_values[0])
    selected = _select_blocks(rows, args.selected_blocks, args.metric)
    all_positive_vals = np.array([float(r["all_positive_metric"]) for r in rows], dtype=np.float64)
    all_positive_baseline_mean = float(np.nanmean(all_positive_vals)) if np.any(np.isfinite(all_positive_vals)) else float("nan")

    out_name = args.name if args.name else f"{args.model_type}_{args.group}"
    out_dir = Path(args.out_dir) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = _emphasize_model_display_label("VGGT")
    group_prefix = _group_display_label(args.group)
    title_prefix = " | ".join(part for part in (model_prefix, group_prefix) if part)

    _plot_spectrum_selected(
        rows=rows,
        selected=selected,
        out_path=out_dir / "01_singular_spectrum_selected_blocks.png",
        top_singular=int(args.top_singular),
        matrix_dim=matrix_dim,
        title_prefix=title_prefix,
    )
    _plot_cum_energy_selected(
        rows=rows,
        selected=selected,
        out_path=out_dir / "02_cumulative_energy_selected_blocks.png",
        top_singular=int(args.top_singular),
        matrix_dim=matrix_dim,
        log_y=bool(args.cum_energy_plot_log_y),
        title_prefix=title_prefix,
    )
    _plot_effective_rank_vs_block(
        rows=rows,
        out_path=out_dir / "03_effective_rank_vs_block.png",
        matrix_dim=matrix_dim,
        energy_threshold=float(args.energy_threshold),
        log_y=bool(args.rank_plot_log_y),
        title_prefix=title_prefix,
    )
    _plot_metric_vs_effective_rank(
        rows=rows,
        out_path=out_dir / "04_metric_vs_effective_rank.png",
        metric=args.metric,
        metric_split=args.metric_split,
        all_positive_baseline_mean=all_positive_baseline_mean,
        title_prefix=title_prefix,
    )

    # Save summary table.
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "block_idx",
                f"{args.metric_split}_{args.metric}",
                f"all_positive_{args.metric_split}_{args.metric}",
                "effective_rank_entropy",
                "effective_rank_participation",
                f"rank_at_{args.energy_threshold:.2f}_energy",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    int(r["block_idx"]),
                    float(r["metric"]),
                    float(r["all_positive_metric"]),
                    float(r["effective_rank_entropy"]),
                    float(r["effective_rank_participation"]),
                    int(r["rank_at_energy_threshold"]),
                ]
            )

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_root": str(args.results_root),
                "model_type": args.model_type,
                "group": args.group,
                "metric": args.metric,
                "metric_split": args.metric_split,
                "all_positive_baseline": {
                    "type": "all_positive",
                    "mean": all_positive_baseline_mean,
                },
                "energy_threshold": float(args.energy_threshold),
                "selected_blocks": selected,
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"Saved plots and tables under: {out_dir}")
    print(f"- {out_dir / '01_singular_spectrum_selected_blocks.png'}")
    print(f"- {out_dir / '02_cumulative_energy_selected_blocks.png'}")
    print(f"- {out_dir / '03_effective_rank_vs_block.png'}")
    print(f"- {out_dir / '04_metric_vs_effective_rank.png'}")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
