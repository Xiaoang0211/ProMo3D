from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.feature_io import load_token_npz
from .scoremap_utils import load_probe_from_run


MODEL_DISPLAY_NAMES = {
    "pi3": r"$\mathbf{\pi^3}$",
    "pi3x": r"$\mathbf{\pi^3X}$",
    "vggt": r"$\mathbf{VGGT}$",
    "depth_anything_3": r"$\mathbf{DA3}$",
    "da3": r"$\mathbf{DA3}$",
    "map_anything": r"$\mathbf{MapAnything}$",
    "dust3r": r"$\mathbf{DUSt3R}$",
}

GROUP_DISPLAY_NAMES = {
    "inter_frame_block": "Inter-Frame Block",
    "intra_frame_block": "Intra-Frame Block",
    "dec_view1": "Decoder View-1",
    "dec_view2": "Decoder View-2",
    "head_in_view1": "Head-in-View-1",
    "head_in_view2": "Head-in-View-2",
}


def _group_display_name(group: str, model_name: str = "") -> str:
    model_name = str(model_name or "").strip().lower()
    if group in {"depth_head", "dual_dpt_head"}:
        return "depth head"
    return GROUP_DISPLAY_NAMES.get(group, group.replace("_", " ").title())

BENCHMARK_DISPLAY_NAMES = {
    "test_davis": "DAVIS-all",
    "test_fbms59": "FBMS-59",
    "test_segtrackv2": "SegTrackv2",
}


class FeatureReservoir:
    """Fixed-size reservoir sampler for vectors or scores."""

    def __init__(self, max_size: int, rng: np.random.Generator):
        self.max_size = int(max_size)
        self.rng = rng
        self.seen = 0
        self.size = 0
        self.buf: Optional[np.ndarray] = None

    def add_batch(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got {x.shape}")
        if x.shape[0] == 0:
            return
        if self.buf is None:
            self.buf = np.empty((self.max_size, x.shape[1]), dtype=np.float32)
        elif self.buf.shape[1] != x.shape[1]:
            raise ValueError(
                f"Inconsistent feature dimension: existing={self.buf.shape[1]} new={x.shape[1]}"
            )
        for row in x:
            self.seen += 1
            if self.size < self.max_size:
                self.buf[self.size] = row
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.max_size:
                    self.buf[j] = row

    def as_array(self) -> np.ndarray:
        if self.buf is None or self.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        return self.buf[: self.size].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize query-conditioned probe separability with PCA. "
            "Produces both (1) PCA of query-conditioned probe evidence and "
            "(2) score histograms from the trained pairwise probe."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--flat_npz", type=str, default=None, help="Flat NPZ path.")
    src.add_argument("--sharded_split_root", type=str, default=None, help="Sharded split root.")
    parser.add_argument("--feature_key", type=str, required=True, help="Feature key / block name.")
    parser.add_argument("--probe_run_dir", type=str, required=True, help="Exact run dir with config.json and checkpoint.")
    parser.add_argument("--label_key", type=str, default="is_dynamic")
    parser.add_argument("--valid_key", type=str, default="valid_mask")
    parser.add_argument("--group_key", type=str, default="group_ids")
    parser.add_argument(
        "--query_label",
        type=str,
        default="static",
        choices=("static", "dynamic"),
        help="Choose one query token per group from this class.",
    )
    parser.add_argument(
        "--query_pick",
        type=str,
        default="first",
        choices=("first", "random"),
        help="How to choose the query token inside each group.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_groups", type=int, default=0, help="0 means use all groups.")
    parser.add_argument("--max_samples_per_class", type=int, default=10000)
    parser.add_argument("--plot_max_points_per_class", type=int, default=2500)
    parser.add_argument("--hist_max_scores_per_class", type=int, default=50000)
    parser.add_argument("--marker_size", type=float, default=3.0, help="2D scatter marker size.")
    parser.add_argument("--marker_size_3d", type=float, default=3.0, help="3D scatter marker size.")
    parser.add_argument("--marker_alpha", type=float, default=0.22)
    parser.add_argument("--make_teaser_3d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teaser_grid_size", type=int, default=96)
    parser.add_argument("--teaser_surface_alpha", type=float, default=0.22)
    parser.add_argument("--teaser_elev", type=float, default=60.0)
    parser.add_argument("--teaser_azim", type=float, default=-80.0)
    parser.add_argument("--hist_bins", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="outputs/isdynamic_visualizations/PCA")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--make_3d", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_feature_key_for_npz(npz_path: Path, requested_key: str) -> str:
    with np.load(npz_path, allow_pickle=False) as data:
        if requested_key in data:
            return requested_key
        if "tokens" in data:
            return "tokens"
        raise KeyError(
            f"{npz_path} missing feature key '{requested_key}' and fallback key 'tokens'. "
            f"Available keys: {list(data.files)}"
        )


def _candidate_feature_dirs(feature_key: str) -> List[str]:
    candidates = [feature_key]
    alias_prefixes = {
        "inter_frame_block_": "glob_attn_",
        "intra_frame_block_": "perframe_attn_",
    }
    for src_prefix, dst_prefix in alias_prefixes.items():
        if feature_key.startswith(src_prefix):
            candidates.append(dst_prefix + feature_key[len(src_prefix) :])
            break
    return candidates


def _iter_sharded_npz(split_root: Path, feature_key: str) -> List[Path]:
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")
    paths: List[Path] = []
    candidate_dirs = _candidate_feature_dirs(feature_key)
    for seq_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
        for candidate in candidate_dirs:
            feat_dir = seq_dir / candidate
            if not feat_dir.exists():
                continue
            paths.extend(sorted(feat_dir.glob("part_*.npz")))
            break
    if not paths:
        raise FileNotFoundError(
            f"No shard npz found for feature_key='{feature_key}' "
            f"(candidates={candidate_dirs}) under split root: {split_root}"
        )
    return paths


def _query_local_index(labels: np.ndarray, query_label: str, query_pick: str, rng: np.random.Generator) -> Optional[int]:
    cls = 0 if query_label == "static" else 1
    idx = np.flatnonzero(labels == cls)
    if idx.size == 0:
        return None
    if query_pick == "first":
        return int(idx[0])
    return int(rng.choice(idx))


def _compute_query_conditioned_features(
    model: torch.nn.Module,
    token_features: torch.Tensor,
    query_idx: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    query = token_features[query_idx : query_idx + 1]

    with torch.no_grad():
        if hasattr(model, "fixed_matrix") and hasattr(model, "a"):
            residual_w = 0.5 * (model.a + model.a.transpose(0, 1)) / float(getattr(model, "scale", 1.0))
            total_w = -(model.fixed_matrix + residual_w)
            wy = token_features @ total_w.transpose(0, 1)
            evidence = query.expand_as(wy) * wy
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "a"):
            w = 0.5 * (model.a + model.a.transpose(0, 1)) / float(getattr(model, "scale", 1.0))
            wy = token_features @ w.transpose(0, 1)
            evidence = query.expand_as(wy) * wy
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "diag"):
            evidence = (query * token_features * model.diag) / float(getattr(model, "scale", 1.0))
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "attention_matrix") and not hasattr(model, "u") and not hasattr(model, "a"):
            total_w = -model.attention_matrix
            wy = token_features @ total_w.transpose(0, 1)
            evidence = query.expand_as(wy) * wy
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "value_matrix") and not hasattr(model, "u") and not hasattr(model, "a"):
            total_w = -model.value_matrix
            wy = token_features @ total_w.transpose(0, 1)
            evidence = query.expand_as(wy) * wy
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "u") and hasattr(model, "v"):
            q_u = query @ model.u.transpose(0, 1)
            q_v = query @ model.v.transpose(0, 1)
            y_u = token_features @ model.u.transpose(0, 1)
            y_v = token_features @ model.v.transpose(0, 1)
            evidence = 0.5 * ((q_u * y_v) + (q_v * y_u)) / float(getattr(model, "scale", 1.0))
            if hasattr(model, "fixed_matrix"):
                base = -torch.einsum("bi,ij,nj->n", query, model.fixed_matrix, token_features)
                evidence = torch.cat([base[:, None], -evidence], dim=1)
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        if hasattr(model, "u") and hasattr(model, "sign"):
            q_u = query @ model.u.transpose(0, 1)
            y_u = token_features @ model.u.transpose(0, 1)
            evidence = q_u * y_u * model.sign
            if hasattr(model, "fixed_matrix"):
                base = -torch.einsum("bi,ij,nj->n", query, model.fixed_matrix, token_features)
                evidence = torch.cat([base[:, None], -evidence], dim=1)
            scores: List[torch.Tensor] = []
            for start in range(0, token_features.shape[0], batch_size):
                end = min(start + batch_size, token_features.shape[0])
                q_batch = query.expand(end - start, -1)
                logits = model(q_batch, token_features[start:end])
                scores.append(torch.sigmoid(logits))
            return (
                evidence.detach().cpu().numpy().astype(np.float32),
                torch.cat(scores, dim=0).detach().cpu().numpy().astype(np.float32),
            )

        raise ValueError(
            f"Unsupported probe type for query-conditioned PCA: {type(model).__name__}. "
            "Expected a supported pairwise probe."
        )


def _load_npz_arrays(
    npz_path: Path,
    feature_key: str,
    label_key: str,
    valid_key: str,
    group_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_feature_key = _resolve_feature_key_for_npz(npz_path, feature_key)
    feats_t, labels_t, valid_t, groups_t = load_token_npz(
        str(npz_path),
        feature_key=resolved_feature_key,
        label_key=label_key,
        valid_key=valid_key,
        group_key=group_key,
    )
    return (
        feats_t.numpy().astype(np.float32, copy=False),
        labels_t.numpy().astype(np.int64, copy=False),
        valid_t.numpy().astype(np.bool_, copy=False),
        groups_t.numpy().astype(np.int64, copy=False),
    )


def _collect_query_conditioned_samples(
    npz_paths: Sequence[Path],
    feature_key: str,
    label_key: str,
    valid_key: str,
    group_key: str,
    model: torch.nn.Module,
    device: torch.device,
    query_label: str,
    query_pick: str,
    max_groups: int,
    max_samples_per_class: int,
    hist_max_scores_per_class: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    feat_reservoirs = {
        0: FeatureReservoir(max_samples_per_class, rng),
        1: FeatureReservoir(max_samples_per_class, rng),
    }
    score_reservoirs = {
        0: FeatureReservoir(hist_max_scores_per_class, rng),
        1: FeatureReservoir(hist_max_scores_per_class, rng),
    }
    raw_counts = {0: 0, 1: 0}
    used_groups = 0
    skipped_groups_no_query = 0

    for npz_path in npz_paths:
        feats, labels, valid, group_ids = _load_npz_arrays(
            npz_path=npz_path,
            feature_key=feature_key,
            label_key=label_key,
            valid_key=valid_key,
            group_key=group_key,
        )
        feats = feats[valid]
        labels = labels[valid]
        group_ids = group_ids[valid]
        if feats.shape[0] == 0:
            continue

        for gid in np.unique(group_ids):
            if max_groups > 0 and used_groups >= max_groups:
                break
            idx = np.flatnonzero(group_ids == gid)
            if idx.size == 0:
                continue
            group_feats = feats[idx]
            group_labels = labels[idx]
            q_local = _query_local_index(group_labels, query_label=query_label, query_pick=query_pick, rng=rng)
            if q_local is None:
                skipped_groups_no_query += 1
                continue
            used_groups += 1
            token_features = torch.from_numpy(group_feats).to(device=device, dtype=torch.float32)
            evidence_np, score_np = _compute_query_conditioned_features(
                model=model,
                token_features=token_features,
                query_idx=q_local,
                batch_size=batch_size,
            )
            for cls in (0, 1):
                cls_mask = group_labels == cls
                cls_evidence = evidence_np[cls_mask]
                cls_scores = score_np[cls_mask]
                raw_counts[cls] += int(cls_mask.sum())
                feat_reservoirs[cls].add_batch(cls_evidence.astype(np.float32, copy=False))
                score_reservoirs[cls].add_batch(cls_scores.astype(np.float32, copy=False))
        if max_groups > 0 and used_groups >= max_groups:
            break

    x0 = feat_reservoirs[0].as_array()
    x1 = feat_reservoirs[1].as_array()
    s0 = score_reservoirs[0].as_array().reshape(-1)
    s1 = score_reservoirs[1].as_array().reshape(-1)
    if x0.size == 0 or x1.size == 0:
        raise RuntimeError(
            f"Need both classes for query-conditioned PCA, got static={x0.shape[0]} dynamic={x1.shape[0]}"
        )
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate(
        [np.zeros(x0.shape[0], dtype=np.int64), np.ones(x1.shape[0], dtype=np.int64)],
        axis=0,
    )
    info = {
        "raw_static_count": raw_counts[0],
        "raw_dynamic_count": raw_counts[1],
        "sampled_static_count": int(x0.shape[0]),
        "sampled_dynamic_count": int(x1.shape[0]),
        "sampled_static_scores": int(s0.shape[0]),
        "sampled_dynamic_scores": int(s1.shape[0]),
        "used_groups": int(used_groups),
        "skipped_groups_without_query": int(skipped_groups_no_query),
    }
    scores = np.concatenate([s0, s1], axis=0)
    score_labels = np.concatenate(
        [np.zeros(s0.shape[0], dtype=np.int64), np.ones(s1.shape[0], dtype=np.int64)],
        axis=0,
    )
    return x, y, scores, score_labels, {**info, "score_label_count": int(score_labels.shape[0])}


def _pca_3d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected [N,C], got {x.shape}")
    if x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError(f"Need enough data for PCA, got {x.shape}")
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    n_comp = min(3, vt.shape[0])
    comp = vt[:n_comp]
    z = xc @ comp.T
    var = (s**2) / max(1, (x.shape[0] - 1))
    var_ratio = var / max(1e-12, var.sum())
    if n_comp < 3:
        z = np.pad(z, ((0, 0), (0, 3 - n_comp)))
        var_ratio = np.pad(var_ratio[:n_comp], (0, 3 - n_comp))
    else:
        var_ratio = var_ratio[:3]
    return z, var_ratio


def _subsample_for_plot(
    z: np.ndarray,
    y: np.ndarray,
    max_points_per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    keep_idx: List[np.ndarray] = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        if idx.size > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        keep_idx.append(np.sort(idx))
    idx_all = np.concatenate(keep_idx, axis=0)
    return z[idx_all], y[idx_all]


def _quadratic_design_matrix(z: np.ndarray) -> np.ndarray:
    x = z[:, 0]
    y = z[:, 1]
    w = z[:, 2]
    return np.stack(
        [
            np.ones_like(x),
            x,
            y,
            w,
            x * x,
            y * y,
            w * w,
            x * y,
            x * w,
            y * w,
        ],
        axis=1,
    ).astype(np.float64, copy=False)


def _fit_quadratic_separator(z: np.ndarray, labels: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    phi = _quadratic_design_matrix(z)
    targets = np.where(labels > 0, 1.0, -1.0).astype(np.float64, copy=False)
    eye = np.eye(phi.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    lhs = phi.T @ phi + ridge * eye
    rhs = phi.T @ targets
    return np.linalg.solve(lhs, rhs)


def _quadratic_scores_from_coeffs(z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return _quadratic_design_matrix(z) @ coeffs


def _build_teaser_surface(
    z: np.ndarray,
    coeffs: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad = 0.08
    mins = z.min(axis=0)
    maxs = z.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    mins = mins - pad * spans
    maxs = maxs + pad * spans

    xs = np.linspace(float(mins[0]), float(maxs[0]), int(grid_size), dtype=np.float64)
    ys = np.linspace(float(mins[1]), float(maxs[1]), int(grid_size), dtype=np.float64)
    zs = np.linspace(float(mins[2]), float(maxs[2]), int(grid_size), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz_surface = np.full_like(xx, np.nan, dtype=np.float64)

    flat_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z_candidates = np.repeat(zs[None, :], flat_xy.shape[0], axis=0)
    x_candidates = np.repeat(flat_xy[:, 0:1], zs.shape[0], axis=1)
    y_candidates = np.repeat(flat_xy[:, 1:2], zs.shape[0], axis=1)
    pts = np.stack(
        [x_candidates.reshape(-1), y_candidates.reshape(-1), z_candidates.reshape(-1)],
        axis=1,
    )
    scores = _quadratic_scores_from_coeffs(pts, coeffs).reshape(flat_xy.shape[0], zs.shape[0])
    best_idx = np.argmin(np.abs(scores), axis=1)
    chosen = zs[best_idx]
    best_score = scores[np.arange(scores.shape[0]), best_idx]
    tolerance = max(0.08, float(np.percentile(np.abs(best_score), 65)))
    chosen[np.abs(best_score) > tolerance] = np.nan
    zz_surface[:] = chosen.reshape(xx.shape)
    return xx, yy, zz_surface


def _save_figure_png_pdf(
    fig,
    out_path: Path,
    dpi: int = 220,
    *,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.03,
) -> Path:
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return pdf_path


def _display_model_name_from_out_dir(out_dir: Path) -> str:
    parts = [part.lower() for part in out_dir.parts]
    for key, label in MODEL_DISPLAY_NAMES.items():
        if key in parts:
            return label
    return ""


def _display_group_name(feature_key: str) -> str:
    base = feature_key
    block_idx = ""
    if "_" in feature_key:
        maybe_idx = feature_key.rsplit("_", 1)[1]
        if maybe_idx.isdigit():
            base = feature_key.rsplit("_", 1)[0]
            block_idx = maybe_idx
    label = _group_display_name(base, args.model)
    if block_idx:
        return f"{label} {block_idx}"
    return label


def _display_benchmark_name(flat_npz: str | None, sharded_split_root: str | None) -> str:
    candidate = ""
    if sharded_split_root:
        candidate = Path(sharded_split_root).name
    elif flat_npz:
        candidate = Path(flat_npz).stem
    candidate = candidate.lower()
    return BENCHMARK_DISPLAY_NAMES.get(candidate, candidate.replace("_", " ").title() if candidate else "")


def _panel_title(out_dir: Path, feature_key: str, flat_npz: str | None, sharded_split_root: str | None) -> str:
    pieces = [
        _display_model_name_from_out_dir(out_dir),
        _display_group_name(feature_key),
        _display_benchmark_name(flat_npz, sharded_split_root),
    ]
    return " | ".join(piece for piece in pieces if piece)


def _plot_2d_pairs(
    z: np.ndarray,
    y: np.ndarray,
    var_ratio: np.ndarray,
    out_path: Path,
    marker_size: float,
    marker_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    title_fontsize = 16
    label_fontsize = 15
    tick_fontsize = 14
    legend_fontsize = 14
    pairs = [(0, 1), (1, 2), (2, 0)]
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "static", 1: "dynamic"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        for cls in (0, 1):
            pts = z[y == cls]
            ax.scatter(pts[:, i], pts[:, j], s=marker_size, alpha=marker_alpha, c=colors[cls], edgecolors="none")
        ax.set_title(f"PC{i+1} vs PC{j+1}", fontweight="bold", fontsize=title_fontsize)
        ax.set_xlabel(f"PC{i+1} ({var_ratio[i] * 100:.1f}%)", fontsize=label_fontsize)
        ax.set_ylabel(f"PC{j+1} ({var_ratio[j] * 100:.1f}%)", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(alpha=0.2, linewidth=0.6)
    axes[0][0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=7, label=labels[0]),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[1], markeredgecolor=colors[1], markersize=7, label=labels[1]),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=legend_fontsize,
    )
    fig.tight_layout()
    _save_figure_png_pdf(fig, out_path, dpi=220)
    plt.close(fig)


def _plot_panel(
    z: np.ndarray,
    y: np.ndarray,
    var_ratio: np.ndarray,
    out_path: Path,
    title: str,
    marker_size_2d: float,
    marker_size_3d: float,
    marker_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    title_fontsize = 15
    label_fontsize = 14
    tick_fontsize = 13
    legend_fontsize = 15
    pairs = [(0, 1), (1, 2), (2, 0)]
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "static", 1: "dynamic"}
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=9, label=labels[0]),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[1], markeredgecolor=colors[1], markersize=9, label=labels[1]),
    ]
    fig = plt.figure(figsize=(21.0, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 1.0, 1.0, 1.0], wspace=0.24)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    for cls in (0, 1):
        pts = z[y == cls]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=marker_size_3d, alpha=marker_alpha, c=colors[cls], depthshade=False)
    ax3d.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=15, labelpad=10)
    ax3d.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=15, labelpad=10)
    ax3d.set_zlabel(f"PC3 ({var_ratio[2] * 100:.1f}%)", fontsize=15, labelpad=14)
    ax3d.tick_params(axis="both", which="major", labelsize=13)
    ax3d.tick_params(axis="z", which="major", labelsize=13)
    axes_2d = [fig.add_subplot(gs[0, idx]) for idx in range(1, 4)]
    for ax, (i, j) in zip(axes_2d, pairs):
        for cls in (0, 1):
            pts = z[y == cls]
            ax.scatter(pts[:, i], pts[:, j], s=marker_size_2d, alpha=marker_alpha, c=colors[cls], edgecolors="none")
        ax.set_title(f"PC{i+1} vs PC{j+1}", fontweight="bold", fontsize=title_fontsize)
        ax.set_xlabel(f"PC{i+1} ({var_ratio[i] * 100:.1f}%)", fontsize=label_fontsize)
        ax.set_ylabel(f"PC{j+1} ({var_ratio[j] * 100:.1f}%)", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=16,
        bbox_to_anchor=(0.03, 0.89),
    )
    if title:
        fig.suptitle(title, fontsize=17, y=0.975)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.12, top=0.79, wspace=0.22)
    _save_figure_png_pdf(fig, out_path, dpi=220, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def _plot_3d(
    z: np.ndarray,
    y: np.ndarray,
    var_ratio: np.ndarray,
    out_path: Path,
    marker_size: float,
    marker_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    label_fontsize = 22
    tick_fontsize = 17
    legend_fontsize = 24
    colors = {0: "#1f77b4", 1: "#d62728"}
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for cls in (0, 1):
        pts = z[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=marker_size, alpha=marker_alpha, c=colors[cls], depthshade=False)
    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=label_fontsize, labelpad=1)
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=label_fontsize, labelpad=1)
    ax.set_zlabel(f"PC3 ({var_ratio[2] * 100:.1f}%)", fontsize=label_fontsize, labelpad=3)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="z", which="major", labelsize=tick_fontsize)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=11, label="static"),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[1], markeredgecolor=colors[1], markersize=11, label="dynamic"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=legend_fontsize,
    )
    fig.subplots_adjust(left=0.01, right=0.95, bottom=0.015, top=0.96)
    _save_figure_png_pdf(fig, out_path, dpi=220, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def _plot_3d_teaser(
    z: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    marker_size: float,
    marker_alpha: float,
    surface_alpha: float,
    grid_size: int,
    teaser_elev: float,
    teaser_azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    colors = {0: "#2b6cb0", 1: "#d62728"}
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    teaser_marker_size = marker_size * 3.4
    for cls in (0, 1):
        pts = z[y == cls]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=teaser_marker_size,
            alpha=marker_alpha,
            c=colors[cls],
            depthshade=False,
            edgecolors="none",
        )
    q_lo = np.percentile(z, 1.0, axis=0)
    q_hi = np.percentile(z, 99.0, axis=0)
    spans = np.maximum(q_hi - q_lo, 1e-6)
    centers = (q_lo + q_hi) / 2.0
    radius = 0.42 * np.max(spans)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.88))
    ax.view_init(elev=teaser_elev, azim=teaser_azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="major", length=0, pad=-2)
    ax.tick_params(axis="z", which="major", length=0, pad=-1)
    ax.grid(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=14, label="static"),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colors[1], markeredgecolor=colors[1], markersize=14, label="dynamic"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=28,
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    _save_figure_png_pdf(fig, out_path, dpi=240, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def _plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    bins: int,
) -> Dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_fontsize = 13
    tick_fontsize = 12
    legend_fontsize = 12
    static_scores = scores[labels == 0]
    dynamic_scores = scores[labels == 1]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(static_scores, bins=bins, range=(0.0, 1.0), density=True, alpha=0.45, color="#1f77b4", label="static")
    ax.hist(dynamic_scores, bins=bins, range=(0.0, 1.0), density=True, alpha=0.45, color="#d62728", label="dynamic")
    ax.set_xlabel("isDynamic score", fontsize=label_fontsize)
    ax.set_ylabel("Density", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95, fontsize=legend_fontsize)
    fig.tight_layout()
    _save_figure_png_pdf(fig, out_path, dpi=220)
    plt.close(fig)
    return {
        "static_mean_score": float(static_scores.mean()) if static_scores.size else float("nan"),
        "dynamic_mean_score": float(dynamic_scores.mean()) if dynamic_scores.size else float("nan"),
        "static_std_score": float(static_scores.std()) if static_scores.size else float("nan"),
        "dynamic_std_score": float(dynamic_scores.std()) if dynamic_scores.size else float("nan"),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model = load_probe_from_run(Path(args.probe_run_dir), device=device)

    if args.flat_npz is not None:
        npz_paths = [Path(args.flat_npz)]
        source_desc = f"flat:{args.flat_npz}"
    else:
        npz_paths = _iter_sharded_npz(Path(args.sharded_split_root), args.feature_key)
        source_desc = f"sharded:{args.sharded_split_root}"

    x, y, scores, score_labels, info = _collect_query_conditioned_samples(
        npz_paths=npz_paths,
        feature_key=args.feature_key,
        label_key=args.label_key,
        valid_key=args.valid_key,
        group_key=args.group_key,
        model=model,
        device=device,
        query_label=args.query_label,
        query_pick=args.query_pick,
        max_groups=int(args.max_groups),
        max_samples_per_class=int(args.max_samples_per_class),
        hist_max_scores_per_class=int(args.hist_max_scores_per_class),
        batch_size=int(args.batch_size),
        rng=rng,
    )
    z, var_ratio = _pca_3d(x)
    z_plot, y_plot = _subsample_for_plot(z, y, max_points_per_class=int(args.plot_max_points_per_class), rng=rng)

    out_dir = Path(args.out_dir)
    name = args.name if args.name else args.feature_key
    p2 = out_dir / f"{name}_PCA_2d.png"
    _plot_2d_pairs(
        z_plot,
        y_plot,
        var_ratio,
        p2,
        marker_size=float(args.marker_size),
        marker_alpha=float(args.marker_alpha),
    )
    panel_path = out_dir / f"{name}_PCA_panel.png"
    panel_title = _panel_title(
        out_dir=out_dir,
        feature_key=args.feature_key,
        flat_npz=args.flat_npz,
        sharded_split_root=args.sharded_split_root,
    )
    _plot_panel(
        z_plot,
        y_plot,
        var_ratio,
        panel_path,
        panel_title,
        marker_size_2d=float(args.marker_size),
        marker_size_3d=float(args.marker_size_3d),
        marker_alpha=float(args.marker_alpha),
    )
    p3 = None
    if args.make_3d:
        p3 = out_dir / f"{name}_PCA_3d.png"
        _plot_3d(
            z_plot,
            y_plot,
            var_ratio,
            p3,
            marker_size=float(args.marker_size_3d),
            marker_alpha=float(args.marker_alpha),
        )
    teaser_p3 = None
    if args.make_teaser_3d:
        teaser_p3 = out_dir / f"{name}_PCA_3d_teaser.png"
        _plot_3d_teaser(
            z_plot,
            y_plot,
            teaser_p3,
            marker_size=float(args.marker_size_3d),
            marker_alpha=float(args.marker_alpha),
            surface_alpha=float(args.teaser_surface_alpha),
            grid_size=int(args.teaser_grid_size),
            teaser_elev=float(args.teaser_elev),
            teaser_azim=float(args.teaser_azim),
        )
    hist_path = out_dir / f"{name}_PCA_score_hist.png"
    score_stats = _plot_score_histogram(scores=scores, labels=score_labels, out_path=hist_path, bins=int(args.hist_bins))

    stats = {
        "name": name,
        "source": source_desc,
        "feature_key": args.feature_key,
        "probe_run_dir": str(args.probe_run_dir),
        "query_label": args.query_label,
        "query_pick": args.query_pick,
        "max_groups": int(args.max_groups),
        "max_samples_per_class": int(args.max_samples_per_class),
        "plot_max_points_per_class": int(args.plot_max_points_per_class),
        "hist_max_scores_per_class": int(args.hist_max_scores_per_class),
        "sample_info": info,
        "pca_explained_variance_ratio_pc1_pc2_pc3": [float(v) for v in var_ratio.tolist()],
        "score_stats": score_stats,
        "outputs": {
            "PCA_2d": str(p2),
            "PCA_2d_pdf": str(p2.with_suffix(".pdf")),
            "PCA_panel": str(panel_path),
            "PCA_panel_pdf": str(panel_path.with_suffix(".pdf")),
            "PCA_3d": str(p3) if p3 else None,
            "PCA_3d_pdf": str(p3.with_suffix(".pdf")) if p3 else None,
            "PCA_3d_teaser": str(teaser_p3) if teaser_p3 else None,
            "PCA_3d_teaser_pdf": str(teaser_p3.with_suffix(".pdf")) if teaser_p3 else None,
            "PCA_score_hist": str(hist_path),
            "PCA_score_hist_pdf": str(hist_path.with_suffix(".pdf")),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"{name}_PCA_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Saved 2D PCA plot: {p2}")
    print(f"Saved 2D PCA PDF: {p2.with_suffix('.pdf')}")
    if p3:
        print(f"Saved 3D PCA plot: {p3}")
        print(f"Saved 3D PCA PDF: {p3.with_suffix('.pdf')}")
    if teaser_p3:
        print(f"Saved 3D teaser PCA plot: {teaser_p3}")
        print(f"Saved 3D teaser PCA PDF: {teaser_p3.with_suffix('.pdf')}")
    print(f"Saved PCA score histogram: {hist_path}")
    print(f"Saved PCA score histogram PDF: {hist_path.with_suffix('.pdf')}")
    print(f"Saved PCA stats: {stats_path}")


if __name__ == "__main__":
    main()
