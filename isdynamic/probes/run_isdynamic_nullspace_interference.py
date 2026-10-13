from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from isdynamic.core.common_models import build_probe
from isdynamic.dispatch import run_legacy_entry


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _savez_compressed_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flat_cache_is_valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return all(key in data for key in ("tokens", "is_dynamic", "valid_mask", "group_ids"))
    except Exception:
        return False


def _candidate_feature_keys(sharded_root: Path, limit: int = 12) -> list[str]:
    feature_keys: set[str] = set()
    try:
        sequence_dirs = sorted(path for path in sharded_root.iterdir() if path.is_dir())
    except FileNotFoundError:
        return []
    for seq_dir in sequence_dirs:
        try:
            for feat_dir in seq_dir.iterdir():
                if feat_dir.is_dir():
                    feature_keys.add(feat_dir.name)
        except FileNotFoundError:
            continue
        if len(feature_keys) >= limit:
            break
    return sorted(feature_keys)[:limit]


def _npz_stem_candidates(paths: Iterable[object]) -> list[str]:
    stems: list[str] = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        stem = Path(raw_path).stem
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def _infer_source_feature_key(source_run_dir: Path, source_cfg: dict) -> str:
    dataset_cfg = source_cfg.get("dataset", {})
    out_cfg = source_cfg.get("output", {})
    candidates = _npz_stem_candidates(
        [
            dataset_cfg.get("train_npz"),
            dataset_cfg.get("val_npz"),
            *(dataset_cfg.get("test_npzs", {}) or {}).values(),
            out_cfg.get("out_dir"),
            str(source_run_dir),
        ]
    )
    if not candidates:
        raise ValueError(f"Could not infer feature key from source run: {source_run_dir}")
    return candidates[0]


def _feature_key_candidates(feature_key: str) -> list[str]:
    candidates = [feature_key]
    if feature_key.startswith("inter_frame_block_"):
        candidates.append(feature_key.replace("inter_frame_block_", "glob_attn_", 1))
    if feature_key.startswith("intra_frame_block_"):
        candidates.append(feature_key.replace("intra_frame_block_", "perframe_attn_", 1))
    seen = set()
    ordered = []
    for key in candidates:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def materialize_feature_npz(
    sharded_root: Path,
    cache_dir: Path,
    feature_key: str,
    split_name: str,
    rebuild: bool,
) -> Path:
    split_cache_dir = cache_dir / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_cache_dir / f"{feature_key}.npz"
    if out_path.exists() and not rebuild and _flat_cache_is_valid(out_path):
        return out_path

    sequence_dirs = sorted([p for p in sharded_root.iterdir() if p.is_dir()])
    raw_parts = []
    dim_token_count: Dict[int, int] = {}

    for seq_dir in sequence_dirs:
        feat_dir = None
        for candidate_key in _feature_key_candidates(feature_key):
            candidate_dir = seq_dir / candidate_key
            if candidate_dir.exists() and candidate_dir.is_dir():
                feat_dir = candidate_dir
                break
        if feat_dir is None:
            continue
        for part_path in sorted(feat_dir.glob("part_*.npz")):
            try:
                with np.load(part_path, allow_pickle=False) as part:
                    tokens = part["tokens"].astype(np.float32, copy=False)
                    labels = part["is_dynamic"].astype(np.int64, copy=False)
                    valid = part["valid_mask"].astype(np.bool_, copy=False)
                    groups = part["group_ids"].astype(np.int64, copy=False)
            except Exception as exc:
                print(f"[warn] skipping unreadable part: {part_path} ({exc})")
                continue

            if tokens.ndim != 2:
                print(f"[warn] skipping malformed part (tokens.ndim={tokens.ndim}): {part_path}")
                continue
            n = int(tokens.shape[0])
            if labels.shape[0] != n or valid.shape[0] != n or groups.shape[0] != n:
                print(f"[warn] skipping malformed part (length mismatch): {part_path}")
                continue

            d = int(tokens.shape[1])
            dim_token_count[d] = dim_token_count.get(d, 0) + n
            raw_parts.append((d, tokens, labels, valid, groups))

    if not raw_parts:
        available = _candidate_feature_keys(sharded_root)
        available_hint = f" Available feature dirs include: {', '.join(available)}" if available else ""
        raise RuntimeError(
            f"No valid part files found for feature '{feature_key}' in {sharded_root}.{available_hint}"
        )

    target_dim = sorted(dim_token_count.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    token_parts = []
    label_parts = []
    valid_parts = []
    group_parts = []
    for d, tokens, labels, valid, groups in raw_parts:
        if d != target_dim:
            continue
        token_parts.append(tokens)
        label_parts.append(labels)
        valid_parts.append(valid)
        group_parts.append(groups)

    tokens = np.concatenate(token_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    valid = np.concatenate(valid_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)

    _savez_compressed_atomic(
        out_path,
        tokens=tokens,
        is_dynamic=labels,
        valid_mask=valid,
        group_ids=groups,
        meta=np.array(json.dumps({"feature_key": feature_key, "source_root": str(sharded_root)})),
    )
    return out_path


def _resolve_split_root(sharded_root: Path, split_name: str) -> Path:
    candidate = sharded_root / split_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"split root not found: {candidate}")


def _orthonormalize_columns(matrix: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError("expected a non-empty 2D matrix for basis extraction")
    q, r = torch.linalg.qr(matrix.float(), mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    keep = diag > tol
    if int(keep.sum()) == 0:
        raise ValueError("basis extraction produced no non-zero directions")
    return q[:, keep].contiguous()


def _basis_complement(basis: torch.Tensor, input_dim: int) -> torch.Tensor:
    if basis.ndim != 2 or basis.shape[0] != input_dim:
        raise ValueError(f"basis shape mismatch: expected (*, {input_dim}), got {tuple(basis.shape)}")
    q_full, _ = torch.linalg.qr(basis.float(), mode="complete")
    return q_full[:, basis.shape[1]:].contiguous()


def _extract_basis_from_lowrank_run(
    run_dir: Path,
    subspace_rank: int,
    subspace_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, dict]:
    cfg = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")
    model_cfg = cfg["model"]
    model_type = str(model_cfg["model_type"])
    input_dim = int(model_cfg["input_dim"])
    source_rank = int(model_cfg.get("rank", subspace_rank))
    if model_type not in {"pair_lowrank_quadratic", "pair_lowrank_quadratic_strict"}:
        raise ValueError(f"Expected low-rank source probe, got {model_type}")

    checkpoint_path = Path(metrics.get("checkpoint") or (run_dir / cfg["output"].get("save_name", "best_probe.pt")))
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / cfg["output"].get("save_name", "best_probe.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    probe = build_probe(model_type=model_type, input_dim=input_dim, rank=source_rank)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    probe.load_state_dict(state_dict)

    if model_type == "pair_lowrank_quadratic":
        u = probe.u.detach().float()
        v = probe.v.detach().float()
        w = 0.5 * (u.transpose(0, 1) @ v + v.transpose(0, 1) @ u) / (input_dim ** 0.5)
    else:
        u = probe.u.detach().float()
        v = None
        sign = probe.sign.detach().float()
        w = (u.transpose(0, 1) @ torch.diag(sign) @ u) / (input_dim ** 0.5)

    eigvals, eigvecs = torch.linalg.eigh(w)
    order = torch.argsort(torch.abs(eigvals), descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if subspace_mode == "eig_topk":
        if subspace_rank <= 0 or subspace_rank >= eigvecs.shape[1]:
            raise ValueError(f"Invalid subspace_rank={subspace_rank} for input_dim={eigvecs.shape[1]}")
        subspace_basis = eigvecs[:, :subspace_rank].contiguous()
    elif subspace_mode == "uv_span":
        support = u.transpose(0, 1)
        if v is not None:
            support = torch.cat([support, v.transpose(0, 1)], dim=1)
        subspace_basis = _orthonormalize_columns(support)
    else:
        raise ValueError(f"Unsupported subspace_mode={subspace_mode!r}")

    nullspace_basis = _basis_complement(subspace_basis, input_dim)
    return subspace_basis, nullspace_basis, eigvals, cfg, metrics


def _project_tokens(tokens: np.ndarray, basis: torch.Tensor) -> np.ndarray:
    x = torch.from_numpy(tokens.astype(np.float32, copy=False))
    projected = x @ basis.float()
    return projected.cpu().numpy().astype(np.float32, copy=False)


def _write_projected_npz(source_npz: Path, target_npz: Path, basis: torch.Tensor) -> int:
    with np.load(source_npz, allow_pickle=False) as data:
        tokens = data["tokens"].astype(np.float32, copy=False)
        labels = data["is_dynamic"].astype(np.int64, copy=False)
        valid = data["valid_mask"].astype(np.bool_, copy=False)
        groups = data["group_ids"].astype(np.int64, copy=False)
    projected = _project_tokens(tokens, basis)
    _savez_compressed_atomic(
        target_npz,
        tokens=projected,
        is_dynamic=labels,
        valid_mask=valid,
        group_ids=groups,
        meta=np.array(json.dumps({"source_npz": str(source_npz), "basis_dim": int(basis.shape[1])})),
    )
    return int(projected.shape[1])


def _build_variant_config(
    source_cfg: dict,
    npz_by_split: dict[str, Path],
    input_dim: int,
    model_type: str,
    output_dir: Path,
    rank: int,
) -> dict:
    cfg = copy.deepcopy(source_cfg)
    cfg["dataset"]["train_npz"] = str(npz_by_split["train"])
    cfg["dataset"]["val_npz"] = str(npz_by_split["val"])
    test_items = {k: str(v) for k, v in npz_by_split.items() if k not in {"train", "val"}}
    cfg["dataset"]["test_npz"] = None
    cfg["dataset"]["test_npzs"] = test_items
    cfg["dataset"]["feature_key"] = "tokens"
    cfg["model"]["model_type"] = model_type
    cfg["model"]["input_dim"] = int(input_dim)
    cfg["model"]["rank"] = int(rank)
    cfg["model"]["use_value_features"] = False
    cfg["model"]["value_feature_matrix_path"] = None
    cfg["model"]["attention_matrix_path"] = None
    cfg["model"]["value_matrix_path"] = None
    cfg["output"]["out_dir"] = str(output_dir)
    return cfg


def _run_probe(model_name: str, config_path: Path, eval_only: bool) -> None:
    forwarded_args = ["--config", str(config_path)]
    if eval_only:
        forwarded_args.append("--eval_only")
    run_legacy_entry("run_isdynamic_probe.py", forwarded_args=forwarded_args, model_name=model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run null-space interference reprobes from a trained low-rank probe.")
    parser.add_argument("--model", required=True, help="Model label used by the shared IsDynamic entrypoints, e.g. pi3x or vggt.")
    parser.add_argument("--source_run_dir", required=True, help="Existing low-rank run directory containing config.json, metrics.json, and best_probe.pt.")
    parser.add_argument("--sharded_root", required=True, help="Feature shard root, e.g. .../youtubevos/tokens")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_splits", default="test_davis,test_segtrackv2,test_fbms59")
    parser.add_argument("--feature_key", default=None, help="Feature key to materialize. Defaults to source run leaf directory name.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--subspace_rank", type=int, default=8)
    parser.add_argument("--subspace_mode", choices=("eig_topk", "uv_span"), default="uv_span")
    parser.add_argument("--reprobe_model_type", default="pair_quadratic")
    parser.add_argument("--reprobe_rank", type=int, default=256)
    parser.add_argument("--variants", default="subspace,nullspace")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    sharded_root = Path(args.sharded_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else out_dir / "_cache"
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    allowed = {"subspace", "nullspace"}
    invalid = [v for v in variants if v not in allowed]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}; allowed={sorted(allowed)}")

    subspace_basis, nullspace_basis, eigvals, source_cfg, source_metrics = _extract_basis_from_lowrank_run(
        run_dir=source_run_dir,
        subspace_rank=args.subspace_rank,
        subspace_mode=args.subspace_mode,
    )
    inferred_feature_key = _infer_source_feature_key(source_run_dir, source_cfg)
    feature_key = inferred_feature_key
    if args.feature_key and args.feature_key != inferred_feature_key:
        print(
            "[warn] ignoring requested --feature_key="
            f"{args.feature_key!r}; source run {source_run_dir} was trained on {inferred_feature_key!r}"
        )
    elif args.feature_key:
        feature_key = args.feature_key

    print(
        f"[info] source={source_run_dir} feature_key={feature_key} subspace_mode={args.subspace_mode} "
        f"subspace_dim={int(subspace_basis.shape[1])} nullspace_dim={int(nullspace_basis.shape[1])}"
    )

    split_roots = {
        "train": _resolve_split_root(sharded_root, args.train_split),
        "val": _resolve_split_root(sharded_root, args.val_split),
    }
    for split_name in [s.strip() for s in args.test_splits.split(",") if s.strip()]:
        split_roots[split_name] = _resolve_split_root(sharded_root, split_name)

    source_flat_dir = cache_dir / "flat_source"
    print(f"[info] materializing source npz files under {source_flat_dir}")
    source_npzs = {
        split_name: materialize_feature_npz(
            sharded_root=split_root,
            cache_dir=source_flat_dir,
            feature_key=feature_key,
            split_name=split_name,
            rebuild=args.rebuild_cache,
        )
        for split_name, split_root in split_roots.items()
    }

    basis_by_variant = {
        "subspace": subspace_basis,
        "nullspace": nullspace_basis,
    }
    results: dict[str, dict] = {}

    for variant in variants:
        variant_out_dir = out_dir / variant
        metrics_path = variant_out_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[skip] variant={variant} existing metrics: {metrics_path}")
            results[variant] = {
                "metrics_path": str(metrics_path),
                "metrics": _load_json(metrics_path),
                "skipped_existing": True,
                "input_dim": int(basis_by_variant[variant].shape[1]),
            }
            continue

        projected_npzs: dict[str, Path] = {}
        projected_root = cache_dir / "projected" / variant
        input_dim = None
        print(f"[info] projecting variant={variant} into {projected_root}")
        for split_name, source_npz in source_npzs.items():
            target_npz = projected_root / split_name / f"{feature_key}.npz"
            input_dim = _write_projected_npz(source_npz, target_npz, basis_by_variant[variant])
            projected_npzs[split_name] = target_npz

        run_dir = variant_out_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = _build_variant_config(
            source_cfg=source_cfg,
            npz_by_split=projected_npzs,
            input_dim=int(input_dim),
            model_type=args.reprobe_model_type,
            output_dir=run_dir,
            rank=args.reprobe_rank,
        )
        cfg_path = run_dir / "config.json"
        _write_json(cfg_path, cfg)
        print(f"[info] running reprobe variant={variant} input_dim={int(input_dim)} config={cfg_path}")
        _run_probe(args.model, cfg_path, eval_only=args.eval_only)
        results[variant] = {
            "metrics_path": str(metrics_path),
            "metrics": _load_json(metrics_path),
            "input_dim": int(input_dim),
            "skipped_existing": False,
        }

    summary = {
        "model": args.model,
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": source_metrics.get("checkpoint"),
        "source_model_type": source_cfg.get("model", {}).get("model_type"),
        "source_rank": source_cfg.get("model", {}).get("rank"),
        "feature_key": feature_key,
        "sharded_root": str(sharded_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_splits": [s.strip() for s in args.test_splits.split(",") if s.strip()],
        "subspace_rank": int(args.subspace_rank),
        "subspace_mode": args.subspace_mode,
        "effective_subspace_dim": int(subspace_basis.shape[1]),
        "nullspace_dim": int(nullspace_basis.shape[1]),
        "reprobe_model_type": args.reprobe_model_type,
        "reprobe_rank": int(args.reprobe_rank),
        "top_eigenvalues": [float(x) for x in eigvals[: min(32, eigvals.numel())].tolist()],
        "source_best": source_metrics.get("best"),
        "source_best_tests": source_metrics.get("best_tests"),
        "variants": results,
    }
    _write_json(out_dir / "summary.json", summary)
    print(f"Saved null-space interference summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
