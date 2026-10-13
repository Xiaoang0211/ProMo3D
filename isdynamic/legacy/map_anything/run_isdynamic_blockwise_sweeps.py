#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.models.map_anything import load_mapanything
from isdynamic.core.artifacts import artifact_path
from isdynamic.probes.blockwise_sweep_common import (
    build_standard_blockwise_summary,
    build_standard_probe_cfg,
    load_json as _load_json,
    partition_existing_and_pending,
    save_json as _save_json,
    standard_run_result_record,
    standard_existing_result_record,
    write_probe_config_and_run,
    resolve_sweep_results_dir,
)


GROUP_PREFIX = {
    "inter_frame_block": "inter_frame_block_",
    "intra_frame_block": "intra_frame_block_",
    "dense_head": "dense_head_",
    "pose_head": "pose_head_",
    "scale_head": "scale_head_",
}

GROUP_SOURCE_PREFIXES = {
    "inter_frame_block": ["inter_frame_block_", "glob_attn_"],
    "intra_frame_block": ["intra_frame_block_", "perframe_attn_"],
    "dense_head": ["dense_head_"],
    "pose_head": ["pose_head_"],
    "scale_head": ["scale_head_"],
}

FIXED_MATRIX_MODEL_TYPES = {
    "pair_attention",
    "pair_value",
    "pair_attention_biased_quadratic",
    "pair_value_biased_quadratic",
    "pair_attention_biased_lowrank_quadratic",
    "pair_value_biased_lowrank_quadratic",
    "pair_attention_biased_lowrank_quadratic_strict",
    "pair_value_biased_lowrank_quadratic_strict",
}
MATRIX_SUPPORTED_GROUPS = {"inter_frame_block", "intra_frame_block"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blockwise IsDynamic probe sweeps from sequence-sharded MapAnything token exports."
    )
    parser.add_argument("--base_config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "map_anything" / "isdynamic_probe.json"))
    parser.add_argument(
        "--sharded_root",
        type=str,
        default=artifact_path("data", "isdynamic", "youtubevos", "tokens", model_name="mapanything"),
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--test_splits",
        type=str,
        default="test_davis,test_segtrackv2,test_fbms59",
        help="Comma-separated named test splits to evaluate separately.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=artifact_path("data", "isdynamic", "flat_features", model_name="mapanything"),
    )
    parser.add_argument(
        "--ephemeral_cache",
        action="store_true",
        help="Use a temporary flat-feature cache directory and delete it after the run.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=artifact_path("outputs", "isdynamic_blockwise_mapanything", model_name="mapanything"),
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="inter_frame_block,intra_frame_block,dense_head,pose_head,scale_head",
        help="Comma-separated group names to sweep.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="pair_linear,pair_diag_quadratic,pair_quadratic",
        help="Comma-separated probe model types.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=256,
        help="Low-rank dimension k for low-rank probe variants.",
    )
    parser.add_argument(
        "--matrix_checkpoint",
        type=str,
        default=None,
        help=(
            "MapAnything checkpoint/dir used to auto-extract fixed matrices for fixed-matrix probe types. "
            "Required when those model types are requested."
        ),
    )
    parser.add_argument(
        "--matrix_cache_dir",
        type=str,
        default=artifact_path("outputs", "isdynamic_blockwise_matrices", model_name="mapanything"),
        help="Directory to cache extracted fixed matrices per feature block.",
    )
    parser.add_argument(
        "--ephemeral_matrix_cache",
        action="store_true",
        help="Use a temporary matrix cache directory and delete it after the run.",
    )
    parser.add_argument(
        "--matrix_projection",
        type=str,
        default="next",
        choices=["next", "current"],
        help="Use projection matrix from current block or next same-group block.",
    )
    parser.add_argument(
        "--matrix_fallback_current_for_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If next-block projection is out of range for last block, fallback to current block.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
        help="Primary metric to print while running.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a run if its metrics.json already exists.",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild flat feature NPZ cache even if cached files already exist.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate existing checkpoints only (no training).",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_probe.pt",
        help="Checkpoint filename under each run directory for --eval_only.",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _canonical_feature_key(feature_key: str) -> str:
    for group_name, prefixes in GROUP_SOURCE_PREFIXES.items():
        canonical_prefix = GROUP_PREFIX[group_name]
        for prefix in prefixes:
            if feature_key.startswith(prefix):
                return canonical_prefix + feature_key[len(prefix) :]
    return feature_key

def _scaled_eval_batch(base_eval_batch: int, input_dim: int, ref_dim: int = 2048) -> int:
    if input_dim <= 0:
        return max(1, int(base_eval_batch))
    scaled = int(round(float(base_eval_batch) * float(ref_dim) / float(input_dim)))
    return max(1, scaled)


def _matrix_mode_for_model_type(model_type: str) -> str:
    if model_type in {
        "pair_attention",
        "pair_attention_biased_quadratic",
        "pair_attention_biased_lowrank_quadratic",
        "pair_attention_biased_lowrank_quadratic_strict",
    }:
        return "attention"
    if model_type in {
        "pair_value",
        "pair_value_biased_quadratic",
        "pair_value_biased_lowrank_quadratic",
        "pair_value_biased_lowrank_quadratic_strict",
    }:
        return "value"
    raise ValueError(f"Unsupported fixed-matrix model_type: {model_type}")


def _resolve_projection_block_idx(
    feature_block_idx: int,
    projection_block_idx: int | None,
    depth: int,
    fallback_current_for_last: bool,
) -> int:
    idx = feature_block_idx + 1 if projection_block_idx is None else projection_block_idx
    if idx >= depth:
        if fallback_current_for_last:
            idx = feature_block_idx
        else:
            raise ValueError(
                f"Projection block index {idx} is out of range for depth {depth}. "
                "Use --fallback_current_for_last or provide --projection_block_idx."
            )
    if idx < 0:
        raise ValueError(f"Invalid projection block index: {idx}")
    return idx


def _extract_matrix_from_block(block: torch.nn.Module, mode: str) -> torch.Tensor:
    attn = getattr(block, "attn", None)
    if attn is None or not hasattr(attn, "qkv"):
        raise ValueError(f"Block {type(block).__name__} does not expose attn.qkv.")
    qkv = attn.qkv.weight.detach().float()
    c_dim = qkv.shape[1]
    if qkv.shape[0] != 3 * c_dim:
        raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")
    wq, wk, wv = qkv.chunk(3, dim=0)
    if mode == "attention":
        return wq.transpose(0, 1) @ wk
    if mode == "value":
        return wv
    raise ValueError(f"Unsupported mode: {mode}")


def _matrix_for_feature(
    model: torch.nn.Module,
    group_name: str,
    block_idx: int,
    mode: str,
    projection: str,
    fallback_current_for_last: bool,
) -> torch.Tensor:
    if group_name not in MATRIX_SUPPORTED_GROUPS:
        raise ValueError(
            f"Fixed matrices are only supported for groups {sorted(MATRIX_SUPPORTED_GROUPS)}, got {group_name}"
        )

    all_blocks = list(model.info_sharing.self_attention_blocks)
    actual_block_indices = list(range(0, len(all_blocks), 2)) if group_name == "inter_frame_block" else list(
        range(1, len(all_blocks), 2)
    )
    blocks = [all_blocks[idx] for idx in actual_block_indices]
    projection_block_idx = block_idx if projection == "current" else None
    proj_idx = _resolve_projection_block_idx(
        feature_block_idx=block_idx,
        projection_block_idx=projection_block_idx,
        depth=len(blocks),
        fallback_current_for_last=fallback_current_for_last,
    )
    return _extract_matrix_from_block(blocks[proj_idx], mode=mode)


def materialize_fixed_matrix(
    model: torch.nn.Module,
    matrix_cache_dir: Path,
    feature_key: str,
    group_name: str,
    block_idx: int,
    mode: str,
    projection: str,
    fallback_current_for_last: bool,
) -> Path:
    out_path = matrix_cache_dir / mode / projection / group_name / f"{feature_key}.npy"
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _matrix_for_feature(
        model=model,
        group_name=group_name,
        block_idx=block_idx,
        mode=mode,
        projection=projection,
        fallback_current_for_last=fallback_current_for_last,
    )
    np.save(out_path, matrix.cpu().numpy())
    return out_path



def _parse_feature_key(feature_key: str) -> Tuple[str, int]:
    canonical_key = _canonical_feature_key(feature_key)
    for group_name, prefix in GROUP_PREFIX.items():
        if canonical_key.startswith(prefix):
            return group_name, int(canonical_key[len(prefix) :])
    raise ValueError(f"Could not parse feature key group/index: {feature_key}")


def discover_feature_keys(sharded_root: Path, groups: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if not sharded_root.exists():
        raise FileNotFoundError(f"Sharded root not found: {sharded_root}")

    sequence_dirs = sorted(path for path in sharded_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise RuntimeError(f"No sequence directories found under: {sharded_root}")

    all_features = set()
    for sequence_dir in sequence_dirs:
        for feat_dir in sequence_dir.iterdir():
            if feat_dir.is_dir():
                all_features.add(feat_dir.name)

    grouped: Dict[str, List[str]] = {group: [] for group in groups}
    source_key_by_canonical: Dict[str, str] = {}
    for group in groups:
        canonical_prefix = GROUP_PREFIX[group]
        candidates: Dict[str, str] = {}
        for actual_key in all_features:
            canonical_key = _canonical_feature_key(actual_key)
            if not canonical_key.startswith(canonical_prefix):
                continue
            candidates.setdefault(canonical_key, actual_key)
        keys = sorted(candidates.keys(), key=lambda item: int(item[len(canonical_prefix) :]))
        grouped[group] = keys
        source_key_by_canonical.update(candidates)
    return grouped, source_key_by_canonical


def materialize_feature_npz(
    sharded_root: Path,
    cache_dir: Path,
    feature_key: str,
    split_name: str,
    rebuild: bool,
    source_feature_key: str | None = None,
) -> Path:
    split_cache_dir = cache_dir / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_cache_dir / f"{feature_key}.npz"
    if out_path.exists() and not rebuild:
        return out_path

    sequence_dirs = sorted(path for path in sharded_root.iterdir() if path.is_dir())
    token_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []

    for sequence_dir in sequence_dirs:
        feat_dir = sequence_dir / (source_feature_key or feature_key)
        if not feat_dir.exists() or not feat_dir.is_dir():
            continue
        for part_path in sorted(feat_dir.glob("part_*.npz")):
            try:
                with np.load(part_path, allow_pickle=False) as part:
                    token_parts.append(part["tokens"].astype(np.float32, copy=False))
                    label_parts.append(part["is_dynamic"].astype(np.int64, copy=False))
                    valid_parts.append(part["valid_mask"].astype(np.bool_, copy=False))
                    group_parts.append(part["group_ids"].astype(np.int64, copy=False))
            except (EOFError, OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
                print(f"[warn] skipping unreadable part: {part_path} ({exc})")
                continue

    if not token_parts:
        raise RuntimeError(f"No part files found for feature '{feature_key}' in {sharded_root}")

    np.savez_compressed(
        out_path,
        tokens=np.concatenate(token_parts, axis=0),
        is_dynamic=np.concatenate(label_parts, axis=0),
        valid_mask=np.concatenate(valid_parts, axis=0),
        group_ids=np.concatenate(group_parts, axis=0),
        meta=np.array(
            json.dumps(
                {
                    "feature_key": feature_key,
                    "source_feature_key": source_feature_key or feature_key,
                    "source_root": str(sharded_root),
                }
            )
        ),
    )
    return out_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    tmp_cache_ctx = None
    tmp_matrix_cache_ctx = None

    try:
        base_cfg_path = Path(args.base_config)
        if not base_cfg_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
        base_cfg = _load_json(base_cfg_path)

        groups = _parse_csv(args.groups)
        model_types = _parse_csv(args.model_types)
        test_splits = _parse_csv(args.test_splits)

        for group in groups:
            if group not in GROUP_PREFIX:
                raise ValueError(f"Unknown group '{group}'. Allowed: {sorted(GROUP_PREFIX.keys())}")
        fixed_matrix_model_types = [
            model_type for model_type in model_types if model_type in FIXED_MATRIX_MODEL_TYPES
        ]
        if fixed_matrix_model_types and not args.matrix_checkpoint:
            raise ValueError(
                "Fixed-matrix probe types requested but --matrix_checkpoint was not provided: "
                + ", ".join(sorted(fixed_matrix_model_types))
            )

        sharded_root = Path(args.sharded_root)
        train_root = sharded_root / args.train_split
        val_root = sharded_root / args.val_split
        if not train_root.exists():
            raise FileNotFoundError(f"Train split root not found: {train_root}")
        if not val_root.exists():
            raise FileNotFoundError(f"Val split root not found: {val_root}")

        test_roots = {}
        for test_split in test_splits:
            test_root = sharded_root / test_split
            if not test_root.exists():
                raise FileNotFoundError(f"Test split root not found: {test_root}")
            test_roots[test_split] = test_root

        if args.ephemeral_cache:
            tmp_cache_ctx = tempfile.TemporaryDirectory(prefix="isdynamic_flat_cache_")
            cache_dir = Path(tmp_cache_ctx.name)
            print(f"[info] using ephemeral cache dir: {cache_dir}")
        else:
            cache_dir = Path(args.cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = (repo_root / cache_dir).resolve()

        fixed_matrix_model = None
        if fixed_matrix_model_types:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fixed_matrix_model = load_mapanything(
                checkpoint_path=args.matrix_checkpoint,
                device=device,
            )
            if args.ephemeral_matrix_cache:
                tmp_matrix_cache_ctx = tempfile.TemporaryDirectory(prefix="isdynamic_matrix_cache_")
                matrix_cache_dir = Path(tmp_matrix_cache_ctx.name)
                print(f"[info] using ephemeral matrix cache dir: {matrix_cache_dir}")
            else:
                matrix_cache_dir = Path(args.matrix_cache_dir)
                if not matrix_cache_dir.is_absolute():
                    matrix_cache_dir = (repo_root / matrix_cache_dir).resolve()

        results_dir = resolve_sweep_results_dir(args.results_dir, base_cfg.get("train"))
        grouped_keys, source_key_by_canonical = discover_feature_keys(train_root, groups)
        results: List[Dict[str, object]] = []

        for group in groups:
            feature_keys = grouped_keys.get(group, [])
            if not feature_keys:
                print(f"[warn] no keys found for group={group}")
                continue

            for feature_key in feature_keys:
                source_feature_key = source_key_by_canonical.get(feature_key, feature_key)
                group_name, block_idx = _parse_feature_key(feature_key)
                eligible_model_types = [
                    model_type
                    for model_type in model_types
                    if model_type not in FIXED_MATRIX_MODEL_TYPES or group_name in MATRIX_SUPPORTED_GROUPS
                ]
                existing_records, pending_model_types = partition_existing_and_pending(
                    model_types=eligible_model_types,
                    skip_existing=bool(args.skip_existing),
                    existing_record_for_model_type=lambda model_type: standard_existing_result_record(
                        group_name=group_name,
                        block_idx=block_idx,
                        feature_key=feature_key,
                        model_type=model_type,
                        metric_name=args.metric,
                        run_dir=results_dir / model_type / group_name / feature_key,
                        metrics_path=results_dir / model_type / group_name / feature_key / "metrics.json",
                    ),
                )
                if not pending_model_types:
                    print(
                        f"[skip feature] group={group_name} block={block_idx} feature={feature_key} "
                        f"all requested model_types already have metrics.json; skipping feature materialization"
                    )
                    results.extend(existing_records)
                    continue

                train_feature_npz = materialize_feature_npz(
                    sharded_root=train_root,
                    cache_dir=cache_dir,
                    feature_key=feature_key,
                    split_name=args.train_split,
                    rebuild=args.rebuild_cache,
                    source_feature_key=source_feature_key,
                )
                val_feature_npz = materialize_feature_npz(
                    sharded_root=val_root,
                    cache_dir=cache_dir,
                    feature_key=feature_key,
                    split_name=args.val_split,
                    rebuild=args.rebuild_cache,
                    source_feature_key=source_feature_key,
                )
                test_feature_npzs = {
                    test_split: str(
                        materialize_feature_npz(
                            sharded_root=test_root,
                            cache_dir=cache_dir,
                            feature_key=feature_key,
                            split_name=test_split,
                            rebuild=args.rebuild_cache,
                            source_feature_key=source_feature_key,
                        )
                    )
                    for test_split, test_root in test_roots.items()
                }

                with np.load(train_feature_npz, allow_pickle=False) as train_npz:
                    input_dim = int(train_npz["tokens"].shape[1])

                results.extend(existing_records)

                for model_type in pending_model_types:
                    run_dir = results_dir / model_type / group_name / feature_key
                    run_dir.mkdir(parents=True, exist_ok=True)

                    base_eval_batch = int(base_cfg.get("train", {}).get("eval_batch_size", 262144))
                    cfg = build_standard_probe_cfg(
                        base_cfg=base_cfg,
                        train_npz=str(train_feature_npz),
                        val_npz=str(val_feature_npz),
                        test_npzs=test_feature_npzs,
                        model_type=model_type,
                        input_dim=input_dim,
                        rank=int(args.rank),
                        out_dir=str(run_dir),
                        eval_batch_size=_scaled_eval_batch(base_eval_batch, input_dim),
                        include_matrix_keys=False,
                    )

                    effective_run_dir = Path(cfg["output"]["out_dir"])
                    checkpoint_path = effective_run_dir / args.checkpoint_name
                    if args.eval_only and not checkpoint_path.exists():
                        print(
                            f"[skip] missing checkpoint for eval-only: {checkpoint_path} "
                            f"(group={group_name} feature={feature_key} model={model_type})"
                        )
                        continue

                    if model_type in FIXED_MATRIX_MODEL_TYPES:
                        mode = _matrix_mode_for_model_type(model_type)
                        matrix_path = materialize_fixed_matrix(
                            model=fixed_matrix_model,
                            matrix_cache_dir=matrix_cache_dir,
                            feature_key=feature_key,
                            group_name=group_name,
                            block_idx=block_idx,
                            mode=mode,
                            projection=args.matrix_projection,
                            fallback_current_for_last=args.matrix_fallback_current_for_last,
                        )
                        matrix_dim = int(np.load(matrix_path, mmap_mode="r").shape[0])
                        if matrix_dim != input_dim:
                            raise ValueError(
                                f"Matrix/feature dim mismatch for {feature_key} ({model_type}): "
                                f"matrix_dim={matrix_dim}, input_dim={input_dim}, path={matrix_path}"
                            )
                        if mode == "attention":
                            cfg["model"]["attention_matrix_path"] = str(matrix_path)
                            cfg["model"]["value_matrix_path"] = None
                        else:
                            cfg["model"]["value_matrix_path"] = str(matrix_path)
                            cfg["model"]["attention_matrix_path"] = None
                    else:
                        cfg["model"]["attention_matrix_path"] = None
                        cfg["model"]["value_matrix_path"] = None

                    cfg_path = run_dir / "config.json"
                    probe_script = Path(__file__).resolve().with_name("run_isdynamic_probe.py")
                    write_probe_config_and_run(
                        cfg_path=cfg_path,
                        cfg=cfg,
                        probe_script_path=probe_script,
                        python_executable=sys.executable,
                        cwd=repo_root,
                        eval_only=bool(args.eval_only),
                        checkpoint_path=checkpoint_path,
                    )

                    metrics_path = effective_run_dir / "metrics.json"
                    result_row = standard_run_result_record(
                        group_name=group_name,
                        block_idx=block_idx,
                        feature_key=feature_key,
                        model_type=model_type,
                        metric_name=args.metric,
                        run_dir=effective_run_dir,
                        metrics_path=metrics_path,
                    )
                    best = result_row["best"]
                    best_tests = result_row["best_tests"]
                    score = float(best.get(args.metric, float("nan")))
                    print(
                        f"[result] group={group_name} block={block_idx} feature={feature_key} "
                        f"model={model_type} {args.metric}={score:.6f}"
                    )
                    for test_name, test_metrics in sorted(best_tests.items()):
                        test_score = float(test_metrics.get(args.metric, float("nan")))
                        print(f"         test[{test_name}] {args.metric}={test_score:.6f}")

                    results.append(result_row)

        summary = build_standard_blockwise_summary(
            metric_primary=args.metric,
            rank=int(args.rank),
            groups=groups,
            model_types=model_types,
            sharded_root=str(sharded_root),
            train_split=args.train_split,
            val_split=args.val_split,
            test_splits=test_splits,
            cache_dir="__ephemeral__" if args.ephemeral_cache else str(cache_dir),
            ephemeral_cache=bool(args.ephemeral_cache),
            skip_existing=bool(args.skip_existing),
            results=results,
            extra_fields={
                "matrix_cache_dir": (
                    "__ephemeral__"
                    if fixed_matrix_model_types and args.ephemeral_matrix_cache
                    else str(matrix_cache_dir)
                    if fixed_matrix_model_types
                    else None
                ),
            },
        )
        summary_path = results_dir / "summary.json"
        _save_json(summary_path, summary)
        print(f"Saved summary: {summary_path}")
    finally:
        if tmp_cache_ctx is not None:
            tmp_cache_ctx.cleanup()
        if tmp_matrix_cache_ctx is not None:
            tmp_matrix_cache_ctx.cleanup()


if __name__ == "__main__":
    main()
