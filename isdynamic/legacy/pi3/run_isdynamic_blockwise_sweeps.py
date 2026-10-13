#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.models.pi3 import (  # noqa: E402
    MODEL_CHOICES,
    get_projection_matrix,
    load_pi3_family_model,
)
from isdynamic.probes.blockwise_sweep_common import (
    build_standard_blockwise_summary,
    build_standard_probe_cfg,
    load_json as _load_json,
    partition_existing_and_pending,
    save_json as _save_json,
    write_probe_config_and_run,
    resolve_sweep_results_dir,
    standard_run_result_record,
    standard_existing_result_record,
)


GROUP_PREFIX = {
    "inter_frame_block": "inter_frame_block_",
    "intra_frame_block": "intra_frame_block_",
    "image_encoder": "image_encoder_",
    "cam_head": "cam_head_",
    "conf_head": "conf_head_",
    "point_head": "point_head_",
}

FEATURE_KEY_ALIASES = {
    "inter_frame_block": ["inter_frame_block_", "glob_attn_"],
    "intra_frame_block": ["intra_frame_block_", "perframe_attn_"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blockwise IsDynamic probe sweeps from sequence-sharded token exports."
    )
    parser.add_argument("--base_config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "pi3" / "isdynamic_probe.json"))
    parser.add_argument("--sharded_root", type=str, default="data/isdynamic/davis/tokens")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test_davis,test_segtrackv2,test_fbms59", help=argparse.SUPPRESS)
    parser.add_argument(
        "--test_splits",
        type=str,
        default="test_davis,test_segtrackv2,test_fbms59",
        help="Comma-separated named test splits to evaluate separately.",
    )
    parser.add_argument("--eval_split", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache_dir", type=str, default="data/isdynamic/davis/flat_features")
    parser.add_argument(
        "--ephemeral_cache",
        action="store_true",
        help="Use a temporary flat-feature cache directory and delete it after the run.",
    )
    parser.add_argument("--results_dir", type=str, default="outputs/isdynamic_blockwise")
    parser.add_argument(
        "--groups",
        type=str,
        default="inter_frame_block,intra_frame_block,cam_head,conf_head,point_head",
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
            "Pi3/Pi3X checkpoint used to auto-extract fixed matrices for fixed-matrix probe types "
            "(pair_attention, pair_value, pair_attention_biased_quadratic, pair_value_biased_quadratic, "
            "pair_attention_biased_lowrank_quadratic, pair_value_biased_lowrank_quadratic, "
            "pair_attention_biased_lowrank_quadratic_strict, pair_value_biased_lowrank_quadratic_strict). "
            "Required when those model types are requested."
        ),
    )
    parser.add_argument(
        "--matrix_model",
        type=str,
        choices=MODEL_CHOICES,
        default="pi3x",
        help="Which model architecture to instantiate for matrix extraction.",
    )
    parser.add_argument(
        "--matrix_use_multimodal_pi3x",
        action="store_true",
        help="Enable Pi3X multimodal branch when matrix_model=pi3x.",
    )
    parser.add_argument(
        "--matrix_cache_dir",
        type=str,
        default="outputs/isdynamic_blockwise_matrices",
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
        help="Use projection matrix from current block or next block.",
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
    return [x.strip() for x in raw.split(",") if x.strip()]

def _scaled_eval_batch(base_eval_batch: int, input_dim: int, ref_dim: int = 2048) -> int:
    if input_dim <= 0:
        return max(1, int(base_eval_batch))
    scaled = int(round(float(base_eval_batch) * float(ref_dim) / float(input_dim)))
    return max(1, scaled)



def _parse_feature_key(feature_key: str) -> Tuple[str, int]:
    for group, prefixes in FEATURE_KEY_ALIASES.items():
        for prefix in prefixes:
            if feature_key.startswith(prefix):
                idx = int(feature_key[len(prefix) :])
                return group, idx
    for group, prefix in GROUP_PREFIX.items():
        if feature_key.startswith(prefix):
            idx = int(feature_key[len(prefix) :])
            return group, idx
    raise ValueError(f"Could not parse feature key group/index: {feature_key}")


def _canonical_feature_key(group_name: str, block_idx: int) -> str:
    prefix = GROUP_PREFIX.get(group_name)
    if prefix is None:
        raise ValueError(f"Unknown group for canonical feature key: {group_name}")
    return f"{prefix}{block_idx}"


def materialize_fixed_matrix(
    model,
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
    projection_block_idx = block_idx if projection == "current" else None
    matrix, _ = get_projection_matrix(
        model=model,
        group=group_name,
        feature_block_idx=block_idx,
        mode=mode,
        projection_block_idx=projection_block_idx,
        fallback_current_for_last=fallback_current_for_last,
    )
    np.save(out_path, matrix.detach().cpu().numpy())
    return out_path


def discover_feature_keys(sharded_root: Path, groups: List[str]) -> Dict[str, List[str]]:
    if not sharded_root.exists():
        raise FileNotFoundError(f"Sharded root not found: {sharded_root}")

    sequence_dirs = sorted([p for p in sharded_root.iterdir() if p.is_dir()])
    if not sequence_dirs:
        raise RuntimeError(f"No sequence directories found under: {sharded_root}")

    all_features = set()
    for seq_dir in sequence_dirs:
        for feat_dir in seq_dir.iterdir():
            if feat_dir.is_dir():
                all_features.add(feat_dir.name)

    grouped: Dict[str, List[str]] = {g: [] for g in groups}
    for group in groups:
        prefixes = FEATURE_KEY_ALIASES.get(group, [GROUP_PREFIX[group]])
        keys = []
        for prefix in prefixes:
            keys.extend([k for k in all_features if k.startswith(prefix)])
        keys = sorted(set(keys), key=lambda x: int(x.split("_")[-1]))
        grouped[group] = keys

    return grouped


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
    if out_path.exists() and not rebuild:
        return out_path

    sequence_dirs = sorted([p for p in sharded_root.iterdir() if p.is_dir()])
    raw_parts = []
    dim_token_count: Dict[int, int] = {}

    for seq_dir in sequence_dirs:
        feat_dir = seq_dir / feature_key
        if not feat_dir.exists() or not feat_dir.is_dir():
            continue
        part_files = sorted(feat_dir.glob("part_*.npz"))
        for part_path in part_files:
            with np.load(part_path, allow_pickle=False) as part:
                tokens = part["tokens"].astype(np.float32, copy=False)
                labels = part["is_dynamic"].astype(np.int64, copy=False)
                valid = part["valid_mask"].astype(np.bool_, copy=False)
                groups = part["group_ids"].astype(np.int64, copy=False)

            if tokens.ndim != 2:
                print(f"[warn] skipping malformed part (tokens.ndim={tokens.ndim}): {part_path}")
                continue
            n = int(tokens.shape[0])
            if labels.shape[0] != n or valid.shape[0] != n or groups.shape[0] != n:
                print(f"[warn] skipping malformed part (length mismatch): {part_path}")
                continue

            d = int(tokens.shape[1])
            dim_token_count[d] = dim_token_count.get(d, 0) + n
            raw_parts.append((d, part_path, tokens, labels, valid, groups))

    if not raw_parts:
        raise RuntimeError(f"No valid part files found for feature '{feature_key}' in {sharded_root}")

    target_dim = sorted(dim_token_count.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    token_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []
    skipped = []
    for d, part_path, tokens, labels, valid, groups in raw_parts:
        if d != target_dim:
            skipped.append((part_path, d))
            continue
        token_parts.append(tokens)
        label_parts.append(labels)
        valid_parts.append(valid)
        group_parts.append(groups)

    if skipped:
        preview = ", ".join([f"{p} (d={d})" for p, d in skipped[:5]])
        print(
            "[warn] feature '" + feature_key + "' split='" + split_name + "' has mixed token dims "
            + f"{sorted(dim_token_count.keys())}; keeping d={target_dim}, skipped {len(skipped)} part(s). "
            + (f"Examples: {preview}" if preview else "")
        )

    tokens = np.concatenate(token_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    valid = np.concatenate(valid_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)

    np.savez_compressed(
        out_path,
        tokens=tokens,
        is_dynamic=labels,
        valid_mask=valid,
        group_ids=groups,
        meta=np.array(json.dumps({"feature_key": feature_key, "source_root": str(sharded_root)})),
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

        sharded_root = Path(args.sharded_root)
        val_split = args.eval_split if args.eval_split else args.val_split
        test_splits = _parse_csv(args.test_splits)
        if not test_splits:
            raise ValueError("--test_splits resolved to empty list")
        train_root = sharded_root / args.train_split
        val_root = sharded_root / val_split
        test_roots = {}
        if not train_root.exists():
            raise FileNotFoundError(f"Train split root not found: {train_root}")
        if not val_root.exists():
            raise FileNotFoundError(f"Val split root not found: {val_root}")
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

        results_dir = resolve_sweep_results_dir(args.results_dir, base_cfg.get("train"), repo_root=repo_root)
        groups = _parse_csv(args.groups)
        model_types = _parse_csv(args.model_types)

        for g in groups:
            if g not in GROUP_PREFIX:
                raise ValueError(f"Unknown group '{g}'. Allowed: {sorted(GROUP_PREFIX.keys())}")
        if not model_types:
            raise ValueError("--model_types resolved to empty list")

        fixed_matrix_model_types = [m for m in model_types if m in FIXED_MATRIX_MODEL_TYPES]
        matrix_model = None
        matrix_cache_dir = Path(args.matrix_cache_dir)
        if not matrix_cache_dir.is_absolute():
            matrix_cache_dir = (repo_root / matrix_cache_dir).resolve()
        if fixed_matrix_model_types:
            if args.ephemeral_matrix_cache:
                tmp_matrix_cache_ctx = tempfile.TemporaryDirectory(prefix="isdynamic_matrix_cache_")
                matrix_cache_dir = Path(tmp_matrix_cache_ctx.name)
                print(f"[info] using ephemeral matrix cache dir: {matrix_cache_dir}")
            if args.matrix_checkpoint is None:
                raise ValueError(
                    f"model_types includes {fixed_matrix_model_types} but --matrix_checkpoint was not provided."
                )
            matrix_model = load_pi3_family_model(
                model_name=args.matrix_model,
                checkpoint=Path(args.matrix_checkpoint),
                device=torch.device("cpu"),
                use_multimodal_pi3x=args.matrix_use_multimodal_pi3x,
            )

        grouped_keys = discover_feature_keys(sharded_root=train_root, groups=groups)

        results: List[Dict[str, object]] = []

        for group in groups:
            keys = grouped_keys.get(group, [])
            if not keys:
                print(f"[warn] no keys found for group={group}")
                continue

            for source_feature_key in keys:
                group_name, block_idx = _parse_feature_key(source_feature_key)
                feature_key = _canonical_feature_key(group_name, block_idx)
                existing_rows, pending_model_types = partition_existing_and_pending(
                    model_types=model_types,
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
                    results.extend(existing_rows)
                    continue

                train_feature_npz = materialize_feature_npz(
                    sharded_root=train_root,
                    cache_dir=cache_dir,
                    feature_key=source_feature_key,
                    split_name=args.train_split,
                    rebuild=args.rebuild_cache,
                )
                eval_feature_npz = materialize_feature_npz(
                    sharded_root=val_root,
                    cache_dir=cache_dir,
                    feature_key=source_feature_key,
                    split_name=val_split,
                    rebuild=args.rebuild_cache,
                )
                test_feature_npzs = {
                    test_split: materialize_feature_npz(
                        sharded_root=test_root,
                        cache_dir=cache_dir,
                        feature_key=source_feature_key,
                        split_name=test_split,
                        rebuild=args.rebuild_cache,
                    )
                    for test_split, test_root in test_roots.items()
                }

                with np.load(train_feature_npz, allow_pickle=False) as d:
                    input_dim = int(d["tokens"].shape[1])

                results.extend(existing_rows)

                for model_type in pending_model_types:
                    run_dir = results_dir / model_type / group_name / feature_key
                    run_dir.mkdir(parents=True, exist_ok=True)

                    base_eval_batch = int(base_cfg.get("train", {}).get("eval_batch_size", 262144))
                    cfg = build_standard_probe_cfg(
                        base_cfg=base_cfg,
                        train_npz=str(train_feature_npz),
                        val_npz=str(eval_feature_npz),
                        test_npzs={test_name: str(path) for test_name, path in test_feature_npzs.items()},
                        model_type=model_type,
                        input_dim=input_dim,
                        rank=int(args.rank),
                        out_dir=str(run_dir),
                        eval_batch_size=_scaled_eval_batch(base_eval_batch, input_dim),
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
                        if matrix_model is None:
                            raise RuntimeError("Internal error: matrix_model is not initialized.")
                        mode = _matrix_mode_for_model_type(model_type)
                        matrix_path = materialize_fixed_matrix(
                            model=matrix_model,
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
                            print(
                                f"[warn] skip fixed-matrix run for {feature_key} ({model_type}) due to dim mismatch: "
                                f"matrix_dim={matrix_dim}, input_dim={input_dim}, path={matrix_path}"
                            )
                            continue
                        matrix_path = matrix_path.resolve()
                        if mode == "attention":
                            cfg["model"]["attention_matrix_path"] = str(matrix_path)
                        else:
                            cfg["model"]["value_matrix_path"] = str(matrix_path)
                        cfg["model"]["use_value_features"] = False

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
                    score = float(best.get(args.metric, float("nan")))
                    print(
                        f"[result] group={group_name} block={block_idx} feature={feature_key} "
                        f"model={model_type} {args.metric}={score:.6f}"
                    )

                    results.append(result_row)

        summary = build_standard_blockwise_summary(
            metric_primary=args.metric,
            rank=int(args.rank),
            groups=groups,
            model_types=model_types,
            sharded_root=str(sharded_root),
            train_split=args.train_split,
            val_split=val_split,
            test_splits=test_splits,
            cache_dir="__ephemeral__" if args.ephemeral_cache else str(cache_dir),
            ephemeral_cache=bool(args.ephemeral_cache),
            skip_existing=bool(args.skip_existing),
            results=results,
            extra_fields={
                "test_split": args.test_split,
                "matrix_checkpoint": args.matrix_checkpoint,
                "matrix_model": args.matrix_model if fixed_matrix_model_types else None,
                "matrix_cache_dir": (
                    "__ephemeral__"
                    if fixed_matrix_model_types and args.ephemeral_matrix_cache
                    else (str(matrix_cache_dir) if fixed_matrix_model_types else None)
                ),
                "ephemeral_matrix_cache": bool(args.ephemeral_matrix_cache),
                "matrix_projection": args.matrix_projection,
                "matrix_fallback_current_for_last": bool(args.matrix_fallback_current_for_last),
            },
        )
        summary_path = results_dir / "summary.json"
        _save_json(summary_path, summary)
        print(f"Saved summary: {summary_path}")
    finally:
        if tmp_matrix_cache_ctx is not None:
            tmp_matrix_cache_ctx.cleanup()
        if tmp_cache_ctx is not None:
            tmp_cache_ctx.cleanup()


if __name__ == "__main__":
    main()
