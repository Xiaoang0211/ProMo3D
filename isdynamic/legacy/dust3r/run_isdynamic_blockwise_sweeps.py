#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.probes.blockwise_sweep_common import (
    build_standard_probe_cfg,
    build_best_val_summary,
    best_val_existing_result_record,
    best_val_run_result_record,
    load_json as _load_json,
    merge_best_val_summary_rows,
    partition_existing_and_pending,
    save_json as _save_json,
    write_probe_config_and_run,
    resolve_sweep_results_dir,
)


GROUP_PREFIX = {
    "enc": "enc_",
    "dec_view1": "dec_view1_",
    "dec_view2": "dec_view2_",
    "head_in_view1": "head_in_view1_",
    "head_in_view2": "head_in_view2_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blockwise IsDynamic probe sweeps from sequence-sharded Dust3R token exports."
    )
    parser.add_argument("--base_config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "dust3r" / "isdynamic_probe.json"))
    parser.add_argument("--sharded_root", type=str, default="data/isdynamic/youtubevos/tokens")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--test_splits",
        type=str,
        default="test_davis,test_segtrackv2,test_fbms59",
        help="Comma-separated named test splits to evaluate separately.",
    )
    parser.add_argument("--cache_dir", type=str, default="data/isdynamic/youtubevos/flat_features")
    parser.add_argument(
        "--ephemeral_cache",
        action="store_true",
        help="Use a temporary flat-feature cache directory and delete it after the run.",
    )
    parser.add_argument("--results_dir", type=str, default="outputs/isdynamic_blockwise")
    parser.add_argument(
        "--groups",
        type=str,
        default="enc,dec_view1,dec_view2,head_in_view1,head_in_view2",
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
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall", "loss"],
        help="Primary metric to report in the summary.",
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


def _suffix_sort_key(key: str, prefix: str):
    suffix = key[len(prefix) :]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def discover_feature_keys(sharded_root: Path, groups: List[str]) -> Dict[str, List[str]]:
    if not sharded_root.exists():
        raise FileNotFoundError(f"Sharded root not found: {sharded_root}")

    sequence_dirs = sorted(path for path in sharded_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise RuntimeError(f"No sequence directories found under: {sharded_root}")

    all_features = set()
    for seq_dir in sequence_dirs:
        for feat_dir in seq_dir.iterdir():
            if feat_dir.is_dir():
                all_features.add(feat_dir.name)

    grouped: Dict[str, List[str]] = {group: [] for group in groups}
    for group in groups:
        prefix = GROUP_PREFIX[group]
        keys = [key for key in all_features if key.startswith(prefix)]
        keys.sort(key=lambda key: _suffix_sort_key(key, prefix))
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

    sequence_dirs = sorted(path for path in sharded_root.iterdir() if path.is_dir())
    token_parts = []
    label_parts = []
    valid_parts = []
    group_parts = []

    for seq_dir in sequence_dirs:
        feat_dir = seq_dir / feature_key
        if not feat_dir.exists() or not feat_dir.is_dir():
            continue
        for part_path in sorted(feat_dir.glob("part_*.npz")):
            with np.load(part_path, allow_pickle=False) as part:
                token_parts.append(part["tokens"].astype(np.float32, copy=False))
                label_parts.append(part["is_dynamic"].astype(np.int64, copy=False))
                valid_parts.append(part["valid_mask"].astype(np.bool_, copy=False))
                group_parts.append(part["group_ids"].astype(np.int64, copy=False))

    if not token_parts:
        raise RuntimeError(f"No part files found for feature '{feature_key}' in {sharded_root}")

    np.savez_compressed(
        out_path,
        tokens=np.concatenate(token_parts, axis=0),
        is_dynamic=np.concatenate(label_parts, axis=0),
        valid_mask=np.concatenate(valid_parts, axis=0),
        group_ids=np.concatenate(group_parts, axis=0),
        meta=np.array(json.dumps({"feature_key": feature_key, "source_root": str(sharded_root)})),
    )
    return out_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    tmp_cache_ctx = None

    try:
        base_cfg_path = Path(args.base_config)
        if not base_cfg_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
        base_cfg = _load_json(base_cfg_path)

        sharded_root = Path(args.sharded_root)
        train_root = sharded_root / args.train_split
        val_root = sharded_root / args.val_split
        test_splits = _parse_csv(args.test_splits)
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

        results_dir = resolve_sweep_results_dir(args.results_dir, base_cfg.get("train"))
        groups = _parse_csv(args.groups)
        model_types = _parse_csv(args.model_types)
        for group in groups:
            if group not in GROUP_PREFIX:
                raise ValueError(f"Unknown group '{group}'. Allowed: {sorted(GROUP_PREFIX.keys())}")
        if not model_types:
            raise ValueError("--model_types resolved to empty list")

        grouped_keys = discover_feature_keys(train_root, groups)
        summary_rows = []

        for group in groups:
            feature_keys = grouped_keys.get(group, [])
            if not feature_keys:
                print(f"[warn] no keys found for group={group}")
                continue

            for feature_key in feature_keys:
                existing_rows, pending_model_types = partition_existing_and_pending(
                    model_types=model_types,
                    skip_existing=bool(args.skip_existing),
                    existing_record_for_model_type=lambda model_type: best_val_existing_result_record(
                        group_name=group,
                        feature_key=feature_key,
                        model_type=model_type,
                        metric_name=args.metric,
                        metrics_path=results_dir / model_type / group / feature_key / "metrics.json",
                        extra_payload={"rank": int(args.rank)},
                    ),
                )

                if not pending_model_types:
                    print(
                        f"[skip feature] group={group} feature={feature_key} "
                        f"all requested model_types already have metrics.json; skipping feature materialization"
                    )
                    summary_rows.extend(existing_rows)
                    continue

                train_npz = materialize_feature_npz(
                    sharded_root=train_root,
                    cache_dir=cache_dir,
                    feature_key=feature_key,
                    split_name=args.train_split,
                    rebuild=args.rebuild_cache,
                )
                val_npz = materialize_feature_npz(
                    sharded_root=val_root,
                    cache_dir=cache_dir,
                    feature_key=feature_key,
                    split_name=args.val_split,
                    rebuild=args.rebuild_cache,
                )
                named_test_npzs = {
                    test_name: materialize_feature_npz(
                        sharded_root=test_root,
                        cache_dir=cache_dir,
                        feature_key=feature_key,
                        split_name=test_name,
                        rebuild=args.rebuild_cache,
                    )
                    for test_name, test_root in test_roots.items()
                }

                with np.load(train_npz, allow_pickle=False) as data:
                    input_dim = int(data["tokens"].shape[1])

                summary_rows.extend(existing_rows)

                for model_type in pending_model_types:
                    run_dir = results_dir / model_type / group / feature_key
                    run_dir.mkdir(parents=True, exist_ok=True)
                    cfg = build_standard_probe_cfg(
                        base_cfg=base_cfg,
                        train_npz=str(train_npz),
                        val_npz=str(val_npz),
                        test_npzs={name: str(path) for name, path in named_test_npzs.items()},
                        model_type=model_type,
                        input_dim=input_dim,
                        rank=int(args.rank),
                        out_dir=str(run_dir),
                        eval_batch_size=int(base_cfg.get("train", {}).get("eval_batch_size", 262144)),
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

                    tmp_cfg = run_dir / "config.json"
                    probe_script = Path(__file__).resolve().with_name("run_isdynamic_probe.py")
                    write_probe_config_and_run(
                        cfg_path=tmp_cfg,
                        cfg=cfg,
                        probe_script_path=probe_script,
                        python_executable=sys.executable,
                        cwd=repo_root,
                        eval_only=bool(args.eval_only),
                        checkpoint_path=checkpoint_path,
                    )

                    metrics_path = effective_run_dir / "metrics.json"
                    result_row = best_val_run_result_record(
                        group_name=group,
                        feature_key=feature_key,
                        model_type=model_type,
                        metric_name=args.metric,
                        metrics_path=metrics_path,
                        extra_payload={"rank": int(args.rank), "run_dir": str(effective_run_dir)},
                    )
                    summary_rows.append(result_row)
                    print(
                        f"[result] group={group} feature={feature_key} model={model_type} "
                        f"val_{args.metric}={result_row['best_val']}"
                    )

        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = results_dir / "summary.json"

        merged_groups, merged_model_types, merged_rows = merge_best_val_summary_rows(
            summary_path=summary_path,
            groups=groups,
            model_types=model_types,
            rank=int(args.rank),
            current_rows=summary_rows,
        )
        summary = build_best_val_summary(
            groups=merged_groups,
            model_types=merged_model_types,
            metric=args.metric,
            rank=int(args.rank),
            skip_existing=bool(args.skip_existing),
            rows=merged_rows,
        )
        _save_json(summary_path, summary)
        print(f"Saved summary to: {summary_path}")
    finally:
        if tmp_cache_ctx is not None:
            tmp_cache_ctx.cleanup()


if __name__ == "__main__":
    main()
