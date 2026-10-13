from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from ..core.output_paths import ANCHOR_OUTPUTS_DIR_NAME, ENFORCE_TO_ANCHOR_SUBDIR, resolve_probe_out_dir_from_mapping


def resolve_existing_metrics_path(metrics_path: Path) -> Path:
    if metrics_path.exists():
        return metrics_path

    # Only anchor-output runs may use the legacy nested anchor layout. Non-anchor
    # experiments must not silently reuse metrics from outputs_anchor0.
    if ANCHOR_OUTPUTS_DIR_NAME not in metrics_path.parts:
        return metrics_path
    legacy_candidate = metrics_path.parent / ENFORCE_TO_ANCHOR_SUBDIR / metrics_path.name
    return legacy_candidate if legacy_candidate.exists() else metrics_path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def apply_train_env_overrides(train_cfg: Dict | None) -> Dict | None:
    if train_cfg is None:
        return None
    cfg = copy.deepcopy(train_cfg)

    safeguard = os.environ.get("ANCHOR0_SAFEGUARD")
    if safeguard is not None and safeguard.strip():
        enabled = safeguard.strip().lower() not in {"0", "false", "off", "none", "no"}
        cfg["pair_force_anchor_static_label"] = enabled
        if not enabled:
            cfg["pair_min_anchor_static_pairs_per_batch"] = 0
            cfg["pair_min_anchor_dynamic_pairs_per_batch"] = 0

    static_pairs = os.environ.get("ANCHOR0_STATIC_PAIRS_PER_BATCH")
    if static_pairs is not None and static_pairs.strip():
        cfg["pair_min_anchor_static_pairs_per_batch"] = int(static_pairs)
    dynamic_pairs = os.environ.get("ANCHOR0_DYNAMIC_PAIRS_PER_BATCH")
    if dynamic_pairs is not None and dynamic_pairs.strip():
        cfg["pair_min_anchor_dynamic_pairs_per_batch"] = int(dynamic_pairs)

    mode = os.environ.get("POSITION_RESIDUALIZATION")
    if mode is not None and mode.strip():
        cfg["position_residualization"] = mode.strip()
    freqs = os.environ.get("POSITION_RESIDUALIZATION_FOURIER_FREQS")
    if freqs is not None and freqs.strip():
        cfg["position_residualization_fourier_freqs"] = int(freqs)
    ridge = os.environ.get("POSITION_RESIDUALIZATION_RIDGE")
    if ridge is not None and ridge.strip():
        cfg["position_residualization_ridge"] = float(ridge)
    label_control = os.environ.get("TRAIN_LABEL_CONTROL")
    if label_control is not None and label_control.strip():
        cfg["train_label_control"] = label_control.strip()
    return cfg


def resolve_sweep_results_dir(results_dir: str | Path, train_cfg: Dict | None, *, repo_root: Path | None = None) -> Path:
    path = Path(results_dir).expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    return resolve_probe_out_dir_from_mapping(path, apply_train_env_overrides(train_cfg)).resolve()


def run_cmd(cmd: Sequence[str], *, cwd: Path) -> None:
    print("[cmd] " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def write_probe_config_and_run(
    *,
    cfg_path: Path,
    cfg: Dict,
    probe_script_path: Path,
    python_executable: str,
    cwd: Path,
    eval_only: bool = False,
    checkpoint_path: Path | None = None,
) -> None:
    save_json(cfg_path, cfg)
    probe_cmd = [python_executable, str(probe_script_path), "--config", str(cfg_path)]
    if eval_only:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when eval_only=True")
        probe_cmd.extend(["--eval_only", "--checkpoint", str(checkpoint_path)])
    run_cmd(probe_cmd, cwd=cwd)


def standard_existing_result_record(
    *,
    group_name: str,
    block_idx: int,
    feature_key: str,
    model_type: str,
    metric_name: str,
    run_dir: Path,
    metrics_path: Path,
) -> Dict[str, object] | None:
    metrics_path = resolve_existing_metrics_path(metrics_path)
    if not metrics_path.exists():
        return None
    try:
        report = load_json(metrics_path)
    except Exception:
        return None
    return {
        "group": group_name,
        "block_idx": block_idx,
        "feature_key": feature_key,
        "model_type": model_type,
        "metric_primary": metric_name,
        "best": report.get("best", {}),
        "best_tests": report.get("best_tests", {}),
        "run_dir": str(metrics_path.parent),
        "skipped_existing": True,
    }


def best_val_existing_result_record(
    *,
    group_name: str,
    feature_key: str,
    model_type: str,
    metric_name: str,
    metrics_path: Path,
    extra_payload: Dict[str, object] | None = None,
) -> Dict[str, object] | None:
    metrics_path = resolve_existing_metrics_path(metrics_path)
    if not metrics_path.exists():
        return None
    try:
        metrics = load_json(metrics_path)
    except Exception:
        return None
    row: Dict[str, object] = {
        "group": group_name,
        "feature_key": feature_key,
        "model_type": model_type,
        "metric": metric_name,
        "best_val": metrics["best_val"].get(metric_name),
        "best_tests": metrics.get("best_tests", {}),
        "metrics_path": str(metrics_path),
        "skipped_existing": True,
    }
    if extra_payload:
        row.update(extra_payload)
    return row


def standard_run_result_record(
    *,
    group_name: str,
    block_idx: int,
    feature_key: str,
    model_type: str,
    metric_name: str,
    run_dir: Path,
    metrics_path: Path,
) -> Dict[str, object]:
    report = load_json(metrics_path)
    return {
        "group": group_name,
        "block_idx": block_idx,
        "feature_key": feature_key,
        "model_type": model_type,
        "metric_primary": metric_name,
        "best": report.get("best", {}),
        "best_tests": report.get("best_tests", {}),
        "run_dir": str(run_dir),
    }


def best_val_run_result_record(
    *,
    group_name: str,
    feature_key: str,
    model_type: str,
    metric_name: str,
    metrics_path: Path,
    extra_payload: Dict[str, object] | None = None,
) -> Dict[str, object]:
    metrics = load_json(metrics_path)
    row: Dict[str, object] = {
        "group": group_name,
        "feature_key": feature_key,
        "model_type": model_type,
        "metric": metric_name,
        "best_val": metrics["best_val"].get(metric_name),
        "best_tests": metrics.get("best_tests", {}),
        "metrics_path": str(metrics_path),
    }
    if extra_payload:
        row.update(extra_payload)
    return row


def build_standard_probe_cfg(
    *,
    base_cfg: Dict,
    train_npz: str,
    val_npz: str,
    test_npzs: Dict[str, str],
    model_type: str,
    input_dim: int,
    rank: int,
    out_dir: str,
    eval_batch_size: int,
    include_matrix_keys: bool = True,
) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset"]["train_npz"] = train_npz
    cfg["dataset"]["val_npz"] = val_npz
    cfg["dataset"]["test_npz"] = next(iter(test_npzs.values())) if len(test_npzs) == 1 else None
    cfg["dataset"]["test_npzs"] = dict(test_npzs)
    cfg["dataset"]["feature_key"] = "tokens"
    cfg["model"]["model_type"] = model_type
    cfg["model"]["input_dim"] = int(input_dim)
    cfg["model"]["rank"] = int(rank)
    cfg["train"] = apply_train_env_overrides(cfg.setdefault("train", {})) or cfg.setdefault("train", {})
    cfg["train"]["eval_batch_size"] = int(eval_batch_size)
    if include_matrix_keys:
        cfg["model"]["attention_matrix_path"] = None
        cfg["model"]["attention_matrix_key"] = None
        cfg["model"]["value_matrix_path"] = None
        cfg["model"]["value_matrix_key"] = None
    cfg["output"]["out_dir"] = str(
        resolve_probe_out_dir_from_mapping(out_dir, cfg.get("train")).resolve()
    )
    return cfg


def build_standard_blockwise_summary(
    *,
    metric_primary: str,
    rank: int,
    groups: Sequence[str],
    model_types: Sequence[str],
    sharded_root: str,
    train_split: str,
    val_split: str,
    test_splits: Sequence[str],
    cache_dir: str,
    ephemeral_cache: bool,
    skip_existing: bool,
    results: Sequence[Dict[str, object]],
    extra_fields: Dict[str, object] | None = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "metric_primary": metric_primary,
        "rank": int(rank),
        "groups": list(groups),
        "model_types": list(model_types),
        "sharded_root": sharded_root,
        "train_split": train_split,
        "val_split": val_split,
        "test_splits": list(test_splits),
        "cache_dir": cache_dir,
        "ephemeral_cache": bool(ephemeral_cache),
        "skip_existing": bool(skip_existing),
        "results": list(results),
    }
    if extra_fields:
        summary.update(extra_fields)
    return summary


def merge_best_val_summary_rows(
    *,
    summary_path: Path,
    groups: Sequence[str],
    model_types: Sequence[str],
    rank: int,
    current_rows: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    merged_rows: List[Dict[str, object]] = []
    merged_model_types = list(model_types)
    merged_groups = list(groups)
    if summary_path.exists():
        try:
            existing_summary = load_json(summary_path)
            existing_rows = existing_summary.get("rows", [])
            if isinstance(existing_rows, list):
                merged_rows.extend(existing_rows)
            existing_model_types = existing_summary.get("model_types", [])
            if isinstance(existing_model_types, list):
                merged_model_types = list(dict.fromkeys([*existing_model_types, *merged_model_types]))
            existing_groups = existing_summary.get("groups", [])
            if isinstance(existing_groups, list):
                merged_groups = list(dict.fromkeys([*existing_groups, *merged_groups]))
        except Exception:
            pass

    row_by_key = {}
    for row in merged_rows + list(current_rows):
        key = (row.get("group"), row.get("feature_key"), row.get("model_type"), row.get("rank", int(rank)))
        row_by_key[key] = row
    return merged_groups, merged_model_types, list(row_by_key.values())


def build_best_val_summary(
    *,
    groups: Sequence[str],
    model_types: Sequence[str],
    metric: str,
    rank: int,
    skip_existing: bool,
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "groups": list(groups),
        "model_types": list(model_types),
        "metric": metric,
        "rank": int(rank),
        "skip_existing": bool(skip_existing),
        "rows": list(rows),
    }


def partition_existing_and_pending(
    *,
    model_types: Sequence[str],
    skip_existing: bool,
    existing_record_for_model_type: Callable[[str], Dict[str, object] | None],
) -> Tuple[List[Dict[str, object]], List[str]]:
    existing_rows: List[Dict[str, object]] = []
    pending_model_types: List[str] = []
    for model_type in model_types:
        if skip_existing:
            record = existing_record_for_model_type(model_type)
            if record is not None:
                existing_rows.append(record)
                continue
        pending_model_types.append(model_type)
    return existing_rows, pending_model_types
