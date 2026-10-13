from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from ..dispatch import MODEL_REPO_DIRS, ensure_arg, pop_optional_arg
from ..core.output_paths import resolve_probe_out_dir_from_mapping
from ..paths import PROJECT_ROOT, probe_cache_root, probe_experiment_root, token_dataset_root, token_split_root


def _repo_root(model_name: str) -> Path:
    repo_dir = MODEL_REPO_DIRS.get(model_name)
    if repo_dir is None:
        raise SystemExit(f"Unsupported model for shared IsDynamic pipeline: {model_name}")
    return PROJECT_ROOT / repo_dir


def _config_owner(model_name: str) -> str:
    return "pi3" if model_name == "pi3x" else model_name


def _supports_matrix_cache_arg(model_name: str) -> bool:
    return model_name in {"vggt", "vggt_omega", "pi3", "pi3x", "da3", "mapanything"}


def default_config_path(model_name: str) -> Path:
    return PROJECT_ROOT / "isdynamic" / "configs" / _config_owner(model_name) / "isdynamic_probe.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _resolve_config_path(model_name: str, config_path: str | None) -> Path:
    if config_path is None:
        return default_config_path(model_name)
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    return (_repo_root(model_name) / candidate).resolve()


def _resolve_aux_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _infer_dataset_name_from_config(raw_cfg: dict[str, Any]) -> str:
    dataset_cfg = raw_cfg.get("dataset") or {}
    candidates: list[str] = []
    for key in ("train_npz", "val_npz", "test_npz"):
        value = dataset_cfg.get(key)
        if isinstance(value, str):
            candidates.append(value)
    test_npzs = dataset_cfg.get("test_npzs")
    if isinstance(test_npzs, dict):
        candidates.extend(str(value) for value in test_npzs.values())

    for candidate in candidates:
        lowered = candidate.lower()
        if "youtubevos" in lowered:
            return "youtubevos"
        if "davis" in lowered:
            return "davis"
        if "segtrack" in lowered:
            return "segtrackv2"
        if "fbms" in lowered:
            return "fbms59"
    return "youtubevos"


def _maybe_rewrite_npz_path(model_name: str, dataset_name: str, base_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("data/isdynamic/")
        or "/data/isdynamic/" in normalized
        or "/isdynamic/" in normalized
    ):
        return str((token_dataset_root(model_name, dataset_name, project_root=PROJECT_ROOT) / Path(value).name).resolve())
    return _resolve_aux_path(base_dir, value)


def _derive_run_name(config_path: Path, raw_cfg: dict[str, Any]) -> str:
    output_cfg = raw_cfg.get("output") or {}
    out_dir = output_cfg.get("out_dir")
    if isinstance(out_dir, str) and out_dir:
        leaf = Path(out_dir).name.strip()
        if leaf and leaf not in {"outputs", "out", "isdynamic_probe"}:
            return leaf

    dataset_cfg = raw_cfg.get("dataset") or {}
    feature_key = dataset_cfg.get("feature_key")
    if isinstance(feature_key, str) and feature_key:
        return feature_key.replace("/", "_")
    return config_path.stem


def infer_dataset_name(argv: List[str]) -> str:
    sharded_root, _ = pop_optional_arg(argv, "--sharded_root")
    if sharded_root:
        lowered = sharded_root.lower()
        if "youtubevos" in lowered:
            return "youtubevos"
        if "davis" in lowered:
            return "davis"
    return "youtubevos"


def rewrite_blockwise_probing_args(model_name: str, argv: List[str]) -> List[str]:
    experiment_name, rewritten = pop_optional_arg(argv, "--experiment")
    dataset_name = infer_dataset_name(argv)
    if experiment_name is None:
        experiment_name = f"{dataset_name}_main"

    rewritten = ensure_arg(rewritten, "--base_config", str(default_config_path(model_name)))
    sharded_root_default = token_split_root(model_name=model_name, dataset_name=dataset_name, project_root=PROJECT_ROOT)
    results_root_default = probe_experiment_root(model_name=model_name, experiment_name=experiment_name, project_root=PROJECT_ROOT)
    cache_root_default = probe_cache_root(model_name=model_name, experiment_name=experiment_name, project_root=PROJECT_ROOT)
    matrix_cache_root_default = cache_root_default / "matrices"

    rewritten = ensure_arg(rewritten, "--sharded_root", str(sharded_root_default))
    rewritten = ensure_arg(rewritten, "--results_dir", str(results_root_default))
    rewritten = ensure_arg(rewritten, "--cache_dir", str(cache_root_default / "flat_features"))
    if _supports_matrix_cache_arg(model_name):
        rewritten = ensure_arg(rewritten, "--matrix_cache_dir", str(matrix_cache_root_default))
    return rewritten


def rewrite_probe_config(model_name: str, argv: List[str]) -> List[str]:
    experiment_name, rewritten = pop_optional_arg(argv, "--experiment")
    config_arg, rewritten = pop_optional_arg(rewritten, "--config")
    config_path = _resolve_config_path(model_name, config_arg)
    raw_cfg = _read_json(config_path)

    dataset_name = _infer_dataset_name_from_config(raw_cfg)
    if experiment_name is None:
        experiment_name = f"{dataset_name}_main"

    config_dir = config_path.parent
    dataset_cfg = raw_cfg.setdefault("dataset", {})
    for key in ("train_npz", "val_npz", "test_npz"):
        dataset_cfg[key] = _maybe_rewrite_npz_path(
            model_name=model_name,
            dataset_name=dataset_name,
            base_dir=config_dir,
            value=dataset_cfg.get(key),
        )

    test_npzs = dataset_cfg.get("test_npzs")
    if isinstance(test_npzs, dict):
        dataset_cfg["test_npzs"] = {
            split_name: _maybe_rewrite_npz_path(
                model_name=model_name,
                dataset_name=dataset_name,
                base_dir=config_dir,
                value=split_path,
            )
            for split_name, split_path in test_npzs.items()
        }

    model_cfg = raw_cfg.setdefault("model", {})
    for key in ("value_feature_matrix_path", "attention_matrix_path", "value_matrix_path"):
        model_cfg[key] = _resolve_aux_path(config_dir, model_cfg.get(key))

    model_type = str(model_cfg.get("model_type") or "probe")
    run_name = _derive_run_name(config_path, raw_cfg)
    experiment_root = probe_experiment_root(model_name=model_name, experiment_name=experiment_name, project_root=PROJECT_ROOT)

    output_cfg = raw_cfg.setdefault("output", {})
    default_out_dir = (experiment_root / "single_runs" / model_type / run_name).resolve()
    output_cfg["out_dir"] = str(
        resolve_probe_out_dir_from_mapping(default_out_dir, raw_cfg.get("train")).resolve()
    )

    generated_config_path = experiment_root / "_generated_configs" / f"{config_path.stem}__{run_name}.json"
    _write_json(generated_config_path, raw_cfg)
    return [*rewritten, "--config", str(generated_config_path)]
