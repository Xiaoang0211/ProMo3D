from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


ENFORCE_TO_ANCHOR_SUBDIR = "output_enforce_t0_anchor"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR_NAME = "outputs"
ANCHOR_OUTPUTS_DIR_NAME = "outputs_anchor0"
POSITION_RESIDUALIZED_SUBDIR = "pos_residualized"


def anchor_enforcement_enabled_from_train_cfg(train_cfg: Any) -> bool:
    force_anchor_static_label = bool(getattr(train_cfg, "pair_force_anchor_static_label", False))
    min_anchor_static_pairs = int(getattr(train_cfg, "pair_min_anchor_static_pairs_per_batch", 0) or 0)
    min_anchor_dynamic_pairs = int(getattr(train_cfg, "pair_min_anchor_dynamic_pairs_per_batch", 0) or 0)
    return force_anchor_static_label or min_anchor_static_pairs > 0 or min_anchor_dynamic_pairs > 0


def anchor_enforcement_enabled_from_mapping(train_cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(train_cfg, Mapping):
        return False
    force_anchor_static_label = bool(train_cfg.get("pair_force_anchor_static_label", False))
    min_anchor_static_pairs = int(train_cfg.get("pair_min_anchor_static_pairs_per_batch", 0) or 0)
    min_anchor_dynamic_pairs = int(train_cfg.get("pair_min_anchor_dynamic_pairs_per_batch", 0) or 0)
    return force_anchor_static_label or min_anchor_static_pairs > 0 or min_anchor_dynamic_pairs > 0


def outputs_root() -> Path:
    return Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", str(PROJECT_ROOT / OUTPUTS_DIR_NAME))).expanduser()


def anchor_outputs_root() -> Path:
    return Path(os.environ.get("ISDYNAMIC_ANCHOR_OUTPUTS_ROOT", str(PROJECT_ROOT / ANCHOR_OUTPUTS_DIR_NAME))).expanduser()


def _replace_outputs_segment(path: Path) -> Path | None:
    parts = path.parts
    if OUTPUTS_DIR_NAME not in parts:
        return None
    idx = parts.index(OUTPUTS_DIR_NAME)
    return Path(*parts[:idx], ANCHOR_OUTPUTS_DIR_NAME, *parts[idx + 1 :])


def build_anchor_enforced_out_dir(base_out_dir: str | Path) -> Path:
    out_dir = Path(base_out_dir).expanduser()
    if out_dir.name == ENFORCE_TO_ANCHOR_SUBDIR:
        return out_dir

    configured_outputs_root = outputs_root().resolve()
    configured_anchor_root = anchor_outputs_root().resolve()

    if out_dir.is_absolute():
        try:
            out_dir.resolve().relative_to(configured_anchor_root)
            return out_dir
        except ValueError:
            pass
        try:
            relative = out_dir.resolve().relative_to(configured_outputs_root)
            return configured_anchor_root / relative
        except ValueError:
            replaced = _replace_outputs_segment(out_dir)
            if replaced is not None:
                return replaced
            return configured_anchor_root / out_dir.name

    if out_dir.parts and out_dir.parts[0] == ANCHOR_OUTPUTS_DIR_NAME:
        return out_dir
    if out_dir.parts and out_dir.parts[0] == OUTPUTS_DIR_NAME:
        return Path(ANCHOR_OUTPUTS_DIR_NAME, *out_dir.parts[1:])
    replaced = _replace_outputs_segment(out_dir)
    if replaced is not None:
        return replaced
    return Path(ANCHOR_OUTPUTS_DIR_NAME) / out_dir


def position_residualization_mode_from_train_cfg(train_cfg: Any) -> str:
    mode = str(getattr(train_cfg, "position_residualization", "none") or "none").strip().lower()
    return "none" if mode in {"", "none", "off", "false", "0"} else mode


def position_residualization_mode_from_mapping(train_cfg: Mapping[str, Any] | None) -> str:
    if not isinstance(train_cfg, Mapping):
        return "none"
    mode = str(train_cfg.get("position_residualization", "none") or "none").strip().lower()
    return "none" if mode in {"", "none", "off", "false", "0"} else mode


def _append_position_residualization_subdir(path: Path, mode: str) -> Path:
    if mode == "none":
        return path
    safe_mode = mode.replace("/", "_").replace(" ", "_")
    parts = path.parts
    if POSITION_RESIDUALIZED_SUBDIR in parts:
        idx = parts.index(POSITION_RESIDUALIZED_SUBDIR)
        if idx + 1 < len(parts) and parts[idx + 1] == safe_mode:
            return path
    return path / POSITION_RESIDUALIZED_SUBDIR / safe_mode


def resolve_probe_out_dir(base_out_dir: str | Path, train_cfg: Any) -> Path:
    out_dir = build_anchor_enforced_out_dir(base_out_dir) if anchor_enforcement_enabled_from_train_cfg(train_cfg) else Path(base_out_dir)
    return _append_position_residualization_subdir(out_dir, position_residualization_mode_from_train_cfg(train_cfg))


def resolve_probe_out_dir_from_mapping(base_out_dir: str | Path, train_cfg: Mapping[str, Any] | None) -> Path:
    out_dir = build_anchor_enforced_out_dir(base_out_dir) if anchor_enforcement_enabled_from_mapping(train_cfg) else Path(base_out_dir)
    return _append_position_residualization_subdir(out_dir, position_residualization_mode_from_mapping(train_cfg))
