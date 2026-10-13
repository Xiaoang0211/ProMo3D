from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXTERNAL_BASE = Path("/media/xiaoang/Extreme SSD")


def _normalize_model_name(model_name: str) -> str:
    text = model_name.strip().lower().replace("-", "_")
    if text == "mapanything":
        return "map_anything"
    if text in {"vggtomega", "omega"}:
        return "vggt_omega"
    if text == "depth_anything3":
        return "depth_anything_3"
    return text


def _raw_owner(model_name: str) -> str:
    normalized = _normalize_model_name(model_name)
    if normalized == "pi3x":
        return "pi3"
    return normalized


def token_root_for_model(model_name: str) -> Path:
    return artifact_root_for_model(model_name) / "data" / "isdynamic"


def token_path_for_model(model_name: str, *parts: str) -> str:
    return str(token_root_for_model(model_name).joinpath(*parts))


def artifact_root_for_model(model_name: str) -> Path:
    normalized = _normalize_model_name(model_name)
    env_key = f"{normalized.upper()}_ARTIFACT_ROOT"
    env_key = env_key.replace("MAP_ANYTHING", "MAPANYTHING")
    if normalized == "pi3":
        env_key = "PI3_ARTIFACT_ROOT"
    elif normalized == "pi3x":
        env_key = "PI3X_ARTIFACT_ROOT"
    elif normalized == "depth_anything_3":
        env_key = "DA3_ARTIFACT_ROOT"
    external_default_map = {
        "vggt": _DEFAULT_EXTERNAL_BASE / "vggt_tokens",
        "vggt_omega": _DEFAULT_EXTERNAL_BASE / "vggt_omega_tokens",
        "pi3": _DEFAULT_EXTERNAL_BASE / "pi3_tokens",
        "pi3x": _DEFAULT_EXTERNAL_BASE / "pi3x_tokens",
        "dust3r": _DEFAULT_EXTERNAL_BASE / "dust3r_tokens",
        "map_anything": _DEFAULT_EXTERNAL_BASE / "mapanything_tokens",
        "depth_anything_3": _DEFAULT_EXTERNAL_BASE / "da3_tokens",
    }
    default_root = external_default_map.get(normalized, _PROJECT_ROOT)
    return Path(os.environ.get(env_key, str(default_root))).expanduser()


def artifact_path(*parts: str, model_name: str = "vggt") -> str:
    return str(artifact_root_for_model(model_name).joinpath(*parts))
