from __future__ import annotations

import os
from pathlib import Path

from .layout import normalize_model_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_TOKEN_BASE = Path(os.environ.get("ISDYNAMIC_TOKEN_ROOT_BASE", str(PROJECT_ROOT / "outputs/tokens"))).expanduser()


def outputs_root(project_root: Path = PROJECT_ROOT) -> Path:
    return Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", str(project_root / "outputs"))).expanduser()


def model_outputs_root(project_root: Path = PROJECT_ROOT) -> Path:
    return outputs_root(project_root) / "model_outputs"


def visualization_root(project_root: Path = PROJECT_ROOT) -> Path:
    return outputs_root(project_root) / "isdynamic_visualizations"


def _raw_output_owner(model_name: str) -> str:
    normalized = normalize_model_name(model_name)
    if normalized == "pi3x":
        return "pi3"
    return normalized


def _external_token_owner(model_name: str) -> str:
    normalized = normalize_model_name(model_name)
    mapping = {
        "vggt": "vggt_tokens",
        "vggt_omega": "vggt_omega_tokens",
        "pi3": "pi3_tokens",
        "pi3x": "pi3x_tokens",
        "dust3r": "dust3r_tokens",
        "map_anything": "mapanything_tokens",
        "depth_anything_3": "da3_tokens",
    }
    return mapping.get(normalized, f"{normalized}_tokens")


def raw_model_output_root(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return model_outputs_root(project_root) / _raw_output_owner(model_name)


def visualization_model_root(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return visualization_root(project_root) / normalize_model_name(model_name)


def visualization_experiment_root(model_name: str, experiment_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return visualization_model_root(model_name, project_root) / experiment_name


def tokens_root(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return EXTERNAL_TOKEN_BASE / _external_token_owner(model_name) / "data" / "isdynamic"


def token_dataset_root(model_name: str, dataset_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return tokens_root(model_name, project_root) / dataset_name


def token_split_root(model_name: str, dataset_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return token_dataset_root(model_name, dataset_name, project_root) / "tokens"


def probes_root(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return raw_model_output_root(model_name, project_root)


def probe_experiment_root(model_name: str, experiment_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return probes_root(model_name, project_root) / experiment_name


def probe_cache_root(model_name: str, experiment_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return probe_experiment_root(model_name, experiment_name, project_root) / "cache"
