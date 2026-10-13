from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from isdynamic.layout import normalize_model_name
from isdynamic.paths import PROJECT_ROOT, outputs_root


PCA_KMEANS_ROOT = outputs_root(PROJECT_ROOT) / "pca_kmeans"


def ensure_pca_kmeans_root(project_root: Path = PROJECT_ROOT) -> Path:
    root = outputs_root(project_root) / "pca_kmeans"
    for child in ("configs", "masks", "eval", "tables", "visualizations", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def configs_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "configs"


def masks_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "masks"


def eval_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "eval"


def tables_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "tables"


def visualizations_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "visualizations"


def logs_root(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_pca_kmeans_root(project_root) / "logs"


def selected_config_path(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return configs_root(project_root) / f"selected_config_{normalize_model_name(model_name)}.json"


def mask_tree_root_for_variant(
    *,
    model_name: str,
    variant_name: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return masks_root(project_root) / normalize_model_name(model_name) / variant_name


def benchmark_root_for_variant(
    *,
    model_name: str,
    variant_name: str,
    benchmark_name: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return mask_tree_root_for_variant(
        model_name=model_name,
        variant_name=variant_name,
        project_root=project_root,
    ) / str(benchmark_name)


def sequence_root_for_variant(
    *,
    model_name: str,
    variant_name: str,
    benchmark_name: str,
    sequence_tag: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return benchmark_root_for_variant(
        model_name=model_name,
        variant_name=variant_name,
        benchmark_name=benchmark_name,
        project_root=project_root,
    ) / str(sequence_tag)


def frame_output_dir_for_variant(
    *,
    model_name: str,
    variant_name: str,
    benchmark_name: str,
    sequence_tag: str,
    frame_idx: int,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return sequence_root_for_variant(
        model_name=model_name,
        variant_name=variant_name,
        benchmark_name=benchmark_name,
        sequence_tag=sequence_tag,
        project_root=project_root,
    ) / f"frame_{int(frame_idx):05d}"


def eval_json_path(
    *,
    model_name: str,
    variant_name: str,
    benchmark_name: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    out_dir = eval_root(project_root) / normalize_model_name(model_name) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{benchmark_name}.json"


def backbone_csv_path(model_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    return tables_root(project_root) / f"{normalize_model_name(model_name)}_pca_kmeans_results.csv"


def combined_csv_path(project_root: Path = PROJECT_ROOT) -> Path:
    return tables_root(project_root) / "combined_pca_kmeans_results.csv"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
