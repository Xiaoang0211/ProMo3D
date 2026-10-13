from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


SCHEMA_VERSION = "0.1.0"

SUPPORTED_MODELS = (
    "vggt",
    "vggt_omega",
    "pi3",
    "pi3x",
    "dust3r",
    "map_anything",
    "depth_anything_3",
)

RAW_OUTPUT_MODELS = ("vggt", "vggt_omega", "pi3", "dust3r", "map_anything", "depth_anything_3")


def normalize_model_name(name: str) -> str:
    text = name.strip().lower().replace("-", "_")
    if text == "depth_anything3":
        return "depth_anything_3"
    if text == "mapanything":
        return "map_anything"
    if text in {"vggtomega", "omega"}:
        return "vggt_omega"
    return text


def project_root_from_package(package_root: Path) -> Path:
    return package_root.resolve().parent.parent


def build_layout_lines() -> List[str]:
    lines = [
        "isDynamicProbe/",
        "  models/",
    ]
    lines.extend(f"    {name}/" for name in SUPPORTED_MODELS)
    lines.extend(
        [
            "  isdynamic/",
            "    core/",
            "    models/",
            "    legacy/",
            "  outputs/",
            "    model_outputs/",
        ]
    )
    lines.extend(f"      {name}/" for name in RAW_OUTPUT_MODELS)
    lines.append("    isdynamic_visualizations/")
    lines.extend(f"      {name}/" for name in SUPPORTED_MODELS)
    lines.append("      comparisons/")
    return lines


def ensure_output_skeleton(project_root: Path) -> Iterable[Path]:
    created = []
    outputs_dir = Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", str(project_root / "outputs"))).expanduser()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    created.append(outputs_dir)

    raw_root = outputs_dir / "model_outputs"
    raw_root.mkdir(parents=True, exist_ok=True)
    created.append(raw_root)
    for name in RAW_OUTPUT_MODELS:
        path = raw_root / normalize_model_name(name)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    viz_root = outputs_dir / "isdynamic_visualizations"
    viz_root.mkdir(parents=True, exist_ok=True)
    created.append(viz_root)
    viz_subdirs = ("blockwise", "scoremaps", "lowrank", "tables")
    for name in SUPPORTED_MODELS:
        path = viz_root / normalize_model_name(name)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
        for subdir in viz_subdirs:
            child = path / subdir
            child.mkdir(parents=True, exist_ok=True)
            created.append(child)
    comparisons_root = viz_root / "comparisons"
    comparisons_root.mkdir(parents=True, exist_ok=True)
    created.append(comparisons_root)
    for subdir in ("blockwise", "tables"):
        child = comparisons_root / subdir
        child.mkdir(parents=True, exist_ok=True)
        created.append(child)
    return created
