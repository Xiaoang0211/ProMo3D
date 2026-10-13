from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO_DIRS: Dict[str, str] = {
    "vggt": "models/vggt",
    "vggt_omega": "models/vggt-omega",
    "pi3": "models/Pi3",
    "pi3x": "models/Pi3",
    "dust3r": "models/dust3r",
    "map_anything": "models/map-anything",
    "depth_anything_3": "models/Depth-Anything-3",
}


LEGACY_IMPL_DIRS: Dict[str, str] = {
    "vggt": "vggt",
    "vggt_omega": "vggt_omega",
    "pi3": "pi3",
    "pi3x": "pi3",
    "dust3r": "dust3r",
    "map_anything": "map_anything",
    "depth_anything_3": "depth_anything_3",
}




def _normalize_model_name(name: str) -> str:
    text = name.strip().lower().replace("-", "_")
    if text == "mapanything":
        return "map_anything"
    if text in {"da3", "depth_anything3"}:
        return "depth_anything_3"
    if text in {"vggtomega", "omega"}:
        return "vggt_omega"
    return text


def _pop_model_arg(argv: List[str]) -> Tuple[str, List[str]]:
    for idx, token in enumerate(argv):
        if token == "--model":
            if idx + 1 >= len(argv):
                raise SystemExit("--model requires a value")
            model_name = _normalize_model_name(argv[idx + 1])
            remaining = argv[:idx] + argv[idx + 2 :]
            return model_name, remaining
        if token.startswith("--model="):
            model_name = _normalize_model_name(token.split("=", 1)[1])
            remaining = argv[:idx] + argv[idx + 1 :]
            return model_name, remaining
    raise SystemExit("Shared IsDynamic entrypoints require --model <name>.")


def has_arg(argv: List[str], name: str) -> bool:
    return any(token == name or token.startswith(f"{name}=") for token in argv)


def pop_optional_arg(argv: List[str], name: str) -> Tuple[str | None, List[str]]:
    for idx, token in enumerate(argv):
        if token == name:
            if idx + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[idx + 1]
            remaining = argv[:idx] + argv[idx + 2 :]
            return value, remaining
        if token.startswith(f"{name}="):
            value = token.split("=", 1)[1]
            remaining = argv[:idx] + argv[idx + 1 :]
            return value, remaining
    return None, argv


def ensure_arg(argv: List[str], name: str, value: str) -> List[str]:
    if has_arg(argv, name):
        return list(argv)
    return [*argv, name, value]


def _legacy_script_path(model_name: str, script_filename: str) -> Path:
    normalized = _normalize_model_name(model_name)
    repo_dir = MODEL_REPO_DIRS.get(normalized)
    legacy_dir = LEGACY_IMPL_DIRS.get(normalized)
    if repo_dir is None or legacy_dir is None:
        raise SystemExit(f"Unsupported model for shared IsDynamic script: {model_name}")
    path = PROJECT_ROOT / "isdynamic" / "legacy" / legacy_dir / script_filename
    if not path.exists():
        raise SystemExit(f"Legacy script implementation not found: {path}")
    return path


def run_legacy_entry(script_filename: str, forwarded_args: List[str] | None = None, model_name: str | None = None) -> None:
    if model_name is None or forwarded_args is None:
        model_name, forwarded_args = _pop_model_arg(sys.argv[1:])
    legacy_path = _legacy_script_path(model_name, script_filename)
    normalized_model_name = _normalize_model_name(model_name)
    repo_dir = MODEL_REPO_DIRS.get(normalized_model_name)
    original_argv = sys.argv[:]
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str((PROJECT_ROOT / "models").resolve()))
        if repo_dir is not None:
            sys.path.insert(0, str((PROJECT_ROOT / repo_dir).resolve()))
        legacy_argv = list(forwarded_args)
        pi3_model_entries = {"build_youtubevos_isdynamic_npz.py", "build_davis_isdynamic_npz.py",
                             "visualize_motion_mask.py", "visualize_isdynamic.py"}
        if normalized_model_name in {"pi3", "pi3x"} and script_filename in pi3_model_entries and not has_arg(legacy_argv, "--model"):
            legacy_argv = ["--model", model_name, *legacy_argv]
        sys.argv = [str(legacy_path), *legacy_argv]
        runpy.run_path(str(legacy_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def run_local_wrapper(entry_module: str, model_name: str) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], "--model", model_name, *original_argv[1:]]
        runpy.run_module(entry_module, run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path
