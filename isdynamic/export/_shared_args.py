from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ..dispatch import ensure_arg, pop_optional_arg
from ..paths import PROJECT_ROOT, token_dataset_root


def infer_dataset_name(script_filename: str, argv: List[str]) -> Tuple[str, List[str]]:
    dataset_name, remaining = pop_optional_arg(argv, "--dataset_name")
    if dataset_name is not None:
        return dataset_name, ensure_arg(remaining, "--dataset_name", dataset_name)
    if "youtubevos" in script_filename:
        return "youtubevos", remaining
    if "davis" in script_filename:
        return "davis", remaining
    return "unknown", remaining


def rewrite_export_args(model_name: str, script_filename: str, argv: List[str]) -> List[str]:
    dataset_name, rewritten = infer_dataset_name(script_filename, argv)
    output_dir = token_dataset_root(model_name=model_name, dataset_name=dataset_name, project_root=PROJECT_ROOT).parent
    rewritten = ensure_arg(rewritten, "--output_dir", str(output_dir))
    return rewritten
