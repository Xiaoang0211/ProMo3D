from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import importlib.util
import sys
import types

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "Depth-Anything-3"
SRC_ROOT = MODEL_ROOT / "src"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_local_depth_anything3_class():
    package_name = "depth_anything_3"
    package_root = SRC_ROOT / package_name
    api_path = package_root / "api.py"
    if not api_path.exists():
        raise ModuleNotFoundError(f"Local DA3 api not found: {api_path}")

    # Drop any conflicting preloaded modules first.
    for key in list(sys.modules.keys()):
        if key == package_name or key.startswith(f"{package_name}."):
            sys.modules.pop(key, None)

    package_module = types.ModuleType(package_name)
    package_module.__path__ = [str(package_root)]
    package_module.__package__ = package_name
    sys.modules[package_name] = package_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api",
        api_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not build import spec for local DA3 api: {api_path}")

    api_module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.api"] = api_module
    spec.loader.exec_module(api_module)
    return api_module.DepthAnything3


try:
    from depth_anything_3.api import DepthAnything3
except ModuleNotFoundError:
    DepthAnything3 = _load_local_depth_anything3_class()


ModelPathLike = Union[str, Path]


def canonical_group_name(group: str) -> str:
    if group == "dual_dpt_head":
        return "depth_head"
    return group


def load_da3_model(model_dir: ModelPathLike, device: torch.device) -> torch.nn.Module:
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device)
    model.eval()
    return model


def _resolve_anyview_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    seen: set[int] = set()
    while True:
        obj_id = id(current)
        if obj_id in seen:
            return current
        seen.add(obj_id)

        if hasattr(current, "backbone"):
            return current
        if hasattr(current, "da3"):
            current = current.da3
            continue
        if hasattr(current, "model"):
            current = current.model
            continue
        return current


def _block_modes_for_model(anyview_model: torch.nn.Module) -> Dict[int, str]:
    recorder = getattr(anyview_model, "attention_recorder", None)
    if recorder is not None and getattr(recorder, "block_modes", None):
        return dict(recorder.block_modes)

    vit = anyview_model.backbone.pretrained
    alt_start = getattr(vit, "alt_start", -1)
    return {
        idx: ("global" if alt_start != -1 and idx >= alt_start and idx % 2 == 1 else "local")
        for idx in range(len(vit.blocks))
    }


def _group_block_indices(block_modes: Dict[int, str]) -> Dict[str, List[int]]:
    grouped = {"inter_frame_block": [], "intra_frame_block": []}
    for block_idx in sorted(block_modes):
        if block_modes[block_idx] == "global":
            grouped["inter_frame_block"].append(block_idx)
        else:
            grouped["intra_frame_block"].append(block_idx)
    return grouped


def extract_matrix_from_block(block: torch.nn.Module, mode: str) -> torch.Tensor:
    qkv = block.attn.qkv.weight.detach().float()
    c_dim = qkv.shape[1]
    if qkv.shape[0] != 3 * c_dim:
        raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")
    wq, wk, wv = qkv.chunk(3, dim=0)
    if mode == "attention":
        return wq.transpose(0, 1) @ wk
    if mode == "value":
        return wv
    raise ValueError(f"Unsupported mode: {mode}")


def _resolve_projection_block_idx(
    feature_block_idx: int,
    projection_block_idx: Optional[int],
    depth: int,
    fallback_current_for_last: bool,
) -> int:
    idx = feature_block_idx + 1 if projection_block_idx is None else projection_block_idx
    if idx >= depth:
        if fallback_current_for_last:
            idx = feature_block_idx
        else:
            raise ValueError(
                f"Projection block index {idx} is out of range for depth {depth}. "
                "Use --fallback_current_for_last or provide --projection_block_idx."
            )
    if idx < 0:
        raise ValueError(f"Invalid projection block index: {idx}")
    return idx


def get_projection_matrix(
    model: torch.nn.Module,
    group: str,
    feature_block_idx: int,
    mode: str,
    projection_block_idx: Optional[int] = None,
    fallback_current_for_last: bool = False,
) -> Tuple[torch.Tensor, int]:
    anyview_model = _resolve_anyview_model(model)
    vit = anyview_model.backbone.pretrained
    group = canonical_group_name(group)

    if group in {"inter_frame_block", "intra_frame_block"}:
        grouped = _group_block_indices(_block_modes_for_model(anyview_model))
        actual_block_indices = grouped[group]
        blocks = [vit.blocks[idx] for idx in actual_block_indices]
    elif group == "depth_head":
        out_layers = list(anyview_model.backbone.out_layers)
        blocks = [vit.blocks[idx] for idx in out_layers]
    else:
        raise ValueError(f"Unsupported group '{group}'.")

    proj_idx = _resolve_projection_block_idx(
        feature_block_idx=feature_block_idx,
        projection_block_idx=projection_block_idx,
        depth=len(blocks),
        fallback_current_for_last=fallback_current_for_last,
    )
    return extract_matrix_from_block(blocks[proj_idx], mode=mode), proj_idx
