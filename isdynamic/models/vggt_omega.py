from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from isdynamic.core.preprocess_aspect_pad import aspect_pad_arrays

from .vggt import map_to_patch_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VGGT_OMEGA_REPO_ROOT = PROJECT_ROOT / "models" / "vggt-omega"
VGGT_OMEGA_PACKAGE_ROOT = VGGT_OMEGA_REPO_ROOT / "vggt_omega"

VGGT_OMEGA_PATCH_SIZE = 16
VGGT_OMEGA_TARGET_SIZE = 512
VGGT_OMEGA_CACHED_LAYER_INDICES = (4, 11, 17, 23)
VGGT_OMEGA_SOURCE_CHOICES = ["alt_attn", "cam_head", "depth_head"]
SOURCE_CHOICES = list(VGGT_OMEGA_SOURCE_CHOICES)


def _register_pkg(mod_name: str, pkg_dir: Path) -> None:
    init_py = pkg_dir / "__init__.py"
    if init_py.exists():
        spec = importlib.util.spec_from_file_location(
            mod_name,
            init_py,
            submodule_search_locations=[str(pkg_dir)],
        )
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not create import spec for {init_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return

    module = types.ModuleType(mod_name)
    module.__path__ = [str(pkg_dir)]
    module.__package__ = mod_name
    sys.modules[mod_name] = module


def _load_local_vggt_omega_class():
    repo_root = VGGT_OMEGA_REPO_ROOT
    package_root = VGGT_OMEGA_PACKAGE_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules.keys()):
        if name == "vggt_omega" or name.startswith("vggt_omega."):
            sys.modules.pop(name, None)

    _register_pkg("vggt_omega", package_root)
    for subdir in ("models", "heads", "layers", "utils"):
        _register_pkg(f"vggt_omega.{subdir}", package_root / subdir)
    return importlib.import_module("vggt_omega.models.vggt_omega").VGGTOmega


try:
    from vggt_omega.models.vggt_omega import VGGTOmega
except ModuleNotFoundError:
    VGGTOmega = _load_local_vggt_omega_class()


def _load_load_fn_helpers():
    if str(VGGT_OMEGA_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(VGGT_OMEGA_REPO_ROOT))
    module = importlib.import_module("vggt_omega.utils.load_fn")
    return (
        module._balanced_target_shape,
        module._crop_to_supported_aspect_ratio,
        module._load_rgb_image,
    )


def _binarize_mask(mask_like: Path | str | Image.Image) -> Image.Image:
    if isinstance(mask_like, Image.Image):
        mask = mask_like
    else:
        mask = Image.open(mask_like)
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = np.any(mask_np > 0, axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def _crop_mask_to_supported_aspect_ratio(mask: Image.Image, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0) -> Image.Image:
    width, height = mask.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return mask.crop((left, 0, left + crop_width, height))

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return mask.crop((0, top, width, top + crop_height))

    return mask


def preprocess_image_and_mask_ref(
    image_path: Path | str,
    mask_ref: Path | str | Image.Image,
    *,
    image_resolution: int = VGGT_OMEGA_TARGET_SIZE,
    patch_size: int = VGGT_OMEGA_PATCH_SIZE,
    preprocess_mode: str = "native",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if preprocess_mode == "aspect_pad":
        image = Image.open(image_path).convert("RGB")
        mask = _binarize_mask(mask_ref)
        image_arr, mask_arr = aspect_pad_arrays(
            image,
            mask,
            target_size=image_resolution,
            patch_size=patch_size,
        )
    elif preprocess_mode == "native":
        balanced_target_shape, crop_to_supported_aspect_ratio, load_rgb_image = _load_load_fn_helpers()

        image = crop_to_supported_aspect_ratio(load_rgb_image(image_path))
        mask = _crop_mask_to_supported_aspect_ratio(_binarize_mask(mask_ref))

        width, height = image.size
        aspect_ratio = height / max(width, 1)
        target_h, target_w = balanced_target_shape(aspect_ratio, image_resolution, patch_size)

        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        mask = mask.resize((target_w, target_h), Image.Resampling.NEAREST)
        image_arr = np.array(image, dtype=np.uint8, copy=True)
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    image_tensor = torch.from_numpy(image_arr).float().permute(2, 0, 1) / 255.0
    mask_tensor = torch.from_numpy(mask_arr).float()
    return image_tensor, mask_tensor


def _sorted_cached_layer_indices(model: VGGTOmega) -> List[int]:
    cached = getattr(model.aggregator, "cached_layer_indices", None)
    if cached is None:
        return list(VGGT_OMEGA_CACHED_LAYER_INDICES)
    return sorted(int(idx) for idx in cached)


def _slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])


def _run_aggregator_all_layers(model: VGGTOmega, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
    aggregator = model.aggregator
    batch_size, num_frames, num_channels, height, width = images.shape
    if num_channels != 3:
        raise ValueError(f"Expected 3 input channels, got {num_channels}")

    normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
    normalized = normalized.view(batch_size * num_frames, num_channels, height, width)

    camera_token = _slice_expand_and_flatten(aggregator.camera_token, batch_size, num_frames)
    register_token = _slice_expand_and_flatten(aggregator.register_token, batch_size, num_frames)

    patch_tokens = aggregator.patch_embed(normalized)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    _, num_tokens, embed_dim = tokens.shape

    patch_grid_size = (height // aggregator.patch_size, width // aggregator.patch_size)
    with torch.no_grad():
        rope_sin, rope_cos = aggregator.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
        frame_rope = (
            rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
            rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
        )

    outputs: List[torch.Tensor] = []
    for block_idx in range(aggregator.depth):
        tokens, frame_tokens = aggregator._run_frame_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            frame_rope,
        )
        tokens = aggregator._run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            aggregator.inter_frame_attention_types[block_idx],
        )
        outputs.append(torch.cat([frame_tokens, tokens], dim=-1))

    return outputs, aggregator.patch_token_start


def build_feature_key_order(model: VGGTOmega, token_sources: Sequence[str]) -> List[str]:
    keys: List[str] = []

    if "alt_attn" in token_sources:
        for idx in range(model.aggregator.depth):
            keys.append(f"inter_frame_block_{idx}")
        for idx in range(model.aggregator.depth):
            keys.append(f"intra_frame_block_{idx}")

    if "cam_head" in token_sources:
        if model.camera_head is None:
            raise RuntimeError("cam_head requested but model.camera_head is disabled")
        for idx in range(len(model.camera_head.trunk)):
            keys.append(f"cam_head_{idx}")

    if "depth_head" in token_sources:
        if model.dense_head is None:
            raise RuntimeError("depth_head requested but model.dense_head is disabled")
        for idx in range(len(model.dense_head.intermediate_layer_idx)):
            keys.append(f"depth_head_{idx}")

    return keys


def _extract_cam_head_block_tokens(
    model: VGGTOmega,
    final_tokens: torch.Tensor,
    patch_start_idx: int,
) -> Dict[str, torch.Tensor]:
    if model.camera_head is None:
        raise RuntimeError("camera_head is not enabled")

    out: Dict[str, torch.Tensor] = {}
    batch_size, num_frames, num_tokens, _ = final_tokens.shape
    if patch_start_idx > num_tokens:
        raise ValueError(f"patch_start_idx {patch_start_idx} exceeds num_tokens {num_tokens}")

    x = final_tokens
    if x.dtype != torch.float32:
        x = x.float()
    camera_and_register_tokens = x[:, :, :patch_start_idx]
    camera_and_register_tokens = model.camera_head.token_norm(camera_and_register_tokens)
    camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_start_idx, -1)
    patch_count = final_tokens.shape[2] - patch_start_idx

    for idx, block in enumerate(model.camera_head.trunk):
        camera_and_register_tokens = block(camera_and_register_tokens, None)
        reshaped = camera_and_register_tokens.reshape(batch_size, num_frames, patch_start_idx, -1)
        camera_tokens = reshaped[:, :, 0].unsqueeze(2).expand(-1, -1, patch_count, -1).contiguous()
        out[f"cam_head_{idx}"] = camera_tokens
    return out


def _extract_dense_head_block_tokens(
    model: VGGTOmega,
    aggregated_tokens_list: Sequence[torch.Tensor | None],
    patch_start_idx: int,
    prefix: str = "depth_head",
) -> Dict[str, torch.Tensor]:
    if model.dense_head is None:
        raise RuntimeError("dense_head is not enabled")

    out: Dict[str, torch.Tensor] = {}
    for output_idx, layer_idx in enumerate(model.dense_head.intermediate_layer_idx):
        x = aggregated_tokens_list[layer_idx]
        if x is None:
            raise ValueError(f"Aggregator did not cache layer {layer_idx}, which DenseHead needs.")
        x = x[:, :, patch_start_idx:]
        if x.dtype != torch.float32:
            x = x.float()
        batch_size, num_frames, patch_count, channels = x.shape
        x = model.dense_head.norm(x.reshape(batch_size * num_frames, patch_count, channels)).reshape(
            batch_size,
            num_frames,
            patch_count,
            channels,
        )
        out[f"{prefix}_{output_idx}"] = x
    return out


def extract_feature_map(
    model: VGGTOmega,
    images_b: torch.Tensor,
    token_sources: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], int]:
    feature_map: Dict[str, torch.Tensor] = {}
    aggregated_tokens_list, patch_start_idx = _run_aggregator_all_layers(model, images_b)
    embed_dim = int(aggregated_tokens_list[0].shape[-1] // 2)

    if "alt_attn" in token_sources:
        for layer_idx in range(len(aggregated_tokens_list)):
            combined = aggregated_tokens_list[layer_idx]
            frame_tokens = combined[..., :embed_dim]
            inter_tokens = combined[..., embed_dim:]
            feature_map[f"inter_frame_block_{layer_idx}"] = inter_tokens[:, :, patch_start_idx:]
            feature_map[f"intra_frame_block_{layer_idx}"] = frame_tokens[:, :, patch_start_idx:]

    if "cam_head" in token_sources:
        final_tokens = aggregated_tokens_list[-1]
        feature_map.update(_extract_cam_head_block_tokens(model, final_tokens=final_tokens, patch_start_idx=patch_start_idx))

    if "depth_head" in token_sources:
        feature_map.update(_extract_dense_head_block_tokens(model, aggregated_tokens_list, patch_start_idx=patch_start_idx))

    return feature_map, patch_start_idx


def load_vggt_omega(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    enable_camera: bool = True,
    enable_depth: bool = True,
) -> VGGTOmega:
    model = VGGTOmega(enable_camera=enable_camera, enable_depth=enable_depth, enable_alignment=False).to(device)
    state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        stripped = {}
        for key, value in state_dict.items():
            stripped[key[7:] if key.startswith("module.") else key] = value
        state_dict = stripped
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def dense_conf_to_patch_tokens(predictions: Dict[str, torch.Tensor], patch_size: int = 1) -> Dict[str, torch.Tensor]:
    outputs: Dict[str, torch.Tensor] = {}
    if "depth_conf" in predictions:
        depth_conf = predictions["depth_conf"].unsqueeze(2)
        outputs["depth_conf_tokens"] = map_to_patch_tokens(depth_conf, patch_size=patch_size)
    return outputs
