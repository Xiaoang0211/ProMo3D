from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    return (mask > 0).astype(np.uint8)


def aspect_pad_arrays(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    *,
    target_size: int,
    patch_size: int,
    image_fill: int = 255,
) -> Tuple[np.ndarray, np.ndarray]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if target_size % patch_size != 0:
        raise ValueError(f"target_size={target_size} must be divisible by patch_size={patch_size}.")

    image_arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.any(mask_arr > 0, axis=-1)
    mask_arr = (mask_arr > 0).astype(np.uint8)

    height, width = image_arr.shape[:2]
    # Reserve a one-patch zero-mask border so spatial token 0 is static by
    # construction in the padding ablation.
    content_target = max(patch_size, target_size - 2 * patch_size)
    long_edge = max(height, width)
    scale = content_target / max(long_edge, 1)
    new_w = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    new_h = max(patch_size, int(round(height * scale / patch_size)) * patch_size)
    new_w = min(target_size, new_w)
    new_h = min(target_size, new_h)

    image_resized = Image.fromarray(image_arr).resize((new_w, new_h), Image.Resampling.BICUBIC)
    mask_resized = Image.fromarray(mask_arr).resize((new_w, new_h), Image.Resampling.NEAREST)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    padded_image = np.full((target_size, target_size, 3), image_fill, dtype=np.uint8)
    padded_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    padded_image[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = np.asarray(image_resized, dtype=np.uint8)
    padded_mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = (
        np.asarray(mask_resized, dtype=np.uint8) > 0
    ).astype(np.uint8)
    return padded_image, padded_mask
