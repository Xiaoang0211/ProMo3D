from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from isdynamic.legacy.pi3.build_davis_isdynamic_npz import load_sequence_paths as load_davis_sequence_paths
from isdynamic.legacy.pi3.build_youtubevos_isdynamic_npz import load_segtrack_sequence_paths


EXAMPLE_SPECS = (
    {
        "stem": "cheeta",
        "dataset": "segtrackv2",
        "sequence": "cheetah",
        "frame_idx": 0,
    },
    {
        "stem": "f50_birdofparadise",
        "dataset": "segtrackv2",
        "sequence": "bird_of_paradise",
        "frame_idx": 50,
    },
    {
        "stem": "parachute",
        "dataset": "segtrackv2",
        "sequence": "parachute",
        "frame_idx": 0,
    },
    {
        "stem": "f144_worm",
        "dataset": "segtrackv2",
        "sequence": "worm",
        "frame_idx": 144,
    },
    {
        "stem": "f70_koala",
        "dataset": "davis",
        "sequence": "koala",
        "frame_idx": 70,
    },
    {
        "stem": "f70_rhino",
        "dataset": "davis",
        "sequence": "rhino",
        "frame_idx": 70,
    },
)


def resolve_sequence_image_paths(
    *,
    dataset: str,
    sequence: str,
    davis_root: str,
    segtrack_root: str,
) -> List[Path]:
    if dataset == "davis":
        image_paths, _mask_paths = load_davis_sequence_paths(Path(davis_root), sequence)
        return image_paths
    if dataset == "segtrackv2":
        image_paths, _mask_refs = load_segtrack_sequence_paths(Path(segtrack_root), sequence)
        return image_paths
    raise ValueError(f"Unsupported dataset for unseen examples: {dataset}")


def select_frame_window(
    image_paths: List[Path],
    target_frame_idx: int,
    window_size: int,
) -> Tuple[List[Path], int]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if target_frame_idx < 0 or target_frame_idx >= len(image_paths):
        raise IndexError(f"target_frame_idx={target_frame_idx} out of range for {len(image_paths)} frames")
    if len(image_paths) <= window_size:
        return list(image_paths), target_frame_idx

    half = window_size // 2
    start = max(0, target_frame_idx - half)
    end = min(len(image_paths), start + window_size)
    start = max(0, end - window_size)
    clipped = list(image_paths[start:end])
    local_idx = target_frame_idx - start
    return clipped, local_idx


def save_binary_mask_png(mask: np.ndarray, out_path: Path) -> None:
    mask_uint8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(out_path)


def save_heatmap_overlay_png(
    *,
    image_np: np.ndarray,
    score_map: np.ndarray,
    out_path: Path,
    alpha: float = 0.58,
    cmap_name: str = "jet",
) -> None:
    import matplotlib.pyplot as plt

    image = np.asarray(image_np)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    score = np.asarray(score_map, dtype=np.float32)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    height, width = score.shape[:2]
    fig_w = 4.0 * (width / max(height, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image)
    ax.imshow(score, cmap=cmap_name, vmin=0.0, vmax=1.0, alpha=alpha, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
