from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .figure_io import save_png_pdf_figure
from .scoremap_utils import average_block_maps, crop_to_content_bbox, upsample_patch_map


INSTANCE_PALETTE = np.array(
    [
        [255, 64, 64],
        [64, 200, 255],
        [80, 220, 120],
        [255, 190, 64],
        [180, 120, 255],
        [255, 120, 200],
        [120, 255, 220],
        [255, 255, 80],
    ],
    dtype=np.uint8,
)

def otsu_threshold(score_map: np.ndarray) -> float:
    values = np.asarray(score_map, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    values = np.clip(values, 0.0, 1.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return lo

    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    prob = hist / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    sigma_b2 = np.zeros_like(omega)
    valid = (omega > 0.0) & (omega < 1.0)
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / (omega[valid] * (1.0 - omega[valid]))
    return float(bin_centers[int(np.argmax(sigma_b2))])


def resolve_threshold(
    score_map: np.ndarray,
    threshold: float,
    threshold_mode: str,
    *,
    threshold_offset: float = 0.0,
) -> float:
    mode = str(threshold_mode).strip().lower()
    if mode == "fixed":
        return float(np.clip(float(threshold) + float(threshold_offset), 0.0, 1.0))
    if mode == "otsu":
        return float(np.clip(otsu_threshold(score_map) + float(threshold_offset), 0.0, 1.0))
    raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")


def binary_mask_from_score(
    score_map: np.ndarray,
    threshold: float,
    *,
    threshold_mode: str = "fixed",
    threshold_offset: float = 0.0,
) -> Tuple[np.ndarray, float]:
    threshold_value = resolve_threshold(
        score_map,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_offset=threshold_offset,
    )
    return (score_map > threshold_value).astype(np.uint8), float(threshold_value)


def connected_components_4(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    mask = binary_mask.astype(bool)
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            next_label += 1
            labels[y, x] = next_label
            queue = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_label
                        queue.append((ny, nx))

    return labels, next_label


def remove_small_connected_components(binary_mask: np.ndarray, min_component_area: int) -> np.ndarray:
    min_component_area = int(min_component_area)
    if min_component_area <= 0:
        return (np.asarray(binary_mask).astype(np.uint8) > 0).astype(np.uint8)

    labels, num_instances = connected_components_4(binary_mask)
    if num_instances <= 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    counts = np.bincount(labels.reshape(-1), minlength=num_instances + 1)
    keep = counts >= min_component_area
    keep[0] = False
    filtered = keep[labels]
    return filtered.astype(np.uint8)


def fill_small_holes(binary_mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    max_hole_area = int(max_hole_area)
    mask = (np.asarray(binary_mask).astype(np.uint8) > 0).astype(np.uint8)
    if max_hole_area <= 0:
        return mask

    inverse = (1 - mask).astype(np.uint8)
    labels, num_instances = connected_components_4(inverse)
    if num_instances <= 0:
        return mask

    height, width = mask.shape
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[height - 1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, width - 1].tolist())

    counts = np.bincount(labels.reshape(-1), minlength=num_instances + 1)
    filled = mask.copy()
    for label in range(1, num_instances + 1):
        if label in border_labels:
            continue
        if counts[label] <= max_hole_area:
            filled[labels == label] = 1
    return filled.astype(np.uint8)


def filter_instances_by_mean_score(
    binary_mask: np.ndarray,
    score_map: np.ndarray,
    mean_score_threshold: float,
) -> np.ndarray:
    threshold = float(mean_score_threshold)
    mask = (np.asarray(binary_mask).astype(np.uint8) > 0).astype(np.uint8)
    if threshold <= 0.0:
        return mask

    score = np.asarray(score_map, dtype=np.float32)
    if score.shape != mask.shape:
        raise ValueError(f"Score/mask size mismatch: {score.shape} vs {mask.shape}")

    labels, num_instances = connected_components_4(mask)
    if num_instances <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_instances + 1):
        instance = labels == label
        if not np.any(instance):
            continue
        mean_score = float(score[instance].mean())
        if mean_score >= threshold:
            filtered[instance] = 1
    return filtered.astype(np.uint8)


def erode_binary_mask(binary_mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = int(iterations)
    mask = (np.asarray(binary_mask).astype(np.uint8) > 0).astype(np.uint8)
    if iterations <= 0:
        return mask

    for _ in range(iterations):
        if mask.size == 0:
            return mask
        eroded = np.zeros_like(mask, dtype=np.uint8)
        if mask.shape[0] >= 3 and mask.shape[1] >= 3:
            center = mask[1:-1, 1:-1]
            up = mask[:-2, 1:-1]
            down = mask[2:, 1:-1]
            left = mask[1:-1, :-2]
            right = mask[1:-1, 2:]
            eroded[1:-1, 1:-1] = (center & up & down & left & right).astype(np.uint8)
        mask = eroded
    return mask.astype(np.uint8)


def colorize_instances(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    labels, num_instances = connected_components_4(binary_mask)
    canvas = np.zeros(binary_mask.shape + (3,), dtype=np.uint8)
    for label in range(1, num_instances + 1):
        color = INSTANCE_PALETTE[(label - 1) % len(INSTANCE_PALETTE)]
        canvas[labels == label] = color
    return canvas, num_instances


def chw_tensor_to_uint8_image(image_t) -> np.ndarray:
    image_np = image_t.detach().permute(1, 2, 0).cpu().numpy()
    return np.clip(image_np * 255.0, 0, 255).astype(np.uint8)


def add_frame_paths_to_meta(
    meta: Dict[str, object],
    *,
    image_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    ref_image_path: str | Path | None = None,
) -> Dict[str, object]:
    enriched = dict(meta)
    if image_path is not None:
        enriched["image_path"] = str(image_path)
    if mask_path is not None:
        enriched["mask_path"] = str(mask_path)
    if ref_image_path is not None:
        enriched["ref_image_path"] = str(ref_image_path)
    return enriched


def synchronize_device(device) -> None:
    device_type = getattr(device, "type", device)
    if str(device_type).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _save_input_image(out_dir: Path, image_np: np.ndarray | None) -> None:
    if image_np is None:
        return

    image_uint8 = np.asarray(image_np)
    if image_uint8.ndim != 3:
        raise ValueError(f"Expected image_np to have shape [H,W,3], got {image_uint8.shape}")
    if image_uint8.dtype != np.uint8:
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)

    try:
        from PIL import Image

        Image.fromarray(image_uint8).save(out_dir / "input_image.png")
    except ImportError:
        pass


def _overlay_binary_mask_on_image(
    image_np: np.ndarray,
    binary_mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    image_uint8 = np.asarray(image_np)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_uint8.shape}")
    if image_uint8.dtype != np.uint8:
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
    mask = np.asarray(binary_mask).astype(bool)
    if mask.shape != image_uint8.shape[:2]:
        raise ValueError(f"Mask/image size mismatch: {mask.shape} vs {image_uint8.shape[:2]}")
    out = image_uint8.astype(np.float32).copy()
    overlay_color = np.array(color, dtype=np.float32)[None, None, :]
    out[mask] = (1.0 - alpha) * out[mask] + alpha * overlay_color[0, 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_overlay_image(
    out_dir: Path,
    stem: str,
    image_np: np.ndarray | None,
    binary_mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 64, 64),
) -> None:
    if image_np is None:
        return
    try:
        from PIL import Image

        overlay = _overlay_binary_mask_on_image(image_np=image_np, binary_mask=binary_mask, color=color)
        Image.fromarray(overlay).save(out_dir / f"{stem}.png")
    except ImportError:
        pass


def _save_binary_mask_artifacts(
    out_dir: Path,
    stem: str,
    binary_mask: np.ndarray,
    image_np: np.ndarray | None,
    *,
    overlay_color: Tuple[int, int, int] = (255, 64, 64),
) -> None:
    binary_uint8 = (np.asarray(binary_mask).astype(np.uint8) > 0).astype(np.uint8)
    binary_rgb = np.repeat((binary_uint8 * 255)[..., None], 3, axis=2)
    np.save(out_dir / f"{stem}.npy", binary_uint8)
    _save_overlay_image(out_dir, f"{stem}_overlay", image_np, binary_uint8, color=overlay_color)
    try:
        from PIL import Image

        Image.fromarray(binary_rgb).save(out_dir / f"{stem}.png")
    except ImportError:
        pass


def _save_score_heatmap_figure(
    out_dir: Path,
    stem: str,
    score_map: np.ndarray,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return

    score = np.asarray(score_map, dtype=np.float32)
    fig_w = 5.6
    fig_h = max(2.8, fig_w * float(score.shape[0]) / max(float(score.shape[1]), 1.0))
    base_font_size = float(mpl.rcParams.get("font.size", 10.0))
    heatmap_font_size = base_font_size + 4.0
    def _save_single_heatmap(*, with_colorbar: bool, suffix: str) -> None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(score, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if with_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=heatmap_font_size)
        save_png_pdf_figure(fig, out_dir / f"{stem}_{suffix}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": heatmap_font_size,
            "axes.labelsize": heatmap_font_size,
            "xtick.labelsize": heatmap_font_size,
            "ytick.labelsize": heatmap_font_size,
        }
    ):
        _save_single_heatmap(with_colorbar=True, suffix="heatmap")
        _save_single_heatmap(with_colorbar=False, suffix="heatmap_nobar")


def _crop_patch_score_to_content_bbox(
    patch_score: np.ndarray,
    crop_bbox: Tuple[int, int, int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    patch_h, patch_w = patch_score.shape[:2]
    y0, y1, x0, x1 = crop_bbox

    py0 = max(0, min(patch_h, int(np.floor((y0 * patch_h) / max(target_h, 1)))))
    py1 = max(py0 + 1, min(patch_h, int(np.ceil((y1 * patch_h) / max(target_h, 1)))))
    px0 = max(0, min(patch_w, int(np.floor((x0 * patch_w) / max(target_w, 1)))))
    px1 = max(px0 + 1, min(patch_w, int(np.ceil((x1 * patch_w) / max(target_w, 1)))))
    return patch_score[py0:py1, px0:px1]


def save_refined_motion_mask_outputs(
    out_dir: Path,
    score_map: np.ndarray,
    threshold: float,
    threshold_mode: str,
    connected_components: bool,
    meta: Dict[str, object],
    score_stem: str,
    binary_stem: str,
    instance_stem: str | None = None,
    image_np: np.ndarray | None = None,
    min_component_area: int = 0,
    max_hole_area: int = 0,
    threshold_offset: float = 0.0,
    instance_mean_score_threshold: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    score_map = score_map.astype(np.float32)
    inherited_bbox = meta.get("content_bbox_xyxy")
    inherited_hw = meta.get("content_hw")
    preserve_inherited_bbox = (
        isinstance(inherited_bbox, list)
        and len(inherited_bbox) == 4
        and isinstance(inherited_hw, list)
        and len(inherited_hw) == 2
        and image_np is not None
        and tuple(int(x) for x in inherited_hw) == tuple(np.asarray(image_np).shape[:2])
    )
    crop_bbox = None
    if image_np is not None and not preserve_inherited_bbox:
        crop_bbox, image_np, [score_map] = crop_to_content_bbox(image_np, score_map)
    _save_input_image(out_dir, image_np=image_np)

    binary_mask, threshold_value = binary_mask_from_score(
        score_map,
        threshold,
        threshold_mode=threshold_mode,
        threshold_offset=threshold_offset,
    )
    binary_mask = remove_small_connected_components(binary_mask, min_component_area=min_component_area)
    binary_mask = fill_small_holes(binary_mask, max_hole_area=max_hole_area)
    binary_mask = filter_instances_by_mean_score(
        binary_mask,
        score_map,
        mean_score_threshold=instance_mean_score_threshold,
    )
    binary_rgb = (binary_mask * 255).astype(np.uint8)
    binary_rgb = np.repeat(binary_rgb[..., None], 3, axis=2)
    score_uint8 = np.clip(score_map, 0.0, 1.0)
    score_uint8 = (score_uint8 * 255.0).astype(np.uint8)

    np.save(out_dir / f"{score_stem}.npy", score_map)
    np.save(out_dir / f"{binary_stem}.npy", binary_mask.astype(np.uint8))
    _save_score_heatmap_figure(out_dir, score_stem, score_map)
    _save_overlay_image(out_dir, f"{binary_stem}_overlay", image_np, binary_mask)

    try:
        from PIL import Image

        Image.fromarray(score_uint8).save(out_dir / f"{score_stem}.png")
        Image.fromarray(binary_rgb).save(out_dir / f"{binary_stem}.png")
        if connected_components:
            stem = instance_stem or f"{binary_stem}_instances"
            instance_rgb, num_instances = colorize_instances(binary_mask)
            Image.fromarray(instance_rgb).save(out_dir / f"{stem}.png")
            meta = {**meta, "num_instances": int(num_instances)}
    except ImportError:
        pass

    if crop_bbox is not None:
        y0, y1, x0, x1 = crop_bbox
        meta = {
            **meta,
            "content_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "content_hw": [int(y1 - y0), int(x1 - x0)],
        }

    meta = {
        **meta,
        "threshold_mode": str(threshold_mode),
        "threshold_value": float(threshold_value),
        "threshold_offset": float(threshold_offset),
        "min_component_area": int(min_component_area),
        "max_hole_area": int(max_hole_area),
        "instance_mean_score_threshold": float(instance_mean_score_threshold),
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def save_motion_mask_outputs(
    out_dir: Path,
    averaged_patch_score: np.ndarray,
    target_hw: Tuple[int, int],
    threshold: float,
    threshold_mode: str,
    connected_components: bool,
    meta: Dict[str, object],
    image_np: np.ndarray | None = None,
    gt_mask_np: np.ndarray | None = None,
    threshold_offset: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_only = os.environ.get("ISDYNAMIC_MASK_OUTPUT_MODE", "").strip().lower() == "metrics_only"

    upsampled_score = upsample_patch_map(averaged_patch_score, target_hw)
    crop_bbox = None
    if image_np is not None:
        maps_to_crop = [upsampled_score]
        if gt_mask_np is not None:
            maps_to_crop.append(np.asarray(gt_mask_np, dtype=np.float32))
        crop_bbox, image_np, cropped_maps = crop_to_content_bbox(image_np, *maps_to_crop)
        upsampled_score = cropped_maps[0]
        if gt_mask_np is not None:
            gt_mask_np = (cropped_maps[1] > 0.5).astype(np.uint8)
        averaged_patch_score = _crop_patch_score_to_content_bbox(
            patch_score=averaged_patch_score,
            crop_bbox=crop_bbox,
            target_hw=target_hw,
        )
    if not metrics_only:
        _save_input_image(out_dir, image_np=image_np)

    binary_mask, threshold_value = binary_mask_from_score(
        upsampled_score,
        threshold,
        threshold_mode=threshold_mode,
        threshold_offset=threshold_offset,
    )
    np.save(out_dir / "binary_mask_upsampled.npy", binary_mask.astype(np.uint8))
    if not metrics_only:
        binary_rgb = (binary_mask * 255).astype(np.uint8)
        binary_rgb = np.repeat(binary_rgb[..., None], 3, axis=2)
        np.save(out_dir / "avg_score_patch.npy", averaged_patch_score.astype(np.float32))
        np.save(out_dir / "avg_score_upsampled.npy", upsampled_score.astype(np.float32))
        _save_score_heatmap_figure(out_dir, "avg_score_upsampled", upsampled_score)
        _save_overlay_image(out_dir, "binary_mask_upsampled_overlay", image_np, binary_mask)

        try:
            from PIL import Image

            Image.fromarray(binary_rgb).save(out_dir / "binary_mask_upsampled.png")
            if connected_components:
                instance_rgb, num_instances = colorize_instances(binary_mask)
                Image.fromarray(instance_rgb).save(out_dir / "instance_mask_upsampled.png")
                meta = {**meta, "num_instances": int(num_instances)}
        except ImportError:
            pass

        if gt_mask_np is not None:
            _save_binary_mask_artifacts(
                out_dir=out_dir,
                stem="binary_mask_gt",
                binary_mask=gt_mask_np,
                image_np=image_np,
                overlay_color=(64, 160, 255),
            )

    if crop_bbox is not None:
        y0, y1, x0, x1 = crop_bbox
        meta = {
            **meta,
            "content_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "content_hw": [int(y1 - y0), int(x1 - x0)],
        }

    meta = {
        **meta,
        "threshold_mode": str(threshold_mode),
        "threshold_offset": float(threshold_offset),
        "threshold_value": float(threshold_value),
        "output_mode": "metrics_only" if metrics_only else "full",
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
