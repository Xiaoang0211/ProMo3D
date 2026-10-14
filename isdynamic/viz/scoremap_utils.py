from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .figure_io import save_png_pdf_figure

from ..core.common_models import build_probe, load_attention_matrix


def pick_queries(
    mask_token_ratio: np.ndarray,
    query_modes: List[str],
    query_indices: List[int],
) -> List[Tuple[str, int]]:
    flat = mask_token_ratio.reshape(-1)
    total = int(flat.shape[0])

    if query_indices:
        out = []
        for i, idx in enumerate(query_indices):
            if not (0 <= idx < total):
                raise ValueError(f"Query index out of range: {idx} not in [0,{total})")
            out.append((f"q{i}", int(idx)))
        return out

    out: List[Tuple[str, int]] = []
    for mode in query_modes:
        if mode == "dynamic":
            out.append(("dynamic", int(np.argmax(flat))))
        elif mode == "static":
            out.append(("static", int(np.argmin(flat))))
        else:
            raise ValueError(f"Unsupported query mode: {mode}")
    return out


def _load_probe_state(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    return obj


def resolve_probe_input_dim(model: torch.nn.Module) -> int:
    if hasattr(model, "w") and hasattr(model.w, "weight"):
        return int(model.w.weight.shape[-1])
    if hasattr(model, "diag"):
        return int(model.diag.shape[0])
    if hasattr(model, "a"):
        return int(model.a.shape[0])
    if hasattr(model, "u"):
        return int(model.u.shape[1])
    if hasattr(model, "attention_matrix"):
        return int(model.attention_matrix.shape[0])
    if hasattr(model, "value_matrix"):
        return int(model.value_matrix.shape[0])
    raise ValueError("Could not resolve probe input dimension.")


def load_probe_from_run(run_dir: Path, device: torch.device) -> torch.nn.Module:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing probe config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    model_cfg = cfg["model"]
    model_type = model_cfg["model_type"]
    input_dim = int(model_cfg["input_dim"])

    fixed_matrix = None
    if model_type in {
        "pair_attention",
        "pair_attention_biased_quadratic",
        "pair_attention_biased_lowrank_quadratic",
        "pair_attention_biased_lowrank_quadratic_strict",
    }:
        path = model_cfg.get("attention_matrix_path")
        if not path:
            raise ValueError(f"{model_type} run missing attention_matrix_path: {run_dir}")
        fixed_matrix = load_attention_matrix(path, key=model_cfg.get("attention_matrix_key"))
    elif model_type in {
        "pair_value",
        "pair_value_biased_quadratic",
        "pair_value_biased_lowrank_quadratic",
        "pair_value_biased_lowrank_quadratic_strict",
    }:
        path = model_cfg.get("value_matrix_path")
        if not path:
            raise ValueError(f"{model_type} run missing value_matrix_path: {run_dir}")
        fixed_matrix = load_attention_matrix(path, key=model_cfg.get("value_matrix_key"))

    model = build_probe(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        rank=int(model_cfg.get("rank", 256)),
        attention_matrix=fixed_matrix,
    ).to(device)

    save_name = cfg.get("output", {}).get("save_name", "best_probe.pt")
    ckpt_path = run_dir / save_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best_probe.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found in run dir: {run_dir}")

    state_dict = _load_probe_state(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if not bool(getattr(model, "is_pairwise", False)):
        raise ValueError(f"Run is not pairwise and cannot produce score maps: {run_dir}")
    return model


def score_map_for_query(
    model: torch.nn.Module,
    token_features: torch.Tensor,
    query_idx: int,
    batch_size: int,
) -> torch.Tensor:
    num_tokens = token_features.shape[0]
    query = token_features[query_idx : query_idx + 1]

    probs = []
    with torch.no_grad():
        for start in range(0, num_tokens, batch_size):
            end = min(start + batch_size, num_tokens)
            target = token_features[start:end]
            query_batch = query.expand(end - start, -1)
            logits = model(query_batch, target)
            probs.append(torch.sigmoid(logits))
    return torch.cat(probs, dim=0)


def expand_patch_map(score_map: np.ndarray, patch_size: int) -> np.ndarray:
    return np.kron(score_map, np.ones((patch_size, patch_size), dtype=np.float32))


def upsample_patch_map(score_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    tensor = torch.from_numpy(score_map).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return upsampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def average_block_maps(maps_by_block: Dict[str, np.ndarray]) -> np.ndarray:
    if not maps_by_block:
        raise RuntimeError("No block maps available to average.")
    stacked = np.stack(list(maps_by_block.values()), axis=0).astype(np.float32)
    return stacked.mean(axis=0)


def display_block_label(key: str) -> str:
    text = str(key)
    if "_" in text:
        suffix = text.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return f"Block {int(suffix)}"
    return text


def infer_probe_rank(probe_root: str | Path) -> Optional[int]:
    probe_root = Path(probe_root)
    text = str(probe_root)
    match = re.search(r"(?:^|/)rank_(\d+)(?:_|/|$)", text)
    if match:
        return int(match.group(1))
    try:
        for cfg_path in probe_root.rglob("config.json"):
            with cfg_path.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            model_cfg = cfg.get("model", {})
            rank = model_cfg.get("rank")
            if rank is not None:
                return int(rank)
            break
    except Exception:
        pass
    return None


def infer_training_video_count(probe_root: str | Path) -> Optional[int]:
    text = str(probe_root)
    for pattern in (
        r"data_starvation_random_(\d+)",
        r"data_starvation_full_(\d+)",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    if "youtubevos" in text.lower():
        return 160
    return None


def describe_scoremap_group(group: str, model_name: str = "") -> str:
    model_name = str(model_name or "").strip().lower()
    if group in {"depth_head", "dual_dpt_head"}:
        return "depth head"
    mapping = {
        "inter_frame_block": "Inter-Frame Blocks",
        "intra_frame_block": "Intra-Frame Blocks",
        "cam_head": "Camera Head Blocks",
        "conf_head": "Confidence Head Blocks",
        "point_head": "Point Head Blocks",
        "enc": "Encoder Blocks",
        "dec_view1": "Decoder View-1 Blocks",
        "dec_view2": "Decoder View-2 Blocks",
        "head_in_view1": "Head-in-View-1 Blocks",
        "head_in_view2": "Head-in-View-2 Blocks",
    }
    return mapping.get(group, group.replace("_", " ").title())


def build_scoremap_title(probe_root: str | Path, group: str) -> str:
    rank = infer_probe_rank(probe_root)
    videos = infer_training_video_count(probe_root)
    probe_text = str(probe_root).lower()
    model_name = ""
    for candidate in ("depth_anything_3", "da3", "vggt", "pi3x", "pi3", "map_anything", "dust3r"):
        if candidate in probe_text:
            model_name = candidate
            break
    first = "Low-rank Quad"
    if rank is not None:
        first += f" (k = {rank})"
    parts = [first]
    if videos is not None:
        parts.append(f"{videos} training videos (YouTube-VOS)")
    parts.append("Dynamic score maps")
    parts.append(describe_scoremap_group(group, model_name))
    return " | ".join(parts)


def _draw_query_panel(ax, image_np: np.ndarray, query_idx: int, patch_size: int) -> None:
    from matplotlib.patches import Rectangle

    height, width = image_np.shape[:2]
    grid_w = width // patch_size
    qy, qx = divmod(query_idx, grid_w)
    ax.imshow(image_np)
    rect = Rectangle(
        (qx * patch_size, qy * patch_size),
        patch_size,
        patch_size,
        edgecolor="white",
        facecolor="none",
        linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.set_title(f"Query idx={query_idx}", fontsize=16)
    ax.axis("off")


def choose_full_grid(n_blocks: int, rows_hint: int, cols_hint: int) -> Tuple[int, int]:
    rows_hint = max(1, rows_hint)
    cols_hint = max(1, cols_hint)
    target_aspect = 4.0 / 3.0
    hint_aspect = cols_hint / rows_hint
    candidates = []
    for rows in range(1, n_blocks + 1):
        cols = (n_blocks + rows - 1) // rows
        unused = (rows * cols) - n_blocks
        aspect = cols / rows
        score = (
            0 if cols >= rows else 1,
            abs(aspect - target_aspect),
            unused,
            abs(rows - rows_hint) + abs(cols - cols_hint),
            abs(aspect - hint_aspect),
            cols,
        )
        candidates.append((score, rows, cols))
    if not candidates:
        return rows_hint, cols_hint
    _, rows, cols = min(candidates, key=lambda item: item[0])
    return rows, cols


def _left_panel_width_ratio(image_np: np.ndarray, rows: int, min_ratio: float = 3.2) -> float:
    height, width = image_np.shape[:2]
    aspect = width / max(height, 1)
    # Make the left image wide enough that it can occupy the full stacked-grid
    # height while preserving aspect ratio, without introducing visible padding.
    return max(min_ratio, 1.08 * rows * aspect)


def infer_content_bbox_from_image(image_np: np.ndarray, tol: int = 0) -> Tuple[int, int, int, int]:
    image_np = np.asarray(image_np)
    if image_np.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape for content bbox: {image_np.shape}")

    height, width = image_np.shape[:2]

    def _allclose_row(row_idx: int, bg_value: np.ndarray) -> bool:
        row = image_np[row_idx]
        return bool(np.all(np.abs(row.astype(np.int16) - bg_value.astype(np.int16)) <= tol))

    def _allclose_col(col_idx: int, bg_value: np.ndarray) -> bool:
        col = image_np[:, col_idx]
        return bool(np.all(np.abs(col.astype(np.int16) - bg_value.astype(np.int16)) <= tol))

    if image_np.ndim == 2:
        corners = np.array(
            [image_np[0, 0], image_np[0, -1], image_np[-1, 0], image_np[-1, -1]],
            dtype=np.int16,
        )
    else:
        corners = np.stack(
            [image_np[0, 0], image_np[0, -1], image_np[-1, 0], image_np[-1, -1]],
            axis=0,
        ).astype(np.int16)

    bg_value = corners[0]
    if np.all(np.abs(corners - bg_value) <= tol):
        top = 0
        bottom = height
        left = 0
        right = width
        while top < bottom and _allclose_row(top, bg_value):
            top += 1
        while bottom > top and _allclose_row(bottom - 1, bg_value):
            bottom -= 1
        while left < right and _allclose_col(left, bg_value):
            left += 1
        while right > left and _allclose_col(right - 1, bg_value):
            right -= 1
        if top < bottom and left < right:
            return int(top), int(bottom), int(left), int(right)

    if image_np.ndim == 2:
        valid = image_np > tol
    else:
        valid = np.any(image_np > tol, axis=2)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, height, 0, width
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def crop_to_content_bbox(
    image_np: np.ndarray,
    *arrays: np.ndarray | None,
    tol: int = 0,
) -> Tuple[Tuple[int, int, int, int], np.ndarray, list[np.ndarray | None]]:
    y0, y1, x0, x1 = infer_content_bbox_from_image(image_np=image_np, tol=tol)
    cropped_image = image_np[y0:y1, x0:x1]
    cropped_arrays: list[np.ndarray | None] = []
    for array in arrays:
        if array is None:
            cropped_arrays.append(None)
            continue
        if array.shape[:2] != image_np.shape[:2]:
            cropped_arrays.append(array)
            continue
        cropped_arrays.append(array[y0:y1, x0:x1])
    return (y0, y1, x0, x1), cropped_image, cropped_arrays


def plot_block_maps(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    maps_by_block: Dict[str, np.ndarray],
    query_idx: int,
    out_path: Path,
    rows: int,
    cols: int,
    patch_size: int,
    heatmap_renderer: Callable[[np.ndarray, Tuple[int, int]], np.ndarray],
    title: str = "Per-block dynamic score maps",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    block_keys = list(maps_by_block.keys())
    n_blocks = len(block_keys)
    if n_blocks == 0:
        raise RuntimeError("No block maps to plot.")

    rows, cols = choose_full_grid(n_blocks=n_blocks, rows_hint=rows, cols_hint=cols)

    left_ratio = _left_panel_width_ratio(image_np=image_np, rows=rows)
    fig = plt.figure(figsize=((cols + left_ratio) * 2.2, rows * 2.4))
    gs = GridSpec(rows, cols + 1, figure=fig, width_ratios=[left_ratio] + [1.0] * cols)
    height, width = image_np.shape[:2]
    grid_w = width // patch_size
    qy, qx = divmod(query_idx, grid_w)
    gray = np.dot(image_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(gray, cmap="gray", vmin=0.0, vmax=255.0)
    ax0.imshow(mask_np, cmap="jet", alpha=0.22, vmin=0.0, vmax=1.0)
    rect = Rectangle(
        (qx * patch_size, qy * patch_size),
        patch_size,
        patch_size,
        edgecolor="white",
        facecolor="none",
        linewidth=1.8,
    )
    ax0.add_patch(rect)
    ax0.axis("off")

    for i, key in enumerate(block_keys):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c + 1])
        heat = heatmap_renderer(maps_by_block[key], (height, width))
        ax.imshow(gray, cmap="gray", vmin=0.0, vmax=255.0)
        ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
        rect = Rectangle(
            (qx * patch_size, qy * patch_size),
            patch_size,
            patch_size,
            edgecolor="white",
            facecolor="none",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.set_title(display_block_label(key), fontsize=14, pad=3)
        ax.axis("off")

    fig.suptitle(title, fontsize=22, y=0.957)
    fig.tight_layout(rect=[0, 0.0, 1, 0.948])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_pdf_figure(fig, out_path, dpi=220)
    plt.close(fig)

    fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    axes_grid_flat = list(axes_grid.reshape(-1))
    for ax, key in zip(axes_grid_flat, block_keys):
        heat = heatmap_renderer(maps_by_block[key], (height, width))
        ax.imshow(gray, cmap="gray", vmin=0.0, vmax=255.0)
        ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
        ax.set_title(display_block_label(key), fontsize=14, pad=3)
        ax.axis("off")
    for ax in axes_grid_flat[len(block_keys) :]:
        ax.axis("off")
    fig_grid.tight_layout(rect=[0, 0.0, 1, 0.985])
    save_png_pdf_figure(fig_grid, out_path.with_name(out_path.stem + "_grid.png"), dpi=220)
    plt.close(fig_grid)

    fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax2.imshow(gray, cmap="gray", vmin=0.0, vmax=255.0)
    ax2.imshow(mask_np, cmap="jet", alpha=0.25, vmin=0.0, vmax=1.0)
    rect = Rectangle(
        (qx * patch_size, qy * patch_size),
        patch_size,
        patch_size,
        edgecolor="white",
        facecolor="none",
        linewidth=1.8,
    )
    ax2.add_patch(rect)
    ax2.axis("off")
    fig2.tight_layout()
    if not out_path.stem.endswith("_upsampled"):
        save_png_pdf_figure(fig2, out_path.with_name(out_path.stem + "_context.png"), dpi=220)
    plt.close(fig2)


def plot_average_map(
    image_np: np.ndarray,
    maps_by_block: Dict[str, np.ndarray],
    out_path: Path,
    heatmap_renderer: Callable[[np.ndarray, Tuple[int, int]], np.ndarray],
    target_hw: Tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    averaged_map = average_block_maps(maps_by_block)
    heat = heatmap_renderer(averaged_map, target_hw)
    _, image_np, [heat] = crop_to_content_bbox(image_np, heat)

    height, width = heat.shape[:2]
    fig_w = 4.0 * (width / max(height, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image_np)
    ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_pdf_figure(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_upsampled_maps_npz(
    maps_by_block: Dict[str, np.ndarray],
    out_path: Path,
    target_hw: Tuple[int, int],
) -> None:
    arrays = {key: upsample_patch_map(value, target_hw) for key, value in maps_by_block.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
