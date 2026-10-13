#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.legacy.pi3.build_davis_isdynamic_npz import (  # noqa: E402
    PI3_TARGET_SIZE,
    compute_resize_dims,
    load_sequence_paths,
    preprocess_image_and_mask,
    resolve_davis_root,
)
from isdynamic.legacy.pi3.build_youtubevos_isdynamic_npz import (  # noqa: E402
    load_fbms_sequence_paths,
    load_segtrack_sequence_paths,
    preprocess_image_and_mask_ref,
    resolve_fbms_root,
    resolve_segtrack_root,
)
from isdynamic.models.pi3 import (  # noqa: E402
    GROUP_CHOICES,
    MODEL_CHOICES,
    PI3_PATCH_SIZE,
    extract_feature_map,
    load_pi3_family_model,
)
from isdynamic.core.common_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.viz.scoremap_export_common import (  # noqa: E402
    build_scoremap_meta,
    build_scoremap_output_dir,
    load_scoremap_sequence_context,
    save_scoremap_meta,
)
from isdynamic.viz.motion_mask_benchmarks import load_binary_mask  # noqa: E402
from isdynamic.viz.figure_io import save_raster_png_pdf_figure, save_png_pdf_figure  # noqa: E402
from isdynamic.viz.scoremap_utils import (  # noqa: E402
    average_block_maps,
    build_scoremap_title,
    choose_full_grid,
    crop_to_content_bbox,
    display_block_label,
    expand_patch_map,
    load_probe_from_run,
    score_map_for_query,
    upsample_patch_map,
)

GROUP_TO_SOURCE = {
    "inter_frame_block": "alt_attn",
    "intra_frame_block": "alt_attn",
    "cam_head": "cam_head",
    "conf_head": "conf_head",
    "point_head": "point_head",
}
HEAD_GROUPS = {"cam_head", "conf_head", "point_head"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image, per-block IsDynamic score-map visualization (Figure-5 style)."
    )
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "segtrackv2", "fbms59"])
    parser.add_argument("--benchmark_name", type=str, default=None)
    parser.add_argument("--davis_root", type=str, default=None)
    parser.add_argument("--segtrack_root", type=str, default=None)
    parser.add_argument("--fbms_root", type=str, default=None)
    parser.add_argument("--fbms_split", type=str, default="Testset", choices=["Trainingset", "Testset"])
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument("--context_stride", type=int, default=1)
    parser.add_argument("--model", type=str, default="pi3x", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--probe_root", type=str, required=True, help="e.g. outputs/isdynamic_blockwise/pair_quadratic")
    parser.add_argument("--group", type=str, default="inter_frame_block", choices=GROUP_CHOICES)
    parser.add_argument(
        "--query_modes",
        type=str,
        default="dynamic",
        help="Comma list from {dynamic,static}.",
    )
    parser.add_argument(
        "--query_indices",
        type=str,
        default=None,
        help="Optional comma-separated token indices to use as queries.",
    )
    parser.add_argument("--max_blocks", type=int, default=0, help="Limit number of blocks to visualize (0=all).")
    parser.add_argument("--rows", type=int, default=3, help="Rows per visualization grid.")
    parser.add_argument("--cols", type=int, default=6, help="Columns per visualization grid.")
    parser.add_argument(
        "--head_single_column",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a single-row layout for head groups (cam/conf/point).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="outputs/isdynamic_visualization")
    parser.add_argument("--batch_size", type=int, default=8192, help="Pairwise scoring batch size.")
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert probabilities for query token idx=0 when using auto-selected query modes.",
    )
    parser.add_argument(
        "--use_multimodal_pi3x",
        action="store_true",
        help="Enable Pi3X multimodal branch while loading checkpoints.",
    )
    parser.add_argument(
        "--save_upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save bilinearly upsampled score-map outputs alongside the original patch-expanded heatmaps.",
    )
    parser.add_argument(
        "--hide_global_title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress the top figure title while keeping the query/block titles.",
    )
    return parser.parse_args()


def _load_frame_tensors(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, str]:
    dataset = str(args.dataset).lower()
    if dataset == "davis":
        if not args.davis_root:
            raise ValueError("--davis_root is required when --dataset davis.")
        davis_root = resolve_davis_root(args.davis_root)
        image_paths, mask_paths = load_sequence_paths(davis_root, args.sequence)
        if args.frame_idx < 0 or args.frame_idx >= len(image_paths):
            raise IndexError(f"frame_idx out of range: {args.frame_idx} for sequence {args.sequence} ({len(image_paths)} frames)")
        image_path = image_paths[args.frame_idx]
        mask_ref = mask_paths[args.frame_idx]
        image_t, mask_t = preprocess_image_and_mask(image_path, mask_ref)
        sequence_tag = args.sequence
    elif dataset == "segtrackv2":
        if not args.segtrack_root:
            raise ValueError("--segtrack_root is required when --dataset segtrackv2.")
        segtrack_root = resolve_segtrack_root(args.segtrack_root)
        image_paths, mask_refs = load_segtrack_sequence_paths(segtrack_root, args.sequence)
        if args.frame_idx < 0 or args.frame_idx >= len(image_paths):
            raise IndexError(f"frame_idx out of range: {args.frame_idx} for sequence {args.sequence} ({len(image_paths)} frames)")
        image_path = image_paths[args.frame_idx]
        mask_ref = mask_refs[args.frame_idx]
        image_t, mask_t = preprocess_image_and_mask_ref(image_path, mask_ref)
        sequence_tag = args.sequence
    elif dataset == "fbms59":
        if not args.fbms_root:
            raise ValueError("--fbms_root is required when --dataset fbms59.")
        fbms_root = resolve_fbms_root(args.fbms_root)
        image_paths, mask_refs = load_fbms_sequence_paths(
            fbms_root,
            args.fbms_split,
            args.sequence,
            include_unannotated=False,
        )
        if args.frame_idx < 0 or args.frame_idx >= len(image_paths):
            raise IndexError(
                f"frame_idx out of range: {args.frame_idx} for sequence {args.fbms_split}/{args.sequence} ({len(image_paths)} frames)"
            )
        image_path = image_paths[args.frame_idx]
        mask_ref = mask_refs[args.frame_idx]
        image_t, mask_t = preprocess_image_and_mask_ref(image_path, mask_ref)
        sequence_tag = f"{args.fbms_split}__{args.sequence}"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    image_np = (image_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask_np = mask_t.cpu().numpy().astype(np.float32)
    return image_t, mask_t, image_np, mask_np, sequence_tag


def _load_context_tensors(
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], torch.Tensor, np.ndarray, np.ndarray, str, List[int], List[Path], object]:
    spec, image_paths, mask_refs, sequence_tag, input_frame_indices = load_scoremap_sequence_context(
        dataset=args.dataset,
        benchmark_name=args.benchmark_name,
        sequence=args.sequence,
        frame_idx=args.frame_idx,
        context_size=args.context_size,
        context_stride=args.context_stride,
        davis_root=args.davis_root or "",
        segtrack_root=args.segtrack_root or "",
        fbms_root=args.fbms_root or "",
        fbms_split=args.fbms_split,
    )
    image_tensors: List[torch.Tensor] = []
    target_mask_t = None
    target_mask_ref = None
    for input_pos, frame_idx in enumerate(input_frame_indices):
        image_t, mask_t = preprocess_image_and_mask_ref(image_paths[frame_idx], mask_refs[frame_idx])
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
            target_mask_ref = mask_refs[frame_idx]
    if target_mask_t is None:
        raise RuntimeError("Missing target mask after preprocessing.")

    image_np = (image_tensors[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask_np = target_mask_t.cpu().numpy().astype(np.float32)
    return image_tensors, target_mask_t, image_np, mask_np, sequence_tag, input_frame_indices, image_paths, target_mask_ref


def _build_left_panel_canvas(
    image_path: Path,
    mask_ref,
) -> Tuple[np.ndarray, np.ndarray, float]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    orig_h, orig_w = image_np.shape[:2]

    binary_mask = load_binary_mask(mask_ref, fallback_size=(orig_w, orig_h)).astype(np.uint8) * 255

    new_w, new_h = compute_resize_dims(orig_w, orig_h)
    if orig_w >= orig_h:
        scale = orig_w / float(new_w)
    else:
        scale = orig_h / float(new_h)

    canvas_side = int(round(PI3_TARGET_SIZE * scale))
    content_w = int(round(new_w * scale))
    content_h = int(round(new_h * scale))
    x0 = (canvas_side - content_w) // 2
    y0 = (canvas_side - content_h) // 2

    image_resized = np.array(
        image.resize((content_w, content_h), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    mask_resized = np.array(
        Image.fromarray(binary_mask).resize((content_w, content_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )

    canvas_image = np.full((canvas_side, canvas_side, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((canvas_side, canvas_side), dtype=np.float32)
    canvas_image[y0 : y0 + content_h, x0 : x0 + content_w] = image_resized
    canvas_mask[y0 : y0 + content_h, x0 : x0 + content_w] = (mask_resized > 0).astype(np.float32)
    return canvas_image, canvas_mask, float(scale)


def _parse_feature_idx(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected feature name '{name}' for prefix '{prefix}'")
    return int(name[len(prefix) :])


def _uniform_subsample_keys(keys: List[str], limit: int) -> List[str]:
    if limit <= 0 or len(keys) <= limit:
        return list(keys)
    if limit == 1:
        return [keys[len(keys) // 2]]
    positions = [round(i * (len(keys) - 1) / (limit - 1)) for i in range(limit)]
    seen = set()
    sampled: List[str] = []
    for pos in positions:
        idx = int(pos)
        if idx in seen:
            continue
        seen.add(idx)
        sampled.append(keys[idx])
    if len(sampled) < limit:
        for idx, key in enumerate(keys):
            if idx in seen:
                continue
            sampled.append(key)
            if len(sampled) == limit:
                break
    return sampled


def discover_run_dirs(probe_root: Path, group: str) -> List[Tuple[str, Path]]:
    group_dir = probe_root / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Probe group directory not found: {group_dir}")

    prefix = f"{group}_"
    run_dirs = [d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    run_dirs.sort(key=lambda p: _parse_feature_idx(p.name, prefix))
    return [(d.name, d) for d in run_dirs]


def _resolve_probe_input_dim(model: torch.nn.Module) -> int:
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


def _sorted_feature_keys(feature_map: Dict[str, torch.Tensor], group: str) -> List[str]:
    prefix = f"{group}_"
    keys = [k for k in feature_map.keys() if k.startswith(prefix)]
    keys.sort(key=lambda k: _parse_feature_idx(k, prefix))
    return keys


def extract_frame_feature_map(
    model_name: str,
    model: torch.nn.Module,
    image_t: torch.Tensor,
    group: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    source = GROUP_TO_SOURCE[group]
    with torch.no_grad():
        if image_t.ndim == 3:
            images_b = image_t.unsqueeze(0).unsqueeze(0).to(device)
        elif image_t.ndim == 4:
            images_b = image_t.unsqueeze(0).to(device)
        else:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(image_t.shape)}")
        feature_map = extract_feature_map(
            model_name=model_name,
            model=model,
            images_b=images_b,
            token_sources=[source],
        )

    out: Dict[str, torch.Tensor] = {}
    for k in _sorted_feature_keys(feature_map, group):
        out[k] = feature_map[k][0, 0].detach().to(device)
    return out


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


def _left_panel_width_ratio(image_np: np.ndarray, rows: int, min_ratio: float = 3.2) -> float:
    height, width = image_np.shape[:2]
    aspect = width / max(height, 1)
    return max(min_ratio, 1.08 * rows * aspect)


def _alpha_blend_rgb(
    base_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    alpha: np.ndarray | float,
) -> np.ndarray:
    base = base_rgb.astype(np.float32) / 255.0
    overlay = overlay_rgb.astype(np.float32)
    if overlay.max() > 1.0:
        overlay = overlay / 255.0
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim == 0:
        alpha_arr = np.full(base.shape[:2], float(alpha_arr), dtype=np.float32)
    alpha_arr = np.clip(alpha_arr, 0.0, 1.0)[..., None]
    blended = base * (1.0 - alpha_arr) + overlay * alpha_arr
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def _grayscale_rgb(image_np: np.ndarray) -> np.ndarray:
    gray = np.dot(image_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return np.repeat(gray[..., None], 3, axis=-1).clip(0.0, 255.0).astype(np.uint8)


def _build_context_overlay(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    jet_min_rgb: np.ndarray,
    jet_max_rgb: np.ndarray,
) -> np.ndarray:
    tinted_gray = _alpha_blend_rgb(_grayscale_rgb(image_np), jet_min_rgb, 0.24)
    return _alpha_blend_rgb(tinted_gray, jet_max_rgb, 0.58 * (mask_np > 0.5).astype(np.float32))


def _save_labeled_grid(
    tile_rgbs: List[np.ndarray],
    block_keys: List[str],
    *,
    rows: int,
    cols: int,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tile_h, tile_w = tile_rgbs[0].shape[:2]
    grid_cell_w = 2.2
    grid_cell_h = grid_cell_w * (tile_h / max(tile_w, 1)) + 0.34
    fig, axes = plt.subplots(rows, cols, figsize=(cols * grid_cell_w, rows * grid_cell_h), squeeze=False)
    axes_flat = list(axes.reshape(-1))
    for ax, key, tile_rgb in zip(axes_flat, block_keys, tile_rgbs):
        ax.imshow(tile_rgb, interpolation="bilinear")
        ax.text(
            0.5,
            1.02,
            display_block_label(key),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
            color=(0.0, 0.0, 0.0, 0.95),
            clip_on=False,
        )
        ax.axis("off")
    for ax in axes_flat[len(block_keys) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.98, wspace=0.02, hspace=0.12)
    save_png_pdf_figure(fig, out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _save_context_exports(
    *,
    out_path: Path,
    left_context_rgb: np.ndarray,
    left_panel_rgb: np.ndarray,
    qx: int,
    qy: int,
    patch_size: int,
    left_scale: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if out_path.stem.endswith("_upsampled"):
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.imshow(left_context_rgb)
    ax.add_patch(
        Rectangle(
            (qx * patch_size * left_scale, qy * patch_size * left_scale),
            patch_size * left_scale,
            patch_size * left_scale,
            edgecolor=(0.98, 0.98, 1.0, 0.95),
            facecolor=(0.72, 0.72, 0.88, 0.90),
            linewidth=1.8,
        )
    )
    ax.axis("off")
    fig.tight_layout()
    save_raster_png_pdf_figure(fig, out_path.with_name(out_path.stem + "_context.png"), dpi=dpi)
    plt.close(fig)

    fig_h = 4.5
    nopad_h, nopad_w = left_panel_rgb.shape[:2]
    fig_w = fig_h * (nopad_w / max(nopad_h, 1))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(left_panel_rgb)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_raster_png_pdf_figure(
        fig,
        out_path.with_name(out_path.stem + "_context_nopad.png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(fig)


def plot_block_maps(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    left_image_np: np.ndarray,
    left_mask_np: np.ndarray,
    left_scale: float,
    maps_by_block: Dict[str, np.ndarray],
    query_idx: int,
    out_path: Path,
    rows: int,
    cols: int,
    patch_size: int,
    heatmap_renderer,
    title: str = "Per-block IsDynamic score maps",
    show_global_title: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    block_keys = sorted(maps_by_block.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))
    n_blocks = len(block_keys)
    if n_blocks == 0:
        raise RuntimeError("No block maps to plot.")

    rows, cols = choose_full_grid(n_blocks=n_blocks, rows_hint=rows, cols_hint=cols)
    left_ratio = _left_panel_width_ratio(image_np=left_image_np, rows=rows)
    fig = plt.figure(figsize=((cols + left_ratio) * 2.2, rows * 2.4))
    gs = GridSpec(rows, cols + 1, figure=fig, width_ratios=[left_ratio] + [1.0] * cols)

    orig_h, orig_w = image_np.shape[:2]
    grid_w = orig_w // patch_size
    qy, qx = divmod(query_idx, grid_w)
    (y0, y1, x0, x1), image_np_cropped, [mask_np_cropped] = crop_to_content_bbox(image_np, mask_np)
    jet_cmap = matplotlib.colormaps["jet"]
    jet_min_rgb = np.asarray(jet_cmap(0.0), dtype=np.float32)[:3]
    jet_max_rgb = np.asarray(jet_cmap(1.0), dtype=np.float32)[:3]
    base_rgb = image_np_cropped

    (left_y0, _left_y1, left_x0, _left_x1), left_image_np_cropped, [left_mask_np_cropped] = crop_to_content_bbox(
        left_image_np,
        left_mask_np,
    )
    left_context_rgb = _build_context_overlay(
        left_image_np,
        left_mask_np,
        jet_min_rgb=jet_min_rgb,
        jet_max_rgb=jet_max_rgb,
    )
    left_panel_rgb = _build_context_overlay(
        left_image_np_cropped,
        left_mask_np_cropped,
        jet_min_rgb=jet_min_rgb,
        jet_max_rgb=jet_max_rgb,
    )

    tile_rgbs: List[np.ndarray] = []
    for key in block_keys:
        heat = heatmap_renderer(maps_by_block[key], (orig_h, orig_w))[y0:y1, x0:x1]
        heat_rgb = jet_cmap(np.clip(heat, 0.0, 1.0))[..., :3]
        tile_rgbs.append(_alpha_blend_rgb(base_rgb, heat_rgb, 0.58))

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(left_panel_rgb)
    rect = Rectangle(
        (qx * patch_size * left_scale, qy * patch_size * left_scale),
        patch_size * left_scale,
        patch_size * left_scale,
        edgecolor=(0.98, 0.98, 1.0, 0.95),
        facecolor=(0.72, 0.72, 0.88, 0.90),
        linewidth=1.8,
    )
    rect.set_xy(
        (
            qx * patch_size * left_scale - left_x0,
            qy * patch_size * left_scale - left_y0,
        )
    )
    ax0.add_patch(rect)
    ax0.axis("off")

    for i, (k, tile_rgb) in enumerate(zip(block_keys, tile_rgbs)):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c + 1])
        ax.imshow(tile_rgb, interpolation="bilinear")
        ax.text(
            0.5,
            1.02,
            display_block_label(k),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
            color=(0.0, 0.0, 0.0, 0.95),
            clip_on=False,
        )
        ax.axis("off")

    if show_global_title:
        fig.suptitle(title, fontsize=22)
        fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    else:
        fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_raster_png_pdf_figure(fig, out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    _save_labeled_grid(
        tile_rgbs,
        block_keys,
        rows=rows,
        cols=cols,
        out_path=out_path.with_name(out_path.stem + "_grid.png"),
        dpi=300,
    )
    _save_context_exports(
        out_path=out_path,
        left_context_rgb=left_context_rgb,
        left_panel_rgb=left_panel_rgb,
        qx=qx,
        qy=qy,
        patch_size=patch_size,
        left_scale=left_scale,
        dpi=300,
    )


def plot_average_map(
    image_np: np.ndarray,
    maps_by_block: Dict[str, np.ndarray],
    out_path: Path,
    heatmap_renderer,
    target_hw: Tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    averaged_map = average_block_maps(maps_by_block)
    heat = heatmap_renderer(averaged_map, target_hw)
    _, image_np, [heat] = crop_to_content_bbox(image_np, heat)

    h, w = heat.shape[:2]
    fig_w = 4.0 * (w / max(h, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image_np)
    ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_raster_png_pdf_figure(fig, out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_upsampled_maps_npz(
    maps_by_block: Dict[str, np.ndarray],
    out_path: Path,
    target_hw: Tuple[int, int],
) -> None:
    arrays = {k: upsample_patch_map(v, target_hw) for k, v in maps_by_block.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def main() -> None:
    args = parse_args()

    image_tensors, target_mask_t, image_np, mask_np, sequence_tag, input_frame_indices, image_paths, target_mask_ref = _load_context_tensors(args)
    left_image_np, left_mask_np, left_scale = _build_left_panel_canvas(image_paths[input_frame_indices[0]], target_mask_ref)
    ratio = mask_to_token_dynamic_ratio(target_mask_t.unsqueeze(0), patch_size=PI3_PATCH_SIZE).squeeze(0).cpu().numpy()

    query_indices: List[int] = []
    if args.query_indices:
        query_indices = [int(x.strip()) for x in args.query_indices.split(",") if x.strip()]
    query_modes = [x.strip() for x in args.query_modes.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=ratio, query_modes=query_modes, query_indices=query_indices)
    apply_query0_inversion = bool(args.invert_query_idx_zero)

    device = torch.device(args.device)
    model = load_pi3_family_model(
        model_name=args.model,
        checkpoint=Path(args.checkpoint),
        device=device,
        use_multimodal_pi3x=args.use_multimodal_pi3x,
    )
    feature_map = extract_frame_feature_map(
        model_name=args.model,
        model=model,
        image_t=torch.stack(image_tensors, dim=0),
        group=args.group,
        device=device,
    )

    run_items = discover_run_dirs(Path(args.probe_root), group=args.group)
    run_dir_by_key = {k: d for k, d in run_items}

    usable_keys = [k for k in feature_map.keys() if k in run_dir_by_key]
    usable_keys.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))

    if args.head_single_column and args.group in HEAD_GROUPS:
        if args.max_blocks > 0:
            usable_keys = _uniform_subsample_keys(usable_keys, args.max_blocks)
        plot_rows = 1
        plot_cols = max(1, len(usable_keys))
    else:
        plot_rows = max(1, args.rows)
        plot_cols = max(1, args.cols)
        grid_capacity = plot_rows * plot_cols
        effective_limit = grid_capacity if args.max_blocks <= 0 else min(args.max_blocks, grid_capacity)
        usable_keys = _uniform_subsample_keys(usable_keys, effective_limit)

    if not usable_keys:
        raise RuntimeError(
            f"No overlapping block keys between features and probe runs. "
            f"group={args.group}, probe_root={args.probe_root}"
        )

    h, w = image_np.shape[:2]
    grid_h = h // PI3_PATCH_SIZE
    grid_w = w // PI3_PATCH_SIZE
    if grid_h * grid_w != next(iter(feature_map.values())).shape[0]:
        raise ValueError("Token count does not match expected patch grid.")

    maps_by_name: Dict[str, Dict[str, np.ndarray]] = {name: {} for name, _ in queries}

    for k in usable_keys:
        feats = feature_map[k]
        probe = load_probe_from_run(run_dir_by_key[k], device=device)

        c_probe = _resolve_probe_input_dim(probe)
        if feats.shape[-1] != c_probe:
            raise ValueError(f"Feature dim mismatch for {k}: feature={feats.shape[-1]} probe={c_probe}")

        for name, q_idx in queries:
            probs = score_map_for_query(
                model=probe,
                token_features=feats,
                query_idx=q_idx,
                batch_size=args.batch_size,
            )
            if apply_query0_inversion and q_idx == 0:
                probs = 1.0 - probs
            maps_by_name[name][k] = probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)

    out_dir = build_scoremap_output_dir(
        out_root=args.out_dir,
        group_label=args.group,
        dataset=args.dataset,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    title = build_scoremap_title(args.probe_root, args.group)
    for name, q_idx in queries:
        fig_path = out_dir / f"{name}_query_{q_idx:04d}.png"
        plot_block_maps(
            image_np=image_np,
            mask_np=mask_np,
            left_image_np=left_image_np,
            left_mask_np=left_mask_np,
            left_scale=left_scale,
            maps_by_block=maps_by_name[name],
            query_idx=q_idx,
            out_path=fig_path,
            rows=plot_rows,
            cols=plot_cols,
            patch_size=PI3_PATCH_SIZE,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, PI3_PATCH_SIZE),
            title=title,
            show_global_title=not args.hide_global_title,
        )
        avg_fig_path = out_dir / f"{name}_query_{q_idx:04d}_avg.png"
        plot_average_map(
            image_np=image_np,
            maps_by_block=maps_by_name[name],
            out_path=avg_fig_path,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, PI3_PATCH_SIZE),
            target_hw=(h, w),
        )
        if args.save_upsampled:
            upsampled_fig_path = out_dir / f"{name}_query_{q_idx:04d}_upsampled.png"
            plot_block_maps(
                image_np=image_np,
                mask_np=mask_np,
                left_image_np=left_image_np,
                left_mask_np=left_mask_np,
                left_scale=left_scale,
                maps_by_block=maps_by_name[name],
                query_idx=q_idx,
                out_path=upsampled_fig_path,
                rows=plot_rows,
                cols=plot_cols,
                patch_size=PI3_PATCH_SIZE,
                heatmap_renderer=lambda score_map, _target_hw: upsample_patch_map(score_map, (h, w)),
                title=title,
                show_global_title=not args.hide_global_title,
            )
            avg_upsampled_fig_path = out_dir / f"{name}_query_{q_idx:04d}_avg_upsampled.png"
            plot_average_map(
                image_np=image_np,
                maps_by_block=maps_by_name[name],
                out_path=avg_upsampled_fig_path,
                heatmap_renderer=lambda score_map, _target_hw: upsample_patch_map(score_map, (h, w)),
                target_hw=(h, w),
            )
            save_upsampled_maps_npz(
                maps_by_block=maps_by_name[name],
                out_path=out_dir / f"{name}_query_{q_idx:04d}_upsampled_maps.npz",
                target_hw=(h, w),
            )

    meta = build_scoremap_meta(
        dataset=args.dataset,
        sequence=args.sequence,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
        group=args.group,
        probe_root=args.probe_root,
        usable_blocks=usable_keys,
        queries=queries,
        patch_size=PI3_PATCH_SIZE,
        save_upsampled=bool(args.save_upsampled),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        extra_meta={
            "fbms_split": args.fbms_split if args.dataset == "fbms59" else None,
            "model": args.model,
            "plot_rows": plot_rows,
            "plot_cols": plot_cols,
            "invert_when_query_idx_zero": apply_query0_inversion,
        },
    )
    save_scoremap_meta(out_dir, meta)

    print(f"Saved visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
