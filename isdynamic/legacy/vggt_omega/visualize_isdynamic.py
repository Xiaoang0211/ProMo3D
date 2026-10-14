#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isdynamic.legacy.vggt_omega.build_davis_isdynamic_npz import (  # noqa: E402
    VGGT_PATCH_SIZE,
    _extract_state_dict,
    _strip_module_prefix,
    extract_feature_map,
)
from isdynamic.core.artifacts import artifact_path  # noqa: E402
from isdynamic.core.common_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.legacy.vggt_omega.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref  # noqa: E402
from isdynamic.models.vggt_omega import VGGTOmega, extract_feature_map as extract_vggt_omega_feature_map, load_vggt_omega  # noqa: E402
from isdynamic.viz.scoremap_export_common import (  # noqa: E402
    build_scoremap_meta,
    build_scoremap_output_dir,
    load_scoremap_sequence_context,
    save_scoremap_meta,
)
from isdynamic.viz.figure_io import save_png_pdf_figure  # noqa: E402
from isdynamic.viz.scoremap_utils import (  # noqa: E402
    build_scoremap_title,
    choose_full_grid,
    display_block_label,
    expand_patch_map,
    load_probe_from_run,
    plot_average_map,
    save_upsampled_maps_npz,
    score_map_for_query,
    upsample_patch_map,
)

GROUP_TO_SOURCE = {
    "inter_frame_block": "alt_attn",
    "intra_frame_block": "alt_attn",
    "cam_head": "cam_head",
    "depth_head": "depth_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image, per-block dynamic score-map visualization (Figure-5 style)."
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
    parser.add_argument("--vggt_checkpoint", type=str, required=True)
    parser.add_argument("--probe_root", type=str, required=True, help="e.g. outputs/isdynamic_blockwise/pair_quadratic")
    parser.add_argument(
        "--group",
        type=str,
        default="inter_frame_block",
        choices=["inter_frame_block", "intra_frame_block", "cam_head", "depth_head"],
    )
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
    parser.add_argument("--cols", type=int, default=8, help="Columns per visualization grid.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default=artifact_path("outputs", "isdynamic_visualization", model_name="vggt_omega"))
    parser.add_argument("--batch_size", type=int, default=8192, help="Pairwise scoring batch size.")
    parser.add_argument(
        "--save_upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save bilinearly upsampled score-map outputs alongside the original patch-expanded heatmaps.",
    )
    return parser.parse_args()


def _parse_feature_idx(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected feature name '{name}' for prefix '{prefix}'")
    return int(name[len(prefix) :])


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


def load_vggt(vggt_checkpoint: Path, group: str, device: torch.device) -> VGGTOmega:
    enable_camera = group == "cam_head"
    enable_depth = group == "depth_head"
    return load_vggt_omega(vggt_checkpoint, device=device, enable_camera=enable_camera, enable_depth=enable_depth)


def _sorted_feature_keys(feature_map: Dict[str, torch.Tensor], group: str) -> List[str]:
    prefix = f"{group}_"
    keys = [k for k in feature_map.keys() if k.startswith(prefix)]
    keys.sort(key=lambda k: _parse_feature_idx(k, prefix))
    return keys


def extract_frame_feature_map(vggt: VGGTOmega, image_t: torch.Tensor, group: str, device: torch.device) -> Dict[str, torch.Tensor]:
    source = GROUP_TO_SOURCE[group]
    with torch.no_grad():
        if image_t.ndim == 3:
            images_b = image_t.unsqueeze(0).unsqueeze(0).to(device)
        elif image_t.ndim == 4:
            images_b = image_t.unsqueeze(0).to(device)
        else:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(image_t.shape)}")
        feature_map, _patch_start_idx = extract_vggt_omega_feature_map(
            model=vggt,
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


def _left_panel_width_ratio(image_np: np.ndarray, rows: int, min_ratio: float = 2.6) -> float:
    height, width = image_np.shape[:2]
    aspect = width / max(height, 1)
    return max(min_ratio, 0.95 * rows * aspect)


def plot_block_maps(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    maps_by_block: Dict[str, np.ndarray],
    query_idx: int,
    out_path: Path,
    cols: int,
    patch_size: int,
    heatmap_renderer,
    title: str = "Per-block dynamic score maps",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    block_keys = sorted(maps_by_block.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))
    n = len(block_keys)
    if n == 0:
        raise RuntimeError("No block maps to plot.")

    rows, cols = choose_full_grid(n_blocks=n, rows_hint=max(1, int(math.ceil(n / max(cols, 1)))), cols_hint=max(1, cols))
    left_ratio = _left_panel_width_ratio(image_np=image_np, rows=rows)
    fig = plt.figure(figsize=((cols + left_ratio) * 2.2, rows * 2.4))
    gs = GridSpec(rows, cols + 1, figure=fig, width_ratios=[left_ratio] + [1.0] * cols)

    h, w = image_np.shape[:2]
    grid_w = w // patch_size
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
    ax0.set_title(f"Query token idx={query_idx}", fontsize=24)
    ax0.axis("off")

    for i, k in enumerate(block_keys):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c + 1])

        heat = heatmap_renderer(maps_by_block[k], (h, w))
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
        ax.set_title(display_block_label(k), fontsize=13, pad=3)
        ax.axis("off")

    fig.suptitle(title, fontsize=22)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_pdf_figure(fig, out_path, dpi=220)
    plt.close(fig)

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
    ax2.set_title(f"Query token idx={query_idx}", fontsize=22)
    ax2.axis("off")
    fig2.tight_layout()
    if not out_path.stem.endswith("_upsampled"):
        save_png_pdf_figure(fig2, out_path.with_name(out_path.stem + "_context.png"), dpi=220)
    plt.close(fig2)

def main() -> None:
    args = parse_args()
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
    for input_pos, frame_idx in enumerate(input_frame_indices):
        image_t, mask_t = preprocess_image_and_mask_ref(image_paths[frame_idx], mask_refs[frame_idx])
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask after preprocessing.")

    image_np = (image_tensors[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask_np = target_mask_t.cpu().numpy().astype(np.float32)
    ratio = mask_to_token_dynamic_ratio(target_mask_t.unsqueeze(0), patch_size=VGGT_PATCH_SIZE).squeeze(0).cpu().numpy()

    query_indices: List[int] = []
    if args.query_indices:
        query_indices = [int(x.strip()) for x in args.query_indices.split(",") if x.strip()]
    query_modes = [x.strip() for x in args.query_modes.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=ratio, query_modes=query_modes, query_indices=query_indices)

    device = torch.device(args.device)
    vggt = load_vggt(Path(args.vggt_checkpoint), group=args.group, device=device)
    feature_map = extract_frame_feature_map(vggt=vggt, image_t=torch.stack(image_tensors, dim=0), group=args.group, device=device)

    run_items = discover_run_dirs(Path(args.probe_root), group=args.group)
    run_dir_by_key = {k: d for k, d in run_items}

    usable_keys = [k for k in feature_map.keys() if k in run_dir_by_key]
    usable_keys.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))
    if args.max_blocks > 0:
        usable_keys = usable_keys[: args.max_blocks]
    if not usable_keys:
        raise RuntimeError(
            f"No overlapping block keys between features and probe runs. "
            f"group={args.group}, probe_root={args.probe_root}"
        )

    h, w = image_np.shape[:2]
    grid_h = h // VGGT_PATCH_SIZE
    grid_w = w // VGGT_PATCH_SIZE
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
            if q_idx == 0:
                probs = 1.0 - probs
            maps_by_name[name][k] = probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)

    dataset = str(args.dataset).lower()
    out_dir = build_scoremap_output_dir(
        out_root=args.out_dir,
        group_label=args.group,
        dataset=dataset,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cols = args.cols
    if args.group in {"cam_head", "depth_head"}:
        # Force a single row for head visualizations.
        plot_cols = max(1, len(usable_keys))

    for name, q_idx in queries:
        title = build_scoremap_title(args.probe_root, args.group)
        fig_path = out_dir / f"{name}_query_{q_idx:04d}.png"
        plot_block_maps(
            image_np=image_np,
            mask_np=mask_np,
            maps_by_block=maps_by_name[name],
            query_idx=q_idx,
            out_path=fig_path,
            cols=plot_cols,
            patch_size=VGGT_PATCH_SIZE,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, VGGT_PATCH_SIZE),
            title=title,
        )
        avg_fig_path = out_dir / f"{name}_query_{q_idx:04d}_avg.png"
        plot_average_map(
            image_np=image_np,
            maps_by_block=maps_by_name[name],
            out_path=avg_fig_path,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, VGGT_PATCH_SIZE),
            target_hw=(h, w),
        )
        if args.save_upsampled:
            upsampled_fig_path = out_dir / f"{name}_query_{q_idx:04d}_upsampled.png"
            plot_block_maps(
                image_np=image_np,
                mask_np=mask_np,
                maps_by_block=maps_by_name[name],
                query_idx=q_idx,
                out_path=upsampled_fig_path,
                cols=plot_cols,
                patch_size=VGGT_PATCH_SIZE,
                heatmap_renderer=lambda score_map, _target_hw: upsample_patch_map(score_map, (h, w)),
                title=title,
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
        dataset=dataset,
        sequence=args.sequence,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
        group=args.group,
        probe_root=args.probe_root,
        usable_blocks=usable_keys,
        queries=queries,
        patch_size=VGGT_PATCH_SIZE,
        save_upsampled=bool(args.save_upsampled),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        extra_meta={
            "invert_when_query_idx_zero": True,
            "fbms_split": args.fbms_split if dataset == "fbms59" else None,
        },
    )
    save_scoremap_meta(out_dir, meta)

    print(f"Saved visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
