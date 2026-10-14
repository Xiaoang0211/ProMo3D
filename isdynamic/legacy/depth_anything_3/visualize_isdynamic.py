#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isdynamic.models.depth_anything_3 import load_da3_model  # noqa: E402
from isdynamic.viz.scoremap_export_common import (  # noqa: E402
    build_scoremap_meta,
    build_scoremap_output_dir,
    load_scoremap_sequence_context,
    save_scoremap_meta,
)
from isdynamic.viz.scoremap_utils import (  # noqa: E402
    build_scoremap_title,
    plot_average_map,
    expand_patch_map,
    load_probe_from_run,
    pick_queries,
    plot_block_maps,
    resolve_probe_input_dim,
    save_upsampled_maps_npz,
    score_map_for_query,
    upsample_patch_map,
)
from isdynamic.legacy.depth_anything_3.build_davis_isdynamic_npz import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
    _resolve_anyview_model,
    _run_anyview_backbone,
    build_feature_key_order,
    extract_feature_map,
    mask_to_token_dynamic_ratio,
)
from isdynamic.legacy.depth_anything_3.build_youtubevos_isdynamic_npz import (  # noqa: E402
    preprocess_image_and_mask_ref,
)


GROUP_CHOICES = ["inter_frame_block", "intra_frame_block", "depth_head", "dual_dpt_head"]


def canonical_group_name(group: str) -> str:
    if group == "dual_dpt_head":
        return "depth_head"
    return group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image, per-block dynamic score-map visualization for Depth-Anything-3."
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
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--group", type=str, default="inter_frame_block", choices=GROUP_CHOICES)
    parser.add_argument("--query_modes", type=str, default="dynamic")
    parser.add_argument("--query_indices", type=str, default=None)
    parser.add_argument("--max_blocks", type=int, default=0)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="outputs/isdynamic_visualization")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert probabilities for query token idx=0 so maps visualize dynamic likelihood.",
    )
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
    )
    parser.add_argument(
        "--save_upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save bilinearly upsampled score-map outputs alongside the original patch-expanded heatmaps.",
    )
    return parser.parse_args()


def _recover_image_np(image_t: torch.Tensor) -> np.ndarray:
    image = image_t.detach().cpu().numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = image.transpose(1, 2, 0) * 255.0
    return image.clip(0, 255).astype(np.uint8)


def discover_run_dirs(probe_root: Path, group: str) -> Dict[str, Path]:
    group = canonical_group_name(group)
    group_dir = probe_root / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Probe group directory not found: {group_dir}")
    prefix = f"{group}_"
    run_dirs = [path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    run_dirs.sort(key=lambda path: int(path.name[len(prefix) :]))
    return {path.name: path for path in run_dirs}


def _load_context_tensors(
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], torch.Tensor, np.ndarray, np.ndarray, str, List[int], List[Path]]:
    preprocess_kwargs = {
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
    }
    spec, image_paths, mask_refs, sequence_tag, input_frame_indices = load_scoremap_sequence_context(
        dataset=args.dataset,
        benchmark_name=args.benchmark_name,
        sequence=args.sequence,
        frame_idx=args.frame_idx,
        context_size=args.context_size,
        context_stride=args.context_stride,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
        fbms_root=args.fbms_root,
        fbms_split=args.fbms_split,
    )
    image_tensors: List[torch.Tensor] = []
    target_mask_t = None
    for input_pos, frame_idx in enumerate(input_frame_indices):
        image_t, mask_t = preprocess_image_and_mask_ref(
            image_path=image_paths[frame_idx],
            mask_ref=mask_refs[frame_idx],
            **preprocess_kwargs,
        )
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask after preprocessing.")

    image_np = _recover_image_np(image_tensors[0])
    mask_np = target_mask_t.cpu().numpy().astype(np.float32)
    return image_tensors, target_mask_t, image_np, mask_np, sequence_tag, input_frame_indices, image_paths


def main() -> None:
    args = parse_args()
    args.group = canonical_group_name(args.group)
    image_tensors, target_mask_t, image_np, mask_np, sequence_tag, input_frame_indices, image_paths = _load_context_tensors(args)
    ratio = mask_to_token_dynamic_ratio(target_mask_t.unsqueeze(0), patch_size=PATCH_SIZE).squeeze(0).cpu().numpy()

    query_indices: List[int] = []
    if args.query_indices:
        query_indices = [int(x.strip()) for x in args.query_indices.split(",") if x.strip()]
    query_modes = [x.strip() for x in args.query_modes.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=ratio, query_modes=query_modes, query_indices=query_indices)

    device = torch.device(args.device)
    model = load_da3_model(args.model_dir, device=device)
    anyview_model = _resolve_anyview_model(model)
    export_feat_layers = list(range(len(anyview_model.backbone.pretrained.blocks)))
    token_sources = ["alt_attn"] if args.group in {"inter_frame_block", "intra_frame_block"} else ["depth_head"]

    images = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    feats, aux_feats = _run_anyview_backbone(
        anyview_model=anyview_model,
        images_b=images,
        export_feat_layers=export_feat_layers,
    )
    feature_map = extract_feature_map(
        anyview_model=anyview_model,
        feats=feats,
        aux_feats=aux_feats,
        export_feat_layers=export_feat_layers,
        token_sources=token_sources,
    )

    feature_key_order, _, _ = build_feature_key_order(anyview_model=anyview_model, token_sources=token_sources)
    run_dir_by_key = discover_run_dirs(Path(args.probe_root), group=args.group)
    usable_keys = [key for key in feature_key_order if key.startswith(f"{args.group}_") and key in feature_map and key in run_dir_by_key]
    if args.max_blocks > 0:
        usable_keys = usable_keys[: args.max_blocks]
    if not usable_keys:
        raise RuntimeError(
            f"No overlapping block keys between features and probe runs. group={args.group}, probe_root={args.probe_root}"
        )

    height, width = mask_np.shape
    grid_h = height // PATCH_SIZE
    grid_w = width // PATCH_SIZE
    token_count = feature_map[usable_keys[0]][0, 0].shape[0]
    if grid_h * grid_w != token_count:
        raise ValueError(
            f"Token count does not match expected patch grid: grid={grid_h}x{grid_w}, tokens={token_count}"
        )

    maps_by_name: Dict[str, Dict[str, np.ndarray]] = {name: {} for name, _ in queries}
    for key in usable_keys:
        token_tensor = feature_map[key][0, 0]
        probe = load_probe_from_run(run_dir_by_key[key], device=device)
        probe_input_dim = resolve_probe_input_dim(probe)
        if token_tensor.shape[-1] != probe_input_dim:
            raise ValueError(f"Feature dim mismatch for {key}: feature={token_tensor.shape[-1]} probe={probe_input_dim}")
        for name, query_idx in queries:
            probs = score_map_for_query(
                model=probe,
                token_features=token_tensor,
                query_idx=query_idx,
                batch_size=args.batch_size,
            )
            if args.invert_query_idx_zero and query_idx == 0:
                probs = 1.0 - probs
            maps_by_name[name][key] = probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)

    plot_rows = max(1, args.rows)
    plot_cols = max(1, args.cols)
    dataset = str(args.dataset).lower()
    out_dir = build_scoremap_output_dir(
        out_root=args.out_dir,
        group_label=args.group,
        dataset=dataset,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_maps_by_name = {
        name: {key: maps_by_name[name][key] for key in usable_keys}
        for name, _ in queries
    }
    title = build_scoremap_title(args.probe_root, args.group)
    for name, query_idx in queries:
        fig_path = out_dir / f"{name}_query_{query_idx:04d}.png"
        plot_block_maps(
            image_np=image_np,
            mask_np=mask_np,
            maps_by_block=ordered_maps_by_name[name],
            query_idx=query_idx,
            out_path=fig_path,
            rows=plot_rows,
            cols=plot_cols,
            patch_size=PATCH_SIZE,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, PATCH_SIZE),
            title=title,
        )
        avg_fig_path = out_dir / f"{name}_query_{query_idx:04d}_avg.png"
        plot_average_map(
            image_np=image_np,
            maps_by_block=ordered_maps_by_name[name],
            out_path=avg_fig_path,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, PATCH_SIZE),
            target_hw=(height, width),
        )
        if args.save_upsampled:
            upsampled_path = out_dir / f"{name}_query_{query_idx:04d}_upsampled.png"
            plot_block_maps(
                image_np=image_np,
                mask_np=mask_np,
                maps_by_block=ordered_maps_by_name[name],
                query_idx=query_idx,
                out_path=upsampled_path,
                rows=plot_rows,
                cols=plot_cols,
                patch_size=PATCH_SIZE,
                heatmap_renderer=lambda score_map, target_hw: upsample_patch_map(score_map, target_hw),
                title=title,
            )
            avg_upsampled_path = out_dir / f"{name}_query_{query_idx:04d}_avg_upsampled.png"
            plot_average_map(
                image_np=image_np,
                maps_by_block=ordered_maps_by_name[name],
                out_path=avg_upsampled_path,
                heatmap_renderer=lambda score_map, target_hw: upsample_patch_map(score_map, target_hw),
                target_hw=(height, width),
            )
            save_upsampled_maps_npz(
                maps_by_block=ordered_maps_by_name[name],
                out_path=out_dir / f"{name}_query_{query_idx:04d}_upsampled_maps.npz",
                target_hw=(height, width),
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
        patch_size=PATCH_SIZE,
        save_upsampled=bool(args.save_upsampled),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        extra_meta={
            "fbms_split": args.fbms_split if dataset == "fbms59" else None,
            "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
        },
    )
    save_scoremap_meta(out_dir, meta)

    print(f"Saved visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
