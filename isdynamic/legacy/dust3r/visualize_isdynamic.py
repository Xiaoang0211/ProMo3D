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

from isdynamic.core.stereo_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.models.dust3r import (  # noqa: E402
    DUST3R_PATCH_SIZE,
    GROUP_CHOICES,
    MODEL_NAME_CHOICES,
    build_feature_key_order,
    collate_views,
    extract_feature_map,
    load_dust3r_model,
    sort_feature_keys,
)
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
from isdynamic.legacy.dust3r.build_youtubevos_isdynamic_npz import (  # noqa: E402
    preprocess_image_and_mask_ref,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image, per-block IsDynamic score-map visualization for Dust3R."
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
    parser.add_argument("--ref_frame_idx", type=int, default=None)
    parser.add_argument("--pair_stride", type=int, default=1)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, default=None)
    parser_weights.add_argument("--model_name", type=str, default=None, choices=MODEL_NAME_CHOICES)
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--group", type=str, default="enc", choices=GROUP_CHOICES)
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
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    parser.add_argument(
        "--save_upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save bilinearly upsampled score-map outputs alongside the original patch-expanded heatmaps.",
    )
    return parser.parse_args()


def _resolve_ref_frame_idx(frame_idx: int, num_frames: int, pair_stride: int, ref_frame_idx: int | None) -> int:
    if ref_frame_idx is not None:
        if not (0 <= ref_frame_idx < num_frames):
            raise IndexError(f"ref_frame_idx out of range: {ref_frame_idx} for sequence with {num_frames} frames")
        if ref_frame_idx == frame_idx:
            raise ValueError("ref_frame_idx must differ from frame_idx for Dust3R visualization.")
        return ref_frame_idx

    if pair_stride <= 0:
        raise ValueError("pair_stride must be > 0 when ref_frame_idx is not provided.")
    ref_idx = frame_idx + pair_stride
    if ref_idx >= num_frames:
        ref_idx = frame_idx - pair_stride
    if ref_idx < 0 or ref_idx >= num_frames or ref_idx == frame_idx:
        raise ValueError(
            f"Could not resolve a valid reference frame for frame_idx={frame_idx}, "
            f"pair_stride={pair_stride}, num_frames={num_frames}"
        )
    return ref_idx


def _recover_image_np(view: Dict[str, torch.Tensor]) -> np.ndarray:
    image = view["img"].detach().cpu().permute(1, 2, 0).numpy()
    image = ((image * 0.5) + 0.5) * 255.0
    return image.clip(0, 255).astype(np.uint8)


def _feature_key_sort(key: str):
    suffix = key.rsplit("_", 1)[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def discover_run_dirs(probe_root: Path, group: str) -> Dict[str, Path]:
    group_dir = probe_root / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Probe group directory not found: {group_dir}")
    prefix = f"{group}_"
    run_dirs = [path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    run_dirs.sort(key=lambda path: _feature_key_sort(path.name))
    return {path.name: path for path in run_dirs}


def _token_tensor_for_feature(feature_key: str, feature_value) -> torch.Tensor:
    if isinstance(feature_value, tuple):
        return 0.5 * (feature_value[0][0] + feature_value[1][0])
    return feature_value[0]


def _token_tensor_batch_for_feature(feature_value) -> torch.Tensor:
    if isinstance(feature_value, tuple):
        return 0.5 * (feature_value[0] + feature_value[1])
    return feature_value


def _load_context_tensors(
    args: argparse.Namespace,
    patch_size: int,
    square_ok: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]], np.ndarray, np.ndarray, str, List[int], List[Path]]:
    _spec, image_paths, mask_refs, sequence_tag, input_frame_indices = load_scoremap_sequence_context(
        dataset=args.dataset,
        benchmark_name=args.benchmark_name,
        sequence=args.sequence,
        frame_idx=args.frame_idx,
        context_size=args.context_size,
        context_stride=args.context_stride,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
        fbms_root=args.fbms_root,
        fbms_split=args.fbms_split if str(args.dataset).lower() == "fbms59" else None,
    )
    if len(input_frame_indices) < 2:
        raise ValueError("Dust3R scoremap visualization requires at least 2 frames so the target has at least 1 reference frame.")
    ref_frame_indices = input_frame_indices[1:]

    view1, mask1 = preprocess_image_and_mask_ref(
        image_path=image_paths[input_frame_indices[0]],
        mask_ref=mask_refs[input_frame_indices[0]],
        image_size=args.image_size,
        patch_size=patch_size,
        square_ok=square_ok,
    )
    ref_views: List[Dict[str, torch.Tensor]] = []
    for ref_idx in ref_frame_indices:
        view2, _ = preprocess_image_and_mask_ref(
            image_path=image_paths[ref_idx],
            mask_ref=mask_refs[ref_idx],
            image_size=args.image_size,
            patch_size=patch_size,
            square_ok=square_ok,
        )
        ref_views.append(view2)

    image_np = _recover_image_np(view1)
    mask_np = mask1.cpu().numpy().astype(np.float32)
    return view1, mask1, ref_views, image_np, mask_np, sequence_tag, input_frame_indices, image_paths


def main() -> None:
    args = parse_args()

    patch_size = DUST3R_PATCH_SIZE
    square_ok = args.image_size == 224
    view1, mask1, ref_views, image_np, mask_np, sequence_tag, input_frame_indices, image_paths = _load_context_tensors(
        args=args,
        patch_size=patch_size,
        square_ok=square_ok,
    )
    ratio = mask_to_token_dynamic_ratio(mask1.unsqueeze(0), patch_size=patch_size).squeeze(0).cpu().numpy()

    query_indices: List[int] = []
    if args.query_indices:
        query_indices = [int(x.strip()) for x in args.query_indices.split(",") if x.strip()]
    query_modes = [x.strip() for x in args.query_modes.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=ratio, query_modes=query_modes, query_indices=query_indices)

    device = torch.device(args.device)
    weights_or_name = args.weights or args.model_name
    if not weights_or_name:
        raise ValueError("Either --weights or --model_name is required.")
    model = load_dust3r_model(weights_or_name=weights_or_name, device=device)

    batch_view1 = collate_views([view1 for _ in ref_views])
    batch_view2 = collate_views(ref_views)
    batch_view1 = {key: value.to(device) for key, value in batch_view1.items()}
    batch_view2 = {key: value.to(device) for key, value in batch_view2.items()}
    with torch.no_grad():
        feature_map = extract_feature_map(
            model=model,
            view1=batch_view1,
            view2=batch_view2,
            token_sources=[args.group],
        )

    run_dir_by_key = discover_run_dirs(Path(args.probe_root), group=args.group)
    candidate_keys = build_feature_key_order(model=model, token_sources=[args.group])
    usable_keys = [key for key in candidate_keys if key in feature_map and key in run_dir_by_key]
    usable_keys = sort_feature_keys(usable_keys)
    if args.max_blocks > 0:
        usable_keys = usable_keys[: args.max_blocks]
    if not usable_keys:
        raise RuntimeError(
            f"No overlapping block keys between features and probe runs. group={args.group}, probe_root={args.probe_root}"
        )

    height, width = mask_np.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    token_count = _token_tensor_for_feature(usable_keys[0], feature_map[usable_keys[0]]).shape[0]
    if grid_h * grid_w != token_count:
        raise ValueError(
            f"Token count does not match expected patch grid: grid={grid_h}x{grid_w}, tokens={token_count}"
        )

    maps_by_name: Dict[str, Dict[str, np.ndarray]] = {name: {} for name, _ in queries}
    for key in usable_keys:
        token_tensor_batch = _token_tensor_batch_for_feature(feature_map[key])
        probe = load_probe_from_run(run_dir_by_key[key], device=device)
        if token_tensor_batch.shape[-1] != resolve_probe_input_dim(probe):
            raise ValueError(
                f"Feature dim mismatch for {key}: feature={token_tensor_batch.shape[-1]} probe={resolve_probe_input_dim(probe)}"
            )
        for name, query_idx in queries:
            pair_maps = []
            for pair_idx in range(token_tensor_batch.shape[0]):
                probs = score_map_for_query(
                    model=probe,
                    token_features=token_tensor_batch[pair_idx],
                    query_idx=query_idx,
                    batch_size=args.batch_size,
                )
                if args.invert_query_idx_zero and query_idx == 0:
                    probs = 1.0 - probs
                pair_maps.append(probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32))
            maps_by_name[name][key] = np.stack(pair_maps, axis=0).mean(axis=0)

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
            patch_size=patch_size,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, patch_size),
            title=title,
        )
        avg_fig_path = out_dir / f"{name}_query_{query_idx:04d}_avg.png"
        plot_average_map(
            image_np=image_np,
            maps_by_block=ordered_maps_by_name[name],
            out_path=avg_fig_path,
            heatmap_renderer=lambda score_map, _target_hw: expand_patch_map(score_map, patch_size),
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
                patch_size=patch_size,
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
        patch_size=patch_size,
        save_upsampled=args.save_upsampled,
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        pairwise=True,
        extra_meta={
            "fbms_split": args.fbms_split if dataset == "fbms59" else None,
            "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
        },
    )
    save_scoremap_meta(out_dir, meta)

    print(f"Saved visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
