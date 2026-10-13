#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.common_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.models.map_anything import (  # noqa: E402
    build_feature_key_order,
    extract_feature_map,
    load_mapanything,
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
from isdynamic.legacy.map_anything.build_davis_isdynamic_npz import (  # noqa: E402
    preprocess_entries,
)
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT  # noqa: E402


GROUP_CHOICES = ["inter_frame_block", "intra_frame_block", "dense_head", "pose_head", "scale_head"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image, per-block IsDynamic score-map visualization for Map-Anything."
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
    parser.add_argument("--checkpoint", type=str, required=True)
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
    parser.add_argument("--machine", type=str, default="aws")
    parser.add_argument("--resolution_set", type=int, default=518, choices=[504, 512, 518])
    parser.add_argument(
        "--save_upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save bilinearly upsampled score-map outputs alongside the original patch-expanded heatmaps.",
    )
    return parser.parse_args()


def _recover_image_np(image_t: torch.Tensor, norm_type: str) -> np.ndarray:
    normalization = IMAGE_NORMALIZATION_DICT[norm_type]
    mean = np.asarray(normalization.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(normalization.std, dtype=np.float32).reshape(3, 1, 1)
    image = image_t.detach().cpu().numpy()
    image = (image * std) + mean
    image = image.transpose(1, 2, 0) * 255.0
    return image.clip(0, 255).astype(np.uint8)


def discover_run_dirs(probe_root: Path, group: str) -> Dict[str, Path]:
    group_dir = probe_root / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Probe group directory not found: {group_dir}")
    prefix = f"{group}_"
    run_dirs = [path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    run_dirs.sort(key=lambda path: int(path.name[len(prefix) :]))
    return {path.name: path for path in run_dirs}


def main() -> None:
    args = parse_args()
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

    device = torch.device(args.device)
    model = load_mapanything(
        checkpoint_path=args.checkpoint,
        device=device,
        machine=args.machine,
    )
    norm_type = model.encoder.data_norm_type

    image_tensors, mask_tensors = preprocess_entries(
        [(image_paths[idx], mask_refs[idx]) for idx in input_frame_indices],
        norm_type=norm_type,
        resolution_set=args.resolution_set,
    )
    image_t = image_tensors[0]
    mask_t = mask_tensors[0]
    image_np = _recover_image_np(image_t, norm_type=norm_type)
    mask_np = mask_t.cpu().numpy().astype(np.float32)

    views = [
        {
            "img": image_t.unsqueeze(0).to(device),
            "data_norm_type": [norm_type],
            "true_shape": np.int32([image_t.shape[-2:]]),
            "idx": 0,
            "instance": "0",
        }
    ]
    with torch.no_grad():
        feature_map = extract_feature_map(model, views, [args.group])

    patch_size = int(model.encoder.patch_size)
    ratio = mask_to_token_dynamic_ratio(mask_t.unsqueeze(0).unsqueeze(0), patch_size=patch_size).squeeze(0).squeeze(0).cpu().numpy()

    query_indices: List[int] = []
    if args.query_indices:
        query_indices = [int(x.strip()) for x in args.query_indices.split(",") if x.strip()]
    query_modes = [x.strip() for x in args.query_modes.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=ratio, query_modes=query_modes, query_indices=query_indices)

    run_dir_by_key = discover_run_dirs(Path(args.probe_root), group=args.group)
    candidate_keys = build_feature_key_order(model=model, token_sources=[args.group])
    usable_keys = [key for key in candidate_keys if key in feature_map and key in run_dir_by_key]
    if args.max_blocks > 0:
        usable_keys = usable_keys[: args.max_blocks]
    if not usable_keys:
        raise RuntimeError(
            f"No overlapping block keys between features and probe runs. group={args.group}, probe_root={args.probe_root}"
        )

    height, width = mask_np.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
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
        save_upsampled=bool(args.save_upsampled),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        extra_meta={
            "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
            "fbms_split": args.fbms_split if dataset == "fbms59" else None,
        },
    )
    save_scoremap_meta(out_dir, meta)

    print(f"Saved visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
