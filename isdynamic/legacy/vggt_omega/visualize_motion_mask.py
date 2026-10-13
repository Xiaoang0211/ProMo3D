#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isdynamic.core.common_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.viz.motion_mask_utils import (  # noqa: E402
    average_block_maps,
    add_frame_paths_to_meta,
    chw_tensor_to_uint8_image,
    save_motion_mask_outputs,
    synchronize_device,
)
from isdynamic.viz.motion_mask_export_common import (  # noqa: E402
    build_motion_mask_meta,
    build_motion_mask_output_dir,
    get_cached_probe,
    load_motion_mask_sequence_context,
    select_single_query,
    single_target_input_protocol,
)
from isdynamic.viz.scoremap_utils import (  # noqa: E402
    resolve_probe_input_dim,
    score_map_for_query,
)
from isdynamic.legacy.vggt_omega.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref  # noqa: E402
from isdynamic.legacy.vggt_omega.visualize_isdynamic import (  # noqa: E402
    GROUP_TO_SOURCE,
    VGGTOmega,
    VGGT_PATCH_SIZE,
    _sorted_feature_keys,
    discover_run_dirs,
    load_vggt,
)
from isdynamic.models.vggt_omega import extract_feature_map as extract_vggt_omega_feature_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export averaged binary motion masks for VGGT-Omega.")
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
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--group", type=str, default="inter_frame_block")
    parser.add_argument("--query_modes", type=str, default="dynamic")
    parser.add_argument("--query_indices", type=str, default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "otsu"])
    parser.add_argument("--threshold_offset", type=float, default=0.0)
    parser.add_argument("--max_blocks", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default=str(Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", "outputs")) / "isdynamic_motion_masks"))
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--offload_block_features_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For VGGT-Omega only, keep extracted per-block features on CPU and move back one block at a time during probe scoring.",
    )
    parser.add_argument("--preprocess_mode", type=str, default="native", choices=["native", "aspect_pad"])
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--color_instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also export an instance-colored mask based on connected components.",
    )
    return parser.parse_args()


def score_single_stream_blocks_vggt(
    *,
    feature_map,
    usable_keys,
    run_dir_by_key,
    query_idx,
    batch_size,
    device,
    grid_shape,
    feature_batch_idx=0,
    invert_query_idx_zero=False,
):
    grid_h, grid_w = grid_shape
    averaged_inputs = {}
    probe_scoring_seconds = 0.0

    for key in usable_keys:
        token_tensor = feature_map[key][feature_batch_idx, 0]
        if token_tensor.device != device:
            token_tensor = token_tensor.to(device, non_blocking=device.type == "cuda")

        probe = get_cached_probe(
            key=key,
            run_dir_by_key=run_dir_by_key,
            device=device,
            probe_cache=None,
        )
        probe_input_dim = resolve_probe_input_dim(probe)
        if token_tensor.shape[-1] != probe_input_dim:
            raise ValueError(f"Feature dim mismatch for {key}: feature={token_tensor.shape[-1]} probe={probe_input_dim}")

        synchronize_device(device)
        score_t0 = perf_counter()
        probs = score_map_for_query(
            model=probe,
            token_features=token_tensor,
            query_idx=query_idx,
            batch_size=batch_size,
        )
        synchronize_device(device)
        probe_scoring_seconds += perf_counter() - score_t0

        if invert_query_idx_zero and query_idx == 0:
            probs = 1.0 - probs
        averaged_inputs[key] = probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)

        del probe
        del token_tensor
        del probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    average_t0 = perf_counter()
    averaged_patch_score = average_block_maps(averaged_inputs)
    averaging_seconds = perf_counter() - average_t0
    return averaged_patch_score, float(probe_scoring_seconds), float(averaging_seconds)

def main() -> None:
    args = parse_args()
    spec, image_paths, mask_refs, sequence_tag, input_frame_indices = load_motion_mask_sequence_context(
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
    image_tensors = []
    target_mask_t = None
    for input_pos, frame_idx in enumerate(input_frame_indices):
        image_t, mask_t = preprocess_image_and_mask_ref(
            image_paths[frame_idx],
            mask_refs[frame_idx],
            preprocess_mode=args.preprocess_mode,
        )
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask after preprocessing.")

    image_np = chw_tensor_to_uint8_image(image_tensors[0])
    mask_np = target_mask_t.cpu().numpy().astype(np.float32)
    ratio = mask_to_token_dynamic_ratio(target_mask_t.unsqueeze(0), patch_size=VGGT_PATCH_SIZE).squeeze(0).cpu().numpy()

    query_name, query_idx = select_single_query(
        mask_token_ratio=ratio,
        query_modes_raw=args.query_modes,
        query_indices_raw=args.query_indices,
    )

    device = torch.device(args.device)
    vggt: VGGTOmega = load_vggt(Path(args.vggt_checkpoint), group=args.group, device=device)
    with torch.no_grad():
        images_b = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
        synchronize_device(device)
        backbone_t0 = perf_counter()
        source = GROUP_TO_SOURCE[args.group]
        feature_map, patch_start_idx = extract_vggt_omega_feature_map(
            model=vggt,
            images_b=images_b,
            token_sources=[source],
        )
        if args.offload_block_features_to_cpu:
            feature_map = {key: value.detach().cpu() for key, value in feature_map.items()}

        del images_b
        if device.type == "cuda":
            torch.cuda.empty_cache()
        synchronize_device(device)
        backbone_inference_seconds = perf_counter() - backbone_t0

    run_dir_by_key = {key: path for key, path in discover_run_dirs(Path(args.probe_root), group=args.group)}
    usable_keys = [key for key in _sorted_feature_keys(feature_map, args.group) if key in run_dir_by_key]
    if args.max_blocks > 0:
        usable_keys = usable_keys[: args.max_blocks]
    if not usable_keys:
        raise RuntimeError(f"No overlapping block keys between features and probe runs. group={args.group}")

    height, width = mask_np.shape
    grid_h = height // VGGT_PATCH_SIZE
    grid_w = width // VGGT_PATCH_SIZE
    averaged_patch_score, probe_scoring_seconds, averaging_seconds = score_single_stream_blocks_vggt(
        feature_map=feature_map,
        usable_keys=usable_keys,
        run_dir_by_key=run_dir_by_key,
        query_idx=query_idx,
        batch_size=args.batch_size,
        device=device,
        grid_shape=(grid_h, grid_w),
        invert_query_idx_zero=bool(args.invert_query_idx_zero),
    )
    out_dir = build_motion_mask_output_dir(
        out_root=args.out_dir,
        model_name="vggt_omega",
        group_label=args.group,
        benchmark=spec.benchmark,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
    )
    target_mask_ref = mask_refs[args.frame_idx]
    meta = build_motion_mask_meta(
        dataset=args.dataset,
        benchmark=spec.benchmark,
        sequence=args.sequence,
        sequence_tag=sequence_tag,
        frame_idx=args.frame_idx,
        probe_root=args.probe_root,
        threshold=args.threshold,
        query_name=query_name,
        query_idx=query_idx,
        patch_size=VGGT_PATCH_SIZE,
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        target_mask_ref=target_mask_ref,
        backbone_inference_seconds=backbone_inference_seconds,
        probe_scoring_seconds=probe_scoring_seconds,
        averaging_seconds=averaging_seconds,
        extra_meta={
            "group": args.group,
            "usable_blocks": usable_keys,
            "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
            "input_protocol": single_target_input_protocol(total_input_frames=len(input_frame_indices)),
        },
    )
    if spec.fbms_split is not None:
        meta["fbms_split"] = spec.fbms_split

    save_motion_mask_outputs(
        out_dir=out_dir,
        averaged_patch_score=averaged_patch_score,
        target_hw=(height, width),
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        threshold_offset=args.threshold_offset,
        connected_components=bool(args.color_instances),
        image_np=image_np,
        gt_mask_np=(mask_np > 0.5).astype(np.uint8),
        meta=add_frame_paths_to_meta(
            meta,
            image_path=image_paths[args.frame_idx],
            mask_path=target_mask_ref if isinstance(target_mask_ref, Path) else None,
        ),
    )
    print(f"Saved motion-mask outputs under: {out_dir}")


if __name__ == "__main__":
    main()
