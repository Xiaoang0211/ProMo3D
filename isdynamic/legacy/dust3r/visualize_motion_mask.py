#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.stereo_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.models.dust3r import (  # noqa: E402
    DUST3R_PATCH_SIZE,
    MODEL_NAME_CHOICES,
    build_feature_key_order,
    collate_views,
    extract_feature_map,
    load_dust3r_model,
    sort_feature_keys,
)
from isdynamic.viz.motion_mask_utils import add_frame_paths_to_meta, save_motion_mask_outputs, synchronize_device  # noqa: E402
from isdynamic.viz.motion_mask_export_common import (  # noqa: E402
    build_motion_mask_meta,
    build_motion_mask_output_dir,
    chunked_indices,
    collect_pending_target_indices,
    load_motion_mask_sequence_context,
    pairwise_target_input_protocol,
    resolve_input_frame_indices_for_target,
    resolve_target_indices,
    score_pairwise_grouped_blocks,
    select_single_query,
)
from isdynamic.legacy.dust3r.visualize_isdynamic import _recover_image_np, discover_run_dirs  # noqa: E402
from isdynamic.legacy.dust3r.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export averaged binary motion masks for Dust3R.")
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "segtrackv2", "fbms59"])
    parser.add_argument("--benchmark_name", type=str, default=None)
    parser.add_argument("--davis_root", type=str, default=None)
    parser.add_argument("--segtrack_root", type=str, default=None)
    parser.add_argument("--fbms_root", type=str, default=None)
    parser.add_argument("--fbms_split", type=str, default="Testset", choices=["Trainingset", "Testset"])
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument("--context_stride", type=int, default=1)
    parser.add_argument("--ref_frame_idx", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pair_stride", type=int, default=1, help=argparse.SUPPRESS)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, default=None)
    parser_weights.add_argument("--model_name", type=str, default=None, choices=MODEL_NAME_CHOICES)
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--groups", type=str, default="dec_view1,head_in_view1")
    parser.add_argument("--query_modes", type=str, default="dynamic")
    parser.add_argument("--query_indices", type=str, default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "otsu"])
    parser.add_argument("--threshold_offset", type=float, default=0.0)
    parser.add_argument("--max_blocks", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default=str(Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", "outputs")) / "isdynamic_motion_masks"))
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert probabilities for query token idx=0 so masks represent dynamic likelihood.",
    )
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--preprocess_mode", type=str, default="native", choices=["native", "aspect_pad"])
    parser.add_argument(
        "--color_instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also export an instance-colored mask based on connected components.",
    )
    return parser.parse_args()
def _token_tensor_for_feature_batch(feature_value) -> torch.Tensor:
    if isinstance(feature_value, tuple):
        return 0.5 * (feature_value[0] + feature_value[1])
    return feature_value


def _slice_feature_map_batch(feature_map, start_idx: int, end_idx: int):
    sliced = {}
    for key, value in feature_map.items():
        if isinstance(value, tuple):
            sliced[key] = tuple(item[start_idx:end_idx] for item in value)
        else:
            sliced[key] = value[start_idx:end_idx]
    return sliced


def _resolve_ref_frame_idx(frame_idx: int, num_frames: int, pair_stride: int, ref_frame_idx: int | None) -> int:
    if ref_frame_idx is not None:
        if not (0 <= ref_frame_idx < num_frames):
            raise IndexError(f"ref_frame_idx out of range: {ref_frame_idx} for sequence with {num_frames} frames")
        if ref_frame_idx == frame_idx:
            raise ValueError("ref_frame_idx must differ from frame_idx for Dust3R motion-mask export.")
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


def main() -> None:
    args = parse_args()
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    if not groups:
        raise ValueError("Expected at least one group in --groups.")

    bootstrap_idx = args.frame_start if args.frame_start is not None else args.frame_idx
    spec, image_paths, mask_refs, sequence_tag, _ = load_motion_mask_sequence_context(
        dataset=args.dataset,
        benchmark_name=args.benchmark_name,
        sequence=args.sequence,
        frame_idx=bootstrap_idx,
        context_size=args.context_size,
        context_stride=args.context_stride,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
        fbms_root=args.fbms_root,
        fbms_split=args.fbms_split,
    )

    target_indices = resolve_target_indices(args, len(image_paths))
    group_label = "__".join(groups)
    pending_indices = collect_pending_target_indices(
        target_indices=target_indices,
        skip_existing=bool(args.skip_existing),
        out_root=args.out_dir,
        model_name="dust3r",
        group_label=group_label,
        benchmark=spec.benchmark,
        sequence_tag=sequence_tag,
    )

    if not pending_indices:
        print(f"No pending frames for dust3r :: {spec.benchmark} :: {sequence_tag}")
        return

    device = torch.device(args.device)
    weights_or_name = args.weights or args.model_name
    if not weights_or_name:
        raise ValueError("Either --weights or --model_name is required.")
    model = load_dust3r_model(weights_or_name=weights_or_name, device=device)
    square_ok = args.image_size == 224
    probe_root = Path(args.probe_root)
    probe_cache: dict[str, torch.nn.Module] = {}

    for frame_chunk in chunked_indices(pending_indices, args.feature_batch_size):
        grouped_chunk_infos: dict[tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]], list[dict[str, object]]] = {}
        for target_idx in frame_chunk:
            input_frame_indices = resolve_input_frame_indices_for_target(
                dataset=args.dataset,
                benchmark_name=args.benchmark_name,
                sequence=args.sequence,
                frame_idx=target_idx,
                context_size=args.context_size,
                context_stride=args.context_stride,
                davis_root=args.davis_root,
                segtrack_root=args.segtrack_root,
                fbms_root=args.fbms_root,
                fbms_split=args.fbms_split,
            )
            if len(input_frame_indices) < 2:
                ref_idx = _resolve_ref_frame_idx(
                    target_idx,
                    len(image_paths),
                    int(args.pair_stride),
                    args.ref_frame_idx,
                )
                input_frame_indices = [int(target_idx), int(ref_idx)]
            ref_indices = input_frame_indices[1:]

            target_view, target_mask = preprocess_image_and_mask_ref(
                image_paths[input_frame_indices[0]],
                mask_refs[input_frame_indices[0]],
                image_size=args.image_size,
                patch_size=DUST3R_PATCH_SIZE,
                square_ok=square_ok,
                preprocess_mode=args.preprocess_mode,
            )
            ref_views = []
            for ref_idx in ref_indices:
                ref_view, _ = preprocess_image_and_mask_ref(
                    image_paths[ref_idx],
                    mask_refs[ref_idx],
                    image_size=args.image_size,
                    patch_size=DUST3R_PATCH_SIZE,
                    square_ok=square_ok,
                    preprocess_mode=args.preprocess_mode,
                )
                ref_views.append(ref_view)

            image_np = _recover_image_np(target_view)
            mask_np = target_mask.cpu().numpy().astype(np.float32)
            ratio = mask_to_token_dynamic_ratio(target_mask.unsqueeze(0), patch_size=DUST3R_PATCH_SIZE).squeeze(0).cpu().numpy()
            query_name, query_idx = select_single_query(
                mask_token_ratio=ratio,
                query_modes_raw=args.query_modes,
                query_indices_raw=args.query_indices,
            )
            stack_key = (
                len(ref_indices),
                tuple(int(x) for x in target_view["img"].shape),
                tuple(tuple(int(x) for x in ref_view["img"].shape) for ref_view in ref_views),
            )
            grouped_chunk_infos.setdefault(stack_key, []).append(
                {
                    "frame_idx": target_idx,
                    "input_frame_indices": input_frame_indices,
                    "ref_indices": ref_indices,
                    "target_view": target_view,
                    "ref_views": ref_views,
                    "image_np": image_np,
                    "mask_np": mask_np,
                    "query_name": query_name,
                    "query_idx": query_idx,
                }
            )

        for _stack_key, chunk_infos in grouped_chunk_infos.items():
            view1_list = []
            view2_list = []
            pair_slices: list[tuple[int, int]] = []
            start_idx = 0
            for info in chunk_infos:
                num_refs = len(info["ref_views"])
                view1_list.extend([info["target_view"] for _ in range(num_refs)])
                view2_list.extend(info["ref_views"])
                end_idx = start_idx + num_refs
                pair_slices.append((start_idx, end_idx))
                start_idx = end_idx

            batch_view1 = collate_views(view1_list)
            batch_view2 = collate_views(view2_list)
            batch_view1 = {key: value.to(device) for key, value in batch_view1.items()}
            batch_view2 = {key: value.to(device) for key, value in batch_view2.items()}

            synchronize_device(device)
            backbone_t0 = perf_counter()
            with torch.no_grad():
                feature_map = extract_feature_map(model=model, view1=batch_view1, view2=batch_view2, token_sources=groups)
            synchronize_device(device)
            chunk_backbone_seconds = perf_counter() - backbone_t0
            amortized_backbone_seconds = float(chunk_backbone_seconds / len(chunk_infos))

            for info, (pair_start, pair_end) in zip(chunk_infos, pair_slices):
                feature_map_slice = _slice_feature_map_batch(feature_map, pair_start, pair_end)
                mask_np = info["mask_np"]
                height, width = mask_np.shape
                grid_h = height // DUST3R_PATCH_SIZE
                grid_w = width // DUST3R_PATCH_SIZE
                averaged_patch_score, used_blocks, probe_scoring_seconds, averaging_seconds = score_pairwise_grouped_blocks(
                    groups=groups,
                    feature_map=feature_map_slice,
                    probe_root=probe_root,
                    candidate_keys_for_group=lambda group: sort_feature_keys(build_feature_key_order(model=model, token_sources=[group])),
                    discover_run_dirs_for_group=discover_run_dirs,
                    query_idx=info["query_idx"],
                    batch_size=args.batch_size,
                    device=device,
                    grid_shape=(grid_h, grid_w),
                    token_tensor_for_feature_batch=_token_tensor_for_feature_batch,
                    invert_query_idx_zero=bool(args.invert_query_idx_zero),
                    max_blocks=args.max_blocks,
                    probe_cache=probe_cache,
                )
                out_dir = build_motion_mask_output_dir(
                    out_root=args.out_dir,
                    model_name="dust3r",
                    group_label=group_label,
                    benchmark=spec.benchmark,
                    sequence_tag=sequence_tag,
                    frame_idx=info["frame_idx"],
                )
                target_mask_ref = mask_refs[info["frame_idx"]]
                meta = build_motion_mask_meta(
                    dataset=args.dataset,
                    benchmark=spec.benchmark,
                    sequence=args.sequence,
                    sequence_tag=sequence_tag,
                    frame_idx=info["frame_idx"],
                    probe_root=args.probe_root,
                    threshold=args.threshold,
                    query_name=info["query_name"],
                    query_idx=info["query_idx"],
                    patch_size=DUST3R_PATCH_SIZE,
                    input_frame_indices=info["input_frame_indices"],
                    image_paths=image_paths,
                    target_mask_ref=target_mask_ref,
                    backbone_inference_seconds=amortized_backbone_seconds,
                    probe_scoring_seconds=probe_scoring_seconds,
                    averaging_seconds=averaging_seconds,
                    extra_meta={
                        "groups": groups,
                        "usable_blocks": used_blocks,
                        "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
                        "image_size": int(args.image_size),
                        "input_protocol": pairwise_target_input_protocol(total_input_frames=len(info["input_frame_indices"])),
                        "reference_image_paths": [str(image_paths[idx]) for idx in info["ref_indices"]],
                        "backbone_batch_size": int(len(chunk_infos)),
                        "backbone_chunk_seconds_total": float(chunk_backbone_seconds),
                        "backbone_time_accounting": "amortized_per_target_frame",
                        "pair_batch_span": [int(pair_start), int(pair_end)],
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
                    image_np=info["image_np"],
                    gt_mask_np=(mask_np > 0.5).astype(np.uint8),
                    meta=add_frame_paths_to_meta(
                        meta,
                        image_path=image_paths[info["frame_idx"]],
                        mask_path=target_mask_ref if isinstance(target_mask_ref, Path) else None,
                    ),
                )
                print(f"Saved motion-mask outputs under: {out_dir}")


if __name__ == "__main__":
    main()
