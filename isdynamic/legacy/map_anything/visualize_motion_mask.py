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

from isdynamic.core.common_labels import mask_to_token_dynamic_ratio  # noqa: E402
from isdynamic.models.map_anything import build_feature_key_order, extract_feature_map, load_mapanything  # noqa: E402
from isdynamic.viz.motion_mask_utils import add_frame_paths_to_meta, save_motion_mask_outputs, synchronize_device  # noqa: E402
from isdynamic.viz.motion_mask_export_common import (  # noqa: E402
    build_motion_mask_meta,
    build_motion_mask_output_dir,
    chunked_indices,
    collect_pending_target_indices,
    get_cached_probe,
    load_motion_mask_sequence_context,
    resolve_input_frame_indices_for_target,
    resolve_target_indices,
    score_single_stream_blocks,
    select_single_query,
    single_target_input_protocol,
)
from isdynamic.legacy.map_anything.visualize_isdynamic import _recover_image_np, discover_run_dirs  # noqa: E402
from isdynamic.legacy.map_anything.build_davis_isdynamic_npz import preprocess_entries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export averaged binary motion masks for Map-Anything.")
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
    parser.add_argument("--checkpoint", type=str, required=True)
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
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--machine", type=str, default="aws")
    parser.add_argument("--resolution_set", type=int, default=518, choices=[504, 512, 518])
    parser.add_argument("--preprocess_mode", type=str, default="native", choices=["native", "aspect_pad"])
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert probabilities for query token idx=0 so masks represent dynamic likelihood.",
    )
    parser.add_argument(
        "--color_instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also export an instance-colored mask based on connected components.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
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
    pending_indices = collect_pending_target_indices(
        target_indices=target_indices,
        skip_existing=bool(args.skip_existing),
        out_root=args.out_dir,
        model_name="map_anything",
        group_label=args.group,
        benchmark=spec.benchmark,
        sequence_tag=sequence_tag,
    )

    if not pending_indices:
        print(f"No pending frames for map_anything :: {spec.benchmark} :: {sequence_tag}")
        return

    device = torch.device(args.device)
    model = load_mapanything(checkpoint_path=args.checkpoint, device=device, machine=args.machine)
    norm_type = model.encoder.data_norm_type
    patch_size = int(model.encoder.patch_size)
    run_dir_by_key = discover_run_dirs(Path(args.probe_root), group=args.group)
    candidate_keys = build_feature_key_order(model=model, token_sources=[args.group])
    probe_cache: dict[str, torch.nn.Module] = {}
    usable_keys_verified: list[str] | None = None

    for frame_chunk in chunked_indices(pending_indices, args.feature_batch_size):
        grouped_chunk_infos: dict[tuple[tuple[int, ...], ...], list[dict[str, object]]] = {}
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
            entries = [(image_paths[idx], mask_refs[idx]) for idx in input_frame_indices]
            image_tensors, mask_tensors = preprocess_entries(
                entries,
                norm_type=norm_type,
                resolution_set=args.resolution_set,
                preprocess_mode=args.preprocess_mode,
            )
            image_t = image_tensors[0]
            mask_t = mask_tensors[0]
            image_np = _recover_image_np(image_t, norm_type=norm_type)
            mask_np = mask_t.cpu().numpy().astype(np.float32)
            ratio = mask_to_token_dynamic_ratio(mask_t.unsqueeze(0).unsqueeze(0), patch_size=patch_size).squeeze(0).squeeze(0).cpu().numpy()
            query_name, query_idx = select_single_query(
                mask_token_ratio=ratio,
                query_modes_raw=args.query_modes,
                query_indices_raw=args.query_indices,
            )
            stack_shape_key = tuple(tuple(int(x) for x in image_tensor.shape) for image_tensor in image_tensors)
            grouped_chunk_infos.setdefault(stack_shape_key, []).append(
                {
                    "frame_idx": target_idx,
                    "input_frame_indices": input_frame_indices,
                    "image_tensors": image_tensors,
                    "image_np": image_np,
                    "mask_np": mask_np,
                    "query_name": query_name,
                    "query_idx": query_idx,
                }
            )

        for stack_shape_key, chunk_infos in grouped_chunk_infos.items():
            num_views = len(chunk_infos[0]["image_tensors"])
            views = []
            for view_idx in range(num_views):
                image_batch = torch.stack(
                    [info["image_tensors"][view_idx] for info in chunk_infos],
                    dim=0,
                ).to(device)
                true_shape = np.stack(
                    [
                        np.int32(info["image_tensors"][view_idx].shape[-2:])
                        for info in chunk_infos
                    ],
                    axis=0,
                )
                views.append(
                    {
                        "img": image_batch,
                        "data_norm_type": [norm_type] * len(chunk_infos),
                        "true_shape": true_shape,
                        "idx": view_idx,
                        "instance": str(view_idx),
                    }
                )

            synchronize_device(device)
            backbone_t0 = perf_counter()
            with torch.no_grad():
                feature_map = extract_feature_map(model, views, [args.group])
            synchronize_device(device)
            chunk_backbone_seconds = perf_counter() - backbone_t0
            amortized_backbone_seconds = float(chunk_backbone_seconds / len(chunk_infos))

            if usable_keys_verified is None:
                usable_keys = [key for key in candidate_keys if key in feature_map and key in run_dir_by_key]
                if args.max_blocks > 0:
                    usable_keys = usable_keys[: args.max_blocks]
                if not usable_keys:
                    raise RuntimeError(f"No overlapping block keys between features and probe runs. group={args.group}")
                usable_keys_verified = usable_keys
                for key in usable_keys_verified:
                    get_cached_probe(
                        key=key,
                        run_dir_by_key=run_dir_by_key,
                        device=device,
                        probe_cache=probe_cache,
                    )

            for feature_batch_idx, info in enumerate(chunk_infos):
                mask_np = info["mask_np"]
                height, width = mask_np.shape
                grid_h = height // patch_size
                grid_w = width // patch_size
                averaged_patch_score, probe_scoring_seconds, averaging_seconds = score_single_stream_blocks(
                    feature_map=feature_map,
                    usable_keys=usable_keys_verified,
                    run_dir_by_key=run_dir_by_key,
                    query_idx=info["query_idx"],
                    batch_size=args.batch_size,
                    device=device,
                    grid_shape=(grid_h, grid_w),
                    feature_batch_idx=feature_batch_idx,
                    probe_cache=probe_cache,
                    invert_query_idx_zero=bool(args.invert_query_idx_zero),
                )
                out_dir = build_motion_mask_output_dir(
                    out_root=args.out_dir,
                    model_name="map_anything",
                    group_label=args.group,
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
                    patch_size=patch_size,
                    input_frame_indices=info["input_frame_indices"],
                    image_paths=image_paths,
                    target_mask_ref=target_mask_ref,
                    backbone_inference_seconds=amortized_backbone_seconds,
                    probe_scoring_seconds=probe_scoring_seconds,
                    averaging_seconds=averaging_seconds,
                    extra_meta={
                        "group": args.group,
                        "usable_blocks": usable_keys_verified,
                        "norm_type": norm_type,
                        "resolution_set": int(args.resolution_set),
                        "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
                        "input_protocol": single_target_input_protocol(total_input_frames=len(info["input_frame_indices"])),
                        "backbone_batch_size": int(len(chunk_infos)),
                        "backbone_chunk_seconds_total": float(chunk_backbone_seconds),
                        "backbone_time_accounting": "amortized_per_target_frame",
                        "context_stack_shape": [list(item) for item in stack_shape_key],
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
