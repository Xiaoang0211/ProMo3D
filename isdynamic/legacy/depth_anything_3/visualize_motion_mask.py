#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isdynamic.models.depth_anything_3 import load_da3_model  # noqa: E402
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
from isdynamic.legacy.depth_anything_3.visualize_isdynamic import _recover_image_np, discover_run_dirs  # noqa: E402
from isdynamic.legacy.depth_anything_3.build_davis_isdynamic_npz import (  # noqa: E402
    PATCH_SIZE,
    _resolve_anyview_model,
    _run_anyview_backbone,
    build_feature_key_order,
    extract_feature_map,
    mask_to_token_dynamic_ratio,
)
from isdynamic.legacy.depth_anything_3.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref  # noqa: E402


def canonical_group_name(group: str) -> str:
    if group == "dual_dpt_head":
        return "depth_head"
    return group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export averaged binary motion masks for Depth-Anything-3.")
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
    parser.add_argument("--model_dir", type=str, required=True)
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
    parser.add_argument(
        "--invert_query_idx_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert probabilities for query token idx=0 so masks represent dynamic likelihood.",
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
    parser.add_argument("--preprocess_mode", type=str, default="native", choices=["native", "aspect_pad"])
    parser.add_argument(
        "--color_instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also export an instance-colored mask based on connected components.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    args.group = canonical_group_name(args.group)
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
        model_name="depth_anything_3",
        group_label=args.group,
        benchmark=spec.benchmark,
        sequence_tag=sequence_tag,
    )

    if not pending_indices:
        print(f"No pending frames for depth_anything_3 :: {spec.benchmark} :: {sequence_tag}")
        return

    device = torch.device(args.device)
    model = load_da3_model(args.model_dir, device=device)
    anyview_model = _resolve_anyview_model(model)
    export_feat_layers = list(range(len(anyview_model.backbone.pretrained.blocks)))
    token_sources = ["alt_attn"] if args.group in {"inter_frame_block", "intra_frame_block"} else ["depth_head"]

    feature_key_order, _, _ = build_feature_key_order(anyview_model=anyview_model, token_sources=token_sources)
    run_dir_by_key = discover_run_dirs(Path(args.probe_root), group=args.group)
    probe_cache: dict[str, torch.nn.Module] = {}
    frame_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_preprocessed(frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = frame_cache.get(frame_idx)
        if cached is not None:
            return cached
        image_t, mask_t = preprocess_image_and_mask_ref(
            image_paths[frame_idx],
            mask_refs[frame_idx],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            preprocess_mode=args.preprocess_mode,
        )
        frame_cache[frame_idx] = (image_t, mask_t)
        return image_t, mask_t

    usable_keys_verified: list[str] | None = None

    for frame_chunk in chunked_indices(pending_indices, args.feature_batch_size):
        grouped_chunk_infos: dict[tuple[int, ...], list[dict[str, object]]] = {}
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
            image_tensors = []
            target_mask_t = None
            for input_pos, input_frame_idx in enumerate(input_frame_indices):
                image_t, mask_t = get_preprocessed(input_frame_idx)
                image_tensors.append(image_t)
                if input_pos == 0:
                    target_mask_t = mask_t
            if target_mask_t is None:
                raise RuntimeError("Missing target mask after preprocessing.")

            image_np = _recover_image_np(image_tensors[0])
            mask_np = target_mask_t.cpu().numpy().astype(np.float32)
            ratio = mask_to_token_dynamic_ratio(target_mask_t.unsqueeze(0), patch_size=PATCH_SIZE).squeeze(0).cpu().numpy()
            query_name, query_idx = select_single_query(
                mask_token_ratio=ratio,
                query_modes_raw=args.query_modes,
                query_indices_raw=args.query_indices,
            )
            image_stack = torch.stack(image_tensors, dim=0)
            stack_shape_key = tuple(int(x) for x in image_stack.shape)
            grouped_chunk_infos.setdefault(stack_shape_key, []).append(
                {
                    "frame_idx": target_idx,
                    "input_frame_indices": input_frame_indices,
                    "image_np": image_np,
                    "mask_np": mask_np,
                    "query_name": query_name,
                    "query_idx": query_idx,
                    "image_stack": image_stack,
                }
            )

        for stack_shape_key, chunk_infos in grouped_chunk_infos.items():
            images_b = torch.stack([info["image_stack"] for info in chunk_infos], dim=0).to(device=device, dtype=torch.float32)
            synchronize_device(device)
            backbone_t0 = perf_counter()
            feats, aux_feats = _run_anyview_backbone(
                anyview_model=anyview_model,
                images_b=images_b,
                export_feat_layers=export_feat_layers,
            )
            feature_map = extract_feature_map(
                anyview_model=anyview_model,
                feats=feats,
                aux_feats=aux_feats,
                export_feat_layers=export_feat_layers,
                token_sources=token_sources,
            )
            synchronize_device(device)
            chunk_backbone_seconds = perf_counter() - backbone_t0
            amortized_backbone_seconds = float(chunk_backbone_seconds / len(chunk_infos))

            if usable_keys_verified is None:
                usable_keys = [
                    key for key in feature_key_order if key.startswith(f"{args.group}_") and key in feature_map and key in run_dir_by_key
                ]
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
                grid_h = height // PATCH_SIZE
                grid_w = width // PATCH_SIZE
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
                    model_name="depth_anything_3",
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
                    patch_size=PATCH_SIZE,
                    input_frame_indices=info["input_frame_indices"],
                    image_paths=image_paths,
                    target_mask_ref=target_mask_ref,
                    backbone_inference_seconds=amortized_backbone_seconds,
                    probe_scoring_seconds=probe_scoring_seconds,
                    averaging_seconds=averaging_seconds,
                    extra_meta={
                        "group": args.group,
                        "usable_blocks": usable_keys_verified,
                        "invert_when_query_idx_zero": bool(args.invert_query_idx_zero),
                        "process_res": int(args.process_res),
                        "process_res_method": args.process_res_method,
                        "input_protocol": single_target_input_protocol(total_input_frames=len(info["input_frame_indices"])),
                        "backbone_batch_size": int(len(chunk_infos)),
                        "backbone_chunk_seconds_total": float(chunk_backbone_seconds),
                        "backbone_time_accounting": "amortized_per_target_frame",
                        "context_stack_shape": [int(x) for x in stack_shape_key],
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
