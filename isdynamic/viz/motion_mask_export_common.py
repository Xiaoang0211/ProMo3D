from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .motion_mask_benchmarks import (
    benchmark_output_dirname,
    build_sequence_spec,
    format_context_summary,
    format_frame_dir,
    load_sequence_paths_from_spec,
    select_context_indices,
)
from .motion_mask_utils import average_block_maps, synchronize_device
from .scoremap_utils import load_probe_from_run, pick_queries, resolve_probe_input_dim, score_map_for_query
from ..core.position_residualization import apply_position_residualizer_to_token_features, load_position_residualizer_from_run


def mask_ref_to_meta(mask_ref):
    if mask_ref is None:
        return None
    if isinstance(mask_ref, Path):
        return str(mask_ref)
    return [str(path) for path in mask_ref]


def load_motion_mask_sequence_context(
    *,
    dataset: str,
    benchmark_name: str | None,
    sequence: str,
    frame_idx: int,
    context_size: int,
    context_stride: int,
    davis_root: str | None,
    segtrack_root: str | None,
    fbms_root: str | None,
    fbms_split: str | None = None,
):
    spec = build_sequence_spec(
        dataset=dataset,
        benchmark=benchmark_name,
        sequence=sequence,
        fbms_split=fbms_split if dataset == "fbms59" else None,
    )
    image_paths, mask_refs, sequence_tag = load_sequence_paths_from_spec(
        spec,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
        fbms_include_unannotated=False,
    )
    if frame_idx < 0 or frame_idx >= len(image_paths):
        raise IndexError(f"frame_idx out of range: {frame_idx} for sequence {sequence_tag} ({len(image_paths)} frames)")
    input_frame_indices = select_context_indices(
        len(image_paths),
        frame_idx,
        total_frames=context_size,
        stride=context_stride,
    )
    return spec, image_paths, mask_refs, sequence_tag, input_frame_indices


def resolve_target_indices(args: argparse.Namespace, num_frames: int) -> List[int]:
    if args.frame_step <= 0:
        raise ValueError("--frame_step must be positive.")

    if args.frame_start is None and args.frame_end is None:
        if args.frame_idx < 0 or args.frame_idx >= num_frames:
            raise IndexError(f"frame_idx out of range: {args.frame_idx} for sequence with {num_frames} frames")
        return [int(args.frame_idx)]

    start = 0 if args.frame_start is None else int(args.frame_start)
    end_exclusive = num_frames if args.frame_end is None else int(args.frame_end)
    if start < 0 or start >= num_frames:
        raise IndexError(f"frame_start out of range: {start} for sequence with {num_frames} frames")
    end_exclusive = min(num_frames, end_exclusive)
    if end_exclusive <= start:
        return []
    return list(range(start, end_exclusive, int(args.frame_step)))


def chunked_indices(items: Sequence[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def collect_pending_target_indices(
    *,
    target_indices: Iterable[int],
    skip_existing: bool,
    out_root: str | Path,
    model_name: str,
    group_label: str,
    benchmark,
    sequence_tag: str,
) -> List[int]:
    pending_indices: List[int] = []
    for frame_idx in target_indices:
        out_dir = build_motion_mask_output_dir(
            out_root=out_root,
            model_name=model_name,
            group_label=group_label,
            benchmark=benchmark,
            sequence_tag=sequence_tag,
            frame_idx=int(frame_idx),
        )
        if skip_existing and (out_dir / "meta.json").exists():
            print(f"[skip] {model_name} :: {benchmark} :: {sequence_tag} :: frame_idx={int(frame_idx)}")
            continue
        pending_indices.append(int(frame_idx))
    return pending_indices


def resolve_input_frame_indices_for_target(
    *,
    dataset: str,
    benchmark_name: str | None,
    sequence: str,
    frame_idx: int,
    context_size: int,
    context_stride: int,
    davis_root: str | None,
    segtrack_root: str | None,
    fbms_root: str | None,
    fbms_split: str | None = None,
) -> List[int]:
    return load_motion_mask_sequence_context(
        dataset=dataset,
        benchmark_name=benchmark_name,
        sequence=sequence,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
        fbms_split=fbms_split,
    )[4]


def select_single_query(
    *,
    mask_token_ratio: np.ndarray,
    query_modes_raw: str,
    query_indices_raw: str,
) -> Tuple[str, int]:
    query_indices: List[int] = [int(x.strip()) for x in query_indices_raw.split(",") if x.strip()] if query_indices_raw else []
    query_modes = [x.strip() for x in query_modes_raw.split(",") if x.strip()]
    queries = pick_queries(mask_token_ratio=mask_token_ratio, query_modes=query_modes, query_indices=query_indices)
    if len(queries) != 1:
        raise ValueError("Motion-mask export expects exactly one query token. Pass a single --query_indices value.")
    return queries[0]




def maybe_apply_position_residualizer(token_tensor: torch.Tensor, run_dir: Path) -> torch.Tensor:
    residualizer = load_position_residualizer_from_run(run_dir)
    return apply_position_residualizer_to_token_features(token_tensor, residualizer)


def get_cached_probe(
    *,
    key: str,
    run_dir_by_key: Dict[str, Path],
    device: torch.device,
    probe_cache: Dict[str, torch.nn.Module] | None = None,
) -> torch.nn.Module:
    if probe_cache is not None and key in probe_cache:
        return probe_cache[key]

    probe = load_probe_from_run(run_dir_by_key[key], device=device)
    if probe_cache is not None:
        probe_cache[key] = probe
    return probe


def score_single_stream_blocks(
    *,
    feature_map: Dict[str, torch.Tensor],
    usable_keys: Sequence[str],
    run_dir_by_key: Dict[str, Path],
    query_idx: int,
    batch_size: int,
    device: torch.device,
    grid_shape: Tuple[int, int],
    feature_batch_idx: int = 0,
    probe_cache: Dict[str, torch.nn.Module] | None = None,
    invert_query_idx_zero: bool = False,
) -> Tuple[np.ndarray, float, float]:
    grid_h, grid_w = grid_shape
    averaged_inputs: Dict[str, np.ndarray] = {}
    probe_scoring_seconds = 0.0
    for key in usable_keys:
        run_dir = run_dir_by_key[key]
        token_tensor = maybe_apply_position_residualizer(feature_map[key][feature_batch_idx, 0], run_dir)
        probe = get_cached_probe(
            key=key,
            run_dir_by_key=run_dir_by_key,
            device=device,
            probe_cache=probe_cache,
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

    average_t0 = perf_counter()
    averaged_patch_score = average_block_maps(averaged_inputs)
    averaging_seconds = perf_counter() - average_t0
    return averaged_patch_score, float(probe_scoring_seconds), float(averaging_seconds)


def score_pairwise_grouped_blocks(
    *,
    groups: Sequence[str],
    feature_map: Dict[str, object],
    probe_root: Path,
    candidate_keys_for_group: Callable[[str], Sequence[str]],
    discover_run_dirs_for_group: Callable[[Path, str], Dict[str, Path]],
    query_idx: int,
    batch_size: int,
    device: torch.device,
    grid_shape: Tuple[int, int],
    token_tensor_for_feature_batch: Callable[[object], torch.Tensor],
    invert_query_idx_zero: bool = False,
    max_blocks: int = 0,
    probe_cache: Dict[str, torch.nn.Module] | None = None,
) -> Tuple[np.ndarray, List[str], float, float]:
    grid_h, grid_w = grid_shape
    averaged_inputs: Dict[str, np.ndarray] = {}
    used_blocks: List[str] = []
    probe_scoring_seconds = 0.0

    for group in groups:
        run_dir_by_key = discover_run_dirs_for_group(probe_root, group)
        candidate_keys = list(candidate_keys_for_group(group))
        usable_keys = [key for key in candidate_keys if key in feature_map and key in run_dir_by_key]
        if max_blocks > 0:
            usable_keys = usable_keys[:max_blocks]
        for key in usable_keys:
            run_dir = run_dir_by_key[key]
            token_tensor_batch = token_tensor_for_feature_batch(feature_map[key])
            token_tensor_batch = torch.stack([
                maybe_apply_position_residualizer(token_tensor_batch[pair_idx], run_dir)
                for pair_idx in range(token_tensor_batch.shape[0])
            ], dim=0)
            probe = get_cached_probe(
                key=key,
                run_dir_by_key=run_dir_by_key,
                device=device,
                probe_cache=probe_cache,
            )
            probe_input_dim = resolve_probe_input_dim(probe)
            if token_tensor_batch.shape[-1] != probe_input_dim:
                raise ValueError(f"Feature dim mismatch for {key}: feature={token_tensor_batch.shape[-1]} probe={probe_input_dim}")
            synchronize_device(device)
            score_t0 = perf_counter()
            pair_maps = []
            for pair_idx in range(token_tensor_batch.shape[0]):
                probs = score_map_for_query(
                    model=probe,
                    token_features=token_tensor_batch[pair_idx],
                    query_idx=query_idx,
                    batch_size=batch_size,
                )
                if invert_query_idx_zero and query_idx == 0:
                    probs = 1.0 - probs
                pair_maps.append(probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32))
            synchronize_device(device)
            probe_scoring_seconds += perf_counter() - score_t0
            averaged_inputs[key] = np.stack(pair_maps, axis=0).mean(axis=0)
            used_blocks.append(key)

    if not averaged_inputs:
        raise RuntimeError(f"No usable blocks found for groups={list(groups)} under probe_root={probe_root}")

    average_t0 = perf_counter()
    averaged_patch_score = average_block_maps(averaged_inputs)
    averaging_seconds = perf_counter() - average_t0
    return averaged_patch_score, used_blocks, float(probe_scoring_seconds), float(averaging_seconds)


def build_motion_mask_output_dir(
    *,
    out_root: str | Path,
    model_name: str,
    group_label: str,
    benchmark,
    sequence_tag: str,
    frame_idx: int,
) -> Path:
    return (
        Path(out_root)
        / model_name
        / group_label
        / benchmark_output_dirname(benchmark)
        / sequence_tag
        / "bilinear_upsample"
        / format_frame_dir(frame_idx)
    )


def single_target_input_protocol(*, total_input_frames: int) -> Dict[str, int]:
    return {
        "target_frames": 1,
        "context_frames": max(0, int(total_input_frames) - 1),
        "total_input_frames": int(total_input_frames),
        "target_view_index": 0,
    }


def pairwise_target_input_protocol(*, total_input_frames: int) -> Dict[str, int]:
    context_frames = max(0, int(total_input_frames) - 1)
    return {
        "target_frames": 1,
        "context_frames": context_frames,
        "total_input_frames": int(total_input_frames),
        "target_view_index": 0,
        "pairwise_references": context_frames,
        "total_model_image_slots": 2 * context_frames,
    }


def build_motion_mask_meta(
    *,
    dataset: str,
    benchmark,
    sequence: str,
    sequence_tag: str,
    frame_idx: int,
    probe_root: str | Path,
    threshold: float,
    query_name: str,
    query_idx: int,
    patch_size: int,
    input_frame_indices: Sequence[int],
    image_paths: Sequence[Path | str],
    target_mask_ref,
    backbone_inference_seconds: float,
    probe_scoring_seconds: float,
    averaging_seconds: float,
    extra_meta: Dict[str, object] | None = None,
) -> Dict[str, object]:
    additional_inference_seconds = float(probe_scoring_seconds + averaging_seconds)
    total_input_frames = max(1, len(input_frame_indices))
    backbone_per_input_frame = float(backbone_inference_seconds) / total_input_frames
    additional_per_input_frame = additional_inference_seconds / total_input_frames
    meta: Dict[str, object] = {
        "dataset": dataset,
        "benchmark": benchmark,
        "sequence": sequence,
        "sequence_tag": sequence_tag,
        "frame_idx": int(frame_idx),
        "probe_root": str(probe_root),
        "threshold": float(threshold),
        "query": {"name": query_name, "idx": int(query_idx)},
        "patch_size": int(patch_size),
        "output_mode": "bilinear_upsample",
        "input_frame_indices": [int(idx) for idx in input_frame_indices],
        "input_image_paths": [str(image_paths[idx]) for idx in input_frame_indices],
        "target_mask_ref": mask_ref_to_meta(target_mask_ref),
        "timing_seconds": {
            "backbone_feature_inference": float(backbone_inference_seconds),
            "backbone_feature_inference_per_input_frame": backbone_per_input_frame,
            "additional_motion_mask_inference": additional_inference_seconds,
            "additional_motion_mask_inference_per_input_frame": additional_per_input_frame,
            "probe_scoring": float(probe_scoring_seconds),
            "score_averaging": float(averaging_seconds),
            "total_compute_excluding_load_and_io": float(backbone_inference_seconds + additional_inference_seconds),
            "total_compute_excluding_load_and_io_per_input_frame": backbone_per_input_frame + additional_per_input_frame,
        },
        **format_context_summary(input_frame_indices),
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta
