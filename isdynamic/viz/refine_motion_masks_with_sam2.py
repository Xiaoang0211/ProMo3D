from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from isdynamic.viz.motion_mask_utils import (
    _overlay_binary_mask_on_image,
    binary_mask_from_score,
    connected_components_4,
    erode_binary_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine motion masks with SAM2 using Pi3-style region-growing instance prompts "
            "and parity/single video propagation."
        )
    )
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--sam2_repo_root", type=str, default=str(PROJECT_ROOT / "sam2"))
    parser.add_argument("--sam2_cfg", type=str, required=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variant_name", type=str, default="sam2_prompt")
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--prompt_score_source", type=str, default="upsampled", choices=["upsampled", "patch"])
    parser.add_argument("--prompt_threshold", type=float, default=0.9)
    parser.add_argument("--prompt_threshold_mode", type=str, default="fixed", choices=["fixed", "otsu"])
    parser.add_argument("--prompt_threshold_offset", type=float, default=0.0)
    parser.add_argument("--prompt_erode_iterations", type=int, default=0)
    parser.add_argument("--instance_mean_score_threshold", type=float, default=0.0)
    parser.add_argument("--instance_mean_score_threshold_mode", type=str, default="fixed", choices=["fixed", "otsu"])
    parser.add_argument("--instance_mean_score_threshold_offset", type=float, default=0.0)
    parser.add_argument("--propagate_mode", type=str, default="parity", choices=["parity", "single"])
    parser.add_argument("--max_area_increase", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--window_overlap", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload_video_to_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload_state_to_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async_loading_frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile_image_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile only SAM2's image encoder, preserving the standard video predictor.",
    )
    parser.add_argument(
        "--cache_image_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache every image feature in each window for reuse across parity passes.",
    )
    parser.add_argument(
        "--runtime_warmup_mask_root",
        type=str,
        default=None,
        help="Optional coarse-mask sequence used to warm up SAM2 before timing the requested sequence.",
    )
    return parser.parse_args()


def _resolve_existing_path(path_text: str, repo_root: Path, fallback_rel: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        repo_root / path,
        repo_root / "sam2" / path,
        repo_root / fallback_rel,
        PROJECT_ROOT / "sam2" / fallback_rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_root / path


def _to_hydra_config_name(config_path: Path, repo_root: Path, original_text: str) -> str:
    package_root = repo_root / "sam2"
    for base in (package_root, repo_root):
        try:
            rel = config_path.relative_to(base)
            return rel.as_posix()
        except ValueError:
            continue
    return original_text


def _import_sam2_build(repo_root: Path):
    package_root = repo_root.resolve()
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor


def _sam2_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _frame_sort_key(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def _load_overlay_image_for_frame(mask_root: Path, frame_idx: int) -> np.ndarray | None:
    frame_dir = mask_root / f"frame_{frame_idx:05d}"
    meta_path = frame_dir / "meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            image_path = meta.get("image_path")
            if isinstance(image_path, str):
                path = Path(image_path)
                if path.exists():
                    return np.array(Image.open(path).convert("RGB"))
        except Exception:
            pass

    input_image = frame_dir / "input_image.png"
    if input_image.exists():
        return np.array(Image.open(input_image).convert("RGB"))

    return None


def _resolve_source_image_path(mask_root: Path, frame_idx: int) -> Path:
    frame_dir = mask_root / f"frame_{frame_idx:05d}"
    meta_path = frame_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        image_path = meta.get("image_path")
        if isinstance(image_path, str):
            path = Path(image_path)
            if path.exists():
                return path

    input_image = frame_dir / "input_image.png"
    if input_image.exists():
        return input_image
    raise FileNotFoundError(f"Could not resolve source image for frame_{frame_idx:05d} under {mask_root}")


def _build_window_video_dir(
    mask_root: Path,
    global_frame_indices: Sequence[int],
    *,
    pad_to_length: int = 0,
) -> Tuple[TemporaryDirectory, Dict[int, int], List[Optional[int]], str]:
    tmpdir = TemporaryDirectory(prefix="sam2_window_", dir="/tmp")
    tmp_path = Path(tmpdir.name)
    global_to_local: Dict[int, int] = {}
    local_to_global: List[Optional[int]] = []
    actual_global_frame_indices = [int(idx) for idx in global_frame_indices]
    padded_global_frame_indices: List[Optional[int]] = list(actual_global_frame_indices)
    if pad_to_length > len(actual_global_frame_indices) and actual_global_frame_indices:
        padded_global_frame_indices.extend([None] * (pad_to_length - len(actual_global_frame_indices)))

    last_actual_global_idx = actual_global_frame_indices[-1] if actual_global_frame_indices else None
    for local_idx, global_idx in enumerate(padded_global_frame_indices):
        source_global_idx = last_actual_global_idx if global_idx is None else int(global_idx)
        if source_global_idx is None:
            continue
        src = _resolve_source_image_path(mask_root, int(source_global_idx))
        dst = tmp_path / f"{local_idx:05d}.jpg"
        try:
            os.symlink(src, dst)
        except OSError:
            Image.open(src).convert("RGB").save(dst)
        local_to_global.append(None if global_idx is None else int(global_idx))
        if global_idx is not None:
            global_to_local[int(global_idx)] = int(local_idx)
    return tmpdir, global_to_local, local_to_global, str(tmp_path)


def _load_score_map(frame_dir: Path, prompt_score_source: str) -> np.ndarray | None:
    if prompt_score_source == "upsampled":
        candidates = [frame_dir / "avg_score_upsampled.npy", frame_dir / "avg_score_patch.npy"]
    else:
        candidates = [frame_dir / "avg_score_patch.npy", frame_dir / "avg_score_upsampled.npy"]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float32)
    return None


def _extract_instance_masks_from_thresholded_score(
    score_map: np.ndarray,
    threshold: float,
    *,
    threshold_mode: str,
    threshold_offset: float,
    prompt_erode_iterations: int,
    instance_mean_score_threshold: float,
    instance_mean_score_threshold_mode: str,
    instance_mean_score_threshold_offset: float,
) -> Tuple[List[np.ndarray], float]:
    if score_map.ndim != 2:
        raise ValueError(f"Expected 2D score map, got {score_map.shape}")
    binary, threshold_value = binary_mask_from_score(
        np.asarray(score_map, dtype=np.float32),
        threshold=float(threshold),
        threshold_mode=str(threshold_mode),
        threshold_offset=float(threshold_offset),
    )
    binary = erode_binary_mask(binary, iterations=prompt_erode_iterations)
    labels, num_instances = connected_components_4(binary)
    if num_instances <= 0:
        return [], float(threshold_value)
    instance_masks: List[np.ndarray] = []
    for label in range(1, num_instances + 1):
        mask = (labels == label).astype(np.uint8)
        if np.any(mask):
            instance_masks.append(mask)
    mean_threshold_value = 0.0
    if instance_masks and (
        instance_mean_score_threshold > 0.0
        or instance_mean_score_threshold_mode == "otsu"
        or instance_mean_score_threshold_offset != 0.0
    ):
        _unused_binary, mean_threshold_value = binary_mask_from_score(
            np.asarray(score_map, dtype=np.float32),
            threshold=float(instance_mean_score_threshold),
            threshold_mode=str(instance_mean_score_threshold_mode),
            threshold_offset=float(instance_mean_score_threshold_offset),
        )
    if mean_threshold_value > 0.0 and instance_masks:
        filtered_masks: List[np.ndarray] = []
        for mask in instance_masks:
            masked_scores = score_map[mask > 0]
            if masked_scores.size == 0:
                continue
            if float(masked_scores.mean()) >= mean_threshold_value:
                filtered_masks.append(mask)
        instance_masks = filtered_masks
    instance_masks.sort(key=lambda m: int(np.count_nonzero(m)), reverse=True)
    return instance_masks, float(threshold_value)


def collect_frame_instance_prompts(mask_root: Path, args: argparse.Namespace) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, float]]:
    frame_dirs = sorted([path for path in mask_root.iterdir() if path.is_dir() and path.name.startswith("frame_")], key=_frame_sort_key)
    frame_prompts: Dict[int, List[np.ndarray]] = {}
    prompt_threshold_values: Dict[int, float] = {}
    for frame_dir in frame_dirs:
        frame_idx = _frame_sort_key(frame_dir)
        score_map = _load_score_map(frame_dir, args.prompt_score_source)
        if score_map is None:
            masks = []
        else:
            masks, threshold_value = _extract_instance_masks_from_thresholded_score(
                score_map,
                args.prompt_threshold,
                threshold_mode=args.prompt_threshold_mode,
                threshold_offset=args.prompt_threshold_offset,
                prompt_erode_iterations=args.prompt_erode_iterations,
                instance_mean_score_threshold=args.instance_mean_score_threshold,
                instance_mean_score_threshold_mode=args.instance_mean_score_threshold_mode,
                instance_mean_score_threshold_offset=args.instance_mean_score_threshold_offset,
            )
            prompt_threshold_values[frame_idx] = float(threshold_value)

        if masks:
            frame_prompts[frame_idx] = [m.astype(np.uint8) for m in masks]
    return frame_prompts, prompt_threshold_values


def collect_frame_indices(mask_root: Path) -> List[int]:
    frame_dirs = sorted([path for path in mask_root.iterdir() if path.is_dir() and path.name.startswith("frame_")], key=_frame_sort_key)
    return [_frame_sort_key(path) for path in frame_dirs]


def build_windows(frame_indices: List[int], window_size: int, window_overlap: int) -> List[Tuple[int, int]]:
    if not frame_indices:
        return []
    if window_size <= 0:
        return [(frame_indices[0], frame_indices[-1])]
    if window_overlap < 0 or window_overlap >= window_size:
        raise ValueError("--window_overlap must satisfy 0 <= overlap < window_size")
    step = window_size - window_overlap
    windows: List[Tuple[int, int]] = []
    start = frame_indices[0]
    last = frame_indices[-1]
    while start <= last:
        end = min(start + window_size - 1, last)
        windows.append((start, end))
        if end >= last:
            break
        start += step
    return windows


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _should_drop_by_area(pred: np.ndarray, prompt: np.ndarray, max_area_increase: Optional[float]) -> bool:
    if max_area_increase is None or max_area_increase < 0:
        return False
    prompt_area = _mask_area(prompt)
    if prompt_area <= 0:
        return False
    pred_area = _mask_area(pred)
    return pred_area > prompt_area * (1.0 + max_area_increase)


def save_sam2_masks(
    output_dir: Path,
    video_segments: Dict[int, np.ndarray],
    *,
    mask_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, mask in sorted(video_segments.items()):
        binary_mask = (mask > 0).astype(np.uint8)
        binary = binary_mask * 255
        Image.fromarray(binary).save(output_dir / f"frame_{frame_idx:05d}.png")
        np.save(output_dir / f"frame_{frame_idx:05d}.npy", binary_mask)
        rgb = _load_overlay_image_for_frame(mask_root, frame_idx)
        if rgb is not None and rgb.shape[:2] != binary_mask.shape:
            input_image = mask_root / f"frame_{frame_idx:05d}" / "input_image.png"
            if input_image.exists():
                fallback_rgb = np.array(Image.open(input_image).convert("RGB"))
                if fallback_rgb.shape[:2] == binary_mask.shape:
                    rgb = fallback_rgb
        if rgb is not None:
            if rgb.shape[:2] == binary_mask.shape:
                overlay = _overlay_binary_mask_on_image(rgb, binary_mask)
                Image.fromarray(overlay).save(output_dir / f"frame_{frame_idx:05d}_overlay.png")


def write_sam2_meta(
    output_dir: Path,
    *,
    repo_root: Path,
    sam2_cfg: Path,
    sam2_cfg_name: str,
    sam2_checkpoint: Path,
    mask_root: Path,
    video_path: str,
    score_thresh: float,
    prompt_score_source: str,
    prompt_threshold: float,
    prompt_threshold_mode: str,
    prompt_threshold_offset: float,
    prompt_erode_iterations: int,
    instance_mean_score_threshold: float,
    instance_mean_score_threshold_mode: str,
    instance_mean_score_threshold_offset: float,
    propagate_mode: str,
    max_area_increase: Optional[float],
    prompt_frame_counts: Dict[int, int],
    prompt_threshold_values: Dict[int, float],
    windows: List[Tuple[int, int]],
    window_size: int,
    window_overlap: int,
    window_timings: List[Dict[str, float | int]],
    video_segments: Dict[int, np.ndarray],
    total_refine_seconds: float,
    status: str,
    variant_name: str,
    cache_image_features: bool,
    runtime_profile: Optional[Dict[str, object]] = None,
) -> None:
    num_output_frames = int(len(video_segments))
    per_frame_refine_seconds = float(total_refine_seconds / max(num_output_frames, 1))
    meta = {
        "variant": str(variant_name),
        "status": status,
        "mask_root": str(mask_root),
        "video_path": video_path,
        "sam2_repo_root": str(repo_root),
        "sam2_cfg": str(sam2_cfg),
        "sam2_cfg_name": sam2_cfg_name,
        "sam2_checkpoint": str(sam2_checkpoint),
        "score_thresh": float(score_thresh),
        "prompt_score_source": str(prompt_score_source),
        "prompt_threshold": float(prompt_threshold),
        "prompt_threshold_mode": str(prompt_threshold_mode),
        "prompt_threshold_offset": float(prompt_threshold_offset),
        "prompt_erode_iterations": int(prompt_erode_iterations),
        "instance_mean_score_threshold": float(instance_mean_score_threshold),
        "instance_mean_score_threshold_mode": str(instance_mean_score_threshold_mode),
        "instance_mean_score_threshold_offset": float(instance_mean_score_threshold_offset),
        "propagate_mode": str(propagate_mode),
        "cache_image_features": bool(cache_image_features),
        "max_area_increase": None if max_area_increase is None else float(max_area_increase),
        "prompt_frame_counts": {
            f"frame_{frame_idx:05d}": int(count) for frame_idx, count in sorted(prompt_frame_counts.items())
        },
        "prompt_threshold_values": {
            f"frame_{frame_idx:05d}": float(value) for frame_idx, value in sorted(prompt_threshold_values.items())
        },
        "num_output_frames": num_output_frames,
        "window_size": int(window_size),
        "window_overlap": int(window_overlap),
        "windows": [[int(start), int(end)] for start, end in windows],
        "window_timings_seconds": window_timings,
        "timing_seconds": {
            "additional_motion_mask_inference_total": total_refine_seconds,
            "additional_motion_mask_inference_per_frame": per_frame_refine_seconds,
        },
        "frame_timings_seconds": {
            f"frame_{frame_idx:05d}": per_frame_refine_seconds
            for frame_idx in sorted(video_segments)
        },
    }
    if runtime_profile is not None:
        meta["runtime_profile"] = runtime_profile
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _add_prompt_masks(
    predictor,
    inference_state,
    prompt_entries: Sequence[Tuple[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    prompt_areas: Dict[int, np.ndarray] = {}
    obj_id = 1
    for frame_idx, mask in prompt_entries:
        mask_tensor = torch.from_numpy(mask.astype(np.float32, copy=False))
        with torch.inference_mode(), _sam2_autocast(getattr(predictor, "device", torch.device("cpu"))):
            predictor.add_new_mask(
                inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=mask_tensor,
            )
        prompt_areas[obj_id] = mask.astype(np.uint8, copy=False)
        obj_id += 1
    return prompt_areas


def _build_prompt_entries(
    frame_prompts: Dict[int, List[np.ndarray]],
    frame_indices: Sequence[int],
    *,
    skip_parity: Optional[int],
) -> List[Tuple[int, np.ndarray]]:
    prompt_entries: List[Tuple[int, np.ndarray]] = []
    for frame_idx in frame_indices:
        if skip_parity is not None and frame_idx % 2 == skip_parity:
            continue
        for mask in frame_prompts.get(frame_idx, []):
            prompt_entries.append((int(frame_idx), mask.astype(np.uint8, copy=False)))
    return prompt_entries


def _merge_propagated_masks(
    output_masks: Dict[int, np.ndarray],
    out_frame_idx: int,
    out_obj_ids,
    out_mask_logits,
    prompt_areas: Dict[int, np.ndarray],
    *,
    score_thresh: float,
    max_area_increase: Optional[float],
) -> None:
    frame_mask = output_masks.get(int(out_frame_idx))
    if frame_mask is None:
        frame_mask = np.zeros(out_mask_logits.shape[-2:], dtype=np.uint8)
    for i, out_obj_id in enumerate(out_obj_ids):
        pred = (out_mask_logits[i] > score_thresh).detach().cpu().numpy()
        if pred.ndim == 3:
            pred = pred[0]
        pred = pred.astype(np.uint8)
        prompt = prompt_areas.get(int(out_obj_id))
        if prompt is not None and _should_drop_by_area(pred, prompt, max_area_increase):
            continue
        frame_mask = np.maximum(frame_mask, pred)
    output_masks[int(out_frame_idx)] = frame_mask


def run_window(
    *,
    predictor,
    args: argparse.Namespace,
    video_path: str,
    window_start: int,
    window_end: int,
    frame_prompts: Dict[int, List[np.ndarray]],
    mask_root: Optional[Path] = None,
    frame_compute_seconds: Optional[Dict[int, float]] = None,
) -> Dict[int, np.ndarray]:
    window_frame_indices = list(range(int(window_start), int(window_end) + 1))
    prompt_global_frame_indices = [idx for idx in sorted(frame_prompts) if window_start <= idx <= window_end]
    if not prompt_global_frame_indices:
        return {}

    output_masks: Dict[int, np.ndarray] = {}
    pad_to_length = 0
    if int(args.window_size) > 0:
        pad_to_length = int(args.window_size)
    tmpdir, global_to_local, local_to_global, local_video_path = _build_window_video_dir(
        mask_root=Path(args.mask_root) if mask_root is None else mask_root,
        global_frame_indices=window_frame_indices,
        pad_to_length=pad_to_length,
    )
    try:
        local_frame_prompts = {
            global_to_local[global_idx]: masks
            for global_idx, masks in frame_prompts.items()
            if global_idx in global_to_local
        }
        local_prompt_frame_indices = sorted(local_frame_prompts)
        if args.propagate_mode == "single":
            pass_specs: List[Tuple[Optional[int], Optional[int]]] = [(None, None)]
        else:
            available_prompt_parities = {int(frame_idx % 2) for frame_idx in local_prompt_frame_indices}
            if len(available_prompt_parities) < 2:
                pass_specs = [(None, None)]
            else:
                pass_specs = [(0, 0), (1, 1)]
        device = getattr(predictor, "device", torch.device("cpu"))
        with torch.inference_mode(), _sam2_autocast(device):
            inference_state = predictor.init_state(
                video_path=local_video_path,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
                async_loading_frames=args.async_loading_frames,
            )
            if args.cache_image_features:
                feature_cache = {}
                for frame_idx in range(len(local_to_global)):
                    predictor._get_image_feature(
                        inference_state,
                        frame_idx=frame_idx,
                        batch_size=1,
                    )
                    feature_cache[frame_idx] = inference_state["cached_features"][frame_idx]
                inference_state["cached_features"] = feature_cache

        for skip_parity, output_parity in pass_specs:
            prompt_entries = _build_prompt_entries(
                local_frame_prompts,
                local_prompt_frame_indices,
                skip_parity=skip_parity,
            )
            if not prompt_entries:
                continue
            with torch.inference_mode(), _sam2_autocast(device):
                predictor.reset_state(inference_state)
            prompt_areas = _add_prompt_masks(
                predictor,
                inference_state,
                prompt_entries,
            )
            if not prompt_areas:
                continue
            start_frame_idx = 0
            max_frames = int(len(local_to_global))
            with torch.inference_mode(), _sam2_autocast(device):
                iterator = predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frames,
                )
                while True:
                    _synchronize(device)
                    frame_t0 = perf_counter()
                    try:
                        out_frame_idx, out_obj_ids, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    _synchronize(device)
                    if out_frame_idx < 0 or out_frame_idx >= len(local_to_global):
                        continue
                    global_out_frame_idx = local_to_global[int(out_frame_idx)]
                    if global_out_frame_idx is None:
                        continue
                    if output_parity is not None and global_out_frame_idx % 2 != output_parity:
                        continue
                    _merge_propagated_masks(
                        output_masks,
                        int(global_out_frame_idx),
                        out_obj_ids,
                        out_mask_logits,
                        prompt_areas,
                        score_thresh=args.score_thresh,
                        max_area_increase=args.max_area_increase,
                    )
                    if frame_compute_seconds is not None:
                        elapsed = float(perf_counter() - frame_t0)
                        frame_compute_seconds[int(global_out_frame_idx)] = (
                            frame_compute_seconds.get(int(global_out_frame_idx), 0.0) + elapsed
                        )
    finally:
        tmpdir.cleanup()

    return output_masks


def build_runtime_profile(
    *,
    window_timings: Sequence[Dict[str, float | int]],
    frame_compute_seconds: Dict[int, float],
    peak_vram_gib: float,
    warmup_frames: int,
) -> Dict[str, object]:
    total_seconds = float(sum(float(item["seconds"]) for item in window_timings))
    raw_total = float(sum(frame_compute_seconds.values()))
    frame_count = len(frame_compute_seconds)
    overhead_per_frame = max(total_seconds - raw_total, 0.0) / max(frame_count, 1)
    amortized = {
        f"frame_{frame_idx:05d}": float(seconds + overhead_per_frame)
        for frame_idx, seconds in sorted(frame_compute_seconds.items())
    }
    values = np.asarray(list(amortized.values()), dtype=np.float64)
    return {
        "timing_scope": (
            "SAM2 sequence inference including frame decode, state initialization, prompting, and propagation; "
            "model loading and output-mask writes excluded"
        ),
        "warmup_frames": int(warmup_frames),
        "measured_output_frames": int(frame_count),
        "sequence_total_seconds": total_seconds,
        "sequence_overhead_seconds": float(max(total_seconds - raw_total, 0.0)),
        "amortized_overhead_per_frame_seconds": float(overhead_per_frame),
        "frame_compute_seconds_raw": {
            f"frame_{frame_idx:05d}": float(seconds)
            for frame_idx, seconds in sorted(frame_compute_seconds.items())
        },
        "frame_latency_seconds_amortized": amortized,
        "mean_ms": float(values.mean() * 1000.0) if values.size else 0.0,
        "median_ms": float(np.median(values) * 1000.0) if values.size else 0.0,
        "p95_ms": float(np.percentile(values, 95) * 1000.0) if values.size else 0.0,
        "peak_vram_gib": float(peak_vram_gib),
    }


def main() -> None:
    args = parse_args()
    mask_root = Path(args.mask_root)
    repo_root = Path(args.sam2_repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not repo_root.exists() and (PROJECT_ROOT / "sam2").exists():
        repo_root = PROJECT_ROOT / "sam2"
    sam2_cfg = _resolve_existing_path(args.sam2_cfg, repo_root, "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    sam2_checkpoint = _resolve_existing_path(args.sam2_checkpoint, repo_root, "checkpoints/sam2.1_hiera_large.pt")
    sam2_cfg_name = _to_hydra_config_name(sam2_cfg, repo_root, args.sam2_cfg)

    if not mask_root.exists():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    if not sam2_cfg.exists():
        raise FileNotFoundError(f"SAM2 config not found: {sam2_cfg}")
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")
    if args.window_size > 0 and args.window_overlap >= args.window_size:
        raise ValueError("--window_overlap must be smaller than --window_size")
    frame_prompts, prompt_threshold_values = collect_frame_instance_prompts(mask_root, args)
    frame_indices = collect_frame_indices(mask_root)
    if not frame_indices:
        raise RuntimeError(f"No frame_* directories found under {mask_root}")
    if not frame_prompts:
        write_sam2_meta(
            output_dir,
            repo_root=repo_root,
            sam2_cfg=sam2_cfg,
            sam2_cfg_name=sam2_cfg_name,
            sam2_checkpoint=sam2_checkpoint,
            mask_root=mask_root,
            video_path=args.video_path,
            score_thresh=args.score_thresh,
            prompt_score_source=args.prompt_score_source,
            prompt_threshold=args.prompt_threshold,
            prompt_threshold_mode=args.prompt_threshold_mode,
            prompt_threshold_offset=args.prompt_threshold_offset,
            prompt_erode_iterations=args.prompt_erode_iterations,
            instance_mean_score_threshold=args.instance_mean_score_threshold,
            instance_mean_score_threshold_mode=args.instance_mean_score_threshold_mode,
            instance_mean_score_threshold_offset=args.instance_mean_score_threshold_offset,
            propagate_mode=args.propagate_mode,
            max_area_increase=args.max_area_increase,
            prompt_frame_counts={},
            prompt_threshold_values=prompt_threshold_values,
            windows=[],
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            window_timings=[],
            video_segments={},
            total_refine_seconds=0.0,
            status="empty_prompt",
            variant_name=args.variant_name,
            cache_image_features=args.cache_image_features,
        )
        print(f"[warn] No non-empty prompt instances found under {mask_root}; wrote empty_prompt meta to {output_dir}")
        return
    windows = build_windows(frame_indices, int(args.window_size), int(args.window_overlap))

    build_sam2_video_predictor = _import_sam2_build(repo_root)
    device = torch.device(args.device)
    predictor = build_sam2_video_predictor(
        config_file=sam2_cfg_name,
        ckpt_path=str(sam2_checkpoint),
        device=device,
        hydra_overrides_extra=(
            ["++model.compile_image_encoder=True"]
            if args.compile_image_encoder
            else []
        ),
    )

    warmup_frames = 0
    if args.runtime_warmup_mask_root:
        warmup_root = Path(args.runtime_warmup_mask_root)
        warmup_prompts, _ = collect_frame_instance_prompts(warmup_root, args)
        warmup_indices = collect_frame_indices(warmup_root)
        if warmup_indices and warmup_prompts:
            warmup_frames = len(warmup_indices)
            _synchronize(device)
            run_window(
                predictor=predictor,
                args=args,
                video_path=args.video_path,
                window_start=warmup_indices[0],
                window_end=warmup_indices[-1],
                frame_prompts=warmup_prompts,
                mask_root=warmup_root,
            )
            _synchronize(device)

    prompt_frame_counts = {frame_idx: len(masks) for frame_idx, masks in frame_prompts.items()}
    video_segments: Dict[int, np.ndarray] = {}
    window_timings: List[Dict[str, float | int]] = []
    frame_compute_seconds: Dict[int, float] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    total_refine_t0 = perf_counter()

    for window_start, window_end in windows:
        window_frame_indices = [idx for idx in sorted(frame_prompts) if window_start <= idx <= window_end]
        _synchronize(device)
        window_t0 = perf_counter()
        window_segments = run_window(
            predictor=predictor,
            args=args,
            video_path=args.video_path,
            window_start=window_start,
            window_end=window_end,
            frame_prompts=frame_prompts,
            mask_root=mask_root,
            frame_compute_seconds=frame_compute_seconds,
        )
        _synchronize(device)
        window_seconds = float(perf_counter() - window_t0)
        window_timings.append(
            {
                "start_frame": int(window_start),
                "end_frame": int(window_end),
                "num_prompt_frames": int(len(window_frame_indices)),
                "num_prompt_instances": int(sum(prompt_frame_counts.get(idx, 0) for idx in window_frame_indices)),
                "num_output_frames": int(len(window_segments)),
                "seconds": window_seconds,
            }
        )
        video_segments.update(window_segments)
        if window_segments:
            save_sam2_masks(
                output_dir,
                window_segments,
                mask_root=mask_root,
            )
            write_sam2_meta(
                output_dir,
                repo_root=repo_root,
                sam2_cfg=sam2_cfg,
                sam2_cfg_name=sam2_cfg_name,
                sam2_checkpoint=sam2_checkpoint,
                mask_root=mask_root,
                video_path=args.video_path,
                score_thresh=args.score_thresh,
                prompt_score_source=args.prompt_score_source,
                prompt_threshold=args.prompt_threshold,
                prompt_threshold_mode=args.prompt_threshold_mode,
                prompt_threshold_offset=args.prompt_threshold_offset,
                prompt_erode_iterations=args.prompt_erode_iterations,
                instance_mean_score_threshold=args.instance_mean_score_threshold,
                instance_mean_score_threshold_mode=args.instance_mean_score_threshold_mode,
                instance_mean_score_threshold_offset=args.instance_mean_score_threshold_offset,
                propagate_mode=args.propagate_mode,
                max_area_increase=args.max_area_increase,
                prompt_frame_counts=prompt_frame_counts,
                prompt_threshold_values=prompt_threshold_values,
                windows=windows,
                window_size=args.window_size,
                window_overlap=args.window_overlap,
                window_timings=window_timings,
                video_segments=video_segments,
                total_refine_seconds=float(perf_counter() - total_refine_t0),
                status="partial",
                variant_name=args.variant_name,
                cache_image_features=args.cache_image_features,
            )

    total_refine_seconds = float(perf_counter() - total_refine_t0)
    peak_vram_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**3))
        if device.type == "cuda"
        else 0.0
    )
    runtime_profile = build_runtime_profile(
        window_timings=window_timings,
        frame_compute_seconds=frame_compute_seconds,
        peak_vram_gib=peak_vram_gib,
        warmup_frames=warmup_frames,
    )
    if not video_segments:
        write_sam2_meta(
            output_dir,
            repo_root=repo_root,
            sam2_cfg=sam2_cfg,
            sam2_cfg_name=sam2_cfg_name,
            sam2_checkpoint=sam2_checkpoint,
            mask_root=mask_root,
            video_path=args.video_path,
            score_thresh=args.score_thresh,
            prompt_score_source=args.prompt_score_source,
            prompt_threshold=args.prompt_threshold,
            prompt_threshold_mode=args.prompt_threshold_mode,
            prompt_threshold_offset=args.prompt_threshold_offset,
            prompt_erode_iterations=args.prompt_erode_iterations,
            instance_mean_score_threshold=args.instance_mean_score_threshold,
            instance_mean_score_threshold_mode=args.instance_mean_score_threshold_mode,
            instance_mean_score_threshold_offset=args.instance_mean_score_threshold_offset,
            propagate_mode=args.propagate_mode,
            max_area_increase=args.max_area_increase,
            prompt_frame_counts=prompt_frame_counts,
            prompt_threshold_values=prompt_threshold_values,
            windows=windows,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            window_timings=window_timings,
            video_segments={},
            total_refine_seconds=total_refine_seconds,
            status="empty_output",
            variant_name=args.variant_name,
            cache_image_features=args.cache_image_features,
            runtime_profile=runtime_profile,
        )
        print(f"[warn] SAM2 propagation produced no output masks for {mask_root}; wrote empty_output meta to {output_dir}")
        return

    save_sam2_masks(
        output_dir,
        video_segments,
        mask_root=mask_root,
    )
    write_sam2_meta(
        output_dir,
        repo_root=repo_root,
        sam2_cfg=sam2_cfg,
        sam2_cfg_name=sam2_cfg_name,
        sam2_checkpoint=sam2_checkpoint,
        mask_root=mask_root,
        video_path=args.video_path,
        score_thresh=args.score_thresh,
        prompt_score_source=args.prompt_score_source,
        prompt_threshold=args.prompt_threshold,
        prompt_threshold_mode=args.prompt_threshold_mode,
        prompt_threshold_offset=args.prompt_threshold_offset,
        prompt_erode_iterations=args.prompt_erode_iterations,
        instance_mean_score_threshold=args.instance_mean_score_threshold,
        instance_mean_score_threshold_mode=args.instance_mean_score_threshold_mode,
        instance_mean_score_threshold_offset=args.instance_mean_score_threshold_offset,
        propagate_mode=args.propagate_mode,
        max_area_increase=args.max_area_increase,
        prompt_frame_counts=prompt_frame_counts,
        prompt_threshold_values=prompt_threshold_values,
        windows=windows,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        window_timings=window_timings,
        video_segments=video_segments,
        total_refine_seconds=total_refine_seconds,
        status="complete",
        variant_name=args.variant_name,
        cache_image_features=args.cache_image_features,
        runtime_profile=runtime_profile,
    )
    print(f"Saved SAM2-refined masks under: {output_dir}")


if __name__ == "__main__":
    main()
