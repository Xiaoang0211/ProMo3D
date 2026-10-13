#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from isdynamic.core.preprocess_aspect_pad import aspect_pad_arrays
from isdynamic.viz.motion_mask_benchmarks import (
    BENCHMARK_CHOICES,
    list_sequence_specs,
    load_sequence_paths_from_spec,
    select_context_indices,
)


MODEL_CACHE: Dict[tuple, object] = {}


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def evenly_spaced_indices(count: int, max_count: int) -> List[int]:
    if count <= 0:
        return []
    if max_count <= 0 or max_count >= count:
        return list(range(count))
    return sorted(set(int(round(x)) for x in np.linspace(0, count - 1, max_count)))


def tensor_to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def squeeze_depth(depth) -> np.ndarray | None:
    arr = tensor_to_numpy(depth)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0][None]
    if arr.ndim == 2:
        arr = arr[None]
    return arr.astype(np.float32, copy=False)


def to_homogeneous_poses(mats) -> np.ndarray | None:
    arr = tensor_to_numpy(mats)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[-2:] == (4, 4):
        return arr
    if arr.shape[-2:] == (3, 4):
        out = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
        out[:, :3, :4] = arr
        return out
    return None


def resize_depth_stack(depth: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(depth[:, None].astype(np.float32))
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized[:, 0].numpy()


def depth_consistency_metrics(reference: np.ndarray | None, candidate: np.ndarray | None, *, size: int) -> Dict[str, float]:
    if reference is None or candidate is None:
        return {}
    frames = min(reference.shape[0], candidate.shape[0])
    if frames <= 0:
        return {}
    ref = resize_depth_stack(reference[:frames], size=size)
    cand = resize_depth_stack(candidate[:frames], size=size)
    mask = np.isfinite(ref) & np.isfinite(cand) & (ref > 1e-6) & (cand > 1e-6)
    if int(mask.sum()) < 32:
        return {}
    scale = float(np.median(ref[mask]) / max(np.median(cand[mask]), 1e-6))
    cand_aligned = cand * scale
    ref_valid = ref[mask]
    cand_valid = cand_aligned[mask]
    log_diff = np.log(np.maximum(ref_valid, 1e-6)) - np.log(np.maximum(cand_valid, 1e-6))
    abs_rel = np.mean(np.abs(ref_valid - cand_valid) / np.maximum(ref_valid, 1e-6))
    rmse_log = math.sqrt(float(np.mean(log_diff * log_diff)))
    if ref_valid.size > 1 and np.std(ref_valid) > 1e-9 and np.std(cand_valid) > 1e-9:
        pearson = float(np.corrcoef(ref_valid, cand_valid)[0, 1])
    else:
        pearson = float("nan")
    return {
        "depth_absrel_median_aligned": float(abs_rel),
        "depth_rmse_log": float(rmse_log),
        "depth_pearson": pearson,
        "depth_median_scale_aspect_to_native": scale,
        "depth_valid_pixels": int(mask.sum()),
    }


def relative_pose_stack(poses: np.ndarray | None) -> np.ndarray | None:
    if poses is None or poses.shape[0] < 2:
        return None
    try:
        inv0 = np.linalg.inv(poses[0])
    except np.linalg.LinAlgError:
        return None
    return np.stack([poses[i] @ inv0 for i in range(1, poses.shape[0])], axis=0)


def rotation_angle_deg(rotation_delta: np.ndarray) -> float:
    trace = float(np.trace(rotation_delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos_angle))


def pose_consistency_metrics(reference: np.ndarray | None, candidate: np.ndarray | None) -> Dict[str, float]:
    ref_rel = relative_pose_stack(reference)
    cand_rel = relative_pose_stack(candidate)
    if ref_rel is None or cand_rel is None:
        return {}
    count = min(ref_rel.shape[0], cand_rel.shape[0])
    if count <= 0:
        return {}
    rot_errors = []
    trans_cos_errors = []
    for ref_pose, cand_pose in zip(ref_rel[:count], cand_rel[:count]):
        r_delta = cand_pose[:3, :3] @ ref_pose[:3, :3].T
        rot_errors.append(rotation_angle_deg(r_delta))
        ref_t = ref_pose[:3, 3]
        cand_t = cand_pose[:3, 3]
        ref_norm = np.linalg.norm(ref_t)
        cand_norm = np.linalg.norm(cand_t)
        if ref_norm > 1e-9 and cand_norm > 1e-9:
            cos = float(np.dot(ref_t, cand_t) / (ref_norm * cand_norm))
            trans_cos_errors.append(1.0 - max(-1.0, min(1.0, cos)))
    metrics = {
        "pose_rel_rot_deg_mean": float(np.mean(rot_errors)),
        "pose_rel_rot_deg_max": float(np.max(rot_errors)),
    }
    if trans_cos_errors:
        metrics["pose_trans_dir_cos_error_mean"] = float(np.mean(trans_cos_errors))
    return metrics


def load_vggt_predictions(args: argparse.Namespace, image_paths: Sequence[Path], mask_refs: Sequence, mode: str) -> Dict[str, np.ndarray | None]:
    from isdynamic.legacy.vggt.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref
    from isdynamic.legacy.vggt.build_davis_isdynamic_npz import VGGT, _extract_state_dict, _strip_module_prefix
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(args.device)
    cache_key = ("vggt", str(Path(args.vggt_checkpoint).resolve()), str(device))
    if cache_key not in MODEL_CACHE:
        model = VGGT(enable_camera=True, enable_point=False, enable_depth=True, enable_track=False).to(device)
        ckpt = torch.load(Path(args.vggt_checkpoint), map_location="cpu")
        model.load_state_dict(_strip_module_prefix(_extract_state_dict(ckpt)), strict=False)
        model.eval()
        MODEL_CACHE[cache_key] = model
    model = MODEL_CACHE[cache_key]
    tensors = [
        preprocess_image_and_mask_ref(image_path, mask_ref, preprocess_mode=mode)[0]
        for image_path, mask_ref in zip(image_paths, mask_refs)
    ]
    images = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model(images)
    depth = squeeze_depth(pred.get("depth"))
    pose_enc = pred.get("pose_enc")
    extrinsics = None
    if pose_enc is not None:
        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    return {"depth": depth, "pose": to_homogeneous_poses(extrinsics)}


def load_vggt_omega_predictions(args: argparse.Namespace, image_paths: Sequence[Path], mask_refs: Sequence, mode: str) -> Dict[str, np.ndarray | None]:
    from isdynamic.legacy.vggt_omega.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref
    from isdynamic.models.vggt_omega import load_vggt_omega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = torch.device(args.device)
    cache_key = ("vggt_omega", str(Path(args.vggt_omega_checkpoint).resolve()), str(device))
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_vggt_omega(Path(args.vggt_omega_checkpoint), device=device, enable_camera=True, enable_depth=True)
    model = MODEL_CACHE[cache_key]
    tensors = [
        preprocess_image_and_mask_ref(image_path, mask_ref, preprocess_mode=mode)[0]
        for image_path, mask_ref in zip(image_paths, mask_refs)
    ]
    images = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model(images)
    depth = squeeze_depth(pred.get("depth"))
    extrinsics = None
    if pred.get("pose_enc") is not None:
        extrinsics, _ = encoding_to_camera(pred["pose_enc"], images.shape[-2:])
    return {"depth": depth, "pose": to_homogeneous_poses(extrinsics)}


def aspect_padded_pils(image_paths: Sequence[Path], *, target_size: int, patch_size: int) -> List[Image.Image]:
    out = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        image_arr, _ = aspect_pad_arrays(image, mask, target_size=target_size, patch_size=patch_size)
        out.append(Image.fromarray(image_arr))
    return out


def load_da3_predictions(args: argparse.Namespace, image_paths: Sequence[Path], _mask_refs: Sequence, mode: str) -> Dict[str, np.ndarray | None]:
    from isdynamic.models.depth_anything_3 import load_da3_model

    device = torch.device(args.device)
    cache_key = ("depth_anything_3", str(Path(args.da3_model_dir).resolve()), str(device))
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_da3_model(args.da3_model_dir, device=device)
    model = MODEL_CACHE[cache_key]
    if mode == "aspect_pad":
        images = aspect_padded_pils(image_paths, target_size=args.da3_process_res, patch_size=14)
    else:
        images = [str(path) for path in image_paths]
    pred = model.inference(
        images,
        process_res=args.da3_process_res,
        process_res_method=args.da3_process_res_method,
        export_format="mini_npz",
    )
    return {"depth": squeeze_depth(pred.depth), "pose": to_homogeneous_poses(pred.extrinsics)}


def load_mapanything_predictions(args: argparse.Namespace, image_paths: Sequence[Path], _mask_refs: Sequence, mode: str) -> Dict[str, np.ndarray | None]:
    from isdynamic.models.map_anything import load_mapanything
    from mapanything.utils.image import load_images

    device = torch.device(args.device)
    cache_key = ("map_anything", str(Path(args.mapanything_checkpoint).resolve()), args.mapanything_machine, str(device))
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_mapanything(args.mapanything_checkpoint, device=device, machine=args.mapanything_machine)
    model = MODEL_CACHE[cache_key]
    if mode == "aspect_pad":
        with tempfile.TemporaryDirectory(prefix="isdynamic_mapanything_aspectpad_") as tmp:
            tmp_dir = Path(tmp)
            padded_paths = []
            for idx, image in enumerate(aspect_padded_pils(image_paths, target_size=args.mapanything_resolution_set, patch_size=14)):
                out_path = tmp_dir / f"{idx:05d}.png"
                image.save(out_path)
                padded_paths.append(str(out_path))
            views = load_images(padded_paths, resolution_set=args.mapanything_resolution_set)
            outputs = model.infer(
                views,
                memory_efficient_inference=True,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=False,
                mask_edges=False,
                apply_confidence_mask=False,
            )
    else:
        views = load_images([str(path) for path in image_paths], resolution_set=args.mapanything_resolution_set)
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=False,
            mask_edges=False,
            apply_confidence_mask=False,
        )
    depths = []
    poses = []
    for pred in outputs:
        if "depth_z" in pred:
            depths.append(squeeze_depth(pred["depth_z"])[0])
        if "camera_poses" in pred:
            poses.append(to_homogeneous_poses(pred["camera_poses"])[0])
    return {
        "depth": np.stack(depths, axis=0) if depths else None,
        "pose": np.stack(poses, axis=0) if poses else None,
    }


def load_dust3r_predictions(args: argparse.Namespace, image_paths: Sequence[Path], mask_refs: Sequence, mode: str) -> Dict[str, np.ndarray | None]:
    from isdynamic.models.dust3r import DUST3R_PATCH_SIZE, collate_views, load_dust3r_model, preprocess_image_and_mask

    if len(image_paths) < 2:
        return {"depth": None, "pose": None}
    device = torch.device(args.device)
    cache_key = ("dust3r", str(Path(args.dust3r_weights).resolve()), str(device))
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_dust3r_model(args.dust3r_weights, device=device)
    model = MODEL_CACHE[cache_key]
    square_ok = args.dust3r_image_size == 224
    view1, _ = preprocess_image_and_mask(
        image_paths[0],
        mask_refs[0],
        image_size=args.dust3r_image_size,
        patch_size=DUST3R_PATCH_SIZE,
        square_ok=square_ok,
        preprocess_mode=mode,
    )
    view2, _ = preprocess_image_and_mask(
        image_paths[1],
        mask_refs[1],
        image_size=args.dust3r_image_size,
        patch_size=DUST3R_PATCH_SIZE,
        square_ok=square_ok,
        preprocess_mode=mode,
    )
    batch_view1 = {key: value.to(device) for key, value in collate_views([view1]).items()}
    batch_view2 = {key: value.to(device) for key, value in collate_views([view2]).items()}
    with torch.inference_mode():
        pred1, _pred2 = model(batch_view1, batch_view2)
    pts = pred1.get("pts3d")
    depth = None
    if pts is not None:
        depth = squeeze_depth(pts[..., 2])
    return {"depth": depth, "pose": None}


MODEL_LOADERS = {
    "vggt": load_vggt_predictions,
    "vggt_omega": load_vggt_omega_predictions,
    "depth_anything_3": load_da3_predictions,
    "da3": load_da3_predictions,
    "map_anything": load_mapanything_predictions,
    "mapanything": load_mapanything_predictions,
    "dust3r": load_dust3r_predictions,
}


def evaluate_frame(args: argparse.Namespace, model_name: str, spec, image_paths, mask_refs, frame_idx: int) -> Dict[str, object]:
    context_indices = select_context_indices(
        len(image_paths),
        frame_idx,
        total_frames=args.context_size,
        stride=args.context_stride,
    )
    context_images = [image_paths[idx] for idx in context_indices]
    context_masks = [mask_refs[idx] for idx in context_indices]
    loader = MODEL_LOADERS[model_name]
    native = loader(args, context_images, context_masks, "native")
    aspect = loader(args, context_images, context_masks, "aspect_pad")
    row: Dict[str, object] = {
        "model": model_name,
        "benchmark": spec.benchmark,
        "dataset": spec.dataset,
        "sequence": spec.sequence,
        "sequence_tag": spec.sequence_tag,
        "frame_idx": int(frame_idx),
        "context_indices": json.dumps([int(idx) for idx in context_indices]),
        "has_depth": native.get("depth") is not None and aspect.get("depth") is not None,
        "has_pose": native.get("pose") is not None and aspect.get("pose") is not None,
    }
    row.update(depth_consistency_metrics(native.get("depth"), aspect.get("depth"), size=args.depth_compare_size))
    row.update(pose_consistency_metrics(native.get("pose"), aspect.get("pose")))
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model",
        "benchmark",
        "dataset",
        "sequence",
        "sequence_tag",
        "frame_idx",
        "context_indices",
        "has_depth",
        "has_pose",
        "depth_absrel_median_aligned",
        "depth_rmse_log",
        "depth_pearson",
        "pose_rel_rot_deg_mean",
        "pose_rel_rot_deg_max",
        "pose_trans_dir_cos_error_mean",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["benchmark"])), []).append(row)
    summary = []
    metric_keys = [
        "depth_absrel_median_aligned",
        "depth_rmse_log",
        "depth_pearson",
        "pose_rel_rot_deg_mean",
        "pose_rel_rot_deg_max",
        "pose_trans_dir_cos_error_mean",
    ]
    for (model, benchmark), items in sorted(grouped.items()):
        out: Dict[str, object] = {
            "model": model,
            "benchmark": benchmark,
            "num_frames": len(items),
            "depth_frames": sum(bool(item.get("has_depth")) for item in items),
            "pose_frames": sum(bool(item.get("has_pose")) for item in items),
        }
        for key in metric_keys:
            values = [float(item[key]) for item in items if key in item and item[key] not in ("", None) and np.isfinite(float(item[key]))]
            if values:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_median"] = float(np.median(values))
        summary.append(out)
    return summary


def canonical_model_name(name: str) -> str:
    text = name.strip().lower().replace("-", "_")
    aliases = {
        "da3": "depth_anything_3",
        "depthanything3": "depth_anything_3",
        "mapanything": "map_anything",
    }
    return aliases.get(text, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native vs aspect-pad geometry outputs.")
    parser.add_argument("--models", type=str, default="vggt,vggt_omega,depth_anything_3,map_anything,dust3r")
    parser.add_argument("--benchmarks", type=str, default="davis_all,segtrackv2,fbms59")
    parser.add_argument("--davis_root", type=str, default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS")
    parser.add_argument("--segtrack_root", type=str, default="/home/xiaoang/Datasets/SegTrackv2")
    parser.add_argument("--fbms_root", type=str, default="/home/xiaoang/Datasets/FBMS-59")
    parser.add_argument("--out_dir", type=str, default="outputs_aspectpad/geometry_padding_stability")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--context_stride", type=int, default=2)
    parser.add_argument("--max_sequences", type=int, default=0, help="0 means all sequences.")
    parser.add_argument("--max_frames_per_sequence", type=int, default=5, help="0 means all annotated frames.")
    parser.add_argument("--depth_compare_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vggt_checkpoint", type=str, default="models/vggt/ckpt/model.pt")
    parser.add_argument("--vggt_omega_checkpoint", type=str, default="models/vggt-omega/checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--da3_model_dir", type=str, default="models/Depth-Anything-3/src/depth_anything_3/ckpt/DA3-giant-1.1")
    parser.add_argument("--da3_process_res", type=int, default=518)
    parser.add_argument("--da3_process_res_method", type=str, default="upper_bound_resize")
    parser.add_argument("--mapanything_checkpoint", type=str, default="models/map-anything/ckpt")
    parser.add_argument("--mapanything_machine", type=str, default="aws")
    parser.add_argument("--mapanything_resolution_set", type=int, default=518)
    parser.add_argument("--dust3r_weights", type=str, default="models/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--dust3r_image_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [canonical_model_name(model) for model in parse_csv(args.models)]
    benchmarks = parse_csv(args.benchmarks)
    for model in models:
        if model not in MODEL_LOADERS:
            raise ValueError(f"Unsupported model: {model}")
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_CHOICES:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

    all_rows: List[Dict[str, object]] = []
    out_dir = Path(args.out_dir)
    for model in models:
        model_rows: List[Dict[str, object]] = []
        for benchmark in benchmarks:
            specs = list_sequence_specs(
                benchmark,
                davis_root=args.davis_root,
                segtrack_root=args.segtrack_root,
                fbms_root=args.fbms_root,
            )
            if args.max_sequences > 0:
                specs = specs[: args.max_sequences]
            for spec in specs:
                image_paths, mask_refs, _sequence_tag = load_sequence_paths_from_spec(
                    spec,
                    davis_root=args.davis_root,
                    segtrack_root=args.segtrack_root,
                    fbms_root=args.fbms_root,
                    fbms_include_unannotated=False,
                )
                frame_indices = evenly_spaced_indices(len(image_paths), args.max_frames_per_sequence)
                for frame_idx in frame_indices:
                    try:
                        row = evaluate_frame(args, model, spec, image_paths, mask_refs, frame_idx)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        row = {
                            "model": model,
                            "benchmark": spec.benchmark,
                            "dataset": spec.dataset,
                            "sequence": spec.sequence,
                            "sequence_tag": spec.sequence_tag,
                            "frame_idx": int(frame_idx),
                            "error": repr(exc),
                        }
                    model_rows.append(row)
                    all_rows.append(row)
                    print(f"[{model}] {spec.benchmark}/{spec.sequence_tag} frame={frame_idx}")
        model_dir = out_dir / model
        write_csv(model_dir / "per_frame.csv", model_rows)
        write_csv(model_dir / "summary.csv", summarize(model_rows))
    write_csv(out_dir / "per_frame_all_models.csv", all_rows)
    write_csv(out_dir / "summary_all_models.csv", summarize(all_rows))


if __name__ == "__main__":
    main()
