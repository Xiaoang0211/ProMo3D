from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .motion_mask_benchmarks import (
    BENCHMARK_CHOICES,
    DEFAULT_DAVIS_ROOT,
    DEFAULT_FBMS_ROOT,
    DEFAULT_SEGTRACK_ROOT,
    benchmark_display_name,
    benchmark_output_dirname,
    list_sequence_specs,
    load_binary_mask,
    load_sequence_paths_from_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exported motion masks with the official DAVIS J/F definitions "
            "(region IoU and boundary F-measure) on the repository's binary foreground masks."
        )
    )
    parser.add_argument("--mask_root", type=str, required=True, help="Root directory, e.g. outputs/isdynamic_motion_masks.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True, choices=BENCHMARK_CHOICES)
    parser.add_argument("--variant", type=str, default="bilinear_upsample")
    parser.add_argument("--davis_root", type=str, default=str(DEFAULT_DAVIS_ROOT))
    parser.add_argument("--segtrack_root", type=str, default=str(DEFAULT_SEGTRACK_ROOT))
    parser.add_argument("--fbms_root", type=str, default=str(DEFAULT_FBMS_ROOT))
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--boundary_th", type=float, default=0.008, help="Official DAVIS boundary tolerance relative to image diagonal.")
    return parser.parse_args()


def _normalize_model_name(name: str) -> str:
    text = name.strip().lower()
    aliases = {
        "pi3": "pi3",
        "pi3x": "pi3x",
        "vggt": "vggt",
        "da3": "depth_anything_3",
        "depth_anything_3": "depth_anything_3",
        "mapanything": "map_anything",
        "map_anything": "map_anything",
        "dust3r": "dust3r",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported model: {name}")
    return aliases[text]


def _load_binary_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        with Image.open(path) as image:
            arr = np.array(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def _find_existing_path(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    names = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find predicted mask. Checked:\n{names}")


def _prediction_path_for_frame(variant_root: Path, variant: str, frame_idx: int) -> Path:
    frame_name = f"frame_{frame_idx:05d}"
    if variant.startswith("bilinear_upsample"):
        frame_dir = variant_root / frame_name
        return _find_existing_path(
            [
                frame_dir / "binary_mask_upsampled.npy",
                frame_dir / "binary_mask_upsampled.png",
                frame_dir / "binary_mask.npy",
                frame_dir / "binary_mask.png",
            ]
        )
    if variant.startswith("upsample_anything"):
        frame_dir = variant_root / frame_name
        return _find_existing_path(
            [
                frame_dir / "binary_mask_upsample_anything.npy",
                frame_dir / "binary_mask_upsample_anything.png",
                frame_dir / "binary_mask_upsampled.npy",
                frame_dir / "binary_mask_upsampled.png",
                frame_dir / "binary_mask.png",
            ]
        )
    return _find_existing_path(
        [
            variant_root / f"{frame_name}.npy",
            variant_root / f"{frame_name}.png",
        ]
    )


def _load_prediction_for_frame(variant_root: Path, variant: str, frame_idx: int, gt_mask: np.ndarray) -> np.ndarray:
    try:
        pred_path = _prediction_path_for_frame(variant_root=variant_root, variant=variant, frame_idx=frame_idx)
        return _load_binary_prediction(pred_path)
    except FileNotFoundError:
        if variant.startswith("sam2_prompt"):
            return np.zeros_like(gt_mask, dtype=np.uint8)
        raise


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resize_and_pad_binary_mask(
    mask: np.ndarray,
    *,
    target_hw: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    orig_h, orig_w = mask.shape[:2]
    target_h, target_w = target_hw
    if orig_h <= 0 or orig_w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    if orig_w >= orig_h:
        new_w = target_w
        new_h = int(round(orig_h * (new_w / orig_w) / max(patch_size, 1)) * max(patch_size, 1))
    else:
        new_h = target_h
        new_w = int(round(orig_w * (new_h / orig_h) / max(patch_size, 1)) * max(patch_size, 1))
    new_w = max(1, min(target_w, new_w))
    new_h = max(1, min(target_h, new_h))

    resized = Image.fromarray(mask * 255).resize((new_w, new_h), Image.Resampling.NEAREST)
    resized_arr = (np.array(resized) > 0).astype(np.uint8)

    pad_w = max(0, target_w - new_w)
    pad_h = max(0, target_h - new_h)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return np.pad(
        resized_arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    ).astype(np.uint8)


def _prediction_meta_path(variant_root: Path, pred_path: Path | None) -> Path | None:
    candidates: List[Path] = []
    if pred_path is not None:
        candidates.append(pred_path.parent / "meta.json")
    candidates.append(variant_root / "meta.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _shape_tail_hw(shape: object) -> Tuple[int, int] | None:
    if isinstance(shape, (list, tuple)):
        if len(shape) >= 2 and all(isinstance(v, (int, float)) for v in shape[-2:]):
            return (int(shape[-2]), int(shape[-1]))
        if len(shape) > 0:
            return _shape_tail_hw(shape[-1])
    return None


def _align_gt_to_prediction(
    gt_mask: np.ndarray,
    variant_root: Path,
    pred_path: Path | None,
    pred_mask: np.ndarray | None,
) -> np.ndarray:
    if pred_mask is None or gt_mask.shape == pred_mask.shape:
        return gt_mask
    if pred_path is None:
        return gt_mask

    meta_path = _prediction_meta_path(variant_root, pred_path)
    if meta_path is not None:
        meta = _load_json(meta_path)
        stack_shape = meta.get("context_stack_shape")
        patch_size = int(meta.get("patch_size", 14))
        target_hw = _shape_tail_hw(stack_shape)
        padded_gt: np.ndarray | None = None
        if target_hw is not None:
            target_h, target_w = target_hw
            if target_h > 0 and target_w > 0:
                padded_gt = _resize_and_pad_binary_mask(
                    gt_mask,
                    target_hw=(target_h, target_w),
                    patch_size=patch_size,
                )
                if tuple(padded_gt.shape) == tuple(pred_mask.shape):
                    return padded_gt
        bbox = meta.get("content_bbox_xyxy")
        content_hw = meta.get("content_hw")
        if (
            isinstance(content_hw, list)
            and len(content_hw) == 2
            and tuple(int(v) for v in content_hw) == tuple(pred_mask.shape)
        ):
            resized = Image.fromarray((gt_mask > 0).astype(np.uint8) * 255).resize(
                (pred_mask.shape[1], pred_mask.shape[0]),
                Image.Resampling.NEAREST,
            )
            return (np.array(resized) > 0).astype(np.uint8)
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and isinstance(content_hw, list)
            and len(content_hw) == 2
            and padded_gt is not None
        ):
            x0, y0, x1, y1 = (int(v) for v in bbox)
            cropped_gt = padded_gt[y0:y1, x0:x1]
            if tuple(cropped_gt.shape) == tuple(pred_mask.shape):
                return cropped_gt
        guidance_image_size = meta.get("guidance_image_size")
        if (
            isinstance(guidance_image_size, list)
            and len(guidance_image_size) == 2
            and tuple(int(v) for v in guidance_image_size) == tuple(pred_mask.shape)
        ):
            resized = Image.fromarray((gt_mask > 0).astype(np.uint8) * 255).resize(
                (pred_mask.shape[1], pred_mask.shape[0]),
                Image.Resampling.NEAREST,
            )
            return (np.array(resized) > 0).astype(np.uint8)

    return gt_mask


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg).astype(bool)
    if seg.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {seg.shape}")

    height, width = seg.shape
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    south_east = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    south_east[:-1, :-1] = seg[1:, 1:]

    bmap = (seg ^ east) | (seg ^ south) | (seg ^ south_east)
    bmap[-1, :] = seg[-1, :] ^ east[-1, :]
    bmap[:, -1] = seg[:, -1] ^ south[:, -1]
    bmap[-1, -1] = False

    if height == 1:
        bmap[0, :] = seg[0, :]
    if width == 1:
        bmap[:, 0] = seg[:, 0]
    return bmap


def _disk_offsets(radius: float) -> List[Tuple[int, int]]:
    radius = float(max(radius, 0.0))
    ceil_radius = int(np.ceil(radius))
    offsets: List[Tuple[int, int]] = []
    radius_sq = radius * radius
    for dy in range(-ceil_radius, ceil_radius + 1):
        for dx in range(-ceil_radius, ceil_radius + 1):
            if dy * dy + dx * dx <= radius_sq:
                offsets.append((dy, dx))
    if not offsets:
        offsets.append((0, 0))
    return offsets


def _binary_dilate(mask: np.ndarray, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    for dy, dx in offsets:
        if dy >= 0:
            src_y = slice(0, height - dy)
            dst_y = slice(dy, height)
        else:
            src_y = slice(-dy, height)
            dst_y = slice(0, height + dy)

        if dx >= 0:
            src_x = slice(0, width - dx)
            dst_x = slice(dx, width)
        else:
            src_x = slice(-dx, width)
            dst_x = slice(0, width + dx)

        if src_y.stop <= src_y.start or src_x.stop <= src_x.start:
            continue
        out[dst_y, dst_x] |= mask[src_y, src_x]
    return out


def db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> float:
    annotation = np.asarray(annotation).astype(bool)
    segmentation = np.asarray(segmentation).astype(bool)
    if annotation.shape != segmentation.shape:
        raise ValueError(f"Shape mismatch for J: gt={annotation.shape} pred={segmentation.shape}")

    union = np.logical_or(annotation, segmentation).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    intersection = np.logical_and(annotation, segmentation).sum(dtype=np.float64)
    return float(intersection / union)


def db_eval_boundary(annotation: np.ndarray, segmentation: np.ndarray, *, bound_th: float = 0.008) -> float:
    annotation = np.asarray(annotation).astype(bool)
    segmentation = np.asarray(segmentation).astype(bool)
    if annotation.shape != segmentation.shape:
        raise ValueError(f"Shape mismatch for F: gt={annotation.shape} pred={segmentation.shape}")

    if not annotation.any() and not segmentation.any():
        return 1.0

    gt_boundary = _seg2bmap(annotation)
    pred_boundary = _seg2bmap(segmentation)
    num_gt = float(gt_boundary.sum(dtype=np.float64))
    num_pred = float(pred_boundary.sum(dtype=np.float64))

    if bound_th >= 1:
        bound_pix = int(bound_th)
    else:
        bound_pix = int(np.ceil(bound_th * np.linalg.norm(annotation.shape)))
    bound_pix = max(bound_pix, 1)
    offsets = _disk_offsets(bound_pix)

    gt_dil = _binary_dilate(gt_boundary, offsets)
    pred_dil = _binary_dilate(pred_boundary, offsets)

    gt_match = gt_boundary & pred_dil
    pred_match = pred_boundary & gt_dil

    if num_pred == 0 and num_gt > 0:
        precision = 1.0
        recall = 0.0
    elif num_pred > 0 and num_gt == 0:
        precision = 0.0
        recall = 1.0
    elif num_pred == 0 and num_gt == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(pred_match.sum(dtype=np.float64) / max(num_pred, 1.0))
        recall = float(gt_match.sum(dtype=np.float64) / max(num_gt, 1.0))

    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def db_statistics(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D metric vector, got {arr.shape}")
    if arr.size == 0:
        return {"M": float("nan"), "R": float("nan"), "D": float("nan")}

    mean_value = float(np.nanmean(arr))
    recall = float(np.nanmean(arr > 0.5))

    if arr.size == 1:
        decay = 0.0
    else:
        bins = np.round(np.linspace(0, arr.size, 5)).astype(int)
        first = arr[bins[0] : bins[1]]
        last = arr[bins[3] : bins[4]]
        decay = float(np.nanmean(first) - np.nanmean(last))

    return {"M": mean_value, "R": recall, "D": decay}


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def evaluate_sequence(
    *,
    sequence_root: Path,
    mask_root: Path,
    model: str,
    group: str,
    benchmark: str,
    variant: str,
    image_paths: Sequence[Path],
    mask_refs: Sequence[object],
    boundary_th: float,
) -> dict[str, object]:
    variant_root = sequence_root / variant
    if not variant_root.exists():
        if benchmark in {"davis2016", "davis2017"}:
            fallback_root = mask_root / model / group / "davis_all" / sequence_root.name / variant
            if fallback_root.exists():
                variant_root = fallback_root
        if not variant_root.exists():
            raise FileNotFoundError(f"Variant directory not found: {variant_root}")

    frame_records: List[dict[str, object]] = []
    j_values: List[float] = []
    f_values: List[float] = []

    for frame_idx, (image_path, mask_ref) in enumerate(zip(image_paths, mask_refs)):
        if mask_ref is None:
            continue
        pred_path = None
        try:
            pred_path = _prediction_path_for_frame(variant_root=variant_root, variant=variant, frame_idx=frame_idx)
        except FileNotFoundError:
            pass
        pred_mask_for_alignment = _load_binary_prediction(pred_path) if pred_path is not None else None
        # SAM2-refined outputs are written at the original benchmark resolution,
        # so they should be evaluated against the benchmark GT directly rather
        # than against any cropped/aligned GT exported by the bilinear stage.
        gt_mask = load_binary_mask(mask_ref, fallback_size=_image_size(image_path))
        gt_mask = _align_gt_to_prediction(
            gt_mask,
            variant_root,
            pred_path,
            pred_mask_for_alignment,
        )
        if pred_mask_for_alignment is not None:
            pred_mask = pred_mask_for_alignment
        else:
            pred_mask = _load_prediction_for_frame(variant_root, variant, frame_idx, gt_mask)
        if pred_path is None:
            pred_path_text = f"{variant_root}/frame_{frame_idx:05d}.[npy|png] (missing -> treated as empty mask)"
        else:
            pred_path_text = str(pred_path)

        if gt_mask.shape != pred_mask.shape:
            raise ValueError(
                f"Prediction shape mismatch for {pred_path_text}: "
                f"gt={gt_mask.shape} pred={pred_mask.shape}"
            )

        j_score = db_eval_iou(gt_mask, pred_mask)
        f_score = db_eval_boundary(gt_mask, pred_mask, bound_th=boundary_th)
        j_values.append(j_score)
        f_values.append(f_score)
        frame_records.append(
            {
                "frame_idx": int(frame_idx),
                "image_path": str(image_path),
                "prediction_path": pred_path_text,
                "J": float(j_score),
                "F": float(f_score),
                "J_and_F": float(0.5 * (j_score + f_score)),
            }
        )

    j_stats = db_statistics(j_values)
    f_stats = db_statistics(f_values)
    return {
        "variant_root": str(variant_root),
        "num_frames": len(frame_records),
        "metrics": {
            "J": j_stats,
            "F": f_stats,
            "J_and_F": {
                "M": float(0.5 * (j_stats["M"] + f_stats["M"])),
                "R": float(0.5 * (j_stats["R"] + f_stats["R"])),
                "D": float(0.5 * (j_stats["D"] + f_stats["D"])),
            },
        },
        "frames": frame_records,
    }


def main() -> None:
    args = parse_args()
    mask_root = Path(args.mask_root)
    model = _normalize_model_name(args.model)
    benchmark = benchmark_output_dirname(args.benchmark)

    sequence_specs = list_sequence_specs(
        args.benchmark,
        davis_root=args.davis_root,
        segtrack_root=args.segtrack_root,
        fbms_root=args.fbms_root,
    )

    sequence_results: List[dict[str, object]] = []
    global_j_mean: List[float] = []
    global_j_recall: List[float] = []
    global_j_decay: List[float] = []
    global_f_mean: List[float] = []
    global_f_recall: List[float] = []
    global_f_decay: List[float] = []

    for spec in sequence_specs:
        image_paths, mask_refs, sequence_tag = load_sequence_paths_from_spec(
            spec,
            davis_root=args.davis_root,
            segtrack_root=args.segtrack_root,
            fbms_root=args.fbms_root,
            fbms_include_unannotated=False,
        )
        sequence_root = mask_root / model / args.group / benchmark / sequence_tag
        result = evaluate_sequence(
            sequence_root=sequence_root,
            mask_root=mask_root,
            model=model,
            group=args.group,
            benchmark=args.benchmark,
            variant=args.variant,
            image_paths=image_paths,
            mask_refs=mask_refs,
            boundary_th=float(args.boundary_th),
        )
        payload = {
            "sequence": spec.sequence,
            "sequence_tag": sequence_tag,
            "dataset": spec.dataset,
            "benchmark": spec.benchmark,
            "fbms_split": spec.fbms_split,
            **result,
        }
        sequence_results.append(payload)

        global_j_mean.append(payload["metrics"]["J"]["M"])
        global_j_recall.append(payload["metrics"]["J"]["R"])
        global_j_decay.append(payload["metrics"]["J"]["D"])
        global_f_mean.append(payload["metrics"]["F"]["M"])
        global_f_recall.append(payload["metrics"]["F"]["R"])
        global_f_decay.append(payload["metrics"]["F"]["D"])

    summary = {
        "model": model,
        "group": args.group,
        "benchmark": benchmark,
        "benchmark_display_name": benchmark_display_name(args.benchmark),
        "variant": args.variant,
        "mask_root": str(mask_root),
        "num_sequences": len(sequence_results),
        "num_frames": int(sum(int(item["num_frames"]) for item in sequence_results)),
        "official_metric_definition": {
            "J": "DAVIS region similarity IoU",
            "F": "DAVIS contour accuracy boundary F-measure",
            "J_and_F": "0.5 * (J_mean + F_mean)",
            "note": "Ground truth is evaluated as binary foreground union for the current motion-mask protocol.",
        },
        "metrics": {
            "J": {
                "M": _nanmean(global_j_mean),
                "R": _nanmean(global_j_recall),
                "D": _nanmean(global_j_decay),
            },
            "F": {
                "M": _nanmean(global_f_mean),
                "R": _nanmean(global_f_recall),
                "D": _nanmean(global_f_decay),
            },
        },
        "sequences": sequence_results,
    }
    summary["metrics"]["J_and_F"] = {
        "M": float(0.5 * (summary["metrics"]["J"]["M"] + summary["metrics"]["F"]["M"])),
        "R": float(0.5 * (summary["metrics"]["J"]["R"] + summary["metrics"]["F"]["R"])),
        "D": float(0.5 * (summary["metrics"]["J"]["D"] + summary["metrics"]["F"]["D"])),
    }

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "model": model,
                "group": args.group,
                "benchmark": benchmark,
                "variant": args.variant,
                "J": summary["metrics"]["J"],
                "F": summary["metrics"]["F"],
                "J_and_F": summary["metrics"]["J_and_F"],
                "num_sequences": summary["num_sequences"],
                "num_frames": summary["num_frames"],
                "output_json": str(output_json) if output_json is not None else "",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
