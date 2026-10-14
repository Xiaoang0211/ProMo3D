from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from .motion_mask_utils import save_refined_motion_mask_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine saved averaged motion heatmaps with Upsample-Anything. "
            "The guidance image is the processed input image saved in each frame directory when available."
        )
    )
    parser.add_argument("--mask_root", type=str, required=True, help="Directory containing frame_* subdirectories exported by visualize_motion_mask.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variant_name", type=str, default="upsample_anything")
    parser.add_argument("--upa_repo_root", type=str, default=str(PROJECT_ROOT / "Upsample-Anything_Pytorch"))
    parser.add_argument(
        "--heatmap_source",
        type=str,
        default="patch",
        choices=["patch", "upsampled"],
        help="Use avg_score_patch.npy or avg_score_upsampled.npy as the low-resolution input to Upsample-Anything.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "otsu"])
    parser.add_argument(
        "--threshold_offset",
        type=float,
        default=0.0,
        help="Additive offset applied to the resolved threshold. Useful for making Otsu slightly stricter.",
    )
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=0,
        help="Remove connected components smaller than this area from the final binary mask. Set 0 to disable.",
    )
    parser.add_argument(
        "--max_hole_area",
        type=int,
        default=0,
        help="Fill interior holes smaller than this area in the final binary mask. Set 0 to disable.",
    )
    parser.add_argument(
        "--instance_mean_score_threshold",
        type=float,
        default=0.0,
        help="Discard a connected component in the final binary mask if its mean refined dynamic score is below this threshold.",
    )
    parser.add_argument(
        "--color_instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also export an instance-colored mask based on connected components.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional fallback directory of RGB frames. Used only when a frame directory does not contain input_image.png.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="",
        help="Optional image extension filter for --image_root, for example jpg or png.",
    )
    return parser.parse_args()


def _frame_sort_key(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _import_upa(repo_root: Path):
    try:
        from upsample_anything import UPA

        return UPA
    except ImportError:
        src_root = (repo_root / "src").resolve()
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from upsample_anything import UPA

        return UPA


def _resolve_heatmap_path(frame_dir: Path, source: str) -> Path:
    if source == "patch":
        return frame_dir / "avg_score_patch.npy"
    return frame_dir / "avg_score_upsampled.npy"


def _resize_pil_if_needed(image: Image.Image, target_hw: tuple[int, int] | None) -> Image.Image:
    if target_hw is None:
        return image
    target_h, target_w = target_hw
    if image.size == (target_w, target_h):
        return image
    return image.resize((target_w, target_h), Image.BICUBIC)


def _find_fallback_image(image_root: Path, frame_idx: int, meta: Dict[str, object], image_ext: str) -> Path | None:
    image_path_text = meta.get("image_path")
    if isinstance(image_path_text, str):
        candidate = image_root / Path(image_path_text).name
        if candidate.exists():
            return candidate

    suffixes = [image_ext.lstrip(".")] if image_ext else ["jpg", "jpeg", "png", "bmp", "webp"]
    for suffix in suffixes:
        candidate = image_root / f"{frame_idx:05d}.{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_guidance_image(
    frame_dir: Path,
    frame_idx: int,
    target_hw: tuple[int, int] | None,
    image_root: Path | None,
    image_ext: str,
) -> Image.Image:
    direct_path = frame_dir / "input_image.png"
    if direct_path.exists():
        image = Image.open(direct_path).convert("RGB")
        return _resize_pil_if_needed(image, target_hw)

    meta = _load_json(frame_dir / "meta.json")
    image_path_text = meta.get("image_path")
    if isinstance(image_path_text, str):
        image_path = Path(image_path_text)
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            return _resize_pil_if_needed(image, target_hw)

    if image_root is not None:
        fallback = _find_fallback_image(image_root, frame_idx=frame_idx, meta=meta, image_ext=image_ext)
        if fallback is not None:
            image = Image.open(fallback).convert("RGB")
            return _resize_pil_if_needed(image, target_hw)

    raise FileNotFoundError(
        f"Could not resolve guidance image for {frame_dir}. "
        "Expected input_image.png, a valid meta.json image_path, or --image_root."
    )


def _load_source_heatmap(frame_dir: Path, source: str) -> np.ndarray:
    path = _resolve_heatmap_path(frame_dir, source)
    if not path.exists():
        raise FileNotFoundError(f"Missing heatmap source for {frame_dir}: {path.name}")
    heatmap = np.load(path).astype(np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap in {path}, got {heatmap.shape}")
    return heatmap


def _run_upa(UPA, guidance_image: Image.Image, source_heatmap: np.ndarray) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("Upsample-Anything currently requires CUDA in this local repository implementation.")

    lr_feature = torch.from_numpy(source_heatmap).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device="cuda")
    refined = UPA(guidance_image, lr_feature)
    refined_score = refined.detach().float().cpu().numpy().squeeze()
    if refined_score.ndim != 2:
        raise ValueError(f"Expected Upsample-Anything output to be 2D after squeeze, got {refined_score.shape}")
    return np.clip(refined_score.astype(np.float32), 0.0, 1.0)


def collect_frame_dirs(mask_root: Path) -> List[Path]:
    frame_dirs = [path for path in mask_root.iterdir() if path.is_dir() and path.name.startswith("frame_")]
    return sorted(frame_dirs, key=_frame_sort_key)


def main() -> None:
    args = parse_args()

    mask_root = Path(args.mask_root)
    output_dir = Path(args.output_dir)
    repo_root = Path(args.upa_repo_root)
    image_root = Path(args.image_root) if args.image_root else None
    if not mask_root.exists():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    if not repo_root.exists():
        raise FileNotFoundError(f"upa_repo_root not found: {repo_root}")
    if image_root is not None and not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    frame_dirs = collect_frame_dirs(mask_root)
    if not frame_dirs:
        raise RuntimeError(f"No frame_* subdirectories found under {mask_root}")

    UPA = _import_upa(repo_root)
    processed_frames: List[int] = []
    frame_timings_seconds: Dict[str, float] = {}
    total_refine_t0 = perf_counter()
    for frame_dir in frame_dirs:
        frame_idx = _frame_sort_key(frame_dir)
        base_meta = _load_json(frame_dir / "meta.json")
        source_heatmap = _load_source_heatmap(frame_dir, source=args.heatmap_source)
        guidance_target_hw: tuple[int, int] | None = None
        if args.heatmap_source == "upsampled":
            guidance_target_hw = (int(source_heatmap.shape[0]), int(source_heatmap.shape[1]))
        else:
            patch_size = int(base_meta.get("patch_size", 0))
            target_h = patch_size * int(source_heatmap.shape[0])
            target_w = patch_size * int(source_heatmap.shape[1])
            if target_h <= 0 or target_w <= 0:
                raise ValueError(
                    f"Could not infer target guidance image size for patch heatmap in {frame_dir}. "
                    "Expected patch_size in meta.json or a saved input_image.png."
                )
            guidance_target_hw = (target_h, target_w)

        guidance_image = _load_guidance_image(
            frame_dir=frame_dir,
            frame_idx=frame_idx,
            target_hw=guidance_target_hw,
            image_root=image_root,
            image_ext=args.image_ext,
        )

        frame_t0 = perf_counter()
        refined_score = _run_upa(UPA=UPA, guidance_image=guidance_image, source_heatmap=source_heatmap)
        frame_seconds = float(perf_counter() - frame_t0)
        frame_timings_seconds[f"frame_{frame_idx:05d}"] = frame_seconds
        frame_output_dir = output_dir / frame_dir.name
        save_refined_motion_mask_outputs(
            out_dir=frame_output_dir,
            score_map=refined_score,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            threshold_offset=args.threshold_offset,
            connected_components=bool(args.color_instances),
            score_stem="refined_score_upsample_anything",
            binary_stem="binary_mask_upsample_anything",
            instance_stem="instance_mask_upsample_anything",
            image_np=np.array(guidance_image),
            min_component_area=args.min_component_area,
            max_hole_area=args.max_hole_area,
            meta={
                **base_meta,
                "variant": str(args.variant_name),
                "source_mask_root": str(mask_root),
                "upa_repo_root": str(repo_root),
                "heatmap_source": args.heatmap_source,
                "threshold": float(args.threshold),
                "threshold_mode": str(args.threshold_mode),
                "threshold_offset": float(args.threshold_offset),
                "min_component_area": int(args.min_component_area),
                "max_hole_area": int(args.max_hole_area),
                "instance_mean_score_threshold": float(args.instance_mean_score_threshold),
                "guidance_image_size": [int(guidance_image.size[1]), int(guidance_image.size[0])],
                "timing_seconds": {
                    "additional_motion_mask_inference": frame_seconds,
                },
            },
            instance_mean_score_threshold=args.instance_mean_score_threshold,
        )
        processed_frames.append(frame_idx)
    total_refine_seconds = float(perf_counter() - total_refine_t0)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "variant": str(args.variant_name),
                "mask_root": str(mask_root),
                "output_dir": str(output_dir),
                "upa_repo_root": str(repo_root),
                "heatmap_source": args.heatmap_source,
                "threshold": float(args.threshold),
                "threshold_mode": str(args.threshold_mode),
                "threshold_offset": float(args.threshold_offset),
                "instance_mean_score_threshold": float(args.instance_mean_score_threshold),
                "processed_frames": processed_frames,
                "timing_seconds": {
                    "additional_motion_mask_inference_total": total_refine_seconds,
                    "additional_motion_mask_inference_per_frame_mean": (
                        float(total_refine_seconds / max(len(processed_frames), 1))
                    ),
                },
                "frame_timings_seconds": frame_timings_seconds,
            },
            handle,
            indent=2,
        )

    print(f"Saved Upsample-Anything refined masks under: {output_dir}")


if __name__ == "__main__":
    main()
