from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .motion_mask_utils import binary_mask_from_score, colorize_instances, _save_overlay_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild binary motion masks from existing avg_score_upsampled.npy heatmaps."
    )
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument(
        "--output_mask_root",
        type=str,
        default="",
        help="Optional separate output root. Defaults to mask_root for backward-compatible in-place rewriting.",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--variant", type=str, default="bilinear_upsample")
    parser.add_argument(
        "--output_variant",
        type=str,
        default="",
        help="Destination variant name. Defaults to variant.",
    )
    parser.add_argument("--sequence_tags_csv", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_mode", type=str, default="otsu", choices=["fixed", "otsu"])
    parser.add_argument("--threshold_offset", type=float, default=0.2)
    parser.add_argument("--save_numpy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_missing_heatmap", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _iter_sequence_dirs(
    *,
    mask_root: Path,
    model: str,
    group: str,
    benchmark: str,
    sequence_tags_csv: str,
) -> Iterable[Path]:
    benchmark_root = mask_root / model / group / benchmark
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_root}")

    requested = [item.strip() for item in sequence_tags_csv.split(",") if item.strip()]
    if requested:
        for tag in requested:
            seq_dir = benchmark_root / tag
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
            yield seq_dir
        return

    for seq_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
        yield seq_dir


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _maybe_load_image(frame_dir: Path) -> np.ndarray | None:
    image_path = frame_dir / "input_image.png"
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return np.array(image)


def _rewrite_frame(
    source_frame_dir: Path,
    output_frame_dir: Path,
    *,
    threshold: float,
    threshold_mode: str,
    threshold_offset: float,
    save_numpy: bool,
    save_visualizations: bool,
    skip_missing_heatmap: bool,
) -> bool:
    score_path = source_frame_dir / "avg_score_upsampled.npy"
    meta_path = source_frame_dir / "meta.json"
    if not score_path.exists():
        if skip_missing_heatmap:
            return False
        raise FileNotFoundError(f"Missing heatmap: {score_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    score_map = np.load(score_path).astype(np.float32)
    binary_mask, threshold_value = binary_mask_from_score(
        score_map,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_offset=threshold_offset,
    )
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    if save_numpy:
        np.save(output_frame_dir / "binary_mask_upsampled.npy", binary_mask.astype(np.uint8))

    binary_rgb = np.repeat((binary_mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(binary_rgb).save(output_frame_dir / "binary_mask_upsampled.png")

    if save_visualizations:
        image_np = _maybe_load_image(source_frame_dir)
        _save_overlay_image(output_frame_dir, "binary_mask_upsampled_overlay", image_np, binary_mask)

    instance_path = source_frame_dir / "instance_mask_upsampled.png"
    meta = _load_json(meta_path)
    if save_visualizations and instance_path.exists():
        instance_rgb, num_instances = colorize_instances(binary_mask)
        Image.fromarray(instance_rgb).save(output_frame_dir / instance_path.name)
        meta["num_instances"] = int(num_instances)

    meta["threshold"] = float(threshold)
    meta["threshold_mode"] = str(threshold_mode)
    meta["threshold_offset"] = float(threshold_offset)
    meta["threshold_value"] = float(threshold_value)
    with (output_frame_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return True


def main() -> None:
    args = parse_args()
    mask_root = Path(args.mask_root)
    output_mask_root = Path(args.output_mask_root) if args.output_mask_root else mask_root
    output_variant = args.output_variant or args.variant
    rewritten = 0
    skipped = 0

    for seq_dir in _iter_sequence_dirs(
        mask_root=mask_root,
        model=args.model,
        group=args.group,
        benchmark=args.benchmark,
        sequence_tags_csv=args.sequence_tags_csv,
    ):
        source_variant_dir = seq_dir / args.variant
        if not source_variant_dir.exists():
            raise FileNotFoundError(f"Variant directory not found: {source_variant_dir}")
        output_variant_dir = (
            output_mask_root / args.model / args.group / args.benchmark / seq_dir.name / output_variant
        )
        for source_frame_dir in sorted(
            path for path in source_variant_dir.iterdir() if path.is_dir() and path.name.startswith("frame_")
        ):
            output_frame_dir = output_variant_dir / source_frame_dir.name
            if args.skip_existing and (
                (output_frame_dir / "binary_mask_upsampled.npy").exists()
                or (output_frame_dir / "binary_mask_upsampled.png").exists()
            ):
                skipped += 1
                continue
            if _rewrite_frame(
                source_frame_dir,
                output_frame_dir,
                threshold=args.threshold,
                threshold_mode=args.threshold_mode,
                threshold_offset=args.threshold_offset,
                save_numpy=bool(args.save_numpy),
                save_visualizations=bool(args.save_visualizations),
                skip_missing_heatmap=bool(args.skip_missing_heatmap),
            ):
                rewritten += 1
            else:
                skipped += 1

    print(
        json.dumps(
            {
                "mask_root": str(mask_root),
                "output_mask_root": str(output_mask_root),
                "model": args.model,
                "group": args.group,
                "benchmark": args.benchmark,
                "variant": args.variant,
                "output_variant": output_variant,
                "threshold": float(args.threshold),
                "threshold_mode": str(args.threshold_mode),
                "threshold_offset": float(args.threshold_offset),
                "save_numpy": bool(args.save_numpy),
                "save_visualizations": bool(args.save_visualizations),
                "rewritten_frames": int(rewritten),
                "skipped_frames": int(skipped),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
