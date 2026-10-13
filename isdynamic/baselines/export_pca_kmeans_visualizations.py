from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from isdynamic.baselines.pca_kmeans_data import canonical_model_name
from isdynamic.baselines.pca_kmeans_io import mask_tree_root_for_variant, visualizations_root
from isdynamic.viz.evaluate_motion_masks import _load_prediction_for_frame
from isdynamic.viz.motion_mask_benchmarks import (
    benchmark_output_dirname,
    load_binary_mask,
    load_sequence_paths_from_spec,
    normalize_benchmark_name,
    resolve_davis_root,
    resolve_fbms_root,
    resolve_segtrack_root,
    SequenceSpec,
)

DEFAULT_PMD_GROUP = {
    "pi3": "inter_frame_block",
    "pi3x": "inter_frame_block",
    "vggt": "inter_frame_block",
    "vggt_omega": "inter_frame_block",
    "depth_anything_3": "inter_frame_block",
    "map_anything": "inter_frame_block",
    "dust3r": "dec_view1__head_in_view1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export side-by-side PCA+kmeans qualitative comparisons.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["davis_all", "davis2017", "davis2016", "segtrackv2", "fbms59"])
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame_indices", type=str, required=True, help="Comma-separated frame indices.")
    parser.add_argument("--anchor_variant", type=str, default="pca_kmeans")
    parser.add_argument("--val_variant", type=str, default="pca_kmeans_val_selected")
    parser.add_argument("--oracle_variant", type=str, default="pca_kmeans_oracle")
    parser.add_argument("--pca_group", type=str, default="")
    parser.add_argument("--pmd_group", type=str, default="")
    parser.add_argument("--pmd_variant", type=str, default="bilinear_upsample")
    parser.add_argument("--pca_mask_root", type=str, default="")
    parser.add_argument("--pmd_mask_root", type=str, default=str(Path(os.environ.get("ISDYNAMIC_OUTPUTS_ROOT", "outputs")) / "isdynamic_motion_masks"))
    parser.add_argument("--davis_root", type=str, default="")
    parser.add_argument("--segtrack_root", type=str, default="")
    parser.add_argument("--fbms_root", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    return parser.parse_args()


def _parse_frame_indices(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--frame_indices must not be empty")
    return [int(item) for item in values]


def _resolve_spec(dataset: str, sequence: str) -> SequenceSpec:
    benchmark = normalize_benchmark_name(dataset)
    if benchmark == "fbms59":
        if "__" not in sequence:
            raise ValueError("For fbms59, --sequence must be a sequence_tag like Trainingset__cats02")
        split_name, seq = sequence.split("__", 1)
        return SequenceSpec(dataset="fbms59", benchmark=benchmark, sequence=seq, sequence_tag=sequence, fbms_split=split_name)
    ds = "davis" if benchmark.startswith("davis") else "segtrackv2"
    return SequenceSpec(dataset=ds, benchmark=benchmark, sequence=sequence, sequence_tag=sequence)


def _load_sequence(spec: SequenceSpec, args: argparse.Namespace):
    return load_sequence_paths_from_spec(
        spec,
        davis_root=resolve_davis_root(args.davis_root or None),
        segtrack_root=resolve_segtrack_root(args.segtrack_root or None),
        fbms_root=resolve_fbms_root(args.fbms_root or None),
        fbms_include_unannotated=False,
    )


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _fit_tile(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.NEAREST if image.mode in {"1", "L"} else Image.Resampling.BILINEAR)


def _add_label(image: Image.Image, text: str, *, band: int = 28) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + band), (255, 255, 255))
    canvas.paste(image, (0, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill=(0, 0, 0))
    return canvas


def _load_pred(mask_root: Path, model: str, group: str, benchmark: str, sequence_tag: str, frame_idx: int, variant: str, gt_mask: np.ndarray) -> np.ndarray:
    del model, group
    variant_root = mask_root / benchmark_output_dirname(benchmark) / sequence_tag
    return _load_prediction_for_frame(variant_root, variant, frame_idx, gt_mask)


def main() -> None:
    args = parse_args()
    model = canonical_model_name(args.model)
    benchmark = normalize_benchmark_name(args.dataset)
    frame_indices = _parse_frame_indices(args.frame_indices)
    pca_group = args.pca_group or DEFAULT_PMD_GROUP[model]
    pmd_group = args.pmd_group or DEFAULT_PMD_GROUP[model]
    spec = _resolve_spec(benchmark, args.sequence)
    image_paths, mask_refs, sequence_tag = _load_sequence(spec, args)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = visualizations_root() / model / benchmark / sequence_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_root_base = Path(args.pca_mask_root) if args.pca_mask_root else None
    def variant_root(variant: str) -> Path:
        if pca_root_base is not None:
            return pca_root_base / variant
        return mask_tree_root_for_variant(model_name=model, variant_name=variant)

    pmd_root = Path(args.pmd_mask_root)

    for frame_idx in frame_indices:
        image_path = image_paths[frame_idx]
        gt_mask = load_binary_mask(mask_refs[frame_idx], fallback_size=Image.open(image_path).size[::-1])
        with Image.open(image_path) as im:
            input_tile = _fit_tile(im.convert("RGB"), 320)

        anchor_mask = _load_pred(variant_root(args.anchor_variant), model, pca_group, benchmark, sequence_tag, frame_idx, 'bilinear_upsample', gt_mask)
        val_mask = _load_pred(variant_root(args.val_variant), model, pca_group, benchmark, sequence_tag, frame_idx, 'bilinear_upsample', gt_mask)
        oracle_mask = _load_pred(variant_root(args.oracle_variant), model, pca_group, benchmark, sequence_tag, frame_idx, 'bilinear_upsample', gt_mask)
        pmd_mask = _load_pred(pmd_root, model, pmd_group, benchmark, sequence_tag, frame_idx, args.pmd_variant, gt_mask)

        tiles = [
            _add_label(input_tile, 'Input frame'),
            _add_label(_fit_tile(_mask_to_rgb(gt_mask), 320), 'GT mask'),
            _add_label(_fit_tile(_mask_to_rgb(anchor_mask), 320), 'PCA+k-means'),
            _add_label(_fit_tile(_mask_to_rgb(val_mask), 320), 'PCA+k-means val-selected'),
            _add_label(_fit_tile(_mask_to_rgb(oracle_mask), 320), 'PCA+k-means oracle'),
            _add_label(_fit_tile(_mask_to_rgb(pmd_mask), 320), 'PMD coarse mask'),
        ]
        panel = Image.new('RGB', (tiles[0].width * len(tiles), tiles[0].height), (255, 255, 255))
        for idx, tile in enumerate(tiles):
            panel.paste(tile, (idx * tile.width, 0))
        panel.save(out_dir / f'comparison_frame_{frame_idx:05d}.png')
        print(f'[pca_kmeans_viz] wrote {out_dir / f"comparison_frame_{frame_idx:05d}.png"}')


if __name__ == '__main__':
    main()
