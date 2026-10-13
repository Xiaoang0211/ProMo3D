from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image

from isdynamic.legacy.pi3.build_davis_isdynamic_npz import PI3_TARGET_SIZE, compute_resize_dims
from isdynamic.legacy.pi3.visualize_isdynamic import (
    PI3_PATCH_SIZE,
    _load_context_tensors,
    _resolve_probe_input_dim,
    discover_run_dirs,
    extract_frame_feature_map,
)
from isdynamic.models.pi3 import load_pi3_family_model
from isdynamic.viz.figure_io import save_raster_png_pdf_figure
from isdynamic.viz.motion_mask_benchmarks import load_binary_mask
from isdynamic.viz.scoremap_utils import average_block_maps, load_probe_from_run, score_map_for_query, upsample_patch_map


EXAMPLE_SPECS = (
    {
        "stem": "cheeta",
        "dataset": "segtrackv2",
        "benchmark_name": "segtrackv2",
        "sequence": "cheetah",
        "frame_idx": 0,
    },
    {
        "stem": "f50_birdofparadise",
        "dataset": "segtrackv2",
        "benchmark_name": "segtrackv2",
        "sequence": "bird_of_paradise",
        "frame_idx": 50,
    },
    {
        "stem": "parachute",
        "dataset": "segtrackv2",
        "benchmark_name": "segtrackv2",
        "sequence": "parachute",
        "frame_idx": 0,
    },
    {
        "stem": "f144_worm",
        "dataset": "segtrackv2",
        "benchmark_name": "segtrackv2",
        "sequence": "worm",
        "frame_idx": 144,
    },
    {
        "stem": "f70_koala",
        "dataset": "davis",
        "benchmark_name": None,
        "sequence": "koala",
        "frame_idx": 70,
    },
    {
        "stem": "f70_rhino",
        "dataset": "davis",
        "benchmark_name": None,
        "sequence": "rhino",
        "frame_idx": 70,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unseen-class heatmap/context examples.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/isDynamicProbe/unseen_classes",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/isDynamicProbe/models/Pi3/ckpt/model_pi3x.safetensors",
    )
    parser.add_argument(
        "--probe_root",
        type=str,
        default="/home/xiaoang/FF_Reconstruction/isDynamicProbe/outputs/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/pair_lowrank_quadratic",
    )
    parser.add_argument("--group", type=str, default="inter_frame_block")
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument("--context_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS",
    )
    parser.add_argument(
        "--segtrack_root",
        type=str,
        default="/home/xiaoang/Datasets/SegTrackv2",
    )
    parser.add_argument(
        "--fbms_root",
        type=str,
        default="/home/xiaoang/Datasets/FBMS-59",
    )
    return parser.parse_args()


def _make_args_namespace(cli_args: argparse.Namespace, spec: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=spec["dataset"],
        benchmark_name=spec["benchmark_name"],
        sequence=spec["sequence"],
        frame_idx=int(spec["frame_idx"]),
        context_size=int(cli_args.context_size),
        context_stride=int(cli_args.context_stride),
        davis_root=cli_args.davis_root,
        segtrack_root=cli_args.segtrack_root,
        fbms_root=cli_args.fbms_root,
        fbms_split="Testset",
    )


def _render_heatmap_pdf(image_np: np.ndarray, avg_map: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat = upsample_patch_map(avg_map, image_np.shape[:2])
    h, w = heat.shape[:2]
    fig_w = 4.0 * (w / max(h, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image_np)
    ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_raster_png_pdf_figure(fig, out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _map_square_heat_to_original(avg_map: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = original_hw
    resized_w, resized_h = compute_resize_dims(orig_w, orig_h)
    x0 = (PI3_TARGET_SIZE - resized_w) // 2
    y0 = (PI3_TARGET_SIZE - resized_h) // 2

    square_heat = upsample_patch_map(avg_map, (PI3_TARGET_SIZE, PI3_TARGET_SIZE))
    content_heat = square_heat[y0 : y0 + resized_h, x0 : x0 + resized_w]

    heat_uint8 = np.clip(content_heat * 255.0, 0.0, 255.0).astype(np.uint8)
    restored = Image.fromarray(heat_uint8, mode="L").resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.float32) / 255.0


def _render_heatmap_pdf_on_original(original_image_np: np.ndarray, avg_map: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat = _map_square_heat_to_original(avg_map, original_image_np.shape[:2])
    h, w = heat.shape[:2]
    fig_w = 4.0 * (w / max(h, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(original_image_np)
    ax.imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.58, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_raster_png_pdf_figure(fig, out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _render_red_context_pdf(image_np: np.ndarray, mask_np: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = np.dot(image_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    tinted_gray = np.stack(
        [
            gray * 0.79,
            gray * 0.79,
            np.clip(gray * 0.79 + 28.0, 0.0, 255.0),
        ],
        axis=-1,
    ).astype(np.uint8)

    mask_binary = mask_np > 0.5
    mask_overlay = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.float32)
    mask_overlay[..., 0] = 0.56
    mask_overlay[..., 1] = 0.24
    mask_overlay[..., 2] = 0.16
    mask_overlay[..., 3] = 0.52 * mask_binary.astype(np.float32)

    h, w = image_np.shape[:2]
    fig_w = 4.0 * (w / max(h, 1))
    fig_h = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(tinted_gray)
    ax.imshow(mask_overlay, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_raster_png_pdf_figure(fig, out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _compute_maps_for_query_zero(
    feature_map: Dict[str, torch.Tensor],
    run_dir_by_key: Dict[str, Path],
    usable_keys: Iterable[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    maps_by_block: Dict[str, np.ndarray] = {}
    for key in usable_keys:
        feats = feature_map[key]
        probe = load_probe_from_run(run_dir_by_key[key], device=device)
        c_probe = _resolve_probe_input_dim(probe)
        if feats.shape[-1] != c_probe:
            raise ValueError(f"Feature dim mismatch for {key}: feature={feats.shape[-1]} probe={c_probe}")
        probs = score_map_for_query(
            model=probe,
            token_features=feats,
            query_idx=0,
            batch_size=batch_size,
        )
        probs = 1.0 - probs
        grid_tokens = int(feats.shape[0])
        grid_h = grid_w = int(round(grid_tokens**0.5))
        if grid_h * grid_w != grid_tokens:
            raise ValueError(f"Expected square patch grid, got {grid_tokens} tokens for {key}")
        maps_by_block[key] = probs.reshape(grid_h, grid_w).detach().cpu().numpy().astype(np.float32)
    return maps_by_block


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_pi3_family_model(
        model_name="pi3x",
        checkpoint=Path(args.checkpoint),
        device=device,
        use_multimodal_pi3x=False,
    )

    run_items = discover_run_dirs(Path(args.probe_root), group=args.group)
    run_dir_by_key = {k: d for k, d in run_items}

    for spec in EXAMPLE_SPECS:
        example_args = _make_args_namespace(args, spec)
        image_tensors, target_mask_t, image_np, _mask_np, _sequence_tag, input_frame_indices, image_paths, target_mask_ref = _load_context_tensors(example_args)

        feature_map = extract_frame_feature_map(
            model_name="pi3x",
            model=model,
            image_t=torch.stack(image_tensors, dim=0),
            group=args.group,
            device=device,
        )
        usable_keys = [k for k in feature_map.keys() if k in run_dir_by_key]
        usable_keys.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))
        if not usable_keys:
            raise RuntimeError(f"No overlapping block keys for {spec['sequence']}")

        maps_by_block = _compute_maps_for_query_zero(
            feature_map=feature_map,
            run_dir_by_key=run_dir_by_key,
            usable_keys=usable_keys,
            batch_size=int(args.batch_size),
            device=device,
        )

        image_path = Path(image_paths[input_frame_indices[0]])
        original_image_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        avg_map = average_block_maps(maps_by_block)
        heatmap_out = out_dir / f"avg_score_upsampled_heatmap_{spec['stem']}_nobar.pdf"
        _render_heatmap_pdf_on_original(original_image_np=original_image_np, avg_map=avg_map, out_path=heatmap_out)

        binary_mask = load_binary_mask(
            target_mask_ref,
            fallback_size=(original_image_np.shape[1], original_image_np.shape[0]),
        ).astype(np.float32)
        context_out = out_dir / f"context_{spec['stem']}_red.pdf"
        _render_red_context_pdf(image_np=original_image_np, mask_np=binary_mask, out_path=context_out)

        print(f"Saved unseen-class example outputs: {heatmap_out} and {context_out}")


if __name__ == "__main__":
    main()
