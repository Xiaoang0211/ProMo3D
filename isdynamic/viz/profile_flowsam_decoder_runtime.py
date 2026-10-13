from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from PIL import Image


class SerialBatchEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(images[index : index + 1]) for index in range(len(images))], dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a released Flow-SAM decoder on cached RAFT flows.")
    parser.add_argument("--model", choices=["flowpsam", "flowisam"], default="flowpsam")
    parser.add_argument("--flowsam_root", type=Path, required=True)
    parser.add_argument("--segment_anything_root", type=Path, required=True)
    parser.add_argument("--flowsam_checkpoint", type=Path, required=True)
    parser.add_argument("--sam_vit_h_checkpoint", type=Path, required=True)
    parser.add_argument("--sam_vit_b_checkpoint", type=Path, required=True)
    parser.add_argument("--davis_root", type=Path, required=True)
    parser.add_argument("--flow_cache_root", type=Path, required=True)
    parser.add_argument("--sequence", type=str, default="parkour")
    parser.add_argument("--warmup_sequence", type=str, default="cows")
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--flow_gaps", type=str, default="1,-1,2,-2")
    parser.add_argument("--num_gridside", type=int)
    parser.add_argument("--prompt_batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/experiments/runtime/flowsam"))
    parser.add_argument("--save_masks", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_sequence_dir(davis_root: Path, sequence: str) -> Path:
    candidates = [
        davis_root / "DAVIS" / "JPEGImages" / "480p" / sequence,
        davis_root / "JPEGImages" / "480p" / sequence,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find DAVIS sequence '{sequence}'. Checked:\n{checked}")


def image_paths(sequence_dir: Path) -> list[Path]:
    paths = sorted([*sequence_dir.glob("*.jpg"), *sequence_dir.glob("*.png")])
    if not paths:
        raise FileNotFoundError(f"No frames found under {sequence_dir}")
    return paths


def flow_cache_path(flow_cache_root: Path, sequence: str, frame_idx: int, gap: int) -> Path:
    tag = f"pos{gap}" if gap > 0 else f"neg{abs(gap)}"
    return flow_cache_root / sequence / f"frame_{frame_idx:05d}" / f"gap_{tag}.png"


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean() * 1000.0),
        "median_ms": float(np.median(array) * 1000.0),
        "p95_ms": float(np.percentile(array, 95) * 1000.0),
    }


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The runtime protocol requires an available CUDA device.")
    if args.prompt_batch_size != 10:
        raise ValueError("The released Flow-SAM evaluator batches prompts in fixed groups of 10.")
    gaps = [int(value) for value in args.flow_gaps.split(",")]

    sys.path.insert(0, str(args.segment_anything_root.resolve()))
    sys.path.insert(0, str(args.flowsam_root.resolve()))
    from evaluation import run_flowisam, run_flowpsam  # type: ignore[import-not-found]
    from data.dataset_utils import preprocess  # type: ignore[import-not-found]
    from model.model_config import config_model  # type: ignore[import-not-found]
    from segment_anything.utils.amg import (  # type: ignore[import-not-found]
        batched_mask_to_box,
        build_all_layer_point_grids,
    )
    from segment_anything.utils.transforms import ResizeLongestSide  # type: ignore[import-not-found]
    from torchvision.ops.boxes import batched_nms
    from utils import filter_data, hard_thres, is_bg_mask  # type: ignore[import-not-found]

    measured_paths = image_paths(resolve_sequence_dir(args.davis_root, args.sequence))
    warmup_paths = image_paths(resolve_sequence_dir(args.davis_root, args.warmup_sequence))
    if args.num_frames > len(measured_paths):
        raise ValueError(f"Requested {args.num_frames} measured frames, found {len(measured_paths)}.")
    if args.warmup_frames > len(warmup_paths):
        raise ValueError(f"Requested {args.warmup_frames} warm-up frames, found {len(warmup_paths)}.")

    rgb_cache: dict[tuple[str, int], np.ndarray] = {}
    flow_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for sequence, paths, count in [
        (args.warmup_sequence, warmup_paths, args.warmup_frames),
        (args.sequence, measured_paths, args.num_frames),
    ]:
        for frame_idx, path in enumerate(paths[:count]):
            rgb_cache[(sequence, frame_idx)] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
            for gap in gaps:
                cache_path = flow_cache_path(args.flow_cache_root, sequence, frame_idx, gap)
                if not cache_path.is_file():
                    raise FileNotFoundError(f"Missing RAFT flow cache: {cache_path}")
                flow_cache[(sequence, frame_idx, gap)] = np.asarray(
                    Image.open(cache_path).convert("RGB"),
                    dtype=np.uint8,
                ).copy()

    is_flowp = args.model == "flowpsam"
    num_gridside = args.num_gridside or (20 if is_flowp else 10)
    model_label = "FlowP-SAM" if is_flowp else "FlowI-SAM"
    model_args = SimpleNamespace(
        model=args.model,
        rgb_encoder="vit_h",
        flow_encoder="vit_b" if is_flowp else "vit_h",
        rgb_encoder_ckpt_path=str(args.sam_vit_h_checkpoint),
        flow_encoder_ckpt_path=str(
            args.sam_vit_b_checkpoint if is_flowp else args.sam_vit_h_checkpoint
        ),
        ckpt_path=str(args.flowsam_checkpoint),
        sam_channel=0,
        mod_thres=-0.0,
        max_obj=10 if is_flowp else 5,
    )
    model = config_model(model_args).eval()
    if not is_flowp:
        model.flow_encoder = SerialBatchEncoder(model.flow_encoder)
    for parameter in model.parameters():
        parameter.requires_grad = False
    device = torch.device(args.device)
    transform = ResizeLongestSide(1024)
    point_grid = build_all_layer_point_grids(
        n_per_side=num_gridside,
        n_layers=0,
        scale_per_layer=1,
    )[0]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_info(sequence: str, frame_idx: int) -> dict[str, object]:
        original_size = flow_cache[(sequence, frame_idx, gaps[0])].shape[:2]
        flow_tensors = []
        for gap in gaps:
            flow_image = transform.apply_image(flow_cache[(sequence, frame_idx, gap)])
            flow_tensors.append(
                preprocess(torch.as_tensor(flow_image).permute(2, 0, 1).contiguous()).float()
            )
        grid_points = np.asarray(point_grid) * np.asarray(original_size)[None, ::-1]
        grid_coords = transform.apply_coords(grid_points, original_size)
        info = {
            "flow_image": torch.stack(flow_tensors, dim=0).unsqueeze(0),
            "grid": torch.from_numpy(grid_coords).unsqueeze(0).unsqueeze(2).float(),
            "size": [torch.tensor([original_size[0]]), torch.tensor([original_size[1]])],
        }
        if is_flowp:
            rgb_resized = transform.apply_image(rgb_cache[(sequence, frame_idx)])
            info["rgb_image"] = preprocess(
                torch.as_tensor(rgb_resized).permute(2, 0, 1).contiguous()
            ).float().unsqueeze(0)
        return info

    def run_frame(sequence: str, frame_idx: int, measured: bool) -> dict[str, object]:
        if measured:
            torch.cuda.reset_peak_memory_stats(device)
        total_t0 = perf_counter()
        info = prepare_info(sequence, frame_idx)
        preprocess_seconds = perf_counter() - total_t0

        inference_t0 = perf_counter()
        if is_flowp:
            masks_set, scores_set, boxes_set = run_flowpsam(model_args, model, info)
        else:
            masks_set, scores_set, boxes_set = run_flowisam(model_args, model, info)
        torch.cuda.synchronize(device)
        inference_seconds = perf_counter() - inference_t0

        postprocess_t0 = perf_counter()
        keep_idx = batched_nms(
            boxes_set,
            scores_set,
            torch.zeros_like(boxes_set[:, 0]),
            iou_threshold=0.9,
        )
        masks_fil, scores_fil, boxes_fil = filter_data(
            [masks_set, scores_set, boxes_set],
            keep_idx,
            is_idx=True,
        )
        keep_maskidx = ~is_bg_mask(masks_fil)
        masks_fil, scores_fil, _ = filter_data(
            [masks_fil, scores_fil, boxes_fil],
            keep_maskidx,
        )
        selected = torch.argsort(scores_fil, descending=True)[: model_args.max_obj]
        _, saved_mask = hard_thres(
            masks_fil[selected],
            scores_fil[selected],
            output_savemask=True,
        )
        torch.cuda.synchronize(device)
        postprocess_seconds = perf_counter() - postprocess_t0
        total_seconds = perf_counter() - total_t0
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if measured else 0

        if args.save_masks:
            mask_dir = output_dir / "masks" / sequence
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((saved_mask > 0).astype(np.uint8) * 255).save(
                mask_dir / f"{frame_idx:05d}.png"
            )
        return {
            "sequence": sequence,
            "frame_idx": frame_idx,
            "preprocess_seconds": preprocess_seconds,
            f"{args.model}_inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
            f"{args.model}_stage_total_seconds": total_seconds,
            "peak_vram_gib": peak_bytes / (1024.0**3),
        }

    print(f"{model_label} warm-up: {args.warmup_frames} frames from DAVIS/{args.warmup_sequence}")
    for frame_idx in range(args.warmup_frames):
        run_frame(args.warmup_sequence, frame_idx, measured=False)

    rows = []
    print(f"{model_label} measured run: {args.num_frames} frames from DAVIS/{args.sequence}")
    for frame_idx in range(args.num_frames):
        rows.append(run_frame(args.sequence, frame_idx, measured=True))
        print(f"[{frame_idx + 1:03d}/{args.num_frames:03d}] {args.sequence}/frame_{frame_idx:05d}")

    with (output_dir / f"{args.model}_per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(device),
            "precision": "float32",
            "batch_size_target_frames": 1,
            "model": model_label,
            "rgb_encoder": "SAM ViT-H" if is_flowp else None,
            "flow_encoder": "SAM ViT-B" if is_flowp else "SAM ViT-H",
            "flow_encoder_batching": "batched" if is_flowp else "serial over flow gaps",
            "flow_gaps": gaps,
            "grid_size": [num_gridside, num_gridside],
            "prompt_batch_size": args.prompt_batch_size,
            "warmup_sequence": args.warmup_sequence,
            "warmup_frames": args.warmup_frames,
            "sequence": args.sequence,
            "measured_frames": args.num_frames,
            "model_loading_included": False,
            "source_image_and_flow_disk_io_included": False,
            "mask_writes_included": False,
            "cpu_preprocessing_included": True,
            "host_to_device_transfer_included": True,
            "nms_and_mask_overlay_included": True,
        },
        args.model: {
            **percentile_summary(
                [float(row[f"{args.model}_stage_total_seconds"]) for row in rows]
            ),
            "inference_only": percentile_summary(
                [float(row[f"{args.model}_inference_seconds"]) for row in rows]
            ),
            "peak_vram_gib": max(float(row["peak_vram_gib"]) for row in rows),
        },
    }
    with (output_dir / f"{args.model}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
