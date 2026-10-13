from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch
from PIL import Image


class AttrDict(dict[str, object]):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the RAFT stage required by Flow-SAM.")
    parser.add_argument("--flowsam_root", type=Path, required=True)
    parser.add_argument("--raft_checkpoint", type=Path, required=True)
    parser.add_argument("--davis_root", type=Path, required=True)
    parser.add_argument("--sequence", type=str, default="parkour")
    parser.add_argument("--warmup_sequence", type=str, default="cows")
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--flow_gaps", type=str, default="1,-1,2,-2")
    parser.add_argument("--raft_iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/experiments/runtime/flowsam"))
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


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean() * 1000.0),
        "median_ms": float(np.median(array) * 1000.0),
        "p95_ms": float(np.percentile(array, 95) * 1000.0),
    }


def select_neighbor(frame_idx: int, gap: int, frame_count: int) -> int:
    candidate = frame_idx + gap
    if 0 <= candidate < frame_count:
        return candidate
    fallback = frame_idx - gap
    if 0 <= fallback < frame_count:
        return fallback
    return frame_idx


def load_rgb_cache(paths_by_sequence: dict[str, list[Path]]) -> dict[Path, np.ndarray]:
    unique_paths = sorted({path for paths in paths_by_sequence.values() for path in paths})
    return {path: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy() for path in unique_paths}


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The runtime protocol requires an available CUDA device.")
    gaps = [int(value) for value in args.flow_gaps.split(",")]
    if not gaps:
        raise ValueError("--flow_gaps must contain at least one integer.")

    flow_core = args.flowsam_root.resolve() / "flow" / "core"
    sys.path.insert(0, str(flow_core))
    from raft import RAFT  # type: ignore[import-not-found]
    from utils import flow_viz  # type: ignore[import-not-found]
    from utils.utils import InputPadder  # type: ignore[import-not-found]

    measured_paths = image_paths(resolve_sequence_dir(args.davis_root, args.sequence))
    warmup_paths = image_paths(resolve_sequence_dir(args.davis_root, args.warmup_sequence))
    if args.num_frames > len(measured_paths):
        raise ValueError(f"Requested {args.num_frames} measured frames, found {len(measured_paths)}.")
    if args.warmup_frames > len(warmup_paths):
        raise ValueError(f"Requested {args.warmup_frames} warm-up frames, found {len(warmup_paths)}.")
    rgb_cache = load_rgb_cache({args.sequence: measured_paths, args.warmup_sequence: warmup_paths})

    raft_args = AttrDict(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0)
    wrapped_model = torch.nn.DataParallel(RAFT(raft_args))
    state_dict = torch.load(args.raft_checkpoint, map_location="cpu", weights_only=True)
    wrapped_model.load_state_dict(state_dict)
    model = wrapped_model.module.to(args.device).eval()
    device = torch.device(args.device)

    output_dir = args.output_dir.resolve()
    flow_cache_root = output_dir / "flow_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_frame(
        *,
        sequence: str,
        frame_idx: int,
        paths: list[Path],
        measured: bool,
    ) -> dict[str, object]:
        target = rgb_cache[paths[frame_idx]]
        neighbors = [
            rgb_cache[paths[select_neighbor(frame_idx, gap, len(paths))]]
            for gap in gaps
        ]
        if measured:
            torch.cuda.reset_peak_memory_stats(device)

        total_t0 = perf_counter()
        target_tensor = torch.from_numpy(target.copy()).permute(2, 0, 1).float()
        neighbor_tensor = torch.stack(
            [torch.from_numpy(image.copy()).permute(2, 0, 1).float() for image in neighbors],
            dim=0,
        )
        target_batch = target_tensor.unsqueeze(0).expand(len(gaps), -1, -1, -1).contiguous()
        padder = InputPadder(target_batch.shape)
        target_batch, neighbor_tensor = padder.pad(target_batch, neighbor_tensor)
        target_batch = target_batch.to(device)
        neighbor_tensor = neighbor_tensor.to(device)
        torch.cuda.synchronize(device)
        preprocess_transfer_seconds = perf_counter() - total_t0

        forward_t0 = perf_counter()
        with torch.inference_mode():
            _, flow_up = model(
                target_batch,
                neighbor_tensor,
                iters=args.raft_iters,
                test_mode=True,
            )
        flow_up = padder.unpad(flow_up)
        torch.cuda.synchronize(device)
        forward_seconds = perf_counter() - forward_t0

        conversion_t0 = perf_counter()
        flow_numpy = flow_up.permute(0, 2, 3, 1).float().cpu().numpy()
        flow_images = [flow_viz.flow_to_image(flow) for flow in flow_numpy]
        conversion_seconds = perf_counter() - conversion_t0
        total_seconds = perf_counter() - total_t0
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if measured else 0

        frame_dir = flow_cache_root / sequence / f"frame_{frame_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for gap, flow_image in zip(gaps, flow_images):
            tag = f"pos{gap}" if gap > 0 else f"neg{abs(gap)}"
            Image.fromarray(flow_image).save(frame_dir / f"gap_{tag}.png")

        del target_batch, neighbor_tensor, flow_up
        return {
            "sequence": sequence,
            "frame_idx": frame_idx,
            "preprocess_transfer_seconds": preprocess_transfer_seconds,
            "raft_forward_seconds": forward_seconds,
            "flow_color_conversion_seconds": conversion_seconds,
            "raft_stage_total_seconds": total_seconds,
            "peak_vram_gib": peak_bytes / (1024.0**3),
        }

    print(f"RAFT warm-up: {args.warmup_frames} frames from DAVIS/{args.warmup_sequence}")
    for frame_idx in range(args.warmup_frames):
        run_frame(
            sequence=args.warmup_sequence,
            frame_idx=frame_idx,
            paths=warmup_paths,
            measured=False,
        )

    rows = []
    print(f"RAFT measured run: {args.num_frames} frames from DAVIS/{args.sequence}")
    for frame_idx in range(args.num_frames):
        rows.append(
            run_frame(
                sequence=args.sequence,
                frame_idx=frame_idx,
                paths=measured_paths,
                measured=True,
            )
        )
        print(f"[{frame_idx + 1:03d}/{args.num_frames:03d}] {args.sequence}/frame_{frame_idx:05d}")

    with (output_dir / "raft_per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(device),
            "precision": "float32",
            "batch_size_target_frames": 1,
            "flow_pair_batch_size": len(gaps),
            "flow_gaps": gaps,
            "raft_iterations": args.raft_iters,
            "warmup_sequence": args.warmup_sequence,
            "warmup_frames": args.warmup_frames,
            "sequence": args.sequence,
            "measured_frames": args.num_frames,
            "model_loading_included": False,
            "source_image_disk_io_included": False,
            "flow_cache_writes_included": False,
            "cpu_preprocessing_included": True,
            "host_to_device_transfer_included": True,
            "device_to_host_transfer_included": True,
            "flow_color_conversion_included": True,
        },
        "raft": {
            **percentile_summary([float(row["raft_stage_total_seconds"]) for row in rows]),
            "forward_only": percentile_summary([float(row["raft_forward_seconds"]) for row in rows]),
            "peak_vram_gib": max(float(row["peak_vram_gib"]) for row in rows),
        },
    }
    with (output_dir / "raft_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
