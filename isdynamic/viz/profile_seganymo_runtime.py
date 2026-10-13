from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile released SegAnyMo inference.")
    parser.add_argument("--seganymo_root", type=Path, required=True)
    parser.add_argument("--motion_checkpoint", type=Path, required=True)
    parser.add_argument("--tapir_checkpoint", type=Path, required=True)
    parser.add_argument("--sam2_checkpoint", type=Path, required=True)
    parser.add_argument("--davis_root", type=Path, required=True)
    parser.add_argument("--sequence", default="parkour")
    parser.add_argument("--warmup_sequence", default="cows")
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/experiments/runtime/seganymo"),
    )
    return parser.parse_args()


def frame_paths(root: Path, sequence: str, count: int) -> list[Path]:
    image_root = root / "DAVIS" / "JPEGImages" / "480p" / sequence
    paths = sorted(image_root.glob("*.jpg"))
    if len(paths) < count:
        raise ValueError(f"Requested {count} frames from {image_root}, found {len(paths)}.")
    return paths[:count]


def load_rgb(paths: list[Path]) -> list[np.ndarray]:
    return [np.asarray(Image.open(path).convert("RGB")).copy() for path in paths]


def timed_cuda(fn, device: torch.device) -> tuple[object, float, float]:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    return value, elapsed, peak_gib


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_tapir(seganymo_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str((seganymo_root / "preproc").resolve()))
    from tapnet_torch import tapir_model  # type: ignore[import-not-found]

    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval().to(device)


def run_tapir(
    model,
    frames: list[np.ndarray],
    grid_size: int,
    step: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    import mediapy as media
    from tapnet_torch import transforms  # type: ignore[import-not-found]

    video = np.stack(frames)
    num_frames, height, width = video.shape[:3]
    resized = np.asarray(media.resize_video(video, (256, 256)))
    frame_tensor = torch.from_numpy(resized.copy()).to(device)
    frame_tensor = (frame_tensor.float() / 255.0 * 2.0 - 1.0)[None]

    y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
    y_resize = y / (height - 1) * 255
    x_resize = x / (width - 1) * 255
    outputs_by_query: dict[int, np.ndarray] = {}

    for query_t in range(0, num_frames, step):
        points = np.stack(
            [query_t * np.ones_like(y), y_resize, x_resize],
            axis=-1,
        ).reshape(-1, 3)
        chunks = max(1, len(points) // 128)
        query_outputs = []
        for point_chunk in np.array_split(points, chunks):
            point_tensor = torch.from_numpy(point_chunk.astype(np.float32))[None].to(device)
            with torch.inference_mode():
                prediction = model(frame_tensor, point_tensor)
            tracks = prediction["tracks"][0].detach().cpu().numpy()
            occlusion = prediction["occlusion"][0].detach().cpu().numpy()
            expected_dist = prediction["expected_dist"][0].detach().cpu().numpy()
            tracks = transforms.convert_grid_coordinates(
                tracks,
                (255, 255),
                (width - 1, height - 1),
            )
            query_outputs.append(
                np.concatenate(
                    [tracks, occlusion[..., None], expected_dist[..., None]],
                    axis=-1,
                )
            )
        output = np.concatenate(query_outputs, axis=0)
        output[:, query_t, :2] = np.stack([x, y], axis=-1).reshape(-1, 2)
        outputs_by_query[query_t] = output.astype(np.float32, copy=False)
    return outputs_by_query


def load_dino(seganymo_root: Path):
    sys.path.insert(0, str(seganymo_root.resolve()))
    from core.utils.dino_feat import ViTExtractor  # type: ignore[import-not-found]

    return ViTExtractor(model_type="dinov2_vitb14", stride=7)


def run_dino(extractor, frames: list[np.ndarray], step: int) -> dict[int, np.ndarray]:
    from torchvision import transforms

    height, width = frames[0].shape[:2]
    target_shape = ((height + 13) // 14 * 14, (width + 13) // 14 * 14)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(target_shape),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    outputs: dict[int, np.ndarray] = {}
    for query_t in range(0, len(frames), step):
        image = preprocess(Image.fromarray(frames[query_t])).unsqueeze(0).cuda()
        with torch.inference_mode():
            descriptor = extractor.extract_descriptors(
                image,
                [11],
                "key",
                include_cls=False,
            )
        descriptor = descriptor.reshape(
            descriptor.shape[0],
            extractor.num_patches[0],
            extractor.num_patches[1],
            -1,
        ).squeeze(0)
        outputs[query_t] = descriptor.detach().cpu().numpy().astype(np.float16)
    return outputs


def load_depth_pipeline():
    from transformers import pipeline

    return pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device="cuda",
    )


def to_uint16(values: np.ndarray) -> np.ndarray:
    value_min = values.min()
    value_max = values.max()
    if value_max - value_min <= np.finfo(float).eps:
        return np.zeros(values.shape, dtype=np.uint16)
    return (65535 * (values - value_min) / (value_max - value_min)).astype(np.uint16)


def run_depth(pipe, frames: list[np.ndarray]) -> list[np.ndarray]:
    outputs = []
    for frame in frames:
        image = Image.fromarray(frame)
        disparity = pipe(image)["predicted_depth"]
        disparity = F.interpolate(
            disparity.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        outputs.append(to_uint16(disparity.squeeze().detach().cpu().numpy()))
    return outputs


def load_motion_model(seganymo_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(seganymo_root.resolve()))
    from core.network.traj_oa_depth import traj_oa_depth  # type: ignore[import-not-found]

    model = traj_oa_depth(
        extra_info=True,
        oanet=False,
        enable_pos_embed=True,
        load_dino=True,
        target_feature_dim=None,
        dino_later=True,
        dino_woatt=True,
        time_att=True,
        tracks=True,
        depth=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    return model.eval().to(device)


def prepare_motion_input(
    tracks_by_query: dict[int, np.ndarray],
    dino_by_query: dict[int, np.ndarray],
    depth_maps: list[np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    from core.dataset.data_utils import normalize_point_traj_torch
    from core.dataset.kubric import parse_tapir_track_info
    from core.utils.utils import get_feat

    generator = torch.Generator().manual_seed(seed)
    num_frames = len(depth_maps)
    height, width = depth_maps[0].shape
    query_times = sorted(tracks_by_query)
    ratio = 1.0 / len(query_times)

    sampled_tracks = []
    sampled_visible = []
    sampled_confidence = []
    sampled_visibility_values = []
    sampled_confidence_values = []
    sampled_dino = []

    for query_t in query_times:
        packed = torch.from_numpy(tracks_by_query[query_t])
        tracks, occlusions, distances = packed[..., :2], packed[..., 2], packed[..., 3]
        visible, _, confidence, visibility_value, confidence_value = (
            parse_tapir_track_info(occlusions, distances, 0.5)
        )
        sample_count = int(tracks.shape[0] * ratio)
        indices = torch.randperm(tracks.shape[0], generator=generator)[:sample_count]
        sampled_track = tracks[indices]
        sampled_tracks.append(sampled_track)
        sampled_visible.append(visible[indices])
        sampled_confidence.append(confidence[indices])
        sampled_visibility_values.append(visibility_value[indices])
        sampled_confidence_values.append(confidence_value[indices])

        dino_map = torch.from_numpy(dino_by_query[query_t]).unsqueeze(0).unsqueeze(0)
        scale = (dino_map[0].shape[1] / height, dino_map[0].shape[2] / width)
        dino = get_feat(
            scale,
            sampled_track.permute(2, 0, 1)[..., query_t : query_t + 1],
            dino_map,
        )
        sampled_dino.append(dino)

    tracks = torch.cat(sampled_tracks, dim=0)
    visible = torch.cat(sampled_visible, dim=0)
    confidence = torch.cat(sampled_confidence, dim=0)
    visibility_value = torch.cat(sampled_visibility_values, dim=0)
    confidence_value = torch.cat(sampled_confidence_values, dim=0)
    dino = torch.cat(sampled_dino, dim=2)

    select = torch.randperm(tracks.shape[0], generator=generator)[:5000]
    tracks = tracks[select]
    visible = visible[select]
    confidence = confidence[select]
    visibility_value = visibility_value[select]
    confidence_value = confidence_value[select]
    dino = dino[:, :, select, :]

    valid_rows = ~torch.all(~visible, dim=1)
    tracks = tracks[valid_rows]
    visible = visible[valid_rows]
    confidence = confidence[valid_rows]
    visibility_value = visibility_value[valid_rows]
    confidence_value = confidence_value[valid_rows]
    dino = dino[:, :, valid_rows, :]

    for frame_idx in torch.where(torch.all(~visible, dim=0))[0]:
        visible[torch.argmax(confidence[:, frame_idx]), frame_idx] = True

    track = tracks.permute(2, 0, 1).unsqueeze(0)
    invisible = (~visible).unsqueeze(0).unsqueeze(0)
    depth_tensors = []
    for depth in depth_maps:
        depth_l = np.asarray(Image.fromarray(depth).convert("L"), dtype=np.float32)
        value_range = depth_l.max() - depth_l.min()
        depth_l = (
            (depth_l - depth_l.min()) / value_range
            if value_range > np.finfo(np.float32).eps
            else np.zeros_like(depth_l)
        )
        depth_tensors.append(torch.from_numpy(depth_l))
    depths = torch.stack(depth_tensors).permute(1, 2, 0)[None, None]

    batch = {
        "traj": normalize_point_traj_torch(track, [height, width]).float().to(device),
        "mask": invisible.float().to(device),
        "depth": depths.float().to(device),
        "visib_value": visibility_value[None, None].float().to(device),
        "confi_value": confidence_value[None, None].float().to(device),
        "dino": dino.float().to(device),
    }
    return batch, track, visible, confidence


def run_motion(
    model,
    tracks_by_query: dict[int, np.ndarray],
    dino_by_query: dict[int, np.ndarray],
    depth_maps: list[np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, track, visible, confidence = prepare_motion_input(
        tracks_by_query,
        dino_by_query,
        depth_maps,
        seed,
        device,
    )
    with torch.inference_mode():
        prediction = model(batch).detach().cpu()

    thresholds = [0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7]
    if track.shape[-1] > 300:
        thresholds = [0.99, 0.98, 0.97, 0.96, 0.95]
    for threshold in thresholds:
        dynamic = prediction > threshold
        if dynamic.sum() >= 10:
            break
    else:
        dynamic = torch.zeros_like(prediction, dtype=torch.bool)
        dynamic.view(-1)[torch.topk(prediction.view(-1), 3).indices] = True

    dynamic = dynamic.squeeze(0).squeeze(0)
    visible = visible & (confidence > 0.9)
    return (
        track.squeeze(0)[:, dynamic, :].numpy(),
        visible[dynamic, :].numpy(),
        confidence[dynamic, :].numpy(),
    )


def make_frame_subset(paths: list[Path], target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        link = target / path.name
        if not link.exists():
            link.symlink_to(path.resolve())
    return target


def load_sam2(seganymo_root: Path, checkpoint: Path):
    sys.path.insert(0, str((seganymo_root / "sam2").resolve()))
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

    return build_sam2_video_predictor("sam2_hiera_l.yaml", str(checkpoint))


def run_sam2(
    seganymo_root: Path,
    predictor,
    video_dir: Path,
    dynamic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int:
    module_path = seganymo_root / "sam2" / "run_sam2.py"
    spec = importlib.util.spec_from_file_location("seganymo_run_sam2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SegAnyMo SAM2 helpers from {module_path}")
    sam_runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam_runtime)

    sam_runtime.args = SimpleNamespace(vis=False)
    trajectory, visible, confidence = dynamic_data
    num_frames = trajectory.shape[-1]
    max_iterations = min(max(len(range(0, num_frames, 16)), 5), 10)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(video_dir))
        memory = sam_runtime.process_invisible_traj(
            trajectory,
            visible,
            confidence,
            state,
            predictor,
            dilation_size=6,
            max_iterations=max_iterations,
            timestep=num_frames,
        )
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for object_id, package in memory.items():
            query_times = list(range(0, num_frames, 16))
            predictor.reset_state(state)
            seed_time = package["time"]
            if seed_time in query_times:
                query_times.remove(seed_time)
            query_times.insert(0, seed_time)
            require_reverse = True
            for query_t in query_times:
                points = package["pts_trajs"][:, :, query_t]
                visible_points = points[package["vis_trajs"][:, query_t]]
                if visible_points.shape[0] == 0:
                    continue
                if query_t == 0:
                    require_reverse = False
                prompt = sam_runtime.find_dense_pts(visible_points)
                _, _, logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=query_t,
                    obj_id=object_id,
                    points=prompt,
                    labels=np.ones(len(prompt), dtype=np.int32),
                )
                in_mask = sam_runtime.find_pts_in_mask(
                    (logits[0] > 0).cpu().numpy(),
                    visible_points,
                )
                if in_mask.sum() < visible_points.shape[0] * 0.7:
                    nearest, _, farthest, _, _ = (
                        sam_runtime.find_centroid_and_nearest_farthest(visible_points)
                    )
                    prompt = np.concatenate((nearest, farthest), axis=0)
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=query_t,
                        obj_id=object_id,
                        points=prompt,
                        labels=np.ones(len(prompt), dtype=np.int32),
                    )
            for frame_idx, object_ids, logits in predictor.propagate_in_video(state):
                frame_segments = video_segments.setdefault(frame_idx, {})
                for idx, propagated_id in enumerate(object_ids):
                    mask = (logits[idx] > 0).cpu().numpy()
                    if propagated_id in frame_segments:
                        frame_segments[propagated_id] |= mask
                    else:
                        frame_segments[propagated_id] = mask
            if require_reverse:
                for frame_idx, object_ids, logits in predictor.propagate_in_video(
                    state,
                    reverse=True,
                ):
                    frame_segments = video_segments.setdefault(frame_idx, {})
                    for idx, propagated_id in enumerate(object_ids):
                        mask = (logits[idx] > 0).cpu().numpy()
                        if propagated_id in frame_segments:
                            frame_segments[propagated_id] |= mask
                        else:
                            frame_segments[propagated_id] = mask
        merge_groups = sam_runtime.analyze_frame_merges(video_segments, iou_threshold=0.9)
        merged_segments = sam_runtime.merge_masks(video_segments, merge_groups)
    return sum(len(frame_masks) for frame_masks in merged_segments.values())


def profile_pair(
    warm_frames: list[np.ndarray],
    warm_paths: list[Path],
    test_frames: list[np.ndarray],
    test_paths: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    stage_rows: list[dict[str, object]] = []

    def record(stage: str, elapsed: float, peak: float) -> None:
        stage_rows.append(
            {
                "sequence": args.sequence,
                "stage": stage,
                "num_frames": len(test_frames),
                "total_seconds": elapsed,
                "amortized_ms_per_frame": elapsed * 1000.0 / len(test_frames),
                "peak_vram_gib": peak,
            }
        )

    tapir = load_tapir(args.seganymo_root, args.tapir_checkpoint, device)
    warm_tracks = run_tapir(tapir, warm_frames, args.grid_size, args.step, device)
    test_tracks, elapsed, peak = timed_cuda(
        lambda: run_tapir(tapir, test_frames, args.grid_size, args.step, device),
        device,
    )
    record("BootsTAPIR", elapsed, peak)
    del tapir
    clear_cuda()

    dino = load_dino(args.seganymo_root)
    warm_dino = run_dino(dino, warm_frames, args.step)
    test_dino, elapsed, peak = timed_cuda(
        lambda: run_dino(dino, test_frames, args.step),
        device,
    )
    record("DINOv2", elapsed, peak)
    del dino
    clear_cuda()

    depth = load_depth_pipeline()
    warm_depth = run_depth(depth, warm_frames)
    test_depth, elapsed, peak = timed_cuda(lambda: run_depth(depth, test_frames), device)
    record("Depth Anything V2", elapsed, peak)
    del depth
    clear_cuda()

    motion = load_motion_model(args.seganymo_root, args.motion_checkpoint, device)
    warm_dynamic = run_motion(
        motion,
        warm_tracks,
        warm_dino,
        warm_depth,
        args.seed,
        device,
    )
    test_dynamic, elapsed, peak = timed_cuda(
        lambda: run_motion(
            motion,
            test_tracks,
            test_dino,
            test_depth,
            args.seed,
            device,
        ),
        device,
    )
    record("motion head", elapsed, peak)
    del motion
    clear_cuda()

    sam2 = load_sam2(args.seganymo_root, args.sam2_checkpoint)
    warm_video_dir = make_frame_subset(
        warm_paths,
        output_dir / "_frame_subsets" / args.warmup_sequence,
    )
    test_video_dir = make_frame_subset(
        test_paths,
        output_dir / "_frame_subsets" / args.sequence,
    )
    run_sam2(args.seganymo_root, sam2, warm_video_dir, warm_dynamic)
    _, elapsed, peak = timed_cuda(
        lambda: run_sam2(args.seganymo_root, sam2, test_video_dir, test_dynamic),
        device,
    )
    record("iterative SAM2", elapsed, peak)
    del sam2
    clear_cuda()

    total_seconds = sum(float(row["total_seconds"]) for row in stage_rows)
    summary = {
        "sequence": args.sequence,
        "num_frames": len(test_frames),
        "measured": True,
        "total_seconds": total_seconds,
        "amortized_ms_per_frame": total_seconds * 1000.0 / len(test_frames),
        "peak_vram_gib": max(float(row["peak_vram_gib"]) for row in stage_rows),
        "stages": stage_rows,
    }
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SegAnyMo runtime profiling requires CUDA.")
    if args.num_frames <= 0 or args.warmup_frames <= 0:
        raise ValueError("Frame counts must be positive.")

    args.seganymo_root = args.seganymo_root.resolve()
    args.motion_checkpoint = args.motion_checkpoint.resolve()
    args.tapir_checkpoint = args.tapir_checkpoint.resolve()
    args.sam2_checkpoint = args.sam2_checkpoint.resolve()
    args.davis_root = args.davis_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    warm_paths = frame_paths(args.davis_root, args.warmup_sequence, args.warmup_frames)
    test_paths = frame_paths(args.davis_root, args.sequence, args.num_frames)
    warm_frames = load_rgb(warm_paths)
    test_frames = load_rgb(test_paths)

    print(
        f"SegAnyMo: per-stage warm-up on {len(warm_frames)} "
        f"DAVIS/{args.warmup_sequence} frames, then measurement on "
        f"{len(test_frames)} DAVIS/{args.sequence} frames "
        f"(grid={args.grid_size}, step={args.step})"
    )
    measured = profile_pair(
        warm_frames,
        warm_paths,
        test_frames,
        test_paths,
        args,
        output_dir,
        device,
    )

    result = {
        "method": "SegAnyMo",
        "protocol": {
            "warmup": f"{args.warmup_frames} frames from DAVIS/{args.warmup_sequence}",
            "measured": f"{args.num_frames} frames from DAVIS/{args.sequence}",
            "grid_size": args.grid_size,
            "query_stride": args.step,
            "model_loading_included": False,
            "initial_source_image_disk_reads_included": False,
            "sam2_frame_decode_included": True,
            "result_file_writes_included": False,
            "latency_statistic": "whole-sequence total divided by output frames",
            "peak_statistic": "maximum allocated VRAM across sequential stages",
        },
        "measured": measured,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    with (output_dir / "stages.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "stage",
                "num_frames",
                "total_seconds",
                "amortized_ms_per_frame",
                "peak_vram_gib",
            ],
        )
        writer.writeheader()
        writer.writerows(measured["stages"])

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "amortized_ms_per_frame",
                "peak_vram_gib",
                "num_frames",
                "grid_size",
                "query_stride",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "SegAnyMo",
                "amortized_ms_per_frame": measured["amortized_ms_per_frame"],
                "peak_vram_gib": measured["peak_vram_gib"],
                "num_frames": args.num_frames,
                "grid_size": args.grid_size,
                "query_stride": args.step,
            }
        )
    print(json.dumps(result["measured"], indent=2))


if __name__ == "__main__":
    main()
