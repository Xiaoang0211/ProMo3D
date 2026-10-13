#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.legacy.map_anything.build_davis_isdynamic_npz import (  # noqa: E402
    SOURCE_CHOICES,
    ShardedDirSink,
    SingleNpzSink,
    _parse_token_sources,
    list_sequence_images,
    preprocess_entries,
    read_split_sequences,
    resolve_davis_root,
    sample_frame_indices,
)
from isdynamic.models.map_anything import (  # noqa: E402
    build_feature_key_order,
    extract_feature_map,
    load_mapanything,
)
from isdynamic.core.artifacts import artifact_path  # noqa: E402
from isdynamic.core.common_labels import (  # noqa: E402
    dynamic_ratio_to_binary_labels,
    mask_to_token_dynamic_ratio,
)

MaskRef = Union[Path, List[Path], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from YouTube-VOS motion splits using "
            "MapAnything image-only features, with DAVIS and SegTrackv2 as separate test sets."
        )
    )
    parser.add_argument(
        "--youtubevos_root",
        type=str,
        required=True,
        help="Path to YouTube-VOS root containing train_motion/ and valid_motion/.",
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p",
        help="Path to DAVIS root or parent containing DAVIS/.",
    )
    parser.add_argument(
        "--segtrack_root",
        type=str,
        default="/home/xiaoang/Datasets/SegTrackv2",
        help="Path to SegTrackv2 root.",
    )
    parser.add_argument(
        "--fbms_root",
        type=str,
        default="/home/xiaoang/Datasets/FBMS-59",
        help="Path to FBMS-59 root containing Trainingset/ and Testset/.",
    )
    parser.add_argument(
        "--fbms_infer_all_frames",
        action="store_true",
        help="For FBMS-59 only, run inference on all frames and export tokens only for annotated frames.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a MapAnything checkpoint file (.pt/.pth) or local HF checkpoint dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=artifact_path("data", "isdynamic"),
        help="Output folder for exports.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="youtubevos",
        help="Dataset subfolder name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test", "test_davis", "test_segtrackv2", "test_fbms59"],
        help="Which export split to write.",
    )
    parser.add_argument(
        "--youtubevos_train_split",
        type=str,
        default="train_motion",
        help="YouTube-VOS split directory to use as training split.",
    )
    parser.add_argument(
        "--youtubevos_val_split",
        type=str,
        default="valid_motion",
        help="YouTube-VOS split directory to use as validation split.",
    )
    parser.add_argument(
        "--token_sources",
        type=str,
        default="inter_frame_block,intra_frame_block,dense_head,pose_head,scale_head",
        help=f"Comma-separated source groups. Choices: {', '.join(SOURCE_CHOICES)}.",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="sharded_dirs",
        choices=["sharded_dirs", "single_npz"],
        help="sharded_dirs writes per-feature directories; single_npz writes one file per split.",
    )
    parser.add_argument(
        "--tokens_subdir",
        type=str,
        default="tokens",
        help="Subdirectory under output dataset for sharded feature directories.",
    )
    parser.add_argument(
        "--tokens_per_frame",
        type=int,
        default=64,
        help="Subsample this many valid tokens per frame. Use <=0 for all tokens.",
    )
    parser.add_argument(
        "--frames_per_forward",
        type=int,
        default=8,
        help="Number of frames per model forward chunk.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Dynamic ratio threshold for token labels.")
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="native",
        choices=["native", "aspect_pad"],
        help="native keeps MapAnything crop/resize; aspect_pad uses Pi3-style resize-and-pad.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for token subsampling.")
    parser.add_argument(
        "--machine",
        type=str,
        default="aws",
        help="Hydra machine config used to compose the base model config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    return parser.parse_args()


def resolve_youtubevos_root(path: str) -> Path:
    root = Path(path)
    candidates = [root]
    for child_name in ("Youtube-VOS", "YouTube-VOS", "youtubevos", "youtube_vos"):
        if (root / child_name).exists():
            candidates.append(root / child_name)

    required_splits = ("train_motion", "valid_motion")
    for candidate in candidates:
        if all((candidate / split_name).exists() for split_name in required_splits):
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve YouTube-VOS root with required split folders train_motion/ and valid_motion/.\n"
        f"Checked:\n{checked}"
    )


def _resolve_split_dirs(split_root: Path) -> Tuple[Path, Path]:
    direct_img = split_root / "JPEGImages"
    typo_img = split_root / "JEPGImages"
    direct_ann = split_root / "Annotations"
    if direct_img.exists() and direct_ann.exists():
        return direct_img, direct_ann
    if typo_img.exists() and direct_ann.exists():
        return typo_img, direct_ann
    if (direct_img / "480p").exists() and (direct_ann / "480p").exists():
        return direct_img / "480p", direct_ann / "480p"
    if (typo_img / "480p").exists() and (direct_ann / "480p").exists():
        return typo_img / "480p", direct_ann / "480p"
    if (split_root / "images").exists() and (split_root / "annotations").exists():
        return split_root / "images", split_root / "annotations"
    raise FileNotFoundError(
        f"Unsupported split structure under {split_root}. "
        "Expected JPEGImages/ + Annotations/ (optionally with 480p/)."
    )


def list_youtubevos_sequences(youtubevos_root: Path, split_name: str) -> List[str]:
    split_root = youtubevos_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"YouTube-VOS split not found: {split_root}")
    image_root, ann_root = _resolve_split_dirs(split_root)
    sequences = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No sequence directories found under {image_root}")
    missing = [seq for seq in sequences if not (ann_root / seq).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing annotation directories for {len(missing)} sequences under {ann_root}. "
            f"Examples: {missing[:10]}"
        )
    return sequences


def load_youtubevos_sequence_paths(
    youtubevos_root: Path,
    split_name: str,
    sequence: str,
) -> Tuple[List[Path], List[Path]]:
    split_root = youtubevos_root / split_name
    image_root, ann_root = _resolve_split_dirs(split_root)
    image_dir = image_root / sequence
    mask_dir = ann_root / sequence
    image_paths = list_sequence_images(image_dir)
    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def resolve_segtrack_root(path: str) -> Path:
    root = Path(path)
    required = [
        root / "JPEGImages",
        root / "GroundTruth",
        root / "ImageSets",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Invalid SegTrackv2 root. Missing paths:\n" + "\n".join(missing))
    return root


def list_segtrack_sequences(segtrack_root: Path) -> List[str]:
    all_file = segtrack_root / "ImageSets" / "all.txt"
    if all_file.exists():
        with all_file.open("r", encoding="utf-8") as handle:
            sequences = sorted(
                line.strip().lstrip("*").strip()
                for line in handle
                if line.strip()
            )
        if sequences:
            return sequences
    return sorted(path.name for path in (segtrack_root / "JPEGImages").iterdir() if path.is_dir())




def resolve_fbms_root(path: str) -> Path:
    root = Path(path)
    candidates = [root]
    for child_name in ("FBMS-59", "fbms-59", "FBMS59", "fbms59"):
        if (root / child_name).exists():
            candidates.append(root / child_name)

    for candidate in candidates:
        if (candidate / "Trainingset").exists() and (candidate / "Testset").exists():
            return candidate

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve FBMS-59 root with Trainingset/ and Testset/.\n"
        f"Checked:\n{checked}"
    )


def list_fbms_sequences(fbms_root: Path) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []
    for split_name in ("Trainingset", "Testset"):
        split_dir = fbms_root / split_name
        if not split_dir.exists():
            continue
        for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if (seq_dir / "GroundTruth").exists():
                sequences.append((split_name, seq_dir.name))
    if not sequences:
        raise RuntimeError(f"No FBMS-59 sequences found under {fbms_root}")
    return sequences


def _normalize_fbms_stem(stem: str) -> str:
    normalized = stem.strip().lower()
    if normalized.endswith("_gt"):
        normalized = normalized[:-3]
    return normalized


def _list_fbms_images(sequence_dir: Path) -> List[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts
    )
    if not image_paths:
        raise FileNotFoundError(f"No FBMS images found in {sequence_dir}")
    return image_paths


def _parse_fbms_definition(gt_dir: Path) -> Tuple[int, Dict[str, Path]]:
    dat_paths = sorted(path for path in gt_dir.glob("*Def.dat") if path.is_file())
    if not dat_paths:
        dat_paths = sorted(path for path in gt_dir.glob("*.dat") if path.is_file())
    if not dat_paths:
        raise FileNotFoundError(f"No FBMS .dat definition file found in {gt_dir}")

    dat_path = dat_paths[0]
    background_scale = None
    masks_by_stem: Dict[str, Path] = {}
    pending_mask_name = None
    lines = [line.strip() for line in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines()]

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line == "Scale of region 0:" and idx + 1 < len(lines):
            background_scale = int(lines[idx + 1])
            idx += 2
            continue
        if line == "File name:" and idx + 1 < len(lines):
            pending_mask_name = lines[idx + 1].strip()
            idx += 2
            continue
        if line == "Input file name:" and idx + 1 < len(lines):
            input_name = lines[idx + 1].strip()
            idx += 2
            if pending_mask_name:
                mask_path = gt_dir / pending_mask_name
                if mask_path.exists():
                    masks_by_stem[_normalize_fbms_stem(Path(input_name).stem)] = mask_path
                pending_mask_name = None
            continue
        idx += 1

    if background_scale is None:
        raise RuntimeError(f"Could not parse FBMS background scale from {dat_path}")
    if not masks_by_stem:
        raise RuntimeError(f"Could not parse FBMS official mask mapping from {dat_path}")
    return int(background_scale), masks_by_stem


def _choose_fbms_masks(gt_dir: Path) -> Dict[str, Path]:
    try:
        _, masks_by_stem = _parse_fbms_definition(gt_dir)
        return masks_by_stem
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    mask_exts = {".pgm", ".ppm", ".bmp", ".jpg", ".jpeg"}
    suffix_priority = {".pgm": 0, ".ppm": 1, ".bmp": 2, ".jpg": 3, ".jpeg": 4}
    raw_masks = sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in mask_exts
        and "prob" not in path.stem.lower()
    )
    if not raw_masks:
        raise FileNotFoundError(f"No FBMS official masks found in {gt_dir}")

    chosen: Dict[str, Tuple[Tuple[int, str], Path]] = {}
    for mask_path in raw_masks:
        key = _normalize_fbms_stem(mask_path.stem)
        score = (suffix_priority.get(mask_path.suffix.lower(), 99), mask_path.name.lower())
        current = chosen.get(key)
        if current is None or score < current[0]:
            chosen[key] = (score, mask_path)
    return {key: payload[1] for key, payload in chosen.items()}


def load_fbms_sequence_paths(
    fbms_root: Path,
    split_name: str,
    sequence: str,
    include_unannotated: bool = False,
) -> Tuple[List[Path], List[MaskRef]]:
    sequence_dir = fbms_root / split_name / sequence
    if not sequence_dir.exists():
        raise FileNotFoundError(f"FBMS sequence directory not found: {sequence_dir}")
    gt_dir = sequence_dir / "GroundTruth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"FBMS GroundTruth directory not found: {gt_dir}")

    image_paths = _list_fbms_images(sequence_dir)
    masks_by_stem = _choose_fbms_masks(gt_dir)

    paired_images: List[Path] = []
    paired_masks: List[MaskRef] = []
    for image_path in image_paths:
        key = _normalize_fbms_stem(image_path.stem)
        mask_path = masks_by_stem.get(key)
        if mask_path is None and not include_unannotated:
            continue
        paired_images.append(image_path)
        paired_masks.append(mask_path)

    if not paired_images:
        raise FileNotFoundError(
            f"No overlapping image/mask stems found for FBMS sequence: {split_name}/{sequence}"
        )
    return paired_images, paired_masks

def load_segtrack_sequence_paths(
    segtrack_root: Path,
    sequence: str,
) -> Tuple[List[Path], List[List[Path]]]:
    normalized_sequence = sequence.lstrip("*").strip()
    image_dir = segtrack_root / "JPEGImages" / normalized_sequence
    gt_dir = segtrack_root / "GroundTruth" / normalized_sequence
    image_paths = list_sequence_images(image_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")

    direct_mask_paths: List[Path] = []
    for pattern in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
        direct_mask_paths.extend(sorted(gt_dir.glob(pattern)))
    direct_masks = {path.stem: [path] for path in sorted(set(direct_mask_paths))}
    if direct_masks:
        mask_refs = []
        for image_path in image_paths:
            if image_path.stem not in direct_masks:
                raise FileNotFoundError(f"Missing mask for {image_path}")
            mask_refs.append(direct_masks[image_path.stem])
        return image_paths, mask_refs

    instance_dirs = sorted(path for path in gt_dir.iterdir() if path.is_dir())
    if not instance_dirs:
        raise RuntimeError(f"No direct masks or instance directories found under {gt_dir}")

    mask_refs: List[List[Path]] = []
    for image_path in image_paths:
        refs = [instance_dir / f"{image_path.stem}.png" for instance_dir in instance_dirs]
        refs = [path for path in refs if path.exists()]
        if not refs:
            raise FileNotFoundError(f"Missing SegTrack masks for {image_path}")
        mask_refs.append(refs)
    return image_paths, mask_refs


def build_export_splits(
    youtubevos_root: Path,
    youtubevos_train_split: str,
    youtubevos_val_split: str,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
) -> Dict[str, List[Dict[str, str]]]:
    return {
        "train": [
            {"dataset": "youtubevos", "split_name": youtubevos_train_split, "sequence": sequence}
            for sequence in list_youtubevos_sequences(youtubevos_root, youtubevos_train_split)
        ],
        "val": [
            {"dataset": "youtubevos", "split_name": youtubevos_val_split, "sequence": sequence}
            for sequence in list_youtubevos_sequences(youtubevos_root, youtubevos_val_split)
        ],
        "test_davis": [
            {"dataset": "davis", "split_name": split_name, "sequence": sequence}
            for split_name in ("train", "val")
            for sequence in read_split_sequences(davis_root, split_name)
        ],
        "test_segtrackv2": [
            {"dataset": "segtrackv2", "split_name": "all", "sequence": sequence}
            for sequence in list_segtrack_sequences(segtrack_root)
        ],
        "test_fbms59": [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
        "test": [
            {"dataset": "davis", "split_name": split_name, "sequence": sequence}
            for split_name in ("train", "val")
            for sequence in read_split_sequences(davis_root, split_name)
        ] + [
            {"dataset": "segtrackv2", "split_name": "all", "sequence": sequence}
            for sequence in list_segtrack_sequences(segtrack_root)
        ] + [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
    }


def load_sequence_entries_from_spec(
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    spec: Dict[str, str],
    fbms_infer_all_frames: bool = False,
) -> List[Tuple[Path, MaskRef]]:
    dataset = spec["dataset"]
    sequence = spec["sequence"]
    split_name = spec["split_name"]
    if dataset == "youtubevos":
        image_paths, mask_paths = load_youtubevos_sequence_paths(youtubevos_root, split_name, sequence)
    elif dataset == "davis":
        image_paths, mask_paths = read_sequence_paths_davis(davis_root, sequence)
    elif dataset == "segtrackv2":
        image_paths, mask_paths = load_segtrack_sequence_paths(segtrack_root, sequence)
    elif dataset == "fbms59":
        image_paths, mask_paths = load_fbms_sequence_paths(
            fbms_root,
            split_name,
            sequence,
            include_unannotated=fbms_infer_all_frames,
        )
    else:
        raise ValueError(f"Unsupported dataset spec: {dataset}")
    return list(zip(image_paths, mask_paths))


def read_sequence_paths_davis(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence
    image_paths = list_sequence_images(image_dir)
    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def collect_split_specs(
    model,
    split_name: str,
    sequence_specs: List[Dict[str, str]],
    token_sources: List[str],
    feature_key_order: List[str],
    frames_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    fbms_infer_all_frames: bool,
    preprocess_mode: str,
    device: torch.device,
    seed: int,
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    sink,
) -> None:
    rng = np.random.default_rng(seed)
    group_counter = 0
    norm_type = model.encoder.data_norm_type

    for spec in sequence_specs:
        sequence = spec["sequence"]
        seq_name = f"{spec['dataset']}__{sequence}"
        if isinstance(sink, ShardedDirSink):
            if sink.sequence_complete(seq_name):
                print(f"[{split_name}] skipped existing {spec['dataset']}:{sequence}")
                continue
            sink.reset_sequence(seq_name)
        entries = load_sequence_entries_from_spec(
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            spec=spec,
            fbms_infer_all_frames=fbms_infer_all_frames,
        )
        for start in range(0, len(entries), frames_per_forward):
            chunk = entries[start : start + frames_per_forward]
            image_tensors, mask_tensors = preprocess_entries(
                chunk,
                norm_type=norm_type,
                resolution_set=518,
                preprocess_mode=preprocess_mode,
            )
            views = [
                {
                    "img": image_tensor.unsqueeze(0).to(device),
                    "data_norm_type": [norm_type],
                    "true_shape": np.int32([image_tensor.shape[-2:]]),
                    "idx": view_idx,
                    "instance": str(view_idx),
                }
                for view_idx, image_tensor in enumerate(image_tensors)
            ]
            masks = torch.stack(mask_tensors, dim=0).unsqueeze(0)

            with torch.no_grad():
                feature_map = extract_feature_map(model, views, token_sources)

            patch_size = int(model.encoder.patch_size)
            dynamic_ratio = mask_to_token_dynamic_ratio(masks, patch_size=patch_size)
            labels, valid_mask = dynamic_ratio_to_binary_labels(dynamic_ratio, threshold=threshold)
            labels_flat = labels.reshape(labels.shape[1], -1)
            valid_flat = valid_mask.reshape(valid_mask.shape[1], -1)
            if spec["dataset"] == "fbms59" and fbms_infer_all_frames:
                annotated_flags = torch.tensor([mask_ref is not None for _, mask_ref in chunk], dtype=torch.bool)
                valid_flat[~annotated_flags] = False

            sampled_features: Dict[str, List[np.ndarray]] = {key: [] for key in feature_key_order}
            sampled_labels: List[np.ndarray] = []
            sampled_valids: List[np.ndarray] = []
            sampled_groups: List[np.ndarray] = []

            for frame_idx in range(labels_flat.shape[0]):
                chosen = sample_frame_indices(valid_flat[frame_idx], tokens_per_frame, rng)
                if chosen.size == 0:
                    continue
                sampled_labels.append(labels_flat[frame_idx, chosen].cpu().numpy())
                sampled_valids.append(valid_flat[frame_idx, chosen].cpu().numpy())
                sampled_groups.append(
                    np.full(chosen.shape[0], group_counter, dtype=np.int64)
                )
                group_counter += 1
                for feature_key in feature_key_order:
                    tokens = feature_map[feature_key][0, frame_idx, chosen]
                    sampled_features[feature_key].append(tokens.detach().cpu().numpy())

            if not sampled_labels:
                continue

            packed_features = {
                feature_key: np.concatenate(parts, axis=0)
                for feature_key, parts in sampled_features.items()
            }
            packed_labels = np.concatenate(sampled_labels, axis=0)
            packed_valids = np.concatenate(sampled_valids, axis=0)
            packed_groups = np.concatenate(sampled_groups, axis=0)

            if isinstance(sink, ShardedDirSink):
                sink.append(
                    sequence_name=seq_name,
                    feature_map=packed_features,
                    labels=packed_labels,
                    valid_mask=packed_valids,
                    group_ids=packed_groups,
                )
            else:
                sink.append(
                    packed_features,
                    packed_labels,
                    packed_valids,
                    packed_groups,
                )

        print(f"[{split_name}] processed {spec['dataset']}:{sequence}")


def main() -> None:
    args = parse_args()
    token_sources = _parse_token_sources(args.token_sources)
    device = torch.device(args.device)
    youtubevos_root = resolve_youtubevos_root(args.youtubevos_root)
    davis_root = resolve_davis_root(args.davis_root)
    segtrack_root = resolve_segtrack_root(args.segtrack_root)
    fbms_root = resolve_fbms_root(args.fbms_root)

    model = load_mapanything(
        checkpoint_path=args.checkpoint,
        device=device,
        machine=args.machine,
    )
    feature_key_order = build_feature_key_order(model=model, token_sources=token_sources)
    split_map = build_export_splits(
        youtubevos_root=youtubevos_root,
        youtubevos_train_split=args.youtubevos_train_split,
        youtubevos_val_split=args.youtubevos_val_split,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
    )

    output_root = Path(args.output_dir) / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    selected_splits = (
        ["train", "val", "test_davis", "test_segtrackv2", "test_fbms59"]
        if args.split == "all"
        else [args.split]
    )

    for split_name in selected_splits:
        if args.save_mode == "sharded_dirs":
            sink = ShardedDirSink(
                root_dir=output_root / args.tokens_subdir / split_name,
                feature_keys=feature_key_order,
            )
        else:
            sink = SingleNpzSink(feature_keys=feature_key_order)

        collect_split_specs(
            model=model,
            split_name=split_name,
            sequence_specs=split_map[split_name],
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            frames_per_forward=args.frames_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            fbms_infer_all_frames=args.fbms_infer_all_frames,
            preprocess_mode=args.preprocess_mode,
            device=device,
            seed=args.seed,
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            sink=sink,
        )

        if isinstance(sink, SingleNpzSink):
            sink.save(output_root / f"{split_name}_tokens.npz")

    meta = {
        "dataset_name": args.dataset_name,
        "feature_keys": feature_key_order,
        "token_sources": token_sources,
        "splits": split_map,
        "fbms_infer_all_frames": bool(args.fbms_infer_all_frames),
        "preprocess_mode": args.preprocess_mode,
        "checkpoint": args.checkpoint,
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


if __name__ == "__main__":
    main()
