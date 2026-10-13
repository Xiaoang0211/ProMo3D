#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.models.map_anything import (  # noqa: E402
    SOURCE_CHOICES,
    build_feature_key_order,
    extract_feature_map,
    load_mapanything,
    normalize_token_sources,
)
from isdynamic.core.artifacts import artifact_path  # noqa: E402
from isdynamic.core.common_labels import (  # noqa: E402
    dynamic_ratio_to_binary_labels,
    mask_to_token_dynamic_ratio,
)
from isdynamic.core.preprocess_aspect_pad import aspect_pad_arrays  # noqa: E402
from mapanything.utils.cropping import crop_resize_if_necessary  # noqa: E402
from mapanything.utils.image import find_closest_aspect_ratio  # noqa: E402
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT  # noqa: E402


MaskRef = Union[Path, List[Path], None]
PART_KEYS = ("tokens", "is_dynamic", "valid_mask", "group_ids")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from DAVIS-2017 using MapAnything image-only features."
        )
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p",
        help="Path to DAVIS root or parent containing DAVIS/.",
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
        default="davis",
        help="Dataset subfolder name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=(
            "Which export split to write. "
            "Split policy: train=official train split; val/test are carved from official val split."
        ),
    )
    parser.add_argument(
        "--token_sources",
        type=str,
        default="inter_frame_block,intra_frame_block,dense_head,pose_head,scale_head",
        help=(
            "Comma-separated source groups. "
            f"Choices: {', '.join(SOURCE_CHOICES)}."
        ),
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Dynamic ratio threshold for token labels.",
    )
    parser.add_argument(
        "--val_seq_count",
        type=int,
        default=10,
        help="Number of sequences from official DAVIS val split to use as val.",
    )
    parser.add_argument(
        "--test_seq_count",
        type=int,
        default=20,
        help="Number of sequences from official DAVIS val split to use as test.",
    )
    parser.add_argument(
        "--val_test_split_mode",
        type=str,
        default="shuffled",
        choices=["sorted", "shuffled"],
        help="How to pick val/test sequences from official DAVIS val split.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used when --val_test_split_mode=shuffled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for token subsampling.")
    parser.add_argument(
        "--machine",
        type=str,
        default="aws",
        help="Hydra machine config used to compose the base model config.",
    )
    parser.add_argument(
        "--resolution_set",
        type=int,
        default=518,
        choices=[504, 512, 518],
        help="Resolution mapping set to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    return parser.parse_args()


def _parse_token_sources(raw: str) -> List[str]:
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    if not sources:
        raise ValueError("--token_sources must contain at least one source.")
    unknown = [s for s in sources if s not in SOURCE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown token_sources {unknown}. Choices: {SOURCE_CHOICES}")
    return normalize_token_sources(sources)


def resolve_davis_root(path: str) -> Path:
    root = Path(path)
    if (root / "DAVIS").exists():
        root = root / "DAVIS"
    required = [
        root / "ImageSets" / "2017",
        root / "JPEGImages" / "480p",
        root / "Annotations" / "480p",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Invalid DAVIS root. Missing paths:\n" + "\n".join(missing))
    return root


def read_split_sequences(davis_root: Path, split: str) -> List[str]:
    split_file = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        return sorted(line.strip() for line in handle if line.strip())


def build_davis_export_splits(
    davis_root: Path,
    val_seq_count: int,
    test_seq_count: int,
    val_test_split_mode: str,
    split_seed: int,
) -> Dict[str, List[str]]:
    train_sequences = read_split_sequences(davis_root, "train")
    official_val_sequences = read_split_sequences(davis_root, "val")

    if val_seq_count < 0 or test_seq_count < 0:
        raise ValueError("--val_seq_count and --test_seq_count must be >= 0")
    if val_seq_count + test_seq_count > len(official_val_sequences):
        raise ValueError(
            "Requested val/test counts exceed official DAVIS val split size: "
            f"val={val_seq_count}, test={test_seq_count}, available={len(official_val_sequences)}"
        )

    if val_test_split_mode == "sorted":
        chosen = list(official_val_sequences)
    else:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(official_val_sequences))
        chosen = [official_val_sequences[int(i)] for i in perm]

    return {
        "train": train_sequences,
        "val": chosen[:val_seq_count],
        "test": chosen[val_seq_count : val_seq_count + test_seq_count],
        "unused_official_val": chosen[val_seq_count + test_seq_count :],
    }


def load_davis_sequence_paths(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence
    image_paths = sorted(image_dir.glob("*.jpg"))
    if len(image_paths) == 0:
        image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def list_sequence_images(image_dir: Path) -> List[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_dir.glob(pattern)))
    dedup = sorted(set(paths))
    if not dedup:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return dedup


def _is_fbms_mask_path(mask_path: Path) -> bool:
    return mask_path.parent.name == "GroundTruth" and any(part in {"Trainingset", "Testset"} for part in mask_path.parts)

def _normalize_fbms_stem(stem: str) -> str:
    normalized = stem.strip().lower()
    if normalized.endswith("_gt"):
        normalized = normalized[:-3]
    return normalized


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


def _encode_fbms_mask_values(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        rgb = mask_arr.astype(np.uint32)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return mask_arr.astype(np.uint32)


def _fbms_mask_to_binary(mask_path: Path) -> np.ndarray:
    background_scale, _ = _parse_fbms_definition(mask_path.parent)
    mask_arr = np.array(Image.open(mask_path))
    encoded = _encode_fbms_mask_values(mask_arr)
    return (encoded != np.uint32(background_scale)).astype(np.uint8)

def load_binary_mask(mask_ref: MaskRef, fallback_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if mask_ref is None:
        if fallback_size is None:
            raise ValueError("fallback_size is required when mask_ref is None.")
        width, height = fallback_size
        return np.zeros((height, width), dtype=np.uint8)

    refs = [mask_ref] if isinstance(mask_ref, Path) else mask_ref
    union_mask = None
    for ref in refs:
        if _is_fbms_mask_path(ref):
            mask = _fbms_mask_to_binary(ref)
        else:
            mask = np.array(Image.open(ref))
            if mask.ndim == 3:
                mask = np.any(mask > 0, axis=-1)
            mask = (mask > 0).astype(np.uint8)
        if union_mask is None:
            union_mask = mask
        else:
            union_mask = np.maximum(union_mask, mask)
    if union_mask is None:
        raise ValueError("mask_ref resolved to no masks.")
    return union_mask


def build_image_normalizer(norm_type: str):
    if norm_type not in IMAGE_NORMALIZATION_DICT:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Choices: {sorted(IMAGE_NORMALIZATION_DICT.keys())}"
        )
    img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    return tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
    )


def preprocess_entries(
    entries: Sequence[Tuple[Path, MaskRef]],
    norm_type: str,
    resolution_set: int,
    preprocess_mode: str = "native",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    images = [Image.open(image_path).convert("RGB") for image_path, _ in entries]
    masks = [load_binary_mask(mask_ref, fallback_size=image.size) for image, (_, mask_ref) in zip(images, entries)]

    normalize = build_image_normalizer(norm_type)

    image_tensors: List[torch.Tensor] = []
    mask_tensors: List[torch.Tensor] = []
    if preprocess_mode == "native":
        aspect_ratios = [image.size[0] / image.size[1] for image in images]
        target_size = find_closest_aspect_ratio(
            sum(aspect_ratios) / len(aspect_ratios),
            resolution_set,
        )
        for image, mask_np in zip(images, masks):
            processed_image, additional_quantities = crop_resize_if_necessary(
                image=image,
                resolution=target_size,
                additional_quantities=[mask_np],
            )
            resized_mask = additional_quantities[0]
            image_tensors.append(normalize(processed_image))
            mask_tensors.append(torch.from_numpy((resized_mask > 0).astype(np.float32)))
    elif preprocess_mode == "aspect_pad":
        for image, mask_np in zip(images, masks):
            image_arr, mask_arr = aspect_pad_arrays(image, mask_np, target_size=resolution_set, patch_size=14)
            image_tensors.append(normalize(Image.fromarray(image_arr)))
            mask_tensors.append(torch.from_numpy(mask_arr.astype(np.float32)))
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    return image_tensors, mask_tensors


def sample_frame_indices(
    frame_valid: torch.Tensor,
    tokens_per_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_idx = torch.nonzero(frame_valid, as_tuple=False).squeeze(-1).cpu().numpy()
    if valid_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if tokens_per_frame > 0:
        k = min(tokens_per_frame, valid_idx.size)
        if 0 in valid_idx:
            remaining = valid_idx[valid_idx != 0]
            if k == 1:
                chosen = np.array([0], dtype=np.int64)
            else:
                sampled_rest = rng.choice(remaining, size=k - 1, replace=False)
                chosen = np.concatenate([np.array([0], dtype=np.int64), sampled_rest.astype(np.int64)])
        else:
            chosen = rng.choice(valid_idx, size=k, replace=False)
    else:
        chosen = valid_idx
    return np.asarray(chosen, dtype=np.int64)


def _atomic_savez_compressed(output_path: Path, **payload) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=output_path.stem + "_",
            suffix=".npz.tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(output_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _is_valid_part_file(part_path: Path) -> bool:
    try:
        with np.load(part_path, allow_pickle=False) as part:
            if any(key not in part for key in PART_KEYS):
                return False
            tokens = part["tokens"]
            labels = part["is_dynamic"]
            valid_mask = part["valid_mask"]
            group_ids = part["group_ids"]
        if tokens.ndim != 2:
            return False
        token_count = int(tokens.shape[0])
        return (
            labels.shape[0] == token_count
            and valid_mask.shape[0] == token_count
            and group_ids.shape[0] == token_count
        )
    except (EOFError, OSError, ValueError, zipfile.BadZipFile, KeyError):
        return False


class SingleNpzSink:
    def __init__(self, feature_keys: List[str]):
        self.feature_keys = feature_keys
        self.features: Dict[str, List[np.ndarray]] = {key: [] for key in feature_keys}
        self.labels: List[np.ndarray] = []
        self.valids: List[np.ndarray] = []
        self.groups: List[np.ndarray] = []

    def append(
        self,
        feature_map: Dict[str, np.ndarray],
        labels: np.ndarray,
        valid_mask: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        for feature_key in self.feature_keys:
            self.features[feature_key].append(feature_map[feature_key])
        self.labels.append(labels)
        self.valids.append(valid_mask)
        self.groups.append(group_ids)

    def save(self, output_path: Path) -> None:
        payload = {
            feature_key: np.concatenate(parts, axis=0)
            for feature_key, parts in self.features.items()
        }
        payload["is_dynamic"] = np.concatenate(self.labels, axis=0)
        payload["valid_mask"] = np.concatenate(self.valids, axis=0)
        payload["group_ids"] = np.concatenate(self.groups, axis=0)
        _atomic_savez_compressed(output_path, **payload)


class ShardedDirSink:
    def __init__(self, root_dir: Path, feature_keys: List[str]):
        self.root_dir = root_dir
        self.feature_keys = feature_keys
        self.part_idx = 0

    def sequence_complete(self, sequence_name: str) -> bool:
        part_counts = []
        for feature_key in self.feature_keys:
            feat_dir = self.root_dir / sequence_name / feature_key
            if not feat_dir.exists() or not feat_dir.is_dir():
                return False
            part_files = sorted(feat_dir.glob("part_*.npz"))
            count = len(part_files)
            if count == 0:
                return False
            if not all(_is_valid_part_file(part_path) for part_path in part_files):
                return False
            part_counts.append(count)
        return len(set(part_counts)) == 1

    def reset_sequence(self, sequence_name: str) -> None:
        shutil.rmtree(self.root_dir / sequence_name, ignore_errors=True)

    def append(
        self,
        sequence_name: str,
        feature_map: Dict[str, np.ndarray],
        labels: np.ndarray,
        valid_mask: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        for feature_key in self.feature_keys:
            feat_dir = self.root_dir / sequence_name / feature_key
            feat_dir.mkdir(parents=True, exist_ok=True)
            out_path = feat_dir / f"part_{self.part_idx:06d}.npz"
            _atomic_savez_compressed(
                out_path,
                tokens=feature_map[feature_key],
                is_dynamic=labels,
                valid_mask=valid_mask,
                group_ids=group_ids,
            )
        self.part_idx += 1


def collect_split_samples(
    model,
    davis_root: Path,
    split_name: str,
    sequences: List[str],
    token_sources: List[str],
    feature_key_order: List[str],
    frames_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    device: torch.device,
    seed: int,
    resolution_set: int,
    sink,
) -> None:
    rng = np.random.default_rng(seed)
    group_counter = 0
    norm_type = model.encoder.data_norm_type

    for sequence in sequences:
        if isinstance(sink, ShardedDirSink):
            if sink.sequence_complete(sequence):
                print(f"[{split_name}] skipped existing {sequence}")
                continue
            sink.reset_sequence(sequence)
        image_paths, mask_paths = load_davis_sequence_paths(davis_root, sequence)

        for start in range(0, len(image_paths), frames_per_forward):
            end = min(start + frames_per_forward, len(image_paths))
            entries = list(zip(image_paths[start:end], mask_paths[start:end]))
            image_tensors, mask_tensors = preprocess_entries(
                entries,
                norm_type=norm_type,
                resolution_set=resolution_set,
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

            sampled_features: Dict[str, List[np.ndarray]] = {key: [] for key in feature_key_order}
            sampled_labels: List[np.ndarray] = []
            sampled_valids: List[np.ndarray] = []
            sampled_groups: List[np.ndarray] = []

            labels_flat = labels.reshape(labels.shape[1], -1)
            valid_flat = valid_mask.reshape(valid_mask.shape[1], -1)
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
                    sequence_name=sequence,
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

        print(f"[{split_name}] processed {sequence}")


def main() -> None:
    args = parse_args()
    davis_root = resolve_davis_root(args.davis_root)
    token_sources = _parse_token_sources(args.token_sources)
    device = torch.device(args.device)
    model = load_mapanything(
        checkpoint_path=args.checkpoint,
        device=device,
        machine=args.machine,
    )

    feature_key_order = build_feature_key_order(model=model, token_sources=token_sources)
    split_map = build_davis_export_splits(
        davis_root=davis_root,
        val_seq_count=args.val_seq_count,
        test_seq_count=args.test_seq_count,
        val_test_split_mode=args.val_test_split_mode,
        split_seed=args.split_seed,
    )
    output_root = Path(args.output_dir) / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    selected_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split_name in selected_splits:
        if args.save_mode == "sharded_dirs":
            sink = ShardedDirSink(
                root_dir=output_root / args.tokens_subdir / split_name,
                feature_keys=feature_key_order,
            )
        else:
            sink = SingleNpzSink(feature_keys=feature_key_order)

        collect_split_samples(
            model=model,
            davis_root=davis_root,
            split_name=split_name,
            sequences=split_map[split_name],
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            frames_per_forward=args.frames_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            device=device,
            seed=args.seed,
            resolution_set=args.resolution_set,
            sink=sink,
        )

        if isinstance(sink, SingleNpzSink):
            sink.save(output_root / f"{split_name}_tokens.npz")

    meta = {
        "dataset_name": args.dataset_name,
        "feature_keys": feature_key_order,
        "token_sources": token_sources,
        "splits": split_map,
        "checkpoint": args.checkpoint,
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


if __name__ == "__main__":
    main()
