#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.legacy.dust3r.build_davis_isdynamic_npz import (
    ShardedDirSink,
    SingleNpzSink,
    _parse_token_sources,
    read_split_sequences,
    resolve_davis_root,
    sample_frame_indices,
)
from isdynamic.models.dust3r import (
    MODEL_NAME_CHOICES,
    SOURCE_CHOICES,
    build_feature_key_order,
    collate_views,
    extract_feature_map,
    load_dust3r_model,
    preprocess_image_and_mask,
)
from isdynamic.core.stereo_labels import dynamic_ratio_to_binary_labels, mask_to_token_dynamic_ratio


MaskRef = Union[Path, List[Path], None]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from YouTube-VOS motion splits, "
            "using DAVIS and SegTrackv2 as separate test splits."
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
        help=(
            "For FBMS, run forward on all frames in each sequence and only keep/export "
            "tokens from frames that have annotation masks."
        ),
    )
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, default=None, help="Path to local Dust3R weights.")
    parser_weights.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=MODEL_NAME_CHOICES,
        help="Released Dust3R model name to download/load via huggingface_hub.",
    )
    parser.add_argument("--output_dir", type=str, default="data/isdynamic", help="Output root for exports.")
    parser.add_argument("--dataset_name", type=str, default="youtubevos", help="Dataset subfolder name.")
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
        default="enc,dec_view1,dec_view2,head_in_view1,head_in_view2",
        help=f"Comma-separated source groups. Choices: {SOURCE_CHOICES}.",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="sharded_dirs",
        choices=["sharded_dirs", "single_npz"],
        help="Write per-feature directories or one flat NPZ per split.",
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
        "--pairs_per_forward",
        type=int,
        default=8,
        help="Number of Dust3R frame pairs per model forward chunk.",
    )
    parser.add_argument(
        "--pair_stride",
        type=int,
        default=1,
        help="Reference-frame offset for adjacent pairing.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Dynamic ratio threshold for token labels.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        choices=[224, 512],
        help="Dust3R image size preset used by preprocessing.",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="native",
        choices=["native", "aspect_pad"],
        help="native keeps Dust3R crop preprocessing; aspect_pad uses Pi3-style resize-and-pad.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for token subsampling.")
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

    parent = root.parent
    if parent.exists():
        for sibling_name in ("Youtube-VOS", "YouTube-VOS", "youtubevos", "youtube_vos"):
            sibling = parent / sibling_name
            if sibling.exists():
                candidates.append(sibling)

    deduped_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    required_splits = ("train_motion", "valid_motion")
    for candidate in deduped_candidates:
        if all((candidate / split_name).exists() for split_name in required_splits):
            return candidate
    checked = "\\n".join(str(p) for p in deduped_candidates)
    raise FileNotFoundError(
        "Could not resolve YouTube-VOS root with required split folders train_motion/ and valid_motion/.\\n"
        f"Checked:\\n{checked}"
    )


def _resolve_split_dirs(split_root: Path) -> Tuple[Path, Path]:
    direct_img = split_root / "JPEGImages"
    typo_img = split_root / "JEPGImages"
    direct_ann = split_root / "Annotations"
    if direct_img.exists() and direct_ann.exists():
        return direct_img, direct_ann
    if typo_img.exists() and direct_ann.exists():
        return typo_img, direct_ann

    if (split_root / "JPEGImages" / "480p").exists() and (split_root / "Annotations" / "480p").exists():
        return split_root / "JPEGImages" / "480p", split_root / "Annotations" / "480p"
    if (split_root / "JEPGImages" / "480p").exists() and (split_root / "Annotations" / "480p").exists():
        return split_root / "JEPGImages" / "480p", split_root / "Annotations" / "480p"

    if (split_root / "images").exists() and (split_root / "annotations").exists():
        return split_root / "images", split_root / "annotations"

    raise FileNotFoundError(
        f"Unsupported split structure under {split_root}. "
        "Expected JPEGImages/ + Annotations/ (optionally with 480p/). "
        "Also accepts the typo JEPGImages/ used by some local exports."
    )


def list_youtubevos_sequences(youtubevos_root: Path, split_name: str) -> List[str]:
    split_root = youtubevos_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"YouTube-VOS split directory not found: {split_root}")
    image_root, ann_root = _resolve_split_dirs(split_root)
    image_dirs = {path.name for path in image_root.iterdir() if path.is_dir()}
    ann_dirs = {path.name for path in ann_root.iterdir() if path.is_dir()}
    sequences = sorted(image_dirs & ann_dirs)
    if not sequences:
        raise RuntimeError(f"No YouTube-VOS sequences found under {split_root}")
    return sequences


def load_youtubevos_sequence_paths(youtubevos_root: Path, split_name: str, sequence: str) -> Tuple[List[Path], List[Path]]:
    split_root = youtubevos_root / split_name
    image_root, ann_root = _resolve_split_dirs(split_root)
    image_dir = image_root / sequence
    mask_dir = ann_root / sequence
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing image/mask dir for sequence '{sequence}' under {split_root}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No YouTube-VOS images found in {image_dir}")

    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing YouTube-VOS masks:\\n" + "\\n".join(missing[:20]))
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
        raise FileNotFoundError("Invalid SegTrackv2 root. Missing paths:\\n" + "\\n".join(missing))
    return root


def list_segtrack_sequences(segtrack_root: Path) -> List[str]:
    all_file = segtrack_root / "ImageSets" / "all.txt"
    if all_file.exists():
        with all_file.open("r", encoding="utf-8") as handle:
            sequences = sorted(line.strip().lstrip("*").strip() for line in handle if line.strip())
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

    checked = "\\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve FBMS-59 root with Trainingset/ and Testset/.\\n"
        f"Checked:\\n{checked}"
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


def _list_image_files(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _is_fbms_mask_path(mask_path: Path) -> bool:
    return mask_path.parent.name == "GroundTruth" and any(part in {"Trainingset", "Testset"} for part in mask_path.parts)


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

def load_segtrack_sequence_paths(segtrack_root: Path, sequence: str) -> Tuple[List[Path], List[MaskRef]]:
    image_dir = segtrack_root / "JPEGImages" / sequence
    gt_dir = segtrack_root / "GroundTruth" / sequence
    image_paths = _list_image_files(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir} (extensions: {sorted(IMAGE_EXTENSIONS)})")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")

    direct_masks = {}
    for pattern in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
        for path in gt_dir.glob(pattern):
            direct_masks.setdefault(path.stem, []).append(path)
    if direct_masks:
        refs = []
        for image_path in image_paths:
            options = direct_masks.get(image_path.stem)
            if options is None:
                raise FileNotFoundError(f"Missing mask for {image_path}")
            refs.append(options[0])
        return image_paths, refs

    instance_dirs = sorted(path for path in gt_dir.iterdir() if path.is_dir())
    if not instance_dirs:
        raise RuntimeError(f"No direct masks or instance directories found under {gt_dir}")

    refs = []
    for image_path in image_paths:
        options = [instance_dir / f"{image_path.stem}.png" for instance_dir in instance_dirs]
        options = [path for path in options if path.exists()]
        if not options:
            raise FileNotFoundError(f"Missing SegTrackv2 masks for {image_path}")
        refs.append(options)
    return image_paths, refs


def preprocess_image_and_mask_ref(
    image_path: Path,
    mask_ref: MaskRef,
    image_size: int,
    patch_size: int,
    square_ok: bool,
    preprocess_mode: str = "native",
):
    if mask_ref is None:
        with Image.open(image_path) as image:
            width, height = image.size
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(tmp_path)
            return preprocess_image_and_mask(
                image_path=image_path,
                mask_path=tmp_path,
                image_size=image_size,
                patch_size=patch_size,
                square_ok=square_ok,
                preprocess_mode=preprocess_mode,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
    if isinstance(mask_ref, Path):
        if _is_fbms_mask_path(mask_ref):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                tmp_path = Path(handle.name)
            try:
                Image.fromarray(_fbms_mask_to_binary(mask_ref) * 255).save(tmp_path)
                return preprocess_image_and_mask(
                    image_path=image_path,
                    mask_path=tmp_path,
                    image_size=image_size,
                    patch_size=patch_size,
                    square_ok=square_ok,
                    preprocess_mode=preprocess_mode,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
        return preprocess_image_and_mask(
            image_path=image_path,
            mask_path=mask_ref,
            image_size=image_size,
            patch_size=patch_size,
            square_ok=square_ok,
            preprocess_mode=preprocess_mode,
        )

    merged = None
    for mask_path in mask_ref:
        mask_arr = np.array(Image.open(mask_path), dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = np.any(mask_arr > 0, axis=-1)
        merged = mask_arr if merged is None else np.maximum(merged, mask_arr)
    if merged is None:
        raise FileNotFoundError(f"No masks found for {image_path}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        Image.fromarray((merged > 0).astype(np.uint8) * 255).save(tmp_path)
        return preprocess_image_and_mask(
            image_path=image_path,
            mask_path=tmp_path,
            image_size=image_size,
            patch_size=patch_size,
            square_ok=square_ok,
            preprocess_mode=preprocess_mode,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def build_export_splits(
    youtubevos_root: Path,
    youtubevos_train_split: str,
    youtubevos_val_split: str,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
) -> Dict[str, List[Dict[str, str]]]:
    train_sequences = [
        {"dataset": "youtubevos", "split_name": youtubevos_train_split, "sequence": seq}
        for seq in list_youtubevos_sequences(youtubevos_root, youtubevos_train_split)
    ]
    val_sequences = [
        {"dataset": "youtubevos", "split_name": youtubevos_val_split, "sequence": seq}
        for seq in list_youtubevos_sequences(youtubevos_root, youtubevos_val_split)
    ]
    test_davis = (
        [{"dataset": "davis", "split_name": "train", "sequence": seq} for seq in read_split_sequences(davis_root, "train")]
        + [{"dataset": "davis", "split_name": "val", "sequence": seq} for seq in read_split_sequences(davis_root, "val")]
    )
    test_segtrackv2 = [
        {"dataset": "segtrackv2", "split_name": "all", "sequence": seq}
        for seq in list_segtrack_sequences(segtrack_root)
    ]
    return {
        "train": train_sequences,
        "val": val_sequences,
        "test_davis": test_davis,
        "test_segtrackv2": test_segtrackv2,
        "test_fbms59": [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
        "test": test_davis + test_segtrackv2 + [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
    }


def load_sequence_paths_from_spec(
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    spec: Dict[str, str],
    fbms_infer_all_frames: bool = False,
):
    dataset = spec["dataset"]
    split_name = spec["split_name"]
    sequence = spec["sequence"]
    if dataset == "youtubevos":
        return load_youtubevos_sequence_paths(youtubevos_root, split_name, sequence)
    if dataset == "davis":
        return collect_davis_sequence_paths(davis_root, sequence)
    if dataset == "segtrackv2":
        return load_segtrack_sequence_paths(segtrack_root, sequence)
    if dataset == "fbms59":
        return load_fbms_sequence_paths(
            fbms_root,
            split_name,
            sequence,
            include_unannotated=fbms_infer_all_frames,
        )
    raise ValueError(f"Unsupported dataset spec: {dataset}")


def collect_davis_sequence_paths(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def _pair_indices(num_frames: int, pair_stride: int) -> List[Tuple[int, int]]:
    pairs = []
    for idx in range(num_frames):
        ref_idx = idx + pair_stride
        if ref_idx >= num_frames:
            ref_idx = max(0, idx - pair_stride)
        pairs.append((idx, ref_idx))
    return pairs


def _expected_part_count(num_frames: int, pairs_per_forward: int) -> int:
    if num_frames <= 0:
        return 0
    if pairs_per_forward <= 0:
        raise ValueError("pairs_per_forward must be > 0")
    return (num_frames + pairs_per_forward - 1) // pairs_per_forward


def _inspect_existing_sequence_shards(
    shard_split_root: Path,
    sequence_name: str,
    feature_keys: List[str],
    expected_parts: int,
) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    seq_root = shard_split_root / sequence_name
    if not seq_root.exists():
        return None

    if expected_parts <= 0:
        return None

    part_counts: Dict[str, int] = {}
    first_key_parts: Optional[List[Path]] = None
    for key in feature_keys:
        key_dir = seq_root / key
        if not key_dir.exists():
            return None
        part_files = sorted(key_dir.glob("part_*.npz"))
        if len(part_files) != expected_parts:
            return None
        part_counts[key] = len(part_files)
        if part_files:
            with np.load(part_files[0]) as payload:
                required = {"tokens", "is_dynamic", "valid_mask", "group_ids"}
                if not required.issubset(payload.files):
                    return None
                n_tokens = int(payload["tokens"].shape[0])
                n_labels = int(payload["is_dynamic"].shape[0])
                n_valid = int(payload["valid_mask"].shape[0])
                n_groups = int(payload["group_ids"].shape[0])
                if not (n_tokens == n_labels == n_valid == n_groups):
                    return None
        if first_key_parts is None:
            first_key_parts = part_files

    if not first_key_parts:
        return None

    total_samples = 0
    for part_path in first_key_parts:
        with np.load(part_path) as payload:
            required = {"tokens", "is_dynamic", "valid_mask", "group_ids"}
            if not required.issubset(payload.files):
                return None
            n_tokens = int(payload["tokens"].shape[0])
            n_labels = int(payload["is_dynamic"].shape[0])
            n_valid = int(payload["valid_mask"].shape[0])
            n_groups = int(payload["group_ids"].shape[0])
            if not (n_tokens == n_labels == n_valid == n_groups):
                return None
            total_samples += n_tokens
    sample_counts = {key: total_samples for key in feature_keys}
    return part_counts, sample_counts


def _sample_indices_and_append_meta(
    labels_parts: List[np.ndarray],
    valid_parts: List[np.ndarray],
    groups_parts: List[np.ndarray],
    labels_flat: torch.Tensor,
    valid_flat: torch.Tensor,
    tokens_per_frame: int,
    rng: np.random.Generator,
    group_counter: int,
) -> Tuple[np.ndarray, int]:
    chosen = sample_frame_indices(valid_flat, tokens_per_frame=tokens_per_frame, rng=rng)
    if chosen.size > 0:
        labels_parts.append(labels_flat[chosen].cpu().numpy().astype(np.int64))
        valid_parts.append(valid_flat[chosen].cpu().numpy().astype(np.bool_))
        groups_parts.append(np.full((chosen.shape[0],), group_counter, dtype=np.int64))
    return chosen, group_counter + 1


def _append_feature_samples(
    feature_parts: Dict[str, List[np.ndarray]],
    feature_key: str,
    token_tensor: torch.Tensor,
    chosen: np.ndarray,
) -> None:
    if chosen.size == 0:
        return
    feature_parts[feature_key].append(token_tensor[chosen].detach().cpu().numpy().astype(np.float32))


def collect_split_samples(
    model,
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    split_name: str,
    sequence_specs: List[Dict[str, str]],
    token_sources: List[str],
    feature_key_order: List[str],
    image_size: int,
    pair_stride: int,
    pairs_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    fbms_infer_all_frames: bool,
    preprocess_mode: str,
    device: torch.device,
    seed: int,
    sink,
) -> None:
    rng = np.random.default_rng(seed)
    patch_size = int(model.patch_size)
    square_ok = bool(getattr(model, "square_ok", False))
    group_counter = 0

    for seq_idx, spec in enumerate(sequence_specs):
        sequence = spec["sequence"]
        image_paths, mask_paths = load_sequence_paths_from_spec(
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            spec=spec,
            fbms_infer_all_frames=fbms_infer_all_frames,
        )
        if isinstance(sink, ShardedDirSink):
            expected_parts = _expected_part_count(len(image_paths), pairs_per_forward=pairs_per_forward)
            existing_stats = _inspect_existing_sequence_shards(
                shard_split_root=sink.root_dir,
                sequence_name=sequence,
                feature_keys=feature_key_order,
                expected_parts=expected_parts,
            )
            if existing_stats is not None:
                part_counts, sample_counts = existing_stats
                sink.part_counts_by_seq[sequence] = part_counts
                sink.sample_counts_by_seq[sequence] = sample_counts
                print(
                    f"[{split_name}] skipped sequence {seq_idx + 1}/{len(sequence_specs)} "
                    f"(tokens already generated): {spec['dataset']}:{spec['split_name']}/{sequence}"
                )
                continue

        views = []
        masks = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            view, mask = preprocess_image_and_mask_ref(
                image_path=image_path,
                mask_ref=mask_path,
                image_size=image_size,
                patch_size=patch_size,
                square_ok=square_ok,
                preprocess_mode=preprocess_mode,
            )
            views.append(view)
            masks.append(mask)

        mask_batch = torch.stack(masks, dim=0).unsqueeze(0)
        ratios = mask_to_token_dynamic_ratio(mask_batch, patch_size=patch_size).squeeze(0)
        labels = dynamic_ratio_to_binary_labels(ratios, threshold=threshold)[0].reshape(len(views), -1).cpu()
        valid = torch.ones_like(labels, dtype=torch.bool)
        if spec["dataset"] == "fbms59" and fbms_infer_all_frames:
            annotated_flags = torch.tensor([mask_ref is not None for mask_ref in mask_paths], dtype=torch.bool)
            valid[~annotated_flags] = False

        pairs = _pair_indices(len(views), pair_stride=pair_stride)
        for start in range(0, len(pairs), pairs_per_forward):
            pair_chunk = pairs[start : start + pairs_per_forward]
            batch_view1 = collate_views([views[idx_i] for idx_i, _ in pair_chunk])
            batch_view2 = collate_views([views[idx_j] for _, idx_j in pair_chunk])
            batch_view1 = {key: value.to(device) for key, value in batch_view1.items()}
            batch_view2 = {key: value.to(device) for key, value in batch_view2.items()}

            with torch.no_grad():
                feature_map = extract_feature_map(
                    model=model,
                    view1=batch_view1,
                    view2=batch_view2,
                    token_sources=token_sources,
                )

            feature_parts: Dict[str, List[np.ndarray]] = {key: [] for key in feature_key_order}
            labels_parts: List[np.ndarray] = []
            valid_parts: List[np.ndarray] = []
            groups_parts: List[np.ndarray] = []

            for batch_idx, (frame_idx_i, frame_idx_j) in enumerate(pair_chunk):
                labels_i = labels[frame_idx_i]
                valid_i = valid[frame_idx_i]
                chosen_i, group_counter = _sample_indices_and_append_meta(
                    labels_parts,
                    valid_parts,
                    groups_parts,
                    labels_i,
                    valid_i,
                    tokens_per_frame,
                    rng,
                    group_counter,
                )

                for feature_key in feature_key_order:
                    feat_value = feature_map[feature_key]
                    if isinstance(feat_value, tuple):
                        token_tensor = 0.5 * (feat_value[0][batch_idx] + feat_value[1][batch_idx])
                    elif feature_key.startswith("dec_view1_") or feature_key.startswith("head_in_view1_"):
                        token_tensor = feat_value[batch_idx]
                    elif feature_key.startswith("dec_view2_") or feature_key.startswith("head_in_view2_"):
                        token_tensor = feat_value[batch_idx]
                    else:
                        raise ValueError(f"Unsupported feature key '{feature_key}'")
                    _append_feature_samples(feature_parts, feature_key, token_tensor, chosen_i)

            if labels_parts:
                labels_chunk = np.concatenate(labels_parts, axis=0)
                valid_chunk = np.concatenate(valid_parts, axis=0)
                groups_chunk = np.concatenate(groups_parts, axis=0)
                feature_chunk = {}
                for key in feature_key_order:
                    if not feature_parts[key]:
                        raise RuntimeError(f"Missing token samples for feature key '{key}'")
                    merged = np.concatenate(feature_parts[key], axis=0)
                    if merged.shape[0] != labels_chunk.shape[0]:
                        raise RuntimeError(
                            f"Feature/label sample mismatch for '{key}': "
                            f"tokens={merged.shape[0]} labels={labels_chunk.shape[0]}"
                        )
                    feature_chunk[key] = merged
                sink.add_batch(feature_chunk, labels_chunk, valid_chunk, groups_chunk, sequence_name=sequence)

        print(f"[{split_name}] processed sequence {seq_idx + 1}/{len(sequence_specs)}: {spec['dataset']}:{spec['split_name']}/{sequence}")


def main() -> None:
    args = parse_args()
    youtubevos_root = resolve_youtubevos_root(args.youtubevos_root)
    davis_root = resolve_davis_root(args.davis_root)
    segtrack_root = resolve_segtrack_root(args.segtrack_root)
    fbms_root = resolve_fbms_root(args.fbms_root)
    token_sources = _parse_token_sources(args.token_sources)
    device = torch.device(args.device)
    weights_or_name = args.weights if args.weights is not None else f"naver/{args.model_name}"

    model = load_dust3r_model(weights_or_name=weights_or_name, device=device)
    feature_key_order = build_feature_key_order(model=model, token_sources=token_sources)

    dataset_name = args.dataset_name.strip().lower()
    output_dir = Path(args.output_dir) / dataset_name if dataset_name else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_plan = build_export_splits(
        youtubevos_root=youtubevos_root,
        youtubevos_train_split=args.youtubevos_train_split,
        youtubevos_val_split=args.youtubevos_val_split,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
    )
    export_splits = ["train", "val", "test_davis", "test_segtrackv2", "test_fbms59"] if args.split == "all" else [args.split]

    print(f"Output directory: {output_dir}")
    print(f"Weights: {weights_or_name}")
    print(f"YouTube-VOS root: {youtubevos_root}")
    print(f"DAVIS root: {davis_root}")
    print(f"SegTrackv2 root: {segtrack_root}")
    print(f"FBMS-59 root: {fbms_root}")
    print(f"Token sources: {token_sources}")
    print(f"Feature key order: {feature_key_order[:6]} ... total={len(feature_key_order)}")

    for split_idx, split_name in enumerate(export_splits):
        sequence_specs = split_plan[split_name]
        if not sequence_specs:
            raise RuntimeError(f"Split '{split_name}' resolved to 0 sequences.")

        sink = (
            ShardedDirSink(output_dir / args.tokens_subdir / split_name, feature_key_order)
            if args.save_mode == "sharded_dirs"
            else SingleNpzSink(feature_key_order)
        )

        collect_split_samples(
            model=model,
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            split_name=split_name,
            sequence_specs=sequence_specs,
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            image_size=args.image_size,
            pair_stride=args.pair_stride,
            pairs_per_forward=args.pairs_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            fbms_infer_all_frames=args.fbms_infer_all_frames,
            preprocess_mode=args.preprocess_mode,
            device=device,
            seed=args.seed + split_idx,
            sink=sink,
        )

        meta = {
            "split": split_name,
            "weights_or_name": weights_or_name,
            "split_policy": {
                "train": f"youtubevos:{args.youtubevos_train_split}",
                "val": f"youtubevos:{args.youtubevos_val_split}",
                "test_davis": "davis:official_train_plus_val",
                "test_segtrackv2": "segtrackv2:all",
                "test_fbms59": "fbms59:trainingset_plus_testset",
                "test": "combined:davis_plus_segtrackv2_plus_fbms59",
            },
            "sequence_count": len(sequence_specs),
            "sequence_specs": sequence_specs,
            "token_sources": token_sources,
            "feature_key_order": feature_key_order,
            "image_size": args.image_size,
            "patch_size": int(model.patch_size),
            "pair_stride": args.pair_stride,
            "pair_policy": "adjacent_reference",
            "pairs_per_forward": args.pairs_per_forward,
            "tokens_per_frame": args.tokens_per_frame,
            "threshold": args.threshold,
            "fbms_infer_all_frames": bool(args.fbms_infer_all_frames),
            "preprocess_mode": args.preprocess_mode,
            "save_mode": args.save_mode,
        }

        if args.save_mode == "sharded_dirs":
            sink.finalize(meta=meta)
            print(f"Saved sharded token dirs for split='{split_name}' under: {output_dir / args.tokens_subdir / split_name}")
        else:
            out_path = output_dir / f"{split_name}_tokens.npz"
            sink.finalize(output_path=out_path, meta=meta)
            print(f"Saved single NPZ for split='{split_name}': {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
