#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DA3_ROOT = PROJECT_ROOT / "models" / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"
if str(DA3_ROOT) not in sys.path:
    sys.path.insert(0, str(DA3_ROOT))
if str(DA3_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(DA3_SRC_ROOT))

from depth_anything_3.api import DepthAnything3
from isdynamic.legacy.depth_anything_3.build_davis_isdynamic_npz import (  # noqa: E402
    PATCH_SIZE,
    ShardedDirSink,
    SingleNpzSink,
    _parse_token_sources,
    _resolve_anyview_model,
    _run_anyview_backbone,
    build_feature_key_order,
    dynamic_ratio_to_binary_labels,
    extract_feature_map,
    mask_to_token_dynamic_ratio,
    preprocess_image_and_mask,
    read_split_sequences,
    resolve_davis_root,
    sample_frame_indices,
)
from scripts.token_defaults import token_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from YouTube-VOS motion splits for DA3, "
            "using full DAVIS official train+val as the test split."
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
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="HF repo id or local directory passed to DepthAnything3.from_pretrained.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder for exports.")
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
        default="alt_attn,depth_head",
        help=(
            "Comma-separated source groups. Choices: alt_attn,depth_head. "
            "alt_attn expands to inter_frame_block_i and intra_frame_block_i."
        ),
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="sharded_dirs",
        choices=["sharded_dirs", "single_npz"],
        help=(
            "sharded_dirs: write per-feature directories. "
            "single_npz: write one tokens.npz with many keys."
        ),
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
        help="Number of frames per forward chunk.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Dynamic ratio threshold for token labels.")
    parser.add_argument(
        "--process_res",
        type=int,
        default=518,
        help="Target resize resolution matching DA3 preprocessing.",
    )
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
        help="Resize method matching DA3 InputProcessor.",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="native",
        choices=["native", "aspect_pad"],
        help="native keeps DA3 preprocessing; aspect_pad uses Pi3-style resize-and-pad.",
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

    required_splits = ("train_motion", "valid_motion")
    for candidate in candidates:
        if all((candidate / split_name).exists() for split_name in required_splits):
            return candidate

    checked = "\n".join(str(p) for p in candidates)
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
        raise FileNotFoundError(f"YouTube-VOS split not found: {split_root}")
    image_root, ann_root = _resolve_split_dirs(split_root)
    sequences = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    if not sequences:
        raise RuntimeError(f"No sequence directories found under {image_root}")

    missing_ann = [seq for seq in sequences if not (ann_root / seq).exists()]
    if missing_ann:
        raise FileNotFoundError(
            f"Missing annotation directories for {len(missing_ann)} sequences under {ann_root}. "
            f"Examples: {missing_ann[:10]}"
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

    davis_train = read_split_sequences(davis_root, "train")
    davis_val = read_split_sequences(davis_root, "val")
    test_davis_sequences = (
        [{"dataset": "davis", "split_name": "train", "sequence": seq} for seq in davis_train]
        + [{"dataset": "davis", "split_name": "val", "sequence": seq} for seq in davis_val]
    )
    test_segtrack_sequences = [
        {"dataset": "segtrackv2", "split_name": "all", "sequence": seq}
        for seq in list_segtrack_sequences(segtrack_root)
    ]
    return {
        "train": train_sequences,
        "val": val_sequences,
        "test_davis": test_davis_sequences,
        "test_segtrackv2": test_segtrack_sequences,
        "test_fbms59": [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
        "test": test_davis_sequences + test_segtrack_sequences + [
            {"dataset": "fbms59", "split_name": split_name, "sequence": sequence}
            for split_name, sequence in list_fbms_sequences(fbms_root)
        ],
    }


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
) -> Tuple[List[Path], List[Union[Path, None]]]:
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

def _list_sequence_images(image_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        image_paths.extend(image_dir.glob(pattern))
    image_paths = sorted(set(image_paths))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


MaskRef = Union[Path, List[Path], None]


def _is_fbms_mask_path(mask_path: Path) -> bool:
    return mask_path.parent.name == "GroundTruth" and any(part in {"Trainingset", "Testset"} for part in mask_path.parts)


def _encode_fbms_mask_values(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        rgb = mask_arr.astype(np.uint32)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return mask_arr.astype(np.uint32)


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
    sequence = sequence.lstrip("*").strip()
    image_dir = segtrack_root / "JPEGImages" / sequence
    gt_dir = segtrack_root / "GroundTruth" / sequence
    image_paths = _list_sequence_images(image_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")

    direct_mask_paths: List[Path] = []
    for pattern in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
        direct_mask_paths.extend(sorted(gt_dir.glob(pattern)))
    direct_masks = {path.stem: [path] for path in sorted(set(direct_mask_paths))}
    if direct_masks:
        mask_paths: List[MaskRef] = []
        for image_path in image_paths:
            refs = direct_masks.get(image_path.stem)
            if refs is None:
                raise FileNotFoundError(f"Missing mask for {image_path}")
            mask_paths.append(refs[0])
        return image_paths, mask_paths

    instance_dirs = sorted(path for path in gt_dir.iterdir() if path.is_dir())
    if not instance_dirs:
        raise RuntimeError(f"No direct masks or instance directories found under {gt_dir}")

    mask_paths: List[MaskRef] = []
    for image_path in image_paths:
        refs = [instance_dir / f"{image_path.stem}.png" for instance_dir in instance_dirs]
        refs = [path for path in refs if path.exists()]
        if not refs:
            raise FileNotFoundError(f"Missing SegTrackv2 masks for {image_path}")
        mask_paths.append(refs)
    return image_paths, mask_paths




def preprocess_image_and_mask_ref(image_path: Path, mask_ref: MaskRef, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask_ref is None:
        with Image.open(image_path) as image:
            width, height = image.size
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(tmp_path)
            return preprocess_image_and_mask(image_path, tmp_path, **kwargs)
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
                return preprocess_image_and_mask(image_path, tmp_path, **kwargs)
            finally:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
        return preprocess_image_and_mask(image_path, mask_ref, **kwargs)

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
        return preprocess_image_and_mask(image_path, tmp_path, **kwargs)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def load_sequence_paths_from_spec(
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    spec: Dict[str, str],
    fbms_infer_all_frames: bool = False,
) -> Tuple[List[Path], List[MaskRef]]:
    dataset = spec["dataset"]
    sequence = spec["sequence"]
    split_name = spec["split_name"]
    if dataset == "youtubevos":
        return load_youtubevos_sequence_paths(youtubevos_root, split_name, sequence)
    if dataset == "davis":
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


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def collect_split_samples(
    anyview_model: torch.nn.Module,
    youtubevos_root: Path,
    davis_root: Path,
    segtrack_root: Path,
    fbms_root: Path,
    split_name: str,
    sequence_specs: List[Dict[str, str]],
    token_sources: List[str],
    feature_key_order: List[str],
    export_feat_layers: List[int],
    frames_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    fbms_infer_all_frames: bool,
    process_res: int,
    process_res_method: str,
    preprocess_mode: str,
    device: torch.device,
    seed: int,
    group_start: int,
    sink,
) -> int:
    rng = np.random.default_rng(seed)
    group_counter = group_start

    for seq_idx, spec in enumerate(sequence_specs):
        sequence = spec["sequence"]
        if isinstance(sink, ShardedDirSink) and sink.sequence_complete(sequence):
            source_tag = f"{spec['dataset']}:{spec['split_name']}"
            print(f"[{split_name}] skipped existing sequence {seq_idx + 1}/{len(sequence_specs)}: {source_tag}/{sequence}")
            continue
        image_paths, mask_paths = load_sequence_paths_from_spec(
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            spec=spec,
            fbms_infer_all_frames=fbms_infer_all_frames,
        )
        chunk_size = max(1, int(frames_per_forward))
        start = 0
        while start < len(image_paths):
            end = min(start + chunk_size, len(image_paths))
            chunk_image_paths = image_paths[start:end]
            chunk_mask_paths = mask_paths[start:end]

            try:
                image_list = []
                mask_list = []
                for img_path, mask_path in zip(chunk_image_paths, chunk_mask_paths):
                    img_t, mask_t = preprocess_image_and_mask_ref(
                        img_path,
                        mask_path,
                        process_res=process_res,
                        process_res_method=process_res_method,
                        preprocess_mode=preprocess_mode,
                    )
                    image_list.append(img_t)
                    mask_list.append(mask_t)

                images = torch.stack(image_list, dim=0)
                masks = torch.stack(mask_list, dim=0)

                feats, aux_feats = _run_anyview_backbone(
                    anyview_model,
                    images.unsqueeze(0).to(device=device, dtype=torch.float32),
                    export_feat_layers=export_feat_layers,
                )
                feature_map = extract_feature_map(
                    anyview_model=anyview_model,
                    feats=feats,
                    aux_feats=aux_feats,
                    export_feat_layers=export_feat_layers,
                    token_sources=token_sources,
                )

                ratios = mask_to_token_dynamic_ratio(masks.unsqueeze(0), patch_size=PATCH_SIZE).squeeze(0)
                labels, valid = dynamic_ratio_to_binary_labels(ratios, threshold=threshold)
                labels = labels.reshape(labels.shape[0], -1).cpu()
                valid = valid.reshape(valid.shape[0], -1).cpu()
                if spec["dataset"] == "fbms59" and fbms_infer_all_frames:
                    annotated_flags = torch.tensor(
                        [mask_ref is not None for mask_ref in chunk_mask_paths],
                        dtype=torch.bool,
                    )
                    valid[~annotated_flags] = False

                for key in feature_key_order:
                    if key not in feature_map:
                        raise KeyError(f"Missing expected feature key '{key}' in extracted feature_map")
                    feats_k = feature_map[key]
                    if feats_k.shape[0] != 1 or feats_k.shape[1] != labels.shape[0] or feats_k.shape[2] != labels.shape[1]:
                        raise ValueError(
                            f"Feature/label mismatch for key={key} in {sequence} [{start}:{end}] "
                            f"features={tuple(feats_k.shape)} labels={tuple(labels.shape)}"
                        )

                feature_parts: Dict[str, List[np.ndarray]] = {key: [] for key in feature_key_order}
                labels_parts: List[np.ndarray] = []
                valid_parts: List[np.ndarray] = []
                groups_parts: List[np.ndarray] = []

                for frame_idx in range(labels.shape[0]):
                    chosen = sample_frame_indices(valid[frame_idx], tokens_per_frame=tokens_per_frame, rng=rng)
                    if chosen.size == 0:
                        group_counter += 1
                        continue

                    labels_parts.append(labels[frame_idx, chosen].cpu().numpy().astype(np.int64))
                    valid_parts.append(valid[frame_idx, chosen].cpu().numpy().astype(np.bool_))
                    groups_parts.append(np.full((chosen.shape[0],), group_counter, dtype=np.int64))

                    for key in feature_key_order:
                        x = feature_map[key][0, frame_idx, chosen].detach().cpu().float().numpy().astype(np.float32)
                        feature_parts[key].append(x)

                    group_counter += 1

                if labels_parts:
                    labels_chunk = np.concatenate(labels_parts, axis=0)
                    valid_chunk = np.concatenate(valid_parts, axis=0)
                    groups_chunk = np.concatenate(groups_parts, axis=0)
                    feature_chunk = {
                        key: np.concatenate(feature_parts[key], axis=0) for key in feature_key_order
                    }
                    sink.add_batch(feature_chunk, labels_chunk, valid_chunk, groups_chunk, sequence_name=sequence)
                start = end
            except RuntimeError as exc:
                if device.type == "cuda" and _is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if chunk_size <= 1:
                        raise RuntimeError(
                            f"CUDA OOM for {spec['dataset']}:{spec['split_name']}/{sequence} even with "
                            "frames_per_forward=1. Try lowering --process_res or freeing GPU memory."
                        ) from exc
                    new_chunk_size = max(1, chunk_size // 2)
                    print(
                        f"[warn] CUDA OOM at {spec['dataset']}:{spec['split_name']}/{sequence} "
                        f"frames[{start}:{end}] chunk_size={chunk_size}. Retrying with chunk_size={new_chunk_size}."
                    )
                    chunk_size = new_chunk_size
                    continue
                raise

        source_tag = f"{spec['dataset']}:{spec['split_name']}"
        print(f"[{split_name}] processed sequence {seq_idx + 1}/{len(sequence_specs)}: {source_tag}/{sequence}")

    return group_counter


def main() -> None:
    args = parse_args()
    youtubevos_root = resolve_youtubevos_root(args.youtubevos_root)
    davis_root = resolve_davis_root(args.davis_root)
    segtrack_root = resolve_segtrack_root(args.segtrack_root)
    fbms_root = resolve_fbms_root(args.fbms_root)
    token_sources = _parse_token_sources(args.token_sources)

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)
    model.eval()

    anyview_model = _resolve_anyview_model(model.model)
    export_feat_layers = list(range(len(anyview_model.backbone.pretrained.blocks)))
    feature_key_order, block_groups, depth_head_layers = build_feature_key_order(
        anyview_model=anyview_model,
        token_sources=token_sources,
    )

    tokens_root = args.output_dir if args.output_dir is not None else token_path("data", "isdynamic")
    dataset_name = args.dataset_name.strip().lower()
    output_dir = Path(tokens_root) / dataset_name if dataset_name else Path(tokens_root)
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
    print(f"YouTube-VOS root: {youtubevos_root}")
    print(f"DAVIS root: {davis_root}")
    print(f"Token sources: {token_sources}")
    print(f"Feature key order: {feature_key_order[:6]} ... total={len(feature_key_order)}")
    print(f"Save mode: {args.save_mode}")
    print(
        "Actual block mapping: "
        f"inter_frame_block={block_groups['inter_frame_block'][:6]}... ({len(block_groups['inter_frame_block'])}), "
        f"intra_frame_block={block_groups['intra_frame_block'][:6]}... ({len(block_groups['intra_frame_block'])}), "
        f"depth_head_layers={depth_head_layers}"
    )
    for split_name in export_splits:
        print(f"Split '{split_name}' sequences: {len(split_plan[split_name])}")

    for split_idx, split_name in enumerate(export_splits):
        sequence_specs = split_plan[split_name]
        if not sequence_specs:
            raise RuntimeError(f"Split '{split_name}' resolved to 0 sequences.")

        if args.save_mode == "sharded_dirs":
            sink = ShardedDirSink(output_dir / args.tokens_subdir / split_name, feature_key_order)
        else:
            sink = SingleNpzSink(feature_key_order)

        collect_split_samples(
            anyview_model=anyview_model,
            youtubevos_root=youtubevos_root,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            split_name=split_name,
            sequence_specs=sequence_specs,
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            export_feat_layers=export_feat_layers,
            frames_per_forward=args.frames_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            fbms_infer_all_frames=args.fbms_infer_all_frames,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            preprocess_mode=args.preprocess_mode,
            device=device,
            seed=args.seed + split_idx,
            group_start=0,
            sink=sink,
        )

        meta = {
            "model_dir": args.model_dir,
            "model_name": getattr(model, "model_name", None),
            "branch": "anyview",
            "split": split_name,
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
            "feature_groups": {
                "inter_frame_block_actual_blocks": block_groups["inter_frame_block"],
                "intra_frame_block_actual_blocks": block_groups["intra_frame_block"],
                "depth_head_source_layers": depth_head_layers,
            },
            "frames_per_forward": args.frames_per_forward,
            "tokens_per_frame": args.tokens_per_frame,
            "threshold": args.threshold,
            "patch_size": PATCH_SIZE,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "preprocess_mode": args.preprocess_mode,
            "save_mode": args.save_mode,
        }

        if args.save_mode == "sharded_dirs":
            sink.finalize(meta=meta)
            print(
                f"Saved sharded token dirs for split='{split_name}' under: "
                f"{output_dir / args.tokens_subdir / split_name}"
            )
        else:
            out_path = output_dir / f"{split_name}_tokens.npz"
            sink.finalize(output_path=out_path, meta=meta)
            print(f"Saved single NPZ for split='{split_name}': {out_path}")


if __name__ == "__main__":
    main()
