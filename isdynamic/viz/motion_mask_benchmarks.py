from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image


MaskRef = Union[Path, List[Path], None]

DEFAULT_DAVIS_ROOT = Path(__file__).resolve().parents[2] / "datasets/DAVIS"
DEFAULT_SEGTRACK_ROOT = Path(__file__).resolve().parents[2] / "datasets/SegTrackv2"
DEFAULT_FBMS_ROOT = Path(__file__).resolve().parents[2] / "datasets/FBMS-59"

DAVIS2016_VAL_FILE = DEFAULT_DAVIS_ROOT / "ImageSets" / "2016" / "val.txt"
DAVIS2017_TRAIN_FILE = DEFAULT_DAVIS_ROOT / "ImageSets" / "2017" / "train.txt"
DAVIS2017_VAL_FILE = DEFAULT_DAVIS_ROOT / "ImageSets" / "2017" / "val.txt"

BENCHMARK_CHOICES = ["davis2016", "davis2017", "davis_all", "segtrackv2", "fbms59"]


@dataclass(frozen=True)
class SequenceSpec:
    dataset: str
    benchmark: str
    sequence: str
    sequence_tag: str
    fbms_split: str | None = None


def normalize_benchmark_name(name: str) -> str:
    text = name.strip().lower().replace("-", "_")
    aliases = {
        "davis2016": "davis2016",
        "davis_2016": "davis2016",
        "davis16": "davis2016",
        "davis2017": "davis2017",
        "davis_2017": "davis2017",
        "davis17": "davis2017",
        "davis_all": "davis_all",
        "davisall": "davis_all",
        "segtrackv2": "segtrackv2",
        "segtrack_v2": "segtrackv2",
        "fbms59": "fbms59",
        "fbms_59": "fbms59",
        "fbms": "fbms59",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported benchmark: {name}")
    return aliases[text]


def resolve_davis_root(path: str | Path | None) -> Path:
    root = Path(path) if path else DEFAULT_DAVIS_ROOT
    if (root / "DAVIS").exists():
        root = root / "DAVIS"
    required = [
        root / "ImageSets" / "2016",
        root / "ImageSets" / "2017",
        root / "JPEGImages" / "480p",
        root / "Annotations" / "480p",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Invalid DAVIS root. Missing paths:\n" + "\n".join(missing))
    return root


def resolve_segtrack_root(path: str | Path | None) -> Path:
    root = Path(path) if path else DEFAULT_SEGTRACK_ROOT
    required = [root / "JPEGImages", root / "GroundTruth", root / "ImageSets"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Invalid SegTrackv2 root. Missing paths:\n" + "\n".join(missing))
    return root


def resolve_fbms_root(path: str | Path | None) -> Path:
    root = Path(path) if path else DEFAULT_FBMS_ROOT
    candidates = [root]
    for child_name in ("FBMS-59", "fbms-59", "FBMS59", "fbms59"):
        child = root / child_name
        if child.exists():
            candidates.append(child)
    for candidate in candidates:
        if (candidate / "Trainingset").exists() and (candidate / "Testset").exists():
            return candidate
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve FBMS-59 root with Trainingset/ and Testset/.\n"
        f"Checked:\n{checked}"
    )


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def benchmark_display_name(benchmark: str) -> str:
    names = {
        "davis2016": "DAVIS2016",
        "davis2017": "DAVIS2017",
        "davis_all": "DAVIS-all",
        "segtrackv2": "SegTrackv2",
        "fbms59": "FBMS-59",
    }
    return names[normalize_benchmark_name(benchmark)]


def benchmark_output_dirname(benchmark: str) -> str:
    names = {
        "davis2016": "davis2016",
        "davis2017": "davis2017",
        "davis_all": "davis_all",
        "segtrackv2": "segtrackv2",
        "fbms59": "fbms59",
    }
    return names[normalize_benchmark_name(benchmark)]


def list_segtrack_sequences(segtrack_root: Path) -> List[str]:
    all_file = segtrack_root / "ImageSets" / "all.txt"
    if all_file.exists():
        sequences = sorted(line.strip().lstrip("*").strip() for line in _read_nonempty_lines(all_file))
        if sequences:
            return sequences
    return sorted(path.name for path in (segtrack_root / "JPEGImages").iterdir() if path.is_dir())


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


def list_sequence_specs(
    benchmark: str,
    *,
    davis_root: str | Path = DEFAULT_DAVIS_ROOT,
    segtrack_root: str | Path = DEFAULT_SEGTRACK_ROOT,
    fbms_root: str | Path = DEFAULT_FBMS_ROOT,
) -> List[SequenceSpec]:
    benchmark = normalize_benchmark_name(benchmark)
    if benchmark == "davis2016":
        davis_root = resolve_davis_root(davis_root)
        return [
            SequenceSpec(dataset="davis", benchmark=benchmark, sequence=seq, sequence_tag=seq)
            for seq in _read_nonempty_lines(davis_root / "ImageSets" / "2016" / "val.txt")
        ]
    if benchmark == "davis2017":
        davis_root = resolve_davis_root(davis_root)
        return [
            SequenceSpec(dataset="davis", benchmark=benchmark, sequence=seq, sequence_tag=seq)
            for seq in _read_nonempty_lines(davis_root / "ImageSets" / "2017" / "val.txt")
        ]
    if benchmark == "davis_all":
        davis_root = resolve_davis_root(davis_root)
        sequences = _read_nonempty_lines(davis_root / "ImageSets" / "2017" / "train.txt") + _read_nonempty_lines(
            davis_root / "ImageSets" / "2017" / "val.txt"
        )
        return [
            SequenceSpec(dataset="davis", benchmark=benchmark, sequence=seq, sequence_tag=seq)
            for seq in sequences
        ]
    if benchmark == "segtrackv2":
        segtrack_root = resolve_segtrack_root(segtrack_root)
        return [
            SequenceSpec(dataset="segtrackv2", benchmark=benchmark, sequence=seq, sequence_tag=seq)
            for seq in list_segtrack_sequences(segtrack_root)
        ]
    fbms_root = resolve_fbms_root(fbms_root)
    return [
        SequenceSpec(
            dataset="fbms59",
            benchmark=benchmark,
            sequence=sequence,
            sequence_tag=f"{split_name}__{sequence}",
            fbms_split=split_name,
        )
        for split_name, sequence in list_fbms_sequences(fbms_root)
    ]


def load_davis_sequence_paths(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No DAVIS images found in {image_dir}")
    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DAVIS masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def _list_sequence_images(image_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        image_paths.extend(image_dir.glob(pattern))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


def _normalize_fbms_stem(stem: str) -> str:
    normalized = stem.strip().lower()
    if normalized.endswith("_gt"):
        normalized = normalized[:-3]
    return normalized


def _parse_fbms_definition(gt_dir: Path) -> Tuple[int, dict[str, Path]]:
    dat_paths = sorted(path for path in gt_dir.glob("*Def.dat") if path.is_file())
    if not dat_paths:
        dat_paths = sorted(path for path in gt_dir.glob("*.dat") if path.is_file())
    if not dat_paths:
        raise FileNotFoundError(f"No FBMS .dat definition file found in {gt_dir}")

    dat_path = dat_paths[0]
    background_scale = None
    masks_by_stem: dict[str, Path] = {}
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


def _choose_fbms_masks(gt_dir: Path) -> dict[str, Path]:
    try:
        _, masks_by_stem = _parse_fbms_definition(gt_dir)
        return masks_by_stem
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    mask_exts = {".pgm", ".ppm", ".bmp", ".jpg", ".jpeg", ".png"}
    suffix_priority = {".png": 0, ".pgm": 1, ".ppm": 2, ".bmp": 3, ".jpg": 4, ".jpeg": 5}
    raw_masks = sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in mask_exts
        and "prob" not in path.stem.lower()
    )
    if not raw_masks:
        raise FileNotFoundError(f"No FBMS official masks found in {gt_dir}")

    chosen: dict[str, tuple[tuple[int, str], Path]] = {}
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
    *,
    include_unannotated: bool = False,
) -> Tuple[List[Path], List[MaskRef]]:
    sequence_dir = fbms_root / split_name / sequence
    if not sequence_dir.exists():
        raise FileNotFoundError(f"FBMS sequence directory not found: {sequence_dir}")
    gt_dir = sequence_dir / "GroundTruth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"FBMS GroundTruth directory not found: {gt_dir}")

    image_paths = _list_sequence_images(sequence_dir)
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
        refs: List[Path] = []
        for instance_dir in instance_dirs:
            matches = []
            for pattern in (
                f"{image_path.stem}.png",
                f"{image_path.stem}.bmp",
                f"{image_path.stem}.jpg",
                f"{image_path.stem}.jpeg",
                f"{image_path.stem}.PNG",
                f"{image_path.stem}.BMP",
                f"{image_path.stem}.JPG",
                f"{image_path.stem}.JPEG",
            ):
                candidate = instance_dir / pattern
                if candidate.exists():
                    matches.append(candidate)
            if not matches:
                matches = [path for path in instance_dir.iterdir() if path.is_file() and path.stem == image_path.stem]
            refs.extend(sorted(matches))
        if not refs:
            raise FileNotFoundError(f"Missing SegTrackv2 masks for {image_path}")
        mask_paths.append(refs)
    return image_paths, mask_paths


def load_sequence_paths_from_spec(
    spec: SequenceSpec,
    *,
    davis_root: str | Path = DEFAULT_DAVIS_ROOT,
    segtrack_root: str | Path = DEFAULT_SEGTRACK_ROOT,
    fbms_root: str | Path = DEFAULT_FBMS_ROOT,
    fbms_include_unannotated: bool = False,
) -> Tuple[List[Path], List[MaskRef], str]:
    if spec.dataset == "davis":
        root = resolve_davis_root(davis_root)
        image_paths, mask_refs = load_davis_sequence_paths(root, spec.sequence)
        return image_paths, mask_refs, spec.sequence_tag
    if spec.dataset == "segtrackv2":
        root = resolve_segtrack_root(segtrack_root)
        image_paths, mask_refs = load_segtrack_sequence_paths(root, spec.sequence)
        return image_paths, mask_refs, spec.sequence_tag
    if spec.dataset == "fbms59":
        if spec.fbms_split is None:
            raise ValueError("FBMS-59 sequence spec requires fbms_split.")
        root = resolve_fbms_root(fbms_root)
        image_paths, mask_refs = load_fbms_sequence_paths(
            root,
            spec.fbms_split,
            spec.sequence,
            include_unannotated=fbms_include_unannotated,
        )
        return image_paths, mask_refs, spec.sequence_tag
    raise ValueError(f"Unsupported dataset in sequence spec: {spec.dataset}")


def select_context_indices(
    num_frames: int,
    target_idx: int,
    *,
    total_frames: int = 8,
    stride: int = 1,
) -> List[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not (0 <= target_idx < num_frames):
        raise IndexError(f"target_idx out of range: {target_idx} for {num_frames} frames")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    if total_frames == 1 or num_frames == 1:
        return [target_idx]

    want = min(total_frames, num_frames) - 1
    context: List[int] = []
    step = 1
    while len(context) < want:
        left = target_idx - step * stride
        right = target_idx + step * stride
        added = False
        if left >= 0:
            context.append(left)
            added = True
            if len(context) >= want:
                break
        if right < num_frames:
            context.append(right)
            added = True
            if len(context) >= want:
                break
        if not added:
            break
        step += 1
    return [target_idx, *sorted(context[:want])]


def _encode_fbms_mask_values(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        rgb = mask_arr.astype(np.uint32)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return mask_arr.astype(np.uint32)


def is_fbms_mask_path(mask_path: Path) -> bool:
    return mask_path.parent.name == "GroundTruth" and any(part in {"Trainingset", "Testset"} for part in mask_path.parts)


def _generic_mask_to_binary(mask_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_arr)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = arr.max(axis=-1)
        arr = arr.astype(np.float32, copy=False)
        if arr.size == 0:
            return np.zeros(arr.shape[:2], dtype=np.uint8)
        threshold = 0.5 if float(arr.max()) <= 1.0 else 127.0
        return (arr > threshold).astype(np.uint8)

    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    # Single-channel annotation masks in DAVIS/SegTrack commonly store object ids
    # as small positive integers (e.g. {0,1,2}), so foreground should be any
    # non-zero label rather than only values above 127.
    return (arr > 0).astype(np.uint8)


def load_binary_mask(mask_ref: MaskRef, *, fallback_size: Tuple[int, int] | None = None) -> np.ndarray:
    if mask_ref is None:
        if fallback_size is None:
            raise ValueError("fallback_size is required when mask_ref is None.")
        width, height = fallback_size
        return np.zeros((height, width), dtype=np.uint8)

    refs = [mask_ref] if isinstance(mask_ref, Path) else mask_ref
    union_mask = None
    for ref in refs:
        if is_fbms_mask_path(ref):
            background_scale, _ = _parse_fbms_definition(ref.parent)
            encoded = _encode_fbms_mask_values(np.array(Image.open(ref)))
            mask = (encoded != np.uint32(background_scale)).astype(np.uint8)
        else:
            mask = _generic_mask_to_binary(np.array(Image.open(ref)))
        union_mask = mask if union_mask is None else np.maximum(union_mask, mask)
    if union_mask is None:
        raise ValueError("mask_ref resolved to no masks.")
    return union_mask.astype(np.uint8)


def build_sequence_spec(
    *,
    dataset: str,
    benchmark: str | None,
    sequence: str,
    fbms_split: str | None = None,
) -> SequenceSpec:
    dataset = dataset.strip().lower()
    default_benchmark = {
        "davis": "davis2017",
        "segtrackv2": "segtrackv2",
        "fbms59": "fbms59",
    }
    benchmark_name = normalize_benchmark_name(benchmark or default_benchmark.get(dataset, dataset))
    if dataset == "davis":
        return SequenceSpec(dataset=dataset, benchmark=benchmark_name, sequence=sequence, sequence_tag=sequence)
    if dataset == "segtrackv2":
        return SequenceSpec(dataset=dataset, benchmark=benchmark_name, sequence=sequence, sequence_tag=sequence)
    if dataset == "fbms59":
        if not fbms_split:
            raise ValueError("fbms_split is required for dataset=fbms59.")
        return SequenceSpec(
            dataset=dataset,
            benchmark=benchmark_name,
            sequence=sequence,
            sequence_tag=f"{fbms_split}__{sequence}",
            fbms_split=fbms_split,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def format_frame_dir(frame_idx: int) -> str:
    return f"frame_{frame_idx:05d}"


def format_context_summary(indices: Sequence[int]) -> dict[str, object]:
    if not indices:
        return {"target_frame_idx": None, "context_frame_indices": []}
    return {
        "target_frame_idx": int(indices[0]),
        "context_frame_indices": [int(idx) for idx in indices[1:]],
    }


def sequence_tags_for_specs(specs: Iterable[SequenceSpec]) -> List[str]:
    return [spec.sequence_tag for spec in specs]
