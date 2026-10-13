#!/usr/bin/env python3
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DA3_ROOT = PROJECT_ROOT / "models" / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"
if str(DA3_ROOT) not in sys.path:
    sys.path.insert(0, str(DA3_ROOT))
if str(DA3_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(DA3_SRC_ROOT))

from depth_anything_3.api import DepthAnything3
from isdynamic.core.preprocess_aspect_pad import aspect_pad_arrays
from scripts.token_defaults import token_path


PATCH_SIZE = 14
SOURCE_CHOICES = ["alt_attn", "depth_head"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from DAVIS-2017 for DA3 "
            "(groups: intra_frame_block, inter_frame_block, depth_head)."
        )
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p",
        help="Path to DAVIS root or parent containing DAVIS/.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="HF repo id or local directory passed to DepthAnything3.from_pretrained.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder for exports.")
    parser.add_argument("--dataset_name", type=str, default="davis_da3", help="Dataset subfolder name.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=(
            "Which export split to write. "
            "Split policy: train=official train split; val/test are carved from official val split "
            "using --val_seq_count and --test_seq_count."
        ),
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

    ordered = []
    seen = set()
    for source in sources:
        if source not in seen:
            ordered.append(source)
            seen.add(source)
    return ordered


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
    if val_seq_count < 0 or test_seq_count < 0:
        raise ValueError("--val_seq_count and --test_seq_count must be >= 0")

    train_sequences = read_split_sequences(davis_root, "train")
    official_val_sequences = read_split_sequences(davis_root, "val")

    requested = val_seq_count + test_seq_count
    if requested > len(official_val_sequences):
        raise ValueError(
            "Requested val/test counts exceed official DAVIS val sequences: "
            f"val={val_seq_count}, test={test_seq_count}, available={len(official_val_sequences)}"
        )

    if val_test_split_mode == "sorted":
        chosen_order = official_val_sequences
    else:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(official_val_sequences))
        chosen_order = [official_val_sequences[int(i)] for i in perm]

    val_sequences = chosen_order[:val_seq_count]
    test_sequences = chosen_order[val_seq_count : val_seq_count + test_seq_count]
    unused_val_sequences = chosen_order[val_seq_count + test_seq_count :]
    return {
        "train": train_sequences,
        "val": val_sequences,
        "test": test_sequences,
        "unused_official_val": unused_val_sequences,
    }


def load_sequence_paths(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence

    image_paths: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(image_dir.glob(f"*{ext}")))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    mask_paths = [mask_dir / f"{path.stem}.png" for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


def _nearest_multiple(x: int, patch: int) -> int:
    down = (x // patch) * patch
    up = down + patch
    return up if abs(up - x) <= abs(x - down) else down


def _resize_image(
    img: np.ndarray,
    target_size: int,
    method: str,
    interpolation: Image.Resampling | None = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    if method in ("upper_bound_resize", "upper_bound_crop"):
        longest = max(h, w)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
    elif method in ("lower_bound_resize", "lower_bound_crop"):
        shortest = min(h, w)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
    else:
        raise ValueError(f"Unsupported resize method: {method}")

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resize_interp = interpolation
    if resize_interp is None:
        resize_interp = Image.Resampling.BICUBIC if scale > 1.0 else Image.Resampling.BOX
    pil_img = Image.fromarray(img)
    return np.asarray(pil_img.resize((new_w, new_h), resample=resize_interp))


def _make_divisible(
    arr: np.ndarray,
    patch: int,
    method: str,
    interpolation: Image.Resampling | None = None,
) -> np.ndarray:
    h, w = arr.shape[:2]
    if method.endswith("crop"):
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return arr
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return arr[top : top + new_h, left : left + new_w]

    if method.endswith("resize"):
        new_w = max(1, _nearest_multiple(w, patch))
        new_h = max(1, _nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return arr
        resize_interp = interpolation
        if resize_interp is None:
            upscale = (new_w > w) or (new_h > h)
            resize_interp = Image.Resampling.BICUBIC if upscale else Image.Resampling.BOX
        pil_img = Image.fromarray(arr)
        return np.asarray(pil_img.resize((new_w, new_h), resample=resize_interp))

    raise ValueError(f"Unsupported process_res_method: {method}")


def preprocess_image_and_mask(
    image_path: Path,
    mask_path: Path,
    process_res: int,
    process_res_method: str,
    preprocess_mode: str = "native",
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    mask = (mask > 0).astype(np.uint8)

    if preprocess_mode == "aspect_pad":
        image, mask = aspect_pad_arrays(image, mask, target_size=process_res, patch_size=PATCH_SIZE)
    elif preprocess_mode == "native":
        image = _resize_image(image, process_res, process_res_method)
        mask = _resize_image(mask, process_res, process_res_method, interpolation=Image.Resampling.NEAREST)

        image = _make_divisible(image, PATCH_SIZE, process_res_method)
        mask = _make_divisible(mask, PATCH_SIZE, process_res_method, interpolation=Image.Resampling.NEAREST)
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    image_arr = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    image_arr = (image_arr - IMAGENET_MEAN) / IMAGENET_STD
    image_tensor = torch.from_numpy(image_arr)
    mask_tensor = torch.from_numpy(mask.astype(np.float32))
    return image_tensor, mask_tensor


def mask_to_token_dynamic_ratio(dynamic_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if dynamic_mask.dim() == 3:
        dynamic_mask = dynamic_mask.unsqueeze(1)
        squeeze_seq = True
    elif dynamic_mask.dim() == 4:
        squeeze_seq = False
    else:
        raise ValueError(f"Expected dynamic_mask dim 3 or 4, got {dynamic_mask.shape}")

    mask = dynamic_mask.float().clamp(0.0, 1.0)
    bsz, seq_len, height, width = mask.shape
    mask = mask.view(bsz * seq_len, 1, height, width)
    ratio = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
    ratio = ratio.view(bsz, seq_len, ratio.shape[-2], ratio.shape[-1])
    return ratio.squeeze(1) if squeeze_seq else ratio


def dynamic_ratio_to_binary_labels(
    dynamic_ratio: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = (dynamic_ratio >= threshold).to(torch.long)
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    return labels, valid_mask


def sample_frame_indices(frame_valid: torch.Tensor, tokens_per_frame: int, rng: np.random.Generator) -> np.ndarray:
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


def _resolve_anyview_model(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    while True:
        obj_id = id(current)
        if obj_id in seen:
            return current
        seen.add(obj_id)

        if hasattr(current, "backbone"):
            return current
        if hasattr(current, "da3"):
            current = current.da3
            continue
        if hasattr(current, "model"):
            current = current.model
            continue
        return current


def _block_modes_for_model(anyview_model: nn.Module) -> Dict[int, str]:
    recorder = getattr(anyview_model, "attention_recorder", None)
    if recorder is not None and getattr(recorder, "block_modes", None):
        return dict(recorder.block_modes)

    vit = anyview_model.backbone.pretrained
    alt_start = getattr(vit, "alt_start", -1)
    return {
        idx: ("global" if alt_start != -1 and idx >= alt_start and idx % 2 == 1 else "local")
        for idx in range(len(vit.blocks))
    }


def _group_block_indices(block_modes: Dict[int, str]) -> Dict[str, List[int]]:
    grouped = {"inter_frame_block": [], "intra_frame_block": []}
    for block_idx in sorted(block_modes):
        if block_modes[block_idx] == "global":
            grouped["inter_frame_block"].append(block_idx)
        else:
            grouped["intra_frame_block"].append(block_idx)
    return grouped


def build_feature_key_order(
    anyview_model: nn.Module,
    token_sources: Sequence[str],
) -> Tuple[List[str], Dict[str, List[int]], List[int]]:
    feature_keys: List[str] = []
    block_groups = _group_block_indices(_block_modes_for_model(anyview_model))
    depth_head_layers = list(anyview_model.backbone.out_layers)

    if "alt_attn" in token_sources:
        for idx in range(len(block_groups["inter_frame_block"])):
            feature_keys.append(f"inter_frame_block_{idx}")
        for idx in range(len(block_groups["intra_frame_block"])):
            feature_keys.append(f"intra_frame_block_{idx}")

    if "depth_head" in token_sources:
        for idx in range(len(depth_head_layers)):
            feature_keys.append(f"depth_head_{idx}")

    return feature_keys, block_groups, depth_head_layers


def _extract_depth_head_tokens(anyview_model: nn.Module, feats) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    head = anyview_model.head
    norm = getattr(head, "norm", None)
    if norm is None:
        raise RuntimeError("Expected anyview head to expose a token norm for depth_head export.")

    for idx, feat in enumerate(feats):
        patch_tokens = feat[0]
        bsz, seq_len, num_tokens, channels = patch_tokens.shape
        normed = norm(patch_tokens.reshape(bsz * seq_len, num_tokens, channels)).reshape(
            bsz, seq_len, num_tokens, channels
        )
        out[f"depth_head_{idx}"] = normed
    return out


def extract_feature_map(
    anyview_model: nn.Module,
    feats,
    aux_feats: Sequence[torch.Tensor],
    export_feat_layers: Sequence[int],
    token_sources: Sequence[str],
) -> Dict[str, torch.Tensor]:
    feature_map: Dict[str, torch.Tensor] = {}

    if "alt_attn" in token_sources:
        aux_by_block = {layer: feat for layer, feat in zip(export_feat_layers, aux_feats)}
        grouped = _group_block_indices(_block_modes_for_model(anyview_model))
        for idx, actual_block in enumerate(grouped["inter_frame_block"]):
            feature_map[f"inter_frame_block_{idx}"] = aux_by_block[actual_block]
        for idx, actual_block in enumerate(grouped["intra_frame_block"]):
            feature_map[f"intra_frame_block_{idx}"] = aux_by_block[actual_block]

    if "depth_head" in token_sources:
        feature_map.update(_extract_depth_head_tokens(anyview_model, feats))

    return feature_map


class SingleNpzSink:
    def __init__(self, feature_keys: List[str]):
        self.feature_keys = feature_keys
        self.features: Dict[str, List[np.ndarray]] = {key: [] for key in feature_keys}
        self.labels: List[np.ndarray] = []
        self.valid: List[np.ndarray] = []
        self.groups: List[np.ndarray] = []

    def add_batch(
        self,
        feature_batch: Dict[str, np.ndarray],
        labels: np.ndarray,
        valid: np.ndarray,
        groups: np.ndarray,
        sequence_name: str = "",
    ) -> None:
        for key in self.feature_keys:
            self.features[key].append(feature_batch[key])
        self.labels.append(labels)
        self.valid.append(valid)
        self.groups.append(groups)

    def finalize(self, output_path: Path, meta: Dict[str, object]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {}
        for key in self.feature_keys:
            payload[key] = np.concatenate(self.features[key], axis=0)
        payload["is_dynamic"] = np.concatenate(self.labels, axis=0)
        payload["valid_mask"] = np.concatenate(self.valid, axis=0)
        payload["group_ids"] = np.concatenate(self.groups, axis=0)
        payload["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(output_path, **payload)


class ShardedDirSink:
    def __init__(self, root_dir: Path, feature_keys: List[str]):
        self.root_dir = root_dir
        self.feature_keys = feature_keys
        self.part_counts_by_seq: Dict[str, Dict[str, int]] = {}
        self.sample_counts_by_seq: Dict[str, Dict[str, int]] = {}
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._scan_existing_sequences()


    def _scan_existing_sequences(self) -> None:
        sequence_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        for seq_dir in sequence_dirs:
            sequence_name = seq_dir.name
            self.part_counts_by_seq[sequence_name] = {key: 0 for key in self.feature_keys}
            self.sample_counts_by_seq[sequence_name] = {key: 0 for key in self.feature_keys}
            for key in self.feature_keys:
                feat_dir = seq_dir / key
                if not feat_dir.exists() or not feat_dir.is_dir():
                    continue
                part_files = sorted(feat_dir.glob("part_*.npz"))
                self.part_counts_by_seq[sequence_name][key] = len(part_files)
                sample_count = 0
                for part_path in part_files:
                    with np.load(part_path, allow_pickle=False) as part:
                        sample_count += int(part["is_dynamic"].shape[0])
                self.sample_counts_by_seq[sequence_name][key] = sample_count

    def sequence_complete(self, sequence_name: str) -> bool:
        seq_root = self.root_dir / sequence_name
        if not seq_root.exists() or not seq_root.is_dir():
            return False
        for key in self.feature_keys:
            feat_dir = seq_root / key
            if not feat_dir.exists() or not feat_dir.is_dir():
                return False
            if not any(feat_dir.glob("part_*.npz")):
                return False
        return True

    def _ensure_sequence(self, sequence_name: str) -> None:
        if sequence_name in self.part_counts_by_seq:
            return
        self.part_counts_by_seq[sequence_name] = {key: 0 for key in self.feature_keys}
        self.sample_counts_by_seq[sequence_name] = {key: 0 for key in self.feature_keys}
        seq_root = self.root_dir / sequence_name
        for key in self.feature_keys:
            (seq_root / key).mkdir(parents=True, exist_ok=True)

    def add_batch(
        self,
        feature_batch: Dict[str, np.ndarray],
        labels: np.ndarray,
        valid: np.ndarray,
        groups: np.ndarray,
        sequence_name: str,
    ) -> None:
        self._ensure_sequence(sequence_name)
        count = int(labels.shape[0])
        for key in self.feature_keys:
            idx = self.part_counts_by_seq[sequence_name][key]
            part_path = self.root_dir / sequence_name / key / f"part_{idx:06d}.npz"
            np.savez_compressed(
                part_path,
                tokens=feature_batch[key],
                is_dynamic=labels,
                valid_mask=valid,
                group_ids=groups,
            )
            self.part_counts_by_seq[sequence_name][key] += 1
            self.sample_counts_by_seq[sequence_name][key] += count

    def finalize(self, meta: Dict[str, object]) -> None:
        manifest = {
            "meta": meta,
            "feature_keys": self.feature_keys,
            "part_counts_by_sequence": self.part_counts_by_seq,
            "sample_counts_by_sequence": self.sample_counts_by_seq,
            "sequences": sorted(self.part_counts_by_seq.keys()),
        }
        with (self.root_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)


@torch.no_grad()
def _run_anyview_backbone(
    anyview_model: nn.Module,
    images_b: torch.Tensor,
    export_feat_layers: Sequence[int],
) -> Tuple[Sequence, Sequence[torch.Tensor]]:
    device = images_b.device
    autocast_context = (
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        return anyview_model.backbone(images_b, export_feat_layers=list(export_feat_layers))


def collect_split_samples(
    anyview_model: nn.Module,
    davis_root: Path,
    split_name: str,
    sequences: List[str],
    token_sources: Sequence[str],
    feature_key_order: Sequence[str],
    export_feat_layers: Sequence[int],
    frames_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    process_res: int,
    process_res_method: str,
    device: torch.device,
    seed: int,
    group_start: int,
    sink,
) -> int:
    rng = np.random.default_rng(seed)
    group_counter = group_start

    for seq_idx, sequence in enumerate(sequences):
        if isinstance(sink, ShardedDirSink) and sink.sequence_complete(sequence):
            print(f"[{split_name}] skipped existing sequence {seq_idx + 1}/{len(sequences)}: {sequence}")
            continue
        image_paths, mask_paths = load_sequence_paths(davis_root, sequence)
 
        for start in range(0, len(image_paths), frames_per_forward):
            end = min(start + frames_per_forward, len(image_paths))
            chunk_image_paths = image_paths[start:end]
            chunk_mask_paths = mask_paths[start:end]

            image_list = []
            mask_list = []
            for img_path, mask_path in zip(chunk_image_paths, chunk_mask_paths):
                img_t, mask_t = preprocess_image_and_mask(
                    img_path,
                    mask_path,
                    process_res=process_res,
                    process_res_method=process_res_method,
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

        print(f"[{split_name}] processed sequence {seq_idx + 1}/{len(sequences)}: {sequence}")

    return group_counter


def main() -> None:
    args = parse_args()
    davis_root = resolve_davis_root(args.davis_root)
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

    split_plan = build_davis_export_splits(
        davis_root=davis_root,
        val_seq_count=args.val_seq_count,
        test_seq_count=args.test_seq_count,
        val_test_split_mode=args.val_test_split_mode,
        split_seed=args.split_seed,
    )
    export_splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print(f"Output directory: {output_dir}")
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
        sequences = split_plan[split_name]
        if not sequences:
            raise RuntimeError(
                f"Split '{split_name}' resolved to 0 sequences. "
                "Adjust --val_seq_count / --test_seq_count."
            )

        if args.save_mode == "sharded_dirs":
            sink = ShardedDirSink(output_dir / args.tokens_subdir / split_name, feature_key_order)
        else:
            sink = SingleNpzSink(feature_key_order)

        collect_split_samples(
            anyview_model=anyview_model,
            davis_root=davis_root,
            split_name=split_name,
            sequences=sequences,
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            export_feat_layers=export_feat_layers,
            frames_per_forward=args.frames_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
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
                "train": "official_train",
                "val": f"first_{args.val_seq_count}_from_official_val_{args.val_test_split_mode}",
                "test": f"next_{args.test_seq_count}_from_official_val_{args.val_test_split_mode}",
                "split_seed": args.split_seed,
            },
            "sequence_count": len(sequences),
            "sequences": sequences,
            "unused_official_val_sequences": split_plan["unused_official_val"],
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
