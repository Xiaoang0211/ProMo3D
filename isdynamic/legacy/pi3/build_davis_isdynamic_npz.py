#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.models.pi3 import (  # noqa: E402
    MODEL_CHOICES,
    PI3_PATCH_SIZE,
    PI3_TARGET_SIZE,
    SOURCE_CHOICES,
    build_feature_key_order,
    extract_feature_map,
    load_pi3_family_model,
)
from isdynamic.core.artifacts import token_path_for_model  # noqa: E402
from isdynamic.core.common_labels import dynamic_ratio_to_binary_labels, mask_to_token_dynamic_ratio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from DAVIS-2017 for Pi3/Pi3X. "
            "Dynamic masks are treated as binary: annotation > 0 => dynamic."
        )
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p",
        help="Path to DAVIS root or parent containing DAVIS/.",
    )
    parser.add_argument("--model", type=str, default="pi3x", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors/.pt/.pth).")
    parser.add_argument(
        "--use_multimodal_pi3x",
        action="store_true",
        help="Enable Pi3X multimodal branch while exporting RGB-only features.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder for exports.")
    parser.add_argument("--dataset_name", type=str, default="davis", help="Dataset subfolder name.")
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
        default="alt_attn,cam_head,conf_head,point_head",
        help=(
            "Comma-separated source groups. "
            "Choices: alt_attn,image_encoder,cam_head,conf_head,point_head. "
            "alt_attn expands to inter_frame_block_i and intra_frame_block_i; "
            "image_encoder expands to image_encoder_i."
        ),
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="sharded_dirs",
        choices=["sharded_dirs", "single_npz"],
        help=(
            "sharded_dirs: write per-feature directories (recommended for many blocks). "
            "single_npz: write one tokens.npz with many keys (can be very large)."
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
        help="Number of frames per model forward chunk.",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Dynamic ratio threshold for token labels.")
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
    for s in sources:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
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


def compute_resize_dims(width: int, height: int, target_size: int = PI3_TARGET_SIZE, patch: int = PI3_PATCH_SIZE):
    if width >= height:
        new_w = target_size
        new_h = int(round(height * (new_w / width) / patch) * patch)
    else:
        new_h = target_size
        new_w = int(round(width * (new_h / height) / patch) * patch)
    new_w = max(new_w, patch)
    new_h = max(new_h, patch)
    return new_w, new_h


def preprocess_image_and_mask(image_path: Path, mask_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np)

    width, height = image.size
    new_w, new_h = compute_resize_dims(width, height)

    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    pad_w = PI3_TARGET_SIZE - new_w
    pad_h = PI3_TARGET_SIZE - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    image_arr = np.array(image, dtype=np.uint8)
    mask_arr = np.array(mask, dtype=np.uint8)
    if mask_arr.ndim == 3:
        mask_arr = np.any(mask_arr > 0, axis=-1)
    mask_arr = (mask_arr > 0).astype(np.uint8) * 255

    image_arr = np.pad(
        image_arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    mask_arr = np.pad(
        mask_arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    image_tensor = torch.from_numpy(image_arr).float().permute(2, 0, 1) / 255.0
    mask_tensor = (torch.from_numpy(mask_arr).float() > 0).float()
    return image_tensor, mask_tensor


def read_split_sequences(davis_root: Path, split: str) -> List[str]:
    split_file = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as handle:
        seqs = [line.strip() for line in handle if line.strip()]
    return sorted(seqs)


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
    elif val_test_split_mode == "shuffled":
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(official_val_sequences))
        chosen_order = [official_val_sequences[int(i)] for i in perm]
    else:
        raise ValueError(f"Unsupported val_test_split_mode: {val_test_split_mode}")

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
    image_paths = sorted(image_dir.glob("*.jpg"))
    if len(image_paths) == 0:
        image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")

    mask_paths = [mask_dir / (p.stem + ".png") for p in image_paths]
    missing = [str(p) for p in mask_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing[:20]))
    return image_paths, mask_paths


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


class SingleNpzSink:
    def __init__(self, feature_keys: List[str]):
        self.feature_keys = feature_keys
        self.features: Dict[str, List[np.ndarray]] = {k: [] for k in feature_keys}
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
        del sequence_name
        for k in self.feature_keys:
            self.features[k].append(feature_batch[k])
        self.labels.append(labels)
        self.valid.append(valid)
        self.groups.append(groups)

    def finalize(self, output_path: Path, meta: Dict[str, object]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, object] = {}
        for k in self.feature_keys:
            payload[k] = np.concatenate(self.features[k], axis=0)
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
        for seq_root in sorted(self.root_dir.iterdir() if self.root_dir.exists() else []):
            if not seq_root.is_dir():
                continue
            sequence_name = seq_root.name
            self.part_counts_by_seq[sequence_name] = {}
            self.sample_counts_by_seq[sequence_name] = {}
            for k in self.feature_keys:
                feat_dir = seq_root / k
                part_files = sorted(feat_dir.glob("part_*.npz")) if feat_dir.exists() else []
                self.part_counts_by_seq[sequence_name][k] = len(part_files)
                sample_count = 0
                for part_path in part_files:
                    try:
                        with np.load(part_path) as data:
                            sample_count += int(data["is_dynamic"].shape[0])
                    except Exception:
                        continue
                self.sample_counts_by_seq[sequence_name][k] = sample_count

    def sequence_complete(self, sequence_name: str) -> bool:
        seq_root = self.root_dir / sequence_name
        if not seq_root.exists():
            return False
        for k in self.feature_keys:
            feat_dir = seq_root / k
            if not feat_dir.exists():
                return False
            if not any(feat_dir.glob("part_*.npz")):
                return False
        return True

    def _ensure_sequence(self, sequence_name: str) -> None:
        seq_root = self.root_dir / sequence_name
        if sequence_name not in self.part_counts_by_seq:
            self.part_counts_by_seq[sequence_name] = {}
            self.sample_counts_by_seq[sequence_name] = {}
        for k in self.feature_keys:
            self.part_counts_by_seq[sequence_name].setdefault(k, 0)
            self.sample_counts_by_seq[sequence_name].setdefault(k, 0)
            (seq_root / k).mkdir(parents=True, exist_ok=True)

    def add_batch(
        self,
        feature_batch: Dict[str, np.ndarray],
        labels: np.ndarray,
        valid: np.ndarray,
        groups: np.ndarray,
        sequence_name: str,
    ) -> None:
        self._ensure_sequence(sequence_name)
        n = int(labels.shape[0])
        for k in self.feature_keys:
            idx = self.part_counts_by_seq[sequence_name][k]
            part_path = self.root_dir / sequence_name / k / f"part_{idx:06d}.npz"
            part_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                part_path,
                tokens=feature_batch[k],
                is_dynamic=labels,
                valid_mask=valid,
                group_ids=groups,
            )
            self.part_counts_by_seq[sequence_name][k] += 1
            self.sample_counts_by_seq[sequence_name][k] += n

    def finalize(self, meta: Dict[str, object]) -> None:
        manifest = {
            "meta": meta,
            "feature_keys": self.feature_keys,
            "part_counts_by_sequence": self.part_counts_by_seq,
            "sample_counts_by_sequence": self.sample_counts_by_seq,
            "sequences": sorted(self.part_counts_by_seq.keys()),
        }
        with (self.root_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def collect_split_samples(
    model_name: str,
    model: torch.nn.Module,
    davis_root: Path,
    split_name: str,
    sequences: List[str],
    token_sources: Iterable[str],
    feature_key_order: List[str],
    frames_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
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
            for img_path, msk_path in zip(chunk_image_paths, chunk_mask_paths):
                img_t, msk_t = preprocess_image_and_mask(img_path, msk_path)
                image_list.append(img_t)
                mask_list.append(msk_t)

            images = torch.stack(image_list, dim=0)
            masks = torch.stack(mask_list, dim=0)

            with torch.no_grad():
                images_b = images.unsqueeze(0).to(device)
                feature_map = extract_feature_map(
                    model_name=model_name,
                    model=model,
                    images_b=images_b,
                    token_sources=token_sources,
                )

            ratios = mask_to_token_dynamic_ratio(masks.unsqueeze(0), patch_size=PI3_PATCH_SIZE).squeeze(0)
            labels, valid = dynamic_ratio_to_binary_labels(ratios, threshold=threshold)
            labels = labels.reshape(labels.shape[0], -1).cpu()
            valid = valid.reshape(valid.shape[0], -1).cpu()

            for k in feature_key_order:
                if k not in feature_map:
                    raise KeyError(f"Missing expected feature key '{k}' in extracted feature_map")
                f = feature_map[k]
                if f.shape[0] != 1 or f.shape[1] != labels.shape[0] or f.shape[2] != labels.shape[1]:
                    raise ValueError(
                        f"Feature/label mismatch for key={k} in {sequence} [{start}:{end}] "
                        f"features={tuple(f.shape)} labels={tuple(labels.shape)}"
                    )

            feature_parts: Dict[str, List[np.ndarray]] = {k: [] for k in feature_key_order}
            labels_parts: List[np.ndarray] = []
            valid_parts: List[np.ndarray] = []
            groups_parts: List[np.ndarray] = []

            for frame_idx in range(labels.shape[0]):
                chosen = sample_frame_indices(valid[frame_idx], tokens_per_frame=tokens_per_frame, rng=rng)
                if chosen.size == 0:
                    group_counter += 1
                    continue

                y = labels[frame_idx, chosen].cpu().numpy().astype(np.int64)
                v = valid[frame_idx, chosen].cpu().numpy().astype(np.bool_)
                g = np.full((chosen.shape[0],), group_counter, dtype=np.int64)

                labels_parts.append(y)
                valid_parts.append(v)
                groups_parts.append(g)

                for k in feature_key_order:
                    x = feature_map[k][0, frame_idx, chosen].detach().cpu().numpy().astype(np.float32)
                    feature_parts[k].append(x)

                group_counter += 1

            if labels_parts:
                labels_chunk = np.concatenate(labels_parts, axis=0)
                valid_chunk = np.concatenate(valid_parts, axis=0)
                groups_chunk = np.concatenate(groups_parts, axis=0)
                feature_chunk = {k: np.concatenate(feature_parts[k], axis=0) for k in feature_key_order}
                sink.add_batch(feature_chunk, labels_chunk, valid_chunk, groups_chunk, sequence_name=sequence)

        print(f"[{split_name}] processed sequence {seq_idx + 1}/{len(sequences)}: {sequence}")

    return group_counter


def main() -> None:
    args = parse_args()
    davis_root = resolve_davis_root(args.davis_root)

    token_sources = _parse_token_sources(args.token_sources)
    device = torch.device(args.device)

    model = load_pi3_family_model(
        model_name=args.model,
        checkpoint=Path(args.checkpoint),
        device=device,
        use_multimodal_pi3x=args.use_multimodal_pi3x,
    )
    feature_key_order = build_feature_key_order(model=model, token_sources=token_sources)

    output_root = args.output_dir if args.output_dir is not None else token_path_for_model(args.model, "data", "isdynamic")
    dataset_name = args.dataset_name.strip().lower()
    output_dir = Path(output_root) / dataset_name if dataset_name else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_plan = build_davis_export_splits(
        davis_root=davis_root,
        val_seq_count=args.val_seq_count,
        test_seq_count=args.test_seq_count,
        val_test_split_mode=args.val_test_split_mode,
        split_seed=args.split_seed,
    )
    if args.split == "all":
        export_splits = ["train", "val", "test"]
    else:
        export_splits = [args.split]

    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Token sources: {token_sources}")
    print(f"Feature key order: {feature_key_order[:6]} ... total={len(feature_key_order)}")
    print(f"Save mode: {args.save_mode}")
    for split_name in export_splits:
        print(f"Split '{split_name}' sequences: {len(split_plan[split_name])}")

    for i, split_name in enumerate(export_splits):
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
            model_name=args.model,
            model=model,
            davis_root=davis_root,
            split_name=split_name,
            sequences=sequences,
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            frames_per_forward=args.frames_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            device=device,
            seed=args.seed + i,
            group_start=0,
            sink=sink,
        )

        meta = {
            "split": split_name,
            "model": args.model,
            "use_multimodal_pi3x": bool(args.use_multimodal_pi3x),
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
            "frames_per_forward": args.frames_per_forward,
            "tokens_per_frame": args.tokens_per_frame,
            "threshold": args.threshold,
            "patch_size": PI3_PATCH_SIZE,
            "target_size": PI3_TARGET_SIZE,
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
