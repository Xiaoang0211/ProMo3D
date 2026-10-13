#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.models.dust3r import (
    DUST3R_PATCH_SIZE,
    MODEL_NAME_CHOICES,
    SOURCE_CHOICES,
    build_feature_key_order,
    collate_views,
    extract_feature_map,
    load_dust3r_model,
    preprocess_image_and_mask,
)
from isdynamic.core.stereo_labels import dynamic_ratio_to_binary_labels, mask_to_token_dynamic_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-level IsDynamic exports from DAVIS-2017 for Dust3R. "
            "Dynamic masks are treated as binary: annotation > 0 => dynamic."
        )
    )
    parser.add_argument(
        "--davis_root",
        type=str,
        default="/home/xiaoang/Datasets/DAVIS-2017-trainval-480p",
        help="Path to DAVIS root or parent containing DAVIS/.",
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
    parser.add_argument("--dataset_name", type=str, default="davis", help="Dataset subfolder name.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=(
            "Which export split to write. "
            "train=official train split; val/test are carved from official val split."
        ),
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
    sources = [item.strip() for item in raw.split(",") if item.strip()]
    if not sources:
        raise ValueError("--token_sources must contain at least one source.")
    unknown = [item for item in sources if item not in SOURCE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown token_sources {unknown}. Choices: {SOURCE_CHOICES}")
    ordered = []
    seen = set()
    for item in sources:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
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
    missing = [str(item) for item in required if not item.exists()]
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
    elif val_test_split_mode == "shuffled":
        rng = np.random.default_rng(split_seed)
        chosen_order = [official_val_sequences[int(idx)] for idx in rng.permutation(len(official_val_sequences))]
    else:
        raise ValueError(f"Unsupported val_test_split_mode: {val_test_split_mode}")

    return {
        "train": train_sequences,
        "val": chosen_order[:val_seq_count],
        "test": chosen_order[val_seq_count : val_seq_count + test_seq_count],
        "unused_official_val": chosen_order[val_seq_count + test_seq_count :],
    }


def load_sequence_paths(davis_root: Path, sequence: str) -> Tuple[List[Path], List[Path]]:
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
        del sequence_name
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
        num_samples = int(labels.shape[0])
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
            self.sample_counts_by_seq[sequence_name][key] += num_samples

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


def _pair_indices(num_frames: int, pair_stride: int) -> List[Tuple[int, int]]:
    pairs = []
    for idx in range(num_frames):
        ref_idx = idx + pair_stride
        if ref_idx >= num_frames:
            ref_idx = max(0, idx - pair_stride)
        pairs.append((idx, ref_idx))
    return pairs


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
    davis_root: Path,
    split_name: str,
    sequences: List[str],
    token_sources: Iterable[str],
    feature_key_order: List[str],
    image_size: int,
    pair_stride: int,
    pairs_per_forward: int,
    tokens_per_frame: int,
    threshold: float,
    device: torch.device,
    seed: int,
    sink,
) -> None:
    rng = np.random.default_rng(seed)
    patch_size = int(model.patch_size)
    square_ok = bool(getattr(model, "square_ok", False))
    group_counter = 0

    for seq_idx, sequence in enumerate(sequences):
        image_paths, mask_paths = load_sequence_paths(davis_root, sequence)
        views = []
        masks = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            view, mask = preprocess_image_and_mask(
                image_path=image_path,
                mask_path=mask_path,
                image_size=image_size,
                patch_size=patch_size,
                square_ok=square_ok,
            )
            views.append(view)
            masks.append(mask)

        mask_batch = torch.stack(masks, dim=0).unsqueeze(0)
        ratios = mask_to_token_dynamic_ratio(mask_batch, patch_size=patch_size).squeeze(0)
        labels = dynamic_ratio_to_binary_labels(ratios, threshold=threshold)[0].reshape(len(views), -1).cpu()
        valid = torch.ones_like(labels, dtype=torch.bool)

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
                sink.add_batch(
                    feature_chunk,
                    labels_chunk,
                    valid_chunk,
                    groups_chunk,
                    sequence_name=sequence,
                )

        print(f"[{split_name}] processed sequence {seq_idx + 1}/{len(sequences)}: {sequence}")


def main() -> None:
    args = parse_args()
    davis_root = resolve_davis_root(args.davis_root)
    token_sources = _parse_token_sources(args.token_sources)
    device = torch.device(args.device)
    weights_or_name = args.weights if args.weights is not None else f"naver/{args.model_name}"

    model = load_dust3r_model(weights_or_name=weights_or_name, device=device)
    feature_key_order = build_feature_key_order(model=model, token_sources=token_sources)

    dataset_name = args.dataset_name.strip().lower()
    output_dir = Path(args.output_dir) / dataset_name if dataset_name else Path(args.output_dir)
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
    print(f"Weights: {weights_or_name}")
    print(f"Token sources: {token_sources}")
    print(f"Feature key order: {feature_key_order[:6]} ... total={len(feature_key_order)}")
    print(f"Save mode: {args.save_mode}")

    for split_idx, split_name in enumerate(export_splits):
        sequences = split_plan[split_name]
        if not sequences:
            raise RuntimeError(
                f"Split '{split_name}' resolved to 0 sequences. "
                "Adjust --val_seq_count / --test_seq_count."
            )

        sink = (
            ShardedDirSink(output_dir / args.tokens_subdir / split_name, feature_key_order)
            if args.save_mode == "sharded_dirs"
            else SingleNpzSink(feature_key_order)
        )

        collect_split_samples(
            model=model,
            davis_root=davis_root,
            split_name=split_name,
            sequences=sequences,
            token_sources=token_sources,
            feature_key_order=feature_key_order,
            image_size=args.image_size,
            pair_stride=args.pair_stride,
            pairs_per_forward=args.pairs_per_forward,
            tokens_per_frame=args.tokens_per_frame,
            threshold=args.threshold,
            device=device,
            seed=args.seed + split_idx,
            sink=sink,
        )

        meta = {
            "split": split_name,
            "weights_or_name": weights_or_name,
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
            "image_size": args.image_size,
            "patch_size": int(model.patch_size),
            "pair_stride": args.pair_stride,
            "pair_policy": "adjacent_reference",
            "pairs_per_forward": args.pairs_per_forward,
            "tokens_per_frame": args.tokens_per_frame,
            "threshold": args.threshold,
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
