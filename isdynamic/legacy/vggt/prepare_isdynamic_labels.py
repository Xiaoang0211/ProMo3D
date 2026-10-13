#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isdynamic.core.common_labels import dynamic_ratio_to_binary_labels, mask_to_token_dynamic_ratio

VGGT_PATCH_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert binary dynamic masks to token-level IsDynamic labels and valid mask."
    )
    parser.add_argument("--input_npz", type=str, required=True, help="Input NPZ containing dynamic masks.")
    parser.add_argument("--output_npz", type=str, required=True, help="Output NPZ path.")
    parser.add_argument("--mask_key", type=str, default="dynamic_mask", help="Key name for dynamic masks.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Dynamic ratio threshold.")
    parser.add_argument("--label_key", type=str, default="is_dynamic", help="Output key for labels.")
    parser.add_argument("--valid_key", type=str, default="valid_mask", help="Output key for validity mask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_npz)
    if not in_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    with np.load(in_path, allow_pickle=False) as data:
        if args.mask_key not in data:
            raise KeyError(f"Mask key '{args.mask_key}' not found in {in_path}")
        mask_np = data[args.mask_key]
        extras = {k: data[k] for k in data.files if k != args.mask_key}

    mask_t = torch.from_numpy(mask_np)
    ratios = mask_to_token_dynamic_ratio(mask_t, patch_size=VGGT_PATCH_SIZE)

    labels, valid = dynamic_ratio_to_binary_labels(
        ratios,
        threshold=args.threshold,
    )

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **extras,
        **{
            args.label_key: labels.cpu().numpy(),
            args.valid_key: valid.cpu().numpy(),
            "dynamic_ratio": ratios.cpu().numpy(),
        },
    )
    print(f"Saved token labels to: {out_path}")


if __name__ == "__main__":
    main()
