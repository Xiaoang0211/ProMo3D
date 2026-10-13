#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from isdynamic.models.pi3 import (
    GROUP_CHOICES,
    MODEL_CHOICES,
    blockdiag,
    get_projection_matrix,
    load_pi3_family_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed projection matrices from Pi3/Pi3X blocks for pair_attention / pair_value probes. "
            "By default uses the next block (l+1) for features from block l."
        )
    )
    parser.add_argument("--model", type=str, default="pi3x", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors/.pt/.pth).")
    parser.add_argument("--group", type=str, required=True, choices=GROUP_CHOICES + ["alt_attn_concat"])
    parser.add_argument("--feature_block_idx", type=int, required=True, help="Index l of the feature block whose output features are being probed.")
    parser.add_argument("--projection_block_idx", type=int, default=None, help="Optional explicit projection block index. If omitted, uses l+1.")
    parser.add_argument("--fallback_current_for_last", action="store_true", help="If l+1 is out of range, use l instead of raising an error.")
    parser.add_argument("--mode", type=str, default="attention", choices=["attention", "value"], help="attention => Wq^T Wk, value => Wv.")
    parser.add_argument("--use_multimodal_pi3x", action="store_true", help="Enable Pi3X multimodal branch while loading the checkpoint.")
    parser.add_argument("--out", type=str, required=True, help="Output matrix path (.npy, .npz, .pt/.pth).")
    return parser.parse_args()


def _save_matrix(path: Path, matrix: torch.Tensor, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, matrix.cpu().numpy())
        return
    if suffix == ".npz":
        np.savez_compressed(path, matrix=matrix.cpu().numpy(), **metadata)
        return
    if suffix in {".pt", ".pth"}:
        torch.save({"matrix": matrix.cpu(), "metadata": metadata}, path)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_pi3_family_model(
        model_name=args.model,
        checkpoint=checkpoint_path,
        device=torch.device("cpu"),
        use_multimodal_pi3x=args.use_multimodal_pi3x,
    )
    metadata: Dict[str, object] = {
        "model": args.model,
        "group": args.group,
        "mode": args.mode,
        "feature_block_idx": args.feature_block_idx,
    }
    if args.group == "alt_attn_concat":
        m_frame, frame_proj_idx = get_projection_matrix(
            model=model,
            group="intra_frame_block",
            feature_block_idx=args.feature_block_idx,
            mode=args.mode,
            projection_block_idx=args.projection_block_idx,
            fallback_current_for_last=args.fallback_current_for_last,
        )
        m_global, global_proj_idx = get_projection_matrix(
            model=model,
            group="inter_frame_block",
            feature_block_idx=args.feature_block_idx,
            mode=args.mode,
            projection_block_idx=args.projection_block_idx,
            fallback_current_for_last=args.fallback_current_for_last,
        )
        matrix = blockdiag(m_frame, m_global)
        metadata["projection_block_idx"] = {"intra_frame_block": int(frame_proj_idx), "inter_frame_block": int(global_proj_idx)}
    else:
        matrix, projection_idx = get_projection_matrix(
            model=model,
            group=args.group,
            feature_block_idx=args.feature_block_idx,
            mode=args.mode,
            projection_block_idx=args.projection_block_idx,
            fallback_current_for_last=args.fallback_current_for_last,
        )
        metadata["projection_block_idx"] = int(projection_idx)
    out_path = Path(args.out)
    _save_matrix(out_path, matrix, metadata)
    print(f"Saved matrix: {out_path}")
    print(f"Shape: {tuple(matrix.shape)}")
    print(f"Mode: {args.mode}, group: {args.group}")
    print(f"Projection index info: {metadata['projection_block_idx']}")


if __name__ == "__main__":
    main()
