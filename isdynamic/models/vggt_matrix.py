#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from vggt.models.vggt import VGGT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed projection matrices from VGGT blocks for pair_attention / pair_value "
            "and their biased quadratic variants. By default uses the next block (l+1) for features from block l."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VGGT checkpoint (.pt/.pth).")
    parser.add_argument("--feature_block_idx", type=int, required=True, help="Index l of the block whose output features are being probed.")
    parser.add_argument("--projection_block_idx", type=int, default=None, help="Optional explicit projection block index. If omitted, uses l+1.")
    parser.add_argument("--fallback_current_for_last", action="store_true", help="If l+1 is out of range, use l instead of raising an error.")
    parser.add_argument("--mode", type=str, default="attention", choices=["attention", "value"], help="attention => Wq^T Wk, value => Wv.")
    parser.add_argument("--stream", type=str, default="frame", choices=["frame", "global", "concat"], help="Which aggregator stream to extract from. concat builds a block-diagonal matrix.")
    parser.add_argument("--out", type=str, required=True, help="Output matrix path (.npy, .npz, .pt/.pth).")
    return parser.parse_args()


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                candidate = obj[key]
                if all(isinstance(v, torch.Tensor) for v in candidate.values()):
                    return candidate
    raise ValueError("Could not find a valid tensor state_dict in checkpoint.")


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        out[key[len("module.") :] if key.startswith("module.") else key] = value
    return out


def _load_aggregator_from_checkpoint(checkpoint_path: str) -> VGGT:
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_prefixes(_extract_state_dict(ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _get_projection_block_idx(args: argparse.Namespace, depth: int) -> int:
    idx = args.projection_block_idx if args.projection_block_idx is not None else args.feature_block_idx + 1
    if idx >= depth:
        if args.fallback_current_for_last:
            idx = args.feature_block_idx
        else:
            raise ValueError(
                f"Projection block index {idx} is out of range for depth {depth}. Use --fallback_current_for_last or provide --projection_block_idx."
            )
    if idx < 0:
        raise ValueError(f"Invalid projection block index: {idx}")
    return idx


def _extract_matrix_from_block(block, mode: str) -> torch.Tensor:
    qkv = block.attn.qkv.weight.detach().float()
    c = qkv.shape[1]
    if qkv.shape[0] != 3 * c:
        raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")
    wq, wk, wv = qkv.chunk(3, dim=0)
    if mode == "attention":
        return wq.transpose(0, 1) @ wk
    if mode == "value":
        return wv
    raise ValueError(f"Unsupported mode: {mode}")


def _blockdiag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(a.shape[0] + b.shape[0], a.shape[1] + b.shape[1], dtype=a.dtype)
    out[: a.shape[0], : a.shape[1]] = a
    out[a.shape[0] :, a.shape[1] :] = b
    return out


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
    model = _load_aggregator_from_checkpoint(str(checkpoint_path))
    projection_idx = _get_projection_block_idx(args, depth=model.aggregator.depth)
    if args.stream == "frame":
        matrix = _extract_matrix_from_block(model.aggregator.frame_blocks[projection_idx], mode=args.mode)
    elif args.stream == "global":
        matrix = _extract_matrix_from_block(model.aggregator.global_blocks[projection_idx], mode=args.mode)
    else:
        matrix = _blockdiag(
            _extract_matrix_from_block(model.aggregator.frame_blocks[projection_idx], mode=args.mode),
            _extract_matrix_from_block(model.aggregator.global_blocks[projection_idx], mode=args.mode),
        )
    out_path = Path(args.out)
    metadata = {
        "mode": args.mode,
        "stream": args.stream,
        "feature_block_idx": args.feature_block_idx,
        "projection_block_idx": projection_idx,
    }
    _save_matrix(out_path, matrix, metadata)
    print(f"Saved matrix: {out_path}")
    print(f"Shape: {tuple(matrix.shape)}")
    print(f"Mode: {args.mode}, stream: {args.stream}, projection block: {projection_idx}")


if __name__ == "__main__":
    main()
