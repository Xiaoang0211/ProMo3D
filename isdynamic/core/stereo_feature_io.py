from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _flatten_token_tensor(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.ndim == 4:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"Expected features with ndim in {{2,4}}, got {x.shape}")


def _flatten_label_tensor(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.reshape(-1)
    if x.ndim in {3, 4}:
        return x.reshape(-1)
    raise ValueError(f"Unsupported label shape: {x.shape}")


def load_token_npz(
    npz_path: str,
    feature_key: str = "tokens",
    label_key: str = "is_dynamic",
    valid_key: Optional[str] = "valid_mask",
    group_key: str = "group_ids",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if feature_key not in data or label_key not in data:
            raise KeyError(f"NPZ must contain keys '{feature_key}' and '{label_key}'")

        raw_features = data[feature_key]
        features = _flatten_token_tensor(raw_features).astype(np.float32)
        labels = _flatten_label_tensor(data[label_key]).astype(np.int64)

        if valid_key is not None and valid_key in data:
            valid = _flatten_label_tensor(data[valid_key]).astype(np.bool_)
        else:
            valid = np.ones_like(labels, dtype=np.bool_)

        if group_key in data:
            group_ids = _flatten_label_tensor(data[group_key]).astype(np.int64)
        elif raw_features.ndim == 4:
            b, s, p, _ = raw_features.shape
            group_ids = np.repeat(np.arange(b * s, dtype=np.int64), p)
        else:
            group_ids = np.zeros(features.shape[0], dtype=np.int64)

    if features.shape[0] != labels.shape[0] or labels.shape[0] != valid.shape[0]:
        raise ValueError(
            "Mismatched sample counts after flattening: "
            f"features={features.shape}, labels={labels.shape}, valid={valid.shape}"
        )
    if group_ids.shape[0] != features.shape[0]:
        raise ValueError(
            "Mismatched group count after flattening: "
            f"group_ids={group_ids.shape}, features={features.shape}"
        )

    return (
        torch.from_numpy(features),
        torch.from_numpy(labels),
        torch.from_numpy(valid),
        torch.from_numpy(group_ids),
    )
