from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

TokenData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
RESIDUALIZER_FILENAME = "position_residualizer.pt"


@dataclass
class PositionResidualizer:
    mode: str
    fourier_freqs: int
    ridge: float
    coef: torch.Tensor
    basis_dim: int


def position_residualization_mode(train_cfg: Any) -> str:
    mode = str(getattr(train_cfg, "position_residualization", "none") or "none").strip().lower()
    return "none" if mode in {"", "none", "off", "false", "0"} else mode


def position_residualization_metadata(residualizer: Optional[PositionResidualizer]) -> Optional[Dict[str, object]]:
    if residualizer is None:
        return None
    return {
        "mode": residualizer.mode,
        "fourier_freqs": int(residualizer.fourier_freqs),
        "ridge": float(residualizer.ridge),
        "basis_dim": int(residualizer.basis_dim),
        "note": "Post-hoc residualization from exported token group-local position basis; this is not a no-RoPE backbone re-export.",
    }


def save_position_residualizer(path: str | Path, residualizer: Optional[PositionResidualizer]) -> None:
    if residualizer is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": residualizer.mode,
            "fourier_freqs": int(residualizer.fourier_freqs),
            "ridge": float(residualizer.ridge),
            "coef": residualizer.coef.cpu(),
            "basis_dim": int(residualizer.basis_dim),
        },
        target,
    )


@lru_cache(maxsize=512)
def load_position_residualizer(path_text: str) -> Optional[PositionResidualizer]:
    path = Path(path_text)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    return PositionResidualizer(
        mode=str(payload["mode"]),
        fourier_freqs=int(payload["fourier_freqs"]),
        ridge=float(payload["ridge"]),
        coef=payload["coef"].cpu(),
        basis_dim=int(payload["basis_dim"]),
    )


def load_position_residualizer_from_run(run_dir: str | Path) -> Optional[PositionResidualizer]:
    return load_position_residualizer(str(Path(run_dir) / RESIDUALIZER_FILENAME))


def apply_position_residualizer_to_token_features(
    token_features: torch.Tensor,
    residualizer: Optional[PositionResidualizer],
) -> torch.Tensor:
    if residualizer is None:
        return token_features
    if token_features.ndim != 2:
        raise ValueError(f"Expected token features [N,C], got {tuple(token_features.shape)}")
    group_ids = torch.zeros(token_features.shape[0], dtype=torch.long, device=token_features.device)
    basis = build_position_basis(
        group_ids,
        mode=residualizer.mode,
        fourier_freqs=residualizer.fourier_freqs,
        device=token_features.device,
        dtype=torch.float32,
    )
    residual = basis @ residualizer.coef.to(device=token_features.device, dtype=torch.float32)
    return token_features.float() - residual


def _group_local_xy(group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    groups = group_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    local = np.empty(groups.shape[0], dtype=np.float32)
    counts = np.empty(groups.shape[0], dtype=np.float32)

    start = 0
    while start < sorted_groups.shape[0]:
        end = start + 1
        while end < sorted_groups.shape[0] and sorted_groups[end] == sorted_groups[start]:
            end += 1
        idx = order[start:end]
        n = end - start
        local[idx] = np.arange(n, dtype=np.float32)
        counts[idx] = float(n)
        start = end

    width = np.ceil(np.sqrt(counts)).astype(np.float32)
    height = np.ceil(counts / np.maximum(width, 1.0)).astype(np.float32)
    x = np.mod(local, width) / np.maximum(width - 1.0, 1.0)
    y = np.floor(local / np.maximum(width, 1.0)) / np.maximum(height - 1.0, 1.0)
    return torch.from_numpy(x), torch.from_numpy(y)


def build_position_basis(
    group_ids: torch.Tensor,
    mode: str,
    fourier_freqs: int = 4,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mode = mode.lower()
    x_cpu, y_cpu = _group_local_xy(group_ids)
    device = device or group_ids.device
    x = x_cpu.to(device=device, dtype=dtype)
    y = y_cpu.to(device=device, dtype=dtype)
    cols = [torch.ones_like(x), x, y]

    if mode == "linear":
        pass
    elif mode == "poly2":
        cols.extend([x * x, y * y, x * y])
    elif mode in {"fourier", "local_fourier"}:
        pi2 = 2.0 * torch.pi
        for freq in range(1, int(fourier_freqs) + 1):
            f = float(freq)
            cols.extend([
                torch.sin(pi2 * f * x),
                torch.cos(pi2 * f * x),
                torch.sin(pi2 * f * y),
                torch.cos(pi2 * f * y),
            ])
    else:
        raise ValueError(f"Unsupported position_residualization mode: {mode}")
    return torch.stack(cols, dim=1)


def fit_position_residualizer(data: TokenData, train_cfg: Any) -> Optional[PositionResidualizer]:
    mode = position_residualization_mode(train_cfg)
    if mode == "none":
        return None

    features, _labels, valid_mask, group_ids = data
    keep = valid_mask.bool()
    x = features[keep].float().cpu()
    basis = build_position_basis(
        group_ids[keep].cpu(),
        mode=mode,
        fourier_freqs=int(getattr(train_cfg, "position_residualization_fourier_freqs", 4)),
        dtype=torch.float32,
    ).cpu()
    ridge = float(getattr(train_cfg, "position_residualization_ridge", 1e-3))
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    eye[0, 0] = 0.0
    coef = torch.linalg.solve(gram + ridge * eye, basis.T @ x)
    return PositionResidualizer(
        mode=mode,
        fourier_freqs=int(getattr(train_cfg, "position_residualization_fourier_freqs", 4)),
        ridge=ridge,
        coef=coef.cpu(),
        basis_dim=int(basis.shape[1]),
    )


def apply_position_residualizer(data: Optional[TokenData], residualizer: Optional[PositionResidualizer]) -> Optional[TokenData]:
    if data is None or residualizer is None:
        return data
    features, labels, valid_mask, group_ids = data
    basis = build_position_basis(
        group_ids,
        mode=residualizer.mode,
        fourier_freqs=residualizer.fourier_freqs,
        device=features.device,
        dtype=torch.float32,
    )
    residual = basis @ residualizer.coef.to(device=features.device, dtype=torch.float32)
    return features.float() - residual, labels, valid_mask, group_ids


def apply_position_residualization_to_splits(
    *,
    train_data: TokenData,
    val_data: TokenData,
    train_cfg: Any,
    test_data: Optional[TokenData] = None,
    named_test_data: Optional[Dict[str, TokenData]] = None,
) -> Tuple[TokenData, TokenData, Optional[TokenData], Optional[Dict[str, TokenData]], Optional[Dict[str, object]], Optional[PositionResidualizer]]:
    residualizer = fit_position_residualizer(train_data, train_cfg)
    if residualizer is None:
        return train_data, val_data, test_data, named_test_data, None, None

    transformed_named = None
    if named_test_data is not None:
        transformed_named = {
            name: apply_position_residualizer(split, residualizer)  # type: ignore[arg-type]
            for name, split in named_test_data.items()
        }
    return (
        apply_position_residualizer(train_data, residualizer),  # type: ignore[return-value]
        apply_position_residualizer(val_data, residualizer),  # type: ignore[return-value]
        apply_position_residualizer(test_data, residualizer),
        transformed_named,
        position_residualization_metadata(residualizer),
        residualizer,
    )
