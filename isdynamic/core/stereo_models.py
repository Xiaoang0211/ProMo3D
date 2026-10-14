"""PMD stereo readouts informed by Li et al.'s pairwise object-binding probes.

Methodological reference: https://github.com/liyihao0302/vit-object-binding
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(-1)


class PairwiseLinearProbe(nn.Module):
    is_pairwise = True

    def __init__(self, input_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.w(x).squeeze(-1) + self.w(y).squeeze(-1) + self.b


class PairwiseDiagonalQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, input_dim: int):
        super().__init__()
        self.diag = nn.Parameter(torch.zeros(input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.scale = input_dim ** 0.5

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        score = (x * y * self.diag).sum(dim=-1) / self.scale
        return score + self.b


class PairwiseQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, input_dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.scale = input_dim ** 0.5
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = 0.5 * (self.a + self.a.transpose(0, 1)) / self.scale
        score = torch.einsum("bi,ij,bj->b", x, w, y)
        return score + self.b


class PairwiseLowRankQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, input_dim: int, rank: int = 256):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(rank, input_dim))
        self.v = nn.Parameter(torch.zeros(rank, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.scale = input_dim ** 0.5
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = 0.5 * (self.u.transpose(0, 1) @ self.v + self.v.transpose(0, 1) @ self.u) / self.scale
        score = torch.einsum("bi,ij,bj->b", x, w, y)
        return score + self.b


class PairwiseStrictLowRankQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, input_dim: int, rank: int = 256):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(rank, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        sign = torch.ones(rank, dtype=torch.float32)
        sign[1::2] = -1.0
        self.register_buffer("sign", sign)
        nn.init.normal_(self.u, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_proj = x @ self.u.transpose(0, 1)
        y_proj = y @ self.u.transpose(0, 1)
        score = (x_proj * y_proj * self.sign).sum(dim=-1)
        return score + self.b


class PairwiseAttentionProbe(nn.Module):
    is_pairwise = True

    def __init__(self, attention_matrix: torch.Tensor):
        super().__init__()
        if attention_matrix.ndim != 2 or attention_matrix.shape[0] != attention_matrix.shape[1]:
            raise ValueError(
                "attention_matrix must be square [C, C], got "
                f"{tuple(attention_matrix.shape)}"
            )
        self.register_buffer("attention_matrix", attention_matrix.float())
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        score = -torch.einsum("bi,ij,bj->b", x, self.attention_matrix, y)
        return score + self.b


class PairwiseValueProbe(nn.Module):
    is_pairwise = True

    def __init__(self, value_matrix: torch.Tensor):
        super().__init__()
        if value_matrix.ndim != 2 or value_matrix.shape[0] != value_matrix.shape[1]:
            raise ValueError(
                "value_matrix must be square [C, C], got "
                f"{tuple(value_matrix.shape)}"
            )
        self.register_buffer("value_matrix", value_matrix.float())
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        score = -torch.einsum("bi,ij,bj->b", x, self.value_matrix, y)
        return score + self.b


class PairwiseMatrixBiasedQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, fixed_matrix: torch.Tensor):
        super().__init__()
        if fixed_matrix.ndim != 2 or fixed_matrix.shape[0] != fixed_matrix.shape[1]:
            raise ValueError(
                "fixed_matrix must be square [C, C], got "
                f"{tuple(fixed_matrix.shape)}"
            )
        input_dim = int(fixed_matrix.shape[0])
        self.register_buffer("fixed_matrix", fixed_matrix.float())
        self.a = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.scale = input_dim ** 0.5
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,ij,bj->b", x, self.fixed_matrix, y)
        w = 0.5 * (self.a + self.a.transpose(0, 1)) / self.scale
        residual = torch.einsum("bi,ij,bj->b", x, w, y)
        return -(base + residual) + self.b


class PairwiseAttentionBiasedQuadraticProbe(PairwiseMatrixBiasedQuadraticProbe):
    pass


class PairwiseValueBiasedQuadraticProbe(PairwiseMatrixBiasedQuadraticProbe):
    pass


class PairwiseMatrixBiasedLowRankQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, fixed_matrix: torch.Tensor, rank: int = 256):
        super().__init__()
        if fixed_matrix.ndim != 2 or fixed_matrix.shape[0] != fixed_matrix.shape[1]:
            raise ValueError(
                "fixed_matrix must be square [C, C], got "
                f"{tuple(fixed_matrix.shape)}"
            )
        input_dim = int(fixed_matrix.shape[0])
        self.register_buffer("fixed_matrix", fixed_matrix.float())
        self.u = nn.Parameter(torch.zeros(rank, input_dim))
        self.v = nn.Parameter(torch.zeros(rank, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.scale = input_dim ** 0.5
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,ij,bj->b", x, self.fixed_matrix, y)
        w = 0.5 * (self.u.transpose(0, 1) @ self.v + self.v.transpose(0, 1) @ self.u) / self.scale
        residual = torch.einsum("bi,ij,bj->b", x, w, y)
        return -(base + residual) + self.b


class PairwiseAttentionBiasedLowRankQuadraticProbe(PairwiseMatrixBiasedLowRankQuadraticProbe):
    pass


class PairwiseValueBiasedLowRankQuadraticProbe(PairwiseMatrixBiasedLowRankQuadraticProbe):
    pass


class PairwiseMatrixBiasedStrictLowRankQuadraticProbe(nn.Module):
    is_pairwise = True

    def __init__(self, fixed_matrix: torch.Tensor, rank: int = 256):
        super().__init__()
        if fixed_matrix.ndim != 2 or fixed_matrix.shape[0] != fixed_matrix.shape[1]:
            raise ValueError(
                "fixed_matrix must be square [C, C], got "
                f"{tuple(fixed_matrix.shape)}"
            )
        input_dim = int(fixed_matrix.shape[0])
        self.register_buffer("fixed_matrix", fixed_matrix.float())
        self.u = nn.Parameter(torch.zeros(rank, input_dim))
        self.b = nn.Parameter(torch.zeros(1))
        sign = torch.ones(rank, dtype=torch.float32)
        sign[1::2] = -1.0
        self.register_buffer("sign", sign)
        nn.init.normal_(self.u, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,ij,bj->b", x, self.fixed_matrix, y)
        x_proj = x @ self.u.transpose(0, 1)
        y_proj = y @ self.u.transpose(0, 1)
        residual = (x_proj * y_proj * self.sign).sum(dim=-1)
        return -(base + residual) + self.b


class PairwiseAttentionBiasedStrictLowRankQuadraticProbe(
    PairwiseMatrixBiasedStrictLowRankQuadraticProbe
):
    pass


class PairwiseValueBiasedStrictLowRankQuadraticProbe(
    PairwiseMatrixBiasedStrictLowRankQuadraticProbe
):
    pass


def load_attention_matrix(path: str, key: Optional[str] = None) -> torch.Tensor:
    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Attention matrix file not found: {matrix_path}")

    suffix = matrix_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(matrix_path, allow_pickle=False)
    elif suffix == ".npz":
        npz = np.load(matrix_path, allow_pickle=False)
        use_key = key if key is not None else list(npz.keys())[0]
        if use_key not in npz:
            raise KeyError(f"Key '{use_key}' not found in {matrix_path}")
        arr = npz[use_key]
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(matrix_path, map_location="cpu")
        if isinstance(obj, dict):
            use_key = key if key is not None else next(iter(obj.keys()))
            if use_key not in obj:
                raise KeyError(f"Key '{use_key}' not found in {matrix_path}")
            arr = obj[use_key]
        else:
            arr = obj
        if isinstance(arr, torch.Tensor):
            return arr.float()
        arr = np.asarray(arr)
    else:
        raise ValueError(f"Unsupported attention matrix format: {suffix}")

    return torch.from_numpy(np.asarray(arr)).float()


def build_probe(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    rank: int = 256,
    attention_matrix: Optional[torch.Tensor] = None,
) -> nn.Module:
    if model_type == "linear":
        return LinearProbe(input_dim)
    if model_type == "mlp":
        return MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == "pair_linear":
        return PairwiseLinearProbe(input_dim=input_dim)
    if model_type == "pair_diag_quadratic":
        return PairwiseDiagonalQuadraticProbe(input_dim=input_dim)
    if model_type == "pair_quadratic":
        return PairwiseQuadraticProbe(input_dim=input_dim)
    if model_type == "pair_lowrank_quadratic":
        return PairwiseLowRankQuadraticProbe(input_dim=input_dim, rank=rank)
    if model_type == "pair_lowrank_quadratic_strict":
        return PairwiseStrictLowRankQuadraticProbe(input_dim=input_dim, rank=rank)
    if model_type == "pair_attention":
        if attention_matrix is None:
            raise ValueError("pair_attention requires attention_matrix.")
        return PairwiseAttentionProbe(attention_matrix=attention_matrix)
    if model_type == "pair_value":
        if attention_matrix is None:
            raise ValueError("pair_value requires a fixed matrix.")
        return PairwiseValueProbe(value_matrix=attention_matrix)
    if model_type == "pair_attention_biased_quadratic":
        if attention_matrix is None:
            raise ValueError("pair_attention_biased_quadratic requires attention_matrix.")
        return PairwiseAttentionBiasedQuadraticProbe(fixed_matrix=attention_matrix)
    if model_type == "pair_value_biased_quadratic":
        if attention_matrix is None:
            raise ValueError("pair_value_biased_quadratic requires a fixed matrix.")
        return PairwiseValueBiasedQuadraticProbe(fixed_matrix=attention_matrix)
    if model_type == "pair_attention_biased_lowrank_quadratic":
        if attention_matrix is None:
            raise ValueError("pair_attention_biased_lowrank_quadratic requires attention_matrix.")
        return PairwiseAttentionBiasedLowRankQuadraticProbe(fixed_matrix=attention_matrix, rank=rank)
    if model_type == "pair_value_biased_lowrank_quadratic":
        if attention_matrix is None:
            raise ValueError("pair_value_biased_lowrank_quadratic requires a fixed matrix.")
        return PairwiseValueBiasedLowRankQuadraticProbe(fixed_matrix=attention_matrix, rank=rank)
    if model_type == "pair_attention_biased_lowrank_quadratic_strict":
        if attention_matrix is None:
            raise ValueError("pair_attention_biased_lowrank_quadratic_strict requires attention_matrix.")
        return PairwiseAttentionBiasedStrictLowRankQuadraticProbe(fixed_matrix=attention_matrix, rank=rank)
    if model_type == "pair_value_biased_lowrank_quadratic_strict":
        if attention_matrix is None:
            raise ValueError("pair_value_biased_lowrank_quadratic_strict requires a fixed matrix.")
        return PairwiseValueBiasedStrictLowRankQuadraticProbe(fixed_matrix=attention_matrix, rank=rank)
    raise ValueError(f"Unknown probe model_type: {model_type}")
