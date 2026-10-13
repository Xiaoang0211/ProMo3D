from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


TokenData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def apply_train_label_control(data: TokenData, mode: str, seed: int) -> TokenData:
    mode = str(mode).strip().lower()
    if mode in {"", "none"}:
        return data
    if mode != "within_group_shuffle":
        raise ValueError(f"Unsupported train_label_control: {mode}")

    features, labels, valid_mask, group_ids = data
    shuffled = labels.clone()
    rng = np.random.default_rng(int(seed))

    valid_indices = torch.nonzero(valid_mask.bool(), as_tuple=False).flatten().cpu().numpy()
    valid_groups = group_ids[valid_indices].cpu().numpy()
    order = np.argsort(valid_groups, kind="stable")
    sorted_indices = valid_indices[order]
    sorted_groups = valid_groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1

    for indices_np in np.split(sorted_indices, boundaries):
        if indices_np.size < 2:
            continue
        indices = torch.from_numpy(indices_np).long()
        permutation = torch.from_numpy(rng.permutation(indices_np.size)).long()
        shuffled[indices] = labels[indices[permutation]]

    return features, shuffled, valid_mask, group_ids
