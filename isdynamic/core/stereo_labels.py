from typing import Tuple

import torch
import torch.nn.functional as F


def mask_to_token_dynamic_ratio(dynamic_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if dynamic_mask.dim() == 3:
        dynamic_mask = dynamic_mask.unsqueeze(1)
        squeeze_seq = True
    elif dynamic_mask.dim() == 4:
        squeeze_seq = False
    else:
        raise ValueError(f"Expected dynamic_mask dim 3 or 4, got {dynamic_mask.shape}")

    mask = dynamic_mask.float().clamp(0.0, 1.0)
    b, s, h, w = mask.shape
    mask = mask.view(b * s, 1, h, w)
    ratio = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
    ratio = ratio.view(b, s, ratio.shape[-2], ratio.shape[-1])
    return ratio.squeeze(1) if squeeze_seq else ratio


def dynamic_ratio_to_binary_labels(
    dynamic_ratio: torch.Tensor,
    threshold: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = (dynamic_ratio >= threshold).to(torch.long)
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    return labels, valid_mask
