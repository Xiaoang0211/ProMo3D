from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    Keep token-major format [B, S, P, C].
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [B,S,P,C], got {x.shape}")
    return x


def map_to_patch_tokens(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert dense map to token grid by average pooling.

    Args:
        x: [B, S, C, H, W]
        patch_size: patch side.

    Returns:
        [B, S, P, C] where P = (H/patch_size) * (W/patch_size)
    """
    if x.dim() != 5:
        raise ValueError(f"Expected [B,S,C,H,W], got {x.shape}")
    b, s, c, h, w = x.shape
    pooled = F.avg_pool2d(
        x.view(b * s, c, h, w),
        kernel_size=patch_size,
        stride=patch_size,
    )
    pooled = pooled.view(b, s, c, pooled.shape[-2], pooled.shape[-1])
    pooled = pooled.permute(0, 1, 3, 4, 2).reshape(b, s, -1, c)
    return pooled


def collect_aggregator_patch_tokens(
    aggregated_tokens_list: List[torch.Tensor],
    patch_start_idx: int,
    layer_indices: Optional[Iterable[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract patch tokens from selected aggregator layers.
    """
    if layer_indices is None:
        layer_indices = range(len(aggregated_tokens_list))

    outputs: Dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        tokens = aggregated_tokens_list[idx][:, :, patch_start_idx:]
        outputs[f"agg_layer_{idx}"] = _flatten_tokens(tokens)
    return outputs


def collect_dpt_input_tokens(
    aggregated_tokens_list: List[torch.Tensor],
    patch_start_idx: int,
    intermediate_layer_idx: Iterable[int],
) -> Dict[str, torch.Tensor]:
    """
    Tokens consumed by DPT heads for each selected layer.
    """
    outputs: Dict[str, torch.Tensor] = {}
    for idx in intermediate_layer_idx:
        tokens = aggregated_tokens_list[idx][:, :, patch_start_idx:]
        outputs[f"dpt_in_layer_{idx}"] = _flatten_tokens(tokens)
    return outputs


def collect_confidence_tokens(
    predictions: Dict[str, torch.Tensor],
    patch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert depth/point confidence maps into token-aligned scalar features.
    """
    outputs: Dict[str, torch.Tensor] = {}

    if "depth_conf" in predictions:
        depth_conf = predictions["depth_conf"]  # [B,S,H,W]
        depth_conf = depth_conf.unsqueeze(2)  # [B,S,1,H,W]
        outputs["depth_conf_tokens"] = map_to_patch_tokens(depth_conf, patch_size=patch_size)

    if "world_points_conf" in predictions:
        point_conf = predictions["world_points_conf"]  # [B,S,H,W]
        point_conf = point_conf.unsqueeze(2)  # [B,S,1,H,W]
        outputs["point_conf_tokens"] = map_to_patch_tokens(point_conf, patch_size=patch_size)

    return outputs


@torch.no_grad()
def collect_track_feature_tokens(
    model,
    aggregated_tokens_list: List[torch.Tensor],
    images: torch.Tensor,
    patch_start_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract tokenized track features from track head DPT feature extractor.
    """
    if model.track_head is None:
        return {}

    fmap = model.track_head.feature_extractor(
        aggregated_tokens_list=aggregated_tokens_list,
        images=images,
        patch_start_idx=patch_start_idx,
    )  # [B,S,C,H',W']
    return {"track_feature_tokens": map_to_patch_tokens(fmap, patch_size=1)}

