from pathlib import Path
from typing import Dict, List, Tuple

import torch

from mapanything.models import MapAnything, init_model_from_config
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.prediction_heads.base import (
    PredictionHeadInput,
    PredictionHeadTokenInput,
)


SOURCE_CHOICES = [
    "alt_attn",
    "inter_frame_block",
    "intra_frame_block",
    "dense_head",
    "pose_head",
    "scale_head",
]


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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        out[key[len("module.") :] if key.startswith("module.") else key] = value
    return out


def load_mapanything(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "mapanything",
    machine: str = "aws",
) -> torch.nn.Module:
    checkpoint = Path(checkpoint_path)

    # Support local Hugging Face-style directories such as ckpt/{config.json,model.safetensors}.
    if checkpoint.is_dir():
        model = MapAnything.from_pretrained(str(checkpoint))
        model = model.to(device)
        model.eval()
        return model

    model = init_model_from_config(model_name=model_name, device="cpu", machine=machine)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def normalize_token_sources(token_sources: List[str]) -> List[str]:
    expanded: List[str] = []
    for source in token_sources:
        if source == "alt_attn":
            expanded.extend(["inter_frame_block", "intra_frame_block"])
        else:
            expanded.append(source)

    normalized: List[str] = []
    seen = set()
    for source in expanded:
        if source not in seen:
            normalized.append(source)
            seen.add(source)
    return normalized


def build_feature_key_order(model, token_sources: List[str]) -> List[str]:
    token_sources = normalize_token_sources(token_sources)
    keys: List[str] = []

    if "inter_frame_block" in token_sources:
        for i, _depth_idx in enumerate(range(0, model.info_sharing.depth, 2)):
            keys.append(f"inter_frame_block_{i}")

    if "intra_frame_block" in token_sources:
        for i, _depth_idx in enumerate(range(1, model.info_sharing.depth, 2)):
            keys.append(f"intra_frame_block_{i}")

    if "dense_head" in token_sources:
        dense_blocks = 4 if "dpt" in model.pred_head_type else 1
        for i in range(dense_blocks):
            keys.append(f"dense_head_{i}")

    if "pose_head" in token_sources:
        if not hasattr(model, "pose_head"):
            raise RuntimeError("pose_head requested but the model has no pose head.")
        for i in range(model.pose_head.num_resconv_block):
            keys.append(f"pose_head_{i}")

    if "scale_head" in token_sources:
        for i in range(model.scale_head.num_mlp_layers):
            keys.append(f"scale_head_{i}")

    return keys


def _stacked_bchw_to_tokens(x: torch.Tensor, num_views: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected stacked BCHW tensor, got {tuple(x.shape)}")
    batch_total, channels, height, width = x.shape
    if batch_total % num_views != 0:
        raise ValueError(
            f"Batch {batch_total} is not divisible by num_views={num_views}"
        )
    batch_size = batch_total // num_views
    x = x.view(batch_size, num_views, channels, height, width)
    x = x.permute(0, 1, 3, 4, 2).reshape(batch_size, num_views, height * width, channels)
    return x.contiguous()


def _summary_to_tokens(
    x: torch.Tensor,
    num_views: int,
    tokens_per_view: int,
) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[2] != 1:
            raise ValueError(f"Expected [B,C,1], got {tuple(x.shape)}")
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected summary feature [B,C], got {tuple(x.shape)}")
    batch_size, channels = x.shape
    x = x[:, None, None, :].expand(batch_size, num_views, tokens_per_view, channels)
    return x.contiguous()


def _extract_alt_attn_tokens(
    block_output: torch.Tensor,
    batch_size: int,
    num_views: int,
    spatial_tokens_per_view: int,
    extra_tokens_per_view: int,
    global_tokens: int,
) -> torch.Tensor:
    if block_output.ndim != 3:
        raise ValueError(f"Expected [B,L,C] transformer output, got {tuple(block_output.shape)}")
    _, _, channels = block_output.shape
    tokens_per_view = spatial_tokens_per_view + extra_tokens_per_view

    # Even alternating-attention layers run global attention and keep shape [B, V*T + G, C].
    if block_output.shape[0] == batch_size:
        if block_output.shape[1] < num_views * tokens_per_view:
            raise ValueError(
                "Global attention output is shorter than expected for the per-view token payload. "
                f"Got shape {tuple(block_output.shape)}, expected at least {num_views * tokens_per_view} tokens."
            )
        view_tokens = block_output[:, : num_views * tokens_per_view, :]
        view_tokens = view_tokens.reshape(batch_size, num_views, tokens_per_view, channels)
        view_tokens = view_tokens[:, :, :spatial_tokens_per_view, :]
        return view_tokens.contiguous()

    # Odd alternating-attention layers run per-frame attention and keep shape [B*V, T, C].
    if block_output.shape[0] == batch_size * num_views:
        if block_output.shape[1] != tokens_per_view:
            raise ValueError(
                "Per-frame attention output token count does not match the expected per-view token count: "
                f"got {block_output.shape[1]}, expected {tokens_per_view}."
            )
        view_tokens = block_output.reshape(batch_size, num_views, tokens_per_view, channels)
        view_tokens = view_tokens[:, :, :spatial_tokens_per_view, :]
        return view_tokens.contiguous()

    raise ValueError(
        "Unexpected alternating-attention output shape. "
        f"Got {tuple(block_output.shape)} for batch_size={batch_size}, num_views={num_views}."
    )


@torch.no_grad()
def extract_feature_map(model, views: List[Dict[str, torch.Tensor]], token_sources: List[str]) -> Dict[str, torch.Tensor]:
    token_sources = normalize_token_sources(token_sources)
    batch_size_per_view, _, img_height, img_width = views[0]["img"].shape
    num_views = len(views)

    encoder_features, encoder_registers = model._encode_n_views(views)
    with torch.autocast("cuda", enabled=False):
        encoder_features = model._encode_and_fuse_optional_geometric_inputs(
            views, encoder_features
        )

    spatial_height = int(encoder_features[0].shape[-2])
    spatial_width = int(encoder_features[0].shape[-1])
    spatial_tokens_per_view = spatial_height * spatial_width
    extra_tokens_per_view = 0
    if encoder_registers is not None and len(encoder_registers) > 0:
        extra_tokens_per_view = int(encoder_registers[0].shape[-1])
    global_tokens = 1

    input_scale_token = (
        model.scale_token.unsqueeze(0).unsqueeze(-1).repeat(batch_size_per_view, 1, 1)
    )
    info_sharing_input = MultiViewTransformerInput(
        features=encoder_features,
        additional_input_tokens_per_view=encoder_registers,
        additional_input_tokens=input_scale_token,
    )

    alt_attn_outputs: Dict[int, torch.Tensor] = {}
    pose_outputs: Dict[int, torch.Tensor] = {}
    scale_outputs: Dict[int, torch.Tensor] = {}
    hook_handles = []

    if "inter_frame_block" in token_sources or "intra_frame_block" in token_sources:
        for idx, block in enumerate(model.info_sharing.self_attention_blocks):
            hook_handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output, block_idx=idx: alt_attn_outputs.__setitem__(
                        block_idx, output.detach()
                    )
                )
            )

    if "pose_head" in token_sources and hasattr(model, "pose_head"):
        for idx, block in enumerate(model.pose_head.res_conv):
            hook_handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output, block_idx=idx: pose_outputs.__setitem__(
                        block_idx, output.detach()
                    )
                )
            )

    if "scale_head" in token_sources:
        for idx, layer in enumerate(model.scale_head.mlp):
            hook_handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, block_idx=idx: scale_outputs.__setitem__(
                        block_idx, output.detach()
                    )
                )
            )

    try:
        if model.info_sharing_return_type == "no_intermediate_features":
            final_info = model.info_sharing(info_sharing_input)
            intermediate_info = None
        else:
            final_info, intermediate_info = model.info_sharing(info_sharing_input)

        dense_head_inputs: List[torch.Tensor]
        if model.pred_head_type == "linear":
            dense_head_inputs = [torch.cat(final_info.features, dim=0)]
        elif model.pred_head_type in ["dpt", "dpt+pose"]:
            dense_head_inputs = []
            if model.use_encoder_features_for_dpt:
                dense_head_inputs.append(torch.cat(encoder_features, dim=0))
                dense_head_inputs.append(torch.cat(intermediate_info[0].features, dim=0))
                dense_head_inputs.append(torch.cat(intermediate_info[1].features, dim=0))
                dense_head_inputs.append(torch.cat(final_info.features, dim=0))
            else:
                dense_head_inputs.append(torch.cat(intermediate_info[0].features, dim=0))
                dense_head_inputs.append(torch.cat(intermediate_info[1].features, dim=0))
                dense_head_inputs.append(torch.cat(intermediate_info[2].features, dim=0))
                dense_head_inputs.append(torch.cat(final_info.features, dim=0))
        else:
            raise ValueError(f"Unsupported pred_head_type: {model.pred_head_type}")

        if "pose_head" in token_sources and hasattr(model, "pose_head"):
            model.pose_head(PredictionHeadInput(last_feature=dense_head_inputs[-1]))
        if "scale_head" in token_sources:
            model.scale_head(
                PredictionHeadTokenInput(last_feature=final_info.additional_token_features)
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    feature_map: Dict[str, torch.Tensor] = {}

    if "inter_frame_block" in token_sources or "intra_frame_block" in token_sources:
        glob_idx = 0
        perframe_idx = 0
        for idx in range(model.info_sharing.depth):
            tokens = _extract_alt_attn_tokens(
                alt_attn_outputs[idx],
                batch_size=batch_size_per_view,
                num_views=num_views,
                spatial_tokens_per_view=spatial_tokens_per_view,
                extra_tokens_per_view=extra_tokens_per_view,
                global_tokens=global_tokens,
            )
            if idx % 2 == 0:
                if "inter_frame_block" in token_sources:
                    feature_map[f"inter_frame_block_{glob_idx}"] = tokens
                glob_idx += 1
            else:
                if "intra_frame_block" in token_sources:
                    feature_map[f"intra_frame_block_{perframe_idx}"] = tokens
                perframe_idx += 1

    if "dense_head" in token_sources:
        for idx, dense_block in enumerate(dense_head_inputs):
            feature_map[f"dense_head_{idx}"] = _stacked_bchw_to_tokens(
                dense_block.detach(),
                num_views=num_views,
            )

    if "pose_head" in token_sources and hasattr(model, "pose_head"):
        for idx in range(model.pose_head.num_resconv_block):
            feature_map[f"pose_head_{idx}"] = _stacked_bchw_to_tokens(
                pose_outputs[idx],
                num_views=num_views,
            )

    if "scale_head" in token_sources:
        for idx in range(model.scale_head.num_mlp_layers):
            scale_block = scale_outputs[idx]
            if scale_block.ndim != 3 or scale_block.shape[1] != 1:
                raise ValueError(
                    f"Expected scale head block output [B,1,C], got {tuple(scale_block.shape)}"
                )
            scale_block = scale_block[:, 0, :]
            feature_map[f"scale_head_{idx}"] = _summary_to_tokens(
                scale_block.detach(),
                num_views=num_views,
                tokens_per_view=spatial_tokens_per_view,
            )

    return feature_map
