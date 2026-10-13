import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

_PI3_REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "Pi3"
if str(_PI3_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI3_REPO_ROOT))

_existing_pi3 = sys.modules.get("pi3")
if _existing_pi3 is not None:
    module_file = str(getattr(_existing_pi3, "__file__", "") or "")
    module_paths = [str(p) for p in getattr(_existing_pi3, "__path__", [])]
    expected_pkg = str((_PI3_REPO_ROOT / "pi3").resolve())
    if expected_pkg not in module_file and expected_pkg not in module_paths:
        sys.modules.pop("pi3", None)
        sys.modules.pop("pi3.models", None)
        sys.modules.pop("pi3.models.pi3", None)
        sys.modules.pop("pi3.models.pi3x", None)

try:
    from pi3.models.pi3 import Pi3
    from pi3.models.pi3x import Pi3X
except ModuleNotFoundError:
    import types

    pkg_root = (_PI3_REPO_ROOT / "pi3").resolve()
    pkg = types.ModuleType("pi3")
    pkg.__path__ = [str(pkg_root)]
    sys.modules["pi3"] = pkg

    models_pkg = types.ModuleType("pi3.models")
    models_pkg.__path__ = [str(pkg_root / "models")]
    sys.modules["pi3.models"] = models_pkg

    from pi3.models.pi3 import Pi3
    from pi3.models.pi3x import Pi3X


PI3_PATCH_SIZE = 14
PI3_TARGET_SIZE = 518
SOURCE_CHOICES = ["alt_attn", "image_encoder", "cam_head", "conf_head", "point_head"]
MODEL_CHOICES = ["pi3", "pi3x"]
GROUP_CHOICES = ["inter_frame_block", "intra_frame_block", "image_encoder", "cam_head", "conf_head", "point_head"]


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


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def load_checkpoint_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors checkpoints.") from exc
        state_dict = load_safetensors(str(checkpoint_path), device="cpu")
        return _strip_known_prefixes(state_dict)

    obj = torch.load(checkpoint_path, map_location="cpu")
    return _strip_known_prefixes(_extract_state_dict(obj))


def load_pi3_family_model(
    model_name: str,
    checkpoint: Path,
    device: torch.device,
    use_multimodal_pi3x: bool = False,
) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "pi3":
        model = Pi3()
    elif model_name == "pi3x":
        model = Pi3X(use_multimodal=use_multimodal_pi3x)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state_dict = load_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _run_decoder_blocks(
    decoder: torch.nn.Module,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    bsz: int,
    n_frames: int,
    token_len: int,
    patch_start_idx: int,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    x = decoder.projects(hidden)
    out: Dict[str, torch.Tensor] = {}
    for i, blk in enumerate(decoder.blocks):
        x = blk(x, xpos=pos)
        out[f"{prefix}_{i}"] = x.reshape(bsz, n_frames, token_len, -1)[:, :, patch_start_idx:]
    return out


def _extract_main_decoder_outputs(
    model_name: str,
    model: torch.nn.Module,
    images_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    bsz, n_frames, _, h, w = images_b.shape
    normalized = (images_b - model.image_mean) / model.image_std

    if model_name == "pi3":
        hidden = model.encoder(normalized.reshape(bsz * n_frames, 3, h, w), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        return model.decode(hidden, n_frames, h, w, return_split=True)

    hidden, poses, _, use_pose_mask, _ = model.encode(
        normalized,
        with_prior=False,
        depths=None,
        rays=None,
        intrinsics=None,
        poses=None,
        mask_add_depth=None,
        mask_add_ray=None,
        mask_add_pose=None,
    )
    hidden = hidden.reshape(bsz, n_frames, -1, model.dec_embed_dim)
    return model.decode(hidden, n_frames, h, w, poses, use_pose_mask, return_split=True)


def _extract_image_encoder_outputs(
    model: torch.nn.Module,
    images_b: torch.Tensor,
) -> List[torch.Tensor]:
    bsz, n_frames, _, h, w = images_b.shape
    normalized = (images_b - model.image_mean) / model.image_std
    outputs = model.encoder.get_intermediate_layers(
        normalized.reshape(bsz * n_frames, 3, h, w),
        n=list(range(model.encoder.n_blocks)),
        reshape=False,
        return_class_token=False,
        norm=True,
    )
    return [out.reshape(bsz, n_frames, out.shape[1], out.shape[2]) for out in outputs]


@torch.no_grad()
def extract_feature_map(
    model_name: str,
    model: torch.nn.Module,
    images_b: torch.Tensor,
    token_sources: Iterable[str],
) -> Dict[str, torch.Tensor]:
    token_sources = list(token_sources)
    feature_map: Dict[str, torch.Tensor] = {}

    if "image_encoder" in token_sources:
        for i, out in enumerate(_extract_image_encoder_outputs(model, images_b)):
            feature_map[f"image_encoder_{i}"] = out

    if not any(source in token_sources for source in ("alt_attn", "cam_head", "conf_head", "point_head")):
        return feature_map

    concat_hidden, pos, frame_outputs, global_outputs = _extract_main_decoder_outputs(model_name, model, images_b)

    bsz, n_frames, token_len = images_b.shape[0], images_b.shape[1], concat_hidden.shape[1]
    patch_start_idx = model.patch_start_idx

    if "alt_attn" in token_sources:
        for i, out in enumerate(global_outputs):
            feature_map[f"inter_frame_block_{i}"] = out[:, :, patch_start_idx:]
        for i, out in enumerate(frame_outputs):
            feature_map[f"intra_frame_block_{i}"] = out[:, :, patch_start_idx:]

    if "cam_head" in token_sources:
        feature_map.update(
            _run_decoder_blocks(
                decoder=model.camera_decoder,
                hidden=concat_hidden,
                pos=pos,
                bsz=bsz,
                n_frames=n_frames,
                token_len=token_len,
                patch_start_idx=patch_start_idx,
                prefix="cam_head",
            )
        )

    if "conf_head" in token_sources:
        feature_map.update(
            _run_decoder_blocks(
                decoder=model.conf_decoder,
                hidden=concat_hidden,
                pos=pos,
                bsz=bsz,
                n_frames=n_frames,
                token_len=token_len,
                patch_start_idx=patch_start_idx,
                prefix="conf_head",
            )
        )

    if "point_head" in token_sources:
        feature_map.update(
            _run_decoder_blocks(
                decoder=model.point_decoder,
                hidden=concat_hidden,
                pos=pos,
                bsz=bsz,
                n_frames=n_frames,
                token_len=token_len,
                patch_start_idx=patch_start_idx,
                prefix="point_head",
            )
        )

    return feature_map


def build_feature_key_order(
    model: torch.nn.Module,
    token_sources: Iterable[str],
) -> List[str]:
    token_sources = list(token_sources)
    keys: List[str] = []

    if "alt_attn" in token_sources:
        num_global = sum(1 for i in range(len(model.decoder)) if i % 2 == 1)
        num_frame = sum(1 for i in range(len(model.decoder)) if i % 2 == 0)
        for i in range(num_global):
            keys.append(f"inter_frame_block_{i}")
        for i in range(num_frame):
            keys.append(f"intra_frame_block_{i}")

    if "image_encoder" in token_sources:
        for i in range(model.encoder.n_blocks):
            keys.append(f"image_encoder_{i}")

    if "cam_head" in token_sources:
        for i in range(len(model.camera_decoder.blocks)):
            keys.append(f"cam_head_{i}")

    if "conf_head" in token_sources:
        for i in range(len(model.conf_decoder.blocks)):
            keys.append(f"conf_head_{i}")

    if "point_head" in token_sources:
        for i in range(len(model.point_decoder.blocks)):
            keys.append(f"point_head_{i}")

    return keys


def extract_matrix_from_block(block: torch.nn.Module, mode: str) -> torch.Tensor:
    qkv = block.attn.qkv.weight.detach().float()
    c_dim = qkv.shape[1]
    if qkv.shape[0] != 3 * c_dim:
        raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")

    wq, wk, wv = qkv.chunk(3, dim=0)
    if mode == "attention":
        return wq.transpose(0, 1) @ wk
    if mode == "value":
        return wv
    raise ValueError(f"Unsupported mode: {mode}")


def blockdiag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(a.shape[0] + b.shape[0], a.shape[1] + b.shape[1], dtype=a.dtype)
    out[: a.shape[0], : a.shape[1]] = a
    out[a.shape[0] :, a.shape[1] :] = b
    return out


def _resolve_projection_block_idx(
    feature_block_idx: int,
    projection_block_idx: Optional[int],
    depth: int,
    fallback_current_for_last: bool,
) -> int:
    if projection_block_idx is None:
        idx = feature_block_idx + 1
    else:
        idx = projection_block_idx

    if idx >= depth:
        if fallback_current_for_last:
            idx = feature_block_idx
        else:
            raise ValueError(
                f"Projection block index {idx} is out of range for depth {depth}. "
                "Use --fallback_current_for_last or provide --projection_block_idx."
            )
    if idx < 0:
        raise ValueError(f"Invalid projection block index: {idx}")
    return idx


def get_projection_matrix(
    model: torch.nn.Module,
    group: str,
    feature_block_idx: int,
    mode: str,
    projection_block_idx: Optional[int] = None,
    fallback_current_for_last: bool = False,
) -> Tuple[torch.Tensor, int]:
    if group == "inter_frame_block":
        blocks = [model.decoder[i] for i in range(len(model.decoder)) if i % 2 == 1]
        proj_idx = _resolve_projection_block_idx(
            feature_block_idx=feature_block_idx,
            projection_block_idx=projection_block_idx,
            depth=len(blocks),
            fallback_current_for_last=fallback_current_for_last,
        )
        return extract_matrix_from_block(blocks[proj_idx], mode=mode), proj_idx

    if group == "intra_frame_block":
        blocks = [model.decoder[i] for i in range(len(model.decoder)) if i % 2 == 0]
        proj_idx = _resolve_projection_block_idx(
            feature_block_idx=feature_block_idx,
            projection_block_idx=projection_block_idx,
            depth=len(blocks),
            fallback_current_for_last=fallback_current_for_last,
        )
        return extract_matrix_from_block(blocks[proj_idx], mode=mode), proj_idx

    if group == "image_encoder":
        blocks = model.encoder.blocks
        proj_idx = _resolve_projection_block_idx(
            feature_block_idx=feature_block_idx,
            projection_block_idx=projection_block_idx,
            depth=model.encoder.n_blocks,
            fallback_current_for_last=fallback_current_for_last,
        )
        return extract_matrix_from_block(blocks[proj_idx], mode=mode), proj_idx

    if group == "cam_head":
        blocks = model.camera_decoder.blocks
    elif group == "conf_head":
        blocks = model.conf_decoder.blocks
    elif group == "point_head":
        blocks = model.point_decoder.blocks
    else:
        raise ValueError(f"Unsupported group '{group}'.")

    proj_idx = _resolve_projection_block_idx(
        feature_block_idx=feature_block_idx,
        projection_block_idx=projection_block_idx,
        depth=len(blocks),
        fallback_current_for_last=fallback_current_for_last,
    )
    return extract_matrix_from_block(blocks[proj_idx], mode=mode), proj_idx
