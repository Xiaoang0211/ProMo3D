from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import importlib
import importlib.util
import sys
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from isdynamic.core.preprocess_aspect_pad import aspect_pad_arrays

try:
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import ImgNorm
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _MODELS_ROOT = _PROJECT_ROOT / "models"
    _DUST3R_REPO_ROOT = _MODELS_ROOT / "dust3r"
    _DUST3R_PACKAGE_ROOT = _DUST3R_REPO_ROOT / "dust3r"
    if str(_MODELS_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODELS_ROOT))
    if str(_DUST3R_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_DUST3R_REPO_ROOT))
    for _name in list(sys.modules.keys()):
        if _name == "dust3r" or _name.startswith("dust3r."):
            sys.modules.pop(_name, None)
    _init_py = _DUST3R_PACKAGE_ROOT / "__init__.py"
    _spec = importlib.util.spec_from_file_location(
        "dust3r",
        _init_py,
        submodule_search_locations=[str(_DUST3R_PACKAGE_ROOT)],
    )
    if _spec is None or _spec.loader is None:
        raise ModuleNotFoundError(f"Could not create a local import spec for {_init_py}")
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["dust3r"] = _pkg
    _spec.loader.exec_module(_pkg)
    AsymmetricCroCo3DStereo = importlib.import_module("dust3r.model").AsymmetricCroCo3DStereo
    ImgNorm = importlib.import_module("dust3r.utils.image").ImgNorm


DUST3R_PATCH_SIZE = 16
SOURCE_CHOICES = ["enc", "dec_view1", "dec_view2", "head_in_view1", "head_in_view2"]
GROUP_CHOICES = list(SOURCE_CHOICES)
MODEL_NAME_CHOICES = [
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    "DUSt3R_ViTLarge_BaseDecoder_512_linear",
    "DUSt3R_ViTLarge_BaseDecoder_224_linear",
]


def load_dust3r_model(
    weights_or_name: str,
    device: torch.device,
) -> AsymmetricCroCo3DStereo:
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_or_name)
    model = model.to(device)
    model.eval()
    return model


def _sort_suffix(key: str):
    suffix = key.rsplit("_", 1)[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def build_feature_key_order(
    model: AsymmetricCroCo3DStereo,
    token_sources: Iterable[str],
) -> List[str]:
    token_sources = list(token_sources)
    keys: List[str] = []

    if "enc" in token_sources:
        for idx in range(len(model.enc_blocks)):
            keys.append(f"enc_{idx}")
    if "dec_view1" in token_sources:
        for idx in range(len(model.dec_blocks)):
            keys.append(f"dec_view1_{idx}")
    if "dec_view2" in token_sources:
        for idx in range(len(model.dec_blocks2)):
            keys.append(f"dec_view2_{idx}")
    if "head_in_view1" in token_sources:
        keys.extend(_head_feature_keys(model, "head_in_view1"))
    if "head_in_view2" in token_sources:
        keys.extend(_head_feature_keys(model, "head_in_view2"))
    return keys


def _head_hooks(model: AsymmetricCroCo3DStereo) -> Sequence[Union[int, str]]:
    if model.head_type == "dpt":
        return list(model.downstream_head1.dpt.hooks)
    return ["final"]


def _head_feature_keys(model: AsymmetricCroCo3DStereo, prefix: str) -> List[str]:
    hooks = _head_hooks(model)
    return [f"{prefix}_{hook}" for hook in hooks]


def _resize_pil_image(img: Image.Image, long_edge_size: int, interp) -> Image.Image:
    longest = max(img.size)
    new_size = tuple(int(round(x * long_edge_size / longest)) for x in img.size)
    return img.resize(new_size, interp)


def preprocess_image_and_mask(
    image_path: Path,
    mask_path: Path,
    image_size: int,
    patch_size: int,
    square_ok: bool = False,
    preprocess_mode: str = "native",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    image = exif_transpose(Image.open(image_path)).convert("RGB")
    mask = exif_transpose(Image.open(mask_path)).convert("L")

    if preprocess_mode == "aspect_pad":
        image_arr, mask_arr = aspect_pad_arrays(image, mask, target_size=image_size, patch_size=patch_size)
        image = Image.fromarray(image_arr)
        mask_tensor = torch.from_numpy(mask_arr.astype(np.float32))
    elif preprocess_mode == "native":
        if image_size == 224:
            resize_target = round(image_size * max(image.size[0] / image.size[1], image.size[1] / image.size[0]))
            image = _resize_pil_image(image, resize_target, Image.Resampling.BICUBIC)
            mask = _resize_pil_image(mask, resize_target, Image.Resampling.NEAREST)
        else:
            image = _resize_pil_image(image, image_size, Image.Resampling.BICUBIC)
            mask = _resize_pil_image(mask, image_size, Image.Resampling.NEAREST)

        width, height = image.size
        center_x, center_y = width // 2, height // 2
        if image_size == 224:
            half = min(center_x, center_y)
            crop_box = (center_x - half, center_y - half, center_x + half, center_y + half)
        else:
            half_width = ((2 * center_x) // patch_size) * patch_size / 2
            half_height = ((2 * center_y) // patch_size) * patch_size / 2
            if not square_ok and width == height:
                half_height = 3 * half_width / 4
            crop_box = (
                int(center_x - half_width),
                int(center_y - half_height),
                int(center_x + half_width),
                int(center_y + half_height),
            )

        image = image.crop(crop_box)
        mask = mask.crop(crop_box)
        mask_tensor = (torch.from_numpy(np.array(mask, dtype=np.uint8)) > 0).float()
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    width, height = image.size
    image_tensor = ImgNorm(image)

    if preprocess_mode == "native" and width < height:
        # Dust3R keeps portrait inputs in a landscape tensor layout; keep the
        # binary mask in the same orientation so downstream visualizations and
        # upsampled overlays use consistent H/W.
        image_tensor = image_tensor.swapaxes(1, 2)
        mask_tensor = mask_tensor.swapaxes(0, 1)

    view = {
        "img": image_tensor,
        "true_shape": torch.tensor((height, width), dtype=torch.int64),
    }
    return view, mask_tensor


def collate_views(views: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not views:
        raise ValueError("Expected a non-empty list of views.")
    return {
        "img": torch.stack([view["img"] for view in views], dim=0),
        "true_shape": torch.stack([view["true_shape"] for view in views], dim=0),
    }


def _encode_image_all_blocks(
    model: AsymmetricCroCo3DStereo,
    image: torch.Tensor,
    true_shape: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    x, pos = model.patch_embed(image, true_shape=true_shape)
    outputs: List[torch.Tensor] = []
    for block in model.enc_blocks:
        x = block(x, pos)
        outputs.append(x)
    outputs[-1] = model.enc_norm(outputs[-1])
    return outputs, pos


@torch.no_grad()
def extract_feature_map(
    model: AsymmetricCroCo3DStereo,
    view1: Dict[str, torch.Tensor],
    view2: Dict[str, torch.Tensor],
    token_sources: Iterable[str],
) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    token_sources = list(token_sources)

    enc1_blocks, pos1 = _encode_image_all_blocks(model, view1["img"], view1["true_shape"])
    enc2_blocks, pos2 = _encode_image_all_blocks(model, view2["img"], view2["true_shape"])

    dec1, dec2 = model._decoder(enc1_blocks[-1], pos1, enc2_blocks[-1], pos2)
    dec1 = list(dec1)
    dec2 = list(dec2)

    feature_map: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}

    if "enc" in token_sources:
        for idx, (block1, block2) in enumerate(zip(enc1_blocks, enc2_blocks)):
            feature_map[f"enc_{idx}"] = (block1, block2)

    if "dec_view1" in token_sources:
        for idx in range(1, len(dec1)):
            feature_map[f"dec_view1_{idx - 1}"] = dec1[idx]

    if "dec_view2" in token_sources:
        for idx in range(1, len(dec2)):
            feature_map[f"dec_view2_{idx - 1}"] = dec2[idx]

    hooks = _head_hooks(model)
    if "head_in_view1" in token_sources:
        for hook in hooks:
            key = f"head_in_view1_{hook}"
            feature_map[key] = dec1[-1] if hook == "final" else dec1[int(hook)]

    if "head_in_view2" in token_sources:
        for hook in hooks:
            key = f"head_in_view2_{hook}"
            feature_map[key] = dec2[-1] if hook == "final" else dec2[int(hook)]

    return feature_map


def sort_feature_keys(keys: Iterable[str]) -> List[str]:
    return sorted(keys, key=_sort_suffix)
