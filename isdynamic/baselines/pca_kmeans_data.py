from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from isdynamic.layout import SUPPORTED_MODELS, normalize_model_name
from isdynamic.viz.motion_mask_benchmarks import (
    DEFAULT_DAVIS_ROOT,
    DEFAULT_FBMS_ROOT,
    DEFAULT_SEGTRACK_ROOT,
    SequenceSpec,
    list_sequence_specs,
    load_sequence_paths_from_spec,
    normalize_benchmark_name,
    resolve_davis_root,
    resolve_fbms_root,
    resolve_segtrack_root,
)
from isdynamic.viz.motion_mask_export_common import load_motion_mask_sequence_context, select_context_indices


def resolve_youtubevos_root(path: str | Path) -> Path:
    root = Path(path)
    candidates = [root]
    for child_name in ("Youtube-VOS", "YouTube-VOS", "youtubevos", "youtube_vos"):
        child = root / child_name
        if child.exists():
            candidates.append(child)
    parent = root.parent
    if parent.exists():
        for sibling_name in ("Youtube-VOS", "YouTube-VOS", "youtubevos", "youtube_vos"):
            sibling = parent / sibling_name
            if sibling.exists():
                candidates.append(sibling)
    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    for candidate in deduped:
        if all((candidate / split_name).exists() for split_name in ("train_motion", "valid_motion")):
            return candidate
        subset_root = candidate / "youtubevos_160vids"
        if (subset_root / "train").exists() and (subset_root / "validation").exists():
            return subset_root
    checked = "\n".join(str(item) for item in deduped)
    raise FileNotFoundError(
        'Could not resolve YouTube-VOS root with required split folders train_motion/ and valid_motion/.\n'
        f'Checked:\n{checked}'
    )


def _resolve_youtubevos_split_dirs(split_root: Path) -> tuple[Path, Path]:
    direct_img = split_root / 'JPEGImages'
    typo_img = split_root / 'JEPGImages'
    direct_ann = split_root / 'Annotations'
    if direct_img.exists() and direct_ann.exists():
        return direct_img, direct_ann
    if typo_img.exists() and direct_ann.exists():
        return typo_img, direct_ann
    if (split_root / 'JPEGImages' / '480p').exists() and (split_root / 'Annotations' / '480p').exists():
        return split_root / 'JPEGImages' / '480p', split_root / 'Annotations' / '480p'
    if (split_root / 'JEPGImages' / '480p').exists() and (split_root / 'Annotations' / '480p').exists():
        return split_root / 'JEPGImages' / '480p', split_root / 'Annotations' / '480p'
    if (split_root / 'images').exists() and (split_root / 'annotations').exists():
        return split_root / 'images', split_root / 'annotations'
    raise FileNotFoundError(f'Unsupported YouTube-VOS split structure under {split_root}')


def list_youtubevos_sequences(youtubevos_root: Path, split_name: str) -> list[str]:
    split_root = youtubevos_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f'YouTube-VOS split not found: {split_root}')
    image_root, ann_root = _resolve_youtubevos_split_dirs(split_root)
    seqs = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if not seqs:
        raise RuntimeError(f'No sequence directories found under {image_root}')
    missing = [sequence for sequence in seqs if not (ann_root / sequence).exists()]
    if missing:
        raise FileNotFoundError(
            f'Missing annotation directories for {len(missing)} sequences under {ann_root}. Examples: {missing[:10]}'
        )
    return seqs


def load_youtubevos_sequence_paths(youtubevos_root: Path, split_name: str, sequence: str) -> tuple[list[Path], list[Path]]:
    split_root = youtubevos_root / split_name
    image_root, ann_root = _resolve_youtubevos_split_dirs(split_root)
    image_dir = image_root / sequence
    mask_dir = ann_root / sequence
    image_paths = sorted(image_dir.glob('*.jpg')) or sorted(image_dir.glob('*.png'))
    if not image_paths:
        raise FileNotFoundError(f'No images found in {image_dir}')
    mask_paths = [mask_dir / f'{path.stem}.png' for path in image_paths]
    missing = [str(path) for path in mask_paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing masks:\n' + '\n'.join(missing[:20]))
    return image_paths, mask_paths


YOUTUBE_VOS_VAL = "youtube_vos_val"

DATASET_CHOICES = (
    YOUTUBE_VOS_VAL,
    "davis_all",
    "davis2017",
    "davis2016",
    "segtrackv2",
    "fbms59",
)

DEFAULT_GROUPS_BY_MODEL: dict[str, list[str]] = {
    "pi3": ["inter_frame_block"],
    "pi3x": ["inter_frame_block"],
    "vggt": ["inter_frame_block"],
    "vggt_omega": ["inter_frame_block"],
    "depth_anything_3": ["inter_frame_block"],
    "map_anything": ["inter_frame_block"],
    "dust3r": ["dec_view1", "head_in_view1"],
}

DEFAULT_CONTEXT_SIZE_BY_MODEL: dict[str, int] = {
    "pi3": 3,
    "pi3x": 3,
    "vggt": 3,
    "vggt_omega": 3,
    "depth_anything_3": 3,
    "map_anything": 3,
    "dust3r": 3,
}


@dataclass(frozen=True)
class DatasetRoots:
    youtubevos_root: Path
    davis_root: Path
    segtrack_root: Path
    fbms_root: Path


@dataclass(frozen=True)
class ModelResources:
    model_name: str
    model: Any
    patch_size: int
    default_groups: tuple[str, ...]
    default_context_size: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SampleContext:
    model_name: str
    dataset_name: str
    sequence: str
    sequence_tag: str
    benchmark_name: str | None
    frame_idx: int
    input_frame_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    mask_refs: tuple[Any, ...]
    image_np: np.ndarray
    gt_mask_np: np.ndarray
    feature_map: dict[str, Any]
    feature_key_order: tuple[str, ...]
    patch_size: int


def canonical_model_name(name: str) -> str:
    model_name = normalize_model_name(name)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    return model_name


def default_groups_for_model(model_name: str) -> list[str]:
    canonical = canonical_model_name(model_name)
    return list(DEFAULT_GROUPS_BY_MODEL[canonical])


def resolve_dataset_roots(
    *,
    youtubevos_root: str | Path,
    davis_root: str | Path | None,
    segtrack_root: str | Path | None,
    fbms_root: str | Path | None,
) -> DatasetRoots:
    return DatasetRoots(
        youtubevos_root=resolve_youtubevos_root(str(youtubevos_root)),
        davis_root=resolve_davis_root(davis_root if davis_root is not None else DEFAULT_DAVIS_ROOT),
        segtrack_root=resolve_segtrack_root(segtrack_root if segtrack_root is not None else DEFAULT_SEGTRACK_ROOT),
        fbms_root=resolve_fbms_root(fbms_root if fbms_root is not None else DEFAULT_FBMS_ROOT),
    )


def list_specs_for_dataset(dataset_name: str, roots: DatasetRoots) -> list[SequenceSpec]:
    if dataset_name == YOUTUBE_VOS_VAL:
        return [
            SequenceSpec(
                dataset="youtubevos",
                benchmark=YOUTUBE_VOS_VAL,
                sequence=sequence,
                sequence_tag=f"youtubevos__valid_motion__{sequence}",
            )
            for sequence in list_youtubevos_sequences(roots.youtubevos_root, "valid_motion")
        ]
    benchmark_name = normalize_benchmark_name(dataset_name)
    return list_sequence_specs(
        benchmark=benchmark_name,
        davis_root=roots.davis_root,
        segtrack_root=roots.segtrack_root,
        fbms_root=roots.fbms_root,
    )


def list_target_frame_indices_for_spec(spec: SequenceSpec, roots: DatasetRoots) -> list[int]:
    if spec.dataset == "youtubevos":
        val_split = "valid_motion" if (roots.youtubevos_root / "valid_motion").exists() else "validation"
        image_paths, _ = load_youtubevos_sequence_paths(roots.youtubevos_root, val_split, spec.sequence)
        return list(range(len(image_paths)))
    image_paths, _mask_refs, _sequence_tag = load_sequence_paths_from_spec(
        spec,
        davis_root=roots.davis_root,
        segtrack_root=roots.segtrack_root,
        fbms_root=roots.fbms_root,
        fbms_include_unannotated=False,
    )
    return list(range(len(image_paths)))


def load_model_resources(
    *,
    model_name: str,
    device: torch.device,
    checkpoint: str | None = None,
    model_dir: str | None = None,
    weights: str | None = None,
    dust3r_model_name: str | None = None,
    use_multimodal_pi3x: bool = False,
    mapanything_machine: str = "aws",
) -> ModelResources:
    model_name = canonical_model_name(model_name)
    if model_name in {"pi3", "pi3x"}:
        if not checkpoint:
            raise ValueError(f"--checkpoint is required for model={model_name}")
        from isdynamic.models.pi3 import PI3_PATCH_SIZE, load_pi3_family_model

        model = load_pi3_family_model(
            model_name=model_name,
            checkpoint=Path(checkpoint),
            device=device,
            use_multimodal_pi3x=bool(use_multimodal_pi3x),
        )
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=PI3_PATCH_SIZE,
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={},
        )

    if model_name == "vggt":
        if not checkpoint:
            raise ValueError("--checkpoint is required for model=vggt")
        from isdynamic.legacy.vggt.build_davis_isdynamic_npz import (
            VGGT,
            VGGT_PATCH_SIZE,
            _extract_state_dict as extract_vggt_state_dict,
            _strip_module_prefix as strip_vggt_module_prefix,
        )

        model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False).to(device)
        state_dict = strip_vggt_module_prefix(extract_vggt_state_dict(torch.load(Path(checkpoint), map_location="cpu")))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=VGGT_PATCH_SIZE,
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={},
        )

    if model_name == "vggt_omega":
        if not checkpoint:
            raise ValueError("--checkpoint is required for model=vggt_omega")
        from isdynamic.models.vggt_omega import VGGT_OMEGA_PATCH_SIZE, load_vggt_omega

        model = load_vggt_omega(Path(checkpoint), device=device, enable_camera=True, enable_depth=True)
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=VGGT_OMEGA_PATCH_SIZE,
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={},
        )

    if model_name == "depth_anything_3":
        if not model_dir:
            raise ValueError("--model_dir is required for model=depth_anything_3")
        from isdynamic.legacy.depth_anything_3.build_davis_isdynamic_npz import PATCH_SIZE as DA3_PATCH_SIZE, _resolve_anyview_model
        from isdynamic.models.depth_anything_3 import load_da3_model

        model = load_da3_model(model_dir, device=device)
        anyview_model = _resolve_anyview_model(model)
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=DA3_PATCH_SIZE,
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={
                "anyview_model": anyview_model,
                "export_feat_layers": list(range(len(anyview_model.backbone.pretrained.blocks))),
            },
        )

    if model_name == "map_anything":
        if not checkpoint:
            raise ValueError("--checkpoint is required for model=map_anything")
        from isdynamic.models.map_anything import load_mapanything

        model = load_mapanything(checkpoint_path=checkpoint, device=device, machine=mapanything_machine)
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=int(model.encoder.patch_size),
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={"norm_type": model.encoder.data_norm_type},
        )

    if model_name == "dust3r":
        weights_or_name = weights or dust3r_model_name
        if not weights_or_name:
            raise ValueError("--weights or --dust3r_model_name is required for model=dust3r")
        from isdynamic.models.dust3r import DUST3R_PATCH_SIZE, load_dust3r_model

        model = load_dust3r_model(weights_or_name=weights_or_name, device=device)
        return ModelResources(
            model_name=model_name,
            model=model,
            patch_size=DUST3R_PATCH_SIZE,
            default_groups=tuple(DEFAULT_GROUPS_BY_MODEL[model_name]),
            default_context_size=DEFAULT_CONTEXT_SIZE_BY_MODEL[model_name],
            metadata={"square_ok": False},
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _load_sequence_context_for_spec(
    *,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    context_size: int,
    context_stride: int,
) -> tuple[tuple[Path, ...], tuple[Any, ...], tuple[int, ...]]:
    if spec.dataset == "youtubevos":
        val_split = "valid_motion" if (roots.youtubevos_root / "valid_motion").exists() else "validation"
        image_paths, mask_refs = load_youtubevos_sequence_paths(roots.youtubevos_root, val_split, spec.sequence)
        input_frame_indices = tuple(
            int(x)
            for x in select_context_indices(
                len(image_paths),
                frame_idx,
                total_frames=context_size,
                stride=context_stride,
            )
        )
        return tuple(image_paths), tuple(mask_refs), input_frame_indices

    benchmark_dataset = spec.dataset
    fbms_split = spec.fbms_split
    _loaded_spec, image_paths, mask_refs, _sequence_tag, input_frame_indices = load_motion_mask_sequence_context(
        dataset=benchmark_dataset,
        benchmark_name=spec.benchmark,
        sequence=spec.sequence,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
        davis_root=str(roots.davis_root),
        segtrack_root=str(roots.segtrack_root),
        fbms_root=str(roots.fbms_root),
        fbms_split=fbms_split,
    )
    return tuple(image_paths), tuple(mask_refs), tuple(int(x) for x in input_frame_indices)


def _extract_pi3_like_sample(
    *,
    model_name: str,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
    preprocess_mode: str = "native",
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    image_tensors = []
    target_mask_t = None
    for input_pos, input_frame_idx in enumerate(input_frame_indices):
        from isdynamic.legacy.pi3.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref as preprocess_pi3_image_and_mask_ref

        image_t, mask_t = preprocess_pi3_image_and_mask_ref(image_paths[input_frame_idx], mask_refs[input_frame_idx])
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask for Pi3 sample extraction.")

    images_b = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    token_sources = []
    for group in groups:
        if group in {"inter_frame_block", "intra_frame_block"}:
            token_sources.append("alt_attn")
        else:
            token_sources.append(group)
    from isdynamic.models.pi3 import build_feature_key_order as build_pi3_feature_key_order, extract_feature_map as extract_pi3_feature_map

    feature_map = extract_pi3_feature_map(
        model_name=model_name,
        model=resources.model,
        images_b=images_b,
        token_sources=token_sources,
    )
    feature_key_order = build_pi3_feature_key_order(resources.model, token_sources)
    image_np = np.clip(image_tensors[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    gt_mask_np = target_mask_t.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name=model_name,
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def _load_vggt_model_from_resources(resources: ModelResources):
    model = resources.model
    if not hasattr(model, 'aggregator'):
        raise TypeError(f"Expected VGGT-like resources.model, got {type(model)!r}")
    return model


def _extract_vggt_sample(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
    preprocess_mode: str = "native",
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    image_tensors = []
    target_mask_t = None
    for input_pos, input_frame_idx in enumerate(input_frame_indices):
        from isdynamic.legacy.vggt.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref as preprocess_vggt_image_and_mask_ref

        image_t, mask_t = preprocess_vggt_image_and_mask_ref(
            image_paths[input_frame_idx],
            mask_refs[input_frame_idx],
            preprocess_mode=preprocess_mode,
        )
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask for VGGT sample extraction.")

    images_b = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    vggt = _load_vggt_model_from_resources(resources)
    from isdynamic.legacy.vggt.build_davis_isdynamic_npz import (
        build_feature_key_order as build_vggt_feature_key_order,
        extract_feature_map as extract_vggt_feature_map,
    )

    with torch.no_grad():
        _, patch_start_idx, frame_outputs, global_outputs = vggt.aggregator(
            images_b,
            return_split=True,
            return_concat=False,
        )
        token_sources = []
        for group in groups:
            if group in {"inter_frame_block", "intra_frame_block"}:
                token_sources.append("alt_attn")
            elif group in {"cam_head", "depth_head", "point_head"}:
                token_sources.append(group)
            else:
                raise ValueError(f"Unsupported VGGT group: {group}")
        feature_map = extract_vggt_feature_map(
            model=vggt,
            images_b=images_b,
            frame_outputs=frame_outputs,
            global_outputs=global_outputs,
            patch_start_idx=patch_start_idx,
            token_sources=token_sources,
        )
    feature_key_order = build_vggt_feature_key_order(vggt, token_sources)
    image_np = np.clip(image_tensors[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    gt_mask_np = target_mask_t.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name="vggt",
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def _extract_vggt_omega_sample(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    image_tensors = []
    target_mask_t = None
    for input_pos, input_frame_idx in enumerate(input_frame_indices):
        from isdynamic.models.vggt_omega import preprocess_image_and_mask_ref as preprocess_vggt_omega_image_and_mask_ref

        image_t, mask_t = preprocess_vggt_omega_image_and_mask_ref(
            image_paths[input_frame_idx],
            mask_refs[input_frame_idx],
            preprocess_mode=preprocess_mode,
        )
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask for VGGT-Omega sample extraction.")

    images_b = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    from isdynamic.models.vggt_omega import (
        build_feature_key_order as build_vggt_omega_feature_key_order,
        extract_feature_map as extract_vggt_omega_feature_map,
    )

    token_sources = []
    for group in groups:
        if group in {"inter_frame_block", "intra_frame_block"}:
            token_sources.append("alt_attn")
        elif group in {"cam_head", "depth_head"}:
            token_sources.append(group)
        else:
            raise ValueError(f"Unsupported VGGT-Omega group: {group}")

    with torch.no_grad():
        feature_map, _patch_start_idx = extract_vggt_omega_feature_map(
            model=resources.model,
            images_b=images_b,
            token_sources=token_sources,
        )
    feature_key_order = build_vggt_omega_feature_key_order(resources.model, token_sources)
    image_np = np.clip(image_tensors[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    gt_mask_np = target_mask_t.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name="vggt_omega",
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def _extract_da3_sample(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
    process_res: int = 518,
    process_res_method: str = "upper_bound_resize",
    preprocess_mode: str = "native",
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    image_tensors = []
    target_mask_t = None
    for input_pos, input_frame_idx in enumerate(input_frame_indices):
        from isdynamic.legacy.depth_anything_3.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref as preprocess_da3_image_and_mask_ref

        image_t, mask_t = preprocess_da3_image_and_mask_ref(
            image_paths[input_frame_idx],
            mask_refs[input_frame_idx],
            process_res=process_res,
            process_res_method=process_res_method,
            preprocess_mode=preprocess_mode,
        )
        image_tensors.append(image_t)
        if input_pos == 0:
            target_mask_t = mask_t
    if target_mask_t is None:
        raise RuntimeError("Missing target mask for DA3 sample extraction.")

    token_sources = []
    for group in groups:
        if group in {"inter_frame_block", "intra_frame_block"}:
            token_sources.append("alt_attn")
        elif group in {"depth_head", "dual_dpt_head"}:
            token_sources.append("depth_head")
        else:
            raise ValueError(f"Unsupported DA3 group: {group}")

    from isdynamic.legacy.depth_anything_3.build_davis_isdynamic_npz import (
        _run_anyview_backbone,
        build_feature_key_order as build_da3_feature_key_order,
        extract_feature_map as extract_da3_feature_map,
    )
    from isdynamic.legacy.depth_anything_3.visualize_isdynamic import _recover_image_np as recover_da3_image_np

    anyview_model = resources.metadata["anyview_model"]
    export_feat_layers = resources.metadata["export_feat_layers"]
    feats, aux_feats = _run_anyview_backbone(
        anyview_model=anyview_model,
        images_b=torch.stack(image_tensors, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32),
        export_feat_layers=export_feat_layers,
    )
    feature_map = extract_da3_feature_map(
        anyview_model=anyview_model,
        feats=feats,
        aux_feats=aux_feats,
        export_feat_layers=export_feat_layers,
        token_sources=token_sources,
    )
    feature_key_order, _block_groups, _depth_head_layers = build_da3_feature_key_order(
        anyview_model=anyview_model,
        token_sources=token_sources,
    )
    image_np = recover_da3_image_np(image_tensors[0])
    gt_mask_np = target_mask_t.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name="depth_anything_3",
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def _extract_mapanything_sample(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
    resolution_set: int = 518,
    preprocess_mode: str = "native",
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    entries = [(image_paths[idx], mask_refs[idx]) for idx in input_frame_indices]
    from isdynamic.legacy.map_anything.build_davis_isdynamic_npz import preprocess_entries
    from isdynamic.legacy.map_anything.visualize_isdynamic import _recover_image_np as recover_mapanything_image_np
    from isdynamic.models.map_anything import build_feature_key_order as build_mapanything_feature_key_order, extract_feature_map as extract_mapanything_feature_map

    image_tensors, mask_tensors = preprocess_entries(
        entries,
        norm_type=resources.metadata["norm_type"],
        resolution_set=resolution_set,
        preprocess_mode=preprocess_mode,
    )
    image_t = image_tensors[0]
    mask_t = mask_tensors[0]
    views = [
        {
            "img": image_tensor.unsqueeze(0).to(device),
            "data_norm_type": [resources.metadata["norm_type"]],
            "true_shape": np.int32([image_tensor.shape[-2:]]),
            "idx": view_idx,
            "instance": str(view_idx),
        }
        for view_idx, image_tensor in enumerate(image_tensors)
    ]
    feature_map = extract_mapanything_feature_map(resources.model, views, list(groups))
    feature_key_order = build_mapanything_feature_key_order(resources.model, list(groups))
    image_np = recover_mapanything_image_np(image_t, norm_type=resources.metadata["norm_type"])
    gt_mask_np = mask_t.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name="map_anything",
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def _extract_dust3r_sample(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int,
    context_stride: int,
    device: torch.device,
    image_size: int = 512,
    preprocess_mode: str = "native",
) -> SampleContext:
    image_paths, mask_refs, input_frame_indices = _load_sequence_context_for_spec(
        spec=spec,
        roots=roots,
        frame_idx=frame_idx,
        context_size=context_size,
        context_stride=context_stride,
    )
    if len(input_frame_indices) < 2:
        raise ValueError("Dust3R baseline requires at least one context frame.")
    from isdynamic.legacy.dust3r.build_youtubevos_isdynamic_npz import preprocess_image_and_mask_ref as preprocess_dust3r_image_and_mask_ref
    from isdynamic.legacy.dust3r.visualize_isdynamic import _recover_image_np as recover_dust3r_image_np
    from isdynamic.models.dust3r import (
        DUST3R_PATCH_SIZE,
        build_feature_key_order as build_dust3r_feature_key_order,
        collate_views,
        extract_feature_map as extract_dust3r_feature_map,
    )

    target_view, target_mask = preprocess_dust3r_image_and_mask_ref(
        image_paths[input_frame_indices[0]],
        mask_refs[input_frame_indices[0]],
        image_size=image_size,
        patch_size=DUST3R_PATCH_SIZE,
        square_ok=bool(resources.metadata.get("square_ok", False)),
        preprocess_mode=preprocess_mode,
    )
    ref_views = []
    for ref_idx in input_frame_indices[1:]:
        ref_view, _ = preprocess_dust3r_image_and_mask_ref(
            image_paths[ref_idx],
            mask_refs[ref_idx],
            image_size=image_size,
            patch_size=DUST3R_PATCH_SIZE,
            square_ok=bool(resources.metadata.get("square_ok", False)),
            preprocess_mode=preprocess_mode,
        )
        ref_views.append(ref_view)
    batch_view1 = collate_views([target_view for _ in ref_views])
    batch_view2 = collate_views(ref_views)
    batch_view1 = {key: value.to(device) for key, value in batch_view1.items()}
    batch_view2 = {key: value.to(device) for key, value in batch_view2.items()}
    feature_map = extract_dust3r_feature_map(
        model=resources.model,
        view1=batch_view1,
        view2=batch_view2,
        token_sources=list(groups),
    )
    feature_key_order = build_dust3r_feature_key_order(resources.model, list(groups))
    image_np = recover_dust3r_image_np(target_view)
    gt_mask_np = target_mask.cpu().numpy().astype(np.uint8)
    return SampleContext(
        model_name="dust3r",
        dataset_name=spec.dataset,
        sequence=spec.sequence,
        sequence_tag=spec.sequence_tag,
        benchmark_name=spec.benchmark,
        frame_idx=int(frame_idx),
        input_frame_indices=input_frame_indices,
        image_paths=image_paths,
        mask_refs=mask_refs,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        feature_map=feature_map,
        feature_key_order=tuple(feature_key_order),
        patch_size=resources.patch_size,
    )


def extract_sample_context(
    *,
    resources: ModelResources,
    spec: SequenceSpec,
    roots: DatasetRoots,
    frame_idx: int,
    groups: Sequence[str],
    context_size: int | None,
    context_stride: int,
    device: torch.device,
    process_res: int = 518,
    process_res_method: str = "upper_bound_resize",
    resolution_set: int = 518,
    image_size: int = 512,
    preprocess_mode: str = "native",
) -> SampleContext:
    model_name = canonical_model_name(resources.model_name)
    context_size = int(context_size if context_size is not None else resources.default_context_size)
    groups = [str(group).strip() for group in groups if str(group).strip()]
    if not groups:
        groups = list(resources.default_groups)

    if model_name in {"pi3", "pi3x"}:
        return _extract_pi3_like_sample(
            model_name=model_name,
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
            preprocess_mode=preprocess_mode,
        )
    if model_name == "vggt":
        return _extract_vggt_sample(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
            preprocess_mode=preprocess_mode,
        )
    if model_name == "vggt_omega":
        return _extract_vggt_omega_sample(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
        )
    if model_name == "depth_anything_3":
        return _extract_da3_sample(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
            process_res=process_res,
            process_res_method=process_res_method,
            preprocess_mode=preprocess_mode,
        )
    if model_name == "map_anything":
        return _extract_mapanything_sample(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
            resolution_set=resolution_set,
            preprocess_mode=preprocess_mode,
        )
    if model_name == "dust3r":
        return _extract_dust3r_sample(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=groups,
            context_size=context_size,
            context_stride=context_stride,
            device=device,
            image_size=image_size,
            preprocess_mode=preprocess_mode,
        )
    raise ValueError(f"Unsupported model: {model_name}")
