from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

from .motion_mask_benchmarks import (
    build_sequence_spec,
    format_context_summary,
    load_sequence_paths_from_spec,
    select_context_indices,
)
from .motion_mask_export_common import pairwise_target_input_protocol, single_target_input_protocol


def load_scoremap_sequence_context(
    *,
    dataset: str,
    benchmark_name: str | None,
    sequence: str,
    frame_idx: int,
    context_size: int,
    context_stride: int,
    davis_root: str | None,
    segtrack_root: str | None,
    fbms_root: str | None,
    fbms_split: str | None = None,
):
    spec = build_sequence_spec(
        dataset=dataset,
        benchmark=benchmark_name,
        sequence=sequence,
        fbms_split=fbms_split if str(dataset).lower() == "fbms59" else None,
    )
    image_paths, mask_refs, sequence_tag = load_sequence_paths_from_spec(
        spec,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
    )
    if frame_idx < 0 or frame_idx >= len(image_paths):
        raise IndexError(f"frame_idx out of range: {frame_idx} for sequence {sequence_tag} ({len(image_paths)} frames)")
    input_frame_indices = select_context_indices(
        len(image_paths),
        frame_idx,
        total_frames=context_size,
        stride=context_stride,
    )
    return spec, image_paths, mask_refs, sequence_tag, input_frame_indices


def build_scoremap_output_dir(
    *,
    out_root: str | Path,
    group_label: str,
    dataset: str,
    sequence_tag: str,
    frame_idx: int,
) -> Path:
    return Path(out_root) / group_label / str(dataset).lower() / sequence_tag / f"frame_{int(frame_idx):05d}"


def build_scoremap_meta(
    *,
    dataset: str,
    sequence: str,
    sequence_tag: str,
    frame_idx: int,
    group: str,
    probe_root: str | Path,
    usable_blocks: Sequence[str],
    queries: Sequence[Tuple[str, int]],
    patch_size: int,
    save_upsampled: bool,
    input_frame_indices: Sequence[int],
    image_paths: Sequence[Path | str],
    pairwise: bool = False,
    extra_meta: Dict[str, object] | None = None,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "sequence": sequence,
        "dataset": str(dataset).lower(),
        "sequence_tag": sequence_tag,
        "frame_idx": int(frame_idx),
        "group": group,
        "probe_root": str(probe_root),
        "usable_blocks": list(usable_blocks),
        "queries": [{"name": name, "idx": int(idx)} for name, idx in queries],
        "patch_size": int(patch_size),
        "save_upsampled": bool(save_upsampled),
        "input_protocol": (
            pairwise_target_input_protocol(total_input_frames=len(input_frame_indices))
            if pairwise
            else single_target_input_protocol(total_input_frames=len(input_frame_indices))
        ),
        "input_frame_indices": [int(idx) for idx in input_frame_indices],
        "input_image_paths": [str(image_paths[idx]) for idx in input_frame_indices],
        **format_context_summary(input_frame_indices),
    }
    if pairwise:
        meta["reference_image_paths"] = [str(image_paths[idx]) for idx in input_frame_indices[1:]]
    if extra_meta:
        meta.update(extra_meta)
    return meta


def save_scoremap_meta(out_dir: Path, meta: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
