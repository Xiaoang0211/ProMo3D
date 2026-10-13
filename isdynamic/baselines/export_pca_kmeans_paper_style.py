from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from isdynamic.baselines.pca_kmeans_core import PCAKMeansConfig, fit_pca_kmeans_assignments
from isdynamic.baselines.pca_kmeans_data import (
    DATASET_CHOICES,
    canonical_model_name,
    default_groups_for_model,
    extract_sample_context,
    list_specs_for_dataset,
    list_target_frame_indices_for_spec,
    load_model_resources,
    resolve_dataset_roots,
)
from isdynamic.baselines.pca_kmeans_io import ensure_pca_kmeans_root
from isdynamic.viz.figure_io import save_raster_png_pdf_figure
from isdynamic.viz.motion_mask_benchmarks import benchmark_output_dirname
from isdynamic.viz.scoremap_utils import crop_to_content_bbox, upsample_patch_map

PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export paper-style PCA+kmeans qualitative cluster visualizations.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--frame_indices', type=str, default='', help='Comma-separated target frame indices. Empty means all frames.')
    parser.add_argument('--max_strip_frames', type=int, default=6)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--dust3r_model_name', type=str, default='')
    parser.add_argument('--youtubevos_root', type=str, default='/home/xiaoang/Datasets/Youtube-VOS')
    parser.add_argument('--davis_root', type=str, default='')
    parser.add_argument('--segtrack_root', type=str, default='')
    parser.add_argument('--fbms_root', type=str, default='')
    parser.add_argument('--group', type=str, default='')
    parser.add_argument('--block_idx', type=int, default=0)
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--context_stride', type=int, default=2)
    parser.add_argument('--norm', type=str, default='zscore_l2', choices=['l2', 'zscore_l2'])
    parser.add_argument('--pca_dim', type=int, default=8, choices=[3, 8, 16])
    parser.add_argument('--num_clusters', type=int, default=2, choices=[2, 3, 5])
    parser.add_argument('--selection_rule', type=str, default='minority', choices=['minority', 'manual'])
    parser.add_argument('--selected_cluster', type=int, default=0)
    parser.add_argument('--process_res', type=int, default=518)
    parser.add_argument('--process_res_method', type=str, default='upper_bound_resize', choices=['upper_bound_resize', 'upper_bound_crop', 'lower_bound_resize', 'lower_bound_crop'])
    parser.add_argument('--resolution_set', type=int, default=518, choices=[504, 512, 518])
    parser.add_argument('--image_size', type=int, default=512, choices=[224, 512])
    parser.add_argument('--preprocess_mode', type=str, default='native', choices=['native', 'aspect_pad'])
    parser.add_argument('--mapanything_machine', type=str, default='aws')
    parser.add_argument('--use_multimodal_pi3x', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--n_init', type=int, default=10)
    parser.add_argument('--out_root', type=str, default='')
    return parser.parse_args()


def _parse_groups(model: str, group: str) -> str:
    if group.strip():
        return group.strip()
    groups = default_groups_for_model(model)
    if len(groups) != 1:
        raise ValueError(f'paper_style expects a single group; default groups for {model} are {groups}')
    return groups[0]


def _parse_frame_indices(text: str, all_indices: Sequence[int]) -> list[int]:
    if not text.strip():
        return list(all_indices)
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def _feature_keys_for_group(feature_key_order: Sequence[str], group: str) -> list[str]:
    prefix = f'{group}_'
    return [key for key in feature_key_order if key.startswith(prefix)]


def _select_feature_key(feature_key_order: Sequence[str], group: str, block_idx: int) -> str:
    keys = _feature_keys_for_group(feature_key_order, group)
    if not keys:
        raise RuntimeError(f'No feature keys found for group={group}')
    if not (0 <= int(block_idx) < len(keys)):
        raise IndexError(f'block_idx={block_idx} out of range for group={group} with {len(keys)} blocks')
    return keys[int(block_idx)]


def _feature_frames_from_value(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        array = value.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 4:
        return array[0]
    if array.ndim == 3:
        return array
    raise ValueError(f'Unsupported feature array shape for paper_style export: {array.shape}')


def _cluster_counts(labels_2d: np.ndarray) -> dict[int, int]:
    return {int(idx): int((labels_2d == idx).sum()) for idx in np.unique(labels_2d)}


def _select_cluster(labels_2d: np.ndarray, selection_rule: str, selected_cluster: int) -> int:
    counts = _cluster_counts(labels_2d)
    if selection_rule == 'manual':
        if int(selected_cluster) not in counts:
            raise ValueError(f'selected_cluster={selected_cluster} not present; available={sorted(counts)}')
        return int(selected_cluster)
    return min(counts, key=lambda cluster_id: (counts[cluster_id], cluster_id))


def _assignment_rgb(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for cluster_id in np.unique(label_map):
        rgb[label_map == cluster_id] = PALETTE[int(cluster_id) % len(PALETTE)]
    return rgb


def _overlay_selected(image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    image = image_np.astype(np.float32)
    mask = (mask_np > 0).astype(np.float32)[..., None]
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * (1.0 - 0.45 * mask) + color * (0.45 * mask), 0, 255).astype(np.uint8)


def _save_tile_strip(images: Sequence[np.ndarray], out_path: Path) -> None:
    if not images:
        return
    max_h = max(int(img.shape[0]) for img in images)
    widths = [int(round(img.shape[1] * (max_h / max(img.shape[0], 1)))) for img in images]
    resized = [np.array(Image.fromarray(img).resize((w, max_h), Image.Resampling.BILINEAR)) for img, w in zip(images, widths)]
    strip = np.concatenate(resized, axis=1)
    png_path = out_path.with_suffix('.png')
    pdf_path = out_path.with_suffix('.pdf')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(png_path)
    Image.fromarray(strip).save(pdf_path)


def _evenly_sample(values: Sequence[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return list(values)
    picks = np.linspace(0, len(values) - 1, num=limit)
    indices = sorted({int(round(x)) for x in picks})
    return [int(values[idx]) for idx in indices]


def _paper_style_root(out_root: str | Path | None, model: str, dataset: str, sequence: str, group: str, block_idx: int) -> Path:
    if out_root:
        base = Path(out_root)
    else:
        base = ensure_pca_kmeans_root() / 'paper_style'
    return base / canonical_model_name(model) / benchmark_output_dirname(dataset) / sequence / f'{group}_block_{int(block_idx)}'


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)
    group = _parse_groups(model_name, args.group)
    roots = resolve_dataset_roots(
        youtubevos_root=args.youtubevos_root,
        davis_root=args.davis_root or None,
        segtrack_root=args.segtrack_root or None,
        fbms_root=args.fbms_root or None,
    )
    specs = [spec for spec in list_specs_for_dataset(args.dataset, roots) if spec.sequence == args.sequence]
    if not specs:
        raise RuntimeError(f'No sequence matched dataset={args.dataset} sequence={args.sequence}')
    if len(specs) != 1:
        raise RuntimeError(f'Expected exactly one sequence spec for {args.sequence}, got {len(specs)}')
    spec = specs[0]

    frame_indices = _parse_frame_indices(args.frame_indices, list_target_frame_indices_for_spec(spec, roots))
    if not frame_indices:
        raise RuntimeError('No frame indices selected.')

    resources = load_model_resources(
        model_name=model_name,
        device=args.device,
        checkpoint=args.checkpoint or None,
        model_dir=args.model_dir or None,
        weights=args.weights or None,
        dust3r_model_name=args.dust3r_model_name or None,
        use_multimodal_pi3x=bool(args.use_multimodal_pi3x),
        mapanything_machine=args.mapanything_machine,
    )

    cfg = PCAKMeansConfig(
        norm=args.norm,
        preprocess_scope='frame_group',
        pca_dim=int(args.pca_dim),
        num_clusters=int(args.num_clusters),
        cluster_rule='minority',
        random_state=int(args.random_state),
        n_init=int(args.n_init),
    )
    out_root = _paper_style_root(args.out_root, model_name, args.dataset, spec.sequence_tag, group, int(args.block_idx))
    out_root.mkdir(parents=True, exist_ok=True)

    strip_input: list[np.ndarray] = []
    strip_selected: list[np.ndarray] = []
    strip_assignment: list[np.ndarray] = []
    strip_frame_choices = set(_evenly_sample(frame_indices, int(args.max_strip_frames)))

    summary = {
        'model': model_name,
        'dataset': args.dataset,
        'benchmark': spec.benchmark,
        'sequence': spec.sequence,
        'sequence_tag': spec.sequence_tag,
        'group': group,
        'block_idx': int(args.block_idx),
        'frame_indices': list(frame_indices),
        'context_size': int(args.context_size),
        'context_stride': int(args.context_stride),
        'norm': args.norm,
        'pca_dim': int(args.pca_dim),
        'num_clusters': int(args.num_clusters),
        'selection_rule': args.selection_rule,
        'selected_cluster_manual': int(args.selected_cluster),
        'frames': [],
    }

    for frame_idx in frame_indices:
        sample = extract_sample_context(
            resources=resources,
            spec=spec,
            roots=roots,
            frame_idx=frame_idx,
            groups=[group],
            context_size=args.context_size,
            context_stride=args.context_stride,
            device=args.device,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            resolution_set=args.resolution_set,
            image_size=args.image_size,
            preprocess_mode=args.preprocess_mode,
        )
        feature_key = _select_feature_key(sample.feature_key_order, group, int(args.block_idx))
        frames_features = _feature_frames_from_value(sample.feature_map[feature_key])
        num_frames_local, num_tokens, dim = frames_features.shape
        fit_result = fit_pca_kmeans_assignments(features=frames_features.reshape(num_frames_local * num_tokens, dim), cfg=cfg)
        labels_2d = fit_result.cluster_labels.reshape(num_frames_local, num_tokens)
        target_labels = labels_2d[0]

        patch_h = int(sample.gt_mask_np.shape[0] // sample.patch_size)
        patch_w = int(sample.gt_mask_np.shape[1] // sample.patch_size)
        if patch_h * patch_w != int(target_labels.shape[0]):
            raise ValueError(f'Patch/grid mismatch for frame {frame_idx}: {target_labels.shape[0]} vs {(patch_h, patch_w)}')

        target_label_map = target_labels.reshape(patch_h, patch_w)
        selected_cluster = _select_cluster(target_label_map, args.selection_rule, int(args.selected_cluster))
        target_selected_mask = (target_label_map == selected_cluster).astype(np.float32)
        target_selected_up = upsample_patch_map(target_selected_mask, tuple(int(x) for x in sample.gt_mask_np.shape[:2]))
        target_assignment_up = upsample_patch_map(target_label_map.astype(np.float32), tuple(int(x) for x in sample.gt_mask_np.shape[:2]))

        crop_bbox, cropped_image, cropped_arrays = crop_to_content_bbox(
            sample.image_np,
            target_selected_up,
            target_assignment_up,
        )
        selected_up = cropped_arrays[0]
        assignment_up = cropped_arrays[1]
        assignment_rgb = _assignment_rgb(np.rint(assignment_up).astype(np.int64))
        selected_overlay = _overlay_selected(cropped_image, selected_up > 0.5)

        frame_dir = out_root / f'frame_{int(frame_idx):05d}'
        frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cropped_image).save(frame_dir / 'input.png')
        Image.fromarray(assignment_rgb).save(frame_dir / 'cluster_assignment.png')
        Image.fromarray(selected_overlay).save(frame_dir / 'selected_cluster_overlay.png')
        Image.fromarray(((selected_up > 0.5).astype(np.uint8) * 255)).save(frame_dir / 'selected_cluster_mask.png')

        for cluster_id in np.unique(target_label_map):
            cluster_mask = (target_label_map == int(cluster_id)).astype(np.float32)
            cluster_up = upsample_patch_map(cluster_mask, tuple(int(x) for x in sample.gt_mask_np.shape[:2]))
            _bbox, cluster_image, [cluster_up] = crop_to_content_bbox(sample.image_np, cluster_up)
            Image.fromarray(_overlay_selected(cluster_image, cluster_up > 0.5)).save(frame_dir / f'cluster_{int(cluster_id)}_overlay.png')
            Image.fromarray(((cluster_up > 0.5).astype(np.uint8) * 255)).save(frame_dir / f'cluster_{int(cluster_id)}_mask.png')

        with (frame_dir / 'meta.json').open('w', encoding='utf-8') as handle:
            json.dump({
                'frame_idx': int(frame_idx),
                'feature_key': feature_key,
                'selected_cluster': int(selected_cluster),
                'clusters_present': [int(x) for x in np.unique(target_label_map).tolist()],
                'cluster_counts_target_frame': {str(k): int(v) for k, v in _cluster_counts(target_label_map).items()},
                'content_bbox_xyxy': [int(crop_bbox[2]), int(crop_bbox[0]), int(crop_bbox[3]), int(crop_bbox[1])],
                'input_frame_indices': list(sample.input_frame_indices),
                'projected_shape': list(fit_result.projected_target.shape),
                'pca_dim_used': int(fit_result.pca_dim_used),
            }, handle, indent=2)

        summary['frames'].append({
            'frame_idx': int(frame_idx),
            'selected_cluster': int(selected_cluster),
            'feature_key': feature_key,
            'clusters_present': [int(x) for x in np.unique(target_label_map).tolist()],
        })

        if int(frame_idx) in strip_frame_choices:
            strip_input.append(cropped_image)
            strip_assignment.append(assignment_rgb)
            strip_selected.append(selected_overlay)

    _save_tile_strip(strip_input, out_root / 'strip_input')
    _save_tile_strip(strip_assignment, out_root / 'strip_cluster_assignment')
    _save_tile_strip(strip_selected, out_root / 'strip_selected_cluster')

    with (out_root / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({
        'model': model_name,
        'dataset': args.dataset,
        'sequence': spec.sequence,
        'feature_group': group,
        'block_idx': int(args.block_idx),
        'out_root': str(out_root),
        'num_frames': len(frame_indices),
    }, indent=2))


if __name__ == '__main__':
    main()
