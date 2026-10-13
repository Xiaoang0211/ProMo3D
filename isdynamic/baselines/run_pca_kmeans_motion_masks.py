from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from isdynamic.baselines.pca_kmeans_core import (
    BLOCK_MODE_CHOICES,
    CLUSTER_RULE_CHOICES,
    NORM_CHOICES,
    PREPROCESS_SCOPE_CHOICES,
    PCAKMeansConfig,
    all_cluster_subsets,
    fit_pca_kmeans,
    labels_from_rule,
)
from isdynamic.baselines.pca_kmeans_data import (
    DATASET_CHOICES,
    YOUTUBE_VOS_VAL,
    SampleContext,
    canonical_model_name,
    default_groups_for_model,
    extract_sample_context,
    list_specs_for_dataset,
    list_target_frame_indices_for_spec,
    load_model_resources,
    resolve_dataset_roots,
)
from isdynamic.baselines.pca_kmeans_io import (
    ensure_pca_kmeans_root,
    frame_output_dir_for_variant,
    mask_tree_root_for_variant,
    read_json,
    write_json,
)
from isdynamic.viz.motion_mask_export_common import build_motion_mask_meta
from isdynamic.viz.motion_mask_utils import save_motion_mask_outputs
from isdynamic.viz.scoremap_utils import crop_to_content_bbox, upsample_patch_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PCA+kmeans no-probe motion masks using the same frozen token features as PMD."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--sequence", type=str, default="", help="Optional single-sequence filter.")
    parser.add_argument("--frame_idx", type=int, default=-1, help="Optional single-frame export. Negative means all frames.")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--context_stride", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--dust3r_model_name", type=str, default="")
    parser.add_argument("--youtubevos_root", type=str, default="/home/xiaoang/Datasets/Youtube-VOS")
    parser.add_argument("--davis_root", type=str, default="")
    parser.add_argument("--segtrack_root", type=str, default="")
    parser.add_argument("--fbms_root", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--block_mode", type=str, default="single_block", choices=BLOCK_MODE_CHOICES)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--block_indices", type=str, default="")
    parser.add_argument("--cluster_rule", type=str, default="minority", choices=CLUSTER_RULE_CHOICES)
    parser.add_argument("--norm", type=str, default="zscore_l2", choices=NORM_CHOICES)
    parser.add_argument("--preprocess_scope", type=str, default="frame_group", choices=PREPROCESS_SCOPE_CHOICES)
    parser.add_argument("--pca_dim", type=int, default=8, choices=[3, 8, 16])
    parser.add_argument("--num_clusters", type=int, default=2, choices=[2, 3, 5])
    parser.add_argument("--anchor_idx", type=int, default=0)
    parser.add_argument("--variant_name", type=str, default="")
    parser.add_argument("--out_root", type=str, default="", help="Top-level PCA+kmeans output root. Results are saved under out_root/masks/<model>/<variant>.")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"],
    )
    parser.add_argument("--resolution_set", type=int, default=518, choices=[504, 512, 518])
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--preprocess_mode", type=str, default="native", choices=["native", "aspect_pad"])
    parser.add_argument("--mapanything_machine", type=str, default="aws")
    parser.add_argument("--use_multimodal_pi3x", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--block_workers", type=int, default=4)
    return parser.parse_args()


def _parse_groups(args: argparse.Namespace) -> list[str]:
    if args.groups:
        groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    elif args.group:
        groups = [args.group.strip()]
    else:
        groups = default_groups_for_model(args.model)
    if not groups:
        raise ValueError("No groups resolved for PCA+kmeans baseline.")
    return groups


def _resolve_variant_name(args: argparse.Namespace) -> str:
    if args.variant_name:
        return args.variant_name.strip()
    if args.config_json:
        return "pca_kmeans_val_selected"
    if args.cluster_rule == "oracle":
        return "pca_kmeans_oracle"
    return "pca_kmeans"


def _feature_keys_for_groups(
    feature_key_order: Sequence[str],
    groups: Sequence[str],
) -> list[str]:
    prefixes = tuple(f"{group}_" for group in groups)
    return [key for key in feature_key_order if key.startswith(prefixes)]


def _parse_block_indices(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(',') if item.strip()]
    return [int(item) for item in values]


def _select_feature_keys(
    *,
    feature_key_order: Sequence[str],
    groups: Sequence[str],
    block_mode: str,
    block_idx: int,
    block_indices_raw: str = "",
) -> list[str]:
    candidate_keys = _feature_keys_for_groups(feature_key_order, groups)
    if not candidate_keys:
        raise RuntimeError(f"No feature keys found for groups={list(groups)}")

    block_indices = _parse_block_indices(block_indices_raw)
    if block_indices:
        selected = []
        for idx in block_indices:
            if not (0 <= int(idx) < len(candidate_keys)):
                raise IndexError(
                    f"block index out of range: {idx} for groups={list(groups)} with {len(candidate_keys)} candidate blocks"
                )
            selected.append(candidate_keys[int(idx)])
        return selected

    if block_mode == "multi_block_aggregate":
        return candidate_keys
    if len(groups) != 1:
        raise ValueError("single_block mode requires exactly one group.")
    if not (0 <= int(block_idx) < len(candidate_keys)):
        raise IndexError(
            f"block_idx out of range: {block_idx} for group={groups[0]} with {len(candidate_keys)} candidate blocks"
        )
    return [candidate_keys[int(block_idx)]]


def _infer_grid_shape(num_tokens: int, patch_size: int, target_hw: tuple[int, int]) -> tuple[int, int]:
    grid_h = int(target_hw[0] // patch_size)
    grid_w = int(target_hw[1] // patch_size)
    if grid_h * grid_w != int(num_tokens):
        raise ValueError(
            f"Token/grid mismatch: tokens={num_tokens}, grid=({grid_h},{grid_w}), patch_size={patch_size}, target_hw={target_hw}"
        )
    return grid_h, grid_w


def _iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
    gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
    intersection = int(np.logical_and(pred > 0, gt > 0).sum())
    union = int(np.logical_or(pred > 0, gt > 0).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def _evaluate_patch_labels_against_gt(
    *,
    patch_labels: np.ndarray,
    target_hw: tuple[int, int],
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
) -> float:
    patch_score = np.asarray(patch_labels, dtype=np.float32)
    upsampled = upsample_patch_map(patch_score, target_hw)
    _bbox, _cropped_image, cropped = crop_to_content_bbox(
        image_np,
        upsampled,
        np.asarray(gt_mask_np, dtype=np.float32),
    )
    pred_mask = (cropped[0] > 0.5).astype(np.uint8)
    gt_mask = (cropped[1] > 0.5).astype(np.uint8)
    return _iou_score(pred_mask, gt_mask)


def _oracle_selector_for_grid(
    *,
    target_hw: tuple[int, int],
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    patch_size: int,
) -> Any:
    def selector(cluster_ids: Sequence[int], cluster_labels: np.ndarray, _fit_result) -> Iterable[int]:
        grid_h, grid_w = _infer_grid_shape(cluster_labels.shape[0], patch_size, target_hw)
        best_subset: tuple[int, ...] = tuple()
        best_iou = -1.0
        for subset in all_cluster_subsets(cluster_ids):
            token_labels = np.array([1 if int(cluster_id) in subset else 0 for cluster_id in cluster_labels], dtype=np.uint8)
            patch_labels = token_labels.reshape(grid_h, grid_w)
            iou = _evaluate_patch_labels_against_gt(
                patch_labels=patch_labels,
                target_hw=target_hw,
                image_np=image_np,
                gt_mask_np=gt_mask_np,
            )
            if iou > best_iou:
                best_iou = iou
                best_subset = subset
        return best_subset

    return selector


def _cluster_single_token_matrix(
    *,
    target_features: np.ndarray,
    fit_features: np.ndarray,
    cfg: PCAKMeansConfig,
    patch_size: int,
    target_hw: tuple[int, int],
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
) -> np.ndarray:
    fit_result = fit_pca_kmeans(target_features=target_features, fit_features=fit_features, cfg=cfg)
    oracle_selector = None
    if cfg.cluster_rule == "oracle":
        oracle_selector = _oracle_selector_for_grid(
            target_hw=target_hw,
            image_np=image_np,
            gt_mask_np=gt_mask_np,
            patch_size=patch_size,
        )
    token_labels = labels_from_rule(
        fit_result=fit_result,
        rule=cfg.cluster_rule,
        anchor_idx=cfg.anchor_idx,
        oracle_selector=oracle_selector,
    )
    grid_h, grid_w = _infer_grid_shape(token_labels.shape[0], patch_size, target_hw)
    return token_labels.reshape(grid_h, grid_w).astype(np.float32)


def _patch_mask_for_feature_value(
    *,
    feature_value: Any,
    cfg: PCAKMeansConfig,
    patch_size: int,
    target_hw: tuple[int, int],
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
) -> np.ndarray:
    if torch.is_tensor(feature_value):
        array = feature_value.detach().cpu().numpy().astype(np.float32)
    elif isinstance(feature_value, np.ndarray):
        array = feature_value.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported feature value type: {type(feature_value)!r}")

    if array.ndim == 4:
        # [B,S,P,C]; use batch 0, target frame 0; optionally fit over the whole frame group.
        target_features = array[0, 0]
        fit_features = target_features if cfg.preprocess_scope == "target_frame_only" else array[0].reshape(-1, array.shape[-1])
        return _cluster_single_token_matrix(
            target_features=target_features,
            fit_features=fit_features,
            cfg=cfg,
            patch_size=patch_size,
            target_hw=target_hw,
            image_np=image_np,
            gt_mask_np=gt_mask_np,
        )

    if array.ndim == 3:
        # Dust3R-style repeated target tokens for multiple references: cluster each replica and average.
        replicate_masks: list[np.ndarray] = []
        shared_fit_features = None
        if cfg.preprocess_scope == "frame_group":
            shared_fit_features = array.reshape(-1, array.shape[-1])
        for replicate_idx in range(array.shape[0]):
            target_features = array[replicate_idx]
            fit_features = target_features if shared_fit_features is None else shared_fit_features
            replicate_masks.append(
                _cluster_single_token_matrix(
                    target_features=target_features,
                    fit_features=fit_features,
                    cfg=cfg,
                    patch_size=patch_size,
                    target_hw=target_hw,
                    image_np=image_np,
                    gt_mask_np=gt_mask_np,
                )
            )
        return np.stack(replicate_masks, axis=0).mean(axis=0).astype(np.float32)

    raise ValueError(f"Unsupported feature tensor shape for PCA+kmeans: {array.shape}")


def _build_patch_score(
    *,
    sample: SampleContext,
    selected_keys: Sequence[str],
    cfg: PCAKMeansConfig,
    block_workers: int = 1,
) -> np.ndarray:
    target_hw = tuple(int(x) for x in sample.gt_mask_np.shape[:2])

    def _run_one(key: str) -> np.ndarray:
        return _patch_mask_for_feature_value(
            feature_value=sample.feature_map[key],
            cfg=cfg,
            patch_size=sample.patch_size,
            target_hw=target_hw,
            image_np=sample.image_np,
            gt_mask_np=sample.gt_mask_np,
        )

    if not selected_keys:
        raise RuntimeError("No patch maps computed for PCA+kmeans baseline.")

    if len(selected_keys) == 1 or int(block_workers) <= 1:
        patch_maps = [_run_one(key) for key in selected_keys]
    else:
        max_workers = min(len(selected_keys), max(1, int(block_workers)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            patch_maps = list(executor.map(_run_one, selected_keys))

    if len(patch_maps) == 1:
        return patch_maps[0]
    return np.stack(patch_maps, axis=0).mean(axis=0).astype(np.float32)


def _build_cfg_from_args(args: argparse.Namespace) -> PCAKMeansConfig:
    if args.config_json:
        payload = read_json(Path(args.config_json))
        return PCAKMeansConfig(
            norm=str(payload["norm"]),
            preprocess_scope=str(payload["preprocess_scope"]),
            pca_dim=int(payload["pca_dim"]),
            num_clusters=int(payload["num_clusters"]),
            cluster_rule=str(payload["cluster_rule"]),
            anchor_idx=int(payload.get("anchor_idx", 0)),
            random_state=int(payload.get("random_state", args.random_state)),
            n_init=int(payload.get("n_init", args.n_init)),
        )
    return PCAKMeansConfig(
        norm=args.norm,
        preprocess_scope=args.preprocess_scope,
        pca_dim=int(args.pca_dim),
        num_clusters=int(args.num_clusters),
        cluster_rule=args.cluster_rule,
        anchor_idx=int(args.anchor_idx),
        random_state=int(args.random_state),
        n_init=int(args.n_init),
    )


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)
    selection_payload = read_json(Path(args.config_json)) if args.config_json else {}
    groups = [str(item).strip() for item in selection_payload.get('groups', []) if str(item).strip()] or _parse_groups(args)
    resolved_block_mode = str(selection_payload.get('block_mode', args.block_mode))
    resolved_block_idx = int(selection_payload.get('block_idx', args.block_idx))
    cfg = _build_cfg_from_args(args)
    cfg.validate()
    variant_name = _resolve_variant_name(args)

    roots = resolve_dataset_roots(
        youtubevos_root=args.youtubevos_root,
        davis_root=args.davis_root or None,
        segtrack_root=args.segtrack_root or None,
        fbms_root=args.fbms_root or None,
    )
    device = torch.device(args.device)
    resources = load_model_resources(
        model_name=model_name,
        device=device,
        checkpoint=args.checkpoint or None,
        model_dir=args.model_dir or None,
        weights=args.weights or None,
        dust3r_model_name=args.dust3r_model_name or None,
        use_multimodal_pi3x=bool(args.use_multimodal_pi3x),
        mapanything_machine=args.mapanything_machine,
    )
    sequence_specs = list_specs_for_dataset(args.dataset, roots)
    if args.sequence:
        sequence_specs = [spec for spec in sequence_specs if spec.sequence == args.sequence]
    if not sequence_specs:
        raise RuntimeError(f"No sequence specs resolved for dataset={args.dataset} sequence_filter={args.sequence!r}")

    ensure_pca_kmeans_root()
    if args.out_root:
        base_root = Path(args.out_root)
        out_root = base_root / "masks" / model_name / variant_name
    else:
        out_root = mask_tree_root_for_variant(model_name=model_name, variant_name=variant_name)
    out_root.mkdir(parents=True, exist_ok=True)

    write_json(
        out_root / 'manifest.json',
        {
            'baseline': 'pca_kmeans',
            'variant_name': variant_name,
            'model': model_name,
            'dataset': args.dataset,
            'groups': list(groups),
            'block_mode': resolved_block_mode,
            'block_idx': int(resolved_block_idx),
            'norm': cfg.norm,
            'preprocess_scope': cfg.preprocess_scope,
            'pca_dim': int(cfg.pca_dim),
            'num_clusters': int(cfg.num_clusters),
            'cluster_rule': cfg.cluster_rule,
            'anchor_idx': int(cfg.anchor_idx),
            'random_state': int(cfg.random_state),
            'n_init': int(cfg.n_init),
            'config_json': args.config_json,
            'supervision': 'test-GT oracle' if cfg.cluster_rule == 'oracle' else ('YouTube-VOS val only' if args.config_json else 'none'),
        },
    )

    total_frames = 0
    start_time = perf_counter()
    for spec in sequence_specs:
        frame_indices = [int(args.frame_idx)] if args.frame_idx >= 0 else list_target_frame_indices_for_spec(spec, roots)
        if not frame_indices:
            continue
        print(f"[pca_kmeans] dataset={args.dataset} sequence={spec.sequence} frames={len(frame_indices)}")
        for frame_idx in frame_indices:
            target_frame_dir = frame_output_dir_for_variant(
                model_name=model_name,
                variant_name=variant_name,
                benchmark_name=spec.benchmark,
                sequence_tag=spec.sequence_tag,
                frame_idx=frame_idx,
            )
            if args.skip_existing and (target_frame_dir / "meta.json").exists():
                continue

            sample_t0 = perf_counter()
            sample = extract_sample_context(
                resources=resources,
                spec=spec,
                roots=roots,
                frame_idx=frame_idx,
                groups=groups,
                context_size=args.context_size,
                context_stride=args.context_stride,
                device=device,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                resolution_set=args.resolution_set,
                image_size=args.image_size,
                preprocess_mode=args.preprocess_mode,
            )
            selected_keys = _select_feature_keys(
                feature_key_order=sample.feature_key_order,
                groups=groups,
                block_mode=resolved_block_mode,
                block_idx=int(resolved_block_idx),
                block_indices_raw=args.block_indices,
            )
            patch_score = _build_patch_score(
                sample=sample,
                selected_keys=selected_keys,
                cfg=cfg,
                block_workers=int(args.block_workers),
            )
            additional_seconds = perf_counter() - sample_t0

            extra_meta = {
                "baseline": "pca_kmeans",
                "variant_name": variant_name,
                "supervision": "test-GT oracle" if cfg.cluster_rule == "oracle" else ("YouTube-VOS val only" if args.config_json else "none"),
                "block_mode": resolved_block_mode,
                "groups": list(groups),
                "selected_blocks": list(selected_keys),
                "cluster_rule": cfg.cluster_rule,
                "normalization": cfg.norm,
                "preprocess_scope": cfg.preprocess_scope,
                "pca_dim": int(cfg.pca_dim),
                "num_clusters": int(cfg.num_clusters),
                "anchor_idx": int(cfg.anchor_idx),
                "random_state": int(cfg.random_state),
                "n_init": int(cfg.n_init),
                "oracle_non_deployable_upper_bound": bool(cfg.cluster_rule == "oracle"),
            }
            meta = build_motion_mask_meta(
                dataset=sample.dataset_name,
                benchmark=sample.benchmark_name or sample.dataset_name,
                sequence=sample.sequence,
                sequence_tag=sample.sequence_tag,
                frame_idx=sample.frame_idx,
                probe_root=args.config_json or "pca_kmeans",
                threshold=0.5,
                query_name="pca_kmeans",
                query_idx=int(cfg.anchor_idx),
                patch_size=sample.patch_size,
                input_frame_indices=sample.input_frame_indices,
                image_paths=sample.image_paths,
                target_mask_ref=sample.mask_refs[sample.frame_idx] if sample.frame_idx < len(sample.mask_refs) else None,
                backbone_inference_seconds=0.0,
                probe_scoring_seconds=float(additional_seconds),
                averaging_seconds=0.0,
                extra_meta=extra_meta,
            )
            save_motion_mask_outputs(
                out_dir=target_frame_dir,
                averaged_patch_score=patch_score,
                target_hw=tuple(int(x) for x in sample.gt_mask_np.shape[:2]),
                threshold=0.5,
                threshold_mode="fixed",
                connected_components=False,
                meta=meta,
                image_np=sample.image_np,
                gt_mask_np=sample.gt_mask_np,
                threshold_offset=0.0,
            )
            total_frames += 1

    elapsed = perf_counter() - start_time
    print(
        f"[pca_kmeans] done model={model_name} dataset={args.dataset} variant={variant_name} "
        f"frames={total_frames} out_root={out_root} elapsed_seconds={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
