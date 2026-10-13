from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from isdynamic.baselines.pca_kmeans_core import PCAKMeansConfig
from isdynamic.baselines.pca_kmeans_data import (
    YOUTUBE_VOS_VAL,
    canonical_model_name,
    default_groups_for_model,
    extract_sample_context,
    list_specs_for_dataset,
    list_target_frame_indices_for_spec,
    load_model_resources,
    resolve_dataset_roots,
)
from isdynamic.baselines.pca_kmeans_io import selected_config_path, write_json
from isdynamic.baselines.run_pca_kmeans_motion_masks import _build_patch_score, _select_feature_keys
from isdynamic.viz.scoremap_utils import crop_to_content_bbox, upsample_patch_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PCA+kmeans hyperparameters on YouTube-VOS validation only.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--dust3r_model_name", type=str, default="")
    parser.add_argument("--youtubevos_root", type=str, default="/home/xiaoang/Datasets/Youtube-VOS")
    parser.add_argument("--davis_root", type=str, default="")
    parser.add_argument("--segtrack_root", type=str, default="")
    parser.add_argument("--fbms_root", type=str, default="")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--context_stride", type=int, default=2)
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
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_frames_per_sequence", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=10)
    return parser.parse_args()


def _frame_jm_from_patch_score(
    *,
    patch_score: np.ndarray,
    target_hw: tuple[int, int],
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
) -> float:
    upsampled = upsample_patch_map(np.asarray(patch_score, dtype=np.float32), target_hw)
    _bbox, _cropped_image, cropped = crop_to_content_bbox(
        image_np,
        upsampled,
        np.asarray(gt_mask_np, dtype=np.float32),
    )
    pred = (cropped[0] > 0.5).astype(np.uint8)
    gt = (cropped[1] > 0.5).astype(np.uint8)
    intersection = int(np.logical_and(pred > 0, gt > 0).sum())
    union = int(np.logical_or(pred > 0, gt > 0).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def _candidate_single_block_settings(feature_key_order: tuple[str, ...], groups: list[str]) -> list[tuple[str, list[str], int]]:
    candidates: list[tuple[str, list[str], int]] = []
    for group in groups:
        matching = [key for key in feature_key_order if key.startswith(f"{group}_")]
        for block_idx, _key in enumerate(matching):
            candidates.append(("single_block", [group], block_idx))
    return candidates


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)
    groups = default_groups_for_model(model_name)
    roots = resolve_dataset_roots(
        youtubevos_root=args.youtubevos_root,
        davis_root=args.davis_root or None,
        segtrack_root=args.segtrack_root or None,
        fbms_root=args.fbms_root or None,
    )
    resources = load_model_resources(
        model_name=model_name,
        device=torch.device(args.device),
        checkpoint=args.checkpoint or None,
        model_dir=args.model_dir or None,
        weights=args.weights or None,
        dust3r_model_name=args.dust3r_model_name or None,
        use_multimodal_pi3x=bool(args.use_multimodal_pi3x),
        mapanything_machine=args.mapanything_machine,
    )
    specs = list_specs_for_dataset(YOUTUBE_VOS_VAL, roots)
    if args.max_sequences > 0:
        specs = specs[: int(args.max_sequences)]
    if not specs:
        raise RuntimeError("No YouTube-VOS validation sequences resolved.")

    print(f"[pca_kmeans_select] model={model_name} sequences={len(specs)}")
    bootstrap_sample = extract_sample_context(
        resources=resources,
        spec=specs[0],
        roots=roots,
        frame_idx=0,
        groups=groups,
        context_size=args.context_size,
        context_stride=args.context_stride,
        device=torch.device(args.device),
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        resolution_set=args.resolution_set,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
    )
    block_settings = _candidate_single_block_settings(bootstrap_sample.feature_key_order, groups)
    block_settings.append(("multi_block_aggregate", list(groups), 0))

    search_space: list[tuple[PCAKMeansConfig, str, list[str], int]] = []
    for norm in ("l2", "zscore_l2"):
        for preprocess_scope in ("target_frame_only", "frame_group"):
            for pca_dim in (3, 8, 16):
                for num_clusters in (2, 3, 5):
                    for cluster_rule in ("anchor", "minority", "non_anchor_minority", "farthest_from_anchor"):
                        cfg = PCAKMeansConfig(
                            norm=norm,
                            preprocess_scope=preprocess_scope,
                            pca_dim=pca_dim,
                            num_clusters=num_clusters,
                            cluster_rule=cluster_rule,
                            anchor_idx=0,
                            random_state=int(args.random_state),
                            n_init=int(args.n_init),
                        )
                        for block_mode, block_groups, block_idx in block_settings:
                            search_space.append((cfg, block_mode, block_groups, block_idx))

    best_payload: dict[str, object] | None = None
    best_score = -1.0
    t0 = perf_counter()
    for trial_idx, (cfg, block_mode, block_groups, block_idx) in enumerate(search_space, start=1):
        scores: list[float] = []
        for spec in specs:
            frame_indices = list_target_frame_indices_for_spec(spec, roots)
            if args.max_frames_per_sequence > 0:
                frame_indices = frame_indices[: int(args.max_frames_per_sequence)]
            for frame_idx in frame_indices:
                sample = extract_sample_context(
                    resources=resources,
                    spec=spec,
                    roots=roots,
                    frame_idx=frame_idx,
                    groups=block_groups,
                    context_size=args.context_size,
                    context_stride=args.context_stride,
                    device=torch.device(args.device),
                    process_res=args.process_res,
                    process_res_method=args.process_res_method,
                    resolution_set=args.resolution_set,
                    image_size=args.image_size,
                    preprocess_mode=args.preprocess_mode,
                )
                selected_keys = _select_feature_keys(
                    feature_key_order=sample.feature_key_order,
                    groups=block_groups,
                    block_mode=block_mode,
                    block_idx=block_idx,
                )
                patch_score = _build_patch_score(sample=sample, selected_keys=selected_keys, cfg=cfg)
                scores.append(
                    _frame_jm_from_patch_score(
                        patch_score=patch_score,
                        target_hw=tuple(int(x) for x in sample.gt_mask_np.shape[:2]),
                        image_np=sample.image_np,
                        gt_mask_np=sample.gt_mask_np,
                    )
                )
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_payload = {
                "model": model_name,
                "dataset": YOUTUBE_VOS_VAL,
                "selection_metric": "J-Mean",
                "selection_score": best_score,
                "supervision": "YouTube-VOS val only",
                "norm": cfg.norm,
                "preprocess_scope": cfg.preprocess_scope,
                "pca_dim": cfg.pca_dim,
                "num_clusters": cfg.num_clusters,
                "cluster_rule": cfg.cluster_rule,
                "anchor_idx": cfg.anchor_idx,
                "random_state": cfg.random_state,
                "n_init": cfg.n_init,
                "block_mode": block_mode,
                "groups": list(block_groups),
                "block_idx": int(block_idx),
                "max_sequences": int(args.max_sequences),
                "max_frames_per_sequence": int(args.max_frames_per_sequence),
            }
            print(
                f"[pca_kmeans_select] new_best trial={trial_idx}/{len(search_space)} "
                f"jm={best_score:.4f} cfg={best_payload}"
            )

    if best_payload is None:
        raise RuntimeError("No PCA+kmeans validation config was evaluated.")

    out_path = selected_config_path(model_name)
    write_json(out_path, best_payload)
    elapsed = perf_counter() - t0
    print(f"[pca_kmeans_select] saved {out_path} elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
