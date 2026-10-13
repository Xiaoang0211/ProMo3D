from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from isdynamic.baselines.pca_kmeans_data import canonical_model_name, default_groups_for_model
from isdynamic.baselines.pca_kmeans_io import backbone_csv_path, combined_csv_path, eval_json_path, mask_tree_root_for_variant, read_json

MODELS = ["pi3", "pi3x", "vggt", "vggt_omega", "depth_anything_3", "map_anything", "dust3r"]
BENCHMARKS = ("davis_all", "segtrackv2", "fbms59")
PCA_VARIANTS = ("pca_kmeans", "pca_kmeans_val_selected", "pca_kmeans_oracle")
DEFAULT_PMD_GROUP = {
    "pi3": "inter_frame_block",
    "pi3x": "inter_frame_block",
    "vggt": "inter_frame_block",
    "vggt_omega": "inter_frame_block",
    "depth_anything_3": "inter_frame_block",
    "map_anything": "inter_frame_block",
    "dust3r": "dec_view1__head_in_view1",
}
FIELDNAMES = [
    "Method",
    "Supervision",
    "Mapping rule",
    "Backbone",
    "Block mode",
    "Blocks",
    "Preprocess scope",
    "Normalization",
    "PCA dim",
    "K",
    "DAVIS-all J-M",
    "DAVIS-all J-R",
    "SegTrackv2 J-M",
    "SegTrackv2 J-R",
    "FBMS-59 J-M",
    "FBMS-59 J-R",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PCA+kmeans and PMD comparison CSV tables.")
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    return parser.parse_args()


def _load_eval_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"J_M": float('nan'), "J_R": float('nan')}
    payload = read_json(path)
    return {
        "J_M": float(payload["metrics"]["J"]["M"]),
        "J_R": float(payload["metrics"]["J"]["R"]),
    }


def _load_manifest(model: str, variant: str) -> dict[str, Any]:
    manifest_path = mask_tree_root_for_variant(model_name=model, variant_name=variant) / 'manifest.json'
    if manifest_path.exists():
        return read_json(manifest_path)
    if variant == 'pca_kmeans':
        return {
            'groups': default_groups_for_model(model),
            'block_mode': 'multi_block_aggregate',
            'block_idx': 0,
            'norm': 'zscore_l2',
            'preprocess_scope': 'frame_group',
            'pca_dim': 8,
            'num_clusters': 2,
            'cluster_rule': 'minority',
            'supervision': 'none',
        }
    if variant == 'pca_kmeans_oracle':
        return {
            'groups': default_groups_for_model(model),
            'block_mode': 'multi_block_aggregate',
            'block_idx': 0,
            'norm': 'zscore_l2',
            'preprocess_scope': 'frame_group',
            'pca_dim': 8,
            'num_clusters': 2,
            'cluster_rule': 'oracle',
            'supervision': 'test-GT oracle',
        }
    return {}


def _blocks_text(manifest: dict[str, Any]) -> str:
    groups = [str(item) for item in manifest.get('groups', [])]
    block_mode = str(manifest.get('block_mode', ''))
    if block_mode == 'single_block' and groups:
        return f"{groups[0]}[{int(manifest.get('block_idx', 0))}]"
    return "__".join(groups)


def _build_pca_row(model: str, variant: str) -> dict[str, Any]:
    manifest = _load_manifest(model, variant)
    row = {
        'Method': variant,
        'Supervision': manifest.get('supervision', 'none'),
        'Mapping rule': manifest.get('cluster_rule', ''),
        'Backbone': model,
        'Block mode': manifest.get('block_mode', ''),
        'Blocks': _blocks_text(manifest),
        'Preprocess scope': manifest.get('preprocess_scope', ''),
        'Normalization': manifest.get('norm', ''),
        'PCA dim': manifest.get('pca_dim', ''),
        'K': manifest.get('num_clusters', ''),
    }
    for benchmark, jm_key, jr_key in (
        ('davis_all', 'DAVIS-all J-M', 'DAVIS-all J-R'),
        ('segtrackv2', 'SegTrackv2 J-M', 'SegTrackv2 J-R'),
        ('fbms59', 'FBMS-59 J-M', 'FBMS-59 J-R'),
    ):
        metrics = _load_eval_metrics(eval_json_path(model_name=model, variant_name=variant, benchmark_name=benchmark))
        row[jm_key] = metrics['J_M']
        row[jr_key] = metrics['J_R']
    return row


def _build_pmd_row(model: str) -> dict[str, Any]:
    group = DEFAULT_PMD_GROUP[model]
    row = {
        'Method': 'PMD low-rank probe',
        'Supervision': '160 YouTube-VOS videos',
        'Mapping rule': 'learned low-rank probe',
        'Backbone': model,
        'Block mode': 'multi_block_aggregate',
        'Blocks': group,
        'Preprocess scope': '',
        'Normalization': '',
        'PCA dim': '',
        'K': '',
    }
    for benchmark, jm_key, jr_key in (
        ('davis_all', 'DAVIS-all J-M', 'DAVIS-all J-R'),
        ('segtrackv2', 'SegTrackv2 J-M', 'SegTrackv2 J-R'),
        ('fbms59', 'FBMS-59 J-M', 'FBMS-59 J-R'),
    ):
        path = Path(os.environ.get('ISDYNAMIC_OUTPUTS_ROOT', 'outputs')) / 'isdynamic_motion_masks_eval' / model / group / 'bilinear_upsample' / f'{benchmark}.json'
        metrics = _load_eval_metrics(path)
        row[jm_key] = metrics['J_M']
        row[jr_key] = metrics['J_R']
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    models = [canonical_model_name(item) for item in args.models.split(',') if item.strip()]
    combined_rows: list[dict[str, Any]] = []
    for model in models:
        rows = [_build_pca_row(model, variant) for variant in PCA_VARIANTS]
        rows.append(_build_pmd_row(model))
        _write_csv(backbone_csv_path(model), rows)
        combined_rows.extend(rows)
    _write_csv(combined_csv_path(), combined_rows)
    print(f'Wrote {combined_csv_path()} and per-backbone CSV files.')


if __name__ == '__main__':
    main()
