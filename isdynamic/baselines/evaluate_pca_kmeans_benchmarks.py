from __future__ import annotations

import argparse
import json
from pathlib import Path

from isdynamic.baselines.pca_kmeans_data import canonical_model_name, default_groups_for_model
from isdynamic.baselines.pca_kmeans_io import eval_json_path, mask_tree_root_for_variant, read_json
from isdynamic.viz.evaluate_motion_masks import _nanmean, _normalize_model_name, evaluate_sequence
from isdynamic.viz.motion_mask_benchmarks import (
    benchmark_display_name,
    benchmark_output_dirname,
    list_sequence_specs,
    load_sequence_paths_from_spec,
)

BENCHMARKS = ("davis_all", "segtrackv2", "fbms59")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCA+kmeans coarse masks on DAVIS-all, SegTrackv2, and FBMS-59.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant_name", type=str, required=True)
    parser.add_argument("--mask_root", type=str, default="")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--davis_root", type=str, default="")
    parser.add_argument("--segtrack_root", type=str, default="")
    parser.add_argument("--fbms_root", type=str, default="")
    parser.add_argument("--boundary_th", type=float, default=0.008)
    return parser.parse_args()


def _resolve_group_label(args: argparse.Namespace) -> str:
    if args.groups:
        groups = [item.strip() for item in args.groups.split(",") if item.strip()]
        if groups:
            return "__".join(groups)
    if args.group:
        return args.group.strip()
    if args.config_json:
        payload = read_json(Path(args.config_json))
        groups = [str(item).strip() for item in payload.get("groups", []) if str(item).strip()]
        if groups:
            return "__".join(groups)
    return "__".join(default_groups_for_model(args.model))


def _evaluate_one(*, mask_root: Path, model: str, group_label: str, benchmark: str, davis_root: str, segtrack_root: str, fbms_root: str, boundary_th: float) -> dict[str, object]:
    sequence_specs = list_sequence_specs(
        benchmark,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
    )
    benchmark_dirname = benchmark_output_dirname(benchmark)
    sequence_results = []
    global_j_mean = []
    global_j_recall = []
    global_j_decay = []
    global_f_mean = []
    global_f_recall = []
    global_f_decay = []
    for spec in sequence_specs:
        image_paths, mask_refs, sequence_tag = load_sequence_paths_from_spec(
            spec,
            davis_root=davis_root,
            segtrack_root=segtrack_root,
            fbms_root=fbms_root,
            fbms_include_unannotated=False,
        )
        sequence_root = mask_root / benchmark_dirname / sequence_tag
        result = evaluate_sequence(
            sequence_root=sequence_root,
            mask_root=mask_root,
            model=model,
            group=group_label,
            benchmark=benchmark,
            variant="bilinear_upsample",
            image_paths=image_paths,
            mask_refs=mask_refs,
            boundary_th=float(boundary_th),
        )
        payload = {
            "sequence": spec.sequence,
            "sequence_tag": sequence_tag,
            "dataset": spec.dataset,
            "benchmark": spec.benchmark,
            "fbms_split": spec.fbms_split,
            **result,
        }
        sequence_results.append(payload)
        global_j_mean.append(payload["metrics"]["J"]["M"])
        global_j_recall.append(payload["metrics"]["J"]["R"])
        global_j_decay.append(payload["metrics"]["J"]["D"])
        global_f_mean.append(payload["metrics"]["F"]["M"])
        global_f_recall.append(payload["metrics"]["F"]["R"])
        global_f_decay.append(payload["metrics"]["F"]["D"])
    summary = {
        "model": model,
        "group": group_label,
        "benchmark": benchmark_dirname,
        "benchmark_display_name": benchmark_display_name(benchmark),
        "variant": "bilinear_upsample",
        "mask_root": str(mask_root),
        "num_sequences": len(sequence_results),
        "num_frames": int(sum(int(item["num_frames"]) for item in sequence_results)),
        "metrics": {
            "J": {"M": _nanmean(global_j_mean), "R": _nanmean(global_j_recall), "D": _nanmean(global_j_decay)},
            "F": {"M": _nanmean(global_f_mean), "R": _nanmean(global_f_recall), "D": _nanmean(global_f_decay)},
        },
        "sequences": sequence_results,
    }
    summary["metrics"]["J_and_F"] = {
        "M": float(0.5 * (summary["metrics"]["J"]["M"] + summary["metrics"]["F"]["M"])),
        "R": float(0.5 * (summary["metrics"]["J"]["R"] + summary["metrics"]["F"]["R"])),
        "D": float(0.5 * (summary["metrics"]["J"]["D"] + summary["metrics"]["F"]["D"])),
    }
    return summary


def main() -> None:
    args = parse_args()
    model = _normalize_model_name(args.model)
    canonical_model_name(model)
    group_label = _resolve_group_label(args)
    mask_root = Path(args.mask_root) if args.mask_root else mask_tree_root_for_variant(model_name=model, variant_name=args.variant_name)
    for benchmark in BENCHMARKS:
        summary = _evaluate_one(
            mask_root=mask_root,
            model=model,
            group_label=group_label,
            benchmark=benchmark,
            davis_root=args.davis_root,
            segtrack_root=args.segtrack_root,
            fbms_root=args.fbms_root,
            boundary_th=float(args.boundary_th),
        )
        out_path = eval_json_path(model_name=model, variant_name=args.variant_name, benchmark_name=benchmark)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps({
            'model': model,
            'variant_name': args.variant_name,
            'group': group_label,
            'benchmark': benchmark,
            'J': summary['metrics']['J'],
            'output_json': str(out_path),
        }, indent=2))


if __name__ == '__main__':
    main()
