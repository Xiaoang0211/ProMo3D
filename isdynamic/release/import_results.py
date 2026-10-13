"""Copy small archived metrics to a reference tree without copying masks or tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

LAYOUT = {
    "pi3x": "pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x",
    "pi3": "pi3/rank_8_pi3/isdynamic_blockwise_youtubevos_pi3",
    "vggt": "vggt/rank_8/isdynamic_blockwise_youtubevos",
    "da3": "depth_anything_3/rank_8/isdynamic_blockwise_youtubevos",
    "mapanything": "map_anything/rank_8/isdynamic_blockwise_youtubevos",
    "dust3r": "dust3r/rank_8/isdynamic_blockwise_youtubevos",
}
MASK_MODELS = {"da3": "depth_anything_3", "mapanything": "map_anything"}


def copy_controls(source, destination):
    variants = {"correct_rank_4": "capacity/rank_4", "correct_rank_8": "capacity/rank_8",
                "shuffled_rank_8": "controls/within_group_shuffled_labels/rank_8"}
    for name, rel in variants.items():
        root = source / "pi3x_probe_controls/probes" / rel
        for path in sorted(root.glob("pair_lowrank_quadratic/*/*/metrics.json")):
            compact(path, destination / "controls/pi3x" / name / path.relative_to(root))


def compact(source, destination, keys=None):
    if destination.exists():
        raise FileExistsError(f"Reference destination exists: {destination}; choose a new --destination.")
    data = json.loads(source.read_text())
    if keys:
        data = {key: data[key] for key in keys if key in data}
    data["archived_source"] = str(source.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--experiment_outputs", type=Path)
    parser.add_argument("--source_val_eval", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("results/reference"))
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(f"Choose a fresh reference destination: {args.destination}")
    for model, rel in LAYOUT.items():
        source = args.outputs / "model_outputs" / rel
        dest = args.destination / "main/probes" / model / "rank_8"
        for path in sorted(source.rglob("metrics.json")):
            compact(path, dest / path.relative_to(source))
        if (source / "summary.json").exists():
            compact(source / "summary.json", dest / "summary.json")
        mask_model = MASK_MODELS.get(model, model)
        eval_root = args.outputs / "isdynamic_motion_masks_eval" / mask_model
        for path in sorted(eval_root.glob("*/*/*.json")):
            compact(path, args.destination / "main/eval" / mask_model / path.relative_to(eval_root),
                    keys=("metrics", "model", "group", "benchmark", "variant", "num_sequences", "num_frames"))
        group = "dec_view1__head_in_view1" if model == "dust3r" else "inter_frame_block"
        mask_root = args.outputs / "isdynamic_motion_masks" / mask_model / group / "davis_all"
        meta = next(iter(sorted(mask_root.glob("*/bilinear_upsample/frame_*/meta.json"))), None)
        if meta:
            compact(meta, args.destination / "main/masks" / mask_model / group / "davis_all" / meta.relative_to(mask_root))
        if args.source_val_eval:
            src = args.source_val_eval / mask_model
            for path in sorted(src.glob("*/*/*.json")):
                compact(path, args.destination / "experiments/source_validation/eval" / mask_model / path.relative_to(src), keys=("metrics", "benchmark", "variant"))
    if args.experiment_outputs:
        copy_controls(args.experiment_outputs, args.destination / "experiments")
        runtime = args.experiment_outputs / "runtime"
        for source in sorted(runtime.glob("*/*")):
            if source.is_file() and source.suffix in {".json", ".csv"}:
                destination = args.destination / "experiments/runtime" / source.relative_to(runtime)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    print(f"[reference] {args.destination}")


if __name__ == "__main__":
    main()
