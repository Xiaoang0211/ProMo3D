#!/usr/bin/env python3
"""Portable experiment coordinator; heavy work stays in the validated PMD modules."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs")).expanduser().absolute()
TOKENS = Path(os.environ.get("TOKEN_ROOT", OUTPUT / "tokens")).expanduser().absolute()
MAIN = OUTPUT / "main"
EXPERIMENTS = OUTPUT / "experiments"
SPECS = {
    "pi3x": ("pi3x", "PI3", "pi3", "inter_frame_block", "inter_frame_block,intra_frame_block,cam_head"),
    "pi3": ("pi3", "PI3", "pi3", "inter_frame_block", "inter_frame_block,intra_frame_block,cam_head"),
    "vggt": ("vggt", "VGGT", "vggt", "inter_frame_block", "inter_frame_block,intra_frame_block,cam_head,depth_head,point_head"),
    "da3": ("depth_anything_3", "DA3", "da3", "inter_frame_block", "inter_frame_block,intra_frame_block,depth_head"),
    "mapanything": ("map_anything", "MAPANYTHING", "mapanything", "inter_frame_block", "inter_frame_block,intra_frame_block,dense_head,pose_head,scale_head"),
    "dust3r": ("dust3r", "DUST3R", "dust3r", "dec_view1,head_in_view1", "enc,dec_view1,dec_view2,head_in_view1,head_in_view2"),
}
ALIASES = {"depth_anything_3": "da3", "map_anything": "mapanything", "pi3X": "pi3x"}
TEST_SPLITS = "test_davis,test_segtrackv2,test_fbms59"
MASK_BENCHMARKS = "davis_all,segtrackv2,fbms59"
EVAL_BENCHMARKS = "davis2016,davis2017,davis_all,segtrackv2,fbms59"
REFINED = "sam2_prompt_otsu_off0.2_erode2_w30_mean_otsu_off0.3"


def run(command, env=None, dry=False):
    command = [str(item) for item in command]
    print("[run] " + shlex.join(command), flush=True)
    if env:
        visible = {key: value for key, value in env.items()
                   if key.endswith(("ROOT", "DIR", "BIN", "PREPROCESS_MODE")) or key in {
                       "ANCHOR0_SAFEGUARD", "TRAIN_LABEL_CONTROL", "BENCHMARKS",
                       "CONTEXT_SIZE", "CONTEXT_STRIDE", "FEATURE_BATCH_SIZE", "SCORE_BATCH_SIZE",
                       "ISDYNAMIC_MASK_OUTPUT_MODE"}}
        print("[settings] " + json.dumps(visible, sort_keys=True), flush=True)
    if not dry:
        subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})}, check=True)


def python_for(model):
    return os.environ.get(f"{SPECS[model][1]}_PYTHON_BIN", "python")


def module(model, name, *args, env=None, dry=False):
    run([python_for(model), "-m", name, *args], env, dry)


def common_env(scope=MAIN):
    env = {
        "OUTPUTS_DIR": str(scope), "ISDYNAMIC_OUTPUTS_ROOT": str(scope),
        "ANCHOR0_SAFEGUARD": "0", "ANCHOR0_STATIC_PAIRS_PER_BATCH": "0",
        "ANCHOR0_DYNAMIC_PAIRS_PER_BATCH": "0", "POSITION_RESIDUALIZATION": "none",
        "TRAIN_LABEL_CONTROL": "none", "SKIP_EXISTING": "1",
        "CONTEXT_SIZE": "3", "CONTEXT_STRIDE": "2", "DEVICE": "cuda",
        "SCORE_BATCH_SIZE": os.environ.get("SCORE_BATCH_SIZE", "262144"),
        "SAM2_REPO_ROOT": os.environ.get("SAM2_REPO_ROOT", str(ROOT / "models/sam2")),
        "SAM2_CHECKPOINT": os.environ.get("SAM2_CHECKPOINT", str(ROOT / "models/sam2/checkpoints/sam2.1_hiera_large.pt")),
        "SAM2_PYTHON_BIN": os.environ.get("SAM2_PYTHON_BIN", python_for("pi3x")),
        "SAM2_PROMPT_THRESHOLD_MODE": "otsu", "SAM2_PROMPT_THRESHOLD_OFFSET": "0.2",
        "SAM2_PROMPT_ERODE_ITERATIONS": "2", "SAM2_WINDOW_SIZE": "30",
        "SAM2_WINDOW_OVERLAP": "0", "SAM2_PROPAGATE_MODE": "parity",
        "SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_MODE": "otsu",
        "SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET": "0.3",
        "MOTION_MASK_THRESHOLD_MODE": "otsu", "MOTION_MASK_THRESHOLD_OFFSET": "0.2",
    }
    for name in ("DAVIS_ROOT", "SEGTRACK_ROOT", "FBMS_ROOT", "YOUTUBEVOS_ROOT"):
        env[name] = os.environ[name]
    return env


def token_source(model, preprocessing="native"):
    explicit = os.environ.get(f"{model.upper()}_TOKEN_SOURCE")
    return Path(explicit).expanduser().absolute() if explicit else TOKENS / preprocessing / model / "data/isdynamic/youtubevos/tokens"


def dataset_view(dry=False):
    source = Path(os.environ["YOUTUBEVOS_ROOT"])
    view = OUTPUT / "views/youtubevos"
    for target, candidates in (("train_motion", ("train_motion", "train")),
                               ("valid_motion", ("valid_motion", "validation", "valid"))):
        path = next((source / name for name in candidates if (source / name).is_dir()), None)
        if path is None and not dry:
            raise FileNotFoundError(f"Missing {target} under {source}; download the supplied 160/80 motion-mask subset.")
        if not dry:
            view.mkdir(parents=True, exist_ok=True)
            link = view / target
            if link.is_symlink() and link.resolve() != path.resolve():
                raise ValueError(f"Dataset view points to a different source: {link}")
            if not link.exists():
                link.symlink_to(path.resolve(), target_is_directory=True)
    return view


def export_tokens(models, args, preprocessing="native"):
    env = common_env()
    env.update(YOUTUBEVOS_ROOT=str(dataset_view(args.dry_run)),
               SPLIT=os.environ.get("SPLIT", "all"), TOKENS_PER_FRAME="64",
               FRAMES_PER_FORWARD="8", THRESHOLD="0.35", SEED="42", OUTPUT_DIR="")
    for model in models:
        prefix = "PI3X" if model == "pi3x" else SPECS[model][1]
        env[f"{prefix}_OUTPUT_DIR"] = str(TOKENS / preprocessing / model / "data/isdynamic")
        if model not in {"pi3x", "pi3"}:
            env[f"{prefix}_PREPROCESS_MODE"] = preprocessing
        env[f"{SPECS[model][1]}_PYTHON_BIN"] = python_for(model)
    run(["bash", ROOT / "bash/common/run_youtubevos_token_exports.sh", *models], env, args.dry_run)


def probe_root(model, scope=MAIN, experiment="rank_8"):
    return scope / "probes" / model / experiment


def train(model, destination, args, rank=8, types="pair_lowrank_quadratic", groups=None,
          source=None, label_control="none", residual="none"):
    env = common_env(destination.parent)
    env.update(TRAIN_LABEL_CONTROL=label_control, POSITION_RESIDUALIZATION=residual)
    module(model, "isdynamic.probes.run_isdynamic_blockwise_probing",
           "--model", SPECS[model][0],
           "--base_config", ROOT / "configs/probes" / f"{SPECS[model][2]}.json",
           "--sharded_root", source or token_source(model), "--train_split", "train",
           "--val_split", "val", "--test_splits", TEST_SPLITS,
           "--results_dir", destination, "--cache_dir", destination / "cache",
           "--groups", groups or SPECS[model][3], "--model_types", types,
           "--rank", rank, "--metric", "accuracy", "--ephemeral_cache", "--skip_existing",
           env=env, dry=args.dry_run)


def masks(model, args, scope=MAIN, probe=None, context="temporal", preprocessing="native",
          fixed=False, stage="masks", benchmarks=None, variants=None, qualitative=False):
    env = common_env(scope)
    prefix = "PI3X" if model == "pi3x" else SPECS[model][1]
    env.update(PYTHON_BIN=python_for(model), MODELS=SPECS[model][0],
               BENCHMARKS=benchmarks or os.environ.get("BENCHMARKS", MASK_BENCHMARKS),
               OUT_DIR=str(scope / "masks"), MASK_ROOT=str(scope / "masks"),
               EVAL_ROOT=str(scope / "eval"), CONTEXT_MODE=context,
               FEATURE_BATCH_SIZE=os.environ.get(f"{prefix}_FEATURE_BATCH_SIZE", "1" if model == "da3" else "4"))
    env[f"{prefix}_PROBE_ROOT"] = str(probe or probe_root(model) / "pair_lowrank_quadratic")
    env["DUST3R_GROUPS" if model == "dust3r" else f"{prefix}_GROUP"] = SPECS[model][3]
    if model not in {"pi3x", "pi3"}:
        env[f"{prefix}_PREPROCESS_MODE"] = preprocessing
    if model == "da3":
        env["SCORE_BATCH_SIZE"] = os.environ.get("DA3_SCORE_BATCH_SIZE", "32768")
    if qualitative:
        env.update(ISDYNAMIC_MASK_OUTPUT_MODE="full", VARIANTS="sam2_prompt",
                   PIPELINE_REFINED_AFTER_EACH_SEQUENCE="0")
    if fixed:
        env.update(MOTION_MASK_THRESHOLD_MODE="fixed", MOTION_MASK_THRESHOLD="0.5",
                   MOTION_MASK_THRESHOLD_OFFSET="0.0")
    if stage == "eval":
        env["BENCHMARKS"] = benchmarks or os.environ.get("BENCHMARKS", EVAL_BENCHMARKS)
        for variant in variants or ("bilinear_upsample", REFINED):
            env["VARIANT"] = variant
            run(["bash", ROOT / "bash/common/run_eval_motion_masks_all.sh"], env, args.dry_run)
    else:
        filename = "run_export_motion_masks_all.sh" if stage == "masks" else "run_refine_motion_masks_all.sh"
        run(["bash", ROOT / "bash/common" / filename], env, args.dry_run)


def predict(args, models):
    """Run pretrained probes on images, with the existing qualitative writer and evaluator."""
    def resolve_input(path):
        path = path.expanduser()
        return (path if path.is_absolute() else Path(os.environ.get("PMD_CALLER_CWD", ROOT)) / path).resolve()

    if args.probe_root and args.probes_root:
        raise ValueError("Use --probe-root for one model or --probes-root for a model bundle, not both.")
    if args.probe_root and len(models) != 1:
        raise ValueError("--probe-root requires exactly one backbone.")
    scope = resolve_input(args.output_dir or OUTPUT / "predictions")
    for name in ("davis", "segtrack", "fbms"):
        value = getattr(args, f"{name}_root")
        if value is not None:
            os.environ[f"{name.upper()}_ROOT"] = str(resolve_input(value))
    benchmarks = args.benchmarks or ["davis2017", "segtrackv2", "fbms59"]
    mask_benchmarks = list(dict.fromkeys("davis_all" if name.startswith("davis") else name for name in benchmarks))
    variants = ("bilinear_upsample", REFINED) if args.sam2 else ("bilinear_upsample",)
    jobs = []
    for model in models:
        prefix = "PI3X" if model == "pi3x" else SPECS[model][1]
        bundle_probe = None
        if args.probes_root:
            bundle_model = resolve_input(args.probes_root) / model
            released = bundle_model / "rank_8/pair_lowrank_quadratic"
            bundle_probe = released if released.is_dir() else bundle_model / "pair_lowrank_quadratic"
        probe = resolve_input(args.probe_root or bundle_probe or probe_root(model) / "pair_lowrank_quadratic")
        protocol = {"model": model, "probe_root": str(probe), "benchmarks": benchmarks,
                    "checkpoint_overrides": {key: os.environ[key] for key in
                                             (f"{prefix}_CHECKPOINT", f"{prefix}_MODEL_DIR", f"{prefix}_WEIGHTS")
                                             if key in os.environ},
                    "dataset_roots": {key: os.environ[f"{key}_ROOT"] for key in ("DAVIS", "SEGTRACK", "FBMS")},
                    "preprocessing": "native", "context_size": 3, "context_stride": 2,
                    "threshold": "otsu+0.2", "qualitative_outputs": True}
        manifest = scope / f"run_{model}.json"
        if not args.dry_run:
            if not probe.is_dir():
                raise FileNotFoundError(f"Missing trained probes: {probe}. Use --probe-root or --probes-root; no training is run.")
            for benchmark in benchmarks:
                key = "DAVIS" if benchmark.startswith("davis") else "SEGTRACK" if benchmark == "segtrackv2" else "FBMS"
                if not Path(os.environ[f"{key}_ROOT"]).is_dir():
                    raise FileNotFoundError(f"Missing {key} dataset: {os.environ[f'{key}_ROOT']}")
            if manifest.exists() and json.loads(manifest.read_text()) != protocol:
                raise ValueError(f"Different inference settings already recorded in {manifest}; choose a new --output-dir.")
            mask_dir = scope / "masks" / SPECS[model][0]
            if not manifest.exists() and mask_dir.exists():
                raise ValueError(f"Existing masks have no matching run manifest: {mask_dir}; choose a new --output-dir.")
        jobs.append((model, probe, manifest, protocol))
    if not args.dry_run:
        scope.mkdir(parents=True, exist_ok=True)
        for _, _, manifest, protocol in jobs:
            manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    for model, probe, _, _ in jobs:
        for stage in (("masks", "refine", "eval") if args.sam2 else ("masks", "eval")):
            selected = benchmarks if stage == "eval" else mask_benchmarks
            masks(model, args, scope, probe=probe, stage=stage, benchmarks=",".join(selected),
                  variants=variants, qualitative=True)
    analysis("isdynamic.release.tables", "--root", scope, "--kind", "main", dry=args.dry_run)
    print(f"[results] {scope}\n[tables] {scope / 'tables/motion_masks.csv'}", flush=True)


def subset(model, size, seed, args):
    source = token_source(model)
    dest = OUTPUT / "views/subsets" / model / f"{size}_seed_{seed}"
    if args.dry_run:
        return dest
    train_root = source / "train"
    names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if size > len(names):
        raise ValueError(f"Requested {size} videos but found {len(names)} under {train_root}")
    random.Random(seed).shuffle(names)
    selected = sorted(names[:size])
    manifest = dest / "subset_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["selected_sequences"] != selected:
        raise ValueError(f"Subset manifest differs: {manifest}")
    (dest / "train").mkdir(parents=True, exist_ok=True)
    for name in selected:
        link = dest / "train" / name
        if not link.exists():
            link.symlink_to((train_root / name).resolve(), target_is_directory=True)
    for split in ("val", *TEST_SPLITS.split(",")):
        link = dest / split
        if not link.exists():
            link.symlink_to((source / split).resolve(), target_is_directory=True)
    manifest.write_text(json.dumps({"size": size, "seed": seed, "source": str(source),
                                    "selected_sequences": selected}, indent=2) + "\n")
    return dest


def analysis(name, *params, dry=False):
    run([os.environ.get("ANALYSIS_PYTHON_BIN", "python"), "-m", name, *params], dry=dry)


def main_experiment(args, models):
    if args.experiment == "predict":
        predict(args, models)
    elif args.experiment == "export":
        export_tokens(models, args)
    elif args.experiment in {"train", "layerwise", "ranks"}:
        for model in models:
            if args.experiment == "ranks":
                for rank in args.ranks:
                    train(model, probe_root(model, experiment=f"rank_{rank}"), args, rank=rank)
            else:
                layerwise = args.experiment == "layerwise"
                train(model, probe_root(model, experiment="layerwise" if layerwise else "rank_8"), args,
                      types="pair_linear,pair_diag_quadratic,pair_lowrank_quadratic" if layerwise else "pair_lowrank_quadratic",
                      groups=SPECS[model][4] if layerwise else None)
    elif args.experiment in {"starvation", "starvation_masks"}:
        for model in models:
            for size in args.sizes:
                for seed in args.seeds:
                    if size == 160 and seed != 42:
                        continue
                    destination = MAIN / "probes" / model / "starvation" / f"{size}_seed_{seed}"
                    if args.experiment == "starvation":
                        train(model, destination, args, source=subset(model, size, seed, args),
                              types="pair_lowrank_quadratic,pair_quadratic")
                    else:
                        scope = MAIN / "starvation" / model / f"{size}_seed_{seed}"
                        for stage in ("masks", "refine", "eval"):
                            masks(model, args, scope, destination / "pair_lowrank_quadratic", stage=stage)
    elif args.experiment in {"masks", "refine", "eval"}:
        for model in models:
            masks(model, args, stage=args.experiment)
    elif args.experiment == "figures":
        analysis("isdynamic.release.figures", "--root", MAIN, "--token_root", TOKENS,
                 "--models", ",".join(models), "--panels", ",".join(args.panels), dry=args.dry_run)
    elif args.experiment == "tables":
        analysis("isdynamic.release.tables", "--root", MAIN, "--kind", args.table, dry=args.dry_run)
    elif args.experiment == "qualitative":
        module("pi3x", "isdynamic.viz.export_unseen_class_examples", "--out_dir", MAIN / "qualitative/unseen_classes",
               "--checkpoint", os.environ.get("PI3X_CHECKPOINT", ROOT / "models/Pi3/ckpt/model_pi3x.safetensors"),
               "--probe_root", probe_root("pi3x") / "pair_lowrank_quadratic", "--context_size", "8",
               "--context_stride", "1", "--davis_root", os.environ["DAVIS_ROOT"],
               "--segtrack_root", os.environ["SEGTRACK_ROOT"], "--fbms_root", os.environ["FBMS_ROOT"], dry=args.dry_run)
    elif args.experiment == "all":
        for name in ("export", "train", "masks", "refine", "eval", "tables"):
            args.experiment = name
            main_experiment(args, models)


def additional_experiment(args, models):
    if args.experiment == "tables":
        analysis("isdynamic.release.tables", "--root", EXPERIMENTS,
                 "--kind", "controls" if args.table == "main" else args.table, dry=args.dry_run)
    elif args.experiment == "controls":
        for model in models:
            for rank, labels in ((4, "correct"), (8, "correct"), (8, "shuffled")):
                train(model, EXPERIMENTS / "controls" / model / f"{labels}_rank_{rank}", args, rank=rank,
                      label_control="within_group_shuffle" if labels == "shuffled" else "none")
        analysis("isdynamic.release.tables", "--root", EXPERIMENTS, "--kind", "controls", dry=args.dry_run)
    elif args.experiment == "thresholds":
        analysis("isdynamic.release.thresholds", "--source_mask_root", MAIN / "masks",
                 "--output_root", EXPERIMENTS / "thresholds", "--davis_root", os.environ["DAVIS_ROOT"],
                 "--segtrack_root", os.environ["SEGTRACK_ROOT"], "--fbms_root", os.environ["FBMS_ROOT"], dry=args.dry_run)
    elif args.experiment == "val_masks":
        env = common_env()
        env.update(YOUTUBEVOS_ROOT=str(dataset_view(args.dry_run)),
                   PROBE_OUTPUTS_ROOT=str(MAIN), ARTIFACT_ROOT=str(EXPERIMENTS / "source_validation"),
                   MODELS=",".join(SPECS[m][0] for m in models),
                   DA3_FEATURE_BATCH_SIZE="1", DA3_SCORE_BATCH_SIZE="32768")
        for model in SPECS:
            prefix = "PI3X" if model == "pi3x" else SPECS[model][1]
            env[f"{prefix}_PROBE_ROOT"] = str(probe_root(model) / "pair_lowrank_quadratic")
        run(["bash", ROOT / "bash/common/run_source_validation_masks.sh"], env, args.dry_run)
    elif args.experiment == "diagnostic":
        manifest = {name: str(probe_root(model) / "pair_lowrank_quadratic")
                    for name, model in (("pi3X", "pi3x"), ("pi3", "pi3"), ("VGGT", "vggt"),
                                        ("DA3", "da3"), ("MapAnything", "mapanything"), ("DUSt3R", "dust3r"))}
        path = EXPERIMENTS / "diagnostic/probe_paths.json"
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        analysis("isdynamic.viz.analyze_backbone_probe_diagnostic", "--probe_outputs_root", MAIN,
                 "--probe_paths_json", path, "--mask_outputs_root", MAIN / "masks",
                 "--mask_eval_root", MAIN / "eval", "--source_val_eval_root", EXPERIMENTS / "source_validation/eval",
                 "--output_dir", EXPERIMENTS / "diagnostic", dry=args.dry_run)
    elif args.experiment == "runtime":
        for method in args.methods:
            env = common_env()
            env.update(OUTPUT_DIR=str(EXPERIMENTS / "runtime" / method),
                       PI3X_PROBE_ROOT=str(probe_root("pi3x") / "pair_lowrank_quadratic"),
                       SEQUENCE="parkour", WARMUP_SEQUENCE="cows", NUM_FRAMES="100", WARMUP_FRAMES="10")
            # Runtime profilers expect DAVIS/JPEGImages beneath this parent.
            davis = Path(os.environ["DAVIS_ROOT"])
            if davis.name == "DAVIS":
                env["DAVIS_ROOT"] = str(davis.parent)
            env["PYTHON_BIN"] = os.environ.get(f"{method.upper()}_PYTHON_BIN", python_for("pi3x"))
            if method != "pi3x":
                env[f"{method.upper()}_RUNTIME_ROOT"] = os.environ.get(
                    f"{method.upper()}_RUNTIME_ROOT", str(ROOT / "models" / f"{method}_assets"))
                env[f"{method.upper()}_ROOT"] = os.environ.get(
                    f"{method.upper()}_ROOT", str(ROOT / "models" / {"geomotion":"GeoMotion","seganymo":"SegAnyMo","flowsam":"flowsam"}[method]))
            run(["bash", ROOT / f"bash/common/profile_{method}_runtime.sh"], env, args.dry_run)
        analysis("isdynamic.release.tables", "--root", EXPERIMENTS, "--kind", "runtime", dry=args.dry_run)
    elif args.experiment == "repeated_context":
        for model in models:
            if model not in {"pi3x", "pi3"}:
                raise ValueError("The implemented repeated-target control supports pi3x and pi3.")
            scope = EXPERIMENTS / "repeated_context" / model
            masks(model, args, scope, context="repeated_target", fixed=True)
    elif args.experiment == "position_residual":
        for model in models:
            dest = EXPERIMENTS / "position_residual" / "probes" / model / "rank_8"
            if args.stage == "train":
                train(model, dest, args, residual="fourier")
            else:
                probe = dest / "pos_residualized/fourier/pair_lowrank_quadratic"
                masks(model, args, EXPERIMENTS / "position_residual", probe=probe, stage=args.stage)
    elif args.experiment == "padding":
        if not models or set(models) - {"da3", "mapanything", "dust3r"}:
            raise ValueError("Choose da3, mapanything, or dust3r for the optional padding experiment. "
                             "Pi3X, Pi3, and VGGT already use padding in the native PMD pipeline.")
        scope = EXPERIMENTS / "padding"
        if args.stage == "export":
            export_tokens(models, args, "aspect_pad")
        elif args.stage == "train":
            for model in models:
                train(model, probe_root(model, scope), args, source=token_source(model, "aspect_pad"))
        elif args.stage == "geometry":
            for model in models:
                env = common_env(scope)
                env.update(PYTHON_BIN=python_for(model), MODELS=SPECS[model][0],
                           OUT_DIR=str(scope / "geometry"))
                run(["bash", ROOT / "bash/common/run_padding_geometry_stability.sh"], env, args.dry_run)
        else:
            for model in models:
                masks(model, args, scope, probe_root(model, scope) / "pair_lowrank_quadratic",
                      preprocessing="aspect_pad", stage=args.stage)


def parse_args():
    direct = sys.argv[1:3] == ["main", "predict"]
    parser = argparse.ArgumentParser(
        prog="bash scripts/evaluate.sh" if direct else None,
        description="Pretrained masks, qualitative outputs, and benchmark evaluation." if direct
                    else "Reproduce ProMo3D / PMD experiments using bash entrypoints.")
    if direct:
        parser.set_defaults(suite="main", experiment="predict")
    else:
        parser.add_argument("suite", choices=("main", "experiments"))
        parser.add_argument("experiment")
    parser.add_argument("models", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    if direct:
        parser.add_argument("--probe-root", type=Path, help="Trained pair_lowrank_quadratic directory for one backbone.")
        parser.add_argument("--probes-root", type=Path, help="Model bundle containing <backbone>/[rank_8/]pair_lowrank_quadratic/.")
        parser.add_argument("--output-dir", type=Path, help="Output folder; default: outputs/predictions/.")
        parser.add_argument("--davis-root", type=Path)
        parser.add_argument("--segtrack-root", type=Path)
        parser.add_argument("--fbms-root", type=Path)
        parser.add_argument("--benchmarks", nargs="+", choices=EVAL_BENCHMARKS.split(","))
        parser.add_argument("--sam2", action="store_true", help="Also generate and separately evaluate SAM2-refined masks.")
    else:
        parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
        parser.add_argument("--sizes", type=int, nargs="+", default=[160, 80, 40, 20, 10])
        parser.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7])
        parser.add_argument("--methods", nargs="+", choices=("pi3x", "geomotion", "flowsam", "seganymo"), default=["pi3x"])
        parser.add_argument("--stage", choices=("export", "train", "masks", "refine", "eval", "geometry"), default="train")
        parser.add_argument("--panels", nargs="+", choices=("layerwise", "pca", "starvation", "starvation_masks"), default=["layerwise"])
        parser.add_argument("--table", choices=("main", "ranks", "controls", "runtime"), default="main")
    return parser.parse_args(sys.argv[3:] if direct else None)


if __name__ == "__main__":
    args = parse_args()
    choices = {"main": {"predict", "export", "train", "layerwise", "ranks", "starvation", "starvation_masks", "masks", "refine", "eval", "figures", "tables", "qualitative", "all"},
               "experiments": {"controls", "thresholds", "val_masks", "diagnostic", "runtime", "tables", "repeated_context", "position_residual", "padding"}}
    if args.experiment not in choices[args.suite]:
        raise SystemExit(f"Choose an experiment from: {', '.join(sorted(choices[args.suite]))}")
    defaults = ["pi3x", "pi3", "da3", "dust3r"] if args.suite == "main" else list(SPECS)
    if args.experiment == "predict":
        defaults = list(SPECS)
    if args.experiment == "controls":
        defaults = ["pi3x", "da3", "dust3r"]
    if args.experiment in {"starvation", "starvation_masks", "ranks"}:
        defaults = ["pi3x", "da3", "dust3r"]
    if args.experiment == "repeated_context":
        defaults = ["pi3x"]
    if args.experiment == "padding":
        defaults = ["da3", "mapanything", "dust3r"]
    models = [ALIASES.get(model, model) for model in args.models or defaults]
    unknown = set(models) - SPECS.keys()
    if unknown:
        raise SystemExit(f"Unknown backbone(s): {', '.join(sorted(unknown))}")
    print(f"[workspace] {ROOT}\n[outputs] {OUTPUT}\n[tokens] {TOKENS}", flush=True)
    (main_experiment if args.suite == "main" else additional_experiment)(args, models)
