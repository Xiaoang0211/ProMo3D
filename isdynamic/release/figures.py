from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import matplotlib.pyplot as plt

from isdynamic.viz.figure_io import save_vector_figure
from isdynamic.viz.plot_isdynamic_data_starvation import _score_from_summary


def layerwise(root):
    from isdynamic.viz import plot_blockwise_comparison_panel as plot

    # Reuse the paper plotting code with the release's explicit result paths.
    specs = []
    for model, _primary, _rank8, groups in plot.PANEL_SPECS:
        public = "da3" if model == "depth_anything_3" else model
        path = root / "probes" / public / "layerwise/summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Run layerwise probes first: {path}")
        specs.append((model, path, path, groups))
    original_specs, original_argv = plot.PANEL_SPECS, sys.argv[:]
    try:
        plot.PANEL_SPECS = tuple(specs)
        sys.argv = [sys.argv[0], "--out_path", str(root / "figures/figure_2_layerwise.pdf")]
        plot.main()
    finally:
        plot.PANEL_SPECS, sys.argv = original_specs, original_argv


def starvation(root, models):
    models = [model for model in models if model in {"pi3x", "da3", "dust3r"}]
    if not models:
        raise ValueError("The main starvation panel uses pi3x, da3, and dust3r.")
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5), squeeze=False)
    for model, ax in zip(models, axes[0]):
        files = sorted((root / "probes" / model / "starvation").glob("*/summary.json"))
        if not files:
            raise FileNotFoundError(f"No starvation logs for {model}")
        group = "dec_view1" if model == "dust3r" else "inter_frame_block"
        for kind, rank, label, color in (("pair_lowrank_quadratic", 8, "Low-rank quadratic (k=8)", "#d62728"),
                                         ("pair_quadratic", None, "Full-rank quadratic", "#8e4477")):
            points = {}
            for path in files:
                size = int(path.parent.name.split("_seed_")[0])
                value = _score_from_summary(json.loads(path.read_text()), kind, group, "test_davis", "normalized_gain", rank)
                if value is not None:
                    points.setdefault(size, []).append(100 * value)
            sizes = sorted(points)
            if not sizes:
                continue
            means = [statistics.fmean(points[size]) for size in sizes]
            stds = [statistics.stdev(points[size]) if len(points[size]) > 1 else 0 for size in sizes]
            ax.plot(sizes, means, marker="o", linewidth=2.5, color=color, label=label)
            ax.fill_between(sizes, [mean-std for mean,std in zip(means,stds)],
                            [mean+std for mean,std in zip(means,stds)], color=color, alpha=0.14)
        ax.set(title=model, xlabel="Training videos", ylabel="Normalized gain (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save_vector_figure(fig, root / "figures/figure_5a_starvation_ng.pdf")
    plt.close(fig)


def starvation_masks(root, models):
    for model in models:
        paths = sorted((root / "starvation" / model).glob("*/eval/*/*/*/*.json"))
        if not paths:
            raise FileNotFoundError(f"No starvation mask evaluations for {model}")
        points = {}
        for path in paths:
            if path.stem not in {"davis_all", "segtrackv2", "fbms59"}:
                continue
            run = path.relative_to(root / "starvation" / model).parts[0]
            size = int(run.split("_seed_")[0])
            variant = path.parts[-2]
            key = (path.stem, variant, size)
            points.setdefault(key, []).append(100 * json.loads(path.read_text())["metrics"]["J"]["M"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        for benchmark, ax in zip(("davis_all", "segtrackv2", "fbms59"), axes):
            variants = sorted({key[1] for key in points if key[0] == benchmark})
            for variant in variants:
                sizes = sorted(key[2] for key in points if key[:2] == (benchmark, variant))
                values = [statistics.fmean(points[benchmark, variant, size]) for size in sizes]
                color = "#b5442e" if variant == "bilinear_upsample" else "#d62728"
                ax.plot(sizes, values, marker="o", linewidth=2.5, color=color,
                        label="Coarse" if variant == "bilinear_upsample" else "SAM2 refined")
            ax.set(title=benchmark, xlabel="Training videos", ylabel="J-Mean (%)")
            ax.grid(alpha=0.2)
            ax.legend()
        save_vector_figure(fig, root / f"figures/figure_5b_{model}_starvation_masks.pdf")
        plt.close(fig)


def pca(root, token_root):
    probe = root / "probes/pi3x/rank_8"
    block = "inter_frame_block_0"
    figures = root / "figures/figure_4"
    commands = [
        ["isdynamic.viz.visualize_isdynamic_pca", "--sharded_split_root",
         str(token_root / "native/pi3x/data/isdynamic/youtubevos/tokens/test_davis"),
         "--feature_key", block, "--probe_run_dir", str(probe / f"pair_lowrank_quadratic/inter_frame_block/{block}"),
         "--out_dir", str(figures / "pca"), "--name", "pi3x_block_0", "--make_3d"],
        ["isdynamic.viz.visualize_lowrank_evidence", "--model", "pi3x", "--results_root", str(probe),
         "--model_type", "pair_lowrank_quadratic", "--group", "inter_frame_block", "--metric_split", "test_davis",
         "--out_dir", str(figures), "--name", "lowrank"],
        ["isdynamic.viz.plot_pi3x_rank8_pca_lowrank_panel", "--pca_png", str(figures / "pca/pi3x_block_0_PCA_3d.png"),
         "--spectrum_png", str(figures / "lowrank/01_singular_spectrum_selected_blocks_notitle.png"),
         "--metric_png", str(figures / "lowrank/04_metric_vs_effective_rank_notitle.png"),
         "--out_path", str(root / "figures/figure_4_probe_induced_pca.pdf")],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", *command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token_root", type=Path, required=True)
    parser.add_argument("--models", default="pi3x,da3,dust3r")
    parser.add_argument("--panels", default="layerwise")
    args = parser.parse_args()
    (args.root / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    models = args.models.split(",")
    for panel in args.panels.split(","):
        if panel == "layerwise":
            layerwise(args.root)
        elif panel == "starvation":
            starvation(args.root, models)
        elif panel == "starvation_masks":
            starvation_masks(args.root, models)
        elif panel == "pca":
            pca(args.root, args.token_root)
        else:
            raise ValueError(f"Unknown panel: {panel}")


if __name__ == "__main__":
    main()
