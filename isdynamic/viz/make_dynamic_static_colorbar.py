from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a standalone jet color bar labeled static/dynamic.")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/isdynamic_visualizations/common/jet_dynamic_static_bar.png",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="vertical",
        choices=("vertical", "horizontal"),
    )
    parser.add_argument("--width", type=float, default=1.55)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--label_fontsize", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.orientation == "horizontal":
        fig = plt.figure(figsize=(5.6, 1.55))
        ax = fig.add_axes([0.17, 0.39, 0.66, 0.18])
    else:
        fig = plt.figure(figsize=(args.width, args.height))
        ax = fig.add_axes([0.30, 0.08, 0.34, 0.84])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="jet")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation=args.orientation)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(1.0)
    if args.orientation == "horizontal":
        cbar.ax.text(
            -0.22,
            0.5,
            "Low\nmotion",
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            fontsize=args.label_fontsize,
        )
        cbar.ax.text(
            1.22,
            0.5,
            "High\nmotion",
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            fontsize=args.label_fontsize,
        )
    else:
        cbar.ax.text(
            0.5,
            1.04,
            "High\nmotion",
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=args.label_fontsize,
        )
        cbar.ax.text(
            0.5,
            -0.05,
            "Low\nmotion",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=args.label_fontsize,
        )

    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved color bar: {out_path}")
    print(f"Saved color bar PDF: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
