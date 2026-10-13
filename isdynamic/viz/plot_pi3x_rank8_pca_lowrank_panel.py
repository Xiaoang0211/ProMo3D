from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .figure_io import save_vector_figure


def parse_args() -> argparse.Namespace:
    root = Path("outputs/isdynamic_visualizations/pi3x")
    parser = argparse.ArgumentParser(
        description="Compose a panel from pi3x DAVIS rank-8 PCA 3D and low-rank evidence figures."
    )
    parser.add_argument(
        "--pca_png",
        type=Path,
        default=root / "PCA" / "test_davis_inter_frame_block_0_rank8_isdynamic_PCA_3d.png",
    )
    parser.add_argument(
        "--spectrum_png",
        type=Path,
        default=root / "lowrank" / "pi3x_rank8_inter_frame_block" / "01_singular_spectrum_selected_blocks_notitle.png",
    )
    parser.add_argument(
        "--metric_png",
        type=Path,
        default=root / "lowrank" / "pi3x_rank8_inter_frame_block" / "04_metric_vs_effective_rank_notitle.png",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=root / "comparisons" / "pi3x_test_davis_rank8_pca_lowrank_panel.pdf",
    )
    parser.add_argument(
        "--width_ratios",
        type=str,
        default="1.06,1.37,1.33",
        help="Comma-separated width ratios for PCA, singular spectrum, and metric-vs-rank panels.",
    )
    parser.add_argument(
        "--wspace",
        type=float,
        default=0.02,
        help="Horizontal space between the three panels.",
    )
    return parser.parse_args()


def _parse_ratios(raw: str) -> list[float]:
    ratios = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(ratios) != 3:
        raise ValueError(f"Expected exactly 3 width ratios, got {ratios}")
    return ratios


def _load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input figure not found: {path}")
    return mpimg.imread(path)


def _autocrop_white_border(image: np.ndarray, *, threshold: float = 0.99, pad: int = 10) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    rgb = image[..., :3]
    nonwhite = np.any(rgb < threshold, axis=2)
    if not np.any(nonwhite):
        return image

    rows = np.where(np.any(nonwhite, axis=1))[0]
    cols = np.where(np.any(nonwhite, axis=0))[0]
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, image.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def main() -> None:
    args = parse_args()
    width_ratios = _parse_ratios(args.width_ratios)

    images = [
        _load_image(args.pca_png),
        _load_image(args.spectrum_png),
        _load_image(args.metric_png),
    ]
    images = [
        _autocrop_white_border(images[0], threshold=0.99, pad=10),
        _autocrop_white_border(images[1], threshold=0.99, pad=6),
        _autocrop_white_border(images[2], threshold=0.99, pad=6),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.6, 4.7),
        gridspec_kw={"width_ratios": width_ratios, "wspace": args.wspace},
        squeeze=False,
    )
    axes = axes[0]

    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")

    saved_paths = save_vector_figure(
        fig,
        args.out_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    print("Saved panel plots: " + ", ".join(str(path) for path in saved_paths))


if __name__ == "__main__":
    main()
