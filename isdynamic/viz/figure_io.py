from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List

import matplotlib as mpl
import numpy as np
from PIL import Image


VECTOR_SUFFIXES = (".pdf", ".svg")


def _configure_vector_font_embedding() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"


def resolve_vector_output_paths(out_path: str | Path, suffixes: Iterable[str] = VECTOR_SUFFIXES) -> List[Path]:
    path = Path(out_path)
    ordered_suffixes = []
    for suffix in suffixes:
        normalized = suffix if suffix.startswith(".") else f".{suffix}"
        if normalized not in ordered_suffixes:
            ordered_suffixes.append(normalized)
    if not ordered_suffixes:
        raise ValueError("At least one vector suffix is required.")

    requested_suffix = path.suffix.lower()
    if requested_suffix in ordered_suffixes:
        primary = path
    else:
        primary = path.with_suffix(ordered_suffixes[0])

    paths = [primary]
    for suffix in ordered_suffixes:
        candidate = primary.with_suffix(suffix)
        if candidate not in paths:
            paths.append(candidate)
    return paths


def save_vector_figure(
    fig,
    out_path: str | Path,
    *,
    dpi: int = 220,
    bbox_inches=None,
    pad_inches=None,
    suffixes: Iterable[str] = VECTOR_SUFFIXES,
) -> List[Path]:
    _configure_vector_font_embedding()
    saved_paths = resolve_vector_output_paths(out_path, suffixes=suffixes)
    savefig_kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        savefig_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        savefig_kwargs["pad_inches"] = pad_inches

    for path in saved_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix.lstrip("."), **savefig_kwargs)
    return saved_paths


def save_png_pdf_figure(
    fig,
    out_path: str | Path,
    *,
    dpi: int = 220,
    bbox_inches=None,
    pad_inches=None,
) -> List[Path]:
    return save_vector_figure(
        fig,
        out_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        suffixes=(".png", ".pdf"),
    )


def save_raster_png_pdf_figure(
    fig,
    out_path: str | Path,
    *,
    dpi: int = 220,
    bbox_inches=None,
    pad_inches=None,
) -> List[Path]:
    buffer = io.BytesIO()
    savefig_kwargs = {"dpi": dpi, "format": "png"}
    if bbox_inches is not None:
        savefig_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        savefig_kwargs["pad_inches"] = pad_inches
    fig.savefig(buffer, **savefig_kwargs)
    buffer.seek(0)

    rgba = Image.open(buffer).convert("RGBA")
    rgb = Image.fromarray(np.asarray(rgba)[..., :3], mode="RGB")

    paths = resolve_vector_output_paths(out_path, suffixes=(".png", ".pdf"))
    png_path = next(path for path in paths if path.suffix.lower() == ".png")
    pdf_path = next(path for path in paths if path.suffix.lower() == ".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(png_path)
    rgb.save(pdf_path, resolution=float(dpi))
    return [png_path, pdf_path]
