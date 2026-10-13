"""Prepare complete showcase videos from archived PMD outputs, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFINED = "sam2_prompt_otsu_off0.2_erode2_w30_mean_otsu_off0.3"
SAMPLES = (
    ("rhino", "davis_all", 70, "Rhino behind a foreground tree"),
    ("koala", "davis_all", 70, "Koala against a textured background"),
    ("cheetah", "segtrackv2", 0, "Two animals running in a chase"),
)
FIGURES = {
    "layerwise": "comparisons/blockwise/pi3x_da3_dust3r_blockwise_widescreen_panel.pdf",
    "pca": "pi3x/comparisons/pi3x_test_davis_rank8_pca_lowrank_panel.pdf",
    "starvation": "comparisons/data_starvation/pi3x_da3_dust3r_test_davis_normalized_gain_panel.pdf",
}


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def frame_dirs(sequence_root, expected_count):
    frames = sorted((sequence_root / "bilinear_upsample").glob("frame_*"))
    indices = [int(frame.name.removeprefix("frame_")) for frame in frames if frame.is_dir()]
    if indices != list(range(expected_count)):
        raise ValueError(f"Incomplete sequence {sequence_root.name}: {len(indices)} outputs for {expected_count} source frames")
    return [frame for frame in frames if frame.is_dir()]


def encode(paths, destination, fps):
    if destination.exists():
        return
    if not all(path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(missing)
    # Glob ordering is frame ordering because archive indices are zero-padded.
    pattern = str(paths[0].parent.parent / "frame_*" / paths[0].name) if paths[0].parent.name.startswith("frame_") else str(paths[0].parent / "frame_*_overlay.png")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-framerate", fps,
         "-pattern_type", "glob", "-i", pattern, "-vf", "scale=768:-2:flags=lanczos",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-an", "-movflags", "+faststart", destination])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_outputs", type=Path, required=True)
    parser.add_argument("--davis_root", type=Path, default=ROOT / "datasets/DAVIS")
    parser.add_argument("--segtrack_root", type=Path, default=ROOT / "datasets/SegTrackv2")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    assets = ROOT / "project_page/assets"
    videos = assets / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    samples, provenance = [], {"backbone": "pi3x", "rank": 8, "visualization_fps": args.fps, "sequences": [], "figures": {}}
    for name, benchmark, poster_idx, description in SAMPLES:
        data_root = args.davis_root / "JPEGImages/480p" if benchmark == "davis_all" else args.segtrack_root / "JPEGImages"
        source_images = sorted(path for path in (data_root / name).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not source_images:
            raise ValueError(f"No source images for {name}")
        relative = Path("isdynamic_motion_masks/pi3x/inter_frame_block") / benchmark / name
        sequence_root = args.archive_outputs / relative
        frames = frame_dirs(sequence_root, len(source_images))
        views = {
            "input": [frame / "input_image.png" for frame in frames],
            "scores": [frame / "avg_score_upsampled_heatmap.png" for frame in frames],
            "coarse": [frame / "binary_mask_upsampled_overlay.png" for frame in frames],
            "refined": [sequence_root / REFINED / f"frame_{index:05d}_overlay.png" for index in range(len(frames))],
        }
        for paths in views.values():
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError(next(path for path in paths if not path.is_file()))
        meta_path = frames[poster_idx] / "meta.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != "pi3x" or len(meta.get("usable_blocks", [])) != 18:
            raise ValueError(f"Unexpected probe protocol for {name}")
        sample = {"id": name, "description": description, "benchmark": benchmark,
                  "frames": len(frames), "fps": args.fps, "poster_frame": poster_idx, "views": {}}
        for view, paths in views.items():
            movie, poster = videos / f"{name}-{view}.mp4", videos / f"{name}-{view}.png"
            encode(paths, movie, args.fps)
            shutil.copy2(paths[poster_idx], poster)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=nb_frames", "-of", "json", movie], text=True))
            if int(probe["streams"][0]["nb_frames"]) != len(frames):
                raise ValueError(f"Video is incomplete: {movie}")
            sample["views"][view] = {"video": f"assets/videos/{movie.name}", "poster": f"assets/videos/{poster.name}"}
        samples.append(sample)
        provenance["sequences"].append({"id": name, "source": str(relative), "frames": len(frames),
            "frame_range": [0, len(frames)-1], "poster_frame": poster_idx, "refinement_variant": REFINED,
            "blocks": meta["usable_blocks"], "input_protocol": meta.get("input_protocol"),
            "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()})
        print(f"[sequence] {name}: {len(frames)} frames, four complete views", flush=True)
    figure_dir = assets / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name, relative in FIGURES.items():
        source = args.archive_outputs / "isdynamic_visualizations" / relative
        run(["pdftoppm", "-singlefile", "-f", "1", "-png", "-scale-to", "2200", source, figure_dir / name])
        provenance["figures"][name] = {"source": f"isdynamic_visualizations/{relative}", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (assets / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (assets / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
