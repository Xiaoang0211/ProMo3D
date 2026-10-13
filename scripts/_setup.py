from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def link(source, destination):
    if not source.exists():
        print(f"[missing] {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Resource already exists: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    print(f"[link] {destination} -> {source}")


def local_setup(source):
    source = source.expanduser().resolve()
    lock = json.loads((ROOT / "configs/third_party.lock.json").read_text())
    for name, spec in lock.items():
        if name in {"GeoMotion", "flowsam", "SegAnyMo"}:
            path = source.parent / name
        else:
            path = source / "sam2" if name == "sam2" else source / spec["destination"]
        link(path, ROOT / spec["destination"])
    datasets = source.parent / "PMD/datasets"
    for name in ("youtubevos_160vids", "youtubevos_10vids", "DAVIS", "SegTrackv2", "FBMS-59"):
        link(datasets / name, ROOT / "datasets" / name)
    config = ROOT / "configs/local.env"
    if config.exists():
        print(f"[keep] {config}")
        return
    lines = ["# Local environment settings. Large token exports default to this workspace."]
    environments = {"PI3": "pi3", "VGGT": "vggt", "DA3": "depthanything3",
                    "MAPANYTHING": "mapanything", "DUST3R": "dust3r",
                    "SAM2": "pi3", "ANALYSIS": "pi3", "GEOMOTION": "vggt4d",
                    "FLOWSAM": "pi3", "SEGANYMO": "vggt4d"}
    base = Path(os.environ.get("CONDA_ENV_ROOT", Path.home() / "anaconda3/envs"))
    for prefix, name in environments.items():
        path = base / name / "bin/python"
        if path.exists():
            lines.append(f"{prefix}_PYTHON_BIN={shlex.quote(str(path))}")
    config.write_text("\n".join(lines) + "\n")
    print(f"[settings] {config}")


def clone_models(names):
    lock = json.loads((ROOT / "configs/third_party.lock.json").read_text())
    for name in names or ("Pi3", "Depth-Anything-3", "dust3r", "sam2"):
        spec = lock[name]
        dest = ROOT / spec["destination"]
        if dest.exists():
            print(f"[keep] {dest}; existing repositories are not modified")
            continue
        subprocess.run(["git", "clone", spec["url"], str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", spec["commit"]], check=True)
        subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"], check=True)
        if spec.get("patch"):
            patch = ROOT / spec["patch"]
            subprocess.run(["git", "-C", str(dest), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(dest), "apply", str(patch)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Set up pinned third-party code or reuse local research resources.")
    parser.add_argument("mode", choices=("link", "clone"))
    parser.add_argument("resources", nargs="*")
    args = parser.parse_args()
    if args.mode == "link":
        if len(args.resources) != 1:
            parser.error("Usage: bash scripts/setup_local.sh /path/to/isDynamicProbe")
        local_setup(Path(args.resources[0]))
    else:
        clone_models(args.resources)


if __name__ == "__main__":
    main()
