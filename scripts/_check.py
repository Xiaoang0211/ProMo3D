from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["pi3x", "pi3", "da3", "dust3r"])
    args = parser.parse_args()
    mapping = {"pi3x": ("Pi3", "PI3", "PI3X_CHECKPOINT", "ckpt/model_pi3x.safetensors"),
               "pi3": ("Pi3", "PI3", "PI3_CHECKPOINT", "ckpt/model_pi3.safetensors"),
               "vggt": ("vggt", "VGGT", "VGGT_CHECKPOINT", "ckpt/model.pt"),
               "da3": ("Depth-Anything-3", "DA3", "DA3_MODEL_DIR", "src/depth_anything_3/ckpt/DA3-giant-1.1"),
               "mapanything": ("map-anything", "MAPANYTHING", "MAPANYTHING_CHECKPOINT", "ckpt"),
               "dust3r": ("dust3r", "DUST3R", "DUST3R_WEIGHTS", "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")}
    failures = []
    lock = json.loads((ROOT / "configs/third_party.lock.json").read_text())
    for model in args.models:
        repo, prefix, checkpoint_var, checkpoint = mapping[model]
        code = ROOT / "models" / repo
        python = os.environ[f"{prefix}_PYTHON_BIN"]
        weights = Path(os.environ.get(checkpoint_var, str(code / checkpoint)))
        for label, exists in (("code", code.is_dir()), ("python", shutil.which(python) is not None), ("weights", weights.exists())):
            print(f"[{'ok' if exists else 'missing'}] {model}: {label}")
            if not exists:
                failures.append(f"{model} {label}")
        if code.is_dir():
            commit = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True, text=True)
            if commit.returncode == 0:
                status = "match" if commit.stdout.strip() == lock[repo]["commit"] else "different revision"
                print(f"[revision] {model}: {status}")
    for variable in ("YOUTUBEVOS_ROOT", "DAVIS_ROOT", "SEGTRACK_ROOT", "FBMS_ROOT"):
        path = Path(os.environ[variable])
        print(f"[{'ok' if path.is_dir() else 'missing'}] {variable}: {path}")
        if not path.is_dir():
            failures.append(variable)
    print(f"[outputs] {os.environ['OUTPUT_ROOT']}\n[tokens] {os.environ['TOKEN_ROOT']}")
    if failures:
        raise SystemExit("Missing resources: " + ", ".join(failures))


if __name__ == "__main__":
    main()
