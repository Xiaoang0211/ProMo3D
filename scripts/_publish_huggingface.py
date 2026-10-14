"""Publish reviewed cards or an explicitly prepared, checksummed PMD release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = (("model", "ProMo3D", "MODEL_CARD.md"),
         ("dataset", "ProMo3D-data", "DATASET_CARD.md"))


MODELS = {"pi3x", "pi3", "vggt", "da3", "mapanything", "dust3r"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_release(folder, kind):
    manifest = json.loads((folder / "manifest.json").read_text())
    if kind == "model" and set(manifest["models"]) != MODELS:
        raise ValueError("The model release must include all six paper backbones")
    if kind == "dataset" and {key: value["sequences"] for key, value in manifest["splits"].items()} != {"train": 160, "validation": 80}:
        raise ValueError("The dataset release must contain exactly 160 train and 80 validation sequences")
    if kind == "dataset" and manifest.get("gated") != "auto":
        raise ValueError("The RGB dataset release must require agreement to the upstream terms")
    paths = ["README.md", "manifest.json"]
    for record in manifest["files"]:
        relative = Path(record["path"])
        parts = relative.parts
        if kind == "model":
            allowed = len(parts) == 6 and parts[0] in MODELS and parts[1:3] == ("rank_8", "pair_lowrank_quadratic") and parts[-1] in {"config.json", "best_probe.pt"}
        else:
            allowed = len(parts) == 3 and parts[0] == "youtubevos_160vids" and parts[1] in {"train", "validation"} and parts[2] in {"rgb_images.zip", "binary_motion_masks.zip", "manifest.json"}
        path = folder / relative
        if not allowed or ".." in parts or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"Disallowed release path: {relative}")
        digest = sha256(path)
        if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Release checksum mismatch: {relative}")
        paths.append(relative.as_posix())
    if len(paths) == 2 or len(paths) != len(set(paths)):
        raise ValueError("Empty or duplicate release manifest")
    return paths


def publish(namespace="Xiaoang0211", dry_run=False, release_dir=None):
    plans = []
    for kind, name, filename in CARDS:
        folder = Path(release_dir) / ("models" if kind == "model" else "dataset") if release_dir else None
        paths = checked_release(folder, kind) if folder else None
        card = folder / "README.md" if folder else ROOT / "docs/huggingface" / filename
        plans.append((kind, name, card, folder, paths))
    api = None
    if not dry_run:
        from huggingface_hub import HfApi, get_token
        if not get_token():
            raise SystemExit("Hugging Face CLI/API login is required; Firefox login is separate.\n"
                             "Run: python -c 'from huggingface_hub import login; login()'\n"
                             "Authenticate in your terminal, never paste a token into chat.")
        api = HfApi()
        api.whoami()
    for kind, name, card, folder, paths in plans:
        repo = f"{namespace}/{name}"
        action = "plan" if dry_run else "publish"
        scope = f"{len(paths)} manifest-approved files" if paths else f"{card.name} -> README.md"
        print(f"[{action}] public {kind} repository {repo}: {scope}", flush=True)
        if api is not None:
            api.create_repo(repo_id=repo, repo_type=kind, private=False, exist_ok=True)
            if api.repo_info(repo_id=repo, repo_type=kind).private:
                api.update_repo_visibility(repo_id=repo, repo_type=kind, private=False)
            if folder:
                if kind == "dataset":
                    api.update_repo_settings(repo_id=repo, repo_type=kind, gated="auto")
                    if api.repo_info(repo_id=repo, repo_type=kind).gated != "auto":
                        raise RuntimeError("Dataset access agreement was not enabled; refusing RGB upload")
                api.upload_folder(folder_path=str(folder), allow_patterns=paths, repo_id=repo,
                                  repo_type=kind, commit_message="Release paper PMD probes and source RGB/masks")
            else:
                api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo,
                                repo_type=kind, commit_message="Update ProMo3D release documentation")
            prefix = "datasets/" if kind == "dataset" else ""
            print(f"[page] https://huggingface.co/{prefix}{repo}", flush=True)
    print("[scope] Public probes and agreement-gated source RGB/masks only; no Space, tokens, target data, or upstream backbone weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="Xiaoang0211")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--release-dir", type=Path, help="Publish the explicit manifests from prepare_huggingface.sh")
    args = parser.parse_args()
    publish(args.namespace, args.dry_run, args.release_dir)
