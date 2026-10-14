"""Download the released PMD probes and source RGB/masks without overwriting files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("pi3x", "pi3", "vggt", "da3", "mapanything", "dust3r")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_file(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe release path: {relative}")
    return path


def install_file(source, target, expected):
    if sha256(source) != expected:
        raise ValueError(f"Download checksum mismatch: {source.name}")
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError(f"Refusing to overwrite different existing file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def extract_archive(archive, root, expected):
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(expected):
            raise ValueError(f"Archive membership differs from the split manifest: {archive.name}")
        for name in source.namelist():
            target = destination_file(root, name)
            if target.exists():
                if sha256(target) != expected[name]:
                    raise FileExistsError(f"Refusing to overwrite different existing file: {target}")
                continue
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError(f"Archive member checksum mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)


def download(assets, models, models_root, data_root, *, namespace="Xiaoang0211", dry_run=False):
    selected = ("model", "dataset") if assets == "all" else ("model",) if assets == "models" else ("dataset",)
    if dry_run:
        print(f"[plan] {assets}; models={','.join(models)}; probes={models_root}; data={data_root}")
        return
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    for kind in selected:
        repo = f"{namespace}/ProMo3D" + ("-data" if kind == "dataset" else "")
        if kind == "dataset" and data_root.is_symlink():
            raise FileExistsError(f"Dataset root is an existing link; keep it or choose a fresh --data-root: {data_root}")
        revision = HfApi().repo_info(repo, repo_type=kind).sha

        def fetch(filename):
            try:
                return Path(hf_hub_download(repo, filename, repo_type=kind, revision=revision))
            except GatedRepoError as exc:
                raise SystemExit("Accept the data terms at https://huggingface.co/datasets/" + repo +
                                 " and authenticate your local Hugging Face client before downloading.") from exc

        manifest_path = fetch("manifest.json")
        manifest = json.loads(manifest_path.read_text())
        records = {record["path"]: record for record in manifest["files"]}
        if kind == "model":
            for relative, record in records.items():
                if relative.split("/", 1)[0] in models:
                    install_file(fetch(relative), destination_file(models_root, relative), record["sha256"])
            print(f"[models] {','.join(models)} -> {models_root}")
        else:
            for split in ("train", "validation"):
                prefix = f"youtubevos_160vids/{split}/"
                split_path = fetch(prefix + "manifest.json")
                split_record = records[prefix + "manifest.json"]
                if sha256(split_path) != split_record["sha256"]:
                    raise ValueError(f"Split manifest checksum mismatch: {split}")
                info = json.loads(split_path.read_text())
                for filename, category in (("rgb_images.zip", "images"), ("binary_motion_masks.zip", "frames")):
                    relative = prefix + filename
                    archive = fetch(relative)
                    if sha256(archive) != records[relative]["sha256"]:
                        raise ValueError(f"Archive checksum mismatch: {relative}")
                    expected = {}
                    for sequence in info["sequences"]:
                        for frame in sequence[category]:
                            name = (f"JPEGImages/{sequence['sequence']}/{frame['image']}" if category == "images" else
                                    f"Annotations/{sequence['sequence']}/{frame['mask']}")
                            expected[name] = frame["sha256"]
                    extract_archive(archive, data_root / split, expected)
                install_file(split_path, data_root / split / "manifest.json", split_record["sha256"])
                print(f"[data] {split}: {info['sequence_count']} sequences -> {data_root / split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", choices=("models", "data", "all"), default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--models-root", type=Path, default=ROOT / "outputs/pretrained")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/youtubevos_160vids")
    parser.add_argument("--namespace", default="Xiaoang0211")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.assets, args.models, args.models_root, args.data_root, namespace=args.namespace, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
