"""Prepare the paper's probes and source masks for two separate Hub repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch

from isdynamic.core.common_models import build_probe
from isdynamic.core.output_paths import anchor_enforcement_enabled_from_mapping
from isdynamic.release.import_results import LAYOUT

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {name: {"inter_frame_block": list(range(count))} for name, count in
          (("pi3x", 18), ("pi3", 18), ("vggt", 24), ("da3", 14), ("mapanything", 8))}
GROUPS["dust3r"] = {"dec_view1": list(range(12)), "head_in_view1": [0, 6, 9, 12]}
MODEL_KEYS = ("model_type", "input_dim", "hidden_dim", "dropout", "rank")


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": checksum(path)}


def prepare_models(outputs, destination, *, models=tuple(LAYOUT), reference_root=None):
    reference_root = reference_root or ROOT / "results/reference/main/probes"
    if "anchor0" in str(outputs) or "pos_residualized" in str(outputs):
        raise ValueError("Use the original paper outputs, not a later ablation root")
    files, released = [], {}
    for name in models:
        source = outputs / "model_outputs" / LAYOUT[name] / "pair_lowrank_quadratic"
        summary_path = source.parent / "summary.json"
        if summary_path.exists():
            token_source = str(json.loads(summary_path.read_text()).get("sharded_root", ""))
            if "anchor0" in token_source:
                raise ValueError(f"Later forced-anchor token export: {summary_path}")
        configs = sorted(source.glob("*/*/config.json"))
        if not configs:
            raise FileNotFoundError(f"No probe configurations: {source}")
        available = {path.parent.name for path in configs}
        required = {f"{group}_{i}" for group, indices in GROUPS[name].items() for i in indices}
        if not required <= available:
            raise ValueError(f"Incomplete selected blocks for {name}: {sorted(required - available)}")
        records = []
        for config_path in configs:
            config = json.loads(config_path.read_text())
            model = config["model"]
            train = config.get("train", {})
            if anchor_enforcement_enabled_from_mapping(train):
                raise ValueError(f"Safeguard probe is not part of the original paper release: {config_path}")
            if train.get("position_residualization", "none") != "none" or train.get("label_control", "none") != "none":
                raise ValueError(f"Later ablation probe: {config_path}")
            if model["model_type"] != "pair_lowrank_quadratic" or model.get("rank") != 8:
                raise ValueError(f"Expected rank-8 low-rank probe: {config_path}")
            if model.get("use_value_features") or (config_path.parent / "position_residualizer.pt").exists():
                raise ValueError(f"Not the native, unmodified-feature paper configuration: {config_path}")
            checkpoint = config_path.parent / config.get("output", {}).get("save_name", "best_probe.pt")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not all(isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError(f"Expected the original weights-only checkpoint: {checkpoint}")
            probe = build_probe("pair_lowrank_quadratic", input_dim=model["input_dim"], rank=8)
            probe.load_state_dict(state, strict=True)
            clean_model = {key: model[key] for key in MODEL_KEYS if key in model}
            relative = config_path.parent.relative_to(source)
            metrics = json.loads((config_path.parent / "metrics.json").read_text())
            reference = json.loads((reference_root / name / "rank_8/pair_lowrank_quadratic" / relative / "metrics.json").read_text())
            reference.pop("archived_source", None)
            if metrics != reference:
                raise ValueError(f"Metrics differ from the archived paper run: {config_path}")
            target = destination / name / "rank_8/pair_lowrank_quadratic" / relative
            target.mkdir(parents=True)
            shutil.copy2(checkpoint, target / "best_probe.pt")
            write_json(target / "config.json", {"model": clean_model, "output": {"save_name": "best_probe.pt"}})
            files.extend(file_record(target / filename, destination) for filename in ("config.json", "best_probe.pt"))
            records.append({"group": relative.parts[0], "block": relative.parts[1],
                            "source_config_sha256": checksum(config_path),
                            "source_checkpoint_sha256": checksum(checkpoint),
                            "seed": config.get("train", {}).get("seed")})
        released[name] = {"rank": 8, "preprocessing": "native", "native_padding": name in {"pi3", "pi3x", "vggt"},
                          "additional_padding_ablation": False, "anchor0_safeguard": False,
                          "token0_forced_export": False, "training_videos": 160,
                          "validation_videos": 80, "mask_groups": GROUPS[name], "probes": records}
        print(f"[models] {name}: {len(records)} verified rank-8 probes", flush=True)
    result = {"schema_version": 1, "models": released, "files": files,
              "contents": "PMD probes only; upstream backbone checkpoints are not included"}
    write_json(destination / "manifest.json", result)
    return result


def prepare_dataset(source, destination, *, splits=(("train", 160, "train"), ("validation", 80, "valid"))):
    files, released, memberships = [], {}, []
    for split, expected, official_split in splits:
        annotation_root = source / split / "Annotations"
        sequences = sorted(path for path in annotation_root.iterdir() if path.is_dir())
        if len(sequences) != expected:
            raise ValueError(f"Expected {expected} {split} sequences, found {len(sequences)}")
        directory = destination / "youtubevos_160vids" / split
        directory.mkdir(parents=True)
        sequence_records = []
        archive_path = directory / "binary_motion_masks.zip"
        rgb_path = directory / "rgb_images.zip"
        image_root = source / split / "JPEGImages"
        if not image_root.is_dir():
            image_root = source / split / "JEPGImages"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive, zipfile.ZipFile(rgb_path, "w") as rgb_archive:
            for sequence in sequences:
                masks = sorted(sequence.glob("*.png"))
                if not masks:
                    raise ValueError(f"No PNG motion masks: {sequence}")
                images = sorted(path for path in (image_root / sequence.name).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
                images_by_stem = {path.stem: path for path in images}
                if not images or len(images_by_stem) != len(images):
                    raise ValueError(f"Missing or ambiguous original RGB images for {sequence.name}")
                image_records = []
                for image in images:
                    rgb_archive.write(image, f"JPEGImages/{sequence.name}/{image.name}")
                    image_records.append({"image": image.name, "sha256": checksum(image)})
                frames = []
                for mask in masks:
                    if mask.stem not in images_by_stem:
                        raise ValueError(f"Motion mask has no matching RGB frame: {mask}")
                    with Image.open(mask) as image:
                        array = np.asarray(image)
                        if array.ndim != 2 or not np.isin(array, [0, 1, 255]).all():
                            raise ValueError(f"Not a binary motion mask: {mask}")
                    archive.write(mask, f"Annotations/{sequence.name}/{mask.name}")
                    frames.append({"mask": mask.name, "image": images_by_stem[mask.stem].name, "sha256": checksum(mask)})
                sequence_records.append({"sequence": sequence.name, "frames": frames, "images": image_records})
        manifest = {"split": split, "official_image_split": official_split,
                    "sequences": sequence_records, "sequence_count": len(sequence_records),
                    "frame_count": sum(len(record["frames"]) for record in sequence_records),
                    "image_count": sum(len(record["images"]) for record in sequence_records),
                    "mask_values": "0 is static/background; nonzero is dynamic",
                    "images": "Original subset RGB bytes; official source: https://youtube-vos.org/dataset/vis/"}
        write_json(directory / "manifest.json", manifest)
        files.extend(file_record(path, destination) for path in (archive_path, rgb_path, directory / "manifest.json"))
        released[split] = {"sequences": expected, "frames": manifest["frame_count"], "images": manifest["image_count"]}
        memberships.append({record["sequence"] for record in sequence_records})
        print(f"[data] {split}: {expected} sequences, {manifest['image_count']} images, {manifest['frame_count']} masks", flush=True)
    if len(memberships) == 2 and memberships[0] & memberships[1]:
        raise ValueError("Training and validation sequence memberships overlap")
    result = {"schema_version": 1, "splits": released, "files": files, "gated": "auto",
              "contents": "Original subset RGB images, binary motion masks, and split/frame manifests; no target-benchmark data"}
    write_json(destination / "manifest.json", result)
    return result


def prepare(outputs, youtubevos_root, destination):
    if destination.exists():
        raise FileExistsError(f"Choose a fresh release directory; refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pmd-release-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "release"
        models = prepare_models(outputs, stage / "models")
        masks = prepare_dataset(youtubevos_root, stage / "dataset")
        for folder, filename, status in (
            ("models", "MODEL_CARD.md", "This release contains the original-paper rank-8 PMD probes for all six backbones, trained without safeguard or forced token-0 export. Native backbone preprocessing is preserved. Upstream backbone weights are obtained from their official sources."),
            ("dataset", "DATASET_CARD.md", "This release contains original RGB images and the exact binary motion masks for 160 training and 80 validation sequences, with split/frame manifests. Downloads require agreement to the upstream research-use terms."),
        ):
            card = (ROOT / "docs/huggingface" / filename).read_text()
            start, end = card.index("<!-- RELEASE_STATUS -->"), card.index("<!-- END_RELEASE_STATUS -->")
            card = card[:start] + status + "\n\n" + card[end + len("<!-- END_RELEASE_STATUS -->"):]
            (stage / folder / "README.md").write_text(card)
        stage.rename(destination)
    print(f"[prepared] {destination}: {len(models['files'])} model files, {len(masks['files'])} dataset files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="Archived paper outputs containing model_outputs/")
    parser.add_argument("--youtubevos-root", type=Path, default=ROOT / "datasets/youtubevos_160vids")
    parser.add_argument("--destination", type=Path, default=ROOT / "outputs/releases/huggingface")
    args = parser.parse_args()
    prepare(args.outputs, args.youtubevos_root, args.destination)


if __name__ == "__main__":
    main()
