"""Release only the original paper probes and the approved source motion masks."""
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from isdynamic.core.common_models import build_probe
from isdynamic.release import huggingface as release


@pytest.fixture
def original_run(tmp_path, monkeypatch):
    monkeypatch.setitem(release.GROUPS, "pi3x", {"inter_frame_block": [0]})
    outputs, reference = tmp_path / "original_outputs", tmp_path / "reference"
    relative = Path("inter_frame_block/inter_frame_block_0")
    source = outputs / "model_outputs" / release.LAYOUT["pi3x"] / "pair_lowrank_quadratic" / relative
    source.mkdir(parents=True)
    config = {"model": {"model_type": "pair_lowrank_quadratic", "input_dim": 4, "rank": 8},
              "train": {"seed": 42}, "dataset": {"train_npz": "/private/training/data.npz"},
              "output": {"out_dir": "/private/original/run", "save_name": "best_probe.pt"}}
    (source / "config.json").write_text(json.dumps(config))
    torch.save(build_probe("pair_lowrank_quadratic", input_dim=4, rank=8).state_dict(), source / "best_probe.pt")
    (source / "metrics.json").write_text(json.dumps({"accuracy": 0.9}))
    ref = reference / "pi3x/rank_8/pair_lowrank_quadratic" / relative
    ref.mkdir(parents=True)
    (ref / "metrics.json").write_text(json.dumps({"accuracy": 0.9, "archived_source": "/private/paper/run"}))
    return outputs, reference, source


def test_checkpoint_bytes_preserved_and_config_paths_removed(original_run, tmp_path):
    outputs, reference, source = original_run
    target = tmp_path / "release"
    result = release.prepare_models(outputs, target, models=["pi3x"], reference_root=reference)
    copied = target / "pi3x/rank_8/pair_lowrank_quadratic/inter_frame_block/inter_frame_block_0"
    assert (copied / "best_probe.pt").read_bytes() == (source / "best_probe.pt").read_bytes()
    config = json.loads((copied / "config.json").read_text())
    assert set(config) == {"model", "output"}
    assert "/private" not in (copied / "config.json").read_text()
    assert result["models"]["pi3x"]["native_padding"] is True
    assert result["models"]["pi3x"]["anchor0_safeguard"] is False
    assert result["models"]["pi3x"]["token0_forced_export"] is False


@pytest.mark.parametrize("field,value", [("pair_force_anchor_static_label", True),
                                         ("pair_min_anchor_static_pairs_per_batch", 4),
                                         ("pair_min_anchor_dynamic_pairs_per_batch", 4)])
def test_safeguard_training_cannot_be_packaged_as_original(original_run, tmp_path, field, value):
    outputs, reference, source = original_run
    config = json.loads((source / "config.json").read_text())
    config["train"][field] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Safeguard"):
        release.prepare_models(outputs, tmp_path / "release", models=["pi3x"], reference_root=reference)


def test_mismatched_paper_metrics_are_rejected(original_run, tmp_path):
    outputs, reference, source = original_run
    (source / "metrics.json").write_text(json.dumps({"accuracy": 0.95}))
    with pytest.raises(ValueError, match="Metrics differ"):
        release.prepare_models(outputs, tmp_path / "release", models=["pi3x"], reference_root=reference)


def test_forced_token0_exports_are_rejected(original_run, tmp_path):
    outputs, reference, source = original_run
    (source.parents[2] / "summary.json").write_text(json.dumps({"sharded_root": "/tokens/pi3x_tokens_anchor0"}))
    with pytest.raises(ValueError, match="forced-anchor token export"):
        release.prepare_models(outputs, tmp_path / "release", models=["pi3x"], reference_root=reference)


def test_mask_archives_contain_only_original_binary_pngs_and_split_manifests(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "release"
    originals = {}
    for split in ("train", "validation"):
        path = source / split / "Annotations" / f"{split}_sequence/00000.png"
        path.parent.mkdir(parents=True)
        Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(path)
        image = source / split / "JEPGImages" / f"{split}_sequence/00000.jpg"
        image.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image)
        originals[split] = path.read_bytes()
    result = release.prepare_dataset(source, destination, splits=(("train", 1, "train"), ("validation", 1, "valid")))
    assert len(result["files"]) == 6
    assert result["gated"] == "auto"
    for split in originals:
        with zipfile.ZipFile(destination / "youtubevos_160vids" / split / "binary_motion_masks.zip") as archive:
            assert archive.namelist() == [f"Annotations/{split}_sequence/00000.png"]
            assert archive.read(archive.namelist()[0]) == originals[split]
        with zipfile.ZipFile(destination / "youtubevos_160vids" / split / "rgb_images.zip") as archive:
            assert archive.namelist() == [f"JPEGImages/{split}_sequence/00000.jpg"]
            assert archive.read(archive.namelist()[0]) == (source / split / "JEPGImages" / f"{split}_sequence/00000.jpg").read_bytes()


def test_preparation_refuses_to_overwrite_existing_destination(tmp_path):
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        release.prepare(tmp_path / "outputs", tmp_path / "data", tmp_path)
