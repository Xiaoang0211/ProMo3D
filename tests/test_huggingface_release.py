import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def publisher():
    spec = importlib.util.spec_from_file_location("pmd_hub_publisher", ROOT / "scripts/_publish_huggingface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_publication_does_not_require_auth_or_upload_files(capsys):
    publisher().publish(dry_run=True)
    text = capsys.readouterr().out
    assert "[plan] public model repository Xiaoang0211/ProMo3D" in text
    assert "[plan] public dataset repository Xiaoang0211/ProMo3D-data" in text
    assert "[page]" not in text


def test_publication_is_restricted_to_reviewed_cards(monkeypatch):
    calls = []
    api = SimpleNamespace(
        whoami=lambda: {"name": "Xiaoang0211"},
        create_repo=lambda **kwargs: calls.append(("create", kwargs)),
        repo_info=lambda **kwargs: SimpleNamespace(private=False),
        upload_file=lambda **kwargs: calls.append(("upload", kwargs)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api, get_token=lambda: "test-token"))
    publisher().publish()
    created = [kwargs for action, kwargs in calls if action == "create"]
    assert len(created) == 2 and all(kwargs["private"] is False for kwargs in created)
    uploads = [kwargs for action, kwargs in calls if action == "upload"]
    assert len(uploads) == 2
    assert all(kwargs["path_in_repo"] == "README.md" for kwargs in uploads)
    assert {Path(kwargs["path_or_fileobj"]).name for kwargs in uploads} == {"MODEL_CARD.md", "DATASET_CARD.md"}
    assert all(Path(kwargs["path_or_fileobj"]).is_relative_to(ROOT / "docs/huggingface") for kwargs in uploads)


def test_browser_login_is_not_misreported_as_api_authentication(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: None, get_token=lambda: None))
    with pytest.raises(SystemExit, match="Firefox login is separate"):
        publisher().publish()


def test_cards_do_not_claim_unreleased_weights_or_data_are_available():
    model = (ROOT / "docs/huggingface/MODEL_CARD.md").read_text()
    dataset = (ROOT / "docs/huggingface/DATASET_CARD.md").read_text()
    assert "does not contain weights" in model
    assert "does not contain downloadable datasets" in dataset
    assert "model.safetensors" not in model
    assert "scripts/evaluate.sh" in model


@pytest.fixture
def prepared_release(tmp_path):
    module = publisher()
    for name, kind in (("models", "model"), ("dataset", "dataset")):
        root = tmp_path / name
        root.mkdir()
        (root / "README.md").write_text("Reviewed release card")
        paths = ([f"{model}/rank_8/pair_lowrank_quadratic/inter_frame_block/inter_frame_block_0/best_probe.pt"
                  for model in sorted(module.MODELS)] if kind == "model" else
                 [f"youtubevos_160vids/{split}/binary_motion_masks.zip" for split in ("train", "validation")])
        records = []
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"verified release artifact")
            records.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        manifest = ({"models": {model: {} for model in module.MODELS}} if kind == "model" else
                    {"splits": {"train": {"sequences": 160}, "validation": {"sequences": 80}}, "gated": "auto"})
        manifest["files"] = records
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "not_for_publication.txt").write_text("Private extra file")
    return tmp_path


def test_prepared_publication_uses_explicit_allowlists_and_never_creates_a_space(prepared_release, monkeypatch):
    calls = []
    api = SimpleNamespace(
        whoami=lambda: {"name": "Xiaoang0211"},
        create_repo=lambda **kwargs: calls.append(("create", kwargs)),
        repo_info=lambda **kwargs: SimpleNamespace(private=False, gated="auto"),
        update_repo_settings=lambda **kwargs: calls.append(("gate", kwargs)),
        upload_folder=lambda **kwargs: calls.append(("upload", kwargs)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api, get_token=lambda: "test-token"))
    publisher().publish(release_dir=prepared_release)
    assert {kwargs["repo_type"] for action, kwargs in calls if action == "create"} == {"model", "dataset"}
    dataset_upload = next(i for i, (action, kwargs) in enumerate(calls) if action == "upload" and kwargs["repo_type"] == "dataset")
    gate = next(i for i, (action, kwargs) in enumerate(calls) if action == "gate")
    assert gate < dataset_upload
    for action, kwargs in calls:
        if action == "upload":
            assert "not_for_publication.txt" not in kwargs["allow_patterns"]
            assert {"README.md", "manifest.json"} <= set(kwargs["allow_patterns"])


def test_modified_checkpoint_fails_before_remote_publication(prepared_release):
    checkpoint = next((prepared_release / "models").rglob("best_probe.pt"))
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publisher().publish(dry_run=True, release_dir=prepared_release)


def test_publication_rejects_path_traversal(prepared_release):
    path = prepared_release / "dataset/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = "../../private.txt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Disallowed release path"):
        publisher().publish(dry_run=True, release_dir=prepared_release)


def test_no_huggingface_demo_placeholder_on_public_surfaces():
    for path in (ROOT / "README.md", ROOT / "docs/huggingface/MODEL_CARD.md", ROOT / "docs/huggingface/DATASET_CARD.md"):
        assert "huggingface.co/spaces/" not in path.read_text().lower()
