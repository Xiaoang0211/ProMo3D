from pathlib import Path
from types import SimpleNamespace

import pytest

from isdynamic.probes.blockwise_sweep_common import resolve_existing_metrics_path


@pytest.mark.parametrize("model,canonical", [("pi3x","pi3x"),("pi3","pi3"),("da3","depth_anything_3"),
                                            ("vggt","vggt"),("mapanything","map_anything"),("dust3r","dust3r")])
def test_training_forwards_model_and_explicit_paths(runner, monkeypatch, model, canonical):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append((command, env)))
    destination = runner.probe_root(model)
    runner.train(model, destination, SimpleNamespace(dry_run=True))
    command, env = calls[0]
    assert command[command.index("--model") + 1] == canonical
    assert command[command.index("--rank") + 1] == 8
    assert Path(command[command.index("--results_dir") + 1]) == destination
    assert env["ANCHOR0_SAFEGUARD"] == "0"
    assert env["ANCHOR0_STATIC_PAIRS_PER_BATCH"] == "0"
    assert env["ANCHOR0_DYNAMIC_PAIRS_PER_BATCH"] == "0"


def test_masks_use_images_and_full_inference_with_fixed_context(runner, monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append((command, env)))
    runner.masks("pi3x", SimpleNamespace(dry_run=True))
    command, env = calls[0]
    assert Path(command[1]).name == "run_export_motion_masks_all.sh"
    assert env["CONTEXT_SIZE"] == "3" and env["CONTEXT_STRIDE"] == "2"
    assert env["MOTION_MASK_THRESHOLD_MODE"] == "otsu"
    assert env["MOTION_MASK_THRESHOLD_OFFSET"] == "0.2"
    assert Path(env["OUT_DIR"]).is_relative_to(runner.OUTPUT)
    assert "TOKEN_SOURCE" not in env and "SHARDED_ROOT" not in env


def test_coarse_and_refined_evaluation_remain_separate(runner, monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append(dict(env)))
    runner.masks("pi3x", SimpleNamespace(dry_run=True), stage="eval")
    assert [env["VARIANT"] for env in calls] == ["bilinear_upsample", runner.REFINED]
    assert all(Path(env["EVAL_ROOT"]).is_relative_to(runner.OUTPUT) for env in calls)


def test_dry_export_creates_no_dataset_view(runner, monkeypatch):
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: None)
    runner.export_tokens(["pi3x"], SimpleNamespace(dry_run=True))
    assert not runner.OUTPUT.exists()


def test_non_safeguard_does_not_reuse_old_enforced_metrics(tmp_path):
    current = tmp_path / "outputs/main/probes/pi3x/metrics.json"
    legacy = current.parent / "output_enforce_t0_anchor/metrics.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    assert resolve_existing_metrics_path(current) == current


def test_subset_is_seeded_and_preserves_other_splits(runner):
    source = runner.token_source("pi3x")
    for name in ("one", "two", "three", "four"):
        (source / "train" / name).mkdir(parents=True)
    for split in ("val", *runner.TEST_SPLITS.split(",")):
        (source / split).mkdir(parents=True)
    args = SimpleNamespace(dry_run=False)
    a = runner.subset("pi3x", 2, 42, args)
    b = runner.subset("pi3x", 2, 42, args)
    assert a == b
    assert len(list((a / "train").iterdir())) == 2
    assert (a / "val").resolve() == source / "val"
    assert (a / "test_davis").resolve() == source / "test_davis"


@pytest.mark.parametrize("model", ["pi3x", "pi3", "vggt"])
def test_native_padding_models_cannot_trigger_optional_export(runner, monkeypatch, model):
    calls=[]
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Choose da3"):
        runner.additional_experiment(SimpleNamespace(experiment="padding", stage="export", dry_run=True), [model])
    assert not calls


@pytest.mark.parametrize("model", ["da3", "mapanything", "dust3r"])
def test_optional_padding_export_is_separate_from_native_tokens(runner, monkeypatch, model):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append((command, env)))
    runner.additional_experiment(SimpleNamespace(experiment="padding", stage="export", dry_run=True), [model])
    command, env = calls[0]
    prefix = runner.SPECS[model][1]
    assert command[-1] == model
    assert env[f"{prefix}_PREPROCESS_MODE"] == "aspect_pad"
    assert Path(env[f"{prefix}_OUTPUT_DIR"]) == runner.TOKENS / "aspect_pad" / model / "data/isdynamic"
    assert env["ANCHOR0_SAFEGUARD"] == "0"


@pytest.mark.parametrize("model", ["vggt", "da3", "mapanything", "dust3r"])
def test_main_masks_preserve_native_preprocessing(runner, monkeypatch, model):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append(env))
    runner.masks(model, SimpleNamespace(dry_run=True))
    assert calls[0][f"{runner.SPECS[model][1]}_PREPROCESS_MODE"] == "native"
