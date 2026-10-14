from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest


def options(tmp_path, **overrides):
    values = dict(dry_run=True, probe_root=tmp_path / "trained probes", probes_root=None,
                  output_dir=tmp_path / "predictions", davis_root=tmp_path / "DAVIS",
                  segtrack_root=tmp_path / "SegTrackv2", fbms_root=tmp_path / "FBMS-59",
                  benchmarks=["davis2017", "segtrackv2", "fbms59"], sam2=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("model", ["pi3x", "pi3", "vggt", "da3", "mapanything", "dust3r"])
def test_pretrained_prediction_reuses_dense_inference_and_qualitative_outputs(runner, monkeypatch, tmp_path, model):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append((command, env)))
    args = options(tmp_path)
    runner.predict(args, [model])
    command, env = calls[0]
    assert Path(command[1]).name == "run_export_motion_masks_all.sh"
    assert env["ISDYNAMIC_MASK_OUTPUT_MODE"] == "full"
    assert env["PIPELINE_REFINED_AFTER_EACH_SEQUENCE"] == "0"
    assert env["CONTEXT_SIZE"] == "3" and env["CONTEXT_STRIDE"] == "2"
    assert env["ANCHOR0_SAFEGUARD"] == "0"
    assert env["OUT_DIR"] == str(args.output_dir / "masks")
    assert env["DAVIS_ROOT"] == str(args.davis_root)
    assert env["SEGTRACK_ROOT"] == str(args.segtrack_root)
    assert env["FBMS_ROOT"] == str(args.fbms_root)
    prefix = "PI3X" if model == "pi3x" else runner.SPECS[model][1]
    assert env[f"{prefix}_PROBE_ROOT"] == str(args.probe_root)
    assert calls[1][1]["VARIANT"] == "bilinear_upsample"
    assert calls[0][1]["BENCHMARKS"] == "davis_all,segtrackv2,fbms59"
    assert calls[1][1]["BENCHMARKS"] == "davis2017,segtrackv2,fbms59"
    assert len(calls) == 3
    assert calls[-1][0][2] == "isdynamic.release.tables"
    assert not args.output_dir.exists()
    assert not runner.TOKENS.exists()


def test_sam2_adds_refinement_and_separate_evaluation(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append((command, dict(env or {}))))
    runner.predict(options(tmp_path, sam2=True), ["pi3x"])
    assert Path(calls[1][0][1]).name == "run_refine_motion_masks_all.sh"
    assert calls[1][1]["BENCHMARKS"] == "davis_all,segtrackv2,fbms59"
    assert [env["VARIANT"] for _, env in calls if "VARIANT" in env] == ["bilinear_upsample", runner.REFINED]
    assert all(env["VARIANTS"] == "sam2_prompt" for _, env in calls[:-1])


def test_bundle_resolves_independent_probe_roots(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append(env))
    bundle = tmp_path / "model bundle"
    runner.predict(options(tmp_path, probe_root=None, probes_root=bundle), ["pi3x", "da3"])
    assert calls[0]["PI3X_PROBE_ROOT"] == str(bundle / "pi3x/pair_lowrank_quadratic")
    assert calls[2]["DA3_PROBE_ROOT"] == str(bundle / "da3/pair_lowrank_quadratic")


def test_one_probe_root_cannot_be_reused_for_multiple_backbones(runner, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        runner.predict(options(tmp_path), ["pi3x", "da3"])


def test_huggingface_rank8_bundle_is_resolved_without_training(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, env=None, dry=False: calls.append(env))
    bundle = tmp_path / "released probes"
    probe = bundle / "pi3x/rank_8/pair_lowrank_quadratic"
    probe.mkdir(parents=True)
    runner.predict(options(tmp_path, probe_root=None, probes_root=bundle), ["pi3x"])
    assert calls[0]["PI3X_PROBE_ROOT"] == str(probe)


def test_new_inference_does_not_require_training_data_and_rejects_stale_settings(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: None)
    args = options(tmp_path, dry_run=False, benchmarks=["davis2017"])
    args.probe_root.mkdir()
    args.davis_root.mkdir()
    assert not Path(runner.os.environ["YOUTUBEVOS_ROOT"]).exists()
    runner.predict(args, ["pi3x"])
    assert (args.output_dir / "run_pi3x.json").is_file()
    other = tmp_path / "different probes"
    other.mkdir()
    args.probe_root = other
    with pytest.raises(ValueError, match="new --output-dir"):
        runner.predict(args, ["pi3x"])


def test_existing_untracked_masks_are_not_silently_reused(runner, tmp_path):
    args = options(tmp_path, dry_run=False, benchmarks=["davis2017"])
    args.probe_root.mkdir()
    args.davis_root.mkdir()
    (args.output_dir / "masks/pi3x").mkdir(parents=True)
    with pytest.raises(ValueError, match="no matching run manifest"):
        runner.predict(args, ["pi3x"])


def test_entrypoint_runs_from_a_parent_project_without_changing_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "prediction outputs"
    result = subprocess.run(
        ["bash", str(root / "scripts/evaluate.sh"), "pi3x", "--probe-root", "probes",
         "--davis-root", "DAVIS", "--benchmarks", "davis2017",
         "--output-dir", "prediction outputs", "--dry-run"], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PROMO3D_CONFIG": str(tmp_path / "unused.env"), "RUNNER_PYTHON_BIN": sys.executable})
    assert result.returncode == 0, result.stderr
    assert "run_export_motion_masks_all.sh" in result.stdout
    assert "run_eval_motion_masks_all.sh" in result.stdout
    assert "run_refine_motion_masks_all.sh" not in result.stdout
    assert "build_youtubevos" not in result.stdout
    assert '"ISDYNAMIC_MASK_OUTPUT_MODE": "full"' in result.stdout
    assert str(output) in result.stdout
    assert str(tmp_path / "probes") in result.stdout
    assert str(tmp_path / "DAVIS") in result.stdout
    assert not output.exists()
