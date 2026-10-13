from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "new outputs"))
    monkeypatch.setenv("TOKEN_ROOT", str(tmp_path / "new tokens"))
    for variable in ("YOUTUBEVOS_ROOT", "DAVIS_ROOT", "SEGTRACK_ROOT", "FBMS_ROOT"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    for prefix in ("PI3", "VGGT", "DA3", "MAPANYTHING", "DUST3R"):
        monkeypatch.setenv(f"{prefix}_PYTHON_BIN", sys.executable)
    spec = importlib.util.spec_from_file_location("pmd_release_runner", ROOT / "scripts/_reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
