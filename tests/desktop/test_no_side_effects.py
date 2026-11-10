"""HU-DS-04 · CA-1 y CA-7 — importar no crea nada en disco y el escritorio no usa rutas fijas."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Registra toda creación de carpetas y toda apertura de archivo para escribir mientras se importan los módulos.
PROBE = r"""
import builtins, json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
calls = []
_mkdir, _makedirs, _osmkdir, _open = pathlib.Path.mkdir, os.makedirs, os.mkdir, builtins.open

def mkdir(self, *a, **k):
    calls.append(["mkdir", str(self)]); return _mkdir(self, *a, **k)
def makedirs(p, *a, **k):
    calls.append(["makedirs", str(p)]); return _makedirs(p, *a, **k)
def osmkdir(p, *a, **k):
    calls.append(["mkdir", str(p)]); return _osmkdir(p, *a, **k)
def open_(f, mode="r", *a, **k):
    if any(c in mode for c in "wax+") and not str(f).endswith((".pyc", ".tmp")):
        calls.append(["open", str(f)])
    return _open(f, mode, *a, **k)

pathlib.Path.mkdir, os.makedirs, os.mkdir, builtins.open = mkdir, makedirs, osmkdir, open_
import app.config, wwm_pipeline.pipeline, wwm_pipeline.contract, desktop.main, desktop.paths, desktop.logging_setup
print(json.dumps(calls))
"""


def test_imports_have_no_fs_side_effects(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = {**os.environ, "WWM_DATA_DIR": str(tmp_path / "data"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT)], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    calls = json.loads(r.stdout.strip().splitlines()[-1])
    assert calls == [], f"importar escribió en disco: {calls}"
    assert list(cwd.iterdir()) == []
    assert not (tmp_path / "data").exists()


def test_no_fixed_paths_in_desktop():
    """CA-7: grep de BASE_DIR|MODELS_DIR|DETECTIONS_DIR|TEMP_DIR en desktop/ y wwm_pipeline/ -> 0."""
    pattern = re.compile(r"BASE_DIR|MODELS_DIR|DETECTIONS_DIR|TEMP_DIR")
    hits = [
        f"{path.relative_to(ROOT)}:{n}"
        for folder in ("desktop", "wwm_pipeline")
        for path in (ROOT / folder).rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


def test_api_creates_its_dirs_on_startup(tmp_path, monkeypatch):
    """CA-6 (parte unitaria): las carpetas de la API se crean con ensure(), no al importar."""
    from app.config import api_dirs

    dirs = api_dirs(tmp_path / "api")
    assert not (tmp_path / "api").exists()
    dirs.ensure()
    assert dirs.models.is_dir() and dirs.detections.is_dir() and dirs.temp.is_dir()
