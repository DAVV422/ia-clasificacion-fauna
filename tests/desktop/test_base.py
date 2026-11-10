"""HU-DS-00 · CA-1…CA-6, CA-9 — base del proyecto de escritorio."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from desktop import __version__, paths
from desktop.logging_setup import anonymize, configure, log_event
from desktop.messages import ERROR_CODES, MESSAGES, message_for

ROOT = Path(__file__).resolve().parents[2]
E11_CODES = {
    "NETWORK_UNAVAILABLE",
    "DOWNLOAD_INTEGRITY_FAILED",
    "REQUIREMENTS_NOT_MET",
    "MODEL_LOAD_FAILED",
    "DISK_SPACE_LOW",
    "DESTINATION_NOT_WRITABLE",
    "SAME_OR_NESTED_FOLDER",
    "IA_DECODE_FAILED",
    "IA_INVALID_FORMAT",
    "FILE_LOCKED",
    "FILE_MOVE_VERIFY_FAILED",
    "OUT_OF_MEMORY",
    "EXPORT_INVALID",
    "INTERNAL_ERROR",
}


def _env(tmp_path, **extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"TORCH_HOME", "KAGGLEHUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "YOLO_CONFIG_DIR"}
    }
    env.update({"WWM_DATA_DIR": str(tmp_path / "datos"), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"})
    env.update(extra)
    return env


def test_ensure_dirs_creates_tree(tmp_path, monkeypatch):
    """CA-1"""
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "WWM"))
    paths.ensure_dirs()
    for name in ("models", "cache", "logs", "tmp"):
        assert (tmp_path / "WWM" / name).is_dir()


def test_data_dir_defaults_to_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv("WWM_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.data_dir() == tmp_path / "WWM"


def test_paths_module_has_no_side_effects(tmp_path):
    code = "import desktop.paths, desktop, sys; assert 'torch' not in sys.modules; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "datos").exists() and list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_selftest_and_env_configured_before_torch_import(tmp_path):
    """CA-1, CA-2: --selftest crea las rutas y torch usa la caché de la app (las variables se fijaron antes)."""
    r = subprocess.run(
        [sys.executable, "-m", "desktop.main", "--selftest"],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = dict(line.split(": ", 1) for line in r.stdout.splitlines() if ": " in line)
    data = tmp_path / "datos"
    assert Path(out["torch_hub_dir"]).resolve() == (data / "cache" / "torch" / "hub").resolve()
    for name in ("models", "cache", "logs", "tmp"):
        assert (data / name).is_dir()
    lines = (data / "logs" / "wwm.log").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l)["event"] for l in lines]
    assert "desktop.started" in events and "desktop.selftest.completed" in events


def test_version_flag_and_title_use_same_version(tmp_path):
    """CA-9"""
    r = subprocess.run(
        [sys.executable, "-m", "desktop.main", "--version"],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.stdout.strip() == __version__
    window_src = (ROOT / "desktop/ui/legacy_window.py").read_text(encoding="utf-8")
    assert "v{__version__}" in window_src and "Prototipo v1" not in window_src


def test_log_line_is_json_with_fields(tmp_path):
    """CA-3"""
    path = configure(log_dir=tmp_path)
    try:
        log_event("desktop.test", "hola", job_id_hint="x")
        try:
            raise RuntimeError("falla de prueba")
        except RuntimeError:
            logging.getLogger("desktop").exception("con error", extra={"event": "desktop.error"})
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_wwm", False):
                h.close()
                logging.getLogger().removeHandler(h)
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    first = records[0]
    assert {"ts", "level", "service", "version", "job_id", "event", "message"} <= set(first)
    assert first["service"] == "desktop" and first["version"] == __version__ and first["event"] == "desktop.test"
    assert "falla de prueba" in records[1]["exc_info"]


def test_log_anonymizes_user_paths():
    """CA-3"""
    home = str(Path.home())
    text = anonymize(f"error leyendo {home}\\Desktop\\SD\\video.mp4")
    assert home not in text and "<usuario>" in text


def test_log_rotation_keeps_5(tmp_path, monkeypatch):
    """CA-4"""
    import desktop.logging_setup as ls

    monkeypatch.setattr(ls, "MAX_BYTES", 2000)
    ls.configure(log_dir=tmp_path)
    try:
        for i in range(400):
            log_event("desktop.test", "x" * 80, i=i)
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_wwm", False):
                h.close()
                logging.getLogger().removeHandler(h)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 5 and "wwm.log" in files


def test_every_error_code_has_message():
    """CA-6"""
    assert E11_CODES <= ERROR_CODES
    assert all(MESSAGES[c].strip() for c in E11_CODES)
    assert "AVX2" in message_for("REQUIREMENTS_NOT_MET", motivo="falta AVX2")
    assert "j-123" in message_for("INTERNAL_ERROR", ref="j-123")
    assert message_for("CODIGO_DESCONOCIDO").startswith("Ocurrió un error inesperado")
