"""HU-DS-20 · Requisitos del equipo: niveles, AVX2 por dos vías, BLOQUEADA, aviso y el instalador."""

import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from desktop import paths, settings
from desktop import requirements_check as rc

GOOD = rc.SystemInfo(
    windows="Windows 11 (compilación 26200)",
    windows_build=26200,
    is_64bit=True,
    cpu="Intel i5",
    cores=8,
    avx2=True,
    ram_gb=19.7,
    disk_free_gb=30.0,
    versions={"app": "0.11.0"},
)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    monkeypatch.delenv("WWM_FAKE_REQUIREMENTS", raising=False)
    paths.ensure_dirs()
    yield
    paths.set_models_dir(None)


def levels(report):
    return {c.name: c.level for c in report.checks}


# ---------------------------------------------------------------------------
# CA-2 · CA-3 · CA-4 · Niveles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change,download,name,level,blocked",
    [
        ({}, False, "ram", "ok", False),
        ({"is_64bit": False}, False, "windows", "fail", True),
        ({"windows_build": 9600}, False, "windows", "fail", True),
        ({"avx2": False}, False, "avx2", "fail", True),
        ({"avx2": None}, False, "avx2", "warn", False),  # desconocido: avisa, no bloquea
        ({"cores": 2}, False, "avx2", "warn", False),
        ({"ram_gb": 3.8}, False, "ram", "warn", False),  # bajo el mínimo: aviso (E-02), no bloqueo
        ({"ram_gb": 11.8}, False, "ram", "warn", False),  # entre el mínimo y lo recomendado
        ({"ram_gb": 7.7}, False, "ram", "warn", False),  # 8 GB instalados se informan como ~7,7: cumple el mínimo
        ({"ram_gb": 15.7}, False, "ram", "ok", False),
        ({"disk_free_gb": 1.0}, True, "disk", "warn", False),  # faltan los modelos y no hay lugar
        ({"disk_free_gb": 1.0}, False, "disk", "ok", False),  # los modelos ya están
    ],
)
def test_requirements_report_levels(change, download, name, level, blocked):
    report = rc.check(replace(GOOD, **change), needs_model_download=download)
    assert levels(report)[name] == level
    assert report.blocked is blocked


def test_ram_messages_distinguish_minimum_and_recommended():
    under = rc.check(replace(GOOD, ram_gb=3.8))
    between = rc.check(replace(GOOD, ram_gb=11.8))
    assert "menos de 8 GB" in under.warnings[0].message
    assert "se recomiendan 16 GB" in between.warnings[0].message


def test_blocked_reason_for_message():
    report = rc.check(replace(GOOD, avx2=False))
    assert report.reason == "falta AVX2 en el procesador"
    from desktop.messages import message_for

    assert "falta AVX2" in message_for("REQUIREMENTS_NOT_MET", motivo=report.reason)


def test_warning_text_has_estimated_time():
    text = rc.warning_text(rc.check(replace(GOOD, ram_gb=11.8)))
    assert "se recomiendan 16 GB" in text and f"{rc.SECONDS_PER_15S_VIDEO} s" in text and "estimado" in text


def test_system_info_text_has_every_row():
    text = rc.system_info_text(rc.check(GOOD))
    for label in ("Windows", "Procesador", "Memoria (RAM)", "Disco libre", "app: 0.11.0"):
        assert label in text
    assert "30,0 GB" in text  # coma decimal


# ---------------------------------------------------------------------------
# CA-5 · AVX2 por dos vías
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "windows,numpy_,expected",
    [
        (True, True, True),
        (False, True, True),
        (True, None, True),
        (False, False, False),
        (None, False, False),
        (None, None, None),
    ],
)
def test_avx2_two_sources(windows, numpy_, expected):
    assert rc.combine_avx2([lambda: windows, lambda: numpy_]) is expected


def test_avx2_on_this_machine_is_known():
    assert rc.combine_avx2([rc.avx2_from_windows, rc.avx2_from_numpy]) is not None


# ---------------------------------------------------------------------------
# Simulación para QA-14: nunca en el ejecutable sin --qa
# ---------------------------------------------------------------------------
def test_fakes_only_outside_frozen_app(monkeypatch):
    monkeypatch.setenv("WWM_FAKE_REQUIREMENTS", "no_avx2,ram4")
    info = rc.probe()
    assert info.avx2 is False and info.ram_gb == 4
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["WWMClasificador.exe"])
    assert rc.probe().avx2 is not False  # en el ejecutable se ignora
    monkeypatch.setattr(sys, "argv", ["WWMClasificador.exe", "--qa"])
    assert rc.probe().avx2 is False


def test_check_requirements_cli(tmp_path):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, WWM_DATA_DIR=str(tmp_path / "cli"), WWM_FAKE_REQUIREMENTS="no_avx2")
    proc = subprocess.run(
        [sys.executable, "-m", "desktop.main", "--check-requirements"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    report = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert report["blocked"] is True and {c["name"]: c["level"] for c in report["checks"]}["avx2"] == "fail"
    env["WWM_FAKE_REQUIREMENTS"] = ""
    ok = subprocess.run(
        [sys.executable, "-m", "desktop.main", "--check-requirements"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ok.returncode == 0


# ---------------------------------------------------------------------------
# CA-2 · CA-3 · El arranque
# ---------------------------------------------------------------------------
def test_blocked_view_does_not_load_models(monkeypatch):
    from desktop import main
    from desktop import model_store as ms
    from desktop.ui import first_run, requirements_view

    monkeypatch.setenv("WWM_FAKE_REQUIREMENTS", "no_avx2")
    shown = []
    monkeypatch.setattr(requirements_view, "show_blocked", lambda report: shown.append(report))
    monkeypatch.setattr(first_run, "run_first_run", lambda *a: pytest.fail("no debía preparar los modelos"))
    monkeypatch.setattr(ms, "quick_check", lambda r, m: list(m.artifacts))
    assert main.ensure_models() is False
    assert shown and shown[0].blocked


def test_warning_shown_once_per_version(monkeypatch):
    from desktop import main
    from desktop import model_store as ms
    from desktop.ui import requirements_view

    monkeypatch.setenv("WWM_FAKE_REQUIREMENTS", "ram12")
    monkeypatch.setattr(ms, "quick_check", lambda r, m: [])
    shown = []
    monkeypatch.setattr(requirements_view.messagebox, "showwarning", lambda *a, **k: shown.append(a))
    real_tk = requirements_view.tk.Tk

    class HiddenTk(real_tk):  # type: ignore[misc, valid-type]
        def __init__(self):
            super().__init__()
            self.withdraw()

    monkeypatch.setattr(requirements_view.tk, "Tk", HiddenTk)
    try:
        assert main.ensure_models() is True
    except Exception as exc:  # pragma: no cover - sin pantalla
        pytest.skip(f"sin pantalla: {exc}")
    assert main.ensure_models() is True
    assert len(shown) == 1  # la segunda vez, misma versión: no se repite
    from desktop import __version__

    assert settings.get("requirements_warned_version") == __version__


def test_blocked_window_and_system_info():
    tk = pytest.importorskip("tkinter")
    from desktop.ui import requirements_view

    report = rc.check(replace(GOOD, avx2=False))
    for attempt in range(3):
        try:
            window = requirements_view.BlockedWindow(report)
            break
        except tk.TclError as exc:  # pragma: no cover
            if attempt == 2:
                pytest.skip(f"sin pantalla: {exc}")
            time.sleep(0.2)
    window.withdraw()
    labels = [w.cget("text") for w in window.winfo_children() if isinstance(w, tk.Label)]
    assert any("falta AVX2" in t for t in labels)
    info = window.show_info()
    info.withdraw()
    texts = [
        w.cget("text") for frame in info.winfo_children() for w in frame.winfo_children() if isinstance(w, tk.Label)
    ]
    assert any("AVX2 ✗" in t for t in texts)
    with contextlib.suppress(Exception):
        window.destroy()


# ---------------------------------------------------------------------------
# CA-1 · El instalador usa la misma tabla
# ---------------------------------------------------------------------------
def test_installer_requirements_generated_from_same_table():
    from tools import gen_requirements_iss as gen

    text = gen.render()
    assert f"#define WwmRamMinMB {rc.REQUIREMENTS['ram_min_gb'] * 1024}" in text
    assert "IsWin64" in text and "MB_DEFBUTTON2" in text  # 64 bits obligatorio; "No" por defecto
    assert gen.OUT.read_text(encoding="utf-8-sig") == text, "installer/requirements.iss desactualizado: regenerarlo"
