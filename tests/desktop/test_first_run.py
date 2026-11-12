"""HU-DS-13 · La pantalla de primer inicio y el arranque (E-02 paso 2)."""

import contextlib
import time

import pytest

from desktop import model_store as ms
from desktop import paths
from desktop.ui import first_run


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()
    yield
    paths.set_models_dir(None)


MANIFEST = ms.Manifest(
    1,
    [
        ms.Artifact("megadetector", "MegaDetector v5a", "5.0", "http://x/md.pt", 280 * 1024 * 1024, "a" * 64, "md.pt"),
        ms.Artifact(
            "speciesnet", "SpeciesNet 4.0.2a", "4.0.2a", "http://x/sn.zip", 220 * 1024 * 1024, "b" * 64, "sn/", True
        ),
    ],
)


# ---------------------------------------------------------------------------
# Textos
# ---------------------------------------------------------------------------
def test_progress_text():
    assert first_run.progress_text(232 * 1024 * 1024, 280 * 1024 * 1024) == "82 %  ·  232/280 MB"
    assert first_run.progress_text(0, 0) == "0 %  ·  0/0 MB"


def test_rate_and_remaining():
    now = [0.0]
    meter = first_run.RateMeter(clock=lambda: now[0])
    assert meter.text(100) == "Calculando velocidad…"
    meter.add(0)
    now[0] = 2.0
    meter.add(int(8.4 * 1024 * 1024))  # 4,2 MB/s
    assert meter.text(500 * 1024 * 1024) == "4,2 MB/s  ·  quedan ~2 min"
    assert meter.text(10 * 1024 * 1024) == "4,2 MB/s  ·  quedan ~2 s"
    now[0] = 9.0
    meter.add(int(8.4 * 1024 * 1024))  # la ventana de 5 s descarta lo viejo
    assert len(meter.samples) == 1


def test_error_texts_and_buttons():
    assert "No hay conexión" in first_run.error_text("NETWORK_UNAVAILABLE")
    assert first_run.ERROR_BUTTONS["NETWORK_UNAVAILABLE"] == ("Reintentar", "Salir")
    assert "Abrir carpeta de logs" in first_run.ERROR_BUTTONS["DOWNLOAD_INTEGRITY_FAILED"]


# ---------------------------------------------------------------------------
# La pantalla (Tk oculta)
# ---------------------------------------------------------------------------
def make_window(install):
    tk = pytest.importorskip("tkinter")
    for attempt in range(3):  # Tcl en Windows a veces falla con muchas ventanas en el mismo proceso
        try:
            window = first_run.FirstRunWindow(paths.models_dir(), MANIFEST, install=install)
            window.withdraw()
            return window
        except tk.TclError as exc:  # pragma: no cover
            if attempt == 2:
                pytest.skip(f"sin pantalla: {exc}")
            time.sleep(0.2)


def pump(window, until, timeout=10):
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline
        try:
            window.update()
        except Exception:
            return
        time.sleep(0.01)


def test_first_run_view_progress():
    def fake_install(root, manifest, on_progress, should_stop):
        for done in (0, 100 * 1024 * 1024, 280 * 1024 * 1024):
            on_progress(ms.Progress("megadetector", done, 280 * 1024 * 1024))
            time.sleep(0.05)
        return ms.InstallResult(root, downloaded=["megadetector"], reused=["speciesnet"])

    window = make_window(fake_install)
    statuses = []
    pump(window, lambda: (statuses.append(window.rows["megadetector"][1].cget("text")) or window.result is not None))
    assert window.result is True
    assert any("%" in s and "MB" in s for s in statuses)
    with contextlib.suppress(Exception):
        window.destroy()


def test_cancel_returns_false():
    def slow_install(root, manifest, on_progress, should_stop):
        while not should_stop():
            time.sleep(0.01)
        raise ms.Cancelled()

    window = make_window(slow_install)
    pump(window, lambda: window.stop is not None, timeout=1)
    window.after(200, window._on_cancel)
    pump(window, lambda: window.result is not None)
    assert window.result is False and window.stop.is_set()


def test_network_error_shows_retry_and_exit():
    calls = []

    def failing_install(root, manifest, on_progress, should_stop):
        calls.append(1)
        if len(calls) == 1:
            raise ms.ModelStoreError("NETWORK_UNAVAILABLE")
        return ms.InstallResult(root)

    window = make_window(failing_install)
    pump(window, lambda: "No hay conexión" in window.message.cget("text"))
    labels = [b.cget("text") for b in window.buttons.winfo_children()]
    assert labels == ["Reintentar", "Salir"]
    window.buttons.winfo_children()[0].invoke()  # Reintentar
    pump(window, lambda: window.result is not None)
    assert window.result is True and len(calls) == 2


# ---------------------------------------------------------------------------
# El arranque
# ---------------------------------------------------------------------------
def test_startup_skips_download_when_models_ok(monkeypatch, tmp_path):
    from desktop import main

    root = paths.data_dir() / "models"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ms, "load_manifest", lambda path=None: MANIFEST)
    monkeypatch.setattr(ms, "quick_check", lambda r, m: [])
    monkeypatch.setattr(first_run, "run_first_run", lambda *a: pytest.fail("no debía abrir la descarga"))
    assert main.ensure_models() is True
    assert paths.models_dir() == root


def test_startup_opens_download_when_missing(monkeypatch):
    from desktop import main

    monkeypatch.setattr(ms, "load_manifest", lambda path=None: MANIFEST)
    opened = []
    monkeypatch.setattr(first_run, "run_first_run", lambda root, manifest: opened.append(root) or False)
    assert main.ensure_models() is False  # la persona canceló: la app no abre la ventana principal
    assert opened == [paths.data_dir() / "models"]


def test_full_variant_uses_install_dir(monkeypatch, tmp_path):
    from desktop import main

    install_dir = tmp_path / "Program Files" / "WWM" / "models"
    install_dir.mkdir(parents=True)
    (install_dir / ms.MODELS_OK).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ms, "install_models_dir", lambda: install_dir)
    monkeypatch.setattr(ms, "load_manifest", lambda path=None: MANIFEST)
    monkeypatch.setattr(ms, "quick_check", lambda r, m: [] if r == install_dir else list(m.artifacts))
    monkeypatch.setattr(first_run, "run_first_run", lambda *a: pytest.fail("la completa nunca descarga"))
    assert main.ensure_models() is True and paths.models_dir() == install_dir
