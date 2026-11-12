"""HU-DS-04 · CA-3 — MegaDetector se busca y se descarga en %LOCALAPPDATA%\\WWM\\models (no junto al exe)."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from desktop import paths

PAYLOAD = b"pesos-falsos-de-megadetector" * 1000


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    return tmp_path / "wwm"


@pytest.fixture
def server():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/md_v5a.0.0.pt", hits
    httpd.shutdown()


def test_megadetector_path_is_in_data_dir(data_dir):
    assert paths.megadetector_path() == data_dir / "models" / "md_v5a.0.0.pt"


def test_megadetector_downloaded_to_appdata(data_dir, server):
    from desktop.ui.legacy_window import _download_weights

    url, hits = server
    target = paths.megadetector_path()
    _download_weights(url, target)
    assert target.read_bytes() == PAYLOAD and len(hits) == 1
    assert not target.with_suffix(".tmp").exists()

    _download_weights(url, target)  # ya existe: no se vuelve a descargar
    assert len(hits) == 1


def test_window_uses_appdata_model_path():
    """La ventana crea el detector con las rutas de AppData (no con app.config.MEGADETECTOR_PATH), y SpeciesNet
    desde su carpeta local (HU-DS-11), no desde Kaggle."""
    source = (Path(__file__).resolve().parents[2] / "desktop/ui/legacy_window.py").read_text(encoding="utf-8")
    assert "WildlifeDetector(paths.megadetector_path(), paths.speciesnet_dir())" in source
    assert "MEGADETECTOR_PATH" not in source


def test_migrate_legacy_model_copies_once(data_dir, tmp_path):
    old = tmp_path / "instalacion_vieja" / "models" / "md_v5a.0.0.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(PAYLOAD)

    assert paths.migrate_legacy_model([old]) == old
    assert paths.megadetector_path().read_bytes() == PAYLOAD
    assert old.exists()  # se copia, no se mueve
    assert not paths.megadetector_path().with_suffix(".part").exists()
    assert paths.migrate_legacy_model([old]) is None  # una sola vez


def test_migrate_ignores_missing_or_empty(data_dir, tmp_path):
    empty = tmp_path / "vacio.pt"
    empty.write_bytes(b"")
    assert paths.migrate_legacy_model([tmp_path / "no_existe.pt", empty]) is None
    assert not paths.megadetector_path().exists()
