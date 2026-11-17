"""HU-IA-05 · La IA no guarda nada: ni el archivo recibido, ni recortes, ni mounts (IA §1)."""

import itertools
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint

pytest.importorskip("cv2")


def files_under(folder: Path) -> set:
    return {p for p in folder.rglob("*") if p.is_file()} if folder.exists() else set()


def cwd_outputs() -> set:
    """Donde escribía la API del prototipo: detections/, temp/ y archivos sueltos en la carpeta de trabajo."""
    cwd = Path.cwd()
    return files_under(cwd / "detections") | files_under(cwd / "temp") | {p for p in cwd.iterdir() if p.is_file()}


@pytest.fixture
def setup(tmp_path):
    sint = tmp_path / "sint"
    write_f_set_sint(sint)
    tmp_dir = tmp_path / "tmp"
    app = create_app(Settings(models_dir=tmp_path / "models", tmp_dir=tmp_dir), models=SetModels())
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, sint, tmp_dir, tmp_path


# ---------------------------------------------------------------------------
# CA-1 · Después de 50 peticiones (con éxito y con error), no queda ningún archivo
# ---------------------------------------------------------------------------
def test_no_files_left_after_50_requests(setup):
    client, sint, tmp_dir, tmp_path = setup
    names = ["especie_a.mp4", "especie_b.jpg", "vacio_g.jpg", "danado_j.mp4", "persona_h.mp4"]
    cwd_before = cwd_outputs()
    sint_before = files_under(sint)
    codes = []
    for name in itertools.islice(itertools.cycle(names), 50):
        with open(sint / name, "rb") as f:
            r = client.post("/api/v1/detect", files={"file": (name, f)}, data={"file_name": name})
        codes.append(r.status_code)
    assert set(codes) == {200, 422}  # con éxito y con error (el dañado)
    assert files_under(tmp_dir) == set() and (not tmp_dir.exists() or not any(tmp_dir.iterdir()))
    assert files_under(sint) == sint_before
    assert cwd_outputs() == cwd_before  # tampoco en la carpeta de trabajo (detections/, temp/…)


# ---------------------------------------------------------------------------
# CA-2 · Sin mount
# ---------------------------------------------------------------------------
def test_no_static_mount(setup):
    client = setup[0]
    assert client.get("/detections/cualquier/recorte.jpg").status_code == 404
    assert all(getattr(r, "name", "") != "detections" for r in client.app.routes)


# ---------------------------------------------------------------------------
# CA-3 · Con error inesperado también se borra (el timeout se prueba en HU-IA-04)
# ---------------------------------------------------------------------------
def test_tmp_removed_on_unexpected_error(setup, monkeypatch):
    import api.routes as routes

    client, sint, tmp_dir, _ = setup
    seen = []

    def boom(path, *a, **k):
        seen.append(Path(path))
        assert Path(path).is_file()  # el archivo existía mientras se procesaba
        raise RuntimeError("falla")

    monkeypatch.setattr(routes, "classify_file", boom)
    with open(sint / "especie_a.mp4", "rb") as f:
        r = client.post("/api/v1/detect", files={"file": ("especie_a.mp4", f)}, data={"file_name": "especie_a.mp4"})
    assert r.status_code == 500 and seen
    assert not seen[0].exists() and not seen[0].parent.exists()


def test_file_name_is_not_used_as_path(setup, monkeypatch):
    """Un file_name con "..\\" no escribe fuera de la carpeta temporal de la petición."""
    import api.routes as routes

    client, sint, tmp_dir, tmp_path = setup
    paths = []
    monkeypatch.setattr(routes, "classify_file", lambda path, *a, **k: paths.append(Path(path)) or 1 / 0)
    evil = "..\\..\\fuera.mp4" if os.name == "nt" else "../../fuera.mp4"
    with open(sint / "especie_a.mp4", "rb") as f:
        client.post("/api/v1/detect", files={"file": ("x.mp4", f)}, data={"file_name": evil})
    assert paths and paths[0].parent.parent == tmp_dir and paths[0].name == "upload.mp4"
    assert not (tmp_path / "fuera.mp4").exists()
