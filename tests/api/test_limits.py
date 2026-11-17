"""HU-IA-32 · Límites y validación de la entrada en POST /detect: tamaño, formato, firma y píxeles."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.routes as routes
from api.main import create_app
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def sint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sint")
    write_f_set_sint(folder)
    return folder


def client_for(tmp_path, **overrides):
    settings = Settings(models_dir=tmp_path / "m", tmp_dir=tmp_path / "tmp", ai_api_key="k", **overrides)
    return TestClient(
        create_app(settings, models=SetModels()), raise_server_exceptions=False, headers={"X-API-Key": "k"}
    )


def post(client, name, content):
    return client.post("/api/v1/detect", files={"file": (name, content)}, data={"file_name": name})


def error(r):
    body = r.json()
    return body["error"]["code"], (body["error"]["details"][0]["context"] if body["error"]["details"] else {})


# ---------------------------------------------------------------------------
# CA-1 · Muy grande
# ---------------------------------------------------------------------------
def test_too_large_rejected_by_content_length(tmp_path, monkeypatch):
    """Anunciado más grande que el límite: 413 en el middleware, sin pasar por el endpoint."""
    monkeypatch.setattr(routes, "_save_upload", lambda *a, **k: pytest.fail("no debía recibir el archivo"))
    with client_for(tmp_path, max_file_mb=1) as client:
        r = post(client, "grande.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (3 * 1024 * 1024))
    code, ctx = error(r)
    assert r.status_code == 413 and code == "IA_FILE_TOO_LARGE" and ctx["max_mb"] == 1 and ctx["size_mb"] >= 3


def test_too_large_cut_while_receiving(tmp_path, sint):
    """Dentro del margen del multipart pero más grande que el límite: se corta al recibirlo y no queda el temporal."""
    data = (sint / "especie_a.mp4").read_bytes()
    with client_for(tmp_path, max_file_mb=len(data) / 2 / (1024 * 1024)) as client:
        r = post(client, "especie_a.mp4", data)
    assert r.status_code == 413 and error(r)[0] == "IA_FILE_TOO_LARGE"
    assert not list((tmp_path / "tmp").glob("ia_*"))


# ---------------------------------------------------------------------------
# CA-2 · CA-3 · Formato: extensión, firma, HEIC, vacío
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,content",
    [
        ("notas.txt", b"hola"),
        ("video.mp4", b"esto no es un video, es texto" * 10),  # disfrazado
        ("IMG_0001.heic", b"\x00\x00\x00\x18ftypheic" + b"\x00" * 100),
        ("vacio.jpg", b""),
    ],
)
def test_invalid_format(tmp_path, name, content):
    with client_for(tmp_path) as client:
        r = post(client, name, content)
    code, ctx = error(r)
    assert r.status_code == 415 and code == "IA_INVALID_FORMAT"
    assert ".mp4" in ctx["accepted"] and ".jpg" in ctx["accepted"] and ctx["reason"]


# ---------------------------------------------------------------------------
# CA-4 · Los formatos que se leen pasan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["especie_a.mp4", "especie_b.jpg", "danado_j.mp4"])
def test_supported_formats_pass_validation(tmp_path, sint, name):
    """danado_j.mp4 tiene firma de MP4 pero está roto: pasa la validación y lo rechaza el decodificador (422)."""
    with client_for(tmp_path) as client:
        r = post(client, name, (sint / name).read_bytes())
    assert r.status_code == (422 if name.startswith("danado") else 200)


# ---------------------------------------------------------------------------
# CA-5 · Demasiados píxeles
# ---------------------------------------------------------------------------
def test_too_many_pixels(tmp_path, sint):
    ok, png = cv2.imencode(".png", np.full((200, 300, 3), 200, np.uint8))
    with client_for(tmp_path, max_image_pixels=10_000) as client:
        r = post(client, "grande.png", png.tobytes())
        video = post(client, "especie_a.mp4", (sint / "especie_a.mp4").read_bytes())  # el límite es solo de imágenes
    code, ctx = error(r)
    assert r.status_code == 422 and code == "IA_DECODE_FAILED" and ctx["reason"] == "too_many_pixels"
    assert video.status_code == 200
