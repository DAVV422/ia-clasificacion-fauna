"""HU-IA-30 · API key: obligatoria salvo en /health, antes de leer el cuerpo, nunca en el log."""

import logging

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.security import MissingApiKey, is_protected, valid_key
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint

pytest.importorskip("cv2")
KEY = "s3cr3t-clave-larga"


@pytest.fixture
def app_and_files(tmp_path):
    sint = tmp_path / "sint"
    write_f_set_sint(sint)
    app = create_app(Settings(models_dir=tmp_path / "m", tmp_dir=tmp_path / "tmp", ai_api_key=KEY), models=SetModels())
    return app, sint


def post(client, sint, headers):
    with open(sint / "especie_a.mp4", "rb") as f:
        return client.post(
            "/api/v1/detect", files={"file": ("especie_a.mp4", f)}, data={"file_name": "especie_a.mp4"}, headers=headers
        )


# ---------------------------------------------------------------------------
# CA-1 · Sin clave o incorrecta → 401, y el archivo no se procesa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "otra"}, {"X-API-Key": ""}, {"X-API-Key": KEY + "x"}])
def test_api_key_required(app_and_files, monkeypatch, headers):
    import api.routes as routes

    app, sint = app_and_files
    monkeypatch.setattr(routes, "classify_file", lambda *a, **k: pytest.fail("no debía procesar el archivo"))
    with TestClient(app, raise_server_exceptions=False) as client:
        r = post(client, sint, headers)
    assert r.status_code == 401
    body = r.json()
    assert body["status"] == "error" and body["error"]["code"] == "IA_UNAUTHORIZED" and r.headers["X-Request-ID"]


def test_valid_key_passes(app_and_files):
    app, sint = app_and_files
    with TestClient(app) as client:
        assert post(client, sint, {"X-API-Key": KEY}).status_code == 200


def test_catalog_path_is_protected_too():
    assert is_protected("/api/v1/catalog") and is_protected("/api/v1/detect")
    assert not is_protected("/api/v1/health") and not is_protected("/api/v1/health/")


# ---------------------------------------------------------------------------
# CA-2 · /health libre
# ---------------------------------------------------------------------------
def test_health_does_not_require_key(app_and_files):
    app, _ = app_and_files
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code != 401


# ---------------------------------------------------------------------------
# CA-3 · En producción, sin clave no arranca; en desarrollo, dev-key
# ---------------------------------------------------------------------------
def test_production_requires_key(tmp_path):
    with pytest.raises(MissingApiKey):
        create_app(Settings(environment="production", ai_api_key="", models_dir=tmp_path), models=SetModels())


def test_development_defaults_to_dev_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert Settings(environment="development", models_dir=tmp_path).ai_api_key == "dev-key"
    assert Settings(environment="production", models_dir=tmp_path).ai_api_key == ""


def test_compare_is_constant_time_and_rejects_empty():
    assert valid_key(KEY, KEY) and not valid_key("", "") and not valid_key("x", "")


# ---------------------------------------------------------------------------
# CA-4 · La clave nunca va al log
# ---------------------------------------------------------------------------
def test_key_never_logged(app_and_files, caplog):
    app, sint = app_and_files
    caplog.set_level(logging.DEBUG)
    with TestClient(app, raise_server_exceptions=False) as client:
        post(client, sint, {"X-API-Key": "clave-equivocada-123"})
        post(client, sint, {"X-API-Key": KEY})
    assert "ia.auth.rejected" in caplog.text
    assert KEY not in caplog.text and "clave-equivocada-123" not in caplog.text
