"""HU-IA-18 · GET /api/v1/health (IA §3): lista, cargando, sin modelos; sin API key; misma forma que ia-fake."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from tests.fakes import FakeModels


def make_client(tmp_path, monkeypatch, status):
    import api.main as main

    if status == "ready":
        app = create_app(Settings(models_dir=tmp_path / "m", ai_api_key="k"), models=FakeModels())
    else:
        app = create_app(Settings(models_dir=tmp_path / "vacio", ai_api_key="k"))
        if status == "loading":
            monkeypatch.setattr(main, "_load_models", lambda app: None)  # se queda cargando
    return app, TestClient(app)


@pytest.mark.parametrize(
    "status,http,label", [("ready", 200, "ok"), ("loading", 503, "loading"), ("missing", 503, "models_missing")]
)
def test_health_states(tmp_path, monkeypatch, status, http, label):
    app, client = make_client(tmp_path, monkeypatch, status)
    with client:
        if status == "missing":
            import time

            deadline = time.monotonic() + 5
            while app.state.models_status == "loading" and time.monotonic() < deadline:
                time.sleep(0.05)
        r = client.get("/api/v1/health")  # CA-4: sin X-API-Key
    assert r.status_code == http
    body = r.json()
    assert body["status"] == label and body["models_loaded"] is (status == "ready")
    assert body["queue"] == {"size": 2, "pending": 0}
    assert body["config"]["max_video_seconds"] == 15 and body["config"]["max_file_mb"] == 300
    assert body["config"]["sampling"].startswith("10fps@0-2s+4fps")


def test_versions_match_detect_model(tmp_path, monkeypatch):
    """CA-1: version.detector/classifier son los mismos que model.* de /detect."""
    from tests.desktop.fixtures import SetModels, write_f_set_sint

    pytest.importorskip("cv2")
    sint = tmp_path / "sint"
    write_f_set_sint(sint)
    app = create_app(Settings(models_dir=tmp_path / "m", ai_api_key="k"), models=SetModels())
    with TestClient(app, headers={"X-API-Key": "k"}) as client:
        health = client.get("/api/v1/health").json()
        with open(sint / "especie_a.mp4", "rb") as f:
            model = client.post("/api/v1/detect", files={"file": ("a.mp4", f)}, data={"file_name": "a.mp4"}).json()[
                "data"
            ]["model"]
    assert health["version"]["detector"] == model["detector"] and health["version"]["classifier"] == model["classifier"]
    assert health["config"]["sampling"] == model["sampling"]


def test_same_shape_as_ia_fake(tmp_path, monkeypatch):
    """Las mismas claves que ia-fake (la paridad completa es HU-IA-33)."""
    from fake.app import create_app as create_fake

    _, client = make_client(tmp_path, monkeypatch, "ready")
    with client:
        real = client.get("/api/v1/health").json()
    fake = TestClient(create_fake()).get("/api/v1/health").json()
    assert set(real) == set(fake)
    for key in ("queue", "config"):
        assert set(real[key]) == set(fake[key])
    assert set(fake["version"]) <= set(real["version"])  # la real agrega "app"
