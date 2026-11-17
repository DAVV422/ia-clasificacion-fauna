"""HU-IA-41 · POST /api/v1/detect según el contrato (TestClient + FakeModels, sin PyTorch)."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint
from wwm_pipeline.contract import ApiEnvelope, DetectResult
from wwm_pipeline.errors import PipelineError

pytest.importorskip("cv2")
KEY = "clave-de-prueba"
AUTH = {"X-API-Key": KEY}


@pytest.fixture(scope="module")
def sint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sint")
    write_f_set_sint(folder)
    return folder


@pytest.fixture
def client(tmp_path):
    settings = Settings(models_dir=tmp_path / "models", tmp_dir=tmp_path / "tmp", ai_api_key=KEY)
    with TestClient(create_app(settings, models=SetModels()), raise_server_exceptions=False, headers=AUTH) as c:
        yield c


def post(client, path, name=None, headers=None):
    name = name or path.name
    with open(path, "rb") as f:
        return client.post("/api/v1/detect", files={"file": (name, f)}, data={"file_name": name}, headers=headers or {})


# ---------------------------------------------------------------------------
# CA-1 · CA-2 · CA-4 · Éxito: la envoltura con un DetectResult válido
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,category,taxon",
    [
        ("especie_a.mp4", "animal", "Panthera onca"),
        ("especie_b.jpg", "animal", "Panthera onca"),  # IA-03: la imagen con animal ya no da 500
        ("genero_c.mp4", "animal", "Dasyprocta"),
        ("vacio_g.jpg", "blank", None),
        ("persona_h.mp4", "person", None),
    ],
)
def test_detect_success_envelope(client, sint, name, category, taxon):
    r = post(client, sint / name)
    assert r.status_code == 200, r.text
    body = r.json()
    env = ApiEnvelope[DetectResult].model_validate(body)
    assert env.status == "success" and env.error is None
    data = env.data
    assert data.result.category.value == category
    assert (data.result.taxonomy.scientific_name if data.result.taxonomy else None) == taxon
    assert (data.evidence is None) == (category == "blank")
    assert data.media.file_name == name


def test_file_name_comes_from_form(client, sint):
    """El nombre original lo manda el back en file_name (IA §2.1), aunque el archivo subido se llame distinto."""
    with open(sint / "especie_a.mp4", "rb") as f:
        r = client.post(
            "/api/v1/detect", files={"file": ("tmp_upload_123.mp4", f)}, data={"file_name": "CAM01_20260315_042210.mp4"}
        )
    data = r.json()["data"]
    assert data["media"]["file_name"] == "CAM01_20260315_042210.mp4"
    assert data["capture"]["source"] in ("filename", "container")


# ---------------------------------------------------------------------------
# CA-5 · Errores con su código y HTTP, sin stack en la respuesta
# ---------------------------------------------------------------------------
def test_decode_failed(client, sint):
    r = post(client, sint / "danado_j.mp4")
    assert r.status_code == 422 and r.json()["error"]["code"] == "IA_DECODE_FAILED"
    assert r.json()["status"] == "error" and r.json()["data"] is None


def test_missing_file_name_is_400(client, sint):
    with open(sint / "especie_a.mp4", "rb") as f:
        r = client.post("/api/v1/detect", files={"file": ("a.mp4", f)})
    assert r.status_code == 400
    assert r.json()["error"]["details"][0]["context"]["errors"] == ["file_name"]


def test_unexpected_error_is_internal_without_stack(client, sint, monkeypatch, caplog):
    import api.routes as routes

    def boom(*a, **k):
        raise RuntimeError("detalle secreto del servidor")

    monkeypatch.setattr(routes, "classify_file", boom)
    r = post(client, sint / "especie_a.mp4")
    assert r.status_code == 500 and r.json()["error"]["code"] == "IA_INTERNAL"
    assert "detalle secreto" not in r.text and "Traceback" not in r.text
    assert "detalle secreto" in caplog.text  # el detalle va al log


def test_pipeline_error_maps_to_http(client, sint, monkeypatch):
    import api.routes as routes

    def too_long(*a, **k):
        raise PipelineError("IA_INVALID_FORMAT", "firma no reconocida", accepted=["mp4"])

    monkeypatch.setattr(routes, "classify_file", too_long)
    r = post(client, sint / "especie_a.mp4")
    assert r.status_code == 415 and r.json()["error"]["details"][0]["context"] == {"accepted": ["mp4"]}


def test_temp_file_removed_even_on_error(client, sint, tmp_path):
    post(client, sint / "danado_j.mp4")
    post(client, sint / "especie_a.mp4")
    assert not list((tmp_path / "tmp").glob("ia_*"))


# ---------------------------------------------------------------------------
# CA-6 · Correlación
# ---------------------------------------------------------------------------
def test_correlation_headers(client, sint):
    r = post(client, sint / "danado_j.mp4", headers={"X-Request-ID": "abc123", "X-Flow": "FL-08"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["meta"]["request_id"] == "abc123" and r.json()["error"]["flow"] == "FL-08"
    r = post(client, sint / "especie_a.mp4")
    assert len(r.headers["X-Request-ID"]) >= 8 and r.json()["meta"]["request_id"] == r.headers["X-Request-ID"]


# ---------------------------------------------------------------------------
# CA-7 · Arranque: 503 mientras cargan; nunca descarga nada
# ---------------------------------------------------------------------------
def test_not_ready_while_loading(tmp_path, sint, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_load_models", lambda app: None)  # se queda "cargando"
    settings = Settings(models_dir=tmp_path / "models", tmp_dir=tmp_path / "tmp", ai_api_key=KEY)
    with TestClient(create_app(settings), raise_server_exceptions=False, headers=AUTH) as c:
        r = post(c, sint / "especie_a.mp4")
    assert r.status_code == 503 and r.json()["error"]["code"] == "IA_MODEL_NOT_READY"


def test_missing_models_never_download(tmp_path, monkeypatch):
    import api.main as main

    settings = Settings(models_dir=tmp_path / "vacio", tmp_dir=tmp_path / "tmp")
    app = create_app(settings)
    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("no debía conectarse a nada"))
    main._load_models(app)
    assert app.state.models_status == "missing"


def test_no_cors_and_no_static_mount(client):
    r = client.options("/api/v1/detect", headers={"Origin": "http://x", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert client.get("/detections/algo.jpg").status_code == 404
