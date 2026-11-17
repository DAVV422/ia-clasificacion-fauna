"""HU-IA-42 · Logs JSON (ESTÁNDARES §7): formato, correlación de punta a punta y lo que nunca se registra."""

import io
import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

import api.routes as routes
from api import logs
from api.main import create_app
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint

pytest.importorskip("cv2")
KEY = "clave-que-no-debe-aparecer-1234"
REQUIRED = {"ts", "level", "service", "env", "version", "request_id", "event", "message"}


@pytest.fixture
def captured():
    """Un handler con el formatter JSON real sobre el logger "ia"; devuelve las líneas parseadas."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logs.JsonFormatter(env="test", version="9.9.9"))
    ia = logging.getLogger("ia")
    ia.addHandler(handler)
    old_level = ia.level
    ia.setLevel(logging.DEBUG)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    ia.removeHandler(handler)
    ia.setLevel(old_level)


@pytest.fixture
def client(tmp_path):
    sint = tmp_path / "sint"
    write_f_set_sint(sint)
    app = create_app(Settings(models_dir=tmp_path / "m", tmp_dir=tmp_path / "t", ai_api_key=KEY), models=SetModels())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, sint


def post(c, sint, name, headers):
    with open(sint / name, "rb") as f:
        return c.post("/api/v1/detect", files={"file": (name, f)}, data={"file_name": name}, headers=headers)


# ---------------------------------------------------------------------------
# CA-1 · CA-2 · Formato y correlación (incluidos los eventos del hilo de inferencia)
# ---------------------------------------------------------------------------
def test_logs_json_with_request_id(client, captured):
    c, sint = client
    r = post(c, sint, "especie_a.mp4", {"X-API-Key": KEY, "X-Request-ID": "abc123", "X-Flow": "FL-08"})
    assert r.status_code == 200
    lines = captured()
    assert lines and all(set(line) >= REQUIRED for line in lines)
    assert all(line["service"] == "ia" and line["env"] == "test" and line["version"] == "9.9.9" for line in lines)
    events = [line["event"] for line in lines]
    for expected in (
        "ia.detect.received",
        "pipeline.sampling.completed",
        "pipeline.detection.completed",
        "pipeline.aggregation.completed",
        "ia.detect.succeeded",
    ):
        assert expected in events, (expected, events)
    assert all(line["request_id"] == "abc123" and line["flow"] == "FL-08" for line in lines)
    done = next(line for line in lines if line["event"] == "ia.detect.succeeded")
    assert done["category"] == "animal" and done["rank"] == "species" and done["duration_ms"] >= 0
    stage = next(line for line in lines if line["event"] == "pipeline.detection.completed")
    assert stage["duration_ms"] >= 0 and stage["count"] > 0


def test_each_request_has_its_own_id(client, captured):
    c, sint = client
    post(c, sint, "especie_b.jpg", {"X-API-Key": KEY, "X-Request-ID": "uno"})
    post(c, sint, "vacio_g.jpg", {"X-API-Key": KEY, "X-Request-ID": "dos"})
    ids = {line["request_id"] for line in captured()}
    assert ids == {"uno", "dos"}


# ---------------------------------------------------------------------------
# CA-3 · Lo que nunca va al log
# ---------------------------------------------------------------------------
def test_no_secrets_or_base64_in_logs(client, captured):
    c, sint = client
    post(c, sint, "especie_a.mp4", {"X-API-Key": KEY})
    post(c, sint, "especie_a.mp4", {"X-API-Key": "otra-clave-incorrecta-987"})
    text = "\n".join(json.dumps(line) for line in captured())
    assert KEY not in text and "otra-clave-incorrecta-987" not in text
    assert not re.search(r"[A-Za-z0-9+/]{200,}", text)  # ningún base64 largo (el fotograma)
    assert "ia.auth.rejected" in text


# ---------------------------------------------------------------------------
# CA-4 · Error inesperado: stack en el log, no en la respuesta
# ---------------------------------------------------------------------------
def test_internal_error_logged_with_stack_not_returned(client, captured, monkeypatch):
    c, sint = client

    def boom(*a, **k):
        raise RuntimeError("detalle interno 42")

    monkeypatch.setattr(routes, "classify_file", boom)
    r = post(c, sint, "especie_a.mp4", {"X-API-Key": KEY, "X-Request-ID": "err1"})
    assert r.status_code == 500 and "Traceback" not in r.text and "detalle interno" not in r.text
    failed = [line for line in captured() if line["event"] == "ia.detect.failed"]
    assert failed and failed[0]["error_code"] == "IA_INTERNAL" and failed[0]["request_id"] == "err1"
    assert "Traceback" in failed[0]["exc_info"] and "detalle interno 42" in failed[0]["exc_info"]


# ---------------------------------------------------------------------------
# configure(): idempotente, uvicorn al mismo formato, YOLOv5 en silencio
# ---------------------------------------------------------------------------
def test_configure_is_idempotent_and_quiets_libraries():
    root = logging.getLogger()
    before = list(root.handlers)
    logs.configure("test", "1.0")
    logs.configure("test", "1.0")
    added = [h for h in root.handlers if h not in before]
    try:
        assert len([h for h in root.handlers if getattr(h, "_wwm_ia_handler", False)]) == 1
        assert logging.getLogger("yolov5").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").handlers == [] and logging.getLogger("uvicorn.access").propagate
    finally:
        for h in added:
            root.removeHandler(h)


def test_console_format(monkeypatch):
    record = logging.LogRecord("ia", logging.INFO, __file__, 1, "Archivo recibido", None, None)
    record.event = "ia.detect.received"
    record.ctx = {"file_name": "a.mp4"}
    line = logs.ConsoleFormatter().format(record)
    assert "ia.detect.received" in line and "file_name=a.mp4" in line
