"""HU-IA-22 · ia-fake — un test por criterio de aceptación."""

import asyncio
import hashlib
import json
import logging
import time

import httpx
import pytest

from fake.app import FakeSettings, create_app
from wwm_pipeline.contract import DetectEnvelope, DetectResult

KEY = "dev-key"
HEADERS = {"X-API-Key": KEY}


def make_client(**overrides) -> httpx.AsyncClient:
    settings = FakeSettings(api_key=KEY, delay_seconds=0, slow_seconds=0.3, timeout_seconds=0.3,
                            concurrency=1, queue_size=2, not_ready_seconds=0)
    for k, v in overrides.items():
        setattr(settings, k, v)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://ia")


async def detect(client, name: str, content: bytes = b"contenido de prueba", headers=HEADERS):
    return await client.post("/api/v1/detect", headers=headers,
                             files={"file": (name, content, "application/octet-stream")}, data={"file_name": name})


@pytest.mark.anyio
async def test_default_returns_valid_jaguar_with_real_sha():
    """CA-1"""
    content = b"\x00\x01video falso" * 100
    async with make_client() as client:
        r = await detect(client, "CAM01_20260315_042210.mp4", content)
    assert r.status_code == 200
    env = DetectEnvelope.model_validate(r.json())
    data = env.data
    assert data.result.taxonomy.scientific_name == "Panthera onca"
    assert data.media.sha256 == hashlib.sha256(content).hexdigest()
    assert data.media.size_bytes == len(content)
    assert data.media.file_name == "CAM01_20260315_042210.mp4"
    assert data.capture.source.value == "filename"
    assert data.capture.captured_at.isoformat() == "2026-03-15T04:22:10"


@pytest.mark.anyio
async def test_no_date_in_name_gives_source_none():
    async with make_client() as client:
        r = await detect(client, "video_sin_fecha.mp4")
    assert r.json()["data"]["capture"] == {"captured_at": None, "source": "none"}


SUCCESS_CASES = {
    "x_blank.mp4": ("blank", None),
    "x_person.mp4": ("person", None),
    "x_vehicle.jpg": ("vehicle", None),
    "x_genus.mp4": ("animal", "genus"),
    "foto_image.mp4": ("animal", "species"),
    "IMG_0001.JPG": ("animal", "species"),
    "x_slow.mp4": ("animal", "species"),
}
ERROR_CASES = {
    "x_error_decode.mp4": (422, "IA_DECODE_FAILED"),
    "x_toolong.mp4": (422, "IA_DURATION_EXCEEDED"),
    "x_badformat.cr2": (415, "IA_INVALID_FORMAT"),
    "x_busy.mp4": (503, "IA_BUSY"),
    "x_notready.mp4": (503, "IA_MODEL_NOT_READY"),
    "x_timeout.mp4": (504, "IA_PROCESSING_TIMEOUT"),
    "x_crash.mp4": (500, "IA_INTERNAL"),
}


@pytest.mark.anyio
@pytest.mark.parametrize("name,expected", SUCCESS_CASES.items())
async def test_success_scenarios(name, expected):
    """CA-2 (casos de éxito)"""
    category, rank = expected
    async with make_client() as client:
        r = await detect(client, name)
    assert r.status_code == 200
    data = DetectResult.model_validate(r.json()["data"])
    assert data.result.category.value == category
    if rank:
        assert data.result.taxonomy.rank.value == rank
    if name.lower().endswith(".jpg") or "_image" in name:
        assert data.media.media_type.value == "image"


@pytest.mark.anyio
@pytest.mark.parametrize("name,expected", ERROR_CASES.items())
async def test_error_scenarios(name, expected):
    """CA-2 (casos de error)"""
    status, code = expected
    async with make_client() as client:
        r = await detect(client, name)
    assert r.status_code == status
    env = DetectEnvelope.model_validate(r.json())
    assert env.error.code == code and env.data is None


@pytest.mark.anyio
async def test_duration_error_has_context():
    async with make_client() as client:
        r = await detect(client, "x_toolong.mp4")
    assert r.json()["error"]["details"][0]["context"] == {"duration_seconds": 16.2, "max_seconds": 15}


@pytest.mark.anyio
async def test_bad_contract_scenario_breaks_the_schema():
    """CA-2: _badcontract sirve para probar AI_BAD_RESPONSE en el back."""
    async with make_client() as client:
        r = await detect(client, "x_badcontract.mp4")
    assert r.status_code == 200 and "result" not in r.json()["data"]
    with pytest.raises(ValueError):
        DetectResult.model_validate(r.json()["data"])


@pytest.mark.anyio
async def test_requires_api_key():
    """CA-3"""
    async with make_client() as client:
        missing = await detect(client, "a.mp4", headers={})
        wrong = await detect(client, "a.mp4", headers={"X-API-Key": "otra"})
        catalog = await client.get("/api/v1/catalog")
    for r in (missing, wrong, catalog):
        assert r.status_code == 401 and r.json()["error"]["code"] == "IA_UNAUTHORIZED"


@pytest.mark.anyio
async def test_health_without_key():
    """CA-3, CA-5"""
    async with make_client() as client:
        r = await client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models_loaded"] is True and body["queue"] == {"size": 2, "pending": 0}


@pytest.mark.anyio
async def test_delay():
    """CA-4"""
    async with make_client(delay_seconds=0.5) as client:
        t0 = time.monotonic()
        r = await detect(client, "a.mp4")
        elapsed = time.monotonic() - t0
    assert r.status_code == 200 and 0.4 <= elapsed <= 1.2


@pytest.mark.anyio
async def test_busy_when_queue_full():
    """CA-4: concurrencia 1 + cola 2 → la cuarta petición simultánea recibe IA_BUSY."""
    async with make_client(delay_seconds=0.4) as client:
        responses = await asyncio.gather(*(detect(client, f"v{i}.mp4") for i in range(4)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 200, 200, 503]
    busy = next(r for r in responses if r.status_code == 503)
    assert busy.json()["error"]["code"] == "IA_BUSY"


@pytest.mark.anyio
async def test_not_ready_window():
    """CA-5"""
    async with make_client(not_ready_seconds=60) as client:
        health = await client.get("/api/v1/health")
        r = await detect(client, "a.mp4")
    assert health.status_code == 503 and health.json()["models_loaded"] is False
    assert r.status_code == 503 and r.json()["error"]["code"] == "IA_MODEL_NOT_READY"


@pytest.mark.anyio
async def test_catalog():
    """CA-5"""
    async with make_client() as client:
        r = await client.get("/api/v1/catalog", headers=HEADERS)
    data = r.json()["data"]
    assert r.status_code == 200 and data["count"] == len(data["taxa"]) >= 200


@pytest.mark.anyio
async def test_request_id_echo_and_logged(caplog):
    """CA-6"""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("ia-fake")
    logger.addHandler(handler)
    try:
        async with make_client() as client:
            r = await client.post("/api/v1/detect", headers={**HEADERS, "X-Request-ID": "abc", "X-Flow": "FL-08"},
                                  files={"file": ("a.mp4", b"x")}, data={"file_name": "a.mp4"})
    finally:
        logger.removeHandler(handler)
    assert r.headers["X-Request-ID"] == "abc"
    assert r.json()["meta"]["request_id"] == "abc"
    served = [rec for rec in records if getattr(rec, "event", "") == "fake.detect.served"]
    assert served and served[0].ctx["request_id"] == "abc" and served[0].ctx["flow"] == "FL-08"


@pytest.mark.anyio
async def test_missing_file_is_bad_request():
    async with make_client() as client:
        r = await client.post("/api/v1/detect", headers=HEADERS, data={"file_name": "a.mp4"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "IA_INVALID_FORMAT"


@pytest.fixture
def anyio_backend():
    return "asyncio"
