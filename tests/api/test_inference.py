"""HU-IA-04 · Un archivo a la vez, cola corta (IA_BUSY) y timeout que cancela de verdad (IA §1)."""

import asyncio
import time

import httpx
import pytest

import api.routes as routes
from api.main import create_app
from api.settings import Settings
from tests.desktop.fixtures import SetModels, write_f_set_sint
from wwm_pipeline.pipeline import classify_file as real_classify

pytest.importorskip("cv2")
KEY = {"X-API-Key": "k"}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sint")
    write_f_set_sint(folder)
    return (folder / "especie_a.mp4").read_bytes()


def make_app(tmp_path, queue_size=2, timeout=150.0):
    settings = Settings(
        models_dir=tmp_path / "m",
        tmp_dir=tmp_path / "tmp",
        ai_api_key="k",
        inference_queue_size=queue_size,
        process_timeout_seconds=timeout,
    )
    return create_app(settings, models=SetModels())


def slow_classify(seconds, calls):
    """Como classify_file, pero tarda ``seconds`` (consultando should_stop, como el pipeline entre frames)."""

    def classify(path, name, config, models, should_stop=None, **kwargs):
        calls.append((name, time.monotonic()))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if should_stop and should_stop():
                calls.append(("stopped", time.monotonic()))
                from wwm_pipeline.errors import PipelineCancelled

                raise PipelineCancelled(name)
            time.sleep(0.01)
        return real_classify(path, name, config, models, should_stop=should_stop)

    return classify


async def post(client, video, name="especie_a.mp4"):
    return await client.post("/api/v1/detect", files={"file": (name, video)}, data={"file_name": name}, headers=KEY)


def run(app, coro_fn):
    async def main():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://ia") as client:
            return await coro_fn(client)

    return asyncio.run(main())


# ---------------------------------------------------------------------------
# CA-1 · Uno a la vez: lo que no entra recibe IA_BUSY
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("queue_size,concurrent", [(1, 3), (2, 4)])
def test_extra_concurrent_request_is_busy(tmp_path, monkeypatch, video, queue_size, concurrent):
    calls = []
    monkeypatch.setattr(routes, "classify_file", slow_classify(0.4, calls))
    app = make_app(tmp_path, queue_size=queue_size)

    async def scenario(client):
        return await asyncio.gather(*(post(client, video, f"v{i}.mp4") for i in range(concurrent)))

    responses = run(app, scenario)
    codes = sorted(r.status_code for r in responses)
    assert codes == [200] * (1 + queue_size) + [503]
    busy = next(r for r in responses if r.status_code == 503).json()
    assert busy["error"]["code"] == "IA_BUSY" and busy["error"]["details"][0]["context"] == {"queue_size": queue_size}
    starts = [t for name, t in calls if name != "stopped"]
    assert all(
        b - a >= 0.35 for a, b in zip(starts, starts[1:])
    )  # de a uno: cada uno empieza cuando termina el anterior
    assert app.state.worker.in_system == 0


# ---------------------------------------------------------------------------
# CA-2 · El servidor sigue vivo mientras se procesa
# ---------------------------------------------------------------------------
def test_health_responds_while_processing(tmp_path, monkeypatch, video):
    monkeypatch.setattr(routes, "classify_file", slow_classify(1.0, []))
    app = make_app(tmp_path)

    async def scenario(client):
        task = asyncio.create_task(post(client, video))
        await asyncio.sleep(0.3)  # ya está en el hilo de inferencia
        t0 = time.monotonic()
        health = await client.get("/api/v1/health")
        elapsed = time.monotonic() - t0
        pending_during = app.state.worker.in_system
        await task
        return health, elapsed, pending_during

    health, elapsed, in_system = run(app, scenario)
    assert health.status_code == 200 and elapsed < 0.2 and in_system == 1


# ---------------------------------------------------------------------------
# CA-3 · Timeout: 504, el trabajo se cancela y el hilo queda libre
# ---------------------------------------------------------------------------
def test_timeout_cancels_and_frees_worker(tmp_path, monkeypatch, video):
    calls = []
    app = make_app(tmp_path, timeout=0.5)

    async def scenario(client):
        monkeypatch.setattr(routes, "classify_file", slow_classify(5.0, calls))
        first = await post(client, video, "lento.mp4")
        await asyncio.sleep(0.2)  # el hilo corta en el próximo should_stop (≤ 10 ms aquí)
        monkeypatch.setattr(routes, "classify_file", real_classify)
        second = await post(client, video, "rapido.mp4")
        return first, second

    t0 = time.monotonic()
    first, second = run(app, scenario)
    assert first.status_code == 504
    body = first.json()
    assert (
        body["error"]["code"] == "IA_PROCESSING_TIMEOUT" and body["error"]["details"][0]["context"]["elapsed_ms"] >= 500
    )
    assert "stopped" in [name for name, _ in calls]  # se canceló de verdad, no siguió 5 s
    assert second.status_code == 200 and time.monotonic() - t0 < 4
    assert not list((tmp_path / "tmp").glob("ia_*"))  # HU-IA-05 CA-3: el timeout tampoco deja el temporal


def test_busy_rejected_before_reading_body(tmp_path, monkeypatch, video):
    """Con la cola llena, el middleware responde IA_BUSY sin pasar por el endpoint (no recibe el archivo)."""
    app = make_app(tmp_path, queue_size=0)
    app.state.worker.try_enter()  # simula uno en proceso: capacidad 1, llena
    monkeypatch.setattr(routes, "_save_upload", lambda *a, **k: pytest.fail("no debía recibir el archivo"))

    async def scenario(client):
        return await post(client, video)

    r = run(app, scenario)
    assert r.status_code == 503 and r.json()["error"]["code"] == "IA_BUSY"


def test_health_reports_pending(tmp_path):
    app = make_app(tmp_path, queue_size=2)
    worker = app.state.worker
    assert worker.try_enter() and worker.try_enter() and worker.try_enter() and not worker.try_enter()
    assert worker.pending == 2

    async def scenario(client):
        return await client.get("/api/v1/health")

    assert run(app, scenario).json()["queue"] == {"size": 2, "pending": 2}
