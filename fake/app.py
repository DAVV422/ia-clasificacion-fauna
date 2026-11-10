"""ia-fake · IA simulada para desarrollo (HU-IA-22).

Implementa ``POST /api/v1/detect``, ``GET /api/v1/health`` y ``GET /api/v1/catalog``
según ``contratos/API_IA_BACK.md``, con respuestas predefinidas que se eligen por el
nombre del archivo (ver ``fake/README.md``). No usa torch ni OpenCV.

Ejecutar:
    uvicorn fake.app:app --port 8001
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fake import scenarios
from wwm_pipeline.capture_parse import parse_datetime_from_filename
from wwm_pipeline.contract import IA_ERROR_HTTP, DetectResult

CATALOG_FILE = Path(__file__).resolve().parent.parent / "tests/fixtures/catalog/speciesnet_catalog.sample.json"
VERSION = "fake-1.0.0"


@dataclass
class FakeSettings:
    api_key: str = field(default_factory=lambda: os.getenv("AI_API_KEY", "dev-key"))
    delay_seconds: float = field(default_factory=lambda: float(os.getenv("FAKE_DELAY_SECONDS", "2")))
    slow_seconds: float = field(default_factory=lambda: float(os.getenv("FAKE_SLOW_SECONDS", "60")))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("FAKE_TIMEOUT_SECONDS", "200")))
    concurrency: int = field(default_factory=lambda: int(os.getenv("FAKE_CONCURRENCY", "1")))
    queue_size: int = field(default_factory=lambda: int(os.getenv("FAKE_QUEUE_SIZE", "2")))
    not_ready_seconds: float = field(default_factory=lambda: float(os.getenv("FAKE_NOT_READY_SECONDS", "0")))


# ---------------------------------------------------------------------------
# Logs JSON (una línea por evento)
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": record.levelname, "service": "ia-fake",
                   "version": VERSION, "event": getattr(record, "event", record.getMessage()),
                   "message": record.getMessage()}
        payload.update(getattr(record, "ctx", {}))
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("ia-fake")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_JsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False


def _log(event: str, request: Request, **ctx) -> None:
    ctx = {"request_id": request.state.request_id, "flow": request.state.flow, **ctx}
    logger.info(event, extra={"event": event, "ctx": ctx})


# ---------------------------------------------------------------------------
# Envoltura estándar
# ---------------------------------------------------------------------------
def _meta(request: Request) -> dict:
    return {"request_id": request.state.request_id, "timestamp": datetime.now(timezone.utc).isoformat()}


def ok(request: Request, data: dict, message: str = "Archivo clasificado") -> JSONResponse:
    return JSONResponse({"status": "success", "data": data, "message": message, "error": None, "meta": _meta(request)})


def fail(request: Request, code: str, message: str, context: Optional[dict] = None, status: Optional[int] = None) -> JSONResponse:
    details = [{"code": code, "context": context}] if context else []
    body = {"status": "error", "data": None, "message": message,
            "error": {"code": code, "flow": request.state.flow, "details": details}, "meta": _meta(request)}
    return JSONResponse(body, status_code=status or IA_ERROR_HTTP.get(code, 500))


# ---------------------------------------------------------------------------
# Cola simulada: FAKE_CONCURRENCY en proceso + FAKE_QUEUE_SIZE esperando
# ---------------------------------------------------------------------------
class SimulatedQueue:
    def __init__(self, concurrency: int, queue_size: int):
        self._sem = asyncio.Semaphore(concurrency)
        self.capacity = concurrency + queue_size
        self.size = queue_size
        self.in_system = 0

    @property
    def pending(self) -> int:
        return max(0, self.in_system - (self.capacity - self.size))

    def try_enter(self) -> bool:
        if self.in_system >= self.capacity:
            return False
        self.in_system += 1
        return True

    def leave(self) -> None:
        self.in_system -= 1


def create_app(settings: Optional[FakeSettings] = None) -> FastAPI:
    cfg = settings or FakeSettings()
    app = FastAPI(title="WWM ia-fake", version=VERSION, docs_url="/docs")
    started_at = time.monotonic()
    queue = SimulatedQueue(cfg.concurrency, cfg.queue_size)

    def ready() -> bool:
        return time.monotonic() - started_at >= cfg.not_ready_seconds

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.flow = request.headers.get("X-Flow")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(request.headers.get("X-API-Key", ""), cfg.api_key)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        return fail(request, "IA_INVALID_FORMAT", "La petición debe incluir 'file' y 'file_name'.",
                    {"errors": [e["loc"][-1] for e in exc.errors()]}, status=400)

    @app.get("/api/v1/health")
    async def health(request: Request):
        body = {"status": "ok" if ready() else "loading", "models_loaded": ready(),
                "queue": {"size": queue.size, "pending": queue.pending},
                "config": {"sampling": "fake", "max_video_seconds": 15, "max_file_mb": 300},
                "version": {"pipeline": VERSION, "detector": "fake", "classifier": "fake"}}
        return JSONResponse(body, status_code=200 if ready() else 503)

    @app.get("/api/v1/catalog")
    async def catalog(request: Request):
        if not authorized(request):
            return fail(request, "IA_UNAUTHORIZED", "API key ausente o incorrecta.")
        return ok(request, json.loads(CATALOG_FILE.read_text(encoding="utf-8")), message="Catálogo")

    @app.post("/api/v1/detect")
    async def detect(request: Request, file: UploadFile = File(...), file_name: str = Form(...),
                     client_mtime: Optional[str] = Form(None)):
        if not authorized(request):
            _log("fake.detect.rejected", request, code="IA_UNAUTHORIZED")
            return fail(request, "IA_UNAUTHORIZED", "API key ausente o incorrecta.")
        if not ready():
            return fail(request, "IA_MODEL_NOT_READY", "Los modelos se están cargando.")
        content = await file.read()
        if not queue.try_enter():
            _log("fake.detect.rejected", request, code="IA_BUSY", file_name=file_name)
            return fail(request, "IA_BUSY", "La IA está ocupada; reintenta.", {"queue_size": queue.size})
        t0 = time.monotonic()
        try:
            async with queue._sem:
                return await _serve(request, content, file_name, t0)
        finally:
            queue.leave()

    async def _serve(request: Request, content: bytes, file_name: str, t0: float) -> JSONResponse:
        sc = scenarios.pick(file_name)
        wait = {"normal": cfg.delay_seconds, "slow": cfg.slow_seconds, "timeout": cfg.timeout_seconds}[sc.wait]
        await asyncio.sleep(wait)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if sc.error_code:
            context = dict(sc.error_context or {})
            if sc.error_code == "IA_PROCESSING_TIMEOUT":
                context["elapsed_ms"] = elapsed_ms
            _log("fake.detect.rejected", request, code=sc.error_code, scenario=sc.name, file_name=file_name)
            return fail(request, sc.error_code, sc.error_message, context or None)

        data = scenarios.fixture_data(sc.fixture)  # type: ignore[arg-type]
        data["media"]["file_name"] = file_name
        data["media"]["sha256"] = hashlib.sha256(content).hexdigest()
        data["media"]["size_bytes"] = len(content)
        captured = parse_datetime_from_filename(file_name)
        data["capture"] = ({"captured_at": captured.isoformat(), "source": "filename"} if captured
                           else {"captured_at": None, "source": "none"})
        data["analysis"]["processing_ms"] = elapsed_ms
        data["model"] = {**data["model"], "pipeline_version": VERSION}
        if sc.bad_contract:
            data.pop("result")
        else:
            DetectResult.model_validate(data)  # ia-fake nunca responde fuera de contrato (salvo _badcontract)
        _log("fake.detect.served", request, scenario=sc.name, file_name=file_name, delay_ms=elapsed_ms)
        return ok(request, data)

    return app


app = create_app()
