"""``POST /api/v1/detect`` (IA §2) — HU-IA-41.

1. Guarda el archivo en una carpeta temporal **propia de la petición** (con su extensión: OpenCV la usa para elegir
   el decodificador).
2. Lo clasifica con ``PipelineConfig.web()`` en el hilo de inferencia (HU-IA-04: uno a la vez, cola corta, corte
   por tiempo; ``api/inference.py``).
3. Arma ``DetectResult`` (``wwm_pipeline.response``), validado contra el contrato, y responde la envoltura.
4. Borra esa carpeta entera, también si hubo error (HU-IA-05: la IA no guarda nada, IA §1). Si el archivo llegó al
   hilo, la borra el hilo al terminar (con un *timeout*, todavía puede estar leyéndolo); si no, la petición.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from wwm_pipeline import formats
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.errors import PipelineError
from wwm_pipeline.pipeline import classify_file
from wwm_pipeline.response import to_detect_result

from .envelope import fail, ok
from .errors import MESSAGES
from .logs import log_event

logger = logging.getLogger("ia")
router = APIRouter(prefix="/api/v1")
CHUNK = 1024 * 1024


def _request_dir(tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ia_", dir=str(tmp_dir)))


MB = 1024 * 1024


async def _save_upload(file: UploadFile, file_name: str, folder: Path, max_bytes: int) -> Path:
    """Guarda el archivo comprobando, con el primer bloque, la extensión y la firma (HU-IA-32), y cortando al pasar
    ``MAX_FILE_MB``. El nombre del usuario no se usa como ruta (podría traer "..\"): solo su extensión."""
    path = folder / f"upload{Path(file_name).suffix.lower()}"
    total = 0
    with open(path, "wb") as out:
        while chunk := await file.read(CHUNK):
            if total == 0:
                ok, reason = formats.check(file_name, chunk[: formats.HEAD_BYTES])
                if not ok:
                    raise PipelineError(
                        "IA_INVALID_FORMAT", reason, accepted=formats.ACCEPTED_EXTENSIONS, reason=reason
                    )
            total += len(chunk)
            if total > max_bytes:
                raise PipelineError(
                    "IA_FILE_TOO_LARGE",
                    "supera MAX_FILE_MB",
                    max_mb=round(max_bytes / MB, 1),
                    size_mb=round(total / MB, 1),
                )
            out.write(chunk)
    if total == 0:
        raise PipelineError("IA_INVALID_FORMAT", "archivo vacío", accepted=formats.ACCEPTED_EXTENSIONS, reason="vacío")
    return path


def _check_pixels(path: Path, max_pixels: int) -> None:
    """Una imagen enorme agota la memoria al decodificarla ("bomba de descompresión"): se mira solo la cabecera."""
    if path.suffix.lower() not in formats.IMAGE_EXTENSIONS:
        return
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - si Pillow no la abre, que decida el pipeline (IA_DECODE_FAILED)
        return
    if width * height > max_pixels:
        raise PipelineError("IA_DECODE_FAILED", "demasiados píxeles", reason="too_many_pixels", pixels=width * height)


# ---------------------------------------------------------------------------
# GET /health (HU-IA-18, IA §3): sin API key y sin la envoltura; 503 mientras no esté lista
# ---------------------------------------------------------------------------
HEALTH_STATUS = {"ready": "ok", "loading": "loading", "missing": "models_missing", "error": "error"}


@router.get("/health")
async def health(request: Request):
    from wwm_pipeline import __version__ as pipeline_version

    state = request.app.state
    settings = state.settings
    status = state.models_status
    body = {
        "status": HEALTH_STATUS.get(status, status),
        "models_loaded": status == "ready",
        "queue": {"size": state.worker.queue_size, "pending": state.worker.pending},
        "config": {
            "sampling": PipelineConfig.web().sampling_label,
            "max_video_seconds": settings.max_video_seconds,
            "max_file_mb": settings.max_file_mb,
        },
        "version": {
            "pipeline": pipeline_version,
            "detector": state.model_names.get("detector"),
            "classifier": state.model_names.get("classifier"),
            "app": settings.app_version,
        },
    }
    return JSONResponse(body, status_code=200 if status == "ready" else 503)


# ---------------------------------------------------------------------------
# GET /catalog (HU-IA-19, IA §4): con API key (el middleware); ETag = sha256 del catálogo
# ---------------------------------------------------------------------------
def load_catalog(path: Path) -> Optional[dict]:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log_event("ia.catalog.missing", "No está el catálogo", logging.ERROR, exc_info=True, path=str(path))
        return None


@router.get("/catalog")
async def catalog(request: Request):
    state = request.app.state
    if state.catalog is None:
        state.catalog = load_catalog(state.settings.catalog_path)
        if state.catalog is not None and state.catalog.get("classifier") != state.model_names.get("classifier"):
            log_event(
                "ia.catalog.mismatch",
                "El catálogo es de otro clasificador",
                logging.WARNING,
                catalog=state.catalog.get("classifier"),
                model=state.model_names.get("classifier"),
            )
    if state.catalog is None:
        return fail(request, "IA_INTERNAL", "No está el catálogo de especies.")
    etag = f'"{state.catalog["sha256"]}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response = ok(request, state.catalog, message="Catálogo")
    response.headers["ETag"] = etag
    return response


class StageTimer:
    """``on_progress`` del pipeline → un evento ``pipeline.<etapa>.completed`` por etapa, con su duración."""

    def __init__(self):
        self.stage = None
        self.started = 0.0
        self.total = 0

    def __call__(self, stage: str, done: int, total: int) -> None:
        if stage != self.stage:
            self.finish()
            self.stage, self.started = stage, time.monotonic()
        self.total = total

    def finish(self) -> None:
        if self.stage is not None:
            ms = int((time.monotonic() - self.started) * 1000)
            log_event(f"pipeline.{self.stage}.completed", self.stage, duration_ms=ms, count=self.total)
            self.stage = None


def busy(request: Request, worker):
    log_event("ia.queue.full", "La cola de inferencia está llena", logging.WARNING, in_system=worker.in_system)
    return fail(request, "IA_BUSY", MESSAGES["IA_BUSY"], {"queue_size": worker.queue_size})


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    file_name: str = Form(...),
    # E2-D4: se recibe y no se usa; con capture.source=none, lo aplica el back.
    client_mtime: Optional[str] = Form(None),
):
    state = request.app.state
    if state.models_status != "ready":
        return fail(request, "IA_MODEL_NOT_READY", MESSAGES["IA_MODEL_NOT_READY"])
    worker = state.worker
    if not worker.try_enter():  # el middleware ya lo miró antes de leer el cuerpo; aquí se reserva el lugar
        return busy(request, worker)
    t0 = time.monotonic()
    log_event("ia.detect.received", "Archivo recibido", file_name=file_name)
    folder = _request_dir(state.settings.tmp_dir)
    handed_over = False
    try:
        path = await _save_upload(file, file_name, folder, state.settings.max_file_bytes)
        _check_pixels(path, state.settings.max_image_pixels)
        config = PipelineConfig.web()
        models = state.models

        def job(should_stop):
            stages = StageTimer()
            result = classify_file(path, file_name, config, models, on_progress=stages, should_stop=should_stop)
            stages.finish()
            log_event(
                "pipeline.aggregation.completed",
                "Votación",
                category=result.decision.category,
                votes=len(result.vote.votes) if result.vote else 0,
                frames_sampled=result.frames_sampled,
            )
            return result

        handed_over = True  # desde aquí, el hilo libera el lugar y borra la carpeta
        result = await worker.submit(job, cleanup=lambda: shutil.rmtree(folder, ignore_errors=True))
        data = to_detect_result(
            result, file_name, state.model_names["classifier"], processing_ms=int((time.monotonic() - t0) * 1000)
        )
    finally:
        if not handed_over:
            worker.leave()
            shutil.rmtree(folder, ignore_errors=True)
    log_event(
        "ia.detect.succeeded",
        "Archivo clasificado",
        file_name=file_name,
        category=data.result.category.value,
        rank=data.result.taxonomy.rank.value if data.result.taxonomy else None,
        duration_ms=data.analysis.processing_ms,
    )
    return ok(request, data.model_dump(mode="json", by_alias=True))
