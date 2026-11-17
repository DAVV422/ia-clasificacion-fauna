"""``POST /api/v1/detect`` (IA §2) — HU-IA-41.

1. Guarda el archivo en una carpeta temporal **propia de la petición** (con su extensión: OpenCV la usa para elegir
   el decodificador).
2. Lo clasifica con ``PipelineConfig.web()`` en un hilo (no bloquea el *event loop*; la cola de verdad es HU-IA-04).
3. Arma ``DetectResult`` (``wwm_pipeline.response``), validado contra el contrato, y responde la envoltura.
4. Borra esa carpeta entera en un ``finally``, también si hubo error (HU-IA-05: la IA no guarda nada, IA §1).
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.pipeline import classify_file
from wwm_pipeline.response import to_detect_result

from .envelope import fail, ok
from .errors import MESSAGES

logger = logging.getLogger("ia")
router = APIRouter(prefix="/api/v1")
CHUNK = 1024 * 1024


def _request_dir(tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ia_", dir=str(tmp_dir)))


async def _save_upload(file: UploadFile, file_name: str, folder: Path) -> Path:
    # El nombre del usuario no se usa como ruta (podría traer "..\"): solo su extensión.
    path = folder / f"upload{Path(file_name).suffix.lower()}"
    with open(path, "wb") as out:
        while chunk := await file.read(CHUNK):
            out.write(chunk)
    return path


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
        "queue": {"size": settings.inference_queue_size, "pending": getattr(state, "queue_pending", 0)},
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
    t0 = time.monotonic()
    folder = _request_dir(state.settings.tmp_dir)
    try:
        path = await _save_upload(file, file_name, folder)
        config = PipelineConfig.web()
        result = await run_in_threadpool(classify_file, path, file_name, config, state.models)
        data = to_detect_result(
            result, file_name, state.model_names["classifier"], processing_ms=int((time.monotonic() - t0) * 1000)
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    logger.info(
        "ia.detect.succeeded file=%s category=%s ms=%d",
        file_name,
        data.result.category.value,
        data.analysis.processing_ms,
    )
    return ok(request, data.model_dump(mode="json", by_alias=True))
