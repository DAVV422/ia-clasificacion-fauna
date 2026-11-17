"""``POST /api/v1/detect`` (IA §2) — HU-IA-41.

1. Guarda el archivo en un temporal (con su extensión: OpenCV la usa para elegir el decodificador).
2. Lo clasifica con ``PipelineConfig.web()`` en un hilo (no bloquea el *event loop*; la cola de verdad es HU-IA-04).
3. Arma ``DetectResult`` (``wwm_pipeline.response``), validado contra el contrato, y responde la envoltura.
4. Borra el temporal en un ``finally``.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.pipeline import classify_file
from wwm_pipeline.response import to_detect_result

from .envelope import fail, ok
from .errors import MESSAGES

logger = logging.getLogger("ia")
router = APIRouter(prefix="/api/v1")
CHUNK = 1024 * 1024


async def _save_upload(file: UploadFile, file_name: str, tmp_dir: Path) -> Path:
    suffix = Path(file_name).suffix.lower()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="ia_", suffix=suffix, dir=str(tmp_dir))
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(CHUNK):
            out.write(chunk)
    return Path(name)


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
    path = await _save_upload(file, file_name, state.settings.tmp_dir)
    try:
        config = PipelineConfig.web()
        result = await run_in_threadpool(classify_file, path, file_name, config, state.models)
        data = to_detect_result(
            result, file_name, state.model_names["classifier"], processing_ms=int((time.monotonic() - t0) * 1000)
        )
    finally:
        path.unlink(missing_ok=True)
    logger.info(
        "ia.detect.succeeded file=%s category=%s ms=%d",
        file_name,
        data.result.category.value,
        data.analysis.processing_ms,
    )
    return ok(request, data.model_dump(mode="json", by_alias=True))
