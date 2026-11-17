"""Errores de la IA → HTTP (IA §2.5). Los mensajes son para el log del back y para depurar, no para el usuario final
(el back los traduce a sus propios códigos). Nunca llevan el *stack*: ese va solo al log."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from wwm_pipeline.errors import PipelineError

from .envelope import fail
from .logs import log_event

logger = logging.getLogger("ia")

MESSAGES = {
    "IA_UNAUTHORIZED": "API key ausente o incorrecta.",
    "IA_FILE_TOO_LARGE": "El archivo supera el tamaño máximo.",
    "IA_INVALID_FORMAT": "Formato no soportado.",
    "IA_DECODE_FAILED": "No se pudo leer el archivo (puede estar dañado).",
    "IA_MODEL_NOT_READY": "Los modelos se están cargando.",
    "IA_BUSY": "La IA está ocupada; reintenta.",
    "IA_PROCESSING_TIMEOUT": "El archivo superó el tiempo máximo de procesamiento.",
    "IA_INTERNAL": "Error interno de la IA.",
}


def pipeline_error_response(request: Request, err: PipelineError):
    code = err.code if err.code in MESSAGES else "IA_INTERNAL"
    level = logging.ERROR if code == "IA_INTERNAL" else logging.WARNING
    log_event("ia.detect.failed", err.message, level, error_code=code)
    context = {k: v for k, v in getattr(err, "context", {}).items() if v is not None} or None
    return fail(request, code, MESSAGES[code], context)


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        # Igual que ia-fake: falta "file" o "file_name".
        return fail(
            request,
            "IA_INVALID_FORMAT",
            "La petición debe incluir 'file' y 'file_name'.",
            {"errors": [e["loc"][-1] for e in exc.errors()]},
            status=400,
        )

    @app.exception_handler(PipelineError)
    async def _pipeline(request: Request, exc: PipelineError):
        return pipeline_error_response(request, exc)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log_event(
            "ia.detect.failed",
            f"Excepción no controlada en {request.method} {request.url.path}",
            logging.ERROR,
            exc_info=exc,
            error_code="IA_INTERNAL",
        )
        return fail(request, "IA_INTERNAL", MESSAGES["IA_INTERNAL"])
