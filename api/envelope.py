"""La envoltura estándar ``{status, data, message, error, meta}`` (IA §1, ESTÁNDARES §4) y la correlación.

El mismo formato que ``fake/app.py`` (E2-D2), en código propio: ``ia-fake`` no depende de ``api/``.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from wwm_pipeline.contract import IA_ERROR_HTTP


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]


def flow(request: Request) -> Optional[str]:
    return getattr(request.state, "flow", None)


def _meta(request: Request) -> dict:
    return {"request_id": request_id(request), "timestamp": datetime.now(timezone.utc).isoformat()}


def ok(request: Request, data: Any, message: str = "Archivo clasificado") -> JSONResponse:
    return JSONResponse({"status": "success", "data": data, "message": message, "error": None, "meta": _meta(request)})


def fail(
    request: Request,
    code: str,
    message: str,
    context: Optional[dict] = None,
    status: Optional[int] = None,
) -> JSONResponse:
    details = [{"code": code, "context": context}] if context else []
    body = {
        "status": "error",
        "data": None,
        "message": message,
        "error": {"code": code, "flow": flow(request), "details": details},
        "meta": _meta(request),
    }
    return JSONResponse(body, status_code=status or IA_ERROR_HTTP.get(code, 500))
