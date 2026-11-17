"""Logs JSON en una línea a ``stdout`` (HU-IA-42, ESTÁNDARES §7).

- Campos obligatorios: ``ts, level, service="ia", env, version, request_id, event, message`` (§7.1). Los de
  contexto (``flow``, ``file_name``, ``duration_ms``, ``error_code``, ``category``…) se agregan con ``log_event``.
- ``request_id`` y ``flow`` salen de los *headers* (el *middleware* los pone en ``contextvars``) y viajan al hilo de
  inferencia (``InferenceWorker`` copia el contexto), así los eventos ``pipeline.*`` también los llevan.
- **Nunca** se registran el contenido del archivo, el base64 del fotograma ni la API key (§7.5).
- ``LOG_FORMAT=console`` → texto legible para desarrollo.
- uvicorn (``uvicorn``, ``uvicorn.access``) pasa al mismo formato; los mensajes de YOLOv5/ultralytics
  ("Fusing layers…") se silencian a ``WARNING``.

Se llama ``logs.py`` (no ``logging.py``) para no confundirlo con el módulo estándar.
"""

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
flow_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("flow", default=None)

logger = logging.getLogger("ia")
QUIET_LOGGERS = ("yolov5", "ultralytics", "speciesnet", "absl", "PIL", "matplotlib", "httpx")
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
_MARK = "_wwm_ia_handler"


class JsonFormatter(logging.Formatter):
    def __init__(self, env: str = "development", version: str = "dev"):
        super().__init__()
        self.env = env
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": "ia",
            "env": self.env,
            "version": self.version,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "event": getattr(record, "event", None) or record.name,
            "message": record.getMessage(),
        }
        flow = getattr(record, "flow", None) or flow_var.get()
        if flow:
            payload["flow"] = flow
        payload.update(getattr(record, "ctx", None) or {})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = getattr(record, "ctx", None) or {}
        extra = " ".join(f"{k}={v}" for k, v in ctx.items())
        line = f"{record.levelname:7} {request_id_var.get():12} {getattr(record, 'event', record.name)} {record.getMessage()} {extra}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line.rstrip()


class _StdoutHandler(logging.StreamHandler):
    """Escribe en el ``sys.stdout`` del momento (no en el de cuando se creó): así funciona con Docker y con pytest."""

    @property  # type: ignore[override]
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ lo asigna; se ignora
        pass


def make_formatter(env: str, version: str, fmt: Optional[str] = None) -> logging.Formatter:
    fmt = fmt or os.getenv("LOG_FORMAT", "json")
    return ConsoleFormatter() if fmt == "console" else JsonFormatter(env, version)


def configure(env: str = "development", version: str = "dev", level: Optional[str] = None) -> None:
    """Idempotente: agrega el handler a ``stdout`` una sola vez y no quita los demás (los de pytest, por ejemplo)."""
    root = logging.getLogger()
    if not any(getattr(h, _MARK, False) for h in root.handlers):
        handler = _StdoutHandler()
        setattr(handler, _MARK, True)
        handler.setFormatter(make_formatter(env, version))
        root.addHandler(handler)
    root.setLevel(level or os.getenv("LOG_LEVEL") or "INFO")
    for name in UVICORN_LOGGERS:  # §7.6: el access log de uvicorn, en el mismo formato
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(event: str, message: str = "", level: int = logging.INFO, exc_info: Any = None, **ctx: Any) -> None:
    """Un evento con nombre ``<dominio>.<acción>.<resultado>`` (§7.2) y su contexto."""
    clean = {k: v for k, v in ctx.items() if v is not None}
    logger.log(level, message or event, extra={"event": event, "ctx": clean}, exc_info=exc_info)
