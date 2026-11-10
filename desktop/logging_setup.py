"""Logs JSON rotativos de la app de escritorio (HU-DS-00, ESTANDARES §7).

Cada línea: ``ts, level, service="desktop", version, job_id, event, message`` (+ ``exc_info``).
Las rutas del usuario (``C:\\Users\\<nombre>``) se reemplazan por ``<usuario>``.
"""

import getpass
import json
import logging
import logging.handlers
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from desktop import __version__
from desktop.paths import logs_dir

LOG_FILE_NAME = "wwm.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 4  # 1 activo + 4 rotados = 5 archivos como máximo

_job_id = "-"


def set_job_id(job_id: Optional[str]) -> None:
    global _job_id
    _job_id = job_id or "-"


def _user_patterns():
    patterns = []
    home = str(Path.home())
    if home:
        patterns.append((re.compile(re.escape(home), re.IGNORECASE), "<usuario>"))
        patterns.append((re.compile(re.escape(home.replace("\\", "/")), re.IGNORECASE), "<usuario>"))
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - depende del sistema
        user = os.getenv("USERNAME", "")
    if user and len(user) > 2:
        patterns.append((re.compile(rf"([\\/])Users([\\/]){re.escape(user)}", re.IGNORECASE), r"\1Users\2<usuario>"))
    return patterns


_PATTERNS = _user_patterns()


def anonymize(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": "desktop",
            "version": __version__,
            "job_id": getattr(record, "job_id", _job_id),
            "event": getattr(record, "event", record.name),
            "message": record.getMessage(),
        }
        for key, value in getattr(record, "ctx", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return anonymize(json.dumps(payload, ensure_ascii=False, default=str))


def configure(level: int = logging.INFO, log_dir: Optional[Path] = None) -> Path:
    """Configura el logger raíz con el archivo JSON rotativo. Devuelve la ruta del log."""
    folder = log_dir or logs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_FILE_NAME
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_wwm", False):
            root.removeHandler(h)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler._wwm = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    return path


def log_event(event: str, message: str = "", level: int = logging.INFO, **ctx) -> None:
    logging.getLogger("desktop").log(level, message or event, extra={"event": event, "ctx": ctx})
