"""Eventos que ``FolderJob`` publica en su cola (HU-DS-01). La ventana los consume; no hay ``tkinter`` aquí.

Orden garantizado: ``JobStarted`` → (``Progress``… → ``FileDone``) × N → ``JobDone``. Nada llega después
de ``JobDone``. Si algo impide empezar o seguir (p. ej. la carpeta no existe), llega ``JobError`` y
después ``JobDone``.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class JobStarted:
    total: int


@dataclass(frozen=True)
class Progress:
    file_index: int  # 1..N
    file_name: str
    stage: str  # "sampling" | "detection" | "classification" (lo que informa el pipeline)
    done: int
    total: int


@dataclass(frozen=True)
class FileDone:
    file_index: int
    file_name: str
    status: str  # "done" | "error" | "cancelled"
    elapsed_ms: int
    result: Optional[Any] = None  # wwm_pipeline.pipeline.FileResult si status == "done"
    action: Optional[str] = None  # lo que hizo apply_action (p. ej. "copiado a panthera_onca_jaguar"), o None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class JobSummary:
    total: int
    processed: int = 0
    with_animals: int = 0
    without_animals: int = 0
    errors: int = 0
    stopped: bool = False
    elapsed_ms: int = 0


@dataclass(frozen=True)
class JobDone:
    summary: JobSummary

    @property
    def stopped(self) -> bool:
        return self.summary.stopped


@dataclass(frozen=True)
class JobError:
    code: str
    message: str = ""
