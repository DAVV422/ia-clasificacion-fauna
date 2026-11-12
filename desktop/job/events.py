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
    result: Optional[Any] = None  # wwm_pipeline.pipeline.FileResult si se pudo clasificar
    action: Optional[Any] = None  # wwm_pipeline.package.DesktopAction (copy | move | left_in_place), o None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    # HU-DS-02: la línea de clasificaciones.jsonl (wwm_pipeline.package.PackageLine), ya validada. La escribe
    # HU-DS-14; None solo si ni siquiera se pudo leer el archivo para calcular su sha256.
    line: Optional[Any] = None
    # HU-DS-30: un aviso que no es error (p. ej. SOURCE_NOT_DELETED: se copió pero el original estaba en uso)
    warning_code: Optional[str] = None


@dataclass(frozen=True)
class JobSummary:
    total: int
    processed: int = 0
    with_animals: int = 0
    without_animals: int = 0
    errors: int = 0
    relocated: int = 0  # copiados o movidos a una carpeta de especie, género o familia
    left_in_place: int = 0  # quedaron en el origen (incluye los errores)
    not_deleted: int = 0  # "mover" que quedó como copia porque el original estaba en uso (F3-D8)
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
