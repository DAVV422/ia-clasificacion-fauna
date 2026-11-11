"""Análisis de una carpeta, sin interfaz (HU-DS-01).

``FolderJob.run()`` recorre los archivos soportados de la carpeta (no recursivo, por nombre), clasifica
cada uno con ``classify_file`` y el preset de escritorio, aplica la acción inyectada y publica eventos
(``desktop.job.events``) en una cola. La ventana lo corre en un hilo y solo lee la cola: el hilo de la
interfaz nunca llama al pipeline. No importa ``tkinter``, así que se prueba sin ventana.
"""

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from wwm_pipeline.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PipelineConfig
from wwm_pipeline.errors import PipelineCancelled, PipelineError
from wwm_pipeline.pipeline import classify_file

from .events import FileDone, JobDone, JobError, JobStarted, JobSummary, Progress

logger = logging.getLogger("desktop.job")

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
PROGRESS_INTERVAL_S = 0.25  # como mucho un evento de progreso cada 250 ms por archivo
ApplyAction = Callable[[object, Path], Optional[str]]


def list_media(source: Path) -> List[Path]:
    return sorted(
        (p for p in Path(source).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def configure_threads() -> int:
    """Deja un núcleo para la interfaz: ``torch.set_num_threads(cpu - 1)``. Devuelve los hilos usados."""
    threads = max(1, (os.cpu_count() or 2) - 1)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:  # pragma: no cover - el escritorio siempre trae torch
        pass
    return threads


class FolderJob:
    def __init__(
        self,
        source,
        models,
        events: "queue.Queue",
        stop: threading.Event,
        apply_action: Optional[ApplyAction] = None,
        config: Optional[PipelineConfig] = None,
        tmp_context=None,
        classify=classify_file,
        clock=time.monotonic,
    ):
        self.source = Path(source)
        self.models = models
        self.events = events
        self.stop = stop
        self.apply_action = apply_action
        self.config = config or PipelineConfig.desktop()
        self.tmp_context = tmp_context
        self.classify = classify
        self.clock = clock

    def _tmp(self):
        if self.tmp_context is not None:
            return self.tmp_context()
        from desktop import paths  # HU-DS-04: el temporal de cada archivo se borra al terminar

        return paths.file_tmp()

    def run(self) -> JobSummary:
        t0 = self.clock()
        try:
            files = list_media(self.source)
        except OSError as exc:
            self.events.put(JobError("DESKTOP_SOURCE_UNREADABLE", str(exc)))
            summary = JobSummary(total=0, elapsed_ms=self._ms(t0))
            self.events.put(JobDone(summary))
            return summary

        total = len(files)
        self.events.put(JobStarted(total))
        processed = with_animals = errors = 0
        stopped = False
        for i, path in enumerate(files, start=1):
            if self.stop.is_set():
                stopped = True
                break
            done = self._process(i, path)
            if done.status == "cancelled":
                stopped = True
                self.events.put(done)
                break
            processed += 1
            if done.status == "error":
                errors += 1
            elif done.result is not None and done.result.decision.category == "animal":
                with_animals += 1
            self.events.put(done)
        summary = JobSummary(
            total=total,
            processed=processed,
            with_animals=with_animals,
            without_animals=processed - with_animals - errors,
            errors=errors,
            stopped=stopped or self.stop.is_set() and processed < total,
            elapsed_ms=self._ms(t0),
        )
        self.events.put(JobDone(summary))
        return summary

    def _process(self, index: int, path: Path) -> FileDone:
        t0 = self.clock()
        last: Dict[str, Any] = {"t": None, "stage": None}

        def on_progress(stage: str, done: int, total: int) -> None:
            now = self.clock()
            changed = stage != last["stage"]
            if changed or done >= total or last["t"] is None or now - last["t"] >= PROGRESS_INTERVAL_S:
                last["t"], last["stage"] = now, stage
                self.events.put(Progress(index, path.name, stage, done, total))

        try:
            with self._tmp():
                result = self.classify(
                    path, path.name, self.config, self.models, on_progress=on_progress, should_stop=self.stop.is_set
                )
        except PipelineCancelled:
            return FileDone(index, path.name, "cancelled", self._ms(t0))
        except PipelineError as err:
            return FileDone(index, path.name, "error", self._ms(t0), error_code=err.code, error_message=err.message)
        except Exception as exc:  # un archivo raro no detiene la carpeta
            logger.exception("Falló la clasificación de %s", path.name)
            return FileDone(index, path.name, "error", self._ms(t0), error_code="IA_INTERNAL", error_message=str(exc))

        action = None
        if self.apply_action is not None:
            try:
                action = self.apply_action(result, path)
            except OSError as exc:
                logger.exception("No se pudo copiar o mover %s", path.name)
                return FileDone(
                    index,
                    path.name,
                    "error",
                    self._ms(t0),
                    result=result,
                    error_code="DESKTOP_ACTION_FAILED",
                    error_message=str(exc),
                )
        return FileDone(index, path.name, "done", self._ms(t0), result=result, action=action)

    def _ms(self, t0: float) -> int:
        return int((self.clock() - t0) * 1000)
