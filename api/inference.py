"""La inferencia, fuera del *event loop*: un solo hilo, una cola corta y un corte por tiempo (HU-IA-04, IA §1).

- **Un hilo** con los modelos cargados: un archivo a la vez, así la RAM se mantiene estable (ESTÁNDARES §2.2).
- **Admisión:** como mucho ``1 + INFERENCE_QUEUE_SIZE`` trabajos en el sistema (el que se procesa + los que esperan),
  igual que ``SimulatedQueue`` de ``ia-fake``. Lleno → ``IA_BUSY``. ``pending`` = los que esperan.
- **Timeout:** ``PROCESS_TIMEOUT_SECONDS`` (150 s) → ``IA_PROCESSING_TIMEOUT``. El trabajo se **cancela de verdad**:
  se activa su ``should_stop`` y ``classify_file`` corta entre frames; su lugar se libera cuando el hilo lo suelta.
- **La limpieza** (borrar la carpeta temporal) la hace el hilo al terminar cada trabajo, no la petición: con un
  *timeout*, la petición responde 504 mientras el hilo todavía puede estar leyendo el archivo.
- El contexto (``contextvars``: ``request_id``, ``flow``) viaja con cada trabajo, para los logs (HU-IA-42).
"""

import asyncio
import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from wwm_pipeline.errors import PipelineCancelled, PipelineError

logger = logging.getLogger("ia")


@dataclass
class _Job:
    fn: Callable[[Callable[[], bool]], Any]
    future: "asyncio.Future"
    loop: asyncio.AbstractEventLoop
    context: contextvars.Context
    cleanup: Optional[Callable[[], None]] = None
    stop: threading.Event = field(default_factory=threading.Event)


class InferenceWorker:
    def __init__(self, queue_size: int = 2, timeout_seconds: float = 150.0):
        self.queue_size = queue_size
        self.timeout_seconds = timeout_seconds
        self.capacity = 1 + queue_size
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._lock = threading.Lock()
        self._in_system = 0
        self._thread: Optional[threading.Thread] = None

    # -- estado ---------------------------------------------------------------
    @property
    def in_system(self) -> int:
        return self._in_system

    @property
    def pending(self) -> int:
        """Los que esperan (sin contar el que se procesa): lo que informa ``/health``."""
        return max(0, self._in_system - 1)

    def is_full(self) -> bool:
        return self._in_system >= self.capacity

    def try_enter(self) -> bool:
        with self._lock:
            if self._in_system >= self.capacity:
                return False
            self._in_system += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._in_system = max(0, self._in_system - 1)

    # -- el hilo ----------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="InferenceWorker")
            self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            outcome: Any = None
            failed: Optional[BaseException] = None
            try:
                if job.stop.is_set():  # ya venció mientras esperaba: ni se empieza
                    raise PipelineCancelled("timeout antes de empezar")
                outcome = job.context.run(job.fn, job.stop.is_set)
            except BaseException as exc:  # noqa: BLE001 - se entrega a la petición
                failed = exc
            # Primero limpiar y liberar el lugar, después avisar: cuando la petición responde, ya no queda nada.
            if job.cleanup is not None:
                try:
                    job.cleanup()
                except Exception:  # pragma: no cover - la limpieza nunca tumba el hilo
                    logger.exception("ia.worker.cleanup_failed")
            self.leave()
            if failed is not None:
                job.loop.call_soon_threadsafe(_set_exception, job.future, failed)
            else:
                job.loop.call_soon_threadsafe(_set_result, job.future, outcome)

    async def submit(
        self, fn: Callable[[Callable[[], bool]], Any], cleanup: Optional[Callable[[], None]] = None
    ) -> Any:
        """Encola ``fn(should_stop)`` (el lugar ya reservado con ``try_enter``) y espera su resultado o el *timeout*."""
        self._ensure_started()
        loop = asyncio.get_running_loop()
        job = _Job(fn=fn, future=loop.create_future(), loop=loop, context=contextvars.copy_context(), cleanup=cleanup)
        t0 = time.monotonic()
        self._jobs.put(job)
        try:
            return await asyncio.wait_for(asyncio.shield(job.future), self.timeout_seconds)
        except asyncio.TimeoutError:
            job.stop.set()  # classify_file corta en el próximo frame; el hilo queda libre
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("ia.detect.timeout elapsed_ms=%d", elapsed_ms)
            raise PipelineError("IA_PROCESSING_TIMEOUT", "superó el tiempo máximo", elapsed_ms=elapsed_ms) from None


def _set_result(future: "asyncio.Future", value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(future: "asyncio.Future", exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)
