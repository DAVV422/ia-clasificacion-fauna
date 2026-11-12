"""Pantalla DESCARGANDO del primer inicio (HU-DS-13, "Pantallas" de 02_FLUJOS, E-02).

Se abre antes de la ventana principal solo si faltan modelos o alguno no coincide. El trabajo
(``model_store.install``) corre en un hilo y publica eventos en una cola; la pantalla solo los dibuja.

    run_first_run(root, manifest) -> True  (modelos listos)  |  False  (Cancelar o Salir)

Los textos (progreso, velocidad, tiempo restante, errores) están en funciones sin tkinter, con tests.
"""

import collections
import contextlib
import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Deque, Dict, Optional, Tuple

from desktop import model_store as ms
from desktop.messages import message_for

logger = logging.getLogger("desktop.ui")

BG, FG, DIM, ACCENT, ERROR_FG = "#1a2a1f", "#e8f0ea", "#9fb3a5", "#4caf50", "#ff8a80"
RATE_WINDOW_S = 5.0
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Textos (sin tkinter)
# ---------------------------------------------------------------------------
def progress_text(done: int, total: int) -> str:
    pct = int(done * 100 / total) if total else 0
    return f"{pct} %  ·  {done // MB}/{total // MB} MB"


class RateMeter:
    """Velocidad con una media móvil de 5 s y el tiempo restante."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.samples: Deque[Tuple[float, int]] = collections.deque()

    def add(self, total_done: int) -> None:
        now = self.clock()
        self.samples.append((now, total_done))
        while self.samples and now - self.samples[0][0] > RATE_WINDOW_S:
            self.samples.popleft()

    def rate(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    def text(self, remaining_bytes: int) -> str:
        rate = self.rate()
        if rate <= 0:
            return "Calculando velocidad…"
        seconds = remaining_bytes / rate
        left = f"~{int(seconds // 60) + 1} min" if seconds >= 60 else f"~{max(1, int(seconds))} s"
        return f"{rate / MB:.1f} MB/s  ·  quedan {left}".replace(".", ",")


def error_text(code: str) -> str:
    return message_for(code)


ERROR_BUTTONS = {
    "NETWORK_UNAVAILABLE": ("Reintentar", "Salir"),
    "DOWNLOAD_INTEGRITY_FAILED": ("Reintentar", "Abrir carpeta de logs", "Salir"),
    "DISK_SPACE_LOW": ("Reintentar", "Salir"),
}


# ---------------------------------------------------------------------------
# La pantalla
# ---------------------------------------------------------------------------
class FirstRunWindow(tk.Tk):
    def __init__(self, root: Path, manifest: ms.Manifest, install=ms.install):
        super().__init__()
        self.models_root, self.manifest, self._install = root, manifest, install
        self.result: Optional[bool] = None
        self.events: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.meter = RateMeter()
        self.done_by_id: Dict[str, int] = {}
        self.title("WWM Clasificador — preparando")
        self.configure(bg=BG, padx=24, pady=18)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        tk.Label(
            self, text="Preparando el clasificador (solo la primera vez)", bg=BG, fg=FG, font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(0, 12))
        self.rows: Dict[str, Tuple[ttk.Progressbar, tk.Label]] = {}
        for artifact in manifest.artifacts:
            frame = tk.Frame(self, bg=BG)
            frame.pack(fill="x", pady=3)
            tk.Label(frame, text=artifact.name, bg=BG, fg=FG, width=20, anchor="w").pack(side="left")
            bar = ttk.Progressbar(frame, length=280, maximum=artifact.size)
            bar.pack(side="left", padx=8)
            status = tk.Label(frame, text="en espera", bg=BG, fg=DIM, width=22, anchor="w")
            status.pack(side="left")
            self.rows[artifact.id] = (bar, status)
        self.rate_label = tk.Label(self, text="Verificando los modelos…", bg=BG, fg=DIM)
        self.rate_label.pack(anchor="w", pady=(12, 0))
        tk.Label(self, text="Necesitas internet solo para este paso.", bg=BG, fg=DIM).pack(anchor="w")
        self.message = tk.Label(self, text="", bg=BG, fg=ERROR_FG, wraplength=560, justify="left")
        self.message.pack(anchor="w", pady=(8, 0))
        self.buttons = tk.Frame(self, bg=BG)
        self.buttons.pack(anchor="e", pady=(12, 0))
        self._set_buttons(("Cancelar",))
        self.after(100, self._start)
        self.after(100, self._poll)

    # -- trabajo ------------------------------------------------------------
    def _start(self) -> None:
        self.stop.clear()
        self.message.config(text="")
        self._set_buttons(("Cancelar",))
        threading.Thread(target=self._work, daemon=True, name="ModelDownload").start()

    def _work(self) -> None:
        try:
            result = self._install(
                self.models_root,
                self.manifest,
                on_progress=lambda p: self.events.put(("progress", p)),
                should_stop=self.stop.is_set,
            )
            self.events.put(("done", result))
        except ms.Cancelled:
            self.events.put(("cancelled", None))
        except ms.ModelStoreError as exc:
            logger.error("Descarga de modelos: %s (%s)", exc.code, exc.message)
            self.events.put(("error", exc.code))
        except Exception:  # pragma: no cover - inesperado
            logger.exception("Descarga de modelos")
            self.events.put(("error", "INTERNAL_ERROR"))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                self._handle(kind, value)
                if self.result is not None:
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _handle(self, kind: str, value) -> None:
        if kind == "progress":
            bar, status = self.rows[value.artifact_id]
            bar.config(value=value.done)
            status.config(text=progress_text(value.done, value.total), fg=FG)
            self.done_by_id[value.artifact_id] = value.done
            done = sum(self.done_by_id.values())
            self.meter.add(done)
            sizes = {a.id: a.size for a in self.manifest.artifacts}
            total = sum(sizes[i] for i in self.done_by_id)  # solo lo que se descarga (lo reusado no cuenta)
            self.rate_label.config(text=self.meter.text(max(0, total - done)))
        elif kind == "done":
            for artifact in self.manifest.artifacts:
                bar, status = self.rows[artifact.id]
                bar.config(value=artifact.size)
                reused = artifact.id in getattr(value, "reused", [])
                status.config(text="ya estaba ✓" if reused else "listo ✓", fg=ACCENT)
            self._finish(True)
        elif kind == "cancelled":
            self._finish(False)
        elif kind == "error":
            self.message.config(text=error_text(value))
            self._set_buttons(ERROR_BUTTONS.get(value, ("Reintentar", "Salir")))

    def _finish(self, ok: bool) -> None:
        self.result = ok
        self.after(300 if ok else 0, self.destroy)

    # -- botones -------------------------------------------------------------
    def _set_buttons(self, labels) -> None:
        for child in self.buttons.winfo_children():
            child.destroy()
        actions = {
            "Cancelar": self._on_cancel,
            "Reintentar": self._start,
            "Salir": lambda: self._finish(False),
            "Abrir carpeta de logs": self._open_logs,
        }
        for label in labels:
            tk.Button(self.buttons, text=label, command=actions[label], padx=10).pack(side="left", padx=4)

    def _on_cancel(self) -> None:
        """CA-5: la app se cierra y los .part quedan para reanudar."""
        self.stop.set()
        self.rate_label.config(text="Cancelando…")
        self.after(3000, lambda: self.result is None and self._finish(False))

    def _open_logs(self) -> None:
        from desktop import paths

        with contextlib.suppress(AttributeError, OSError):  # os.startfile solo existe en Windows
            os.startfile(str(paths.logs_dir()))  # type: ignore[attr-defined]


def run_first_run(root: Path, manifest: ms.Manifest) -> bool:
    window = FirstRunWindow(root, manifest)
    window.mainloop()
    return bool(window.result)
