"""Ventana actual de la app de escritorio (prototipo v1), movida desde desktop_app.py.

HU-DS-00: el arranque vive en desktop/main.py. El rediseño de la ventana es HU-DS-22.
HU-DS-01: el análisis lo hace ``desktop.job.FolderJob`` (modo precisión) en un hilo; la ventana solo
lee sus eventos y nunca llama al pipeline.
"""


import os
import sys
import threading
import queue
import logging
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# --- HIDDEN IMPORTS PARA PYINSTALLER ---
# PyInstaller no detecta dependencias dinámicas cargadas por torch.hub (YOLOv5).
# Al importarlas aquí, forzamos a PyInstaller a incluirlas en el .exe.
try:
    import ultralytics
    import pandas
    import yaml
    import tqdm
    import torchvision
    import speciesnet
    import speciesnet.classifier
    import cloudpathlib
    import kagglehub
    import onnx2torch
    import onnx
except ImportError:
    pass
# ---------------------------------------

from desktop import __version__
from desktop import paths
from desktop.paths import resource_dir

# Recursos (íconos, logo): carpeta del repo en desarrollo, _MEIPASS en el ejecutable.
PROJECT_ROOT = resource_dir()

# Los logs de la API (uvicorn) no aplican en el escritorio: se reenvían al logger raíz (JSON).
logging.getLogger("uvicorn.error").handlers = []
logging.getLogger("uvicorn.error").propagate = True

from app.detector import WildlifeDetector
from desktop.job.actions import make_species_action
from desktop.job.events import FileDone, JobDone, JobError, JobStarted, Progress
from desktop.job.folder_job import FolderJob, configure_threads, list_media
from desktop.job.job_log import JobLog
from desktop.ui import texts


# Peso de cada etapa del pipeline dentro del avance de un archivo (para la barra de progreso).
STAGE_SPAN = {
    "sampling": (0.0, 0.15),
    "detection": (0.15, 0.85),
    "classification": (0.85, 0.97),
    "save": (0.97, 1.0),  # HU-DS-31: fotograma + copia o movimiento + línea del registro
}
STAGE_LABEL = {"sampling": "lectura", "detection": "detección", "classification": "clasificación", "save": "guardando"}


# ---------------------------------------------------------------------------
# Colores de la UI (Tema Blanco y Negro)
# ---------------------------------------------------------------------------
BG_DARK     = "#181818"      # Fondo oscuro principal
BG_MID      = "#242424"      # Fondo oscuro secundario
BG_PANEL    = "#0a0a0a"      # Paneles y cabecera
ACCENT      = "#ffffff"      # Blanco puro para contrastar
ACCENT_DARK = "#d4d4d4"      # Gris muy claro para botones
TEXT_MAIN   = "#ffffff"      # Texto principal blanco
TEXT_DIM    = "#a3a3a3"      # Texto secundario gris
LOG_BG      = "#000000"      # Fondo de log negro profundo
LOG_FG      = "#ffffff"      # Texto de log blanco
ERROR_FG    = "#ef4444"      # Rojo
WARN_FG     = "#eab308"      # Amarillo



# ===========================================================================
# Aplicación principal
# ===========================================================================
class WildlifeDesktopApp(tk.Tk):
    def __init__(self):
        self._closed = False
        self._poll_id = None
        super().__init__()

        self.title(f"🦁  WWM Wildlife Detector  — v{__version__}")
        self.geometry("860x650")
        self.minsize(700, 560)
        self.configure(bg=BG_DARK)

        # Aplicar el ícono a la ventana
        try:
            icon_path = PROJECT_ROOT / "public" / "logo.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        # HU-DS-04: pesos en AppData · HU-DS-11: SpeciesNet desde su carpeta local, sin Kaggle
        self.detector = WildlifeDetector(paths.megadetector_path(), paths.speciesnet_dir())
        self.log_queue: queue.Queue = queue.Queue()
        self.job_events: queue.Queue = queue.Queue()  # eventos de FolderJob (HU-DS-01)
        self.stop_event = threading.Event()
        self.is_running = False
        self._stage = None  # la etapa del archivo en curso (HU-DS-31)
        self._current_file = None
        self._close_after_job = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)  # HU-DS-31 CA-4
        self._poll_log_queue()

        # Cargar modelos en segundo plano al arrancar
        self.after(400, self._init_models_async)

        # Cerrar el splash screen de PyInstaller (si existe) una vez que la interfaz carga
        try:
            import pyi_splash
            if pyi_splash.is_alive():
                pyi_splash.close()
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # Construcción de la interfaz
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # ── Cabecera ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, padx=20, pady=12)
        header.pack(fill="x")

        # Intentar cargar y mostrar el logo
        try:
            from PIL import Image, ImageTk
            logo_path = PROJECT_ROOT / "public" / "logo.jpeg"
            if logo_path.exists():
                img = Image.open(logo_path)
                # Redimensionar el logo a una altura de 40px manteniendo la proporción
                ratio = 40 / float(img.size[1])
                new_width = int(float(img.size[0]) * float(ratio))
                img = img.resize((new_width, 40), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                
                logo_lbl = tk.Label(header, image=self.logo_img, bg=BG_PANEL)
                logo_lbl.pack(side="left", padx=(0, 15))
        except Exception as e:
            print(f"Error loading logo: {e}")

        tk.Label(
            header,
            text="WWM Clasificador de especies",
            font=("Segoe UI", 18, "bold"),
            bg=BG_PANEL,
            fg=ACCENT,
        ).pack(side="left")

        self.status_dot = tk.Label(
            header,
            text="⬤  Cargando modelos…",
            font=("Segoe UI", 9),
            bg=BG_PANEL,
            fg=WARN_FG,
        )
        self.status_dot.pack(side="right", padx=5)

        # ── Área de configuración ───────────────────────────────────────────
        config_frame = tk.Frame(self, bg=BG_DARK, padx=20, pady=10)
        config_frame.pack(fill="x")

        # Carpeta de entrada
        self._make_folder_row(
            config_frame,
            label="📁  Carpeta de entrada  (imágenes / videos):",
            var_name="input_var",
            row=0,
            command=self._select_input,
        )

        # Carpeta de salida
        self._make_folder_row(
            config_frame,
            label="📂  Carpeta de salida:",
            var_name="output_var",
            row=1,
            command=self._select_output,
        )

        # Opciones en la misma fila
        opts_frame = tk.Frame(config_frame, bg=BG_DARK)
        opts_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        tk.Label(opts_frame, text="⚙️  Acción:", bg=BG_DARK, fg=TEXT_MAIN,
                 font=("Segoe UI", 10)).pack(side="left")

        self.action_var = tk.StringVar(value="copy")
        for text, val in [("Copiar archivos", "copy"), ("Mover archivos", "move")]:
            rb = tk.Radiobutton(
                opts_frame,
                text=text,
                variable=self.action_var,
                value=val,
                bg=BG_DARK,
                fg=TEXT_MAIN,
                selectcolor=BG_MID,
                activebackground=BG_DARK,
                activeforeground=ACCENT,
                font=("Segoe UI", 10),
            )
            rb.pack(side="left", padx=12)

        config_frame.columnconfigure(1, weight=1)

        # ── Botón de inicio ─────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=20, pady=6)

        self.start_btn = tk.Button(
            btn_frame,
            text="▶   Iniciar Análisis",
            command=self._on_start,
            bg=ACCENT_DARK,
            fg="black",
            activebackground=ACCENT,
            activeforeground="black",
            font=("Segoe UI", 12, "bold"),
            padx=28,
            pady=8,
            relief="flat",
            cursor="hand2",
            bd=0,
        )
        self.start_btn.pack(side="left")

        self.stop_btn = tk.Button(
            btn_frame,
            text="⏹   Detener",
            command=self._on_stop,
            bg="#7f1d1d",
            fg="white",
            activebackground=ERROR_FG,
            font=("Segoe UI", 11),
            padx=18,
            pady=8,
            relief="flat",
            cursor="hand2",
            bd=0,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=10)

        # ── Barra de progreso ───────────────────────────────────────────────
        prog_frame = tk.Frame(self, bg=BG_DARK, padx=20)
        prog_frame.pack(fill="x")

        self.prog_label = tk.Label(
            prog_frame,
            text="Esperando…",
            bg=BG_DARK,
            fg=TEXT_DIM,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.prog_label.pack(fill="x")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "WWF.Horizontal.TProgressbar",
            background=ACCENT_DARK,
            troughcolor=BG_MID,
            bordercolor=BG_MID,
            lightcolor=ACCENT,
            darkcolor=ACCENT_DARK,
        )

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            prog_frame,
            variable=self.progress_var,
            maximum=100,
            style="WWF.Horizontal.TProgressbar",
            length=300,
        )
        self.progress_bar.pack(fill="x", pady=(2, 6))

        # ── Área de log ─────────────────────────────────────────────────────
        log_header = tk.Frame(self, bg=BG_DARK, padx=20)
        log_header.pack(fill="x")
        tk.Label(
            log_header,
            text="📋  Log de procesamiento",
            bg=BG_DARK,
            fg=TEXT_DIM,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        tk.Button(
            log_header,
            text="Limpiar",
            command=self._clear_log,
            bg=BG_MID,
            fg=TEXT_DIM,
            font=("Segoe UI", 8),
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=2,
        ).pack(side="right")

        self.log_text = scrolledtext.ScrolledText(
            self,
            height=14,
            bg=LOG_BG,
            fg=LOG_FG,
            font=("Cascadia Code", 9) if sys.platform == "win32" else ("Courier", 9),
            relief="flat",
            insertbackground="white",
            padx=8,
            pady=6,
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        self.log_text.config(state="disabled")

        # Tags de color dentro del log
        self.log_text.tag_config("ok",    foreground=ACCENT)
        self.log_text.tag_config("warn",  foreground=WARN_FG)
        self.log_text.tag_config("error", foreground=ERROR_FG)
        self.log_text.tag_config("dim",   foreground=TEXT_DIM)
        self.log_text.tag_config("head",  foreground="#60a5fa", font=("Cascadia Code", 9, "bold"))

        # ── Footer ──────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=BG_DARK, padx=20, pady=5)
        footer.pack(fill="x", side="bottom")
        tk.Label(
            footer,
            text="Desarrollado por grupo CETERIS 404 - InnovaHack 2026",
            bg=BG_DARK,
            fg=TEXT_DIM,
            font=("Segoe UI", 8, "italic"),
        ).pack(side="right")

    # -----------------------------------------------------------------------
    # Helper: fila carpeta (label + entry + botón)
    # -----------------------------------------------------------------------
    def _make_folder_row(self, parent, label, var_name, row, command):
        tk.Label(
            parent,
            text=label,
            bg=BG_DARK,
            fg=TEXT_DIM,
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=row * 2, column=0, columnspan=3, sticky="w", pady=(6, 1))

        var = tk.StringVar()
        setattr(self, var_name, var)

        entry = tk.Entry(
            parent,
            textvariable=var,
            bg=BG_MID,
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
        )
        entry.grid(row=row * 2 + 1, column=0, columnspan=2, sticky="ew", ipady=5)

        btn = tk.Button(
            parent,
            text="  Examinar…  ",
            command=command,
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            activebackground=ACCENT_DARK,
            font=("Segoe UI", 9),
            relief="flat",
            cursor="hand2",
            bd=0,
            pady=5,
        )
        btn.grid(row=row * 2 + 1, column=2, padx=(6, 0), sticky="e")

    # -----------------------------------------------------------------------
    # Selectores de carpeta
    # -----------------------------------------------------------------------
    def _select_input(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de entrada")
        if folder:
            self.input_var.set(folder)

    def _select_output(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if folder:
            self.output_var.set(folder)

    # -----------------------------------------------------------------------
    # Sistema de log (thread-safe vía queue)
    # -----------------------------------------------------------------------
    def _log(self, message: str, tag: str = "ok"):
        self.log_queue.put((message, tag))
        # También al log JSON (HU-DS-00): soporte puede ver lo mismo que vio el usuario.
        level = logging.ERROR if tag == "error" else logging.WARNING if tag == "warn" else logging.INFO
        logging.getLogger("desktop.ui").log(level, message.strip())

    def _poll_log_queue(self):
        """Consume mensajes pendientes del log y los eventos del análisis cada 100 ms."""
        self._poll_id = None
        while not self.job_events.empty():
            self._on_job_event(self.job_events.get_nowait())
            if self._closed:  # HU-DS-31: "¿Detener y salir?" cerró la ventana al llegar JobDone
                return
        while not self.log_queue.empty():
            msg, tag = self.log_queue.get_nowait()
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n", tag)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self._poll_id = self.after(100, self._poll_log_queue)

    def destroy(self):
        """Cancela el sondeo pendiente: después de cerrar no se toca ningún widget (HU-DS-31)."""
        self._closed = True
        if getattr(self, "_poll_id", None) is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        super().destroy()

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # -----------------------------------------------------------------------
    # Inicialización de modelos (ejecuta en hilo secundario)
    # -----------------------------------------------------------------------
    def _init_models_async(self):
        self._log("⏳  Cargando el modelo…", "warn")

        def _task():
            try:
                # HU-DS-13: desktop.main ya verificó (y descargó si hacía falta) los modelos antes de abrir la ventana.
                configure_threads()  # deja un núcleo libre para la interfaz
                self.detector.load_model()
                self._log("✅  Modelos listos. ¡Puedes iniciar el análisis!", "ok")
                self.after(0, lambda: self.status_dot.config(
                    text="⬤  Listo", fg=ACCENT))
            except Exception as exc:
                self._log(f"❌  Error cargando modelos: {exc}", "error")
                self.after(0, lambda: self.status_dot.config(
                    text="⬤  Error en modelos", fg=ERROR_FG))

        threading.Thread(target=_task, daemon=True).start()

    # -----------------------------------------------------------------------
    # Botones de control
    # -----------------------------------------------------------------------
    def _on_start(self):
        if self.is_running:
            return

        input_dir  = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        action     = self.action_var.get()

        if not input_dir:
            messagebox.showerror("Faltan datos", "Selecciona una carpeta de entrada.")
            return
        if not output_dir:
            messagebox.showerror("Faltan datos", "Selecciona una carpeta de salida.")
            return
        if not Path(input_dir).is_dir():
            messagebox.showerror("Error", "La carpeta de entrada no existe.")
            return
        if self.detector.model is None:
            messagebox.showwarning("Modelos", "Los modelos aún no han terminado de cargar.\nEspera a que el indicador diga '⬤  Listo'.")
            return

        output_path = Path(output_dir)
        log = JobLog(output_path)
        try:
            if log.exists():  # HU-DS-14 · E-08: continuar o empezar de nuevo
                remaining = log.remaining_estimate(list_media(Path(input_dir)))
                title, message = texts.resume_prompt(log.info(), input_dir, remaining)
                choice = messagebox.askyesnocancel(title, message)
                if choice is None:
                    return
                if choice is False:
                    log.backup()
            output_path.mkdir(parents=True, exist_ok=True)
            log.start(input_dir, action)
        except OSError as exc:
            messagebox.showerror("Error", f"{texts.message_for('DESTINATION_NOT_WRITABLE')}\n\n{exc}")
            return
        self._job_log = log

        self.is_running = True
        self._close_after_job = False
        self.stop_event = threading.Event()
        self.start_btn.config(state="disabled", text="⏳  Procesando…")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self._job_info = {"input": input_dir, "output": output_dir, "action": action, "total": 0, "t0": time.monotonic(), "done_ms": []}
        job = FolderJob(
            input_dir,
            self.detector.models,
            self.job_events,
            self.stop_event,
            apply_action=make_species_action(output_path, action),
            dest=output_path,
            record=log.append,
            has_line=log.has_line,
            skip=log.is_processed,
        )
        # HU-DS-31: no-daemon, para que salir de Python espere el final de la sección de guardado
        self._job_thread = threading.Thread(target=job.run, daemon=False, name="FolderJob")
        self._job_thread.start()

    def _on_stop(self):
        if self.is_running and not self.stop_event.is_set():
            self.stop_event.set()
            self.stop_btn.config(state="disabled", text="⏳  Deteniendo…")
            self._log(texts.stopping_text(self._stage, self._current_file), "warn")

    def _on_close(self):
        """HU-DS-31 CA-4: con un análisis en curso, pregunta; con Sí, detiene y cierra al llegar JobDone."""
        if not self.is_running:
            self.destroy()
            return
        if self._close_after_job:
            return
        if messagebox.askyesno(*texts.CLOSE_PROMPT):
            self._close_after_job = True
            self._on_stop()

    # -----------------------------------------------------------------------
    # Eventos del análisis (HU-DS-01): llegan por la cola, se procesan en el hilo de la UI
    # -----------------------------------------------------------------------
    def _on_job_event(self, event):
        info = getattr(self, "_job_info", None) or {}
        if isinstance(event, JobStarted):
            info["total"] = event.total
            if not event.total and event.already_done:
                self._log(f"✅  Los {event.already_done} archivos de la carpeta ya estaban analizados.", "ok")
                return
            if not event.total:
                self._log("⚠️  No se encontraron archivos de imagen o video en la carpeta.", "warn")
                return
            action_label = "Copiando" if info.get("action") == "copy" else "Moviendo"
            self._log(f"\n{'─'*54}", "head")
            self._log(f"  📁 Entrada : {info.get('input')}", "head")
            self._log(f"  📂 Salida  : {info.get('output')}", "head")
            self._log(f"  ⚙️  Acción  : {action_label}", "head")
            self._log(f"  📊 Archivos: {event.total}", "head")
            if event.already_done:
                self._log(f"  ⏭️  Ya analizados (se saltan): {event.already_done}", "head")
            self._log(f"  🔬 Modo    : precisión (video completo + votación)", "head")
            self._log(f"{'─'*54}\n", "head")
        elif isinstance(event, Progress):
            self._stage, self._current_file = event.stage, event.file_name
            self._show_progress(event, info)
        elif isinstance(event, FileDone):
            self._stage = None
            info.setdefault("done_ms", []).append(event.elapsed_ms)
            self._log_file_done(event, info.get("total", 0))
            total = info.get("total") or 1
            self.progress_var.set(event.file_index / total * 100)
        elif isinstance(event, JobError):
            self._log(f"❌  {texts.job_error_text(event)}", "error")
        elif isinstance(event, JobDone):
            self._finish(event, info)

    def _show_progress(self, event: Progress, info: dict):
        total = info.get("total") or 1
        lo, hi = STAGE_SPAN.get(event.stage, (0.0, 1.0))
        within = lo + (hi - lo) * (event.done / event.total if event.total else 0)
        self.progress_var.set(((event.file_index - 1) + within) / total * 100)
        stage = STAGE_LABEL.get(event.stage, event.stage)
        text = f"[{event.file_index}/{total}]  {event.file_name}  ·  {stage} {event.done}/{event.total}"
        eta = self._eta_text(info, event.file_index)
        self.prog_label.config(text=f"{text}{eta}")

    @staticmethod
    def _eta_text(info: dict, file_index: int) -> str:
        done = info.get("done_ms") or []
        elapsed = time.monotonic() - info.get("t0", time.monotonic())
        text = f"  ·  {texts.format_duration(elapsed)} transcurrido"
        if done:
            remaining = (sum(done) / len(done)) / 1000 * (info.get("total", 0) - file_index + 1)
            text += f", faltan ~{texts.format_duration(remaining)}"
        return text

    def _log_file_done(self, event: FileDone, total: int):
        for message, tag in texts.file_done_lines(event, total):
            self._log(message, tag)

    def _finish(self, event: JobDone, info: dict):
        summary = event.summary
        log = getattr(self, "_job_log", None)
        if log is not None:  # JobDone es el último evento: el hilo ya no escribe
            log.close()
            self._job_log = None
        for message, tag in texts.summary_lines(summary):
            self._log(message, tag)
        self.prog_label.config(
            text=f"{'⏹  Detenido' if summary.stopped else '✅  Listo'} — {summary.relocated} clasificados, "
            f"{summary.left_in_place} quedaron en el origen"
        )
        self._reset_ui()
        if self._close_after_job:  # HU-DS-31 CA-4: "¿Detener y salir?" → Sí
            self.destroy()
            return
        if summary.total:
            messagebox.showinfo(
                "Análisis detenido" if summary.stopped else "Análisis completado",
                texts.summary_dialog_text(summary, info.get("output", "")),
            )

    def _reset_ui(self):
        self.is_running = False
        self.start_btn.config(state="normal", text="▶   Iniciar Análisis")
        self.stop_btn.config(state="disabled", text="⏹   Detener")
        self._stage = self._current_file = None

