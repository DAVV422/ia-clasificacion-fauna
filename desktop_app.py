"""
WWF Wildlife Desktop Detector — Prototipo v1
============================================
Detecta animales en imágenes/videos de una carpeta de entrada
y copia o mueve los archivos positivos a una carpeta de salida,
organizados automáticamente por especie detectada.

Uso:
    python desktop_app.py

Dependencias: las mismas que requirements.txt (no necesita instalar nada extra).
"""

import sys
import os
import shutil
import threading
import queue
import logging
from pathlib import Path
from typing import List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# ---------------------------------------------------------------------------
# Añadir la raíz del proyecto al path para poder importar los módulos de app/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silenciar logs de uvicorn que no aplican en modo escritorio
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Redirigir logger de uvicorn al logger raíz
logging.getLogger("uvicorn.error").handlers = []
logging.getLogger("uvicorn.error").propagate = True

from app.config import (
    MEGADETECTOR_URL,
    MEGADETECTOR_PATH,
)
from app.detector import WildlifeDetector

import cv2
import requests


# ---------------------------------------------------------------------------
# Descarga de pesos independiente (sin FastAPI) para modo escritorio
# ---------------------------------------------------------------------------
def _download_weights(url: str, dest_path: Path, log_fn=None):
    """Descarga los pesos del modelo si no existen. Reporta progreso via log_fn."""
    if dest_path.exists():
        if log_fn:
            log_fn(f"✅  Modelo encontrado en {dest_path.name}", "ok")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(".tmp")

    if log_fn:
        log_fn(f"⬇️   Descargando {dest_path.name} (~144 MB)…", "warn")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and log_fn:
                            pct = downloaded / total * 100
                            log_fn(
                                f"   MegaDetector: {pct:.0f}%  "
                                f"({downloaded/1024/1024:.0f} / {total/1024/1024:.0f} MB)",
                                "dim",
                            )
        tmp.rename(dest_path)
        if log_fn:
            log_fn("✅  MegaDetector descargado correctamente.", "ok")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"No se pudo descargar MegaDetector: {exc}")

# ---------------------------------------------------------------------------
# Extensiones soportadas
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

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
        super().__init__()

        self.title("🦁  WWF Wildlife Detector  — Prototipo v1")
        self.geometry("860x650")
        self.minsize(700, 560)
        self.configure(bg=BG_DARK)

        self.detector = WildlifeDetector()
        self.log_queue: queue.Queue = queue.Queue()
        self.is_running = False

        self._build_ui()
        self._poll_log_queue()

        # Cargar modelos en segundo plano al arrancar
        self.after(400, self._init_models_async)

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
            text="WWF Clasificador de especies",
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

    def _poll_log_queue(self):
        """Consume mensajes pendientes del log cada 100 ms."""
        while not self.log_queue.empty():
            msg, tag = self.log_queue.get_nowait()
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n", tag)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(100, self._poll_log_queue)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # -----------------------------------------------------------------------
    # Inicialización de modelos (ejecuta en hilo secundario)
    # -----------------------------------------------------------------------
    def _init_models_async(self):
        self._log("⏳  Verificando / descargando modelos de IA…", "warn")
        self._log("    (La primera vez puede tardar 3-8 min dependiendo de tu conexión)", "dim")

        def _task():
            try:
                _download_weights(MEGADETECTOR_URL, MEGADETECTOR_PATH, log_fn=self._log)
                self.detector.load_model()
                self.log_queue.put(("✅  Modelos listos. ¡Puedes iniciar el análisis!", "ok"))
                self.after(0, lambda: self.status_dot.config(
                    text="⬤  Listo", fg=ACCENT))
            except Exception as exc:
                self.log_queue.put((f"❌  Error cargando modelos: {exc}", "error"))
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

        self.is_running = True
        self._stop_requested = False
        self.start_btn.config(state="disabled", text="⏳  Procesando…")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)

        threading.Thread(
            target=self._analysis_worker,
            args=(input_dir, output_dir, action),
            daemon=True,
        ).start()

    def _on_stop(self):
        if self.is_running:
            self._stop_requested = True
            self._log("⏹  Deteniendo… (terminará el archivo actual)", "warn")

    # -----------------------------------------------------------------------
    # Worker principal de análisis (hilo secundario)
    # -----------------------------------------------------------------------
    def _analysis_worker(self, input_dir: str, output_dir: str, action: str):
        input_path  = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Recopilar todos los archivos soportados (case-insensitive)
        all_files = []
        for path in input_path.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                all_files.append(path)
        all_files.sort()

        if not all_files:
            self._log("⚠️  No se encontraron archivos de imagen o video en la carpeta.", "warn")
            self.after(0, self._reset_ui)
            return

        action_label = "Copiando" if action == "copy" else "Moviendo"
        self._log(f"\n{'─'*54}", "head")
        self._log(f"  📁 Entrada : {input_dir}", "head")
        self._log(f"  📂 Salida  : {output_dir}", "head")
        self._log(f"  ⚙️  Acción  : {action_label}", "head")
        self._log(f"  📊 Archivos: {len(all_files)}", "head")
        self._log(f"{'─'*54}\n", "head")

        stats = {"total": len(all_files), "with_animals": 0, "skipped": 0, "errors": 0}

        for idx, file_path in enumerate(all_files, start=1):
            if getattr(self, "_stop_requested", False):
                self._log("⏹  Análisis detenido por el usuario.", "warn")
                break

            pct = (idx / stats["total"]) * 100
            self.after(0, lambda p=pct: self.progress_var.set(p))
            self.after(0, lambda f=file_path, i=idx, t=stats["total"]:
                       self.prog_label.config(
                           text=f"[{i}/{t}]  {f.name}"))

            self._log(f"🔍 [{idx}/{stats['total']}]  {file_path.name}", "dim")

            try:
                detections = self._process_file(file_path)

                # Filtrar sólo animales
                animal_dets = [d for d in detections if d.get("class_name") == "animal"]

                if animal_dets:
                    stats["with_animals"] += 1
                    species_folder, top_conf = self._resolve_species(animal_dets)

                    dest_dir = output_path / species_folder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / file_path.name

                    # Evitar colisiones de nombre
                    dest = self._unique_dest(dest)

                    if action == "copy":
                        shutil.copy2(str(file_path), str(dest))
                        verb = "COPIADO"
                    else:
                        shutil.move(str(file_path), str(dest))
                        verb = "MOVIDO"

                    self._log(
                        f"   🐾  {species_folder}  ({top_conf:.0%} conf.)  →  ✅ {verb}",
                        "ok",
                    )
                else:
                    stats["skipped"] += 1
                    self._log(f"   ⬜  Sin animales detectados", "dim")

            except Exception as exc:
                stats["errors"] += 1
                self._log(f"   ❌  Error: {exc}", "error")

        # Resumen final
        self._log(f"\n{'─'*54}", "head")
        self._log("  ✅  ANÁLISIS COMPLETADO", "head")
        self._log(f"  Total analizados : {stats['total']}", "head")
        self._log(f"  Con animales      : {stats['with_animals']}  🐾", "ok")
        self._log(f"  Sin animales      : {stats['skipped']}", "dim")
        self._log(f"  Errores           : {stats['errors']}", "error" if stats["errors"] else "dim")
        self._log(f"{'─'*54}\n", "head")

        self.after(0, lambda: self.prog_label.config(
            text=(f"✅  Listo — {stats['with_animals']} archivos con animales "
                  f"de {stats['total']} analizados")))
        self.after(0, self._reset_ui)

        messagebox.showinfo(
            "Análisis completado",
            f"Proceso terminado.\n\n"
            f"  Archivos con animales : {stats['with_animals']}\n"
            f"  Sin animales          : {stats['skipped']}\n"
            f"  Errores               : {stats['errors']}\n\n"
            f"Resultados guardados en:\n{output_dir}",
        )

    # -----------------------------------------------------------------------
    # Procesar un archivo (imagen o video)
    # -----------------------------------------------------------------------
    def _process_file(self, file_path: Path):
        ext = file_path.suffix.lower()
        detections = []

        if ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(file_path))
            if frame is None:
                raise ValueError("No se pudo leer la imagen (formato no válido o archivo corrupto).")
            detections = self.detector.detect_and_crop(frame, file_path.stem, timestamp=0.0)

        elif ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el video.")

            fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Muestrear hasta 3 frames distribuidos uniformemente
            n_samples     = min(3, max(1, frame_count))
            interval      = frame_count / (n_samples + 1)
            frames_to_check = [int(interval * i) for i in range(1, n_samples + 1)]

            for frame_idx in frames_to_check:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                timestamp = round(frame_idx / fps, 2)
                dets = self.detector.detect_and_crop(frame, file_path.stem, timestamp=timestamp)
                detections.extend(dets)

                # Salida temprana si ya confirmamos un animal
                if any(d.get("class_name") == "animal" for d in dets):
                    break

            cap.release()

        return detections

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _resolve_species(animal_detections: list) -> Tuple[str, float]:
        """Devuelve (nombre_carpeta_especie, confianza) del mejor resultado."""
        best_name = "animal_desconocido"
        best_conf = 0.0

        for det in animal_detections:
            species_list = det.get("species", [])
            if species_list:
                top = species_list[0]
                if top["confidence"] > best_conf:
                    best_conf = top["confidence"]
                    raw = top["species"]
                    # Formatear nombre para usarlo como carpeta segura
                    folder = raw.replace(";", "_").replace(" ", "_")
                    folder = "".join(c for c in folder if c.isalnum() or c in "_-")
                    best_name = folder or "animal_desconocido"

        return best_name, best_conf

    @staticmethod
    def _unique_dest(dest: Path) -> Path:
        """Si el destino ya existe, añade un sufijo numérico."""
        if not dest.exists():
            return dest
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while True:
            new_dest = dest.parent / f"{stem}_{counter}{suffix}"
            if not new_dest.exists():
                return new_dest
            counter += 1

    def _reset_ui(self):
        self.is_running = False
        self.start_btn.config(state="normal", text="▶   Iniciar Análisis")
        self.stop_btn.config(state="disabled")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    app = WildlifeDesktopApp()
    app.mainloop()
