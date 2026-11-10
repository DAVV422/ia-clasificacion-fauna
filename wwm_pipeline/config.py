"""Configuración del pipeline y presets.

Fase 0 (HU-IA-40): los presets ``legacy_*`` reproducen exactamente el prototipo v1. ``web()`` y
``desktop()`` ya existen pero, hasta la fase 1, usan las etapas *legacy*:

- ``legacy_api``: 3 frames (25/50/75 %), **sin** salida temprana (lo que hace hoy ``app/main.py``).
- ``legacy_desktop``: 3 frames con salida temprana al primer animal (lo que hace ``desktop_app.py``).

En la fase 1, IA-11 (ventana inicial densa + muestreo) e IA-13 (votación) cambian ``web()`` y
``desktop()`` sin tocar los presets ``legacy_*``, que quedan como referencia.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"})


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    sampling: str = "legacy3"          # "legacy3" (fase 0) | "dense" (fase 1, IA-11)
    legacy_mode: str = "desktop"       # cómo calcula el prototipo los 3 frames: "desktop" | "api"
    early_exit: bool = True            # detenerse en el primer frame con animal (solo legacy)
    md_threshold: float = 0.30         # umbral de MegaDetector
    correct_class_map: bool = False    # False = tabla del prototipo (persona→animal, vehículo→persona); True en la fase 1
    save_crops: bool = False           # F0-D3: solo la API los guarda, hasta IA-05
    crops_dir: Optional[Path] = None
    crops_url_prefix: str = "/detections"
    # --- fase 1 (valores neutros hasta IA-11/IA-13) ---
    sampling_fps: Optional[float] = None
    head_window_seconds: Optional[float] = None
    head_fps: Optional[float] = None
    max_frames: Optional[int] = None
    max_video_seconds: Optional[float] = None

    def with_crops(self, crops_dir: Path, url_prefix: str = "/detections") -> "PipelineConfig":
        return replace(self, save_crops=True, crops_dir=Path(crops_dir), crops_url_prefix=url_prefix)

    @classmethod
    def legacy_desktop(cls) -> "PipelineConfig":
        return cls(name="legacy_desktop", legacy_mode="desktop", early_exit=True)

    @classmethod
    def legacy_api(cls) -> "PipelineConfig":
        return cls(name="legacy_api", legacy_mode="api", early_exit=False)

    @classmethod
    def web(cls) -> "PipelineConfig":
        """Preset de la API web. Fase 0: igual a ``legacy_api``. Fase 1: ventana densa + 4 fps."""
        return replace(cls.legacy_api(), name="web")

    @classmethod
    def desktop(cls) -> "PipelineConfig":
        """Preset de escritorio (modo precisión). Fase 0: igual a ``legacy_desktop``."""
        return replace(cls.legacy_desktop(), name="desktop")
