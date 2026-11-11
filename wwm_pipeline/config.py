"""Configuración del pipeline y presets.

- ``legacy_api`` / ``legacy_desktop`` (fase 0, HU-IA-40): reproducen exactamente el prototipo v1 y
  quedan como referencia de la línea base. **No se modifican.**
    - ``legacy_api``: 3 frames (25/50/75 %), **sin** salida temprana (lo que hacía ``app/main.py``).
    - ``legacy_desktop``: 3 frames con salida temprana al primer animal (lo que hacía ``desktop_app.py``).
- ``web()`` / ``desktop()`` (fase 1): muestreo denso por tiempo (HU-IA-11), sin salida temprana y
  con el mapa de clases de MegaDetector corregido. La votación por archivo llega con HU-IA-13; hasta
  entonces la decisión usa la regla del prototipo sobre todos los frames analizados.

Los umbrales son valores iniciales conservadores: no se midieron (F1-D11) y se ajustan con el
*feedback* de las pruebas en campo (F1-D12).
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"})


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    sampling: str = "legacy3"  # "legacy3" (presets legacy_*) | "dense" (HU-IA-11)
    legacy_mode: str = "desktop"  # cómo calcula el prototipo los 3 frames: "desktop" | "api"
    early_exit: bool = True  # detenerse en el primer frame con animal (solo legacy)
    md_threshold: float = 0.30  # umbral de MegaDetector
    correct_class_map: bool = False  # False = tabla del prototipo (persona→animal, vehículo→persona)
    save_crops: bool = False  # F0-D3: solo la API los guarda, hasta IA-05
    crops_dir: Optional[Path] = None
    crops_url_prefix: str = "/detections"
    # --- muestreo denso (HU-IA-11); sin efecto con sampling="legacy3" ---
    head_window_seconds: float = 2.0
    head_fps: float = 15.0
    sampling_fps: float = 6.0
    max_frames: int = 120
    max_classified_frames: int = 48  # presupuesto de SpeciesNet (F1-D2)
    max_video_seconds: Optional[float] = None  # None = sin límite; si no, se analiza hasta ahí (F1-D10)
    dup_threshold: float = 0.001  # fracción de píxeles que cambian por debajo de la cual es duplicado
    blur_ratio: float = 0.35  # borroso si la nitidez < blur_ratio × la mediana del video
    max_gap_seconds: float = 1.0  # nunca se descartan frames durante más de este tiempo seguido
    classify_crop_max_px: int = 640  # lado mayor del recorte que se guarda para clasificar (SpeciesNet usa 480)
    # F1-D9: "en" (por defecto, los nombres de SpeciesNet) | "es" (tabla de ~135 especies de Bolivia; las
    # demás siguen en inglés aunque se elija español).
    common_name_lang: str = "en"

    def with_crops(self, crops_dir: Path, url_prefix: str = "/detections") -> "PipelineConfig":
        return replace(self, save_crops=True, crops_dir=Path(crops_dir), crops_url_prefix=url_prefix)

    @property
    def sampling_params(self):
        from .sampling import SamplingParams

        return SamplingParams(
            head_window_seconds=self.head_window_seconds,
            head_fps=self.head_fps,
            sampling_fps=self.sampling_fps,
            max_frames=self.max_frames,
            max_video_seconds=self.max_video_seconds,
            dup_threshold=self.dup_threshold,
            blur_ratio=self.blur_ratio,
            max_gap_seconds=self.max_gap_seconds,
        )

    @property
    def sampling_label(self) -> str:
        """Texto de ``model.sampling``: ``15fps@0-2s+6fps/max120`` (o ``legacy3``)."""
        return self.sampling_params.label if self.sampling == "dense" else "legacy3"

    @classmethod
    def legacy_desktop(cls) -> "PipelineConfig":
        return cls(name="legacy_desktop", legacy_mode="desktop", early_exit=True)

    @classmethod
    def legacy_api(cls) -> "PipelineConfig":
        return cls(name="legacy_api", legacy_mode="api", early_exit=False)

    @classmethod
    def web(cls) -> "PipelineConfig":
        """Preset de la API web: 2 s a 10 fps + 4 fps, tope 60, presupuesto 24; hasta el segundo 15 (F1-D10)."""
        return cls(
            name="web",
            sampling="dense",
            early_exit=False,
            correct_class_map=True,
            head_window_seconds=2.0,
            head_fps=10.0,
            sampling_fps=4.0,
            max_frames=60,
            max_classified_frames=24,
            max_video_seconds=15.0,
        )

    @classmethod
    def desktop(cls) -> "PipelineConfig":
        """Preset de escritorio (modo precisión): 2 s a 15 fps + 6 fps, tope 120, presupuesto 48, sin límite."""
        return cls(
            name="desktop",
            sampling="dense",
            early_exit=False,
            correct_class_map=True,
            head_window_seconds=2.0,
            head_fps=15.0,
            sampling_fps=6.0,
            max_frames=120,
            max_classified_frames=48,
            max_video_seconds=None,
        )
