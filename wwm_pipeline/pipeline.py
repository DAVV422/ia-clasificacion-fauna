"""Punto de entrada único de la clasificación: ``classify_file`` (IA-02).

Lo usan la API (``app/main.py``) y la app de escritorio. No conoce FastAPI ni Tkinter.

Fase 0: con los presets ``legacy_*`` reproduce exactamente el prototipo v1 (verificado contra
``eval/legacy/expected.json``). La fase 1 reemplaza muestreo y agregación etapa por etapa.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .aggregation import Decision, legacy_best
from .classification import classify_crop
from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PipelineConfig
from .detection import detect
from .errors import PipelineCancelled, PipelineError
from .evidence import new_crop_name, save_crop
from .models import Models
from .sampling import legacy_frames

ProgressFn = Callable[[str, int, int], None]
StopFn = Callable[[], bool]


@dataclass
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    detections: List[Dict[str, Any]]  # formato del prototipo: box, class_name, confidence, species, crop_path, crop_url


@dataclass
class FileResult:
    file_name: str
    media_type: str  # image | video
    decision: Decision
    frames: List[FrameResult] = field(default_factory=list)
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    processing_ms: int = 0

    @property
    def frames_analyzed(self) -> List[int]:
        return [f.frame_index for f in self.frames]

    @property
    def all_detections(self) -> List[Dict[str, Any]]:
        return [d for f in self.frames for d in f.detections]


def analyze_frame(frame_bgr, models: Models, config: PipelineConfig, run_name: str) -> List[Dict[str, Any]]:
    """Detección + recorte (opcional) + clasificación de un frame, con la salida del prototipo."""
    detections: List[Dict[str, Any]] = []
    for box in detect(frame_bgr, models, config.md_threshold, config.correct_class_map):
        x1, y1, x2, y2 = box.pixels
        crop = frame_bgr[y1:y2, x1:x2]
        crop_name = new_crop_name()
        crop_path = crop_url = None
        if config.save_crops and config.crops_dir is not None:
            crop_path, crop_url = save_crop(
                crop, config.crops_dir, run_name, box.category, crop_name, config.crops_url_prefix
            )
        species = classify_crop(crop, models, crop_name) if box.category == "animal" else []
        detections.append(
            {
                "box": list(box.box_norm),
                "class_name": box.category,
                "confidence": round(box.confidence, 4),
                "species": species,
                "crop_path": crop_path,
                "crop_url": crop_url,
            }
        )
    return detections


def media_type_of(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise PipelineError(
        "IA_INVALID_FORMAT", f"Extensión no soportada: {ext}", accepted=sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    )


def classify_file(
    path,
    file_name: Optional[str],
    config: PipelineConfig,
    models: Models,
    on_progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
    run_name: Optional[str] = None,
) -> FileResult:
    """Clasifica un archivo (imagen o video) y devuelve una decisión por archivo.

    ``file_name``: el nombre original (para la extensión y, en la fase 1, la fecha); por defecto, el de ``path``.
    ``run_name``: carpeta de los recortes cuando ``config.save_crops`` (por defecto, el nombre sin extensión).
    """
    import cv2

    path = Path(path)
    file_name = file_name or path.name
    run_name = run_name or Path(file_name).stem
    media = media_type_of(file_name)
    t0 = time.perf_counter()

    def check_stop() -> None:
        if should_stop is not None and should_stop():
            raise PipelineCancelled(file_name)

    if media == "image":
        frame = cv2.imread(str(path))
        if frame is None:
            raise PipelineError("IA_DECODE_FAILED", "No se pudo leer la imagen (formato no válido o archivo corrupto).")
        if on_progress:
            on_progress("detection", 1, 1)
        detections = analyze_frame(frame, models, config, run_name)
        result = FileResult(file_name, "image", legacy_best(detections), [FrameResult(0, 0.0, detections)])
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise PipelineError("IA_DECODE_FAILED", "No se pudo abrir el video.")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = legacy_frames(frame_count, config.legacy_mode)
            frames: List[FrameResult] = []
            for i, frame_idx in enumerate(indices, start=1):
                check_stop()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if on_progress:
                    on_progress("detection", i, len(indices))
                detections = analyze_frame(frame, models, config, run_name)
                frames.append(FrameResult(frame_idx, round(frame_idx / fps, 2), detections))
                if config.early_exit and any(d.get("class_name") == "animal" for d in detections):
                    break
        finally:
            cap.release()
        all_dets = [d for f in frames for d in f.detections]
        result = FileResult(
            file_name,
            "video",
            legacy_best(all_dets),
            frames,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=round(frame_count / fps, 2),
        )
    result.processing_ms = int((time.perf_counter() - t0) * 1000)
    return result
