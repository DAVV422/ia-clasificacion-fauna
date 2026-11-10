"""Estrategia del prototipo v1, congelada para medir la línea base (HU-IA-10, F0-D1).

Reproduce exactamente ``desktop_app.py`` (commit c6dac6e):
- imágenes: un solo análisis;
- videos: hasta 3 frames repartidos uniformemente (25/50/75 %) y salida temprana al primer animal;
- especie: la de mayor confianza entre las detecciones de animal (``_resolve_species``).

Este paquete no se importa desde ``wwm_pipeline`` ni desde ``desktop`` (está aislado).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.legacy.config_legacy import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from eval.prediction import Prediction

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from eval.legacy.detector_legacy import WildlifeDetector  # import diferido: carga torch

        _detector = WildlifeDetector()
        _detector.load_model()
    return _detector


def process_file_legacy(file_path: Path) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Copia de ``WildlifeDesktopApp._process_file``. Devuelve (detecciones, frames analizados)."""
    import cv2

    detector = _get_detector()
    ext = file_path.suffix.lower()
    detections: List[Dict[str, Any]] = []
    frames: List[int] = []

    if ext in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(file_path))
        if frame is None:
            raise ValueError("No se pudo leer la imagen (formato no válido o archivo corrupto).")
        detections = detector.detect_and_crop(frame, file_path.stem, timestamp=0.0)
        frames = [0]
    elif ext in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_samples = min(3, max(1, frame_count))
        interval = frame_count / (n_samples + 1)
        frames_to_check = [int(interval * i) for i in range(1, n_samples + 1)]
        for frame_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame_idx)
            timestamp = round(frame_idx / fps, 2)
            dets = detector.detect_and_crop(frame, file_path.stem, timestamp=timestamp)
            detections.extend(dets)
            if any(d.get("class_name") == "animal" for d in dets):
                break
        cap.release()
    else:
        raise ValueError(f"Extensión no soportada: {ext}")
    return detections, frames


def resolve_species_legacy(animal_detections: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    """Copia de ``WildlifeDesktopApp._resolve_species``, devolviendo la etiqueta sin formatear.

    Devuelve (``"género;especie;común"`` o ``None``, confianza).
    """
    best: Optional[str] = None
    best_conf = 0.0
    for det in animal_detections:
        species_list = det.get("species", [])
        if species_list:
            top = species_list[0]
            if top["confidence"] > best_conf:
                best_conf = top["confidence"]
                best = top["species"]
    return best, best_conf


def classify_legacy(file_path: Path, taxonomy_index=None) -> Prediction:
    """Clasifica un archivo con la estrategia del prototipo y la normaliza como ``Prediction``.

    ``taxonomy_index`` (``eval.taxonomy_index.TaxonomyIndex``) completa la familia, que
    la estrategia vieja no devuelve, para poder medir el acierto de familia en la línea base.
    """
    t0 = time.perf_counter()
    detections, frames = process_file_legacy(Path(file_path))
    animal = [d for d in detections if d.get("class_name") == "animal"]
    ms = int((time.perf_counter() - t0) * 1000)
    if animal:
        label, conf = resolve_species_legacy(animal)
        info = taxonomy_index.lookup_short(label) if (taxonomy_index and label) else None
        return Prediction(
            file_name=Path(file_path).name,
            category="animal",
            raw_label=label,
            confidence=round(conf, 4),
            scientific_name=info.scientific_name if info else None,
            rank=info.rank.value if info else None,
            genus=info.genus if info else None,
            family=info.family if info else None,
            frames_analyzed=len(frames),
            frames=frames,
            processing_ms=ms,
        )
    classes = {d.get("class_name") for d in detections}
    category = "person" if "person" in classes else "vehicle" if "vehicle" in classes else "blank"
    return Prediction(
        file_name=Path(file_path).name,
        category=category,
        raw_label=None,
        confidence=0.0,
        frames_analyzed=len(frames),
        frames=frames,
        processing_ms=ms,
    )
