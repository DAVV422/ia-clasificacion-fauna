"""Etapa de detección: MegaDetector sobre un frame → cajas (animal, persona, vehículo).

Extraído sin cambios de ``WildlifeDetector.detect_and_crop`` (prototipo v1).
"""

from dataclasses import dataclass
from typing import List, Tuple

from .models import Models

# Tabla del prototipo v1. ⚠️ ERRÓNEA para MegaDetector v5a, cuyos índices son
# 0 = animal, 1 = person, 2 = vehicle: con esta tabla, persona → "animal" y vehículo → "person".
# Se conserva solo para el preset legacy (equivalencia con el prototipo); la corrección es
# ``correct_class_map=True`` (se activa en la fase 1).
MEGADETECTOR_CLASSES = {"1": "animal", "2": "person", "3": "vehicle", "0": "animal"}
_DEFAULT_MAP = {0: "animal", 1: "person", 2: "vehicle"}
# Tabla correcta para MegaDetector v5a (model.names = {0: 'animal', 1: 'person', 2: 'vehicle'}).
MDV5_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}


@dataclass(frozen=True)
class Box:
    category: str                         # animal | person | vehicle
    confidence: float                     # redondeada a 4 decimales, como el prototipo
    pixels: Tuple[int, int, int, int]     # x1, y1, x2, y2 absolutos, recortados al frame
    box_norm: Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] normalizado (formato del prototipo)


def map_class_name_correct(class_id: int, class_name_str: str) -> str:
    """Por el nombre que da el modelo y, si no, por el índice de MegaDetector v5a."""
    name = class_name_str.lower().strip()
    if name in ("animal", "person", "vehicle"):
        return name
    return MDV5_CLASSES.get(class_id, name)


def map_class_name(class_id: int, class_name_str: str) -> str:
    """Mapeo del prototipo v1 (con el error descrito arriba)."""
    class_str = str(class_id)
    if class_str in MEGADETECTOR_CLASSES:
        return MEGADETECTOR_CLASSES[class_str]
    clean_name = class_name_str.lower().strip()
    if clean_name in MEGADETECTOR_CLASSES:
        return MEGADETECTOR_CLASSES[clean_name]
    return _DEFAULT_MAP.get(class_id, clean_name)


def detect(frame_bgr, models: Models, threshold: float, correct_class_map: bool = False) -> List[Box]:
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_bgr.shape
    boxes: List[Box] = []
    for raw in models.run_detector(frame_rgb):
        conf = float(raw.confidence)
        if conf < threshold:
            continue
        mapper = map_class_name_correct if correct_class_map else map_class_name
        category = mapper(raw.class_id, raw.name)
        x1, y1, x2, y2 = int(raw.xmin), int(raw.ymin), int(raw.xmax), int(raw.ymax)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue
        box_norm = (round(y1 / height, 4), round(x1 / width, 4), round(y2 / height, 4), round(x2 / width, 4))
        boxes.append(Box(category, conf, (x1, y1, x2, y2), box_norm))
    return boxes
