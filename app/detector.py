"""Compatibilidad: ``WildlifeDetector`` delega en ``wwm_pipeline`` (HU-IA-40).

La lógica de detección y clasificación ya no vive aquí: está en ``wwm_pipeline`` (detection,
classification, pipeline). Esta clase solo mantiene la interfaz que usan ``app/main.py`` y
``desktop_app.py`` (``load_model``, ``model``, ``classifier``, ``device``) y expone ``models``
(``ModelRegistry``) para ``classify_file``.

La copia congelada del detector original está en ``eval/legacy/detector_legacy.py``.
"""

import logging
import os
from pathlib import Path

from app.config import CONFIDENCE_THRESHOLD, MEGADETECTOR_PATH, SPECIESNET_MODEL_NAME
from wwm_pipeline.models import ModelRegistry

# torch.hub/ultralytics instalan paquetes por su cuenta al cargar YOLOv5 si no se desactiva
# (hallazgo de la fase 0; la solución de fondo es HU-DS-11).
os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

logger = logging.getLogger("uvicorn.error")


class WildlifeDetector:
    def __init__(self, detector_path: Path = MEGADETECTOR_PATH):
        self.models = ModelRegistry(Path(detector_path), SPECIESNET_MODEL_NAME, md_threshold=CONFIDENCE_THRESHOLD)
        self.device = self.models.device
        logger.info(f"Using device: {self.device} for inference.")

    @property
    def model(self):
        return self.models.detector

    @property
    def classifier(self):
        return self.models.classifier

    def load_model(self):
        self.models.load()
