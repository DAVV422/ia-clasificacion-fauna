"""Carga de los modelos (MegaDetector + SpeciesNet), una sola vez y desde rutas recibidas.

``ModelRegistry`` no usa rutas fijas ni lee variables de entorno: quien lo crea (la API o el
escritorio) le pasa dónde están los pesos. En la fase 0 se mantiene ``torch.hub`` para cargar
YOLOv5 (igual que el prototipo); el modo offline sin ``torch.hub`` es HU-DS-11.

Interfaz que usa el pipeline (y que imitan los modelos falsos de los tests):
- ``run_detector(frame_rgb) -> List[RawBox]``
- ``run_classifier(crop_pil, filepath) -> dict | None``  (resultado crudo de SpeciesNet)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol

logger = logging.getLogger("wwm_pipeline.models")

SPECIESNET_MODEL_NAME = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"


@dataclass(frozen=True)
class RawBox:
    """Una fila de la salida de MegaDetector (``results.pandas().xyxy[0]``)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    name: str


class Models(Protocol):
    def run_detector(self, frame_rgb: Any) -> List[RawBox]: ...

    def run_classifier(self, crop_pil: Any, filepath: str) -> Optional[dict]: ...


class ModelRegistry:
    """Modelos reales. ``load()`` es idempotente: carga cada modelo una sola vez."""

    def __init__(
        self,
        detector_path: Path,
        classifier_model: str = SPECIESNET_MODEL_NAME,
        device: Optional[str] = None,
        md_threshold: float = 0.30,
    ):
        self.detector_path = Path(detector_path)
        self.classifier_model = classifier_model
        self.md_threshold = md_threshold
        self._device = device
        self.detector: Any = None
        self.classifier: Any = None
        self.load_count = 0

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def loaded(self) -> bool:
        return self.detector is not None and self.classifier is not None

    def load(self) -> "ModelRegistry":
        if self.detector is None:
            self.detector = self._load_megadetector()
            self.load_count += 1
        if self.classifier is None:
            from speciesnet.classifier import SpeciesNetClassifier

            logger.info("Cargando SpeciesNet %s", self.classifier_model)
            self.classifier = SpeciesNetClassifier(model_name=self.classifier_model, device=self.device)
        return self

    def _load_megadetector(self):
        import torch

        path = str(self.detector_path.resolve())
        logger.info("Cargando MegaDetector desde %s", path)
        try:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=path, device=self.device, force_reload=False, trust_repo=True
            )
        except Exception as exc:  # mismo reintento que el prototipo (fase 0)
            logger.warning("torch.hub falló (%s); reintentando con force_reload", exc)
            model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=path, device=self.device, force_reload=True, trust_repo=True
            )
        model.conf = self.md_threshold
        return model

    def run_detector(self, frame_rgb) -> List[RawBox]:
        self.load()
        df = self.detector(frame_rgb).pandas().xyxy[0]  # type: ignore[misc]
        return [
            RawBox(
                float(r["xmin"]),
                float(r["ymin"]),
                float(r["xmax"]),
                float(r["ymax"]),
                float(r["confidence"]),
                int(r["class"]),
                str(r["name"]),
            )
            for _, r in df.iterrows()
        ]

    def run_classifier(self, crop_pil, filepath: str) -> Optional[dict]:
        self.load()
        classifier: Any = self.classifier
        preprocessed = classifier.preprocess(crop_pil, bboxes=None)
        if preprocessed is None:
            return None
        return classifier.predict(filepath=filepath, img=preprocessed)
