"""Carga de los modelos (MegaDetector + SpeciesNet), una sola vez y desde rutas recibidas.

``ModelRegistry`` no usa rutas fijas ni lee variables de entorno: quien lo crea (la API o el
escritorio) le pasa dónde están los pesos.

Sin conexión (HU-DS-11): MegaDetector se carga con el código de YOLOv5 copiado en
``third_party/yolov5_hub`` (``torch.hub.load(..., source="local")``: el mismo código que bajaba
``torch.hub``, sin bajarlo), y SpeciesNet desde una carpeta local con su ``info.json``. Ninguna de las dos
cargas sale a internet. (El paquete ``yolov5`` de pip no sirve: al cargar consulta Hugging Face y GitHub.)

Interfaz que usa el pipeline (y que imitan los modelos falsos de los tests):
- ``run_detector(frame_rgb) -> List[RawBox]``
- ``run_classifier(crop_pil, filepath) -> dict | None``  (resultado crudo de SpeciesNet)
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Union

from .errors import PipelineError

logger = logging.getLogger("wwm_pipeline.models")

YOLOV5_HUB_DIR = Path("third_party") / "yolov5_hub"


def default_yolov5_dir() -> Path:
    """El código de YOLOv5 copiado en el repo (o dentro del ejecutable de PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / YOLOV5_HUB_DIR


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
        classifier_model: Union[str, Path],
        device: Optional[str] = None,
        md_threshold: float = 0.30,
        yolov5_dir: Optional[Path] = None,
    ):
        """``classifier_model``: la carpeta local de SpeciesNet (con ``info.json``). La API puede pasar todavía
        un identificador de Kaggle, pero el escritorio siempre pasa una carpeta (HU-DS-11)."""
        self.detector_path = Path(detector_path)
        self.classifier_model = str(classifier_model)
        self.yolov5_dir = Path(yolov5_dir) if yolov5_dir is not None else default_yolov5_dir()
        self.md_threshold = md_threshold
        self._device = device
        self.detector: Any = None
        self.classifier: Any = None
        self.taxonomy: Any = None  # TaxonomyIndex con las etiquetas del modelo cargado (HU-IA-14)
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
        if self.taxonomy is None:
            from .taxonomy import TaxonomyIndex

            self.taxonomy = TaxonomyIndex(self.classifier.labels.values())
        return self

    def _load_megadetector(self):
        import torch

        path = str(self.detector_path.resolve())
        hub_dir = self.yolov5_dir.resolve()
        logger.info("Cargando MegaDetector desde %s (YOLOv5 local: %s)", path, hub_dir)
        if not (hub_dir / "hubconf.py").is_file():
            raise PipelineError("MODEL_LOAD_FAILED", f"No está el código de YOLOv5 en {hub_dir}")
        model = torch.hub.load(str(hub_dir), "custom", path=path, device=self.device, source="local")
        # hubconf atrapa los errores de DetectMultiBackend y devuelve un modelo crudo, sin AutoShape (que espera
        # tensores, no imágenes): mejor fallar al cargar que con un error raro en el primer frame.
        if type(model).__name__ != "AutoShape":
            raise PipelineError("MODEL_LOAD_FAILED", "YOLOv5 no pudo cargar MegaDetector (falta código de YOLOv5)")
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
