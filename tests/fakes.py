"""Modelos falsos para probar el pipeline sin torch ni pesos (misma interfaz que ModelRegistry)."""

from typing import List, Optional

from tools._fixture_taxa import FIXTURE_LABELS
from wwm_pipeline.models import RawBox
from wwm_pipeline.taxonomy import TaxonomyIndex

JAGUAR = "9b7fea59-cfa3-499c-a420-bcf36277dcd8;mammalia;carnivora;felidae;panthera;onca;jaguar"
PUMA = "9c564562-9429-405c-8529-04cff7752282;mammalia;carnivora;felidae;puma;concolor;puma"


class FakeModels:
    """Detecta un animal en los frames cuya media de brillo supera ``bright_threshold``."""

    def __init__(
        self,
        bright_threshold: float = 100.0,
        labels: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        category_id: int = 0,
    ):
        self.bright_threshold = bright_threshold
        self.labels = labels or [JAGUAR, PUMA]
        self.scores = scores or [0.91, 0.05]
        self.category_id = category_id
        self.detector_calls = 0
        self.classifier_calls = 0
        self.taxonomy = TaxonomyIndex(FIXTURE_LABELS + self.labels)

    def run_detector(self, frame_rgb) -> List[RawBox]:
        self.detector_calls += 1
        h, w = frame_rgb.shape[:2]
        if float(frame_rgb.mean()) < self.bright_threshold:
            return []
        name = {0: "animal", 1: "person", 2: "vehicle"}[self.category_id]  # model.names de MegaDetector v5a
        return [RawBox(w * 0.2, h * 0.2, w * 0.6, h * 0.7, 0.88, self.category_id, name)]

    def run_classifier(self, crop_pil, filepath: str) -> Optional[dict]:
        self.classifier_calls += 1
        return {"filepath": filepath, "classifications": {"classes": self.labels, "scores": self.scores}}
