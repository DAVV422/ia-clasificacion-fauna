"""Etapa de clasificación: SpeciesNet sobre el recorte de un animal.

Extraído sin cambios de ``WildlifeDetector.detect_and_crop`` (prototipo v1): devuelve la lista
``[{"species": "género;especie;común", "confidence": 0.9644}, ...]`` en el orden de SpeciesNet
(de mayor a menor puntaje). Sin filtro geográfico: se usa todo el catálogo.
"""

import logging
from typing import Dict, List, Union

from .models import Models

logger = logging.getLogger("wwm_pipeline.classification")


def short_label(raw_species: str) -> str:
    """``uuid;clase;orden;familia;género;especie;común`` → ``género;especie;común`` (formato del prototipo)."""
    parts = raw_species.split(";")
    if len(parts) >= 4:
        return ";".join(parts[-3:])
    if len(parts) > 1:
        return ";".join(parts[1:])
    return raw_species


def classify_crop(crop_bgr, models: Models, filepath: str) -> List[Dict[str, Union[str, float]]]:
    import cv2
    from PIL import Image

    species: List[Dict[str, Union[str, float]]] = []
    try:
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result = models.run_classifier(crop_pil, filepath)
        if isinstance(result, dict):
            classifications = result.get("classifications") or {}
            classes = classifications.get("classes", [])
            scores = classifications.get("scores", [])
            if classes and scores:
                for raw, score in zip(classes, scores):
                    species.append({"species": short_label(raw), "confidence": round(float(score), 4)})
            else:
                fallback = result.get("classes")
                score = result.get("confidence") or result.get("probability") or result.get("score")
                if fallback:
                    species.append({"species": fallback, "confidence": round(float(score), 4) if score else 1.0})
    except Exception as exc:  # el prototipo registra el error y sigue sin especie
        logger.error("Falló SpeciesNet sobre %s: %s", filepath, exc)
    return species
