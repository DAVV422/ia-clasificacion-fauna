"""Etapa de clasificación: SpeciesNet sobre el recorte de un animal.

Extraído sin cambios de ``WildlifeDetector.detect_and_crop`` (prototipo v1): devuelve la lista
``[{"species": "género;especie;común", "confidence": 0.9644}, ...]`` en el orden de SpeciesNet
(de mayor a menor puntaje). Sin filtro geográfico: se usa todo el catálogo.
"""

import logging
from typing import Dict, List, Tuple, Union

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


def classify_crop_raw(crop_bgr, models: Models, filepath: str) -> List[Tuple[str, float]]:
    """``[(etiqueta cruda de 7 partes, puntaje), ...]`` en el orden de SpeciesNet. Vacío si falla."""
    return _run(crop_bgr, models, filepath)[0]


def _run(crop_bgr, models: Models, filepath: str) -> Tuple[List[Tuple[str, float]], bool]:
    """(etiquetas, ``True`` si vino por la respuesta alternativa ``classes``/``confidence``)."""
    import cv2
    from PIL import Image

    labels: List[Tuple[str, float]] = []
    fallback_used = False
    try:
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result = models.run_classifier(crop_pil, filepath)
        if isinstance(result, dict):
            classifications = result.get("classifications") or {}
            classes = classifications.get("classes", [])
            scores = classifications.get("scores", [])
            if classes and scores:
                labels = [(raw, round(float(score), 4)) for raw, score in zip(classes, scores)]
            else:
                fallback = result.get("classes")
                score = result.get("confidence") or result.get("probability") or result.get("score")
                if fallback:
                    labels = [(fallback, round(float(score), 4) if score else 1.0)]
                    fallback_used = True
    except Exception as exc:  # el prototipo registra el error y sigue sin especie
        logger.error("Falló SpeciesNet sobre %s: %s", filepath, exc)
    return labels, fallback_used


def to_species_list(labels: List[Tuple[str, float]]) -> List[Dict[str, Union[str, float]]]:
    """Formato del prototipo: ``[{"species": "género;especie;común", "confidence": 0.96}, ...]``."""
    return [{"species": short_label(raw) if raw.count(";") else raw, "confidence": score} for raw, score in labels]


def classify_crop(crop_bgr, models: Models, filepath: str) -> List[Dict[str, Union[str, float]]]:
    """Resultado de SpeciesNet en el formato del prototipo (presets ``legacy_*``)."""
    labels, fallback_used = _run(crop_bgr, models, filepath)
    if fallback_used:  # la respuesta alternativa se guarda tal cual, como hacía el prototipo
        return [{"species": labels[0][0], "confidence": labels[0][1]}]
    return [{"species": short_label(raw), "confidence": score} for raw, score in labels]
