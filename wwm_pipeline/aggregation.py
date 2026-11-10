"""Agregación por archivo. Fase 0: la regla del prototipo (``_resolve_species``).

La fase 1 (IA-13) agrega la votación ponderada entre frames.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class Decision:
    category: str  # animal | person | vehicle | blank
    label: Optional[str]  # "género;especie;común" (formato del prototipo) o None
    confidence: float


def legacy_best(detections: Iterable[Dict[str, Any]]) -> Decision:
    """La especie de mayor confianza entre las detecciones de animal (primera especie de cada una)."""
    dets = list(detections)
    animal = [d for d in dets if d.get("class_name") == "animal"]
    if not animal:
        classes = {d.get("class_name") for d in dets}
        category = "person" if "person" in classes else "vehicle" if "vehicle" in classes else "blank"
        return Decision(category, None, 0.0)
    best: Optional[str] = None
    best_conf = 0.0
    for det in animal:
        species_list = det.get("species", [])
        if species_list:
            top = species_list[0]
            if top["confidence"] > best_conf:
                best_conf = top["confidence"]
                best = top["species"]
    return Decision("animal", best, best_conf)
