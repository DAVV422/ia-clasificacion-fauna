"""Estrategias que se pueden evaluar con ``eval.py --config <nombre>``.

Fase 0: solo ``baseline`` (el prototipo v1 congelado).
Fase 1: se agregan ``desktop``, ``web`` y las variantes de ablación (IA-11, IA-13), que
usan ``wwm_pipeline.classify_file`` con su ``PipelineConfig``.
"""

from pathlib import Path
from typing import Callable, Dict

from eval.prediction import Prediction

Strategy = Callable[[Path], Prediction]


def baseline(taxonomy_index) -> Strategy:
    from eval.legacy import classify_legacy

    return lambda path: classify_legacy(path, taxonomy_index)


STRATEGIES: Dict[str, Callable[..., Strategy]] = {
    "baseline": baseline,
}
