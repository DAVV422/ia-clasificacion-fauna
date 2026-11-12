"""Qué archivos se reubican y cuáles quedan en el origen (HU-DS-02, E-06, TDR §3.4)."""

import os
from pathlib import Path

from wwm_pipeline.enums import Rank

from .file_ops import RELOCATABLE_RANKS


def should_relocate(result) -> bool:
    """``True`` solo para un animal que llegó a especie, género o familia.

    Quedan en el origen: vacío, persona, vehículo, animal hasta orden o clase (F2-D8) y los errores.
    """
    if result is None or result.decision.category != "animal":
        return False
    taxonomy = result.taxonomy
    return taxonomy is not None and Rank(taxonomy.rank) in RELOCATABLE_RANKS


def is_same_or_inside(dest, source) -> bool:
    """``True`` si ``dest`` es ``source`` o está dentro de él (CB-90), sin distinguir mayúsculas."""
    d = os.path.normcase(str(Path(dest).resolve()))
    s = os.path.normcase(str(Path(source).resolve()))
    try:
        return os.path.commonpath([d, s]) == s
    except ValueError:  # unidades distintas en Windows
        return False
