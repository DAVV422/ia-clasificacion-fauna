"""Operaciones de archivos del escritorio.

- HU-DS-03: ``safe_folder_name`` (nombre de la carpeta por especie, válido en Windows).
- HU-DS-10 agrega ``unique_pair``; la fase 3 (HU-DS-30) ``copy_verified`` y ``move_verified``.
"""

import re
import unicodedata
from typing import Optional

from wwm_pipeline.enums import Rank

MAX_FOLDER_CHARS = 80
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
RELOCATABLE_RANKS = frozenset({Rank.SPECIES, Rank.GENUS, Rank.FAMILY})
_NOT_ALLOWED = re.compile(r"[^a-z0-9_]+")
_REPEATED = re.compile(r"_+")


def clean_part(text: Optional[str]) -> str:
    """Minúsculas, sin tildes, solo ``[a-z0-9_]``, sin ``_`` repetidos ni en los extremos."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _NOT_ALLOWED.sub("_", text)
    return _REPEATED.sub("_", text).strip("_")


def _windows_safe(name: str) -> str:
    name = name[:MAX_FOLDER_CHARS].rstrip("_. ")
    if name in WINDOWS_RESERVED:
        name += "_"
    return name


def safe_folder_name(taxonomy) -> str:
    """Nombre de la carpeta de un resultado ``animal`` (HU-DS-03, F2-D4).

    - ``species`` → ``panthera_onca_jaguar`` (el nombre común es el de ``taxonomy.common_name``, que ya viene
      en el idioma configurado: F2-D7);
    - ``genus`` → ``panthera_sp``;
    - ``family`` → ``felidae_fam``;
    - ``order`` / ``class`` → ``ValueError``: esos archivos quedan en el origen (HU-DS-02, F2-D8).
    """
    rank = Rank(taxonomy.rank)
    if rank is Rank.SPECIES:
        parts = [taxonomy.genus, taxonomy.species, taxonomy.common_name]
    elif rank is Rank.GENUS:
        parts = [taxonomy.genus, "sp"]
    elif rank is Rank.FAMILY:
        parts = [taxonomy.family, "fam"]
    else:
        raise ValueError(f"un resultado con rank={rank.value} no tiene carpeta: queda en el origen")
    name = _windows_safe("_".join(p for p in (clean_part(x) for x in parts) if p))
    if not name:
        raise ValueError(f"no se pudo armar un nombre de carpeta para {taxonomy.scientific_name!r}")
    return name
