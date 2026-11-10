"""Catálogo de especies de SpeciesNet y conversión de etiquetas.

Formato de una etiqueta de SpeciesNet:
``<uuid>;<clase>;<orden>;<familia>;<género>;<especie>;<nombre común>``.
Las etiquetas sin ningún nivel taxonómico (``blank``, ``vehicle``…) no son taxones.

Solo depende de pydantic y de la biblioteca estándar.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import RANK_ORDER, Rank


class CatalogTaxon(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    label_id: str = Field(min_length=1)
    rank: Rank
    class_: Optional[str] = Field(default=None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    scientific_name: str = Field(min_length=1)
    common_name: Optional[str] = None


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifier: str
    count: int = Field(ge=0)
    sha256: str
    taxa: List[CatalogTaxon]


class LabelParseError(ValueError):
    pass


def scientific_name_for(levels: List[Optional[str]], rank: Rank) -> str:
    """Nombre científico del nivel ``rank``: ``Panthera onca``, ``Panthera``, ``Felidae``…"""
    clazz, order, family, genus, species = levels
    if rank is Rank.SPECIES:
        return f"{genus.capitalize()} {species.lower()}"  # type: ignore[union-attr]
    value = {Rank.GENUS: genus, Rank.FAMILY: family, Rank.ORDER: order, Rank.CLASS: clazz}[rank]
    return value.capitalize()  # type: ignore[union-attr]


def parse_label(label_raw: str) -> Optional[CatalogTaxon]:
    """Convierte una etiqueta de SpeciesNet en ``CatalogTaxon``.

    Devuelve ``None`` para las etiquetas que no son taxones (sin ningún nivel).
    Lanza ``LabelParseError`` si el formato no es el esperado.
    """
    parts = label_raw.strip().split(";")
    if len(parts) != 7:
        raise LabelParseError(f"La etiqueta debe tener 7 partes separadas por ';': {label_raw!r}")
    label_id, *levels_raw, common = parts
    levels: List[Optional[str]] = [p.strip().lower() or None for p in levels_raw]
    present = [i for i, v in enumerate(levels) if v]
    if not present:
        return None
    depth = present[-1]
    if present != list(range(depth + 1)):
        raise LabelParseError(f"La etiqueta tiene niveles salteados: {label_raw!r}")
    rank = RANK_ORDER[depth]
    return CatalogTaxon(
        label_id=label_id.strip(),
        rank=rank,
        class_=levels[0],
        order=levels[1],
        family=levels[2],
        genus=levels[3],
        species=levels[4],
        scientific_name=scientific_name_for(levels, rank),
        common_name=common.strip() or None,
    )
