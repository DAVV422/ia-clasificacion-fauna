"""Taxonomía completa de cada resultado (HU-IA-14).

``TaxonomyIndex`` se arma **una vez** con las etiquetas del modelo cargado (``ModelRegistry`` se las
pasa; no hay rutas fijas) y responde:

- ``to_taxonomy(label_raw)`` → ``contract.Taxonomy`` (clase, orden, familia, género, especie,
  nombre científico, nombre común y ``rank``);
- ``ancestor(taxon, rank)`` → el género o la familia de un taxón, para la subida de nivel de HU-IA-13.
  Si el modelo no tiene una etiqueta propia para ese ancestro, se **sintetiza** con un ``label_id``
  ``uuid5`` determinista (F1-D7): el pipeline, el catálogo (IA-19) y el back obtienen el mismo id;
- ``is_person`` (*Homo sapiens*) e ``is_non_taxon`` (``blank``, ``vehicle``), para no votarlos como fauna.

**Nombres comunes (F1-D9):** en inglés por defecto (los de SpeciesNet). Con ``lang="es"`` se usa la
tabla ``data/common_names_es.csv``, que cubre **solo ~150 especies** de Bolivia; las demás siguen en
inglés aunque se pida español.
"""

import csv
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .catalog_schema import CatalogTaxon, parse_label, scientific_name_for
from .contract import Alternative, Taxonomy
from .enums import RANK_ORDER, Rank

# No cambiar: los ids sintéticos de los ancestros dependen de este namespace.
NAMESPACE_WWM = uuid.UUID("6f1c2a4e-9b7d-5e3f-8a21-0c4d5e6f7a8b")
COMMON_NAMES_ES = Path(__file__).resolve().parent / "data" / "common_names_es.csv"
LANGS = ("en", "es")
_SUFFIX = {Rank.GENUS: "species", Rank.FAMILY: "family", Rank.ORDER: "order", Rank.CLASS: "class"}
NON_TAXON_COMMON = {"blank", "vehicle"}

Levels = Tuple[Optional[str], ...]


def synthetic_label_id(rank: Rank, levels: Levels) -> str:
    key = f"{rank.value}:" + ";".join(v or "" for v in levels)
    return str(uuid.uuid5(NAMESPACE_WWM, key))


def _levels(taxon: CatalogTaxon) -> Levels:
    return (taxon.class_, taxon.order, taxon.family, taxon.genus, taxon.species)


def label_raw_of(taxon: CatalogTaxon) -> str:
    levels = ["" if v is None else v for v in _levels(taxon)]
    return ";".join([taxon.label_id, *levels, taxon.common_name or ""])


def load_common_names_es(path: Path = COMMON_NAMES_ES) -> Dict[str, str]:
    """``label_id`` → nombre común en español."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {
            row["label_id"]: row["common_name_es"].strip()
            for row in csv.DictReader(fh)
            if row["common_name_es"].strip()
        }


class TaxonomyIndex:
    def __init__(self, labels: Iterable[str] = (), common_names_es: Optional[Dict[str, str]] = None):
        self.by_id: Dict[str, CatalogTaxon] = {}
        self._raw: Dict[str, str] = {}  # label_id → la etiqueta tal como la da el modelo
        self._by_levels: Dict[Tuple[Rank, Levels], CatalogTaxon] = {}
        self.non_taxon_ids = set()
        for raw in labels:
            raw = raw.strip()
            if not raw:
                continue
            taxon = parse_label(raw)
            if taxon is None:
                self.non_taxon_ids.add(raw.split(";")[0])
                continue
            self._raw[taxon.label_id] = raw
            self._add(taxon)
        self.common_names_es = load_common_names_es() if common_names_es is None else common_names_es

    @classmethod
    def from_file(cls, path: Path, common_names_es: Optional[Dict[str, str]] = None) -> "TaxonomyIndex":
        return cls(Path(path).read_text(encoding="utf-8").splitlines(), common_names_es)

    def _add(self, taxon: CatalogTaxon) -> None:
        self.by_id[taxon.label_id] = taxon
        self._by_levels.setdefault((taxon.rank, self._key(taxon.rank, _levels(taxon))), taxon)

    @staticmethod
    def _key(rank: Rank, levels: Levels) -> Levels:
        depth = RANK_ORDER.index(rank)
        return tuple(levels[: depth + 1])

    def __len__(self) -> int:
        return len(self.by_id)

    # --- búsqueda ---------------------------------------------------------
    def lookup(self, label_raw: str) -> Optional[CatalogTaxon]:
        """El taxón de una etiqueta cruda (o de un ``label_id``). ``None`` si no es fauna (blank, vehicle)."""
        label_id = label_raw.split(";")[0].strip()
        if label_id in self.by_id:
            return self.by_id[label_id]
        if ";" not in label_raw:
            return None
        return parse_label(label_raw)

    def find(self, rank: Rank, scientific_name: str) -> Optional[CatalogTaxon]:
        target = scientific_name.strip().lower()
        for (r, _), taxon in self._by_levels.items():
            if r is rank and taxon.scientific_name.lower() == target:
                return taxon
        return None

    def ancestor(self, taxon: CatalogTaxon, rank: Rank) -> CatalogTaxon:
        """El ancestro de ``taxon`` en ``rank`` (o el propio taxón si ya es de ese nivel o superior)."""
        if RANK_ORDER.index(taxon.rank) <= RANK_ORDER.index(rank):
            return taxon
        levels = _levels(taxon)
        key = self._key(rank, levels)
        found = self._by_levels.get((rank, key))
        if found is not None:
            return found
        depth = RANK_ORDER.index(rank)
        full = tuple(levels[: depth + 1]) + (None,) * (len(levels) - depth - 1)
        name = scientific_name_for(list(full), rank)
        synthetic = CatalogTaxon.model_validate(
            {
                "label_id": synthetic_label_id(rank, key),
                "rank": rank,
                "class": full[0],
                "order": full[1],
                "family": full[2],
                "genus": full[3],
                "species": full[4],
                "scientific_name": name,
                "common_name": f"{name.lower()} {_SUFFIX[rank]}",
            }
        )
        self._add(synthetic)
        return synthetic

    # --- clasificación de etiquetas -----------------------------------------
    def is_non_taxon(self, label_raw: str) -> bool:
        parts = label_raw.split(";")
        if parts[0].strip() in self.non_taxon_ids:
            return True
        return (
            len(parts) == 7 and not any(p.strip() for p in parts[1:6]) and parts[6].strip().lower() in NON_TAXON_COMMON
        )

    def is_person(self, label_raw: str) -> bool:
        taxon = self.lookup(label_raw)
        return taxon is not None and taxon.genus == "homo" and taxon.species == "sapiens"

    # --- contrato -----------------------------------------------------------
    def common_name(self, taxon: CatalogTaxon, lang: str = "en") -> Optional[str]:
        if lang not in LANGS:
            raise ValueError(f"common_name_lang debe ser uno de {LANGS}, no {lang!r}")
        if lang == "es" and taxon.label_id in self.common_names_es:
            return self.common_names_es[taxon.label_id]
        return taxon.common_name

    def taxonomy_of(self, taxon: CatalogTaxon, lang: str = "en") -> Taxonomy:
        return Taxonomy.model_validate(
            {
                "label_id": taxon.label_id,
                "label_raw": self._raw.get(taxon.label_id) or label_raw_of(taxon),
                "class": taxon.class_,
                "order": taxon.order,
                "family": taxon.family,
                "genus": taxon.genus,
                "species": taxon.species,
                "scientific_name": taxon.scientific_name,
                "common_name": self.common_name(taxon, lang),
                "rank": taxon.rank,
            }
        )

    def to_taxonomy(self, label_raw: str, lang: str = "en") -> Optional[Taxonomy]:
        taxon = self.lookup(label_raw)
        return None if taxon is None else self.taxonomy_of(taxon, lang)

    def to_alternative(self, taxon: CatalogTaxon, score: float, lang: str = "en") -> Alternative:
        return Alternative(
            label_id=taxon.label_id,
            scientific_name=taxon.scientific_name,
            common_name=self.common_name(taxon, lang),
            rank=taxon.rank,
            score=round(max(0.0, min(1.0, score)), 4),
        )

    def taxa(self) -> List[CatalogTaxon]:
        return list(self.by_id.values())
