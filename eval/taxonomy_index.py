"""Índice de las etiquetas de SpeciesNet para completar la taxonomía de una predicción."""

from pathlib import Path
from typing import Dict, Optional

from wwm_pipeline.catalog_schema import CatalogTaxon, parse_label

DEFAULT_LABELS = (
    Path.home() / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"
    / "always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt"
)


class TaxonomyIndex:
    def __init__(self, labels_path: Path = DEFAULT_LABELS):
        self.by_id: Dict[str, CatalogTaxon] = {}
        self.by_short: Dict[str, CatalogTaxon] = {}
        self.by_name: Dict[str, CatalogTaxon] = {}
        for raw in Path(labels_path).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            taxon = parse_label(raw)
            if taxon is None:
                continue
            parts = raw.split(";")
            self.by_id[taxon.label_id] = taxon
            # el prototipo v1 devuelve los últimos 3 campos: "género;especie;común"
            self.by_short.setdefault(";".join(parts[-3:]).lower(), taxon)
            self.by_name.setdefault(taxon.scientific_name.lower(), taxon)

    def lookup_short(self, short_label: str) -> Optional[CatalogTaxon]:
        return self.by_short.get(short_label.lower())

    def lookup_name(self, scientific_name: str) -> Optional[CatalogTaxon]:
        return self.by_name.get(scientific_name.strip().lower())
