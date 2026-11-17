"""Genera ``catalog/speciesnet_catalog.json``: el catálogo **completo** de SpeciesNet (HU-IA-19, IA §4).

- **Todas** las etiquetas taxonómicas del modelo (``parse_label``; se excluyen ``blank``, ``vehicle``…).
- **Más sus ancestros** (clase, orden, familia y género) que el modelo no trae como etiqueta propia, con el mismo
  ``label_id`` sintético que usa la votación al subir de nivel (``TaxonomyIndex.ancestor``). Así cualquier
  ``taxonomy.label_id`` que devuelva ``POST /detect`` está en el catálogo, por construcción.
- Orden y ``sha256`` estables: mismo método que la muestra (``tools/build_sample_catalog.py``), así la muestra es un
  subconjunto exacto del completo y el back puede verificar su copia (BE-25).

Uso:
    python tools/build_catalog.py                      # etiquetas de MODELS_DIR (o de la carpeta de la app)
    python tools/build_catalog.py <ruta/al/.labels.txt>
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wwm_pipeline.catalog_schema import Catalog, CatalogTaxon  # noqa: E402
from wwm_pipeline.enums import RANK_ORDER  # noqa: E402
from wwm_pipeline.taxonomy import TaxonomyIndex  # noqa: E402

OUT = ROOT / "catalog" / "speciesnet_catalog.json"
SPECIESNET_DIR = "speciesnet-4.0.2a"


def default_labels() -> Path:
    base = Path(os.getenv("MODELS_DIR") or Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "WWM" / "models")
    info = json.loads((base / SPECIESNET_DIR / "info.json").read_text(encoding="utf-8"))
    return base / SPECIESNET_DIR / info["classifier_labels"]


def sort_key(taxon: CatalogTaxon):
    levels = (taxon.class_ or "", taxon.order or "", taxon.family or "", taxon.genus or "", taxon.species or "")
    return levels + (RANK_ORDER.index(taxon.rank), taxon.label_id)


def catalog_sha256(taxa: List[CatalogTaxon]) -> str:
    payload = json.dumps([t.model_dump(by_alias=True) for t in taxa], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(labels_path: Path, classifier: str = "speciesnet-4.0.2a") -> Catalog:
    index = TaxonomyIndex.from_file(labels_path, common_names_es={})
    for taxon in list(index.by_id.values()):  # los ancestros que el modelo no trae como etiqueta
        depth = RANK_ORDER.index(taxon.rank)
        for rank in RANK_ORDER[:depth]:
            index.ancestor(taxon, rank)
    taxa = sorted(index.by_id.values(), key=sort_key)
    return Catalog(classifier=classifier, count=len(taxa), sha256=catalog_sha256(taxa), taxa=taxa)


def main(argv: Optional[list] = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    labels = Path(args[0]) if args else default_labels()
    catalog = build(labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog.model_dump(by_alias=True), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by_rank = {}
    for t in catalog.taxa:
        by_rank[t.rank.value] = by_rank.get(t.rank.value, 0) + 1
    print(f"{catalog.count} taxones {by_rank} -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
