"""Genera ``tests/fixtures/catalog/speciesnet_catalog.sample.json`` a partir de las
etiquetas reales de SpeciesNet: los taxones de las familias neotropicales más
frecuentes en cámaras trampa, más todos los usados en los fixtures.

Uso:
    python -m tools.build_sample_catalog [ruta/al/archivo.labels.txt]

Sin argumento, busca el archivo en la caché de kagglehub.
"""

import hashlib
import json
import sys
from pathlib import Path

from tools._fixture_taxa import FIXTURE_LABELS
from wwm_pipeline.catalog_schema import Catalog, parse_label

CLASSIFIER = "speciesnet-4.0.2a"
DEFAULT_LABELS = (
    Path.home()
    / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"
    / "always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt"
)
NEOTROPICAL_FAMILIES = {
    # mamíferos
    "felidae",
    "canidae",
    "mustelidae",
    "procyonidae",
    "mephitidae",
    "tapiridae",
    "tayassuidae",
    "cervidae",
    "dasyproctidae",
    "cuniculidae",
    "caviidae",
    "erethizontidae",
    "sciuridae",
    "didelphidae",
    "myrmecophagidae",
    "dasypodidae",
    "chlamyphoridae",
    "bradypodidae",
    "megalonychidae",
    "atelidae",
    "cebidae",
    "callitrichidae",
    "aotidae",
    "pitheciidae",
    # aves terrestres frecuentes en cámaras trampa
    "cracidae",
    "tinamidae",
    "psophiidae",
    "rallidae",
}
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/catalog/speciesnet_catalog.sample.json"


def build(labels_path: Path) -> Catalog:
    fixture_ids = {label.split(";")[0] for label in FIXTURE_LABELS}
    taxa = []
    for raw in labels_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        taxon = parse_label(raw)
        if taxon is None:
            continue
        if taxon.family in NEOTROPICAL_FAMILIES or taxon.label_id in fixture_ids:
            taxa.append(taxon)
    taxa.sort(key=lambda t: (t.class_ or "", t.order or "", t.family or "", t.genus or "", t.species or ""))
    payload = json.dumps([t.model_dump(by_alias=True) for t in taxa], sort_keys=True, ensure_ascii=False)
    return Catalog(
        classifier=CLASSIFIER, count=len(taxa), sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(), taxa=taxa
    )


def main() -> None:
    labels = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LABELS
    catalog = build(labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog.model_dump(by_alias=True), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{catalog.count} taxones -> {OUT}")


if __name__ == "__main__":
    main()
