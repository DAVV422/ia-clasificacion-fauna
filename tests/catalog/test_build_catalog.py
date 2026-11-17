"""HU-IA-19 · El catálogo completo (catalog/speciesnet_catalog.json) y su coherencia con la votación."""

import json
import os
from pathlib import Path

import pytest

from tools import build_catalog
from wwm_pipeline.catalog_schema import Catalog
from wwm_pipeline.enums import RANK_ORDER
from wwm_pipeline.taxonomy import TaxonomyIndex

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "catalog" / "speciesnet_catalog.json"
SAMPLE = ROOT / "tests" / "fixtures" / "catalog" / "speciesnet_catalog.sample.json"


def labels_file():
    try:
        return build_catalog.default_labels()
    except (OSError, KeyError, ValueError):
        return None


LABELS = labels_file()
needs_labels = pytest.mark.skipif(LABELS is None or not LABELS.is_file(), reason="requiere las etiquetas de SpeciesNet")


@pytest.fixture(scope="module")
def catalog():
    return Catalog.model_validate(json.loads(CATALOG.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# CA-1 · Completo y válido
# ---------------------------------------------------------------------------
def test_catalog_complete_and_valid(catalog):
    assert catalog.count == len(catalog.taxa) > 3000
    assert catalog.classifier == "speciesnet-4.0.2a"
    assert len({t.label_id for t in catalog.taxa}) == catalog.count  # sin repetidos
    assert catalog.sha256 == build_catalog.catalog_sha256(catalog.taxa)
    ranks = {t.rank.value for t in catalog.taxa}
    assert ranks == {"class", "order", "family", "genus", "species"}


@needs_labels
def test_every_model_label_is_in_catalog(catalog):
    ids = {t.label_id for t in catalog.taxa}
    index = TaxonomyIndex.from_file(LABELS, common_names_es={})
    assert set(index.by_id) <= ids


# ---------------------------------------------------------------------------
# CA-2 · Coherente con lo que puede devolver /detect (la votación sube de nivel)
# ---------------------------------------------------------------------------
@needs_labels
def test_every_vote_label_in_catalog(catalog):
    """Para cada etiqueta y cada nivel al que la votación puede subir, el label_id está en el catálogo."""
    ids = {t.label_id for t in catalog.taxa}
    index = TaxonomyIndex.from_file(LABELS, common_names_es={})
    for taxon in list(index.by_id.values()):
        for rank in RANK_ORDER[: RANK_ORDER.index(taxon.rank)]:
            assert index.ancestor(taxon, rank).label_id in ids, (taxon.scientific_name, rank)


# ---------------------------------------------------------------------------
# CA-3 · Estable
# ---------------------------------------------------------------------------
@needs_labels
def test_build_is_stable(catalog):
    again = build_catalog.build(LABELS)
    assert again.sha256 == catalog.sha256 and again.count == catalog.count


# ---------------------------------------------------------------------------
# CA-5 · La muestra (la que usan ia-fake, Job y Miguel) es un subconjunto exacto
# ---------------------------------------------------------------------------
def test_sample_is_subset_of_full(catalog):
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    by_id = {t.label_id: t.model_dump(by_alias=True) for t in catalog.taxa}
    for taxon in sample["taxa"]:
        assert by_id.get(taxon["label_id"]) == taxon, taxon["scientific_name"]


def test_catalog_file_is_not_huge():
    assert CATALOG.stat().st_size < 3 * 1024 * 1024
    assert os.path.getsize(CATALOG) > 100 * 1024
