"""HU-IA-14 · Taxonomía completa, ancestros, no-taxones e idioma del nombre común."""

import csv
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeModels
from tools._fixture_taxa import AGOUTI, DASYPROCTA_GENUS, FIXTURE_LABELS, JAGUAR, TAPIR
from wwm_pipeline.catalog_schema import parse_label
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.contract import Taxonomy
from wwm_pipeline.enums import Rank
from wwm_pipeline.taxonomy import COMMON_NAMES_ES, TaxonomyIndex, synthetic_label_id

ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = (
    Path.home()
    / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"
    / "always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt"
)
needs_labels = pytest.mark.skipif(not LABELS_FILE.exists(), reason="requiere el archivo de etiquetas de SpeciesNet")
HUMAN = "x-human;mammalia;primates;hominidae;homo;sapiens;human"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"


@pytest.fixture
def index():
    return TaxonomyIndex(FIXTURE_LABELS + [BLANK, VEHICLE], common_names_es={})


# ---------------------------------------------------------------------------
# CA-1 · Familia siempre presente (archivo completo del modelo)
# ---------------------------------------------------------------------------
@needs_labels
def test_every_label_has_family_when_needed():
    index = TaxonomyIndex.from_file(LABELS_FILE, common_names_es={})
    assert len(index) >= 2400
    checked = 0
    for taxon in index.taxa():
        tax = index.taxonomy_of(taxon)
        Taxonomy.model_validate(tax.model_dump(by_alias=True))
        if taxon.rank in (Rank.SPECIES, Rank.GENUS, Rank.FAMILY):
            assert tax.family, taxon.label_id
            checked += 1
    assert checked > 2400


# ---------------------------------------------------------------------------
# CA-2 · Nombre científico y rank
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,rank,name,family",
    [
        (JAGUAR, Rank.SPECIES, "Panthera onca", "felidae"),
        (DASYPROCTA_GENUS, Rank.GENUS, "Dasyprocta", "dasyproctidae"),
        ("837e12c2-f30c-4875-afeb-32d51938f139;aves;galliformes;;;;galliformes order", Rank.ORDER, "Galliformes", None),
    ],
)
def test_to_taxonomy_examples(index, label, rank, name, family):
    tax = index.to_taxonomy(label)
    assert tax.rank is rank and tax.scientific_name == name and tax.family == family
    assert tax.label_raw == label and tax.label_id == label.split(";")[0]
    assert tax.class_ == label.split(";")[1]  # minúsculas, niveles vacíos → null
    if rank is not Rank.SPECIES:
        assert tax.species is None


# ---------------------------------------------------------------------------
# CA-3 · Ancestros
# ---------------------------------------------------------------------------
def test_ancestor_uses_model_label_when_it_exists(index):
    agouti = index.lookup(AGOUTI)
    genus = index.ancestor(agouti, Rank.GENUS)
    assert genus.label_id == DASYPROCTA_GENUS.split(";")[0]  # "dasyprocta species" sí está en el modelo


def test_ancestor_and_synthetic_ids_are_stable(index):
    jaguar = index.lookup(JAGUAR)
    genus = index.ancestor(jaguar, Rank.GENUS)
    family = index.ancestor(jaguar, Rank.FAMILY)
    assert (genus.rank, genus.scientific_name, genus.family, genus.species) == (Rank.GENUS, "Panthera", "felidae", None)
    assert (family.rank, family.scientific_name, family.genus) == (Rank.FAMILY, "Felidae", None)
    assert uuid.UUID(genus.label_id).version == 5
    again = TaxonomyIndex([JAGUAR], common_names_es={}).ancestor(parse_label(JAGUAR), Rank.GENUS)
    assert (
        again.label_id
        == genus.label_id
        == synthetic_label_id(Rank.GENUS, ("mammalia", "carnivora", "felidae", "panthera"))
    )
    Taxonomy.model_validate(index.taxonomy_of(genus).model_dump(by_alias=True))
    assert index.ancestor(genus, Rank.FAMILY).label_id == family.label_id
    assert index.ancestor(family, Rank.GENUS) is family  # un taxón superior no "baja"


@needs_labels
def test_ancestors_of_real_species_resolve():
    index = TaxonomyIndex.from_file(LABELS_FILE, common_names_es={})
    real_genus = index.find(Rank.GENUS, "Mazama")
    gray_brocket = index.find(Rank.SPECIES, "Mazama gouazoubira")
    assert real_genus is not None and gray_brocket is not None
    assert index.ancestor(gray_brocket, Rank.GENUS).label_id == real_genus.label_id
    assert index.ancestor(gray_brocket, Rank.FAMILY).scientific_name == "Cervidae"


# ---------------------------------------------------------------------------
# CA-4 · No-taxones
# ---------------------------------------------------------------------------
def test_non_taxon_and_person(index):
    assert index.is_non_taxon(BLANK) and index.is_non_taxon(VEHICLE)
    assert index.to_taxonomy(BLANK) is None
    assert not index.is_non_taxon(JAGUAR)
    assert index.is_person(HUMAN) and not index.is_person(JAGUAR)


# ---------------------------------------------------------------------------
# CA-5 · Sin rutas fijas
# ---------------------------------------------------------------------------
def test_no_fixed_cache_paths():
    hits = [
        p.name for p in (ROOT / "wwm_pipeline").rglob("*.py") if ".cache/kagglehub" in p.read_text(encoding="utf-8")
    ]
    assert hits == []


# ---------------------------------------------------------------------------
# CA-6 · Idioma del nombre común (F1-D9)
# ---------------------------------------------------------------------------
def test_common_name_lang():
    index = TaxonomyIndex(FIXTURE_LABELS)  # con la tabla real en español
    assert index.to_taxonomy(TAPIR).common_name == "lowland tapir"  # por defecto, inglés
    assert index.to_taxonomy(TAPIR, "es").common_name == "anta"
    assert index.to_taxonomy(JAGUAR, "es").common_name == "jaguar"
    assert index.to_taxonomy(AGOUTI, "es").common_name == "red-rumped agouti"  # no está en la tabla → inglés
    with pytest.raises(ValueError):
        index.to_taxonomy(JAGUAR, "pt")


def test_es_table_shape():
    with COMMON_NAMES_ES.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert 120 <= len(rows) <= 160
    assert len({r["label_id"] for r in rows}) == len(rows)
    assert all(r["common_name_es"].strip() for r in rows)


@needs_labels
def test_es_table_matches_model_labels():
    index = TaxonomyIndex.from_file(LABELS_FILE, common_names_es={})
    with COMMON_NAMES_ES.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            taxon = index.by_id.get(row["label_id"])
            assert taxon is not None, row
            assert taxon.scientific_name == row["scientific_name"] and taxon.rank is Rank.SPECIES


# ---------------------------------------------------------------------------
# En el pipeline (preset denso, FakeModels)
# ---------------------------------------------------------------------------
def _video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 180))
    for i in range(45):
        img = np.full((180, 320, 3), 170, np.uint8)
        img[50:130, (i * 9) % 260 : (i * 9) % 260 + 60] = 20
        writer.write(img)
    writer.release()
    return path


@pytest.mark.parametrize("lang,common", [("en", "lowland tapir"), ("es", "anta")])
def test_pipeline_result_has_full_taxonomy(tmp_path, lang, common):
    from dataclasses import replace

    from wwm_pipeline.pipeline import classify_file

    models = FakeModels(labels=[TAPIR, JAGUAR], scores=[0.9, 0.05])
    result = classify_file(_video(tmp_path), None, replace(PipelineConfig.desktop(), common_name_lang=lang), models)
    tax = result.taxonomy
    assert tax is not None and tax.scientific_name == "Tapirus terrestris" and tax.family == "tapiridae"
    assert tax.rank is Rank.SPECIES and tax.common_name == common


@pytest.mark.integration
@pytest.mark.skipif(not (ROOT / "models/md_v5a.0.0.pt").exists(), reason="requiere los modelos")
def test_model_registry_builds_index():
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from tests.model_paths import registry

    models = registry().load()
    assert len(models.taxonomy) >= 2400
    assert models.taxonomy.to_taxonomy(JAGUAR).family == "felidae"
