"""HU-IA-21 · CA-6, CA-7, CA-8 — catálogo de muestra, schemas exportados y dependencias mínimas."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools._fixture_taxa import FIXTURE_LABELS
from tools.export_schemas import OUT as SCHEMAS_DIR
from tools.export_schemas import render
from wwm_pipeline.catalog_schema import Catalog, LabelParseError, parse_label
from wwm_pipeline.enums import Rank

ROOT = Path(__file__).resolve().parents[2]
CATALOG_FILE = ROOT / "tests/fixtures/catalog/speciesnet_catalog.sample.json"
LABELS_FILE = (
    Path.home()
    / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"
    / "always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt"
)


def test_sample_catalog_validates_and_is_big_enough():
    catalog = Catalog.model_validate_json(CATALOG_FILE.read_text(encoding="utf-8"))
    assert catalog.count == len(catalog.taxa) >= 200
    assert catalog.classifier == "speciesnet-4.0.2a"


def test_sample_catalog_contains_fixture_taxa():
    ids = {t["label_id"] for t in json.loads(CATALOG_FILE.read_text(encoding="utf-8"))["taxa"]}
    for label in FIXTURE_LABELS:
        assert label.split(";")[0] in ids


@pytest.mark.parametrize(
    "label,rank,name",
    [
        (
            "9b7fea59-cfa3-499c-a420-bcf36277dcd8;mammalia;carnivora;felidae;panthera;onca;jaguar",
            Rank.SPECIES,
            "Panthera onca",
        ),
        (
            "4e6eb99a-baf0-4235-a7bc-1dd7b741b7c3;mammalia;rodentia;dasyproctidae;dasyprocta;;dasyprocta species",
            Rank.GENUS,
            "Dasyprocta",
        ),
        ("837e12c2-f30c-4875-afeb-32d51938f139;aves;galliformes;;;;galliformes order", Rank.ORDER, "Galliformes"),
    ],
)
def test_parse_label(label, rank, name):
    taxon = parse_label(label)
    assert taxon.rank is rank and taxon.scientific_name == name


def test_non_taxon_labels_are_skipped():
    assert parse_label("f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank") is None
    assert parse_label("e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle") is None


def test_malformed_label_raises():
    with pytest.raises(LabelParseError):
        parse_label("id;mammalia;carnivora")


@pytest.mark.skipif(not LABELS_FILE.exists(), reason="requiere el modelo SpeciesNet descargado")
def test_every_real_label_parses():
    """Todas las etiquetas reales del modelo se convierten sin errores (base de IA-14 e IA-19)."""
    taxa = [parse_label(line) for line in LABELS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(taxa) == 2498
    assert sum(t is not None for t in taxa) > 2400


def test_exported_schemas_in_sync():
    """CA-7: si se cambia un modelo, hay que regenerar los schemas (python -m tools.export_schemas)."""
    for file_name, text in render().items():
        assert (SCHEMAS_DIR / file_name).read_text(encoding="utf-8") == text, f"{file_name} desactualizado"


def test_contract_imports_without_torch_or_cv2():
    """CA-8: el back puede usar el contrato sin instalar torch ni OpenCV."""
    code = (
        "import sys\n"
        "for m in ('torch', 'cv2', 'numpy'): sys.modules[m] = None\n"
        "import wwm_pipeline.contract, wwm_pipeline.package, wwm_pipeline.catalog_schema, wwm_pipeline.package_validator\n"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
