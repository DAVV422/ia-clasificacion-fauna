"""HU-IA-21 · CA-4, CA-5 — formato y validación del paquete de escritorio."""

import json
import zipfile
from pathlib import Path

import pytest

from wwm_pipeline.package import Manifest, PackageLine, count_lines
from wwm_pipeline.package_validator import (
    CATALOG_TAXON_NOT_FOUND,
    EVIDENCE_INVALID,
    IMPORT_CHECKSUM_MISMATCH,
    IMPORT_INVALID_PACKAGE,
    IMPORT_TOO_LARGE,
    IMPORT_UNSUPPORTED_VERSION,
    validate_package,
)

PKG = Path(__file__).resolve().parents[1] / "fixtures/package"
CATALOG = json.loads((Path(__file__).resolve().parents[1] / "fixtures/catalog/speciesnet_catalog.sample.json")
                     .read_text(encoding="utf-8"))
CATALOG_IDS = {t["label_id"] for t in CATALOG["taxa"]}


def test_sample_package_validates():
    report = validate_package(PKG / "sample_valid.wwm.zip", catalog_label_ids=CATALOG_IDS)
    assert report.ok, (report.fatal_code, report.issues)
    assert len(report.lines) == 12
    assert report.manifest.camera is not None and report.manifest.camera.is_complete


def test_sample_package_without_camera_validates():
    report = validate_package(PKG / "sample_valid_no_camera.wwm.zip", catalog_label_ids=CATALOG_IDS)
    assert report.ok and report.manifest.camera is None


def test_manifest_counts_match_lines():
    with zipfile.ZipFile(PKG / "sample_valid.wwm.zip") as zf:
        manifest = Manifest.model_validate_json(zf.read("manifest.json"))
        lines = [PackageLine.model_validate_json(t) for t in zf.read("classifications.jsonl").decode().splitlines()]
    counts = count_lines(lines)
    assert counts == manifest.counts
    assert (counts.animal, counts.blank, counts.person, counts.vehicle, counts.error, counts.evidence_files) == (7, 2, 1, 1, 1, 9)


FATAL = {
    "invalid_zip_slip": IMPORT_INVALID_PACKAGE,
    "invalid_checksum": IMPORT_CHECKSUM_MISMATCH,
    "invalid_version_2": IMPORT_UNSUPPORTED_VERSION,
    "invalid_no_manifest": IMPORT_INVALID_PACKAGE,
}
LINE_ISSUES = {
    "invalid_corrupt_jpg": EVIDENCE_INVALID,
    "invalid_unknown_taxon": CATALOG_TAXON_NOT_FOUND,
}


@pytest.mark.parametrize("name,code", FATAL.items())
def test_invalid_packages_fail_for_expected_reason(name, code):
    report = validate_package(PKG / f"{name}.wwm.zip", catalog_label_ids=CATALOG_IDS)
    assert report.fatal_code == code


@pytest.mark.parametrize("name,code", LINE_ISSUES.items())
def test_invalid_lines_reported_as_issues(name, code):
    report = validate_package(PKG / f"{name}.wwm.zip", catalog_label_ids=CATALOG_IDS)
    assert report.fatal_code is None
    codes = [i.code for i in report.issues]
    assert codes == [code]
    assert not report.ok


def test_too_large_package():
    report = validate_package(PKG / "sample_valid.wwm.zip", max_bytes=1024)
    assert report.fatal_code == IMPORT_TOO_LARGE


def test_not_a_zip():
    assert validate_package(b"no soy un zip").fatal_code == IMPORT_INVALID_PACKAGE


def test_line_rules():
    with zipfile.ZipFile(PKG / "sample_valid.wwm.zip") as zf:
        raw = [json.loads(t) for t in zf.read("classifications.jsonl").decode().splitlines()]
    animal = raw[0]
    bad = dict(animal, desktop_action={"type": "copy", "destination_folder": None})
    with pytest.raises(ValueError, match="destination_folder"):
        PackageLine.model_validate(bad)
    bad = dict(animal, evidence=dict(animal["evidence"], file="evidence/" + "a" * 64 + ".jpg"))
    with pytest.raises(ValueError, match="file_sha256"):
        PackageLine.model_validate(bad)
    error_line = raw[-1]
    bad = dict(error_line, error_code=None)
    with pytest.raises(ValueError, match="error_code"):
        PackageLine.model_validate(bad)


def test_camera_rules():
    with zipfile.ZipFile(PKG / "sample_valid.wwm.zip") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    manifest["camera"] = dict(manifest["camera"], longitude=None)
    with pytest.raises(ValueError, match="van juntas"):
        Manifest.model_validate(manifest)
    manifest["camera"] = {"code": "CAM01", "deployed_at": "2026-03-10T00:00:00", "retrieved_at": "2026-03-01T00:00:00"}
    with pytest.raises(ValueError, match="anterior"):
        Manifest.model_validate(manifest)


def test_source_folder_name_without_paths():
    with zipfile.ZipFile(PKG / "sample_valid.wwm.zip") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    manifest["source_folder_name"] = "C:\\Users\\ana\\SD"
    with pytest.raises(ValueError, match="sin rutas"):
        Manifest.model_validate(manifest)


def test_unknown_minor_fields_are_ignored():
    """Formato 1.x: un MINOR nuevo solo agrega campos opcionales, que se ignoran."""
    with zipfile.ZipFile(PKG / "sample_valid.wwm.zip") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    manifest["format_version"] = "1.3"
    manifest["campo_nuevo"] = {"x": 1}
    assert Manifest.model_validate(manifest).major == 1
