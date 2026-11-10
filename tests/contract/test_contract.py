"""HU-IA-21 · CA-1, CA-2, CA-3 — contrato de POST /detect."""

import base64
import copy
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from wwm_pipeline.contract import DetectEnvelope, DetectResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/ai"
SUCCESS = ["animal_species_video", "animal_genus_video", "blank_image", "person_video", "vehicle_image", "animal_image"]
ERRORS = {
    "error_decode": "IA_DECODE_FAILED",
    "error_duration": "IA_DURATION_EXCEEDED",
    "error_unauthorized": "IA_UNAUTHORIZED",
    "error_busy": "IA_BUSY",
}


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SUCCESS)
def test_success_fixtures_validate(name):
    """BE-03 consume estos fixtures: deben cumplir el contrato."""
    env = DetectEnvelope.model_validate(load(name))
    assert env.status == "success"
    DetectResult.model_validate(load(name)["data"])


@pytest.mark.parametrize("name,code", ERRORS.items())
def test_error_fixtures_validate(name, code):
    env = DetectEnvelope.model_validate(load(name))
    assert env.status == "error" and env.data is None and env.error.code == code


def test_detect_result_fields_match_contract():
    """CA-1: las claves de la respuesta son exactamente las de API_IA_BACK §2.2."""
    data = load("animal_species_video")["data"]
    assert set(data) == {"media", "capture", "result", "evidence", "analysis", "model"}
    assert set(data["media"]) == {
        "file_name",
        "media_type",
        "sha256",
        "size_bytes",
        "duration_seconds",
        "fps",
        "width",
        "height",
    }
    assert set(data["capture"]) == {"captured_at", "source"}
    assert set(data["result"]) == {"category", "taxonomy", "confidence", "alternatives"}
    assert set(data["result"]["taxonomy"]) == {
        "label_id",
        "label_raw",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "scientific_name",
        "common_name",
        "rank",
    }
    assert set(data["evidence"]) == {
        "frame_index",
        "timestamp_seconds",
        "bbox",
        "detection_confidence",
        "image_base64",
        "image_mime",
        "width",
        "height",
    }
    assert set(data["analysis"]) == {
        "frames_sampled",
        "frames_with_animal",
        "frames_discarded",
        "votes",
        "processing_ms",
    }
    assert set(data["model"]) == {"detector", "classifier", "sampling", "pipeline_version"}


def _mutated(mutator):
    data = copy.deepcopy(load("animal_species_video")["data"])
    mutator(data)
    return data


INVALID = {
    "blank_with_taxonomy": (lambda d: d["result"].update(category="blank"), "taxonomy"),
    "animal_without_taxonomy": (lambda d: d["result"].update(taxonomy=None), "taxonomy"),
    "animal_without_evidence": (lambda d: d.update(evidence=None), "evidence"),
    "genus_with_species": (lambda d: d["result"]["taxonomy"].update(rank="genus"), "species"),
    "bbox_out_of_range": (lambda d: d["evidence"].update(bbox=[0.1, 0.1, 1.2, 0.5]), "bbox"),
    "bbox_inverted": (lambda d: d["evidence"].update(bbox=[0.8, 0.1, 0.2, 0.5]), "bbox"),
    "confidence_gt_1": (lambda d: d["result"].update(confidence=1.5), "confidence"),
    "too_many_alternatives": (
        lambda d: d["result"].update(alternatives=d["result"]["alternatives"] * 2),
        "alternatives",
    ),
    "bad_sha256": (lambda d: d["media"].update(sha256="xyz"), "sha256"),
    "evidence_too_big": (lambda d: d["evidence"].update(width=1920), "width"),
    "extra_field": (lambda d: d.update(geofence="BOL"), "geofence"),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_combinations(case):
    """CA-2: el schema rechaza las combinaciones que el contrato prohíbe."""
    mutator, needle = INVALID[case]
    with pytest.raises(ValidationError) as exc:
        DetectResult.model_validate(_mutated(mutator))
    assert needle in str(exc.value)


def test_image_evidence_must_be_frame_zero():
    data = copy.deepcopy(load("animal_image")["data"])
    data["evidence"]["timestamp_seconds"] = 3.0
    with pytest.raises(ValidationError, match="imágenes"):
        DetectResult.model_validate(data)


@pytest.mark.parametrize("name", [n for n in SUCCESS if n != "blank_image"])
def test_fixture_frames_are_jpeg_le_960(name):
    """CA-3: el base64 de cada fixture es un JPEG real de ≤ 960 px."""
    ev = load(name)["data"]["evidence"]
    with Image.open(io.BytesIO(base64.b64decode(ev["image_base64"]))) as img:
        assert img.format == "JPEG"
        assert max(img.size) <= 960
        assert img.size == (ev["width"], ev["height"])
