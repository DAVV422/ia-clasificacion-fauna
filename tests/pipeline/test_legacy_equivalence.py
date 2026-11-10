"""HU-IA-40 · CA-1 — el preset legacy de wwm_pipeline reproduce el prototipo (expected.json).

Requiere los modelos reales y las muestras (LEGACY_SAMPLES_DIR); si no están, se salta.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "eval/legacy/expected.json"
SAMPLES = Path(os.getenv("LEGACY_SAMPLES_DIR", str(Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples")))
MD = ROOT / "models/md_v5a.0.0.pt"

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (EXPECTED.exists() and SAMPLES.is_dir() and MD.exists()),
                                 reason="requiere expected.json, las muestras y los modelos")]


@pytest.fixture(scope="module")
def models():
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from wwm_pipeline.models import ModelRegistry

    return ModelRegistry(MD, device="cpu").load()


def test_legacy_preset_matches_expected(models):
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["files"]
    for item in expected:
        result = classify_file(SAMPLES / item["file_name"], None, PipelineConfig.legacy_desktop(), models)
        assert result.frames_analyzed == item["frames"], item["file_name"]
        assert result.decision.label == item["decision"]["label"], item["file_name"]
        assert result.decision.confidence == pytest.approx(item["decision"]["confidence"], abs=1e-4)
        got = [(d["class_name"], d["confidence"], d["box"], d["species"][:5]) for d in result.all_detections]
        want = [(d["class_name"], d["confidence"], d["box"], d["species"]) for d in item["detections"]]
        assert len(got) == len(want)
        for (gc, gconf, gbox, gsp), (wc, wconf, wbox, wsp) in zip(got, want):
            assert gc == wc and gconf == pytest.approx(wconf, abs=1e-4) and gbox == pytest.approx(wbox, abs=1e-4)
            assert [s["species"] for s in gsp] == [s["species"] for s in wsp]


def test_models_loaded_once(models):
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    before = models.load_count
    for _ in range(3):
        classify_file(SAMPLES / "muestra_2.jpg", None, PipelineConfig.legacy_desktop(), models)
    assert models.load_count == before
