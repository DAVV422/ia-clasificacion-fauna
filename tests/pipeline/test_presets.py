"""HU-DS-01 · CA-1 y CA-2: el preset de escritorio (modo precisión)."""

from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeModels
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.pipeline import classify_file

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]


def test_desktop_preset_values():
    c = PipelineConfig.desktop()
    assert c.sampling == "dense" and c.early_exit is False
    assert (c.head_window_seconds, c.head_fps, c.sampling_fps, c.max_frames) == (2.0, 15.0, 6.0, 120)
    assert c.max_video_seconds is None and c.correct_class_map is True
    assert c.sampling_label == "15fps@0-2s+6fps/max120"
    assert c.evidence and c.evidence_label


def test_no_species_filter():
    """CA-1: ningún filtro excluye especies del catálogo (decisión de IA-12)."""
    fields = set(PipelineConfig.__dataclass_fields__)
    assert not {f for f in fields if "exclude" in f or "allow" in f or "geo" in f}
    for py in (ROOT / "wwm_pipeline").rglob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        assert "excluded_species" not in text and "species_allowlist" not in text, py.name


def test_desktop_preset_no_early_exit(tmp_path):
    """CA-2: video de 15 s a 30 fps; se clasifican los frames con animal hasta el presupuesto."""
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(15 * 30):
        img = np.full((180, 320, 3), 170, np.uint8)
        img[::16, :] = 85
        img[:, ::16] = 85
        x = (i * 9) % 260
        img[50:130, x : x + 60] = 20
        writer.write(img)
    writer.release()
    models = FakeModels()
    result = classify_file(path, None, PipelineConfig.desktop(), models)
    assert result.frames_sampled > 3
    assert models.classifier_calls == min(result.frames_with_animal, PipelineConfig.desktop().max_classified_frames)
    assert result.sampling == "15fps@0-2s+6fps/max120"
