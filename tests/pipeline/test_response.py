"""HU-IA-41 · De FileResult a DetectResult (wwm_pipeline/response.py), con FakeModels y el preset web."""

import pytest

from tests.desktop.fixtures import SetModels, write_f_set_sint
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.contract import DetectResult
from wwm_pipeline.errors import PipelineError
from wwm_pipeline.pipeline import classify_file
from wwm_pipeline.response import to_detect_result

pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def sint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sint")
    write_f_set_sint(folder)
    return folder


def detect(folder, name, config=None):
    result = classify_file(folder / name, name, config or PipelineConfig.web(), SetModels())
    return to_detect_result(result, name, "speciesnet-4.0.2a")


def test_animal_video(sint):
    data = detect(sint, "especie_a.mp4")
    DetectResult.model_validate(data.model_dump(by_alias=True))
    assert data.result.category.value == "animal" and data.result.taxonomy.scientific_name == "Panthera onca"
    assert data.result.taxonomy.rank.value == "species" and data.result.taxonomy.label_id
    assert data.media.media_type.value == "video" and data.media.duration_seconds and data.media.fps
    assert len(data.media.sha256) == 64 and data.media.size_bytes > 0
    ev = data.evidence
    assert ev is not None and ev.image_mime == "image/jpeg" and max(ev.width, ev.height) <= 960
    assert ev.timestamp_seconds == round(ev.timestamp_seconds, 2)
    assert data.model.detector == "MDv5a" and data.model.classifier == "speciesnet-4.0.2a"
    assert data.model.sampling.startswith("10fps@0-2s+4fps")
    assert data.analysis.frames_sampled >= data.analysis.frames_with_animal > 0


def test_evidence_has_no_label_strip(sint):
    """El fotograma web va sin franja (IA §2.3 y PKG §4): la altura es la del frame escalado."""
    import base64

    import cv2
    import numpy as np

    data = detect(sint, "especie_a.mp4")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data.evidence.image_base64), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (data.evidence.height, data.evidence.width) == (180, 320)


def test_image(sint):
    data = detect(sint, "especie_b.jpg")
    assert data.media.duration_seconds is None and data.media.fps is None
    assert data.evidence.frame_index == 0 and data.evidence.timestamp_seconds == 0
    assert data.model.sampling == "image"


def test_genus(sint):
    data = detect(sint, "genero_c.mp4")
    assert data.result.taxonomy.rank.value == "genus" and data.result.taxonomy.species is None


def test_blank_has_no_evidence(sint):
    data = detect(sint, "vacio_f.mp4")
    assert data.result.category.value == "blank" and data.evidence is None and data.result.taxonomy is None


@pytest.mark.parametrize("name,category", [("persona_h.mp4", "person"), ("vehiculo_i.jpg", "vehicle")])
def test_person_and_vehicle_have_evidence(sint, name, category):
    """En la web, persona y vehículo sí llevan fotograma (IA §2.3); solo el escritorio no los guarda (F2-D2)."""
    data = detect(sint, name)
    assert data.result.category.value == category and data.evidence is not None and data.result.alternatives == []


def test_capture_from_filename(tmp_path):
    import shutil

    from tests.desktop.fixtures import write_f_set_sint as w

    folder = tmp_path / "x"
    w(folder)
    target = folder / "CAM01_20260315_042210.mp4"
    shutil.copy(folder / "especie_a.mp4", target)
    data = detect(folder, target.name)
    assert data.capture.source.value in ("filename", "container")
    assert data.capture.captured_at is not None


def test_truncated(tmp_path):
    from tests.pipeline.test_evidence import write_video

    video = write_video(tmp_path / "largo.mp4", n=200, fps=10)  # 20 s
    data = to_detect_result(classify_file(video, "largo.mp4", PipelineConfig.web(), SetModels()), "largo.mp4", "sn")
    assert data.analysis.truncated_at_seconds == 15.0


def test_legacy_preset_is_rejected(sint):
    result = classify_file(sint / "especie_a.mp4", "especie_a.mp4", PipelineConfig.legacy_api(), SetModels())
    with pytest.raises(PipelineError) as err:
        to_detect_result(result, "especie_a.mp4", "sn")
    assert err.value.code == "IA_INTERNAL"
