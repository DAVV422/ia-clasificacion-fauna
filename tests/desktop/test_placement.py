"""HU-DS-02 · Qué queda en el origen y la línea del registro (sin escribirla: eso es HU-DS-14)."""

from types import SimpleNamespace

import pytest

from desktop.job.placement import is_same_or_inside, should_relocate
from desktop.job.record import LEFT_IN_PLACE, build_line, error_line
from tests.fakes import FakeModels
from tools._fixture_taxa import JAGUAR
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.enums import Rank
from wwm_pipeline.package import DesktopAction, PackageLine

cv2 = pytest.importorskip("cv2")


def fake_result(category: str, rank=None):
    taxonomy = SimpleNamespace(rank=rank) if rank is not None else None
    return SimpleNamespace(decision=SimpleNamespace(category=category), taxonomy=taxonomy)


# ---------------------------------------------------------------------------
# CA-1 · Matriz de decisión
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", ["animal", "person", "vehicle", "blank"])
@pytest.mark.parametrize("rank", [Rank.SPECIES, Rank.GENUS, Rank.FAMILY, Rank.ORDER, Rank.CLASS])
def test_should_relocate_matrix(category, rank):
    expected = category == "animal" and rank in (Rank.SPECIES, Rank.GENUS, Rank.FAMILY)
    assert should_relocate(fake_result(category, rank if category == "animal" else None)) is expected


def test_should_relocate_edge_cases():
    assert should_relocate(None) is False
    assert should_relocate(fake_result("animal", None)) is False  # animal sin taxonomía


# ---------------------------------------------------------------------------
# La línea, con resultados reales del pipeline (FakeModels)
# ---------------------------------------------------------------------------
def classify(tmp_path, value=200, category_id=0, name="v.mp4"):
    from wwm_pipeline.pipeline import classify_file

    path = tmp_path / name
    if path.suffix == ".jpg":
        cv2.imwrite(str(path), _frame(0, value))
    else:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
        for i in range(20):
            writer.write(_frame(i, value))
        writer.release()
    return classify_file(path, None, PipelineConfig.desktop(), FakeModels(category_id=category_id))


def _frame(i, value):
    import numpy as np

    img = np.full((180, 320, 3), value, np.uint8)
    img[60:120, (i * 9) % 260 : (i * 9) % 260 + 60] = value - 35
    return img


def test_line_for_relocated_animal(tmp_path):
    result = classify(tmp_path)
    line = build_line(result, DesktopAction(type="copy", destination_folder="panthera_onca_jaguar"))
    assert isinstance(line, PackageLine)
    assert line.status.value == "done" and line.category.value == "animal"
    assert line.taxonomy.label_id == JAGUAR.split(";")[0]
    assert line.file_sha256 == result.sha256 and line.file_size_bytes == result.size_bytes
    assert line.evidence.file == f"evidence/{result.sha256}.jpg"
    assert line.desktop_action.destination_folder == "panthera_onca_jaguar"
    assert line.frames_analyzed == result.frames_sampled


@pytest.mark.parametrize("category_id,category", [(1, "person"), (2, "vehicle")])
def test_person_and_vehicle_lines_have_no_evidence(tmp_path, category_id, category):
    """CA-7b (F2-D2, F2-D9): el pipeline generó el fotograma, pero la línea va sin evidence."""
    result = classify(tmp_path, category_id=category_id)
    assert result.decision.category == category and result.evidence is not None
    line = build_line(result)
    assert line.category.value == category and line.evidence is None
    assert line.desktop_action == LEFT_IN_PLACE


def test_blank_and_image_lines(tmp_path):
    blank = build_line(classify(tmp_path, value=10))
    assert blank.category.value == "blank" and blank.evidence is None
    image = build_line(classify(tmp_path, name="a.jpg"))
    assert image.media_type.value == "image" and image.duration_seconds is None
    assert (image.evidence.frame_index, image.evidence.timestamp_seconds) == (0, 0.0)


# ---------------------------------------------------------------------------
# CA-2 · Errores
# ---------------------------------------------------------------------------
def test_error_line_left_in_place():
    line = error_line("roto.mp4", "a" * 64, 123, "IA_DECODE_FAILED")
    assert line.status.value == "error" and line.error_code == "IA_DECODE_FAILED"
    assert line.category is None and line.taxonomy is None and line.evidence is None
    assert line.desktop_action.type.value == "left_in_place"
    assert error_line("foto.JPG", "b" * 64, 1, "IA_INTERNAL").media_type.value == "image"


# ---------------------------------------------------------------------------
# CA-5 · Destino dentro del origen
# ---------------------------------------------------------------------------
def test_is_same_or_inside(tmp_path):
    src = tmp_path / "SD"
    assert is_same_or_inside(src, src)
    assert is_same_or_inside(src / "clasificados", src)
    assert is_same_or_inside(tmp_path / "sd" / "x", src)  # sin distinguir mayúsculas (Windows)
    assert not is_same_or_inside(tmp_path / "SD_clasificados", src)  # prefijo, pero hermana
    assert not is_same_or_inside(tmp_path / "otra", src)
