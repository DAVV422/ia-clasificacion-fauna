"""HU-IA-40 · CA-4…CA-8 — pipeline con modelos falsos (sin torch)."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.fakes import FakeModels
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.errors import PipelineCancelled, PipelineError
from wwm_pipeline.pipeline import classify_file
from wwm_pipeline.sampling import legacy_frames

ROOT = Path(__file__).resolve().parents[2]


def write_video(path: Path, bright_from: int, frames: int = 120, fps: int = 30) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(frames):
        value = 200 if i >= bright_from else 10
        vw.write(np.full((120, 160, 3), value, dtype=np.uint8))
    vw.release()
    return path


def write_image(path: Path, value: int = 200) -> Path:
    cv2.imwrite(str(path), np.full((120, 160, 3), value, dtype=np.uint8))
    return path


def test_legacy_frames_desktop_and_api():
    assert legacy_frames(300, "desktop") == [75, 150, 225]
    assert legacy_frames(300, "api") == [75, 150, 225]
    assert legacy_frames(2, "desktop") == [0, 1]  # el escritorio analiza hasta frame_count frames
    assert legacy_frames(2, "api") == [0, 1, 1]  # la API siempre pide 3
    assert legacy_frames(0, "desktop") == [0] and legacy_frames(0, "api") == [0]


def test_early_exit_desktop_vs_api(tmp_path):
    """El escritorio se detiene en el primer animal; la API analiza los 3 frames."""
    video = write_video(tmp_path / "v.mp4", bright_from=0)
    desk = classify_file(video, None, PipelineConfig.legacy_desktop(), FakeModels())
    api = classify_file(video, None, PipelineConfig.legacy_api(), FakeModels())
    assert desk.frames_analyzed == [30] and api.frames_analyzed == [30, 60, 90]
    assert desk.decision.label == "panthera;onca;jaguar" and desk.decision.confidence == 0.91


def test_late_animal_found_in_second_frame(tmp_path):
    video = write_video(tmp_path / "v.mp4", bright_from=50)
    result = classify_file(video, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert result.frames_analyzed == [30, 60]
    assert result.decision.category == "animal"


def test_blank_and_person(tmp_path):
    dark = write_image(tmp_path / "dark.jpg", value=10)
    assert classify_file(dark, None, PipelineConfig.legacy_desktop(), FakeModels()).decision.category == "blank"
    bright = write_image(tmp_path / "person.jpg")
    from dataclasses import replace

    fixed = replace(PipelineConfig.legacy_desktop(), correct_class_map=True)
    person = classify_file(bright, None, fixed, FakeModels(category_id=1))
    assert person.decision.category == "person" and person.decision.label is None


def test_legacy_class_map_bug_is_preserved_and_fix_available():
    """El prototipo traduce persona→animal y vehículo→persona; legacy lo conserva, la tabla correcta no."""
    from wwm_pipeline.detection import map_class_name, map_class_name_correct

    names = {0: "animal", 1: "person", 2: "vehicle"}  # model.names de MegaDetector v5a
    assert [map_class_name(k, v) for k, v in names.items()] == ["animal", "animal", "person"]
    assert [map_class_name_correct(k, v) for k, v in names.items()] == ["animal", "person", "vehicle"]


def test_progress_order(tmp_path):
    """CA-7"""
    calls = []
    video = write_video(tmp_path / "v.mp4", bright_from=1000)
    classify_file(video, None, PipelineConfig.legacy_api(), FakeModels(), on_progress=lambda *a: calls.append(a))
    assert calls == [("detection", 1, 3), ("detection", 2, 3), ("detection", 3, 3)]


def test_should_stop_cancels_cleanly(tmp_path):
    """CA-7"""
    video = write_video(tmp_path / "v.mp4", bright_from=1000)
    with pytest.raises(PipelineCancelled):
        classify_file(video, None, PipelineConfig.legacy_api(), FakeModels(), should_stop=lambda: True)


def test_save_crops_flag(tmp_path):
    """CA-6: sin save_crops no se escribe nada; con save_crops, en la carpeta recibida."""
    image = write_image(tmp_path / "a.jpg")
    before = set(tmp_path.rglob("*"))
    off = classify_file(image, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert set(tmp_path.rglob("*")) == before
    assert off.all_detections[0]["crop_url"] is None

    crops = tmp_path / "crops"
    on = classify_file(image, None, PipelineConfig.legacy_api().with_crops(crops), FakeModels(), run_name="run1")
    written = list(crops.rglob("*.jpg"))
    assert len(written) == 1 and written[0].parent == crops / "run1" / "animal"
    assert on.all_detections[0]["crop_url"].startswith("/detections/run1/animal/")


def test_decode_error_code(tmp_path):
    """CA-8"""
    bad = tmp_path / "roto.jpg"
    bad.write_bytes(b"no es una imagen")
    with pytest.raises(PipelineError) as exc:
        classify_file(bad, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert exc.value.code == "IA_DECODE_FAILED"
    bad_video = tmp_path / "roto.mp4"
    bad_video.write_bytes(b"no es un video")
    with pytest.raises(PipelineError) as exc:
        classify_file(bad_video, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert exc.value.code == "IA_DECODE_FAILED"


def test_invalid_format_code(tmp_path):
    """CA-8"""
    raw = tmp_path / "foto.cr2"
    raw.write_bytes(b"x")
    with pytest.raises(PipelineError) as exc:
        classify_file(raw, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert exc.value.code == "IA_INVALID_FORMAT"


def test_file_name_overrides_extension(tmp_path):
    """El archivo temporal puede no tener extensión: manda el nombre original."""
    tmp = write_image(tmp_path / "upload.jpg")
    renamed = tmp.rename(tmp_path / "upload.bin")
    result = classify_file(renamed, "IMG_0001.JPG", PipelineConfig.legacy_desktop(), FakeModels())
    assert result.media_type == "image" and result.file_name == "IMG_0001.JPG"


def test_no_framework_imports():
    """CA-4"""
    for py in (ROOT / "wwm_pipeline").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for banned in ("fastapi", "tkinter", "uvicorn", "eval.legacy"):
            assert f"import {banned}" not in text and f"from {banned}" not in text, (py, banned)


def test_import_has_no_side_effects(tmp_path):
    """CA-4: importar wwm_pipeline no crea archivos ni carpetas."""
    code = (
        "import wwm_pipeline, wwm_pipeline.pipeline, wwm_pipeline.models, wwm_pipeline.config, "
        "wwm_pipeline.contract, wwm_pipeline.package\nprint('ok')"
    )
    env = {"PYTHONPATH": str(ROOT), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
