"""HU-IA-20 · CA-4 y la cadena de fechas con el material real de WWF (``estacion_1``, fuera del repo)."""

import os
from pathlib import Path

import pytest

from wwm_pipeline.enums import CaptureSource, MediaType
from wwm_pipeline.metadata import capture, probe, read_mvhd

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", str(ROOT.parent / "docs-interno" / "estacion_1")))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ESTACION_1.is_dir(), reason="requiere la carpeta estacion_1"),
]


def test_duration_on_estacion_1():
    """CA-4: la duración es la del contenedor (FPS variable) y el fps es el declarado, aunque no sea entero."""
    import cv2

    videos = sorted(ESTACION_1.glob("*.mp4"))
    assert len(videos) == 26
    mismatched = 0
    for video in videos:
        info = probe(video, MediaType.VIDEO)
        container = read_mvhd(video).duration_seconds
        assert container is not None, video.name
        assert abs(info.duration_seconds - container) < 0.5, video.name
        cap = cv2.VideoCapture(str(video))
        declared = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert info.fps == pytest.approx(declared, abs=1e-3)
        if info.frame_count and abs(info.frame_count / info.fps - container) > 0.5:
            mismatched += 1
        assert info.width > 0 and info.height > 0
    assert any(probe(v, MediaType.VIDEO).fps % 1 for v in videos)  # hay fps no enteros (8,11 · 13,32 …)


def test_capture_chain_on_estacion_1():
    """Los nombres de WhatsApp no dan fecha; algunos contenedores la conservan y otros la tienen en 0."""
    sources = [capture(v, MediaType.VIDEO).source for v in sorted(ESTACION_1.glob("*.mp4"))]
    assert CaptureSource.FILENAME not in sources
    assert CaptureSource.CONTAINER in sources
    assert set(sources) <= {CaptureSource.CONTAINER, CaptureSource.NONE}
