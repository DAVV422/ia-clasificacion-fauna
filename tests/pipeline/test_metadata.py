"""HU-IA-20 · Huella, duración y fecha de captura (archivos chicos generados en ``tmp_path``)."""

import hashlib
import os
import struct
import subprocess
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from wwm_pipeline.enums import CaptureSource, MediaType
from wwm_pipeline.metadata import capture, file_sha256, probe, read_mvhd

cv2 = pytest.importorskip("cv2")
EPOCH_1904 = datetime(1904, 1, 1)
WHATSAPP = "WhatsApp Video 2026-05-30 at 9.49.20 AM.mp4"


def write_mp4(path: Path, frames: int = 30, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(frames):
        writer.write(np.full((120, 160, 3), (i * 7) % 255, np.uint8))
    writer.release()
    return path


def set_creation_time(path: Path, when) -> None:
    """Escribe ``mvhd.creation_time`` a mano (``when``: datetime o segundos desde 1904)."""
    data = bytearray(path.read_bytes())
    pos = data.find(b"mvhd")
    assert pos > 0
    version = data[pos + 4]
    seconds = when if isinstance(when, int) else int((when - EPOCH_1904).total_seconds())
    if version == 1:
        struct.pack_into(">Q", data, pos + 8, seconds)
    else:
        struct.pack_into(">I", data, pos + 8, seconds)
    path.write_bytes(bytes(data))


def write_jpg(path: Path, exif_date=None, offset=None) -> Path:
    from PIL import Image

    img = Image.new("RGB", (64, 48), (120, 80, 40))
    exif = Image.Exif()
    if exif_date:
        ifd = {36867: exif_date}
        if offset:
            ifd[36881] = offset
        exif[0x8769] = ifd
    img.save(path, "JPEG", exif=exif.tobytes())
    return path


# ---------------------------------------------------------------------------
# CA-1 · Hash en streaming
# ---------------------------------------------------------------------------
def test_sha256_streaming_memory(tmp_path):
    big = tmp_path / "big.bin"
    block = os.urandom(1024 * 1024)
    with big.open("wb") as fh:  # 24 MB: suficiente para ver que no se carga entero (y cuida el disco)
        for _ in range(24):
            fh.write(block)
    tracemalloc.start()
    digest, size = file_sha256(big)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert size == 24 * 1024 * 1024
    assert digest == hashlib.sha256(big.read_bytes()).hexdigest()
    assert peak < 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# CA-2 · Cadena de fuentes
# ---------------------------------------------------------------------------
def test_capture_from_filename(tmp_path):
    video = write_mp4(tmp_path / "CAM01_20260315_042210.mp4")
    cap = capture(video, MediaType.VIDEO, now=datetime(2026, 11, 11))
    assert (cap.captured_at, cap.source) == (datetime(2026, 3, 15, 4, 22, 10), CaptureSource.FILENAME)


def test_capture_exif_wins_over_filename(tmp_path):
    jpg = write_jpg(tmp_path / "IMG_20260101_000000.jpg", "2025:08:01 06:12:00")
    cap = capture(jpg, MediaType.IMAGE)
    assert (cap.captured_at, cap.source) == (datetime(2025, 8, 1, 6, 12), CaptureSource.EXIF)
    assert cap.captured_at.tzinfo is None  # hora local de la cámara


def test_capture_exif_with_offset(tmp_path):
    jpg = write_jpg(tmp_path / "a.jpg", "2025:08:01 06:12:00", "-04:00")
    cap = capture(jpg, MediaType.IMAGE)
    assert cap.captured_at == datetime(2025, 8, 1, 6, 12, tzinfo=timezone(timedelta(hours=-4)))


def test_capture_from_container(tmp_path):
    video = write_mp4(tmp_path / WHATSAPP)
    set_creation_time(video, datetime(2023, 5, 22, 10, 51, 19))
    cap = capture(video, MediaType.VIDEO)
    assert (cap.captured_at, cap.source) == (datetime(2023, 5, 22, 10, 51, 19), CaptureSource.CONTAINER)


def test_capture_none_with_whatsapp_name(tmp_path):
    video = write_mp4(tmp_path / WHATSAPP)
    set_creation_time(video, 0)
    cap = capture(video, MediaType.VIDEO)
    assert (cap.captured_at, cap.source) == (None, CaptureSource.NONE)


def test_capture_uses_original_name(tmp_path):
    """La API recibe un temporal: la fecha del nombre sale del nombre original."""
    video = write_mp4(tmp_path / "upload.tmp")
    cap = capture(video, MediaType.VIDEO, file_name="CAM01_20260315_042210.mp4", now=datetime(2026, 11, 11))
    assert cap.source is CaptureSource.FILENAME


# ---------------------------------------------------------------------------
# CA-3 · Fechas basura
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "when", [1, datetime(1970, 1, 1), datetime(1999, 12, 31), datetime(2035, 1, 1)]
)  # mvhd v0 llega hasta 2040
def test_garbage_dates_ignored(tmp_path, when):
    video = write_mp4(tmp_path / "CAM01_20260315_042210.mp4")
    set_creation_time(video, when)
    cap = capture(video, MediaType.VIDEO, now=datetime(2026, 11, 11))
    assert cap.source is CaptureSource.FILENAME  # se pasó a la siguiente fuente


def test_garbage_exif_falls_through(tmp_path):
    jpg = write_jpg(tmp_path / "x.jpg", "1970:01:01 00:00:00")
    assert capture(jpg, MediaType.IMAGE).source is CaptureSource.NONE


# ---------------------------------------------------------------------------
# Duración y dimensiones
# ---------------------------------------------------------------------------
def test_probe_video_uses_container_duration(tmp_path):
    video = write_mp4(tmp_path / "v.mp4", frames=30, fps=10.0)
    info = probe(video, MediaType.VIDEO)
    mvhd = read_mvhd(video)
    assert (info.width, info.height, info.fps, info.frame_count) == (160, 120, 10.0, 30)
    assert info.duration_seconds == pytest.approx(mvhd.duration_seconds, abs=1e-3)
    assert info.duration_seconds == pytest.approx(3.0, abs=0.15)


def test_probe_video_without_container(tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, np.uint8))
    writer.release()
    assert probe(path, MediaType.VIDEO).duration_seconds == pytest.approx(2.0)


def test_probe_image(tmp_path):
    info = probe(write_jpg(tmp_path / "a.jpg"), MediaType.IMAGE)
    assert (info.width, info.height, info.fps, info.duration_seconds) == (64, 48, None, None)


def test_read_mvhd_on_garbage(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00\x00\x00\x10ftypisom" + os.urandom(64))
    assert read_mvhd(junk) is None
    assert capture(junk, MediaType.VIDEO).source is CaptureSource.NONE


# ---------------------------------------------------------------------------
# CA-5 · Sin binarios externos
# ---------------------------------------------------------------------------
def test_no_subprocess(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("metadata no debe ejecutar binarios")

    for name in ("run", "Popen", "call", "check_output"):
        monkeypatch.setattr(subprocess, name, boom)
    monkeypatch.setattr(os, "system", boom)
    video = write_mp4(tmp_path / "CAM01_20260315_042210.mp4")
    probe(video, MediaType.VIDEO)
    capture(video, MediaType.VIDEO)
    file_sha256(video)
    source = (Path(__file__).parents[2] / "wwm_pipeline/metadata.py").read_text(encoding="utf-8")
    assert "import subprocess" not in source and "os.system" not in source


# ---------------------------------------------------------------------------
# En el pipeline (preset denso, FakeModels)
# ---------------------------------------------------------------------------
def test_pipeline_fills_metadata(tmp_path):
    from tests.fakes import FakeModels
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    video = write_mp4(tmp_path / "CAM01_20260315_042210.mp4", frames=40, fps=10.0)
    result = classify_file(video, None, PipelineConfig.desktop(), FakeModels())
    assert result.sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert result.size_bytes == video.stat().st_size
    assert (result.width, result.height) == (160, 120)
    assert result.duration_seconds == pytest.approx(read_mvhd(video).duration_seconds, abs=1e-3)
    assert result.capture.source is CaptureSource.FILENAME
    legacy = classify_file(video, None, PipelineConfig.legacy_desktop(), FakeModels())
    assert legacy.sha256 is None and legacy.capture is None  # los presets legacy no cambian
