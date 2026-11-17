"""HU-IA-32 · Extensión + firma binaria (wwm_pipeline/formats.py)."""

import os
from pathlib import Path

import numpy as np
import pytest

from wwm_pipeline import formats

cv2 = pytest.importorskip("cv2")
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", Path(__file__).resolve().parents[3] / "docs-interno" / "estacion_1"))


def image_bytes(ext: str) -> bytes:
    ok, buf = cv2.imencode(ext, np.full((20, 30, 3), 120, np.uint8))
    assert ok
    return buf.tobytes()


def video_bytes(tmp_path, ext: str, fourcc: str) -> bytes:
    path = tmp_path / f"v{ext}"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 100, np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.mark.parametrize(
    "ext,kind", [(".jpg", "jpeg"), (".png", "png"), (".bmp", "bmp"), (".tiff", "tiff"), (".webp", "webp")]
)
def test_sniff_images(ext, kind):
    data = image_bytes(ext)
    assert formats.sniff(data[:512]) == kind
    assert formats.check(f"foto{ext}", data[:512]) == (True, "")


@pytest.mark.parametrize("ext,fourcc,kind", [(".mp4", "mp4v", "mp4"), (".avi", "MJPG", "avi")])
def test_sniff_videos_written_by_opencv(tmp_path, ext, fourcc, kind):
    data = video_bytes(tmp_path, ext, fourcc)
    assert formats.sniff(data[:512]) == kind and formats.check(f"clip{ext}", data[:512])[0]


@pytest.mark.parametrize(
    "head,kind",
    [
        (b"\x1a\x45\xdf\xa3" + b"\x00" * 60, "mkv"),
        (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11" + b"\x00" * 60, "wmv"),
        (b"FLV\x01" + b"\x00" * 60, "flv"),
        (bytes([0x47]) + b"\x00" * 187 + bytes([0x47]) + b"\x00" * 50, "mpegts"),
        (b"\x00" * 4 + bytes([0x47]) + b"\x00" * 191 + bytes([0x47]) + b"\x00" * 50, "mpegts"),
        (b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 40, "mp4"),  # MOV
        (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 40, "heic"),
    ],
)
def test_sniff_known_headers(head, kind):
    assert formats.sniff(head) == kind


def test_real_camera_files_pass():
    """Los videos y la imagen de estacion_1 (WhatsApp, cámaras trampa) pasan la validación."""
    files = sorted(ESTACION_1.glob("*.mp4"))[:3] + sorted(ESTACION_1.glob("*.jpg"))[:1] if ESTACION_1.is_dir() else []
    if not files:
        pytest.skip("requiere estacion_1")
    for path in files:
        with open(path, "rb") as f:
            assert formats.check(path.name, f.read(512)) == (True, ""), path.name


@pytest.mark.parametrize(
    "name,head,fragment",
    [
        ("notas.txt", b"hola mundo", "extensión no soportada"),
        ("sin_extension", b"\xff\xd8\xff\xe0", "extensión no soportada"),
        ("video.mp4", b"hola, soy un texto", "no corresponde"),  # disfrazado
        ("foto.jpg", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20, "la extensión dice image"),  # un video con .jpg
        ("IMG_0001.HEIC", b"\x00\x00\x00\x18ftypheic", "HEIC"),
        ("IMG_0001.jpg", b"\x00\x00\x00\x18ftypheic", "HEIC"),  # HEIC renombrado
        ("foto.cr2", b"II*\x00 raw", "extensión no soportada"),
    ],
)
def test_rejected(name, head, fragment):
    ok, reason = formats.check(name, head)
    assert not ok and fragment in reason


def test_config_reexports_the_same_sets():
    from wwm_pipeline import config

    assert config.IMAGE_EXTENSIONS is formats.IMAGE_EXTENSIONS and config.VIDEO_EXTENSIONS is formats.VIDEO_EXTENSIONS
