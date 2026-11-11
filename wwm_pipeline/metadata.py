"""Metadatos de un archivo: huella, dimensiones, duración y fecha de captura (HU-IA-20).

Sin binarios externos (no usa ``ffprobe``) y sin red:

- ``file_sha256`` → ``sha256`` y ``size_bytes`` leyendo en bloques de 1 MB (nunca el archivo entero).
- ``probe`` → ``MediaInfo`` (ancho, alto, fps declarado, frames y duración). En videos de la familia
  MP4 la duración sale del contenedor (``mvhd.duration / timescale``): con FPS variable,
  ``frame_count / fps`` no es confiable.
- ``capture`` → ``contract.Capture``, con esta cadena de fuentes: EXIF ``DateTimeOriginal`` →
  ``mvhd.creation_time`` del contenedor → nombre del archivo → ``none``. Las fechas basura (0, 1904,
  1970, antes de 2000 o en el futuro) se descartan y se prueba la siguiente fuente.

**Zona horaria:** las cámaras trampa graban la hora **local** aunque el contenedor diga UTC. La
fecha se devuelve **sin zona** (hora local de la cámara), salvo que EXIF traiga ``OffsetTimeOriginal``.
"""

import hashlib
import logging
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from .capture_parse import parse_datetime_from_filename
from .contract import Capture
from .enums import CaptureSource, MediaType

logger = logging.getLogger("wwm_pipeline.metadata")

CHUNK = 1024 * 1024
MP4_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".3gp"})
MIN_VALID = datetime(2000, 1, 1)
DURATION_MISMATCH = 0.10  # diferencia relativa entre contenedor y frames/fps que se registra en el log
_MP4_EPOCH = datetime(1904, 1, 1)
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_OFFSET_TIME_ORIGINAL = 36881


# ---------------------------------------------------------------------------
# Huella
# ---------------------------------------------------------------------------
def file_sha256(path) -> Tuple[str, int]:
    """``(sha256 en hex, tamaño en bytes)`` en una lectura secuencial por bloques."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


# ---------------------------------------------------------------------------
# Contenedor MP4 (moov → mvhd)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mvhd:
    creation_time: int  # segundos desde 1904-01-01 (0 = sin fecha)
    timescale: int
    duration: int

    @property
    def duration_seconds(self) -> Optional[float]:
        return self.duration / self.timescale if self.timescale > 0 and self.duration > 0 else None

    @property
    def created(self) -> Optional[datetime]:
        if self.creation_time <= 0:
            return None
        try:
            return _MP4_EPOCH + timedelta(seconds=self.creation_time)
        except OverflowError:
            return None


def _boxes(fh: BinaryIO, start: int, end: int):
    """Recorre las cajas ``(tipo, inicio_del_contenido, fin)`` entre ``start`` y ``end`` sin leer su contenido."""
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        header = fh.read(8)
        if len(header) < 8:
            return
        size, kind = struct.unpack(">I4s", header)
        body = pos + 8
        if size == 1:  # tamaño de 64 bits
            ext = fh.read(8)
            if len(ext) < 8:
                return
            size = struct.unpack(">Q", ext)[0]
            body = pos + 16
        elif size == 0:  # hasta el final
            size = end - pos
        if size < body - pos:
            return  # caja corrupta
        yield kind, body, min(pos + size, end)
        pos += size


def read_mvhd(path) -> Optional[Mvhd]:
    """El átomo ``mvhd`` de un MP4/MOV, o ``None`` si no se encuentra o está dañado."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            file_end = fh.tell()
            for kind, body, end in _boxes(fh, 0, file_end):
                if kind != b"moov":
                    continue
                for sub, sbody, _send in _boxes(fh, body, end):
                    if sub != b"mvhd":
                        continue
                    fh.seek(sbody)
                    version = fh.read(4)[0]
                    if version == 1:
                        data = fh.read(28)
                        creation, _mod, timescale, duration = struct.unpack(">QQIQ", data)
                    else:
                        data = fh.read(16)
                        creation, _mod, timescale, duration = struct.unpack(">IIII", data)
                    return Mvhd(creation, timescale, duration)
                return None
    except (OSError, struct.error, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Dimensiones y duración
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MediaInfo:
    media_type: MediaType
    width: int
    height: int
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None


def probe(path, media_type: MediaType) -> MediaInfo:
    path = Path(path)
    if media_type is MediaType.IMAGE:
        from PIL import Image

        with Image.open(path) as img:  # lee solo la cabecera
            width, height = img.size
        return MediaInfo(MediaType.IMAGE, width, height)

    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    finally:
        cap.release()
    by_frames = frame_count / fps if fps and frame_count else None
    mvhd = read_mvhd(path) if path.suffix.lower() in MP4_EXTENSIONS else None
    by_container = mvhd.duration_seconds if mvhd else None
    duration = by_container if by_container is not None else by_frames
    if by_container and by_frames and abs(by_container - by_frames) > DURATION_MISMATCH * by_container:
        logger.info(
            "metadata.duration_mismatch",
            extra={"file": path.name, "container_s": round(by_container, 3), "frames_s": round(by_frames, 3)},
        )
    return MediaInfo(
        MediaType.VIDEO,
        width,
        height,
        round(fps, 3) if fps else None,
        frame_count,
        round(duration, 3) if duration is not None else None,
    )


# ---------------------------------------------------------------------------
# Fecha de captura
# ---------------------------------------------------------------------------
def is_plausible(value: Optional[datetime], now: Optional[datetime] = None) -> bool:
    if value is None:
        return False
    naive = value.replace(tzinfo=None)
    now = now or datetime.now()
    return MIN_VALID <= naive <= now + timedelta(days=1)


def _parse_offset(text: str) -> Optional[timezone]:
    text = text.strip()
    try:
        sign = -1 if text[0] == "-" else 1
        hours, minutes = int(text[1:3]), int(text[4:6])
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    except (ValueError, IndexError):
        return None


def exif_datetime(path) -> Optional[datetime]:
    """``DateTimeOriginal`` (+ ``OffsetTimeOriginal`` si existe) de una imagen, o ``None``."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            ifd = img.getexif().get_ifd(_EXIF_IFD)
    except Exception:  # imagen sin EXIF o ilegible: se prueba la siguiente fuente
        return None
    raw = ifd.get(_DATETIME_ORIGINAL)
    if not raw:
        return None
    try:
        value = datetime.strptime(str(raw).strip("\x00 ").strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    offset = ifd.get(_OFFSET_TIME_ORIGINAL)
    tz = _parse_offset(str(offset)) if offset else None
    return value.replace(tzinfo=tz) if tz else value


def capture(path, media_type: MediaType, file_name: Optional[str] = None, now: Optional[datetime] = None) -> Capture:
    """La fecha de captura y de dónde salió (EXIF → contenedor → nombre → none)."""
    path = Path(path)
    file_name = file_name or path.name
    if media_type is MediaType.IMAGE:
        value = exif_datetime(path)
        if is_plausible(value, now):
            return Capture(captured_at=value, source=CaptureSource.EXIF)
    ext = Path(file_name).suffix.lower() or path.suffix.lower()
    if media_type is MediaType.VIDEO and ext in MP4_EXTENSIONS:
        mvhd = read_mvhd(path)
        value = mvhd.created if mvhd else None
        if is_plausible(value, now):
            return Capture(captured_at=value, source=CaptureSource.CONTAINER)
    value = parse_datetime_from_filename(file_name)
    if is_plausible(value, now):
        return Capture(captured_at=value, source=CaptureSource.FILENAME)
    return Capture(captured_at=None, source=CaptureSource.NONE)
