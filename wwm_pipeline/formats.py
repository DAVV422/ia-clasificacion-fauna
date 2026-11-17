# SPDX-License-Identifier: AGPL-3.0-only OR MIT
"""¿Qué es este archivo de verdad? La extensión **y** la firma binaria (HU-IA-32; lo mínimo de IA-16).

Puro, sin dependencias: lo usa la API y lo puede copiar el back (BE-21) para validar lo mismo antes de encolar.

- La **extensión** tiene que ser una de las que el *pipeline* lee hoy (``IMAGE_EXTENSIONS`` / ``VIDEO_EXTENSIONS``,
  definidas aquí y reexportadas por ``config``). DD-18 agrega ``tif``, ``m4v``, ``3gp``, ``mts``, ``m2ts`` y HEIC:
  llegan con IA-16, cuando haya archivos de prueba de cada uno.
- La **firma** (los primeros bytes) tiene que ser de la misma **familia** (imagen o video). No se exige la extensión
  exacta: un ``.jpg`` que en realidad es PNG se lee igual (OpenCV decide por el contenido), pero un ``.txt``
  renombrado a ``.mp4`` no.
- **HEIC/HEIF** (fotos de celular) se reconocen, pero todavía no se decodifican: ``IA_INVALID_FORMAT`` con un mensaje
  que lo dice, hasta IA-16 (``pillow-heif``).
"""

from pathlib import Path
from typing import Optional, Tuple

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"})
HEAD_BYTES = 512
ACCEPTED_EXTENSIONS = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}

FAMILY = {
    "jpeg": "image",
    "png": "image",
    "bmp": "image",
    "tiff": "image",
    "webp": "image",
    "heic": "image",
    "mp4": "video",  # ISO BMFF: mp4, mov, m4v, 3gp
    "avi": "video",
    "mkv": "video",  # EBML: mkv, webm
    "wmv": "video",  # ASF
    "flv": "video",
    "mpegts": "video",  # mts, m2ts
}


def sniff(head: bytes) -> Optional[str]:
    """El formato según los primeros bytes, o ``None`` si no es ninguno conocido."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[4:8] == b"ftyp":
        return "heic" if head[8:12] in _HEIC_BRANDS else "mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "mkv"
    if head.startswith(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"):
        return "wmv"
    if head.startswith(b"FLV"):
        return "flv"
    if len(head) > 188 and head[0] == 0x47 and head[188] == 0x47:
        return "mpegts"
    if len(head) > 196 and head[4] == 0x47 and head[196] == 0x47:  # M2TS: 4 bytes de timecode por paquete
        return "mpegts"
    return None


def check(file_name: str, head: bytes) -> Tuple[bool, str]:
    """``(True, "")`` si se puede procesar; si no, ``(False, motivo)`` para ``IA_INVALID_FORMAT``."""
    ext = Path(file_name).suffix.lower()
    kind = sniff(head)
    if ext in HEIC_EXTENSIONS or kind == "heic":
        return False, "HEIC/HEIF todavía no se admite (llega con IA-16); conviértelo a JPEG"
    if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        return False, f"extensión no soportada: {ext or '(sin extensión)'}"
    if kind is None:
        return False, "el contenido no corresponde a ninguna imagen ni video conocido"
    expected = "image" if ext in IMAGE_EXTENSIONS else "video"
    if FAMILY[kind] != expected:
        return False, f"la extensión dice {expected}, pero el contenido es {FAMILY[kind]} ({kind})"
    return True, ""
