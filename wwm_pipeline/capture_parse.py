# SPDX-License-Identifier: AGPL-3.0-only OR MIT
"""Fecha y hora de captura a partir del nombre del archivo.

Python puro (sin dependencias). Lo usan ``ia-fake`` y, en la fase 1, ``metadata.py`` (IA-20)
como tercera fuente después de EXIF y del contenedor del video.

Formatos reconocidos (los más comunes en cámaras trampa y celulares):
    CAM01_20260315_042210.mp4        AAAAMMDD_HHMMSS (también con '-' o sin separador)
    IMG_20260315-042210.jpg
    2026-03-15 04-22-10.avi          AAAA-MM-DD HH-MM-SS (también con '_', '.' o ':')
    2026-03-15_04.22.10.MOV
    VID_2026_03_15_04_22_10.mp4      AAAA_MM_DD_HH_MM_SS
"""

import re
from datetime import datetime
from typing import Optional

_PATTERNS = [
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_\-T ]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[ _T\-](\d{2})[-_.:](\d{2})[-_.:](\d{2})(?!\d)"),
]
_MIN_YEAR = 1990


def parse_datetime_from_filename(file_name: str) -> Optional[datetime]:
    """Devuelve la fecha del nombre, o ``None`` si no hay ninguna válida."""
    for pattern in _PATTERNS:
        for match in pattern.finditer(file_name):
            year, month, day, hour, minute, second = (int(g) for g in match.groups())
            if year < _MIN_YEAR:
                continue
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue
    return None
