# SPDX-License-Identifier: AGPL-3.0-only OR MIT
"""Enumeraciones compartidas por el contrato de la IA y el paquete de escritorio."""

from enum import Enum


class _StrEnum(str, Enum):
    """``str`` + ``Enum`` compatible con Python 3.9+ (``enum.StrEnum`` es 3.11+)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Category(_StrEnum):
    ANIMAL = "animal"
    PERSON = "person"
    VEHICLE = "vehicle"
    BLANK = "blank"


class Rank(_StrEnum):
    """Nivel taxonómico alcanzado, de más general a más específico."""

    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"


# Orden de los niveles taxonómicos, de más general a más específico.
RANK_ORDER = (Rank.CLASS, Rank.ORDER, Rank.FAMILY, Rank.GENUS, Rank.SPECIES)


class CaptureSource(_StrEnum):
    EXIF = "exif"
    CONTAINER = "container"
    FILENAME = "filename"
    NONE = "none"


class MediaType(_StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class LineStatus(_StrEnum):
    DONE = "done"
    ERROR = "error"


class DesktopActionType(_StrEnum):
    COPY = "copy"
    MOVE = "move"
    LEFT_IN_PLACE = "left_in_place"


# Categorías que siempre llevan fotograma de evidencia.
CATEGORIES_WITH_EVIDENCE = frozenset({Category.ANIMAL, Category.PERSON, Category.VEHICLE})
