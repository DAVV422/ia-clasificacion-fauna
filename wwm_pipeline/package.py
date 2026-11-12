"""Formato del paquete de sincronización escritorio → web (``.wwm.zip``, formato 1.x).

Fuente de verdad: ``docs-interno/plan-implementacion/contratos/PAQUETE_ESCRITORIO.md``.
Solo depende de pydantic y de la biblioteca estándar.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contract import SHA256_PATTERN, Alternative, BBox, ModelInfo, Taxonomy, _check_bbox
from .enums import (
    CaptureSource,
    Category,
    DesktopActionType,
    LineStatus,
    MediaType,
)

PACKAGE_FORMAT = "wwm-desktop-export"
SUPPORTED_MAJOR = 1
JSONL_NAME = "classifications.jsonl"
MANIFEST_NAME = "manifest.json"
EVIDENCE_PATH_PATTERN = r"^evidence/[0-9a-f]{64}\.jpg$"
# F2-D2 / F2-D9: el escritorio solo guarda fotogramas de fauna (privacidad). En el paquete, las líneas de
# persona y vehículo van sin evidence. (En el contrato IA → back de la web no cambia: ver contract.py.)
CATEGORIES_WITH_EVIDENCE_IN_PACKAGE = frozenset({Category.ANIMAL})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _Lenient(BaseModel):
    """Los MINOR nuevos solo agregan campos opcionales: se ignoran los desconocidos."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------
class CameraInfo(_Lenient):
    code: Optional[str] = Field(default=None, max_length=50)
    name: Optional[str] = Field(default=None, max_length=120)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    deployed_at: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    still_installed: Optional[bool] = None
    observations: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _consistency(self) -> "CameraInfo":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("camera.latitude y camera.longitude van juntas")
        if self.deployed_at and self.retrieved_at and self.retrieved_at < self.deployed_at:
            raise ValueError("camera.retrieved_at no puede ser anterior a deployed_at")
        if self.still_installed and self.retrieved_at is not None:
            raise ValueError("camera.retrieved_at debe ser null si still_installed=true")
        return self

    @property
    def is_complete(self) -> bool:
        """Tiene lo mínimo para crear la cámara en la web (API §1.5)."""
        return bool(self.code and self.latitude is not None and self.deployed_at and self.still_installed is not None)


class RunInfo(_Lenient):
    job_id: str
    started_at: datetime
    finished_at: datetime
    action: DesktopActionType

    @field_validator("action")
    @classmethod
    def _copy_or_move(cls, v: DesktopActionType) -> DesktopActionType:
        if v is DesktopActionType.LEFT_IN_PLACE:
            raise ValueError("run.action debe ser copy o move")
        return v


class Counts(_Lenient):
    lines: int = Field(ge=0)
    animal: int = Field(ge=0)
    person: int = Field(ge=0)
    vehicle: int = Field(ge=0)
    blank: int = Field(ge=0)
    error: int = Field(ge=0)
    evidence_files: int = Field(ge=0)


class Manifest(_Lenient):
    format: str
    format_version: str = Field(pattern=r"^\d+\.\d+$")
    app_version: str
    created_at: datetime
    device_id: str = Field(min_length=1, max_length=64)
    camera_code: Optional[str] = Field(default=None, max_length=50)
    camera: Optional[CameraInfo] = None
    source_folder_name: str = Field(min_length=1, max_length=255)
    run: RunInfo
    model: ModelInfo
    counts: Counts
    checksums: Dict[str, str]

    @field_validator("format")
    @classmethod
    def _format(cls, v: str) -> str:
        if v != PACKAGE_FORMAT:
            raise ValueError(f"format debe ser {PACKAGE_FORMAT!r}")
        return v

    @field_validator("source_folder_name")
    @classmethod
    def _only_folder_name(cls, v: str) -> str:
        if re.search(r"[\\/:]", v):
            raise ValueError("source_folder_name debe ser solo el nombre de la carpeta, sin rutas")
        return v

    @field_validator("checksums")
    @classmethod
    def _checksum_format(cls, v: Dict[str, str]) -> Dict[str, str]:
        value = v.get(JSONL_NAME, "")
        if not re.match(r"^sha256:[0-9a-f]{64}$", value):
            raise ValueError(f"checksums[{JSONL_NAME!r}] debe ser 'sha256:<64 hex>'")
        return v

    @property
    def major(self) -> int:
        return int(self.format_version.split(".")[0])


# ---------------------------------------------------------------------------
# classifications.jsonl
# ---------------------------------------------------------------------------
class EvidenceRef(_Lenient):
    file: str = Field(pattern=EVIDENCE_PATH_PATTERN)
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox: BBox
    width: int = Field(gt=0, le=960)
    height: int = Field(gt=0, le=960)

    @model_validator(mode="after")
    def _valid_bbox(self) -> "EvidenceRef":
        _check_bbox(self.bbox)
        return self


class DesktopAction(_Lenient):
    type: DesktopActionType
    destination_folder: Optional[str] = None

    @model_validator(mode="after")
    def _destination_when_relocated(self) -> "DesktopAction":
        relocated = self.type in (DesktopActionType.COPY, DesktopActionType.MOVE)
        if relocated and not self.destination_folder:
            raise ValueError("desktop_action.destination_folder es obligatorio con copy o move")
        if not relocated and self.destination_folder:
            raise ValueError("desktop_action.destination_folder debe ser null con left_in_place")
        return self


class PackageLine(_Lenient):
    file_name: str = Field(min_length=1)
    file_sha256: str = Field(pattern=SHA256_PATTERN)
    file_size_bytes: int = Field(ge=0)
    media_type: MediaType
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    captured_at: Optional[datetime] = None
    captured_at_source: CaptureSource
    status: LineStatus
    error_code: Optional[str] = None
    category: Optional[Category] = None
    taxonomy: Optional[Taxonomy] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    alternatives: List[Alternative] = Field(default_factory=list, max_length=3)
    frames_analyzed: Optional[int] = Field(default=None, ge=0)
    evidence: Optional[EvidenceRef] = None
    desktop_action: DesktopAction

    @model_validator(mode="after")
    def _rules(self) -> "PackageLine":
        if self.media_type is MediaType.IMAGE and self.duration_seconds is not None:
            raise ValueError("duration_seconds debe ser null en imágenes")
        if (self.captured_at_source is CaptureSource.NONE) != (self.captured_at is None):
            raise ValueError("captured_at debe ser null si y solo si captured_at_source=none")

        if self.status is LineStatus.ERROR:
            if not self.error_code:
                raise ValueError("error_code es obligatorio con status=error")
            if self.category is not None or self.taxonomy is not None or self.evidence is not None:
                raise ValueError("una línea con status=error no lleva category, taxonomy ni evidence")
            if self.desktop_action.type is not DesktopActionType.LEFT_IN_PLACE:
                raise ValueError("un archivo con error siempre queda en el origen (left_in_place)")
            return self

        # status = done
        if self.error_code:
            raise ValueError("error_code debe ser null con status=done")
        if self.category is None or self.confidence is None or self.frames_analyzed is None:
            raise ValueError("con status=done son obligatorios category, confidence y frames_analyzed")
        is_animal = self.category is Category.ANIMAL
        if is_animal != (self.taxonomy is not None):
            raise ValueError("taxonomy es obligatorio si y solo si category=animal")
        if not is_animal and self.alternatives:
            raise ValueError("alternatives debe estar vacío si category no es animal")
        needs_evidence = self.category in CATEGORIES_WITH_EVIDENCE_IN_PACKAGE
        if needs_evidence != (self.evidence is not None):
            raise ValueError(
                "evidence es obligatorio si y solo si category=animal (persona y vehículo van sin fotograma)"
            )
        if self.evidence is not None and self.evidence.file != f"evidence/{self.file_sha256}.jpg":
            raise ValueError("evidence.file debe ser evidence/<file_sha256>.jpg")
        if self.desktop_action.type is not DesktopActionType.LEFT_IN_PLACE and not is_animal:
            raise ValueError("solo los archivos con animal se copian o mueven")
        return self


def count_lines(lines: List[PackageLine]) -> Counts:
    """Calcula los ``counts`` que corresponden a un conjunto de líneas."""
    by_cat = {c: 0 for c in Category}
    errors = 0
    evidence = 0
    for line in lines:
        if line.status is LineStatus.ERROR:
            errors += 1
            continue
        by_cat[line.category] += 1  # type: ignore[index]
        if line.evidence is not None:
            evidence += 1
    return Counts(
        lines=len(lines),
        animal=by_cat[Category.ANIMAL],
        person=by_cat[Category.PERSON],
        vehicle=by_cat[Category.VEHICLE],
        blank=by_cat[Category.BLANK],
        error=errors,
        evidence_files=evidence,
    )
