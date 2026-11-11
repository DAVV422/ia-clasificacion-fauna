"""Contrato IA → Back (``POST /api/v1/detect``).

Fuente de verdad: ``docs-interno/plan-implementacion/contratos/API_IA_BACK.md``.
El back tiene una copia idéntica (o usa ``schemas/detect_result.schema.json``)
y un test verifica que coincidan. Cualquier cambio incompatible sube el MAJOR
de ``CONTRACT_VERSION`` y se avisa al equipo.

Solo depende de pydantic y de la biblioteca estándar.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    CATEGORIES_WITH_EVIDENCE,
    RANK_ORDER,
    CaptureSource,
    Category,
    MediaType,
    Rank,
)

CONTRACT_VERSION = "1.0.0"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
EVIDENCE_MAX_PX = 960


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Taxonomía
# ---------------------------------------------------------------------------
_LEVEL_FIELDS = ("class_", "order", "family", "genus", "species")


class Taxonomy(_Strict):
    """Taxonomía de una etiqueta de SpeciesNet.

    Los niveles por debajo de ``rank`` van en ``None``. ``scientific_name``
    corresponde al nivel de ``rank`` (``"Panthera"`` si ``rank=genus``).
    """

    label_id: str = Field(min_length=1)
    label_raw: str = Field(min_length=1)
    class_: Optional[str] = Field(default=None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    scientific_name: str = Field(min_length=1)
    common_name: Optional[str] = None
    rank: Rank

    @model_validator(mode="after")
    def _levels_match_rank(self) -> "Taxonomy":
        depth = RANK_ORDER.index(self.rank)
        for i, field in enumerate(_LEVEL_FIELDS):
            value = getattr(self, field)
            if i <= depth and not value:
                raise ValueError(f"taxonomy.{field.rstrip('_')} es obligatorio con rank={self.rank.value}")
            if i > depth and value:
                raise ValueError(f"taxonomy.{field.rstrip('_')} debe ser null con rank={self.rank.value}")
        return self


class Alternative(_Strict):
    label_id: str = Field(min_length=1)
    scientific_name: str = Field(min_length=1)
    common_name: Optional[str] = None
    rank: Rank
    score: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Partes de la respuesta
# ---------------------------------------------------------------------------
class Media(_Strict):
    file_name: str = Field(min_length=1)
    media_type: MediaType
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    fps: Optional[float] = Field(default=None, gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @model_validator(mode="after")
    def _image_has_no_time(self) -> "Media":
        if self.media_type is MediaType.IMAGE and (self.duration_seconds is not None or self.fps is not None):
            raise ValueError("media.duration_seconds y media.fps deben ser null en imágenes")
        if self.media_type is MediaType.VIDEO and (self.duration_seconds is None or self.fps is None):
            raise ValueError("media.duration_seconds y media.fps son obligatorios en videos")
        return self


class Capture(_Strict):
    captured_at: Optional[datetime] = None
    source: CaptureSource

    @model_validator(mode="after")
    def _source_matches_value(self) -> "Capture":
        if self.source is CaptureSource.NONE and self.captured_at is not None:
            raise ValueError("capture.captured_at debe ser null con source=none")
        if self.source is not CaptureSource.NONE and self.captured_at is None:
            raise ValueError("capture.captured_at es obligatorio salvo con source=none")
        return self


class Result(_Strict):
    category: Category
    taxonomy: Optional[Taxonomy] = None
    confidence: float = Field(ge=0, le=1)
    alternatives: List[Alternative] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _taxonomy_only_for_animals(self) -> "Result":
        is_animal = self.category is Category.ANIMAL
        if is_animal and self.taxonomy is None:
            raise ValueError("result.taxonomy es obligatorio con category=animal")
        if not is_animal and self.taxonomy is not None:
            raise ValueError("result.taxonomy debe ser null si category no es animal")
        if not is_animal and self.alternatives:
            raise ValueError("result.alternatives debe estar vacío si category no es animal")
        return self


BBox = Tuple[float, float, float, float]


def _check_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("bbox debe ser [x1, y1, x2, y2] normalizado con 0 ≤ x1 < x2 ≤ 1 y 0 ≤ y1 < y2 ≤ 1")


class Evidence(_Strict):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox: BBox
    detection_confidence: float = Field(ge=0, le=1)
    image_base64: str = Field(min_length=1)
    image_mime: Literal["image/jpeg"] = "image/jpeg"
    width: int = Field(gt=0, le=EVIDENCE_MAX_PX)
    height: int = Field(gt=0, le=EVIDENCE_MAX_PX)

    @model_validator(mode="after")
    def _valid_bbox(self) -> "Evidence":
        _check_bbox(self.bbox)
        return self


class Vote(_Strict):
    label_id: str
    scientific_name: str
    frames: int = Field(ge=0)
    score: float = Field(ge=0, le=1)


class Analysis(_Strict):
    frames_sampled: int = Field(ge=0)
    frames_with_animal: int = Field(ge=0)
    frames_discarded: Dict[str, int] = Field(default_factory=dict)
    votes: List[Vote] = Field(default_factory=list)
    # F1-D10: un video más largo que MAX_VIDEO_SECONDS no se rechaza; se analiza hasta ese segundo y se informa aquí.
    truncated_at_seconds: Optional[float] = Field(default=None, gt=0)
    processing_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _animal_frames_le_sampled(self) -> "Analysis":
        if self.frames_with_animal > self.frames_sampled:
            raise ValueError("analysis.frames_with_animal no puede superar frames_sampled")
        return self


class ModelInfo(_Strict):
    detector: str = Field(min_length=1)
    classifier: str = Field(min_length=1)
    sampling: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)


class DetectResult(_Strict):
    """Contenido de ``data`` en una respuesta exitosa de ``POST /detect``."""

    media: Media
    capture: Capture
    result: Result
    evidence: Optional[Evidence] = None
    analysis: Analysis
    model: ModelInfo

    @model_validator(mode="after")
    def _evidence_rules(self) -> "DetectResult":
        needs_evidence = self.result.category in CATEGORIES_WITH_EVIDENCE
        if needs_evidence and self.evidence is None:
            raise ValueError(f"evidence es obligatorio con category={self.result.category.value}")
        if not needs_evidence and self.evidence is not None:
            raise ValueError("evidence debe ser null con category=blank")
        if self.evidence is not None and self.media.media_type is MediaType.IMAGE:
            if self.evidence.frame_index != 0 or self.evidence.timestamp_seconds != 0:
                raise ValueError("en imágenes, evidence.frame_index y timestamp_seconds deben ser 0")
        return self


# ---------------------------------------------------------------------------
# Envoltura estándar (ESTANDARES_BACKEND_IA §4)
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    field: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class ErrorBody(_Strict):
    code: str = Field(min_length=1)
    flow: Optional[str] = None
    details: List[ErrorDetail] = Field(default_factory=list)


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str = Field(min_length=1)
    timestamp: datetime


T = TypeVar("T")


class ApiEnvelope(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    data: Optional[T] = None
    message: str = ""
    error: Optional[ErrorBody] = None
    meta: Meta

    @model_validator(mode="after")
    def _status_consistency(self) -> "ApiEnvelope[T]":
        if self.status == "success" and (self.data is None or self.error is not None):
            raise ValueError("status=success requiere data y error=null")
        if self.status == "error" and (self.error is None or self.data is not None):
            raise ValueError("status=error requiere error y data=null")
        return self


DetectEnvelope = ApiEnvelope[DetectResult]

# Códigos de error que puede devolver la IA (API_IA_BACK §2.5).
IA_ERROR_HTTP = {
    "IA_UNAUTHORIZED": 401,
    "IA_FILE_TOO_LARGE": 413,
    "IA_INVALID_FORMAT": 415,
    "IA_DECODE_FAILED": 422,
    "IA_DURATION_EXCEEDED": 422,
    "IA_MODEL_NOT_READY": 503,
    "IA_BUSY": 503,
    "IA_PROCESSING_TIMEOUT": 504,
    "IA_INTERNAL": 500,
}
