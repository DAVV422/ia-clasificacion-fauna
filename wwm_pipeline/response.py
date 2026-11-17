"""De ``FileResult`` (lo que devuelve el *pipeline*) a ``DetectResult`` (el contrato IA → back) — HU-IA-41.

Puro: sin FastAPI. Lo usa la API (``api/routes.py``) y se prueba con ``FakeModels``. Sigue IA §2.3:

- ``media``: lo del archivo (``sha256``, tamaño, duración del contenedor, fps, dimensiones);
- ``capture``: EXIF → contenedor → nombre → ``none``. ``client_mtime`` no se usa aquí: con ``none``, lo aplica el
  back (E2-D4);
- ``result``: la votación (categoría, taxonomía completa, confianza, alternativas);
- ``evidence``: el fotograma completo con recuadro, **sin franja**, JPEG ≤ 960 px en base64 (``animal``,
  ``person`` y ``vehicle``; ``null`` en ``blank``);
- ``analysis``: el diagnóstico (frames, votos, truncado, ``low_evidence``, tiempo);
- ``model``: qué detector, clasificador y muestreo produjeron el resultado.

Si el resultado no cumple el contrato, ``to_detect_result`` lanza ``PipelineError("IA_INTERNAL")``: la IA nunca
responde fuera de contrato.
"""

from typing import Any, Dict, Optional

from pydantic import ValidationError

from . import __version__
from .contract import Analysis, Capture, DetectResult, Media, ModelInfo, Result
from .enums import CaptureSource, Category
from .errors import PipelineError

DETECTOR_NAME = "MDv5a"


def model_info(sampling: str, classifier: str, detector: str = DETECTOR_NAME) -> ModelInfo:
    return ModelInfo(detector=detector, classifier=classifier, sampling=sampling, pipeline_version=__version__)


def _media(result, file_name: str) -> Media:
    is_video = result.media_type == "video"
    return Media(
        file_name=file_name,
        media_type=result.media_type,
        sha256=result.sha256,
        size_bytes=result.size_bytes,
        duration_seconds=result.duration_seconds if is_video else None,
        fps=result.fps if is_video else None,
        width=result.width,
        height=result.height,
    )


def _capture(result) -> Capture:
    if result.capture is not None:
        return Capture.model_validate(
            result.capture.model_dump() if hasattr(result.capture, "model_dump") else result.capture
        )
    return Capture(captured_at=None, source=CaptureSource.NONE)


def _result(vote) -> Result:
    is_animal = vote.category is Category.ANIMAL
    return Result(
        category=vote.category,
        taxonomy=vote.taxonomy if is_animal else None,
        confidence=round(float(vote.confidence), 4),
        alternatives=list(vote.alternatives)[:3] if is_animal else [],
    )


def _analysis(result, vote) -> Analysis:
    sampled = result.frames_sampled
    return Analysis(
        frames_sampled=sampled,
        frames_with_animal=min(result.frames_with_animal, sampled),
        frames_discarded=dict(result.frames_discarded or {}),
        votes=list(vote.votes),
        truncated_at_seconds=result.truncated_at_seconds,
        low_evidence=bool(vote.low_evidence),
        processing_ms=int(result.processing_ms),
    )


def to_detect_result(result, file_name: str, classifier: str, processing_ms: Optional[int] = None) -> DetectResult:
    """El ``data`` de ``POST /detect``. ``classifier`` es ``ModelRegistry.describe()["classifier"]``."""
    if result.vote is None:  # los presets legacy no votan: no producen un resultado del contrato
        raise PipelineError("IA_INTERNAL", "el preset usado no produce un resultado del contrato (falta la votación)")
    if processing_ms is not None:
        result.processing_ms = processing_ms
    vote = result.vote
    data: Dict[str, Any] = {
        "media": _media(result, file_name),
        "capture": _capture(result),
        "result": _result(vote),
        "evidence": result.evidence.to_contract()
        if result.evidence is not None and vote.category is not Category.BLANK
        else None,
        "analysis": _analysis(result, vote),
        "model": model_info("image" if result.media_type == "image" else result.sampling, classifier),
    }
    plain = {k: (v.model_dump(by_alias=True) if hasattr(v, "model_dump") else v) for k, v in data.items()}
    try:
        return DetectResult.model_validate(plain)
    except ValidationError as exc:
        raise PipelineError("IA_INTERNAL", f"el resultado no cumple el contrato: {exc.errors()[:3]}") from exc
