"""La línea de ``clasificaciones.jsonl`` de cada archivo (HU-DS-02, PKG §3).

Aquí solo se **arma** y se **valida** con ``wwm_pipeline.package.PackageLine``. Escribirla a disco (con
``flush`` + ``fsync``) y reanudar es HU-DS-14 (fase 3).
"""

from pathlib import Path
from typing import Optional

from wwm_pipeline.config import IMAGE_EXTENSIONS
from wwm_pipeline.enums import Category, DesktopActionType
from wwm_pipeline.package import DesktopAction, PackageLine

LEFT_IN_PLACE = DesktopAction(type=DesktopActionType.LEFT_IN_PLACE)


def media_type_for(file_name: str) -> str:
    return "image" if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS else "video"


def evidence_ref(result) -> Optional[dict]:
    """La referencia al fotograma dentro del paquete; solo para fauna (F2-D2, F2-D9)."""
    ev = result.evidence
    if result.decision.category != Category.ANIMAL.value or ev is None:
        return None
    return {
        "file": f"evidence/{result.sha256}.jpg",
        "frame_index": ev.frame_index,
        "timestamp_seconds": round(ev.timestamp_seconds, 2),
        "bbox": list(ev.bbox),
        "width": ev.width,
        "height": ev.clean_height,  # el fotograma sin la franja, que es el que se exporta (F2-D1)
    }


def build_line(result, action: DesktopAction = LEFT_IN_PLACE) -> PackageLine:
    """La línea de un archivo clasificado (``status=done``)."""
    vote = result.vote
    capture = result.capture
    is_video = result.media_type == "video"
    return PackageLine.model_validate(
        {
            "file_name": result.file_name,
            "file_sha256": result.sha256,
            "file_size_bytes": result.size_bytes,
            "media_type": result.media_type,
            "duration_seconds": result.duration_seconds if is_video else None,
            "captured_at": capture.captured_at if capture else None,
            "captured_at_source": capture.source if capture else "none",
            "status": "done",
            "error_code": None,
            "category": vote.category,
            "taxonomy": vote.taxonomy.model_dump(by_alias=True) if vote.taxonomy else None,
            "confidence": vote.confidence,
            "alternatives": [a.model_dump() for a in vote.alternatives],
            "frames_analyzed": result.frames_sampled,
            "evidence": evidence_ref(result),
            "desktop_action": action.model_dump(),
        }
    )


def error_line(file_name: str, sha256: str, size_bytes: int, error_code: str) -> PackageLine:
    """La línea de un archivo con error: queda en el origen, sin categoría ni fotograma."""
    return PackageLine.model_validate(
        {
            "file_name": file_name,
            "file_sha256": sha256,
            "file_size_bytes": size_bytes,
            "media_type": media_type_for(file_name),
            "captured_at": None,
            "captured_at_source": "none",
            "status": "error",
            "error_code": error_code,
            "desktop_action": LEFT_IN_PLACE.model_dump(),
        }
    )
