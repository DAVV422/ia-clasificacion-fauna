"""Evidencia.

- Fase 0: el guardado opcional de recortes del prototipo (``save_crops``), solo en ``legacy_api``.
- Fase 1 (HU-IA-15): el **fotograma de evidencia** completo, con el recuadro del resultado ganador
  y, en escritorio, una franja con ``Panthera onca (jaguar) · 94 % · 00:07``. Se entrega como JPEG
  (≤ 960 px de lado mayor, sin EXIF) en ``EvidenceImage``; la API lo pasa a base64 con ``to_contract``.
"""

import base64
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .contract import EVIDENCE_MAX_PX, Evidence

logger = logging.getLogger("wwm_pipeline.evidence")

BBox = Tuple[float, float, float, float]
BOX_COLOR_BGR = (0, 220, 255)  # amarillo: contrasta con la vegetación y con el infrarrojo en gris
OUTLINE_BGR = (0, 0, 0)
CATEGORY_LABELS = {
    "en": {"person": "person", "vehicle": "vehicle"},
    "es": {"person": "persona", "vehicle": "vehículo"},
}


# ---------------------------------------------------------------------------
# Fase 0 · recortes del prototipo
# ---------------------------------------------------------------------------
def new_crop_name() -> str:
    return f"{uuid.uuid4()}.jpg"


def save_crop(
    crop_bgr, crops_dir: Path, run_name: str, category: str, crop_filename: str, url_prefix: str = "/detections"
) -> Tuple[Optional[str], Optional[str]]:
    """Guarda el recorte como el prototipo: ``<crops_dir>/<run_name>/<category>/<uuid>.jpg``.

    Devuelve (ruta relativa, URL) o (None, None) si no se pudo escribir.
    """
    import cv2

    category_dir = Path(crops_dir) / run_name / category
    category_dir.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(category_dir / crop_filename), crop_bgr):
        return (
            f"detections/{run_name}/{category}/{crop_filename}",
            f"{url_prefix}/{run_name}/{category}/{crop_filename}",
        )
    logger.error("No se pudo guardar el recorte %s", category_dir / crop_filename)
    return None, None


# ---------------------------------------------------------------------------
# Fase 1 · fotograma de evidencia (HU-IA-15)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceImage:
    frame_index: int
    timestamp_seconds: float
    bbox: BBox  # [x1, y1, x2, y2] normalizado respecto del frame original
    detection_confidence: float
    jpeg: bytes
    width: int
    height: int

    def to_contract(self) -> Evidence:
        return Evidence(
            frame_index=self.frame_index,
            timestamp_seconds=self.timestamp_seconds,
            bbox=self.bbox,
            detection_confidence=self.detection_confidence,
            image_base64=base64.b64encode(self.jpeg).decode("ascii"),
            width=self.width,
            height=self.height,
        )


def xyxy_from_prototype(box: Sequence[float]) -> BBox:
    """``[ymin, xmin, ymax, xmax]`` (formato del prototipo) → ``[x1, y1, x2, y2]`` (contrato), recortado a [0, 1]."""
    ymin, xmin, ymax, xmax = (min(1.0, max(0.0, float(v))) for v in box)
    return (xmin, ymin, xmax, ymax)


def mmss(seconds: float) -> str:
    """``00:07``; ``1:02:03`` desde una hora."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def label_text(
    category: str,
    confidence: float,
    timestamp_seconds: Optional[float],
    scientific_name: Optional[str] = None,
    common_name: Optional[str] = None,
    lang: str = "en",
) -> str:
    """``Panthera onca (jaguar) · 94 % · 00:07``. Sin ``mm:ss`` si ``timestamp_seconds`` es ``None`` (imágenes)."""
    if category == "animal" and scientific_name:
        name = f"{scientific_name} ({common_name})" if common_name else scientific_name
    else:
        name = CATEGORY_LABELS.get(lang, CATEGORY_LABELS["en"]).get(category, category)
    parts = [name, f"{round(confidence * 100)} %"]
    if timestamp_seconds is not None:
        parts.append(mmss(timestamp_seconds))
    return " · ".join(parts)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size)  # Pillow ≥ 10.1: fuente FreeType incluida, con tildes
    except TypeError:  # pragma: no cover - Pillow viejo
        return ImageFont.load_default()


def _append_strip(img_bgr, text: str):
    """Agrega **debajo** del fotograma una franja negra con el texto en blanco.

    Va debajo y no encima porque las cámaras trampa imprimen al pie su propio sello (fecha, hora,
    temperatura, id de cámara), que es información valiosa y no se debe tapar. El texto se dibuja con
    Pillow para que se vean las tildes.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    h, w = img_bgr.shape[:2]
    size = max(12, round(min(h, w) / 22))
    font = _font(size)
    pad = max(4, size // 3)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _left, top, _right, bottom = probe.textbbox((0, 0), text, font=font)
    strip_h = (bottom - top) + 2 * pad
    canvas = Image.new("RGB", (w, h + strip_h), (0, 0, 0))
    canvas.paste(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), (0, 0))
    ImageDraw.Draw(canvas).text((pad, h + pad - top), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_evidence(
    frame_bgr,
    bbox: BBox,
    *,
    label: Optional[str] = None,
    max_px: int = EVIDENCE_MAX_PX,
    quality: int = 80,
) -> Tuple[bytes, int, int]:
    """Redimensiona (sin agrandar), dibuja el recuadro y, si hay ``label``, la franja; devuelve ``(jpeg, ancho, alto)``.

    Con franja, el JPEG es más alto que el fotograma: ``bbox`` sigue siendo relativo al **fotograma**
    (la parte de arriba), como pide el contrato. La web no lleva franja. ``cv2.imencode`` no escribe EXIF.
    """
    import cv2

    max_px = min(max_px, EVIDENCE_MAX_PX)
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    img = (
        frame_bgr
        if scale >= 1
        else cv2.resize(frame_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    )
    img = img.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1 * (w - 1))), int(round(y1 * (h - 1))))
    p2 = (int(round(x2 * (w - 1))), int(round(y2 * (h - 1))))
    thickness = max(2, round(max(h, w) / 320))
    cv2.rectangle(img, p1, p2, OUTLINE_BGR, thickness + 2)
    cv2.rectangle(img, p1, p2, BOX_COLOR_BGR, thickness)
    if label:
        img = _append_strip(img, label)
        h, w = img.shape[:2]
        if max(h, w) > max_px:  # un video vertical con franja podría pasarse del máximo
            scale = max_px / max(h, w)
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover - imencode solo falla con imágenes vacías
        raise ValueError("No se pudo codificar el fotograma de evidencia")
    return buf.tobytes(), w, h
