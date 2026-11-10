"""Evidencia. Fase 0: solo el guardado opcional de recortes del prototipo (``save_crops``).

En la fase 1 (IA-15) se agrega el fotograma de evidencia completo con recuadro y ``mm:ss``.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("wwm_pipeline.evidence")


def new_crop_name() -> str:
    return f"{uuid.uuid4()}.jpg"


def save_crop(crop_bgr, crops_dir: Path, run_name: str, category: str, crop_filename: str,
              url_prefix: str = "/detections") -> Tuple[Optional[str], Optional[str]]:
    """Guarda el recorte como el prototipo: ``<crops_dir>/<run_name>/<category>/<uuid>.jpg``.

    Devuelve (ruta relativa, URL) o (None, None) si no se pudo escribir.
    """
    import cv2

    category_dir = Path(crops_dir) / run_name / category
    category_dir.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(category_dir / crop_filename), crop_bgr):
        return (f"detections/{run_name}/{category}/{crop_filename}",
                f"{url_prefix}/{run_name}/{category}/{crop_filename}")
    logger.error("No se pudo guardar el recorte %s", category_dir / crop_filename)
    return None, None
