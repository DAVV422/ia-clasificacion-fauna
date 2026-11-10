"""Configuración del prototipo v1 (copia de los valores de app/config.py, commit c6dac6e).

Sin efectos al importar. Las rutas se pueden cambiar por variables de entorno.
"""

import os
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# torch.hub carga YOLOv5 desde GitHub y, por defecto, instala paquetes por su cuenta (pip) si cree que
# faltan o están desactualizados. Para medir, eso se desactiva: no cambia la clasificación, solo evita
# que se modifique el entorno. (Hallazgo de la fase 0; se resuelve de fondo en HU-DS-11.)
os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")  # código de YOLOv5 (torch.hub)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # paquete ultralytics

MEGADETECTOR_PATH = Path(os.getenv("LEGACY_MEGADETECTOR_PATH", str(_REPO / "models" / "md_v5a.0.0.pt")))
CONFIDENCE_THRESHOLD = 0.30
# Los recortes que escribe el detector viejo van a un temporal, fuera del repo.
DETECTIONS_DIR = Path(os.getenv("LEGACY_CROPS_DIR", str(Path(tempfile.gettempdir()) / "wwm_legacy_crops")))
DETECTIONS_URL_PREFIX = "/detections"
SPECIESNET_MODEL_NAME = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
SPECIESNET_CONFIDENCE_THRESHOLD = 0.15
MEGADETECTOR_CLASSES = {"1": "animal", "2": "person", "3": "vehicle", "0": "animal"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
