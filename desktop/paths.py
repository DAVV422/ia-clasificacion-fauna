"""Rutas de datos de la app de escritorio (HU-DS-00) y variables de entorno de las librerías.

Todo lo que la app escribe va a ``%LOCALAPPDATA%\\WWM`` (o a ``WWM_DATA_DIR`` si está definida),
nunca junto al ejecutable (HU-DS-04)::

    %LOCALAPPDATA%\\WWM\\models     pesos de los modelos
    %LOCALAPPDATA%\\WWM\\cache      TORCH_HOME, caché de kagglehub, Hugging Face y ultralytics
    %LOCALAPPDATA%\\WWM\\logs       wwm.log (JSON, rotativo)
    %LOCALAPPDATA%\\WWM\\tmp        temporales de cada archivo

Este módulo **no** importa torch ni crea nada al importarse.
"""

import os
import sys
from pathlib import Path

APP_DIR_NAME = "WWM"


def data_dir() -> Path:
    override = os.getenv("WWM_DATA_DIR")
    if override:
        return Path(override)
    base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_DIR_NAME


def models_dir() -> Path:
    return data_dir() / "models"


def cache_dir() -> Path:
    return data_dir() / "cache"


def logs_dir() -> Path:
    return data_dir() / "logs"


def tmp_dir() -> Path:
    return data_dir() / "tmp"


def settings_file() -> Path:
    return data_dir() / "settings.json"


def resource_dir() -> Path:
    """Recursos empaquetados (íconos, logo): ``_MEIPASS`` en el ejecutable, la raíz del repo en desarrollo."""
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return Path(__file__).resolve().parent.parent


def ensure_dirs() -> None:
    for folder in (models_dir(), cache_dir(), logs_dir(), tmp_dir()):
        folder.mkdir(parents=True, exist_ok=True)


def configure_env() -> None:
    """Fija dónde escriben las librerías **antes** de importarlas (torch, kagglehub, ultralytics).

    Se llama al inicio de ``desktop.main``, antes de cualquier import pesado. Usa ``setdefault`` para
    respetar valores ya definidos (por ejemplo, en desarrollo o en los tests).
    """
    cache = cache_dir()
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache / "kagglehub"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    # torch.hub/ultralytics instalan paquetes por su cuenta al cargar YOLOv5 (hallazgo de la fase 0).
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
