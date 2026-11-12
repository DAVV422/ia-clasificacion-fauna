"""Rutas de datos de la app de escritorio (HU-DS-00) y variables de entorno de las librerías.

Todo lo que la app escribe va a ``%LOCALAPPDATA%\\WWM`` (o a ``WWM_DATA_DIR`` si está definida),
nunca junto al ejecutable (HU-DS-04)::

    %LOCALAPPDATA%\\WWM\\models     pesos de los modelos
    %LOCALAPPDATA%\\WWM\\cache      TORCH_HOME, caché de kagglehub, Hugging Face y ultralytics
    %LOCALAPPDATA%\\WWM\\logs       wwm.log (JSON, rotativo)
    %LOCALAPPDATA%\\WWM\\tmp        temporales de cada archivo

Este módulo **no** importa torch ni crea nada al importarse.
"""

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Optional

APP_DIR_NAME = "WWM"
MEGADETECTOR_FILE = "md_v5a.0.0.pt"
SPECIESNET_DIR = "speciesnet-4.0.2a"  # HU-DS-11: SpeciesNet desde una carpeta local (info.json + 4 archivos)


def data_dir() -> Path:
    override = os.getenv("WWM_DATA_DIR")
    if override:
        return Path(override)
    base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_DIR_NAME


_models_dir_override: Optional[Path] = None


def set_models_dir(folder: Optional[Path]) -> None:
    """HU-DS-13 (F4-D5): la carpeta elegida al iniciar (la de la instalación en la variante completa)."""
    global _models_dir_override
    _models_dir_override = Path(folder) if folder is not None else None


def models_dir() -> Path:
    return _models_dir_override or data_dir() / "models"


def cache_dir() -> Path:
    return data_dir() / "cache"


def logs_dir() -> Path:
    return data_dir() / "logs"


def tmp_dir() -> Path:
    return data_dir() / "tmp"


def megadetector_path() -> Path:
    return models_dir() / MEGADETECTOR_FILE


def speciesnet_dir() -> Path:
    return models_dir() / SPECIESNET_DIR


def clear_tmp() -> None:
    """Vacía ``tmp_dir()`` (al terminar cada archivo y al cerrar la app). Nunca lanza."""
    folder = tmp_dir()
    if not folder.is_dir():
        return
    for child in folder.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except OSError:
            pass  # un archivo aún abierto se limpia en la próxima pasada


@contextlib.contextmanager
def file_tmp() -> Iterator[Path]:
    """Carpeta temporal para procesar un archivo; se borra al salir, aunque haya error."""
    tmp_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(tmp_dir()), prefix="job_") as folder:
        yield Path(folder)


def legacy_model_candidates() -> Iterable[Path]:
    """Dónde dejaba el prototipo los pesos: ``models/`` junto al .exe viejo (o la raíz del repo) y en el cwd."""
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    seen = set()
    for base in (exe_dir, Path.cwd()):
        candidate = (base / "models" / MEGADETECTOR_FILE).resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def migrate_legacy_model(candidates: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Si los pesos ya existen donde los dejaba el prototipo, los **copia** a ``models_dir()``.

    Evita volver a descargar ~270 MB. Copia (no mueve) porque la carpeta vieja puede ser de solo
    lectura o seguir en uso por la API en desarrollo. Copia a un ``.part`` y renombra: nunca deja
    un archivo a medias. Devuelve el origen usado, o None si no hizo nada.
    """
    target = megadetector_path()
    if target.exists():
        return None
    for source in candidates if candidates is not None else legacy_model_candidates():
        if source.is_file() and source.stat().st_size > 0:
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(".part")
            shutil.copyfile(source, part)
            os.replace(part, target)
            return source
    return None


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
    # HU-DS-11: ultralytics resuelve one.one.one.one y dns.google al importarse para saber si hay internet.
    os.environ.setdefault("YOLO_OFFLINE", "true")
