"""``%LOCALAPPDATA%\WWM\settings.json``: la configuración de la persona (F5-D8, HU-DS-04).

Por ahora guarda ``requirements_warned_version`` (HU-DS-20: el aviso de requisitos, una vez por versión).
HU-DS-22 le agrega el idioma de los nombres comunes y las últimas carpetas; HU-DS-19 las lee para el
diagnóstico. Se escribe atómico; un archivo corrupto se ignora (con un aviso en el log) y se empieza vacío.
"""

import json
import logging
import os
from typing import Any, Dict

from desktop import paths

logger = logging.getLogger("desktop")


def load() -> Dict[str, Any]:
    path = paths.settings_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("settings.json ilegible; se usa la configuración por defecto")
        return {}


def save(data: Dict[str, Any]) -> None:
    path = paths.settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set_value(key: str, value: Any) -> None:
    data = load()
    data[key] = value
    save(data)
