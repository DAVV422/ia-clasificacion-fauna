"""Qué se hace con cada archivo clasificado.

Copiar o mover a ``<destino>/<carpeta de especie>/`` los archivos con un **animal** que llegó a especie,
género o familia (nombre de carpeta: HU-DS-03). Los demás no se tocan. La decisión completa de qué queda
en el origen y la línea del registro llegan con HU-DS-02; el fotograma junto al archivo, con HU-DS-10.
"""

import shutil
from pathlib import Path
from typing import Callable, Optional

from wwm_pipeline.enums import Rank

from .file_ops import RELOCATABLE_RANKS, safe_folder_name


def unique_dest(dest: Path) -> Path:
    """Si el destino ya existe, añade un sufijo numérico (HU-DS-10 lo cambia por " (2)")."""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while True:
        candidate = dest.parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def make_species_action(dest_root: Path, action: str) -> Callable[[object, Path], Optional[str]]:
    """``apply_action(result, path)`` para ``FolderJob``: copia (``"copy"``) o mueve (``"move"``) los archivos con animal."""
    if action not in ("copy", "move"):
        raise ValueError(f"acción desconocida: {action!r}")
    dest_root = Path(dest_root)

    def apply_action(result, path: Path) -> Optional[str]:
        taxonomy = result.taxonomy
        if result.decision.category != "animal" or taxonomy is None or Rank(taxonomy.rank) not in RELOCATABLE_RANKS:
            return None
        folder = safe_folder_name(taxonomy)
        target_dir = dest_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(target_dir / path.name)
        if action == "copy":
            shutil.copy2(str(path), str(dest))
            return f"copiado a {folder}"
        shutil.move(str(path), str(dest))
        return f"movido a {folder}"

    return apply_action
