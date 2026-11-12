"""Qué se hace en disco con cada archivo clasificado (E-06).

- Animal con especie, género o familia → copiar o mover a ``<destino>/<carpeta>/`` (carpeta: HU-DS-03).
- Todo lo demás → **queda en el origen** (HU-DS-02).

``apply_action(result, path)`` devuelve el ``DesktopAction`` del registro (PKG §3). El fotograma junto al
archivo llega con HU-DS-10 y el mover verificado con HU-DS-30.
"""

import shutil
from pathlib import Path
from typing import Callable

from wwm_pipeline.enums import DesktopActionType
from wwm_pipeline.package import DesktopAction

from .file_ops import safe_folder_name
from .placement import should_relocate
from .record import LEFT_IN_PLACE

ApplyAction = Callable[[object, Path], DesktopAction]


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


def make_species_action(dest_root: Path, action: str) -> ApplyAction:
    """``apply_action`` para ``FolderJob``: copia (``"copy"``) o mueve (``"move"``) solo lo que se reubica."""
    if action not in ("copy", "move"):
        raise ValueError(f"acción desconocida: {action!r}")
    dest_root = Path(dest_root)
    kind = DesktopActionType.COPY if action == "copy" else DesktopActionType.MOVE

    def apply_action(result, path: Path) -> DesktopAction:
        if not should_relocate(result):
            return LEFT_IN_PLACE
        folder = safe_folder_name(result.taxonomy)
        target_dir = dest_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(target_dir / path.name)
        if kind is DesktopActionType.COPY:
            shutil.copy2(str(path), str(dest))
        else:
            shutil.move(str(path), str(dest))
        return DesktopAction(type=kind, destination_folder=folder)

    return apply_action
