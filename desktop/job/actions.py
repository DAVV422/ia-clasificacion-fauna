"""Qué se hace en disco con cada archivo clasificado (E-06).

- Animal con especie, género o familia → ``<destino>/<carpeta>/NOMBRE.ext`` + ``NOMBRE_fotograma.jpg`` al lado
  (carpeta: HU-DS-03; fotograma: HU-DS-10).
- Todo lo demás → **queda en el origen** (HU-DS-02), sin fotograma en disco (F2-D2, F2-D9).

Orden (F2-D3): primero el fotograma y después el archivo. Si falla el fotograma, el archivo no se toca
(``EVIDENCE_WRITE_FAILED``); si falla la copia o el movimiento, se borra el fotograma recién escrito.
``apply_action(result, path)`` devuelve el ``DesktopAction`` del registro (PKG §3). El mover verificado
(con ``.part`` y ``sha256``) llega con HU-DS-30.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable

from wwm_pipeline.enums import DesktopActionType
from wwm_pipeline.package import DesktopAction

from .file_ops import safe_folder_name, unique_pair, write_atomic
from .placement import should_relocate
from .record import LEFT_IN_PLACE

logger = logging.getLogger("desktop.job")
ApplyAction = Callable[[object, Path], DesktopAction]


class ActionError(Exception):
    """Una falla al reubicar un archivo, con su código de E-11. El original siempre queda en el origen."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


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
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ActionError("DESTINATION_NOT_WRITABLE", str(exc)) from exc
        target, evidence_path = unique_pair(target_dir, path.name)

        wrote_evidence = False
        if result.evidence is not None:
            try:
                write_atomic(evidence_path, result.evidence.jpeg)
                wrote_evidence = True
            except OSError as exc:
                raise ActionError("EVIDENCE_WRITE_FAILED", str(exc)) from exc
        else:  # no debería pasar con los presets nuevos: todo animal trae su fotograma
            logger.warning("%s no trae fotograma de evidencia", path.name)

        try:
            if kind is DesktopActionType.COPY:
                shutil.copy2(str(path), str(target))
            else:
                shutil.move(str(path), str(target))
        except OSError as exc:
            if wrote_evidence:
                evidence_path.unlink(missing_ok=True)  # nunca un fotograma huérfano (CA-7)
            raise ActionError("DESKTOP_ACTION_FAILED", str(exc)) from exc
        return DesktopAction(type=kind, destination_folder=folder)

    return apply_action
