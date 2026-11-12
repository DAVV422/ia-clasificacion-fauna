"""Qué se hace en disco con cada archivo clasificado (E-06).

- Animal con especie, género o familia → ``<destino>/<carpeta>/NOMBRE.ext`` + ``NOMBRE_fotograma.jpg`` al lado
  (carpeta: HU-DS-03; fotograma: HU-DS-10).
- Todo lo demás → **queda en el origen** (HU-DS-02), sin fotograma en disco (F2-D2, F2-D9).

Lo que se reubica pasa por la sección de guardado de ``transfer.save_file`` (HU-DS-30): el diario, el fotograma
primero (F2-D3), la copia verificada con ``sha256`` (o el renombre si es "mover" en el mismo disco) y el borrado
del original. ``apply_action(result, path)`` devuelve:

- ``LEFT_IN_PLACE`` (un ``DesktopAction``) si el archivo queda en el origen; o
- un ``transfer.Saved``: el archivo ya está en su lugar y quien llama escribe ``saved.line`` y llama a ``commit()``.

Si algo falla, todo queda como antes y se lanza ``ActionError`` con el código de E-11.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Union

from wwm_pipeline.enums import DesktopActionType
from wwm_pipeline.package import DesktopAction

from .file_ops import safe_folder_name, unique_pair, write_atomic
from .placement import should_relocate
from .record import LEFT_IN_PLACE, build_line
from .transfer import DISK_MARGIN_BYTES, Fault, Saved, TransferError, _no_fault, same_volume, save_file

logger = logging.getLogger("desktop.job")
ApplyAction = Callable[[object, Path], Union[DesktopAction, Saved]]


class ActionError(Exception):
    """Una falla al reubicar un archivo, con su código de E-11. El original siempre queda en el origen."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def make_species_action(
    dest_root: Path,
    action: str,
    *,
    fault: Fault = _no_fault,
    volume: Optional[bool] = None,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> ApplyAction:
    """``apply_action`` para ``FolderJob``: copia (``"copy"``) o mueve (``"move"``) solo lo que se reubica.

    ``fault``, ``volume`` y ``disk_usage`` existen para los tests (cortes inyectados, forzar "otro volumen",
    simular el disco lleno).
    """
    if action not in ("copy", "move"):
        raise ValueError(f"acción desconocida: {action!r}")
    dest_root = Path(dest_root)
    kind = DesktopActionType.COPY if action == "copy" else DesktopActionType.MOVE

    def apply_action(result, path: Path) -> Union[DesktopAction, Saved]:
        if not should_relocate(result):
            return LEFT_IN_PLACE
        folder = safe_folder_name(result.taxonomy)
        target_dir = dest_root / folder
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ActionError("DESTINATION_NOT_WRITABLE", str(exc)) from exc

        jpeg = result.evidence.jpeg if result.evidence is not None else None
        if jpeg is None:  # no debería pasar con los presets nuevos: todo animal trae su fotograma
            logger.warning("%s no trae fotograma de evidencia", path.name)
        rename_only = kind is DesktopActionType.MOVE and (same_volume(path, target_dir) if volume is None else volume)
        needed = len(jpeg or b"") + DISK_MARGIN_BYTES + (0 if rename_only else result.size_bytes or 0)
        if disk_usage(str(target_dir)).free < needed:  # F3-D7 · CB-92
            raise ActionError("DISK_SPACE_LOW", f"no hay espacio para {path.name}")

        target, evidence_path = unique_pair(target_dir, path.name)
        line = build_line(result, DesktopAction(type=kind, destination_folder=folder))
        try:
            return save_file(
                dest_root=dest_root,
                source=Path(path),
                target=target,
                evidence=evidence_path if jpeg is not None else None,
                evidence_jpeg=jpeg,
                sha256=result.sha256,
                move=kind is DesktopActionType.MOVE,
                line=line,
                volume=volume,
                fault=fault,
                write_evidence=write_atomic,
            )
        except TransferError as exc:
            raise ActionError(exc.code, exc.message) from exc

    return apply_action
