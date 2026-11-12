"""Guardar un archivo clasificado sin poder perderlo (HU-DS-30, F3-D1, F3-D2).

La sección de guardado de un archivo, en orden (CA-1), con los puntos de corte de la spec (§6):

    en_curso.json  →C1  fotograma  →C2  NOMBRE.ext.part (fsync)  →C3  verificar tamaño + sha256
    →  renombrar  →C4  borrar el original (solo "mover")  →C5  línea en el registro  →C6  borrar en_curso.json

- **Mismo volumen y "mover":** en lugar de copiar, ``os.replace`` (atómico); C3 no existe.
- **El diario** ``<destino>/_wwm/en_curso.json`` se escribe con ``fsync`` *antes* de tocar el disco y lleva
  la línea ya armada. ``reconcile()`` lo resuelve al iniciar el análisis siguiente: **completa** si el destino
  está entero (``sha256`` igual al del diario) y si no **deshace**. Nunca borra un original sin haber leído
  de nuevo el destino, y nunca borra el destino si el original ya no existe.
- La línea la escribe quien llama (``FolderJob`` con su ``record``), entre ``Saved`` y ``Saved.commit()``.

``fault(step)`` es un gancho para los tests (en producción no hace nada): se llama en cada punto de corte y
puede lanzar ``SimulatedCrash`` (que, como un corte de luz, no lo atrapa nadie) o dañar el ``.part``.
"""

import errno
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from wwm_pipeline.enums import DesktopActionType
from wwm_pipeline.metadata import file_sha256
from wwm_pipeline.package import PackageLine

logger = logging.getLogger("desktop.job")

WWM_DIR = "_wwm"
JOURNAL_NAME = "en_curso.json"
PART_SUFFIX = ".part"
COPY_BUFFER = 1024 * 1024
DISK_MARGIN_BYTES = 50 * 1024 * 1024  # F3-D7
RETRIES = 3
RETRY_WAIT_S = 0.2
ERROR_NOT_SAME_DEVICE = 17  # WinError: "no se puede mover el archivo a otra unidad"

Fault = Callable[[str], None]
Record = Callable[[PackageLine, Dict[str, Any]], None]
HasLine = Callable[[PackageLine], bool]


class SimulatedCrash(BaseException):
    """Un corte inyectado por los tests. Es ``BaseException`` para que ningún ``except Exception`` lo detenga."""


class TransferError(Exception):
    """Una falla al guardar, ya deshecha (el original intacto, sin ``.part`` ni fotograma)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _no_fault(step: str) -> None:
    return None


def journal_path(dest_root: Path) -> Path:
    return Path(dest_root) / WWM_DIR / JOURNAL_NAME


def _retry(fn: Callable[[], Any]) -> Any:
    """Reintenta ante ``PermissionError``: un antivirus o el indexador bloquean un archivo recién escrito."""
    for attempt in range(RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT_S)


def write_durable(path: Path, data: bytes) -> None:
    """Escribe ``path`` entero o nada: ``.tmp`` + ``fsync`` + ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _retry(lambda: os.replace(tmp, path))
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def same_volume(source: Path, target_dir: Path) -> bool:
    """En Windows, ``st_dev`` es el número de serie del volumen."""
    try:
        return os.stat(source).st_dev == os.stat(target_dir).st_dev
    except OSError:
        return False


def copy_with_fsync(source: Path, part: Path) -> None:
    """Copia el contenido y las fechas (como ``copy2``) y fuerza los datos al disco antes de volver."""
    with open(source, "rb") as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, COPY_BUFFER)
        dst.flush()
        os.fsync(dst.fileno())
    shutil.copystat(str(source), str(part))


def is_disk_full(exc: OSError) -> bool:
    return exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in (39, 112)


# ---------------------------------------------------------------------------
# El diario
# ---------------------------------------------------------------------------
@dataclass
class Journal:
    action: str  # "copy" | "move"
    source: str
    source_size: int
    source_mtime_ns: int
    sha256: str
    target: str
    evidence: Optional[str]
    line: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=1).encode("utf-8")

    @classmethod
    def load(cls, path: Path) -> "Journal":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def write_journal(dest_root: Path, journal: Journal) -> None:
    path = journal_path(dest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_durable(path, journal.to_json())


def clear_journal(dest_root: Path) -> None:
    path = journal_path(dest_root)
    _retry(lambda: path.unlink(missing_ok=True))


# ---------------------------------------------------------------------------
# La sección de guardado
# ---------------------------------------------------------------------------
@dataclass
class Saved:
    """El archivo ya está en su lugar. Falta escribir ``line`` y llamar a ``commit()`` (C6)."""

    dest_root: Path
    line: PackageLine
    target: Path
    evidence: Optional[Path]
    warning_code: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    fault: Fault = _no_fault

    def commit(self) -> None:
        self.fault("C6")
        clear_journal(self.dest_root)


def _as_copy(line: PackageLine) -> PackageLine:
    data = line.model_dump(mode="json")
    data["desktop_action"]["type"] = DesktopActionType.COPY.value
    return PackageLine.model_validate(data)


def _delete_original(source: Path) -> bool:
    """``False`` si el original quedó bloqueado por otro programa (F3-D8)."""
    try:
        _retry(lambda: source.unlink())
        return True
    except FileNotFoundError:
        return True
    except PermissionError:
        logger.warning("No se pudo borrar el original %s (en uso); queda como copia", source.name)
        return False


def _undo(source: Path, target: Path, evidence: Optional[Path]) -> None:
    """Deja todo como antes de empezar. El destino solo se borra si el original sigue en el origen."""
    for path in (
        target.with_name(target.name + PART_SUFFIX),
        evidence,
        evidence and evidence.with_name(evidence.name + ".part"),
    ):
        if path is not None:
            Path(path).unlink(missing_ok=True)
    if target.exists() and source.exists():
        target.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def save_file(
    *,
    dest_root: Path,
    source: Path,
    target: Path,
    evidence: Optional[Path],
    evidence_jpeg: Optional[bytes],
    sha256: str,
    move: bool,
    line: PackageLine,
    extra: Optional[Dict[str, Any]] = None,
    volume: Optional[bool] = None,
    fault: Fault = _no_fault,
    write_evidence: Callable[[Path, bytes], None],
) -> Saved:
    """Fotograma + copia o movimiento verificado. Devuelve ``Saved``; si algo falla, deshace y lanza ``TransferError``.

    ``volume``: ``True``/``False`` fuerza "mismo volumen" o "otro" (tests); ``None`` lo averigua.
    """
    st = os.stat(source)
    extra = {
        **(extra or {}),
        "source_mtime_ns": st.st_mtime_ns,  # HU-DS-14: el bloque local de la línea
        "destination_file": _relative(target, dest_root),
        "evidence_file": _relative(evidence, dest_root) if evidence is not None and evidence_jpeg is not None else None,
    }
    rename_only = move and (same_volume(source, target.parent) if volume is None else volume)
    write_journal(
        dest_root,
        Journal(
            action="move" if move else "copy",
            source=str(source),
            source_size=st.st_size,
            source_mtime_ns=st.st_mtime_ns,
            sha256=sha256,
            target=str(target),
            evidence=str(evidence) if evidence is not None else None,
            line=line.model_dump(mode="json"),
            extra=extra,
        ),
    )
    fault("C1")
    part = target.with_name(target.name + PART_SUFFIX)
    try:
        if evidence is not None and evidence_jpeg is not None:
            try:
                write_evidence(evidence, evidence_jpeg)
            except OSError as exc:
                code = "DISK_SPACE_LOW" if is_disk_full(exc) else "EVIDENCE_WRITE_FAILED"
                raise TransferError(code, str(exc)) from exc
        fault("C2")

        moved = False
        if rename_only:
            try:
                _retry(lambda: os.replace(source, target))
                moved = True
            except OSError as exc:
                if getattr(exc, "winerror", None) != ERROR_NOT_SAME_DEVICE:
                    raise
                logger.info("%s: otro volumen; se copia y verifica", source.name)
        if not moved:
            copy_with_fsync(source, part)
            fault("C3")
            got_sha, got_size = file_sha256(part)
            if got_size != st.st_size or got_sha != sha256:
                raise TransferError("FILE_MOVE_VERIFY_FAILED", f"la copia de {source.name} no coincide con el original")
            _retry(lambda: os.replace(part, target))
        elif target.stat().st_size != st.st_size:  # pragma: no cover - un renombre no cambia el tamaño
            logger.error("%s: el tamaño cambió al renombrar", source.name)
        fault("C4")
    except TransferError:
        _undo(source, target, evidence)
        clear_journal(dest_root)
        raise
    except OSError as exc:
        _undo(source, target, evidence)
        clear_journal(dest_root)
        code = "DISK_SPACE_LOW" if is_disk_full(exc) else "DESKTOP_ACTION_FAILED"
        raise TransferError(code, str(exc)) from exc

    warning = None
    if move and not rename_only and not _delete_original(source):
        line, warning = _as_copy(line), "SOURCE_NOT_DELETED"
    fault("C5")
    return Saved(dest_root, line, target, evidence, warning, extra, fault)


# ---------------------------------------------------------------------------
# Al iniciar: resolver lo que quedó a medias
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reconciled:
    outcome: str  # "none" | "completed" | "undone" | "already_recorded"
    file_name: Optional[str] = None
    parts_removed: int = 0


def remove_part_files(dest_root: Path) -> int:
    """Borra los ``*.part`` de ``<destino>/<carpeta>/`` (CA-5): son copias o fotogramas a medias."""
    removed = 0
    root = Path(dest_root)
    if not root.is_dir():
        return 0
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name == WWM_DIR:
            continue
        for part in folder.glob("*" + PART_SUFFIX):
            try:
                part.unlink()
                removed += 1
                logger.warning("Se borró la copia a medias %s", part)
            except OSError:
                logger.exception("No se pudo borrar %s", part)
    return removed


def reconcile(dest_root: Path, record: Optional[Record] = None, has_line: Optional[HasLine] = None) -> Reconciled:
    """Completa o deshace la operación de ``en_curso.json`` (tabla de la spec §6) y limpia los ``.part``."""
    path = journal_path(dest_root)
    if not path.exists():
        return Reconciled("none", parts_removed=remove_part_files(dest_root))
    try:
        journal = Journal.load(path)
    except (OSError, ValueError, TypeError):
        # El diario se escribe con .tmp + replace: si está ilegible es que nunca se completó (antes de C1).
        logger.exception("en_curso.json ilegible; se descarta")
        clear_journal(dest_root)
        return Reconciled("none", parts_removed=remove_part_files(dest_root))

    source, target = Path(journal.source), Path(journal.target)
    evidence = Path(journal.evidence) if journal.evidence else None
    line = PackageLine.model_validate(journal.line)

    complete = False
    if target.exists():
        if not source.exists():
            complete = True  # el único ejemplar es el del destino: nunca se borra
            got_sha, _ = file_sha256(target)
            if got_sha != journal.sha256:
                logger.warning("%s: el destino no coincide con el diario, pero es el único ejemplar", target.name)
        else:
            complete = file_sha256(target)[0] == journal.sha256

    if not complete:
        _undo(source, target, evidence)
        clear_journal(dest_root)
        logger.warning("Se deshizo el guardado a medias de %s; se analizará de nuevo", source.name)
        return Reconciled("undone", source.name, remove_part_files(dest_root))

    if journal.action == "move" and source.exists():
        st = source.stat()
        same = st.st_size == journal.source_size and st.st_mtime_ns == journal.source_mtime_ns
        if not (same and _delete_original(source)):
            line = _as_copy(line)
    if has_line is not None and has_line(line):
        clear_journal(dest_root)
        return Reconciled("already_recorded", source.name, remove_part_files(dest_root))
    if record is not None:
        record(line, journal.extra)
    clear_journal(dest_root)
    logger.warning("Se completó el guardado de %s que quedó a medias", source.name)
    return Reconciled("completed", source.name, remove_part_files(dest_root))
