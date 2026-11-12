"""El registro local del análisis y la reanudación (HU-DS-14, E-05, E-08, E-D6).

``<destino>/_wwm/clasificaciones.jsonl``: una ``PackageLine`` por archivo procesado (``done`` o ``error``), en
UTF-8, una por línea, escrita con ``flush`` + ``os.fsync`` antes de pasar al siguiente archivo. Cada línea lleva
al final un bloque ``local`` (F3-D3) que ``PackageLine`` ignora y la exportación (HU-DS-17) no lleva::

    "local": {"source_mtime_ns": …, "destination_file": "panthera_onca_jaguar/V01.mp4",
              "evidence_file": "panthera_onca_jaguar/V01_fotograma.jpg", "run_id": "j-20261116-0912",
              "processed_at": "2026-11-16T09:14:03"}

``<destino>/_wwm/analisis.json`` es el encabezado: de qué origen y con qué acción se empezó el registro.

**Ya procesado** (F3-D4): misma (nombre, tamaño, ``mtime_ns``) que una línea ``done`` → se salta sin leer el
archivo. Si solo coincide el tamaño, se calcula el ``sha256`` (archivo renombrado). Las líneas ``error`` se
reintentan. Si un archivo tiene varias líneas, cuenta la última.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from desktop import __version__
from wwm_pipeline.metadata import file_sha256
from wwm_pipeline.package import PackageLine

from .transfer import WWM_DIR, write_durable

logger = logging.getLogger("desktop.job")

LOG_NAME = "clasificaciones.jsonl"
HEADER_NAME = "analisis.json"
LOCAL_KEY = "local"
LOCAL_FIELDS = ("source_mtime_ns", "destination_file", "evidence_file")


def new_run_id(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("j-%Y%m%d-%H%M%S")


@dataclass(frozen=True)
class Entry:
    line: PackageLine
    local: Dict[str, Any]


@dataclass(frozen=True)
class LogInfo:
    """Lo que muestra el diálogo de E-08."""

    done: int  # archivos distintos con su última línea en done
    errors: int  # archivos distintos con su última línea en error
    last_at: Optional[datetime]  # processed_at de la última línea (o started_at del encabezado)
    source: Optional[str]  # la carpeta de origen con la que se empezó el registro
    action: Optional[str]


def _identity(line: PackageLine) -> Tuple[str, str]:
    return line.file_name.lower(), line.file_sha256


def _line_key(line: PackageLine) -> Tuple[str, str, str, str, Optional[str]]:
    action = line.desktop_action
    return (*_identity(line), line.status.value, action.type.value, action.destination_folder)


class JobLog:
    def __init__(self, dest_root, run_id: Optional[str] = None, clock=datetime.now):
        self.dest_root = Path(dest_root)
        self.dir = self.dest_root / WWM_DIR
        self.path = self.dir / LOG_NAME
        self.header_path = self.dir / HEADER_NAME
        self.run_id = run_id or new_run_id(clock())
        self.clock = clock
        self._fh: Optional[IO[str]] = None
        self._last: Dict[Tuple[str, str], Entry] = {}
        self._keys: Set[Tuple[str, str, str, str, Optional[str]]] = set()

    # -- lectura -------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def read(self) -> List[Entry]:
        """Las líneas válidas, en orden. Una línea corrupta (p. ej. cortada por un apagón) se saltea con un aviso."""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8", newline="\n") as fh:
            for number, raw in enumerate(fh, start=1):
                if not raw.strip():
                    continue
                try:
                    data = json.loads(raw)
                    local = data.pop(LOCAL_KEY, None) or {}
                    entries.append(Entry(PackageLine.model_validate(data), local))
                except (ValueError, ValidationError) as exc:
                    logger.warning("%s, línea %d ignorada (corrupta): %s", LOG_NAME, number, str(exc)[:200])
        return entries

    def header(self) -> Dict[str, Any]:
        try:
            return json.loads(self.header_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def info(self) -> LogInfo:
        last = self._latest(self.read())
        done = sum(1 for e in last.values() if e.line.status.value == "done")
        stamps = [str(e.local["processed_at"]) for e in last.values() if e.local.get("processed_at")]
        header = self.header()
        when = max(stamps) if stamps else header.get("started_at")
        return LogInfo(
            done=done,
            errors=len(last) - done,
            last_at=datetime.fromisoformat(when) if when else None,
            source=header.get("source"),
            action=header.get("action"),
        )

    @staticmethod
    def _latest(entries: List[Entry]) -> Dict[Tuple[str, str], Entry]:
        last: Dict[Tuple[str, str], Entry] = {}
        for entry in entries:
            last[_identity(entry.line)] = entry
        return last

    # -- escritura -----------------------------------------------------------
    def backup(self) -> Optional[Path]:
        """ "Empezar de nuevo" (CA-5): ``clasificaciones.<AAAAMMDD-HHMMSS>.bak.jsonl``; no toca ningún otro archivo."""
        if not self.path.exists():
            return None
        stamp = self.clock().strftime("%Y%m%d-%H%M%S")
        target = self.dir / f"clasificaciones.{stamp}.bak.jsonl"
        n = 2
        while target.exists():
            target = self.dir / f"clasificaciones.{stamp}-{n}.bak.jsonl"
            n += 1
        os.replace(self.path, target)
        if self.header_path.exists():
            os.replace(self.header_path, target.with_name(target.name.replace(".jsonl", ".analisis.json")))
        logger.info("Registro anterior respaldado en %s", target.name)
        return target

    def start(self, source, action: str) -> None:
        """Abre el registro para anexar (lo crea con su encabezado si no existe) y arma el índice."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.header_path.exists() or not self.exists():
            header = {
                "source": str(source),
                "action": action,
                "run_id": self.run_id,
                "app_version": __version__,
                "started_at": self.clock().isoformat(timespec="seconds"),
            }
            write_durable(self.header_path, json.dumps(header, ensure_ascii=False, indent=1).encode("utf-8"))
        self._repair_tail()
        entries = self.read()
        self._last = self._latest(entries)
        self._keys = {_line_key(e.line) for e in entries}
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115 - abierto todo el análisis; close()

    def _repair_tail(self) -> None:
        """CA-6: si la última línea quedó a medias, se corta el archivo en el último ``\\n`` (el resto va al log)."""
        if not self.path.exists():
            return
        with open(self.path, "rb+") as fh:
            size = fh.seek(0, os.SEEK_END)
            if size == 0:
                return
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return
            block = min(size, 1024 * 1024)
            fh.seek(size - block)
            tail = fh.read(block)
            cut = tail.rfind(b"\n")
            keep = size - block + cut + 1 if cut >= 0 else 0
            logger.warning(
                "%s terminaba con una línea a medias (%d bytes); se descarta: %r",
                LOG_NAME,
                size - keep,
                tail[cut + 1 :][:200],
            )
            fh.truncate(keep)
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, line: PackageLine, extra: Optional[Dict[str, Any]] = None) -> None:
        """Escribe la línea con su bloque ``local`` y la fuerza al disco (CA-1, CA-2)."""
        if self._fh is None:
            raise RuntimeError("JobLog.start() no se llamó")
        extra = extra or {}
        local = {key: extra.get(key) for key in LOCAL_FIELDS}
        local["run_id"] = self.run_id
        local["processed_at"] = self.clock().isoformat(timespec="seconds")
        data = line.model_dump(mode="json")
        data[LOCAL_KEY] = local
        self._fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._last[_identity(line)] = Entry(line, local)
        self._keys.add(_line_key(line))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JobLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reanudación ---------------------------------------------------------
    def has_line(self, line: PackageLine) -> bool:
        """Para ``reconcile()`` (C6): la línea de la operación a medias ya está escrita."""
        return _line_key(line) in self._keys

    def is_processed(self, path: Path) -> bool:
        """``skip`` de ``FolderJob`` (CA-3, CA-4). Usa el índice vivo: incluye lo que ``reconcile()`` acaba de anotar."""
        done = [e for e in self._last.values() if e.line.status.value == "done"]
        if not done:
            return False
        st = os.stat(path)
        name = path.name.lower()
        same_size = [e for e in done if e.line.file_size_bytes == st.st_size]
        if not same_size:
            return False
        for e in same_size:
            if e.line.file_name.lower() == name and e.local.get("source_mtime_ns") == st.st_mtime_ns:
                return True  # sin leer el archivo
        # Mismo tamaño, pero otro nombre o la fecha cambió: se confirma por contenido. Con otro nombre, solo cuenta
        # como "renombrado" si el archivo registrado quedó en el origen (copy o left_in_place) y su nombre ya no
        # está: si sigue ahí, o si se movió (con "mover" no se puede renombrar en el origen), este es un duplicado
        # (dos archivos iguales con nombres distintos) y se analiza como uno más.
        candidates = [
            e
            for e in same_size
            if e.line.file_name.lower() == name
            or (e.line.desktop_action.type.value != "move" and not (path.parent / e.line.file_name).exists())
        ]
        if not candidates:
            return False
        sha, _ = file_sha256(path)
        return any(e.line.file_sha256 == sha for e in candidates)

    def remaining_estimate(self, files: List[Path]) -> int:
        """Para el texto del diálogo: cuántos faltan, sin leer ningún archivo (los renombrados cuentan como faltantes)."""
        keys = {
            (e.line.file_name.lower(), e.line.file_size_bytes, e.local.get("source_mtime_ns"))
            for e in self._latest(self.read()).values()
            if e.line.status.value == "done"
        }
        count = 0
        for p in files:
            try:
                st = os.stat(p)
            except OSError:
                count += 1
                continue
            if (p.name.lower(), st.st_size, st.st_mtime_ns) not in keys:
                count += 1
        return count
