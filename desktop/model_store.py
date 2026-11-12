"""Los modelos de la app: dónde están, si están bien y cómo se descargan (HU-DS-13, E-02, F4-D5).

- ``model_manifest.json`` (empaquetado con la app) dice qué artefactos hacen falta: la URL del *release*
  ``models-v1``, el tamaño, el ``sha256`` y dónde va cada uno dentro de ``models/``. El de SpeciesNet es un zip
  que se descomprime en una carpeta.
- ``models.ok`` (en la carpeta de modelos) se escribe al terminar e inventaría cada archivo instalado con su
  tamaño y su ``sha256``. Con él, el inicio normal solo compara tamaños (< 1 s); si uno no coincide, se
  verifica su hash y se vuelve a descargar solo ese artefacto.
- **Dónde:** ``<instalación>\\models`` si tiene ``models.ok`` (variante completa, de solo lectura); si no,
  ``%LOCALAPPDATA%\\WWM\\models`` (estándar).
- **Descarga:** a ``.part`` con ``Range`` (reanuda), hasta 5 reintentos ante un corte, ``sha256`` contra el
  manifiesto (un reintento si no coincide) y renombre atómico. Un zip se descomprime en una carpeta temporal,
  rechazando rutas peligrosas, y se renombra al final.

Sin ``tkinter``: la pantalla (``desktop/ui/first_run.py``) consume los eventos de ``install``.
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from desktop import paths

logger = logging.getLogger("desktop.models")

MANIFEST_NAME = "model_manifest.json"
MODELS_OK = "models.ok"
CHUNK = 1024 * 1024
RETRY_WAITS_S = (3, 10, 30, 30, 30)  # E-02: hasta 5 reintentos ante un corte
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
DISK_MARGIN_BYTES = 200 * 1024 * 1024


class ModelStoreError(Exception):
    """Con un código de E-11: NETWORK_UNAVAILABLE, DOWNLOAD_INTEGRITY_FAILED, DISK_SPACE_LOW, MODEL_LOAD_FAILED."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class Cancelled(Exception):
    """La persona pulsó Cancelar: los ``.part`` quedan para reanudar la próxima vez (CA-5)."""


# ---------------------------------------------------------------------------
# El manifiesto y la carpeta
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact:
    id: str
    name: str
    version: str
    url: str
    size: int
    sha256: str
    dest: str  # relativo a models/: "md_v5a.0.0.pt" o "speciesnet-4.0.2a/" (si unpack)
    unpack: bool = False
    # Solo con unpack: los archivos del zip ya descomprimido, {ruta relativa a dest: {"size", "sha256"}}. Permite
    # reusar una carpeta que ya está bien sin descargarla de nuevo.
    files: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def download_name(self) -> str:
        return PurePosixPath(self.url).name


@dataclass(frozen=True)
class Manifest:
    manifest_version: int
    artifacts: List[Artifact]


def manifest_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / "desktop" / "models" / MANIFEST_NAME


def load_manifest(path: Optional[Path] = None) -> Manifest:
    data = json.loads((path or manifest_path()).read_text(encoding="utf-8"))
    return Manifest(data["manifest_version"], [Artifact(**a) for a in data["artifacts"]])


def install_models_dir() -> Optional[Path]:
    """La carpeta de la variante completa (junto al ejecutable), si existe."""
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent / "models"


def resolve_models_dir(install_dir: Optional[Path] = None, user_dir: Optional[Path] = None) -> Path:
    """F4-D5: la de la instalación si tiene ``models.ok``; si no, la de AppData."""
    install_dir = install_dir if install_dir is not None else install_models_dir()
    if install_dir is not None and (install_dir / MODELS_OK).is_file():
        return install_dir
    return user_dir if user_dir is not None else paths.models_dir()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(root: Path, artifact: Artifact) -> List[Path]:
    target = root / artifact.dest
    if artifact.unpack:
        return sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else []
    return [target] if target.is_file() else []


# ---------------------------------------------------------------------------
# Verificar (CA-6)
# ---------------------------------------------------------------------------
def read_models_ok(root: Path) -> Optional[dict]:
    try:
        return json.loads((root / MODELS_OK).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def quick_check(root: Path, manifest: Manifest) -> List[Artifact]:
    """Los artefactos que hay que revisar. Solo compara tamaños con ``models.ok``: no lee los archivos."""
    ok = read_models_ok(root)
    if ok is None or ok.get("manifest_version") != manifest.manifest_version:
        return list(manifest.artifacts)
    pending = []
    for artifact in manifest.artifacts:
        entry = ok.get("artifacts", {}).get(artifact.id)
        if not entry or entry.get("sha256") != artifact.sha256:
            pending.append(artifact)
            continue
        for rel, info in entry.get("files", {}).items():
            path = root / rel
            if not path.is_file() or path.stat().st_size != info["size"]:
                pending.append(artifact)
                break
    return pending


def full_check(root: Path, artifact: Artifact, ok: Optional[dict]) -> bool:
    """El artefacto ya instalado está intacto (``sha256`` de cada archivo contra ``models.ok``)."""
    entry = (ok or {}).get("artifacts", {}).get(artifact.id)
    if not entry or entry.get("sha256") != artifact.sha256:
        return False
    for rel, info in entry.get("files", {}).items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != info["size"] or sha256_file(path) != info["sha256"]:
            return False
    return True


def existing_matches(root: Path, artifact: Artifact) -> bool:
    """Sin ``models.ok``: un archivo suelto que ya está y coincide con el manifiesto (p. ej. migrado) se reusa."""
    target = root / artifact.dest
    if artifact.unpack:
        if not artifact.files or not target.is_dir():
            return False
        for rel, info in artifact.files.items():
            path = target / rel
            if not path.is_file() or path.stat().st_size != info["size"] or sha256_file(path) != info["sha256"]:
                return False
        return True
    return target.is_file() and target.stat().st_size == artifact.size and sha256_file(target) == artifact.sha256


# ---------------------------------------------------------------------------
# Descargar (CA-2, CA-3, CA-4, CA-5)
# ---------------------------------------------------------------------------
@dataclass
class Progress:
    artifact_id: str
    done: int
    total: int


Session = Any  # requests.Session (inyectable en los tests)


def _session() -> Session:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "WWM-Clasificador"
    return s


def download(
    artifact: Artifact,
    part: Path,
    *,
    session: Optional[Session] = None,
    on_progress: Callable[[Progress], None] = lambda p: None,
    should_stop: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Descarga a ``part`` reanudando lo que ya tenga. Devuelve ``part`` completo (sin verificar el hash)."""
    import requests

    session = session or _session()
    part.parent.mkdir(parents=True, exist_ok=True)
    started = False  # sin ningún byte recibido, un error de conexión es "no hay red" (sin reintentos)
    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= artifact.size:
            return part
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(
                artifact.url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
            ) as resp:
                if resp.status_code == 416:  # el .part ya tiene todo (o más): se empieza de nuevo
                    part.unlink(missing_ok=True)
                    continue
                resp.raise_for_status()
                mode = "ab" if have and resp.status_code == 206 else "wb"
                done = have if mode == "ab" else 0
                with open(part, mode) as fh:
                    for chunk in resp.iter_content(CHUNK):
                        if should_stop():
                            raise Cancelled()
                        if chunk:
                            fh.write(chunk)
                            done += len(chunk)
                            started = True
                            on_progress(Progress(artifact.id, done, artifact.size))
            if part.stat().st_size >= artifact.size:
                return part
            raise requests.ConnectionError("la descarga terminó antes de tiempo")
        except Cancelled:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            if not started and not have:
                raise ModelStoreError("NETWORK_UNAVAILABLE", str(exc)) from exc
            if attempt >= len(RETRY_WAITS_S):
                raise ModelStoreError("NETWORK_UNAVAILABLE", f"se cortó la descarga de {artifact.name}") from exc
            wait = RETRY_WAITS_S[attempt]
            attempt += 1
            logger.warning("Corte en %s (%s); reintento %d en %d s", artifact.name, exc, attempt, wait)
            for _ in range(int(wait * 10)):
                if should_stop():
                    raise Cancelled() from exc
                sleep(0.1)
        except requests.HTTPError as exc:
            raise ModelStoreError("NETWORK_UNAVAILABLE", f"{artifact.url}: {exc}") from exc


def _safe_unpack(zip_path: Path, target: Path) -> None:
    """Descomprime en ``<target>.tmp`` y renombra; rechaza ``..`` y rutas absolutas (CA-8)."""
    tmp = target.with_name(target.name.rstrip("/\\") + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or ":" in info.filename or info.filename.startswith("\\"):
                raise ModelStoreError("DOWNLOAD_INTEGRITY_FAILED", f"ruta no permitida en el zip: {info.filename}")
        tmp.mkdir(parents=True)
        zf.extractall(tmp)
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)


def _install_one(root: Path, artifact: Artifact, **kwargs) -> None:
    part = root / (artifact.download_name + ".part")
    for verify_attempt in range(2):  # CA-3: un reintento si el hash no coincide
        download(artifact, part, **kwargs)
        if sha256_file(part) == artifact.sha256:
            break
        logger.error("sha256 distinto en %s (intento %d)", artifact.name, verify_attempt + 1)
        part.unlink(missing_ok=True)
    else:
        raise ModelStoreError("DOWNLOAD_INTEGRITY_FAILED", f"{artifact.name}: la descarga está dañada")
    target = root / artifact.dest
    if artifact.unpack:
        _safe_unpack(part, target)
        part.unlink(missing_ok=True)
    else:
        os.replace(part, target)


# ---------------------------------------------------------------------------
# Instalar todo
# ---------------------------------------------------------------------------
@dataclass
class InstallResult:
    root: Path
    downloaded: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)


def write_models_ok(root: Path, manifest: Manifest) -> None:
    data: Dict[str, Any] = {"manifest_version": manifest.manifest_version, "artifacts": {}}
    for artifact in manifest.artifacts:
        files = {
            p.relative_to(root).as_posix(): {"size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in _artifact_files(root, artifact)
        }
        data["artifacts"][artifact.id] = {"version": artifact.version, "sha256": artifact.sha256, "files": files}
    data["written_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = root / (MODELS_OK + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, root / MODELS_OK)


def needs_install(root: Path, manifest: Manifest) -> List[Artifact]:
    """Lo que falta o está dañado, con la verificación rápida primero (CA-6) y el hash solo si hace falta."""
    pending = quick_check(root, manifest)
    if not pending:
        return []
    ok = read_models_ok(root)
    return [a for a in pending if not (full_check(root, a, ok) or existing_matches(root, a))]


def install(
    root: Path,
    manifest: Manifest,
    *,
    on_progress: Callable[[Progress], None] = lambda p: None,
    should_stop: Callable[[], bool] = lambda: False,
    session: Optional[Session] = None,
    sleep: Callable[[float], None] = time.sleep,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> InstallResult:
    root.mkdir(parents=True, exist_ok=True)
    pending = needs_install(root, manifest)
    result = InstallResult(root, reused=[a.id for a in manifest.artifacts if a not in pending])
    if pending:
        needed = sum(a.size * (2 if a.unpack else 1) for a in pending) + DISK_MARGIN_BYTES
        if disk_usage(str(root)).free < needed:
            raise ModelStoreError("DISK_SPACE_LOW", f"hacen falta {needed // (1024 * 1024)} MB libres")
        for artifact in pending:  # la pantalla sabe desde el principio qué se va a descargar
            on_progress(Progress(artifact.id, 0, artifact.size))
        for artifact in pending:
            _install_one(root, artifact, session=session, on_progress=on_progress, should_stop=should_stop, sleep=sleep)
            result.downloaded.append(artifact.id)
    write_models_ok(root, manifest)
    return result
