"""Validador de referencia de un paquete ``.wwm.zip`` (PAQUETE_ESCRITORIO §5).

Lo usan el exportador de escritorio (para autovalidarse) y los tests. El back
(BE-40) aplica las mismas reglas, más las que dependen de su base de datos
(duplicados, cámara, paquete ya importado).

Pillow solo se importa si hay que verificar los JPEG (``check_images=True``).
"""

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import AbstractSet, List, Optional, Union

from pydantic import ValidationError

from .package import (
    EVIDENCE_PATH_PATTERN,
    JSONL_NAME,
    MANIFEST_NAME,
    SUPPORTED_MAJOR,
    Manifest,
    PackageLine,
    count_lines,
)

# Códigos fatales (el paquete entero se rechaza)
IMPORT_INVALID_PACKAGE = "IMPORT_INVALID_PACKAGE"
IMPORT_UNSUPPORTED_VERSION = "IMPORT_UNSUPPORTED_VERSION"
IMPORT_CHECKSUM_MISMATCH = "IMPORT_CHECKSUM_MISMATCH"
IMPORT_TOO_LARGE = "IMPORT_TOO_LARGE"
# Códigos por línea
LINE_INVALID = "LINE_INVALID"
EVIDENCE_INVALID = "EVIDENCE_INVALID"
EVIDENCE_MISSING = "EVIDENCE_MISSING"
CATALOG_TAXON_NOT_FOUND = "CATALOG_TAXON_NOT_FOUND"

MAX_ENTRIES = 20_000


@dataclass
class Issue:
    line: Optional[int]
    file_name: Optional[str]
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class PackageReport:
    fatal_code: Optional[str] = None
    fatal_message: Optional[str] = None
    manifest: Optional[Manifest] = None
    lines: List[PackageLine] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.fatal_code is None and not any(i.severity == "error" for i in self.issues)

    def _fatal(self, code: str, message: str) -> "PackageReport":
        self.fatal_code, self.fatal_message = code, message
        return self


def _allowed_name(name: str) -> bool:
    if name in (MANIFEST_NAME, JSONL_NAME):
        return True
    return bool(re.match(EVIDENCE_PATH_PATTERN, name))


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    is_symlink = (info.external_attr >> 16) & 0o170000 == 0o120000
    return not (path.is_absolute() or ".." in path.parts or "\\" in info.filename or is_symlink)


def validate_package(
    source: Union[str, Path, bytes],
    catalog_label_ids: Optional[AbstractSet[str]] = None,
    max_bytes: Optional[int] = None,
    check_images: bool = True,
) -> PackageReport:
    report = PackageReport()
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if max_bytes is not None and len(raw) > max_bytes:
        return report._fatal(IMPORT_TOO_LARGE, f"El paquete supera {max_bytes} bytes")
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return report._fatal(IMPORT_INVALID_PACKAGE, "No es un archivo ZIP válido")

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            return report._fatal(IMPORT_TOO_LARGE, f"El paquete tiene más de {MAX_ENTRIES} entradas")
        if max_bytes is not None and sum(i.file_size for i in infos) > 3 * max_bytes:
            return report._fatal(IMPORT_TOO_LARGE, "El contenido descomprimido es demasiado grande")
        for info in infos:
            if not _safe_member(info) or not _allowed_name(info.filename):
                return report._fatal(IMPORT_INVALID_PACKAGE, f"Ruta no permitida en el paquete: {info.filename!r}")
        names = {i.filename for i in infos}
        if MANIFEST_NAME not in names or JSONL_NAME not in names:
            return report._fatal(IMPORT_INVALID_PACKAGE, "Faltan manifest.json o classifications.jsonl")

        # --- manifest: primero la versión, después el schema completo ---
        try:
            manifest_raw = json.loads(zf.read(MANIFEST_NAME))
        except (ValueError, UnicodeDecodeError):
            return report._fatal(IMPORT_INVALID_PACKAGE, "manifest.json no es JSON válido")
        version = str(manifest_raw.get("format_version", ""))
        if not re.match(r"^\d+\.\d+$", version) or int(version.split(".")[0]) != SUPPORTED_MAJOR:
            return report._fatal(IMPORT_UNSUPPORTED_VERSION, f"Versión de formato no soportada: {version!r}")
        try:
            manifest = Manifest.model_validate(manifest_raw)
        except ValidationError as exc:
            return report._fatal(IMPORT_INVALID_PACKAGE, f"manifest.json inválido: {exc.errors()[0]['msg']}")
        report.manifest = manifest

        # --- checksum del jsonl ---
        jsonl_bytes = zf.read(JSONL_NAME)
        actual = "sha256:" + hashlib.sha256(jsonl_bytes).hexdigest()
        if manifest.checksums.get(JSONL_NAME) != actual:
            return report._fatal(IMPORT_CHECKSUM_MISMATCH, "El checksum de classifications.jsonl no coincide")

        # --- líneas ---
        for number, text in enumerate(jsonl_bytes.decode("utf-8").splitlines(), start=1):
            if not text.strip():
                continue
            try:
                line = PackageLine.model_validate_json(text)
            except ValidationError as exc:
                report.issues.append(Issue(number, None, "error", LINE_INVALID, exc.errors()[0]["msg"]))
                continue
            report.lines.append(line)
            if catalog_label_ids is not None and line.taxonomy is not None:
                if line.taxonomy.label_id not in catalog_label_ids:
                    report.issues.append(
                        Issue(
                            number,
                            line.file_name,
                            "error",
                            CATALOG_TAXON_NOT_FOUND,
                            f"El taxón {line.taxonomy.label_id} no está en el catálogo",
                        )
                    )
            if line.evidence is not None:
                if line.evidence.file not in names:
                    report.issues.append(
                        Issue(
                            number,
                            line.file_name,
                            "error",
                            EVIDENCE_MISSING,
                            f"Falta {line.evidence.file} en el paquete",
                        )
                    )
                elif check_images and not _is_valid_jpeg(zf.read(line.evidence.file)):
                    report.issues.append(
                        Issue(
                            number,
                            line.file_name,
                            "error",
                            EVIDENCE_INVALID,
                            "El fotograma no es una imagen JPEG válida",
                        )
                    )

        # --- counts del manifiesto contra el contenido ---
        all_lines_valid = not any(i.code == LINE_INVALID for i in report.issues)
        if all_lines_valid and count_lines(report.lines) != manifest.counts:
            return report._fatal(IMPORT_INVALID_PACKAGE, "Los conteos del manifiesto no coinciden con las líneas")
    return report


def _is_valid_jpeg(data: bytes) -> bool:
    from PIL import Image  # import diferido: el schema no depende de Pillow

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "JPEG":
                return False
            img.verify()
        return True
    except Exception:
        return False
