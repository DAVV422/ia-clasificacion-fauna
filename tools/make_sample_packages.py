"""Genera los paquetes ``.wwm.zip`` de prueba (válidos e inválidos).

Salida en ``tests/fixtures/package/``:
    sample_valid.wwm.zip              12 líneas (7 animal [6 especie + 1 género], 2 blank, 1 person, 1 vehicle, 1 error), con camera
    sample_valid_no_camera.wwm.zip    igual, sin bloque camera
    invalid_zip_slip.wwm.zip          contiene "../evil.jpg"                → IMPORT_INVALID_PACKAGE
    invalid_checksum.wwm.zip          checksum del jsonl alterado           → IMPORT_CHECKSUM_MISMATCH
    invalid_version_2.wwm.zip         format_version "2.0"                  → IMPORT_UNSUPPORTED_VERSION
    invalid_corrupt_jpg.wwm.zip       un fotograma no es JPEG               → issue EVIDENCE_INVALID
    invalid_unknown_taxon.wwm.zip     un label_id que no está en el catálogo → issue CATALOG_TAXON_NOT_FOUND
    invalid_no_manifest.wwm.zip       sin manifest.json                     → IMPORT_INVALID_PACKAGE

Requiere haber corrido antes ``python -m tools.make_fixtures`` (usa sus fotogramas).
"""

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from tools import _fixture_taxa as taxa
from tools.make_fixtures import FRAMES_DIR, fake_sha
from wwm_pipeline.catalog_schema import parse_label
from wwm_pipeline.package import JSONL_NAME, MANIFEST_NAME, Manifest, PackageLine, count_lines

OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/package"
MODEL = {
    "detector": "MDv5a",
    "classifier": "speciesnet-4.0.2a",
    "sampling": "15fps@0-2s+6fps",
    "pipeline_version": "1.0.0",
}
CAMERA = {
    "code": "CAM01",
    "name": "Quebrada norte",
    "latitude": -17.78345,
    "longitude": -63.18211,
    "deployed_at": "2026-03-01T08:30:00",
    "retrieved_at": "2026-03-19T17:00:00",
    "still_installed": False,
    "observations": "Sobre sendero de tapires",
}


def _tax(label: str) -> dict:
    t = parse_label(label)
    d = t.model_dump(by_alias=True)  # type: ignore[union-attr]
    d["label_raw"] = label
    return d


def _folder(label: str) -> str:
    t = parse_label(label)
    if t.species:  # type: ignore[union-attr]
        return f"{t.genus}_{t.species}_{(t.common_name or '').replace(' ', '_')}".lower()  # type: ignore[union-attr]
    return f"{t.genus}_sp"  # type: ignore[union-attr]


def _line(
    name: str,
    category: Optional[str],
    label: Optional[str] = None,
    conf: float = 0.9,
    frame: Optional[str] = None,
    ts: float = 3.5,
    media: str = "video",
    error: Optional[str] = None,
    action: str = "copy",
) -> dict:
    sha = fake_sha(name)
    line = {
        "file_name": name,
        "file_sha256": sha,
        "file_size_bytes": 15_000_000 if media == "video" else 2_400_000,
        "media_type": media,
        "duration_seconds": 14.0 if media == "video" else None,
        "captured_at": f"2026-03-{(sum(map(ord, name)) % 18) + 1:02d}T04:22:10",
        "captured_at_source": "filename",
        "status": "error" if error else "done",
        "error_code": error,
        "category": None if error else category,
        "taxonomy": _tax(label) if label and not error else None,
        "confidence": None if error else conf,
        "alternatives": [],
        "frames_analyzed": None if error else 80,
        "evidence": None,
        "desktop_action": {"type": "left_in_place", "destination_folder": None},
    }
    if frame and not error:
        line["evidence"] = {
            "file": f"evidence/{sha}.jpg",
            "frame_index": int(ts * 30) if media == "video" else 0,
            "timestamp_seconds": ts if media == "video" else 0.0,
            "bbox": [0.3, 0.2, 0.7, 0.8],
            "width": 960,
            "height": 540,
        }
    if category == "animal" and not error:
        line["desktop_action"] = {"type": action, "destination_folder": _folder(label)}  # type: ignore[arg-type]
    line["_frame"] = frame
    return line


def base_lines() -> List[dict]:
    return [
        _line("CAM01_20260302_042210.mp4", "animal", taxa.JAGUAR, 0.94, "jaguar", 7.07),
        _line("CAM01_20260303_221501.mp4", "animal", taxa.JAGUAR, 0.88, "jaguar", 2.1),
        _line("CAM01_20260305_031244.mp4", "animal", taxa.PUMA, 0.91, "jaguar", 5.4),
        _line("CAM01_20260307_235959.mp4", "animal", taxa.OCELOT, 0.83, "jaguar", 1.2),
        _line("CAM01_20260309_060102.mp4", "animal", taxa.TAPIR, 0.96, "tapir", 9.9),
        _line("CAM01_20260311_191020.mp4", "animal", taxa.AGOUTI, 0.79, "dasyprocta", 4.0),
        _line("CAM01_20260312_181502.mp4", "animal", taxa.DASYPROCTA_GENUS, 0.71, "dasyprocta", 4.27),
        _line("CAM01_20260313_120000.mp4", "blank", conf=0.97),
        _line("IMG_0042.JPG", "blank", conf=0.95, media="image"),
        _line("CAM01_20260314_090000.mp4", "person", conf=0.88),  # sin fotograma (F2-D2)
        _line("IMG_0107.JPG", "vehicle", conf=0.92, media="image"),  # sin fotograma (F2-D9)
        _line("CAM01_20260315_101010.mp4", None, error="IA_DECODE_FAILED"),
    ]


def build_zip(
    lines: List[dict],
    camera: Optional[dict] = CAMERA,
    version: str = "1.0",
    tamper_checksum: bool = False,
    extra: Optional[Dict[str, bytes]] = None,
    corrupt_frame_of: Optional[int] = None,
    include_manifest: bool = True,
) -> bytes:
    clean = [{k: v for k, v in line.items() if k != "_frame"} for line in lines]
    for line in clean:
        PackageLine.model_validate(line)  # las líneas base siempre son válidas
    jsonl = ("\n".join(json.dumps(line, ensure_ascii=False, separators=(",", ":")) for line in clean) + "\n").encode()
    checksum = "sha256:" + hashlib.sha256(jsonl).hexdigest()
    if tamper_checksum:
        checksum = "sha256:" + "0" * 64
    counts = count_lines([PackageLine.model_validate(line) for line in clean]).model_dump()
    manifest = {
        "format": "wwm-desktop-export",
        "format_version": version,
        "app_version": "1.0.0",
        "created_at": "2026-03-20T15:04:05Z",
        "device_id": "3f9a1c0e",
        "camera_code": "CAM01",
        "camera": camera,
        "source_folder_name": "SD_CAM01_marzo",
        "run": {
            "job_id": "j-20260320-145000",
            "started_at": "2026-03-20T14:50:00Z",
            "finished_at": "2026-03-20T15:03:40Z",
            "action": "copy",
        },
        "model": MODEL,
        "counts": counts,
        "checksums": {JSONL_NAME: checksum},
    }
    if version.startswith("1.") and not tamper_checksum:
        Manifest.model_validate(manifest)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if include_manifest:
            zf.writestr(
                MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False), compress_type=zipfile.ZIP_DEFLATED
            )
        zf.writestr(JSONL_NAME, jsonl, compress_type=zipfile.ZIP_DEFLATED)
        for i, (line, original) in enumerate(zip(clean, lines)):
            if line.get("evidence"):
                data = (FRAMES_DIR / f"{original['_frame']}.jpg").read_bytes()
                if corrupt_frame_of == i:
                    data = b"esto no es un jpeg"
                zf.writestr(line["evidence"]["file"], data, compress_type=zipfile.ZIP_STORED)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def build_all() -> Dict[str, bytes]:
    lines = base_lines()
    unknown = copy.deepcopy(lines)
    unknown[3]["taxonomy"]["label_id"] = "00000000-0000-0000-0000-000000000000"
    return {
        "sample_valid": build_zip(lines),
        "sample_valid_no_camera": build_zip(lines, camera=None),
        "invalid_zip_slip": build_zip(lines, extra={"../evil.jpg": b"x"}),
        "invalid_checksum": build_zip(lines, tamper_checksum=True),
        "invalid_version_2": build_zip(lines, version="2.0"),
        "invalid_corrupt_jpg": build_zip(lines, corrupt_frame_of=0),
        "invalid_unknown_taxon": build_zip(unknown),
        "invalid_no_manifest": build_zip(lines, include_manifest=False),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in build_all().items():
        (OUT / f"{name}.wwm.zip").write_bytes(data)
        print(f"ok  {name}.wwm.zip ({len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
