"""Genera ``eval/legacy/expected.json``: la salida del prototipo v1 sobre unas muestras fijas.

Se ejecuta **una vez, antes** de la reestructura (HU-IA-40). HU-IA-40 compara su preset
``legacy`` contra este archivo para demostrar que no cambió el comportamiento.

Uso:
    set LEGACY_SAMPLES_DIR=C:\\ruta\\a\\muestras      (por defecto: %EVAL_DATA_DIR%\\legacy_samples)
    python -m eval.legacy.make_expected
"""

import json
import os
import sys
from pathlib import Path

from eval.labels import file_sha256
from eval.legacy import process_file_legacy, resolve_species_legacy
from eval.legacy.config_legacy import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

OUT = Path(__file__).resolve().parent / "expected.json"


def samples_dir() -> Path:
    explicit = os.getenv("LEGACY_SAMPLES_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples"


def describe(path: Path) -> dict:
    detections, frames = process_file_legacy(path)
    animal = [d for d in detections if d.get("class_name") == "animal"]
    label, conf = resolve_species_legacy(animal) if animal else (None, 0.0)
    return {
        "file_name": path.name,
        "sha256": file_sha256(path),
        "frames": frames,
        "detections": [
            {
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "box": d["box"],
                "species": d.get("species", [])[:5],
            }
            for d in detections
        ],
        "decision": {"label": label, "confidence": conf},
    }


def main() -> int:
    folder = samples_dir()
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    if not files:
        print(f"No hay muestras en {folder}", file=sys.stderr)
        return 2
    results = [describe(p) for p in files]
    payload = {
        "generated_with": "eval.legacy (prototipo v1, commit c6dac6e)",
        "samples_dir_name": folder.name,
        "files": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['file_name']}: frames={r['frames']} decision={r['decision']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
