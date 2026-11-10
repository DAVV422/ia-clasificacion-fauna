"""HU-IA-40 · CA-3 — la app de escritorio decide la misma carpeta que el prototipo para cada archivo.

Usa la lógica real de ``desktop_app.WildlifeDesktopApp`` (``_process_file`` + ``_resolve_species``)
sin abrir la ventana, y la compara con lo que hacía el prototipo (``eval/legacy/expected.json``).
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "eval/legacy/expected.json"
SAMPLES = Path(os.getenv("LEGACY_SAMPLES_DIR", str(Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples")))

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (EXPECTED.exists() and SAMPLES.is_dir() and (ROOT / "models/md_v5a.0.0.pt").exists()),
                                 reason="requiere expected.json, las muestras y los modelos")]


def prototype_folder(label) -> str:
    """Formato de carpeta del prototipo (copia de _resolve_species, commit c6dac6e)."""
    if not label:
        return "animal_desconocido"
    folder = label.replace(";", "_").replace(" ", "_")
    folder = "".join(c for c in folder if c.isalnum() or c in "_-")
    return folder or "animal_desconocido"


def test_desktop_destination_unchanged():
    import desktop_app
    from app.detector import WildlifeDetector

    app = desktop_app.WildlifeDesktopApp.__new__(desktop_app.WildlifeDesktopApp)  # sin crear la ventana
    app.detector = WildlifeDetector()
    app.detector.load_model()
    for item in json.loads(EXPECTED.read_text(encoding="utf-8"))["files"]:
        detections = app._process_file(SAMPLES / item["file_name"])
        animal = [d for d in detections if d.get("class_name") == "animal"]
        assert bool(animal) == (item["decision"]["label"] is not None)
        folder, conf = app._resolve_species(animal)
        assert folder == prototype_folder(item["decision"]["label"]), item["file_name"]
        assert conf == pytest.approx(item["decision"]["confidence"], abs=1e-4)
        assert all(d["crop_path"] is None for d in detections), "el escritorio ya no guarda recortes (F0-D3)"
