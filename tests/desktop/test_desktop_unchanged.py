"""HU-IA-40 · CA-3 — la línea base del escritorio no cambió: misma carpeta que el prototipo para cada archivo.

Desde HU-DS-01 la app usa el modo precisión (``PipelineConfig.desktop()``); este test queda contra el preset
``legacy`` del escritorio y la regla de carpeta actual (``desktop.job.actions.species_folder_name``), y lo
compara con lo que hacía el prototipo (``eval/legacy/expected.json``). Así se sigue probando que la línea
base no se movió (riesgo 2 de HU-DS-01).
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "eval/legacy/expected.json"
SAMPLES = Path(os.getenv("LEGACY_SAMPLES_DIR", str(Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples")))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (EXPECTED.exists() and SAMPLES.is_dir() and (ROOT / "models/md_v5a.0.0.pt").exists()),
        reason="requiere expected.json, las muestras y los modelos",
    ),
]


def prototype_folder(label) -> str:
    """Formato de carpeta del prototipo (copia de _resolve_species, commit c6dac6e)."""
    if not label:
        return "animal_desconocido"
    folder = label.replace(";", "_").replace(" ", "_")
    folder = "".join(c for c in folder if c.isalnum() or c in "_-")
    return folder or "animal_desconocido"


def test_desktop_destination_unchanged():
    from app.detector import WildlifeDetector
    from desktop.job.actions import species_folder_name
    from wwm_pipeline.aggregation import legacy_best
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    detector = WildlifeDetector()
    detector.load_model()
    baseline = PipelineConfig.legacy_desktop()  # el preset del prototipo
    for item in json.loads(EXPECTED.read_text(encoding="utf-8"))["files"]:
        detections = classify_file(SAMPLES / item["file_name"], None, baseline, detector.models).all_detections
        animal = [d for d in detections if d.get("class_name") == "animal"]
        assert bool(animal) == (item["decision"]["label"] is not None)
        decision = legacy_best(animal)
        folder, conf = species_folder_name(decision.label), decision.confidence
        assert folder == prototype_folder(item["decision"]["label"]), item["file_name"]
        assert conf == pytest.approx(item["decision"]["confidence"], abs=1e-4)
        assert all(d["crop_path"] is None for d in detections), "el escritorio ya no guarda recortes (F0-D3)"
