"""HU-IA-11 · CA-6 y CA-7 con el material real de WWF (``estacion_1``, fuera del repo).

``ESTACION_1_DIR`` apunta a la carpeta (por defecto ``../docs-interno/estacion_1`` junto al repo).
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", str(ROOT.parent / "docs-interno" / "estacion_1")))
MD = ROOT / "models/md_v5a.0.0.pt"
F_EARLY = ESTACION_1 / "WhatsApp Video 2026-05-30 at 9.49.38 AM (1).mp4"  # #21: animal solo entre 0 y 2 s
VFR = ESTACION_1 / "WhatsApp Video 2026-05-30 at 9.49.39 AM.mp4"  # #25: 3,57 fps declarados, FPS variable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ESTACION_1.is_dir(), reason="requiere la carpeta estacion_1"),
]


def test_timestamps_follow_decoder_on_vfr_video():
    """CA-7: el tiempo de cada frame es el del decodificador (no índice / fps, que aquí se desvía > 1 s)."""
    import cv2

    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.sampling import sample_video

    cap = cv2.VideoCapture(str(VFR))
    fps = cap.get(cv2.CAP_PROP_FPS)
    real = []
    while cap.grab():
        real.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    assert max(abs(i / fps - t) for i, t in enumerate(real)) > 1.0  # el video sí es de FPS variable

    sampled = sample_video(str(VFR), PipelineConfig.desktop().sampling_params)
    for frame in sampled.frames:
        assert abs(frame.t - real[frame.index]) < 0.01


@pytest.mark.skipif(not MD.exists(), reason="requiere los modelos")
def test_early_animal_detected():
    """CA-6 (F-EARLY): el prototipo da "sin animal"; el muestreo denso lo encuentra en los primeros 2 s."""
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from tests.model_paths import registry
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    models = registry().load()
    legacy = classify_file(F_EARLY, None, PipelineConfig.legacy_desktop(), models)
    assert legacy.decision.label is None  # lo que hace hoy la app (sondeo del 10/11/2026)

    dense = classify_file(F_EARLY, None, PipelineConfig.desktop(), models)
    assert dense.decision.category == "animal" and dense.decision.label
    animal_times = [f.timestamp_seconds for f in dense.frames if any(d["class_name"] == "animal" for d in f.detections)]
    assert animal_times and min(animal_times) <= 2.0
