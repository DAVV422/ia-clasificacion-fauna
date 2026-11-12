"""HU-DS-01 · CA-3 y CA-4 con los modelos reales y ``estacion_1`` (fuera del repo).

Usa una copia de 3 videos cortos en ``tmp_path`` (no toca la carpeta original): F-EARLY (#21), #24 y #25.
"""

import os
import queue
import shutil
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", str(ROOT.parent / "docs-interno" / "estacion_1")))
MD = ROOT / "models/md_v5a.0.0.pt"
NAMES = [
    "WhatsApp Video 2026-05-30 at 9.49.38 AM (1).mp4",  # #21, F-EARLY
    "WhatsApp Video 2026-05-30 at 9.49.39 AM (1).mp4",  # #24
    "WhatsApp Video 2026-05-30 at 9.49.39 AM.mp4",  # #25, 3,57 fps
]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (ESTACION_1.is_dir() and MD.exists()), reason="requiere estacion_1 y los modelos"),
]


@pytest.fixture(scope="module")
def models():
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from wwm_pipeline.models import ModelRegistry

    return ModelRegistry(MD).load()


@pytest.fixture
def job_results(models, tmp_path, monkeypatch):
    from desktop import paths
    from desktop.job.actions import make_species_action
    from desktop.job.events import FileDone
    from desktop.job.folder_job import FolderJob

    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()
    src = tmp_path / "entrada"
    src.mkdir()
    for name in NAMES:
        shutil.copy2(ESTACION_1 / name, src / name)
    events: queue.Queue = queue.Queue()
    FolderJob(
        src, models, events, threading.Event(), apply_action=make_species_action(tmp_path / "salida", "copy")
    ).run()
    done = {}
    while not events.empty():
        e = events.get_nowait()
        if isinstance(e, FileDone):
            done[e.file_name] = e
    return src, done, tmp_path / "salida"


def test_early_animal_detected_by_folder_job(job_results):
    """CA-3: F-EARLY ya no queda como "sin animal" (con el preset del prototipo sí)."""
    _, done, _ = job_results
    early = done[NAMES[0]]
    assert early.status == "done"
    assert early.result.decision.category == "animal"
    assert early.result.taxonomy.scientific_name == "Panthera onca"
    assert early.action.type.value == "copy" and early.action.destination_folder == "panthera_onca_jaguar"


def test_folder_job_matches_direct_pipeline(job_results, models):
    """CA-4: la app no altera el resultado del pipeline."""
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.pipeline import classify_file

    src, done, _ = job_results
    assert all(d.status == "done" for d in done.values())
    from wwm_pipeline.package import PackageLine

    for d in done.values():  # HU-DS-02 CA-7: cada línea valida
        PackageLine.model_validate(d.line.model_dump(mode="json"))
        assert d.line.desktop_action.destination_folder == "panthera_onca_jaguar"
    for name in NAMES:
        direct = classify_file(src / name, name, PipelineConfig.desktop(), models)
        via_app = done[name].result
        assert via_app.vote.taxonomy == direct.vote.taxonomy
        assert via_app.vote.confidence == direct.vote.confidence
        assert via_app.vote.votes == direct.vote.votes
        assert via_app.vote.alternatives == direct.vote.alternatives
        assert via_app.frames_analyzed == direct.frames_analyzed
        assert via_app.evidence.frame_index == direct.evidence.frame_index
        assert via_app.sha256 == direct.sha256


def test_estacion_1_has_evidence_next_to_files(job_results):
    """HU-DS-10 CA-10: cada video copiado tiene su _fotograma.jpg al lado, con la franja y ≤ 960 px."""
    import cv2
    import numpy as np

    _, done, dest = job_results
    folder = dest / "panthera_onca_jaguar"
    for name in NAMES:
        stem = Path(name).stem
        assert (folder / name).exists()
        jpg = folder / f"{stem}_fotograma.jpg"
        img = cv2.imdecode(np.fromfile(str(jpg), np.uint8), cv2.IMREAD_COLOR)
        ev = done[name].line.evidence
        assert max(img.shape[:2]) <= 960 and img.shape[0] > ev.height and img.shape[1] == ev.width
