"""HU-DS-01 · FolderJob sin interfaz: orden de eventos, errores por archivo, detener y limpieza."""

import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from desktop import paths
from desktop.job.actions import make_species_action, prototype_label, species_folder_name
from desktop.job.events import FileDone, JobDone, JobError, JobStarted, Progress
from desktop.job.folder_job import FolderJob, configure_threads, list_media
from tests.fakes import FakeModels
from wwm_pipeline.errors import PipelineCancelled, PipelineError

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()
    return tmp_path / "wwm"


def scene(i: int, bright: bool = True):
    base = 170 if bright else 40
    img = np.full((180, 320, 3), base, np.uint8)
    img[::16, :] = 255 - base
    img[:, ::16] = 255 - base
    x = (i * 9) % 260
    img[50:130, x : x + 60] = 20 if bright else 90
    return img


def write_video(path: Path, n: int = 30, fps: float = 10, bright: bool = True) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for i in range(n):
        writer.write(scene(i, bright))
    writer.release()
    return path


def drain(q: "queue.Queue"):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def run_job(source, **kwargs):
    events: queue.Queue = queue.Queue()
    stop = kwargs.pop("stop", threading.Event())
    summary = FolderJob(source, kwargs.pop("models", FakeModels()), events, stop, **kwargs).run()
    return summary, drain(events)


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "entrada"
    src.mkdir()
    write_video(src / "a_jaguar.mp4")
    cv2.imwrite(str(src / "b_foto.JPG"), scene(0))
    write_video(src / "c_vacio.mp4", bright=False)
    (src / "notas.txt").write_text("no es un medio")
    (src / "sub").mkdir()
    write_video(src / "sub" / "d_no_recursivo.mp4")
    return src


# ---------------------------------------------------------------------------
# CA-5 · Eventos en orden
# ---------------------------------------------------------------------------
def test_folder_job_events_in_order(folder, tmp_path):
    summary, events = run_job(folder, apply_action=make_species_action(tmp_path / "salida", "copy"))
    assert isinstance(events[0], JobStarted) and events[0].total == 3
    assert isinstance(events[-1], JobDone) and events[-1].summary == summary
    done = [e for e in events if isinstance(e, FileDone)]
    assert [d.file_name for d in done] == ["a_jaguar.mp4", "b_foto.JPG", "c_vacio.mp4"]  # por nombre, no recursivo
    # (Progress… → FileDone) × N: cada Progress pertenece al archivo en curso
    current = 1
    for e in events[1:-1]:
        if isinstance(e, Progress):
            assert e.file_index == current
        elif isinstance(e, FileDone):
            assert e.file_index == current
            current += 1
    assert any(isinstance(e, Progress) and e.stage == "classification" for e in events)
    assert (summary.total, summary.processed, summary.with_animals, summary.without_animals) == (3, 3, 2, 1)
    assert not summary.stopped
    copied = sorted(p.relative_to(tmp_path / "salida").as_posix() for p in (tmp_path / "salida").rglob("*.*"))
    assert copied == ["panthera_onca_jaguar/a_jaguar.mp4", "panthera_onca_jaguar/b_foto.JPG"]
    assert (folder / "a_jaguar.mp4").exists()  # copiar no borra el original
    assert list(paths.tmp_dir().iterdir()) == []


def test_move_action(folder, tmp_path):
    run_job(folder, apply_action=make_species_action(tmp_path / "salida", "move"))
    assert not (folder / "a_jaguar.mp4").exists() and (folder / "c_vacio.mp4").exists()
    assert (tmp_path / "salida" / "panthera_onca_jaguar" / "a_jaguar.mp4").exists()


def test_progress_is_throttled(folder):
    ticks = iter(i * 0.01 for i in range(100_000))  # reloj falso: 10 ms entre llamadas
    _, events = run_job(folder, clock=lambda: next(ticks))
    per_file = [e for e in events if isinstance(e, Progress) and e.file_index == 1 and e.stage == "detection"]
    total = per_file[-1].total
    assert per_file[-1].done == total  # el último siempre llega
    assert len(per_file) < total  # pero no uno por frame


# ---------------------------------------------------------------------------
# CA-6 · Errores por archivo
# ---------------------------------------------------------------------------
def test_folder_job_continues_after_file_error(folder):
    from wwm_pipeline.pipeline import classify_file

    def flaky(path, *args, **kwargs):
        if path.name == "b_foto.JPG":
            raise PipelineError("IA_DECODE_FAILED", "No se pudo leer la imagen.")
        return classify_file(path, *args, **kwargs)

    summary, events = run_job(folder, classify=flaky)
    done = [e for e in events if isinstance(e, FileDone)]
    assert [d.status for d in done] == ["done", "error", "done"]
    assert done[1].error_code == "IA_DECODE_FAILED" and done[1].result is None
    assert summary.errors == 1 and summary.processed == 3


def test_unexpected_exception_and_action_failure(folder):
    def broken_action(result, path):
        raise PermissionError("disco lleno")

    def boom(path, *args, **kwargs):
        raise RuntimeError("algo raro")

    _, events = run_job(folder, classify=boom)
    assert {e.error_code for e in events if isinstance(e, FileDone)} == {"IA_INTERNAL"}
    _, events = run_job(folder, apply_action=broken_action)
    codes = [e.error_code for e in events if isinstance(e, FileDone)]
    assert codes == ["DESKTOP_ACTION_FAILED"] * 3


def test_missing_folder_reports_error(tmp_path):
    summary, events = run_job(tmp_path / "no_existe")
    assert isinstance(events[0], JobError) and isinstance(events[-1], JobDone)
    assert summary.total == 0


def test_empty_folder(tmp_path):
    summary, events = run_job(tmp_path)
    assert [type(e) for e in events] == [JobStarted, JobDone] and summary.total == 0


# ---------------------------------------------------------------------------
# CA-7 · Detener
# ---------------------------------------------------------------------------
def test_folder_job_stop(folder):
    """stop.set() durante el 2.º archivo → se corta ese archivo, no se empieza el 3.º y JobDone(stopped=True)."""
    stop = threading.Event()

    class StopOnSecond(FakeModels):
        def run_detector(self, frame_rgb):
            if self.current == "b_foto.JPG":
                stop.set()
            return super().run_detector(frame_rgb)

    models = StopOnSecond()
    from wwm_pipeline.pipeline import classify_file

    def track(path, *args, **kwargs):
        models.current = path.name
        result = classify_file(path, *args, **kwargs)
        if kwargs["should_stop"]():  # una imagen es un solo frame: el pipeline no alcanza a cortarla
            raise PipelineCancelled(path.name)
        return result

    summary, events = run_job(folder, models=models, stop=stop, classify=track)
    done = [e for e in events if isinstance(e, FileDone)]
    assert [d.file_name for d in done] == ["a_jaguar.mp4", "b_foto.JPG"]
    assert done[1].status == "cancelled"
    assert isinstance(events[-1], JobDone) and events[-1].stopped
    assert summary.processed == 1


def test_stop_before_start(folder):
    stop = threading.Event()
    stop.set()
    summary, events = run_job(folder, stop=stop)
    assert [type(e) for e in events] == [JobStarted, JobDone] and summary.stopped


def test_stop_cancels_a_long_video(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    write_video(src / "largo.mp4", n=150, fps=15)
    stop = threading.Event()

    class StopAfterTen(FakeModels):
        def run_detector(self, frame_rgb):
            if self.detector_calls >= 10:
                stop.set()
            return super().run_detector(frame_rgb)

    models = StopAfterTen()
    _, events = run_job(src, models=models, stop=stop)
    assert [e.status for e in events if isinstance(e, FileDone)] == ["cancelled"]
    assert models.detector_calls < 20 and models.classifier_calls == 0


# ---------------------------------------------------------------------------
# Regla de carpeta (la del prototipo, desde la taxonomía nueva)
# ---------------------------------------------------------------------------
def test_prototype_folder_from_taxonomy():
    from tools._fixture_taxa import DASYPROCTA_GENUS, JAGUAR, TAPIR
    from wwm_pipeline.enums import Rank
    from wwm_pipeline.taxonomy import TaxonomyIndex

    index = TaxonomyIndex([JAGUAR, TAPIR, DASYPROCTA_GENUS])
    assert species_folder_name(prototype_label(index.to_taxonomy(JAGUAR, "es"))) == "panthera_onca_jaguar"
    assert species_folder_name(prototype_label(index.to_taxonomy(TAPIR, "es"))) == "tapirus_terrestris_lowland_tapir"
    genus = index.taxonomy_of(index.ancestor(index.lookup(JAGUAR), Rank.GENUS))
    assert species_folder_name(prototype_label(genus)) == "panthera__panthera_species"
    assert species_folder_name(prototype_label(index.to_taxonomy(DASYPROCTA_GENUS))) == "dasyprocta__dasyprocta_species"
    assert species_folder_name(None) == "animal_desconocido"


def test_list_media_and_threads(folder):
    assert [p.name for p in list_media(folder)] == ["a_jaguar.mp4", "b_foto.JPG", "c_vacio.mp4"]
    assert configure_threads() >= 1


# ---------------------------------------------------------------------------
# CA-9 · Limpieza
# ---------------------------------------------------------------------------
def test_no_legacy_in_desktop():
    for py in (ROOT / "desktop").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for banned in ("legacy_best", "legacy_desktop", "_analysis_worker"):
            assert banned not in text, (py.name, banned)


def test_folder_job_has_no_tkinter():
    for py in (ROOT / "desktop" / "job").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "import tkinter" not in text and "from tkinter" not in text, py.name
