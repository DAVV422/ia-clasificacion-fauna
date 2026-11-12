"""HU-DS-14 · Registro local (clasificaciones.jsonl) y reanudación."""

import json
import os
import queue
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

import desktop.job.job_log as job_log_mod
from desktop import paths
from desktop.job.actions import make_species_action
from desktop.job.events import FileDone, JobStarted
from desktop.job.folder_job import FolderJob, list_media
from desktop.job.job_log import JobLog
from desktop.job.record import error_line
from desktop.job.transfer import SimulatedCrash, journal_path
from desktop.ui.texts import resume_prompt
from tests.desktop.fixtures import RELOCATED, SetModels, write_f_set_sint
from wwm_pipeline.package import PackageLine
from wwm_pipeline.pipeline import classify_file

pytest.importorskip("cv2")
SHA = "a" * 64


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()


@pytest.fixture
def sd(tmp_path):
    src = tmp_path / "SD"
    write_f_set_sint(src)
    return src


def run(src, dest, action="copy", log=None, stop=None, classify=classify_file, fault=None, resume=True):
    events: queue.Queue = queue.Queue()
    log = log or JobLog(dest)
    if not resume:
        log.backup()
    log.start(src, action)
    kwargs = {"fault": fault} if fault else {}
    try:
        summary = FolderJob(
            src,
            SetModels(),
            events,
            stop or threading.Event(),
            make_species_action(dest, action, volume=False, **kwargs),
            dest=dest,
            classify=classify,
            record=log.append,
            has_line=log.has_line,
            skip=log.is_processed,
        ).run()
    finally:
        log.close()
    out = []
    while not events.empty():
        out.append(events.get_nowait())
    return summary, out


def raw_lines(dest):
    text = (dest / "_wwm" / "clasificaciones.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def stop_after(n, stop):
    calls = []

    def classify(*args, **kwargs):
        result = classify_file(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == n:
            stop.set()  # termina este archivo (ya clasificado) y no empieza el siguiente
        return result

    return classify


# ---------------------------------------------------------------------------
# CA-1 · CA-7 · Una línea válida por archivo, sin rutas absolutas en lo exportable
# ---------------------------------------------------------------------------
def test_joblog_lines_validate_package_schema(sd, tmp_path):
    dest = tmp_path / "out"
    run(sd, dest)
    lines = raw_lines(dest)
    assert {x["file_name"] for x in lines} == {p.name for p in list_media(sd)}
    for raw in lines:
        local = raw["local"]
        assert set(local) == {"source_mtime_ns", "destination_file", "evidence_file", "run_id", "processed_at"}
        assert local["run_id"].startswith("j-") and isinstance(local["source_mtime_ns"], int)
        exported = PackageLine.model_validate(raw).model_dump(mode="json")
        assert "local" not in exported
        assert str(tmp_path) not in json.dumps(exported) and str(tmp_path) not in json.dumps(local)
    by_name = {x["file_name"]: x["local"] for x in lines}
    assert by_name["especie_a.mp4"]["destination_file"] == "panthera_onca_jaguar/especie_a.mp4"
    assert by_name["especie_a.mp4"]["evidence_file"] == "panthera_onca_jaguar/especie_a_fotograma.jpg"
    assert by_name["vacio_f.mp4"]["destination_file"] is None
    header = json.loads((dest / "_wwm" / "analisis.json").read_text(encoding="utf-8"))
    assert header["source"] == str(sd) and header["action"] == "copy" and header["app_version"]


# ---------------------------------------------------------------------------
# CA-2 · Durable antes de FileDone
# ---------------------------------------------------------------------------
def test_line_fsynced_before_file_done(sd, tmp_path, monkeypatch):
    dest = tmp_path / "out"
    order = []
    real_fsync = os.fsync
    monkeypatch.setattr(job_log_mod.os, "fsync", lambda fd: order.append("fsync") or real_fsync(fd))

    class SpyQueue(queue.Queue):
        def put(self, item, *a, **k):
            if isinstance(item, FileDone):
                order.append("FileDone")
            super().put(item, *a, **k)

    log = JobLog(dest)
    log.start(sd, "copy")
    FolderJob(
        sd,
        SetModels(),
        SpyQueue(),
        threading.Event(),
        make_species_action(dest, "copy"),
        dest=dest,
        record=log.append,
        has_line=log.has_line,
        skip=log.is_processed,
    ).run()
    log.close()
    done = [i for i, x in enumerate(order) if x == "FileDone"]
    assert done and all(order[i - 1] == "fsync" for i in done)


# ---------------------------------------------------------------------------
# CA-3 · Continuar: salta los done, reintenta los error
# ---------------------------------------------------------------------------
def test_resume_skips_done_and_retries_errors(sd, tmp_path):
    dest = tmp_path / "out"
    stop = threading.Event()
    first, _ = run(sd, dest, stop=stop, classify=stop_after(4, stop))
    assert first.stopped and first.processed == 5  # el dañado + 4 clasificados
    before = Counter(x["file_name"] for x in raw_lines(dest))
    errors_before = {x["file_name"] for x in raw_lines(dest) if x["status"] == "error"}
    assert errors_before == {"danado_j.mp4"}  # el dañado va antes en orden alfabético

    _, events = run(sd, dest)
    started = [e for e in events if isinstance(e, JobStarted)][0]
    listed = len(list_media(sd))
    assert started.already_done == 4 and started.total == listed - 4  # los 4 done se saltan; el error se reintenta
    analyzed = {e.file_name for e in events if isinstance(e, FileDone)}
    assert "danado_j.mp4" in analyzed and not (analyzed & (set(before) - errors_before))
    after = Counter(x["file_name"] for x in raw_lines(dest) if x["status"] == "done")
    assert set(after) == {p.name for p in list_media(sd)} - {"danado_j.mp4"} and set(after.values()) == {1}
    for name, folder in RELOCATED.items():  # ninguna copia repetida
        assert not list((dest / folder).glob(f"{Path(name).stem} (*"))


def test_nothing_left_to_do(sd, tmp_path):
    dest = tmp_path / "out"
    run(sd, dest)
    _, events = run(sd, dest)
    started = [e for e in events if isinstance(e, JobStarted)][0]
    assert started.total == 1 and started.already_done == len(list_media(sd)) - 1  # solo el dañado


# ---------------------------------------------------------------------------
# CA-4 · Identificación
# ---------------------------------------------------------------------------
def test_skip_without_hash(sd, tmp_path, monkeypatch):
    dest = tmp_path / "out"
    run(sd, dest)
    log = JobLog(dest)
    log.start(sd, "copy")
    monkeypatch.setattr(job_log_mod, "file_sha256", lambda p: pytest.fail("no debía leer el archivo"))
    assert log.is_processed(sd / "especie_a.mp4")
    log.close()


def test_renamed_file_recognized_by_hash(sd, tmp_path):
    dest = tmp_path / "out"
    run(sd, dest)
    (sd / "especie_a.mp4").rename(sd / "renombrado.mp4")
    log = JobLog(dest)
    log.start(sd, "copy")
    assert log.is_processed(sd / "renombrado.mp4")
    log.close()


def test_duplicate_content_with_other_name_is_analyzed(sd, tmp_path):
    """Dos archivos idénticos con nombres distintos (especie_a y genero_c en F-SET-SINT): no es un renombre."""
    assert (sd / "especie_a.mp4").read_bytes() == (sd / "genero_c.mp4").read_bytes()
    dest = tmp_path / "out"
    stop = threading.Event()
    run(sd, dest, stop=stop, classify=stop_after(1, stop))  # solo especie_a
    log = JobLog(dest)
    log.start(sd, "copy")
    assert log.is_processed(sd / "especie_a.mp4") and not log.is_processed(sd / "genero_c.mp4")
    log.close()


def test_same_name_other_content_is_analyzed(sd, tmp_path):
    dest = tmp_path / "out"
    run(sd, dest)
    path = sd / "especie_a.mp4"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF  # mismo nombre y tamaño, otro contenido y otra fecha
    path.write_bytes(bytes(data))
    log = JobLog(dest)
    log.start(sd, "copy")
    assert not log.is_processed(path)
    log.close()


# ---------------------------------------------------------------------------
# CA-5 · Empezar de nuevo
# ---------------------------------------------------------------------------
def test_start_over_backs_up_log(sd, tmp_path):
    dest = tmp_path / "out"
    run(sd, dest)
    copied = sorted(p for p in dest.rglob("*") if p.is_file() and "_wwm" not in p.parts)
    n_lines = len(raw_lines(dest))
    log = JobLog(dest, clock=lambda: datetime(2026, 11, 16, 9, 12, 0))
    backup = log.backup()
    assert backup.name == "clasificaciones.20261116-091200.bak.jsonl"
    assert len(backup.read_text(encoding="utf-8").splitlines()) == n_lines
    assert not (dest / "_wwm" / "clasificaciones.jsonl").exists()
    assert sorted(p for p in dest.rglob("*") if p.is_file() and "_wwm" not in p.parts) == copied
    _, events = run(sd, dest, log=JobLog(dest))
    assert [e for e in events if isinstance(e, JobStarted)][0].already_done == 0
    assert (dest / "panthera_onca_jaguar" / "especie_a (2).mp4").exists()  # E-08: regla del nombre repetido


# ---------------------------------------------------------------------------
# CA-6 · Línea truncada
# ---------------------------------------------------------------------------
def test_truncated_last_line_ignored(sd, tmp_path, caplog):
    dest = tmp_path / "out"
    stop = threading.Event()
    run(sd, dest, stop=stop, classify=stop_after(3, stop))
    path = dest / "_wwm" / "clasificaciones.jsonl"
    good = path.read_bytes()
    path.write_bytes(good + b'{"file_name": "especie_b.jpg", "file_sha')  # el corte durante la escritura
    assert len(JobLog(dest).read()) == 4  # el dañado + 3; se ignora la fracción
    run(sd, dest)
    assert path.read_bytes().startswith(good)
    for raw in path.read_text(encoding="utf-8").splitlines():
        json.loads(raw)  # todas las líneas son JSON válido
    assert "a medias" in caplog.text


def test_corrupt_middle_line_is_skipped(tmp_path, caplog):
    log = JobLog(tmp_path)
    log.start(tmp_path, "copy")
    log.append(error_line("a.mp4", SHA, 10, "IA_DECODE_FAILED"))
    log.close()
    with open(log.path, "a", encoding="utf-8") as fh:
        fh.write("no es json\n")
    log2 = JobLog(tmp_path)
    log2.start(tmp_path, "copy")
    log2.append(error_line("b.mp4", SHA, 10, "IA_DECODE_FAILED"))
    log2.close()
    assert [e.line.file_name for e in JobLog(tmp_path).read()] == ["a.mp4", "b.mp4"]
    assert "corrupta" in caplog.text


# ---------------------------------------------------------------------------
# DS-30 + DS-14 · Un corte en "Copiar" ya no duplica al continuar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", ["C4", "C5", "C6"])
def test_copy_crash_then_resume_with_real_log(sd, tmp_path, step):
    from tests.desktop.test_move_verified import crash_at

    dest = tmp_path / "out"
    with pytest.raises(SimulatedCrash):
        run(sd, dest, fault=crash_at(step, nth=2))
    assert journal_path(dest).exists()
    run(sd, dest)
    assert not journal_path(dest).exists()
    done = Counter(x["file_name"] for x in raw_lines(dest) if x["status"] == "done")
    assert set(done.values()) == {1}
    for name, folder in RELOCATED.items():
        assert not list((dest / folder).glob(f"{Path(name).stem} (*")), name
        assert (dest / folder / name).exists()


# ---------------------------------------------------------------------------
# El diálogo
# ---------------------------------------------------------------------------
def test_resume_prompt_text(tmp_path):
    info = job_log_mod.LogInfo(
        done=120, errors=3, last_at=datetime(2026, 11, 16, 9, 12), source=str(tmp_path / "SD"), action="copy"
    )
    title, text = resume_prompt(info, str(tmp_path / "SD"), 220)
    assert "(120 archivos analizados, 16/11/2026 09:12)" in text
    assert "analizar solo los 220 que faltan" in text and "3 con error" in text
    assert "Este registro es de la carpeta" not in text
    _, other = resume_prompt(info, "F:\\DCIM\\100MEDIA", 220)
    assert f"Este registro es de la carpeta {tmp_path / 'SD'}." in other  # F3-D6: se avisa, se deja continuar


def test_info_and_remaining_estimate(sd, tmp_path):
    dest = tmp_path / "out"
    stop = threading.Event()
    run(sd, dest, stop=stop, classify=stop_after(4, stop))
    log = JobLog(dest)
    info = log.info()
    assert (info.done, info.errors) == (4, 1) and info.source == str(sd) and info.last_at is not None
    assert log.remaining_estimate(list_media(sd)) == len(list_media(sd)) - 4
