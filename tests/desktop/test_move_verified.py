"""HU-DS-30 · Mover sin riesgo de pérdida: copia verificada, diario, cortes en cada paso y reanudación."""

import hashlib
import queue
import threading
from collections import Counter, namedtuple
from pathlib import Path

import pytest

import desktop.job.transfer as transfer
from desktop import paths
from desktop.job.actions import make_species_action
from desktop.job.events import FileDone, JobError
from desktop.job.folder_job import FolderJob, list_media
from desktop.job.transfer import SimulatedCrash, journal_path, reconcile
from tests.desktop.fixtures import RELOCATED, STAYS_IN_PLACE, SetModels, write_f_set_sint

pytest.importorskip("cv2")
DiskUsage = namedtuple("DiskUsage", "total used free")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    monkeypatch.setattr(transfer, "RETRY_WAIT_S", 0)
    paths.ensure_dirs()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Log:
    """Un registro en memoria con la interfaz que HU-DS-14 cumplirá con ``JobLog``."""

    def __init__(self):
        self.lines = []

    def record(self, line, extra):
        self.lines.append(line)

    def has_line(self, line):
        return any(
            x.file_sha256 == line.file_sha256 and x.status == line.status and x.desktop_action == line.desktop_action
            for x in self.lines
        )

    def done_names(self):
        return {x.file_name for x in self.lines if x.status.value == "done"}

    def skip(self, path: Path) -> bool:
        return path.name in self.done_names()


def run_job(src, dest, action, log=None, **kwargs):
    events: queue.Queue = queue.Queue()
    log = log or Log()
    summary = FolderJob(
        src,
        kwargs.pop("models", SetModels()),
        events,
        kwargs.pop("stop", threading.Event()),
        make_species_action(dest, action, **kwargs),
        dest=dest,
        record=log.record,
        has_line=log.has_line,
        skip=log.skip,
    ).run()
    out = []
    while not events.empty():
        out.append(events.get_nowait())
    return summary, out, log


def crash_at(step: str, nth: int = 1, steps=None):
    seen = Counter()

    def fault(name):
        if steps is not None:
            steps.append(name)
        seen[name] += 1
        if name == step and seen[name] == nth:
            raise SimulatedCrash(step)

    return fault


@pytest.fixture
def sd(tmp_path):
    src = tmp_path / "SD"
    write_f_set_sint(src)
    return src


def files_in(folder: Path):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# CA-1 · Mover entre discos: el orden
# ---------------------------------------------------------------------------
def test_move_verified_happy_path(sd, tmp_path):
    dest = tmp_path / "out"
    hashes = {p.name: sha(p) for p in sd.iterdir()}
    steps = []
    summary, events, log = run_job(sd, dest, "move", volume=False, fault=crash_at("never", steps=steps))
    assert steps == ["C1", "C2", "C3", "C4", "C5", "C6"] * len(RELOCATED)
    for name, folder in RELOCATED.items():
        assert not (sd / name).exists()
        assert sha(dest / folder / name) == hashes[name]
        assert (dest / folder / f"{Path(name).stem}_fotograma.jpg").exists()
    assert {p.name for p in sd.iterdir()} == STAYS_IN_PLACE
    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    assert summary.relocated == len(RELOCATED)
    moved = [x for x in log.lines if x.desktop_action.type.value == "move"]
    assert {x.file_name for x in moved} == set(RELOCATED)


# ---------------------------------------------------------------------------
# CA-2 · Mismo disco: renombrar, sin copiar
# ---------------------------------------------------------------------------
def test_same_volume_move_is_rename(sd, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transfer, "copy_with_fsync", lambda *a: calls.append(a))
    dest = tmp_path / "out"
    size = (sd / "especie_a.mp4").stat().st_size
    run_job(sd, dest, "move", volume=True)
    assert calls == []
    assert (dest / "panthera_onca_jaguar" / "especie_a.mp4").stat().st_size == size
    assert not (sd / "especie_a.mp4").exists()


def test_same_volume_detected_for_tmp(sd, tmp_path):
    (tmp_path / "out").mkdir()
    assert transfer.same_volume(sd / "especie_a.mp4", tmp_path / "out")


# ---------------------------------------------------------------------------
# CA-3 · Copia dañada
# ---------------------------------------------------------------------------
def test_move_keeps_original_on_corrupt_copy(sd, tmp_path):
    dest = tmp_path / "out"
    original = sha(sd / "especie_a.mp4")

    def corrupt(step):
        if step == "C3":
            for part in dest.rglob("especie_a.mp4.part"):
                data = bytearray(part.read_bytes())
                data[100] ^= 0xFF
                part.write_bytes(bytes(data))

    _, events, log = run_job(sd, dest, "move", volume=False, fault=corrupt)
    done = {e.file_name: e for e in events if isinstance(e, FileDone)}
    assert done["especie_a.mp4"].status == "error"
    assert done["especie_a.mp4"].error_code == "FILE_MOVE_VERIFY_FAILED"
    assert sha(sd / "especie_a.mp4") == original
    assert not (dest / "panthera_onca_jaguar" / "especie_a.mp4").exists()
    assert not (dest / "panthera_onca_jaguar" / "especie_a_fotograma.jpg").exists()
    assert not list(dest.rglob("*.part")) and not journal_path(dest).exists()
    line = [x for x in log.lines if x.file_name == "especie_a.mp4"][0]
    assert line.status.value == "error" and line.desktop_action.type.value == "left_in_place"


# ---------------------------------------------------------------------------
# CA-4 · Un corte en cada paso, y continuar
# ---------------------------------------------------------------------------
MODES = [("copy", False), ("move", False), ("move", True)]


@pytest.mark.parametrize("action,same", MODES, ids=["copiar", "mover-otro-disco", "mover-mismo-disco"])
@pytest.mark.parametrize("step", ["C1", "C2", "C3", "C4", "C5", "C6"])
def test_crash_at_each_step_then_resume(sd, tmp_path, action, same, step):
    if step == "C3" and action == "move" and same:
        pytest.skip("en el mismo disco no hay copia (.part): C3 no existe")
    dest = tmp_path / "out"
    hashes = {p.name: sha(p) for p in sd.iterdir()}
    log = Log()
    with pytest.raises(SimulatedCrash):  # el segundo archivo que se reubica (genero_c) "se corta"
        run_job(sd, dest, action, log=log, volume=same, fault=crash_at(step, nth=2))
    assert journal_path(dest).exists()

    run_job(sd, dest, action, log=log, volume=same)  # "Continuar"

    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    for name, folder in RELOCATED.items():
        copies = [
            p
            for p in (dest / folder).iterdir()
            if p.stem.startswith(Path(name).stem) and p.suffix == Path(name).suffix and "_fotograma" not in p.name
        ]
        assert len(copies) == 1, f"{name}: {copies}"  # ni perdido ni duplicado
        assert sha(copies[0]) == hashes[name]
        assert (dest / folder / f"{copies[0].stem}_fotograma.jpg").exists()
        assert (sd / name).exists() == (action == "copy")
    frames = list(dest.rglob("*_fotograma.jpg"))
    assert len(frames) == len(RELOCATED)  # ningún fotograma huérfano
    for name in STAYS_IN_PLACE:
        assert (sd / name).exists() and sha(sd / name) == hashes[name]
    done = Counter(x.file_name for x in log.lines if x.status.value == "done")
    assert all(n == 1 for n in done.values()), done  # una sola línea done por archivo
    listed = {p.name for p in list_media(tmp_path / "SD")} | set(RELOCATED)
    errors = {x.file_name for x in log.lines if x.status.value == "error"}
    assert set(done) | errors == listed and not (set(done) & errors)


def test_reconcile_without_journal_is_noop(tmp_path):
    assert reconcile(tmp_path / "no_existe").outcome == "none"
    assert reconcile(tmp_path).outcome == "none"


def test_unreadable_journal_is_discarded(tmp_path):
    journal_path(tmp_path).parent.mkdir(parents=True)
    journal_path(tmp_path).write_text("{ a medias", encoding="utf-8")
    assert reconcile(tmp_path).outcome == "none" and not journal_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# CA-5 · .part huérfano
# ---------------------------------------------------------------------------
def test_resume_cleans_part_files(sd, tmp_path):
    dest = tmp_path / "out"
    folder = dest / "panthera_onca_jaguar"
    folder.mkdir(parents=True)
    (folder / "especie_a.mp4.part").write_bytes(b"a medias")
    (folder / "otro_fotograma.jpg.part").write_bytes(b"a medias")
    (dest / "_wwm").mkdir()
    keep = dest / "_wwm" / "clasificaciones.jsonl"
    keep.write_text("", encoding="utf-8")
    run_job(sd, dest, "copy", volume=False)
    assert not list(dest.rglob("*.part")) and keep.exists()
    assert (folder / "especie_a.mp4").exists()


# ---------------------------------------------------------------------------
# CA-6 · Copiar: verificado y sin tocar el original
# ---------------------------------------------------------------------------
def test_copy_verified_leaves_original_untouched(sd, tmp_path):
    src = sd / "especie_a.mp4"
    before = (sha(src), src.stat().st_mtime_ns)
    steps = []
    dest = tmp_path / "out"
    run_job(sd, dest, "copy", volume=True, fault=crash_at("never", steps=steps))
    assert "C3" in steps  # "copiar" siempre copia y verifica, aunque sea el mismo disco
    assert (sha(src), src.stat().st_mtime_ns) == before
    copy = dest / "panthera_onca_jaguar" / "especie_a.mp4"
    assert sha(copy) == before[0] and copy.stat().st_mtime_ns == before[1]  # como copy2


# ---------------------------------------------------------------------------
# CA-7 · Disco lleno
# ---------------------------------------------------------------------------
def test_disk_full_stops_job(sd, tmp_path):
    dest = tmp_path / "out"
    summary, events, log = run_job(sd, dest, "copy", disk_usage=lambda p: DiskUsage(1, 1, 10 * 1024 * 1024))
    errors = [e for e in events if isinstance(e, JobError)]
    assert [e.code for e in errors] == ["DISK_SPACE_LOW"]
    assert summary.stopped and summary.errors == 1  # danado_j.mp4, que va antes; el del disco lleno no cuenta
    stopped = [e for e in events if isinstance(e, FileDone) and e.status == "cancelled"]
    assert [e.file_name for e in stopped] == ["especie_a.mp4"]  # el primero que se reubica
    assert "especie_a.mp4" not in {x.file_name for x in log.lines}
    assert not [p for p in files_in(dest) if not p.startswith("_wwm/")]


def test_disk_full_during_copy(sd, tmp_path, monkeypatch):
    import errno

    def full(source, part):
        Path(part).write_bytes(b"x" * 10)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(transfer, "copy_with_fsync", full)
    dest = tmp_path / "out"
    summary, events, _ = run_job(sd, dest, "move", volume=False)
    assert [e.code for e in events if isinstance(e, JobError)] == ["DISK_SPACE_LOW"]
    assert (sd / "especie_a.mp4").exists() and not list(dest.rglob("*.part"))
    assert not list(dest.rglob("*_fotograma.jpg"))


# ---------------------------------------------------------------------------
# CA-8 · Original bloqueado: queda como copia
# ---------------------------------------------------------------------------
def test_locked_original_becomes_copy(sd, tmp_path, monkeypatch):
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == "especie_a.mp4" and self.parent == sd:
            raise PermissionError("en uso por otro programa")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    dest = tmp_path / "out"
    summary, events, log = run_job(sd, dest, "move", volume=False)
    done = {e.file_name: e for e in events if isinstance(e, FileDone)}
    assert done["especie_a.mp4"].status == "done" and done["especie_a.mp4"].warning_code == "SOURCE_NOT_DELETED"
    assert done["especie_a.mp4"].action.type.value == "copy"
    assert (sd / "especie_a.mp4").exists() and (dest / "panthera_onca_jaguar" / "especie_a.mp4").exists()
    line = [x for x in log.lines if x.file_name == "especie_a.mp4"][0]
    assert line.desktop_action.type.value == "copy"
    assert done["genero_c.mp4"].action.type.value == "move" and not (sd / "genero_c.mp4").exists()
    assert summary.not_deleted == 1
