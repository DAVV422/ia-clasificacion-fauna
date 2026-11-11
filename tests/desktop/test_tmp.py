"""HU-DS-04 · CA-5 — tmp_dir() queda vacío después de cada archivo y al cerrar, aunque haya errores."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from desktop import paths
from tests.fakes import FakeModels

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()
    return tmp_path / "wwm"


def test_file_tmp_removed_on_success_and_error(data_dir):
    with paths.file_tmp() as folder:
        (folder / "frame.jpg").write_bytes(b"x")
    assert list(paths.tmp_dir().iterdir()) == []

    with pytest.raises(RuntimeError), paths.file_tmp() as folder:
        (folder / "frame.jpg").write_bytes(b"x")
        raise RuntimeError("falla a mitad del archivo")
    assert list(paths.tmp_dir().iterdir()) == []


def test_clear_tmp_removes_leftovers(data_dir):
    (paths.tmp_dir() / "job_viejo").mkdir()
    (paths.tmp_dir() / "job_viejo" / "a.bin").write_bytes(b"x")
    (paths.tmp_dir() / "suelto.tmp").write_bytes(b"x")
    paths.clear_tmp()
    assert list(paths.tmp_dir().iterdir()) == []


def test_process_file_leaves_tmp_empty(data_dir, tmp_path):
    """El análisis de la app (FolderJob, HU-DS-01) deja tmp vacío, también cuando un archivo falla."""
    import queue
    import threading

    cv2 = pytest.importorskip("cv2")
    from desktop.job.events import FileDone
    from desktop.job.folder_job import FolderJob

    src = tmp_path / "entrada"
    src.mkdir()
    cv2.imwrite(str(src / "jaguar.jpg"), np.full((64, 64, 3), 200, dtype=np.uint8))
    (src / "roto.mp4").write_bytes(b"no es un video")
    seen = []

    class Spy(FakeModels):
        def run_detector(self, frame_rgb):
            seen.append([p.name for p in paths.tmp_dir().iterdir()])  # durante el archivo: su carpeta job_*
            return super().run_detector(frame_rgb)

    events: queue.Queue = queue.Queue()
    FolderJob(src, Spy(), events, threading.Event()).run()
    done = []
    while not events.empty():
        e = events.get_nowait()
        if isinstance(e, FileDone):
            done.append(e)
    assert [d.status for d in done] == ["done", "error"]
    assert seen and all(len(names) == 1 and names[0].startswith("job_") for names in seen)
    assert list(paths.tmp_dir().iterdir()) == []


def test_tmp_cleared_at_exit(tmp_path):
    """Al terminar el proceso (atexit), lo que quedó en tmp se borra."""
    data = tmp_path / "wwm"
    code = (
        "import sys; sys.argv=['x','--selftest']\n"
        "from desktop import main, paths\n"
        "orig = main.selftest\n"
        "def fake():\n"
        "    (paths.tmp_dir() / 'resto.bin').write_bytes(b'x'); return 0\n"
        "main.selftest = fake\n"
        "sys.exit(main.run())\n"
    )
    env = {**os.environ, "WWM_DATA_DIR": str(data), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert list((data / "tmp").iterdir()) == []
