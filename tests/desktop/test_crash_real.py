"""HU-DS-30 CA-4 (integración) · Un proceso real terminado a la fuerza en medio de un "mover", y reanudar.

El hijo corre ``FolderJob`` con "mover" entre volúmenes (forzado) y se queda dormido en un punto de corte;
el padre lo mata con ``taskkill /F`` (``TerminateProcess``: como un corte de energía para el proceso)
y después reanuda en el mismo destino.
"""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from desktop.job.transfer import journal_path
from tests.desktop.fixtures import RELOCATED, STAYS_IN_PLACE, write_f_set_sint
from tests.desktop.test_move_verified import Log, run_job

pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]

CHILD = r"""
import queue, sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from desktop.job.actions import make_species_action
from desktop.job.folder_job import FolderJob
from tests.desktop.fixtures import SetModels
src, dest, step, marker = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
seen = []
def fault(name):
    seen.append(name)
    if name == step and seen.count(name) == 2:
        marker.write_text(name)
        time.sleep(120)
action = make_species_action(dest, "move", fault=fault, volume=False)
FolderJob(src, SetModels(), queue.Queue(), threading.Event(), action, dest=dest).run()
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill es de Windows")
@pytest.mark.parametrize("step", ["C3", "C4", "C5"])
def test_kill_process_during_move(tmp_path, step):
    src, dest, marker = tmp_path / "SD", tmp_path / "out", tmp_path / "marca"
    write_f_set_sint(src)
    hashes = {p.name: sha(p) for p in src.iterdir()}
    env = dict(os.environ, WWM_DATA_DIR=str(tmp_path / "wwm"), PYTHONIOENCODING="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(ROOT), str(src), str(dest), step, str(marker)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            if child.poll() is not None:
                pytest.fail(f"el hijo terminó antes del corte: {child.stderr.read().decode('utf-8', 'replace')}")
            assert time.monotonic() < deadline, "el hijo no llegó al punto de corte"
            time.sleep(0.1)
        subprocess.run(["taskkill", "/F", "/PID", str(child.pid)], capture_output=True, check=True)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
        child.stderr.close()
    assert journal_path(dest).exists()  # el corte dejó una operación a medias

    log = Log()
    run_job(src, dest, "move", log=log, volume=False)  # "Continuar"

    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    for name, folder in RELOCATED.items():
        assert not (src / name).exists(), name
        assert sha(dest / folder / name) == hashes[name]
        assert (dest / folder / f"{Path(name).stem}_fotograma.jpg").exists()
        assert not list((dest / folder).glob(f"{Path(name).stem} (*"))  # ningún duplicado
    assert {p.name for p in src.iterdir()} == STAYS_IN_PLACE
    assert len(list(dest.rglob("*_fotograma.jpg"))) == len(RELOCATED)
