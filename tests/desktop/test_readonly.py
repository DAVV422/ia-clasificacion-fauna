"""HU-DS-04 · CA-2 y CA-4 — clasificar desde una carpeta de solo lectura (simula "Archivos de programa").

Se ejecuta en un subproceso cuyo cwd es una carpeta con escritura denegada (ACL de Windows con
``icacls``), con el mismo arranque que ``desktop.main.run`` y el análisis real de la app (``FolderJob`` con
el preset de escritorio, HU-DS-01). Se comprueba:
- que el análisis termina sin errores de permisos;
- que el árbol de la carpeta de solo lectura no cambia (hash antes/después);
- que no aparece ninguna carpeta ``detections`` (ni en data_dir, ni en el cwd, ni nueva en el repo).

Requiere los modelos y las muestras (marcador ``integration``).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "models/md_v5a.0.0.pt"
SAMPLES = Path(os.getenv("LEGACY_SAMPLES_DIR", str(Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples")))
EVERYONE = "*S-1-1-0"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "win32", reason="usa ACL de Windows (icacls)"),
    pytest.mark.skipif(not (SAMPLES.is_dir() and MD.exists()), reason="requiere las muestras y los modelos"),
]

SCRIPT = r"""
import json, os, queue, sys, threading
from pathlib import Path
root, samples, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(root))
from desktop import paths
paths.ensure_dirs(); paths.configure_env(); os.chdir(paths.data_dir())   # mismo arranque que desktop.main.run
paths.migrate_legacy_model([root / "models" / "md_v5a.0.0.pt"])
from app.detector import WildlifeDetector
from desktop.job.actions import make_species_action
from desktop.job.events import FileDone
from desktop.job.folder_job import FolderJob
detector = WildlifeDetector(paths.megadetector_path())
detector.load_model()
events = queue.Queue()
FolderJob(samples, detector.models, events, threading.Event(), apply_action=make_species_action(out, "copy"), dest=out).run()
folders = {}
while not events.empty():
    e = events.get_nowait()
    if isinstance(e, FileDone):
        folders[e.file_name] = e.error_code or (e.action and e.action.destination_folder) or e.result.decision.category
print(json.dumps({"folders": folders, "tmp": [p.name for p in paths.tmp_dir().iterdir()]}))
"""


def tree_hash(folder: Path) -> str:
    h = hashlib.sha256()
    files = 0
    for p in sorted(folder.rglob("*")):
        h.update(str(p.relative_to(folder)).encode())
        if p.is_file():
            files += 1
            h.update(p.read_bytes())
    assert files > 0, "el hash no ve los archivos (¿la ACL impide listar?)"
    return h.hexdigest()


def detections_dirs(*roots: Path) -> list:
    return [str(p) for r in roots if r.exists() for p in r.rglob("detections") if p.is_dir()]


@pytest.fixture
def readonly_dir(tmp_path):
    folder = tmp_path / "Program Files" / "WWM"
    folder.mkdir(parents=True)
    (folder / "WWMClasificador.exe.txt").write_text("simula la carpeta de instalación", encoding="utf-8")
    # AD = crear archivos/carpetas, WD = modificar archivos. No usar "W": incluye SYNCHRONIZE e impide listar.
    subprocess.run(["icacls", str(folder), "/deny", f"{EVERYONE}:(OI)(CI)(AD,WD)"], check=True, capture_output=True)
    try:
        try:
            (folder / "probe.txt").write_text("x")
            pytest.skip("no se pudo denegar la escritura en la carpeta")
        except PermissionError:
            pass
        with pytest.raises(PermissionError):
            (folder / "WWMClasificador.exe.txt").write_text("modificado")
        yield folder
    finally:
        subprocess.run(["icacls", str(folder), "/remove:d", EVERYONE], capture_output=True)


def test_run_from_readonly_dir(readonly_dir, tmp_path):
    data, out = tmp_path / "wwm", tmp_path / "salida"
    before = tree_hash(readonly_dir)
    repo_detections_before = sorted(os.listdir(ROOT / "detections")) if (ROOT / "detections").exists() else None
    env = {
        **os.environ,
        "WWM_DATA_DIR": str(data),
        # cachés ya descargadas en esta máquina (evita volver a bajar el repo de YOLOv5 y SpeciesNet)
        "TORCH_HOME": str(Path.home() / ".cache" / "torch"),
        "KAGGLEHUB_CACHE": str(Path.home() / ".cache" / "kagglehub"),
    }
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(ROOT), str(SAMPLES), str(out)],
        cwd=readonly_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    result = json.loads(r.stdout.strip().splitlines()[-1])

    assert len(result["folders"]) == len(list(SAMPLES.iterdir()))
    assert not [v for v in result["folders"].values() if v.startswith(("IA_", "DESKTOP_"))], result["folders"]
    assert result["tmp"] == []  # CA-5 en el flujo real
    assert tree_hash(readonly_dir) == before  # CA-2
    assert (data / "models" / "md_v5a.0.0.pt").exists()  # los pesos quedaron en AppData
    # CA-4: sin carpetas "detections" creadas por la app
    assert detections_dirs(data, readonly_dir, out) == []
    if repo_detections_before is not None:
        assert sorted(os.listdir(ROOT / "detections")) == repo_detections_before
