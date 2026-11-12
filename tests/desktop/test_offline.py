"""HU-DS-11 · Sin conexión real: ni torch.hub ni Kaggle; 0 conexiones al cargar y clasificar."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.model_paths import HAVE_MODELS, MD, SPECIESNET
from wwm_pipeline.errors import PipelineError
from wwm_pipeline.models import ModelRegistry, default_yolov5_dir

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CA-2 · Nada de torch.hub remoto ni Kaggle en el código de la app
# ---------------------------------------------------------------------------
def test_no_hub_or_kaggle_in_code():
    for folder in ("desktop", "wwm_pipeline"):
        for py in (ROOT / folder).rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "force_reload" not in text, py
            assert "kaggle:" not in text, py
            assert 'torch.hub.load("ultralytics' not in text, py
            for call in re.findall(r"torch\.hub\.load\((.*)\)", text):
                assert 'source="local"' in call, (py, call)


def test_vendored_yolov5_is_complete():
    hub = default_yolov5_dir()
    for rel in ("hubconf.py", "export.py", "requirements.txt", "LICENSE", "models/common.py", "utils/general.py"):
        assert (hub / rel).is_file(), rel


def test_missing_yolov5_code_fails_clearly(tmp_path):
    registry = ModelRegistry(tmp_path / "md.pt", tmp_path / "sn", device="cpu", yolov5_dir=tmp_path / "no_hay")
    with pytest.raises(PipelineError) as err:
        registry._load_megadetector()
    assert err.value.code == "MODEL_LOAD_FAILED"


# ---------------------------------------------------------------------------
# CA-1 · CA-3 · Cargar y clasificar con la red bloqueada y las cachés del perfil vacías
# ---------------------------------------------------------------------------
CHILD = r"""
import json, os, socket, sys
attempts = []
def connect(self, address):
    attempts.append(repr(address)); raise OSError("red bloqueada (test)")
def getaddrinfo(host, *a, **k):
    attempts.append("dns:" + str(host)); raise socket.gaierror("red bloqueada (test)")
socket.socket.connect = connect
socket.socket.connect_ex = lambda self, address: connect(self, address)
socket.getaddrinfo = getaddrinfo
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from desktop import paths
paths.configure_env()
from tests.desktop.fixtures import write_f_set_sint
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.models import ModelRegistry
from wwm_pipeline.pipeline import classify_file
src = Path(sys.argv[4]); write_f_set_sint(src)
models = ModelRegistry(Path(sys.argv[2]), Path(sys.argv[3]), device="cpu").load()
out = {}
for name in ("especie_a.mp4", "vacio_g.jpg", "persona_h.mp4"):
    out[name] = classify_file(src / name, name, PipelineConfig.desktop(), models).decision.category
print(json.dumps({"attempts": attempts, "categories": out}))
"""


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_MODELS, reason="requiere MegaDetector y la carpeta local de SpeciesNet")
def test_no_network_during_classification(tmp_path):
    caches = {name: tmp_path / name for name in ("torch", "kagglehub", "hf", "xdg", "yolo")}
    for path in caches.values():
        path.mkdir()
    env = dict(
        os.environ,
        WWM_DATA_DIR=str(tmp_path / "wwm"),
        TORCH_HOME=str(caches["torch"]),
        KAGGLEHUB_CACHE=str(caches["kagglehub"]),
        HF_HOME=str(caches["hf"]),
        XDG_CACHE_HOME=str(caches["xdg"]),
        YOLO_CONFIG_DIR=str(caches["yolo"]),
        PYTHONIOENCODING="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(ROOT), str(MD), str(SPECIESNET), str(tmp_path / "SD")],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["attempts"] == []  # CA-1: 0 intentos de conexión
    assert set(result["categories"]) == {"especie_a.mp4", "vacio_g.jpg", "persona_h.mp4"}
    for name in ("torch", "kagglehub", "hf"):  # CA-3: no se bajó nada a las cachés del perfil
        assert not any(caches[name].rglob("*.pt")), name
        assert not any(p for p in caches[name].rglob("*") if p.is_file() and p.stat().st_size > 1_000_000), name
