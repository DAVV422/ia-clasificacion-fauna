"""HU-IA-04 · CA-4 con los modelos reales: 20 peticiones seguidas, la RAM no crece más de un 10 %."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", ROOT.parent / "docs-interno" / "estacion_1"))
MODELS = Path(os.getenv("MODELS_DIR") or Path(os.getenv("LOCALAPPDATA", "")) / "WWM" / "models")
IMAGE = ESTACION_1 / "d59aee6e-469a-4372-a0e2-2acf205cd1c3.jpg"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (IMAGE.is_file() and (MODELS / "speciesnet-4.0.2a" / "info.json").is_file()),
        reason="requiere estacion_1 y los modelos",
    ),
]


def test_ram_stable_over_20_requests():
    import time

    import psutil
    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.settings import Settings

    app = create_app(Settings(models_dir=MODELS, ai_api_key="k"))
    proc = psutil.Process()
    data = IMAGE.read_bytes()
    with TestClient(app, headers={"X-API-Key": "k"}) as client:
        deadline = time.monotonic() + 300
        while app.state.models_status == "loading":
            assert time.monotonic() < deadline
            time.sleep(0.5)

        def once():
            r = client.post("/api/v1/detect", files={"file": (IMAGE.name, data)}, data={"file_name": IMAGE.name})
            assert r.status_code == 200, r.text[:300]

        once()  # la primera calienta (asigna los buffers de PyTorch)
        baseline = proc.memory_info().rss
        for _ in range(20):
            once()
        after = proc.memory_info().rss
    growth = (after - baseline) / baseline
    print(f"RSS: {baseline / 2**20:.0f} MB → {after / 2**20:.0f} MB ({growth:+.1%})")
    assert growth < 0.10
