"""HU-IA-41 · CA-1 · CA-2 · CA-3 con los modelos reales: lo que fallaba en el análisis del 17/11/2026."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", ROOT.parent / "docs-interno" / "estacion_1"))
MODELS = Path(os.getenv("MODELS_DIR") or Path(os.getenv("LOCALAPPDATA", "")) / "WWM" / "models")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ESTACION_1.is_dir() and (MODELS / "speciesnet-4.0.2a" / "info.json").is_file()),
        reason="requiere estacion_1 y los modelos (tools/prepare_models.py)",
    ),
]

CASES = {
    "d59aee6e-469a-4372-a0e2-2acf205cd1c3.jpg": ("Panthera onca", None),  # antes: 500 (IA-03)
    "WhatsApp Video 2026-05-30 at 9.49.38 AM (1).mp4": ("Panthera onca", None),  # 10 s · antes: "nada"
    "WhatsApp Video 2026-05-30 at 9.49.37 AM (2).mp4": (None, 15.0),  # 52,1 s → se analiza hasta el segundo 15
}


@pytest.fixture(scope="module")
def client():
    import time

    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.settings import Settings

    app = create_app(Settings(models_dir=MODELS, ai_api_key="k"))
    with TestClient(app, headers={"X-API-Key": "k"}) as c:
        deadline = time.monotonic() + 300
        while app.state.models_status == "loading":
            assert time.monotonic() < deadline, "los modelos no cargaron"
            time.sleep(0.5)
        assert app.state.models_status == "ready"
        yield c


@pytest.mark.parametrize("name", list(CASES))
def test_detect_real_image_and_video(client, name):
    from wwm_pipeline.contract import ApiEnvelope, DetectResult

    taxon, truncated = CASES[name]
    with open(ESTACION_1 / name, "rb") as f:
        r = client.post("/api/v1/detect", files={"file": (name, f)}, data={"file_name": name})
    assert r.status_code == 200, r.text[:500]
    data = ApiEnvelope[DetectResult].model_validate(r.json()).data
    assert data.result.category.value == "animal"
    if taxon:
        assert data.result.taxonomy.scientific_name == taxon
    assert data.analysis.truncated_at_seconds == truncated
    assert data.evidence is not None and data.evidence.image_base64
    assert data.model.classifier == "speciesnet-4.0.2a"
