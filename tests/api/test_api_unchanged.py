"""HU-IA-40 · CA-2 — la API responde exactamente igual que antes de la reestructura.

Compara contra ``eval/legacy/api_before.json`` (capturado con el código anterior), normalizando
los UUID de los recortes. Requiere los modelos y las muestras.
"""

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "eval/legacy/api_before.json"
SAMPLES = Path(os.getenv("LEGACY_SAMPLES_DIR", str(Path(os.getenv("EVAL_DATA_DIR", "")) / "legacy_samples")))

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (BEFORE.exists() and SAMPLES.is_dir() and (ROOT / "models/md_v5a.0.0.pt").exists()),
                                 reason="requiere api_before.json, las muestras y los modelos")]


def normalize(body) -> dict:
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", json.dumps(body))
    text = re.sub(r"_[0-9a-f]{8}/", "_<run>/", text)
    return json.loads(text)


def test_api_response_unchanged():
    from fastapi.testclient import TestClient

    from app.main import app

    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    with TestClient(app) as client:
        for name, expected in before.items():
            with open(SAMPLES / name, "rb") as f:
                r = client.post("/api/v1/detect", files={"file": (name, f, "application/octet-stream")})
            assert r.status_code == expected["status"], name
            assert normalize(r.json()) == expected["body"], name
