"""HU-IA-10 · CA-4, CA-8 — la copia congelada reproduce el prototipo (requiere modelos y muestras)."""

import json
import os
from pathlib import Path

import pytest

from eval.legacy.make_expected import OUT as EXPECTED, describe, samples_dir

pytestmark = pytest.mark.integration


def _ready() -> bool:
    return EXPECTED.exists() and samples_dir().is_dir()


@pytest.mark.skipif(not _ready(), reason="requiere expected.json, las muestras (LEGACY_SAMPLES_DIR) y los modelos")
def test_legacy_matches_expected_and_is_reproducible():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["files"]
    for item in expected:
        path = samples_dir() / item["file_name"]
        got = describe(path)
        assert got["sha256"] == item["sha256"], f"la muestra {item['file_name']} cambió"
        assert got["frames"] == item["frames"]
        assert got["decision"]["label"] == item["decision"]["label"]
        assert got["decision"]["confidence"] == pytest.approx(item["decision"]["confidence"], abs=1e-4)


def test_legacy_is_isolated():
    """CA-4: el código viejo no se importa desde wwm_pipeline ni desde desktop."""
    root = Path(__file__).resolve().parents[2]
    for folder in ("wwm_pipeline", "desktop"):
        base = root / folder
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            assert "eval.legacy" not in py.read_text(encoding="utf-8"), py
