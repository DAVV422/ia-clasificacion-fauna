"""HU-IA-06: imagen de la IA (Dockerfile, .dockerignore, fetch_models) y limpieza de la API del prototipo.

Sin Docker: se revisa que lo que copia el Dockerfile coincida con la lista blanca del .dockerignore y que la API
arranque **solo** con esos archivos (si falta un módulo, la imagen se rompería en el servidor). La imagen real se
prueba a mano con Docker local (sin CI, decisión de Diego del 17/11/2026; pasos en la spec de HU-IA-06 §11).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _copy_sources() -> list:
    """Las rutas de origen de los COPY del Dockerfile (sin el --from=builder)."""
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        parts = [p for p in line.split()[1:] if not p.startswith("--")]
        sources.extend(parts[:-1])
    return sources


def _whitelist() -> list:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [line[1:] for line in lines if line.startswith("!")]


def test_no_legacy_api_files():
    """CA: la API del prototipo ya no existe (la reemplazó ``api/``, HU-IA-41)."""
    for name in ("app/main.py", "app/schemas.py", "app/utils.py", "detections/.gitignore"):
        assert not (ROOT / name).exists(), name


def test_dockerfile_copies_exactly_the_whitelist():
    assert sorted(_copy_sources()) == sorted(_whitelist())
    assert (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()[2] == "*"


def test_dockerfile_rules():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--require-hashes" in text and "requirements-api.txt" in text
    assert re.search(r'CMD \[.*"api\.main:app".*"--port", "8001", "--workers", "1"\]', text)
    assert "USER ia" in text and "HEALTHCHECK" in text and "/api/v1/health" in text
    assert "ENVIRONMENT=production" in text and "MODELS_DIR=/models" in text
    # Los modelos nunca van en la imagen.
    for src in _copy_sources():
        assert not src.endswith((".pt", ".pth")) and src.rstrip("/") not in ("models", "models-api"), src
        assert "speciesnet" not in src.lower(), src


def test_requirements_api_is_pinned_with_hashes():
    lines = (ROOT / "requirements-api.txt").read_text(encoding="utf-8")
    for pkg in ("fastapi==", "uvicorn==", "python-multipart==", "torch=="):
        assert pkg in lines, pkg
    assert "--hash=sha256:" in lines


def test_fake_has_its_own_dockerignore():
    """El build de ia-fake (``-f fake/Dockerfile``) no queda limitado por la lista blanca de la IA real."""
    ignore = (ROOT / "fake" / "Dockerfile.dockerignore").read_text(encoding="utf-8")
    for line in (ROOT / "fake" / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            src = line.split()[1]
            assert f"!{src}" in ignore or f"!{src.rstrip('/')}/" in ignore, src


def test_api_starts_with_only_the_image_files(tmp_path):
    """Lo que copia el Dockerfile alcanza para importar la API y fetch_models (sin app/, sin el resto de desktop/)."""
    image = tmp_path / "app"
    for src in _copy_sources():
        if src == "requirements-api.txt":
            continue
        origin = ROOT / src
        if origin.is_dir():
            shutil.copytree(origin, image / src, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            (image / src).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, image / src)
    env = {
        **os.environ,
        "ENVIRONMENT": "development",
        "MODELS_DIR": str(tmp_path / "models"),
        "TMP_DIR": str(tmp_path / "tmp"),
        "PYTHONPATH": str(image),
    }
    code = (
        "import sys\n"
        "import api.main, tools.fetch_models, desktop.model_store\n"
        "for mod in (api.main, tools.fetch_models, desktop.model_store):\n"
        "    assert mod.__file__.startswith(sys.argv[1]), mod.__file__\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(image)], cwd=image, env=env, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("ok")


def test_fetch_models_uses_model_store(tmp_path, monkeypatch, capsys):
    from desktop import model_store as ms
    from tools import fetch_models

    calls = {}

    def fake_install(root, manifest, on_progress):
        calls["root"] = root
        on_progress(ms.Progress(artifact_id="md", done=50, total=100))
        return ms.InstallResult(root, reused=["speciesnet"], downloaded=["md"])

    monkeypatch.setattr(ms, "install", fake_install)
    assert fetch_models.main(["--dir", str(tmp_path)]) == 0
    assert calls["root"] == tmp_path
    out = capsys.readouterr().out
    assert "md: 50 %" in out and "Modelos listos" in out


def test_fetch_models_reports_errors(tmp_path, monkeypatch, capsys):
    from desktop import model_store as ms
    from tools import fetch_models

    def failing(root, manifest, on_progress):
        raise ms.ModelStoreError("MODELS_NO_SPACE", "sin espacio")

    monkeypatch.setattr(ms, "install", failing)
    assert fetch_models.main(["--dir", str(tmp_path)]) == 1
    assert "MODELS_NO_SPACE" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["docker-compose.yml", "deploy/docker-compose.example.yml"])
def test_compose_files(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "models:/models" in text and "AI_API_KEY" in text
    if name.startswith("deploy"):
        ia = text.split("\n  ia:\n")[1].split("\n  ia-fake:")[0]
        assert "ports:" not in ia.replace('sin "ports:"', "")  # IA-31: la IA es interna
        assert "AI_SERVICE_URL: http://ia:8001" in text
