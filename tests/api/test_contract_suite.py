"""HU-IA-33 · La batería de contrato: cada caso contra la IA real y contra ia-fake, validado contra el contrato,
con la misma forma en los dos y aceptado por el cliente del back (el DetectResponse de Job).

- **IA real:** ``api.main.create_app(models=SetModels())`` (sin PyTorch; los modelos reales están en
  ``test_detect_real.py``).
- **ia-fake:** ``fake.app.create_app()``, con los nombres de archivo que disparan cada escenario (``fake/README.md``).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.routes as routes
from api.main import create_app
from api.settings import Settings
from fake.app import FakeSettings
from fake.app import create_app as create_fake
from tests.desktop.fixtures import SetModels, frame, write_f_set_sint
from wwm_pipeline.contract import ApiEnvelope, DetectResult
from wwm_pipeline.pipeline import classify_file as real_classify

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]
KEY = "k"

# escenario → (archivo para la IA real, archivo para ia-fake)
SUCCESS = {
    "jaguar_video": ("especie_a.mp4", "CAM01_20260315_042210.mp4"),
    "animal_image": ("especie_b.jpg", "foto_image.jpg"),
    "blank": ("vacio_f.mp4", "x_blank.mp4"),
    "person": ("persona_h.mp4", "x_person.mp4"),
    "genus": ("genero_c.mp4", "x_genus.mp4"),
    "truncated": ("especie_largo.mp4", "x_toolong.mp4"),
}


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sint")
    write_f_set_sint(folder)
    writer = cv2.VideoWriter(str(folder / "especie_largo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(200):  # 20 s: la web analiza hasta el 15 (F1-D10)
        writer.write(frame(i, 210))
    writer.release()
    (folder / "notas.txt").write_bytes(b"no soy un video")
    return folder


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real")
    app = create_app(Settings(models_dir=tmp / "m", tmp_dir=tmp / "t", ai_api_key=KEY), models=SetModels())
    with TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": KEY}) as client:
        yield client


@pytest.fixture(scope="module")
def fake():
    app = create_fake(FakeSettings(api_key=KEY, delay_seconds=0, timeout_seconds=0.05, slow_seconds=0))
    with TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": KEY}) as client:
        yield client


def post(client, name, content, headers=None):
    return client.post("/api/v1/detect", files={"file": (name, content)}, data={"file_name": name}, headers=headers)


def real_post(real, media, name, **kw):
    return post(real, name, (media / name).read_bytes(), **kw)


def fake_post(fake, name, **kw):
    return post(fake, name, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, **kw)


def keys_tree(obj, prefix=""):
    """Las rutas de claves, sin entrar en las listas (su contenido se compara aparte, si las dos tienen elementos)."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            out |= keys_tree(v, path)
    return out


def list_item_keys(obj, prefix=""):
    """``{ruta_de_la_lista: claves del primer elemento}`` para las listas no vacías de diccionarios."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, list) and v and isinstance(v[0], dict):
                out[path] = keys_tree(v[0])
            out.update(list_item_keys(v, path))
    return out


# ---------------------------------------------------------------------------
# CA-1 · Los casos exitosos validan el contrato en las dos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SUCCESS))
def test_success_cases_validate_contract(real, fake, media, scenario):
    real_name, fake_name = SUCCESS[scenario]
    for r in (real_post(real, media, real_name), fake_post(fake, fake_name)):
        assert r.status_code == 200, r.text[:300]
        data = ApiEnvelope[DetectResult].model_validate(r.json()).data
        if scenario == "truncated":
            assert data.analysis.truncated_at_seconds == 15.0
        if scenario == "blank":
            assert data.evidence is None


# ---------------------------------------------------------------------------
# CA-2 · La misma forma (claves) en la IA real y en ia-fake
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SUCCESS))
def test_shape_parity_with_fake(real, fake, media, scenario):
    real_name, fake_name = SUCCESS[scenario]
    r = real_post(real, media, real_name).json()
    f = fake_post(fake, fake_name).json()
    assert keys_tree(r) == keys_tree(f), keys_tree(r) ^ keys_tree(f)
    ri, fi = list_item_keys(r), list_item_keys(f)
    for path in set(ri) & set(fi):  # las listas con elementos en las dos
        assert ri[path] == fi[path], (path, ri[path] ^ fi[path])


# ---------------------------------------------------------------------------
# CA-1 · CA-2 · Los errores: mismo HTTP y mismo código en las dos
# ---------------------------------------------------------------------------
def _real_busy(real, media):
    worker = real.app.state.worker
    taken = 0
    while worker.try_enter():
        taken += 1
    try:
        return real_post(real, media, "especie_a.mp4")
    finally:
        for _ in range(taken):
            worker.leave()


def _real_timeout(real, media, monkeypatch):
    worker = real.app.state.worker
    old = worker.timeout_seconds
    worker.timeout_seconds = 0.05

    def slow(path, name, config, models, should_stop=None, **kw):
        import time

        while not (should_stop and should_stop()):
            time.sleep(0.01)
        from wwm_pipeline.errors import PipelineCancelled

        raise PipelineCancelled(name)

    monkeypatch.setattr(routes, "classify_file", slow)
    try:
        return real_post(real, media, "especie_a.mp4")
    finally:
        worker.timeout_seconds = old
        monkeypatch.setattr(routes, "classify_file", real_classify)


def _real_not_ready(tmp_path, monkeypatch, media):
    import api.main as main

    monkeypatch.setattr(main, "_load_models", lambda app: None)
    app = create_app(Settings(models_dir=tmp_path / "m", tmp_dir=tmp_path / "t", ai_api_key=KEY))
    with TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": KEY}) as client:
        return real_post(client, media, "especie_a.mp4")


@pytest.mark.parametrize(
    "case,http,code",
    [
        ("decode", 422, "IA_DECODE_FAILED"),
        ("format", 415, "IA_INVALID_FORMAT"),
        ("unauthorized", 401, "IA_UNAUTHORIZED"),
        ("busy", 503, "IA_BUSY"),
        ("timeout", 504, "IA_PROCESSING_TIMEOUT"),
        ("not_ready", 503, "IA_MODEL_NOT_READY"),
    ],
)
def test_error_cases_same_code_in_both(real, fake, media, tmp_path, monkeypatch, case, http, code):
    if case == "decode":
        r, f = real_post(real, media, "danado_j.mp4"), fake_post(fake, "x_error_decode.mp4")
    elif case == "format":
        r, f = real_post(real, media, "notas.txt"), fake_post(fake, "x_badformat.mp4")
    elif case == "unauthorized":
        r = real_post(real, media, "especie_a.mp4", headers={"X-API-Key": "mala"})
        f = fake_post(fake, "a.mp4", headers={"X-API-Key": "mala"})
    elif case == "busy":
        r, f = _real_busy(real, media), fake_post(fake, "x_busy.mp4")
    elif case == "timeout":
        r, f = _real_timeout(real, media, monkeypatch), fake_post(fake, "x_timeout.mp4")
    else:
        r, f = _real_not_ready(tmp_path, monkeypatch, media), fake_post(fake, "x_notready.mp4")
    for resp in (r, f):
        body = resp.json()
        assert resp.status_code == http, (case, resp.status_code, body)
        env = ApiEnvelope[dict].model_validate(body)
        assert env.status == "error" and env.error is not None and env.error.code == code
    assert set(r.json()) == set(f.json()) and set(r.json()["error"]) == set(f.json()["error"])


def test_too_large_real(media, tmp_path):
    """413 solo en la IA real (ia-fake no tiene ese escenario)."""
    app = create_app(
        Settings(models_dir=tmp_path / "m", tmp_dir=tmp_path / "t", ai_api_key=KEY, max_file_mb=0.001),
        models=SetModels(),
    )
    with TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": KEY}) as client:
        r = real_post(client, media, "especie_a.mp4")
    assert r.status_code == 413 and r.json()["error"]["code"] == "IA_FILE_TOO_LARGE"


# ---------------------------------------------------------------------------
# CA-3 · El cliente del back (el DetectResponse de Job) acepta las respuestas de la IA real
# ---------------------------------------------------------------------------
JOB_REPO = ROOT.parent / "BACKWWF"
JOB_REF = "origin/feat/spike-01-ai-walking-skeleton:app/integrations/ai_contract.py"


def load_job_contract():
    """El back es privado: su código no se copia a este repo (público). Se lee del repo local de Job, si está."""
    try:
        source = subprocess.run(
            ["git", "-C", str(JOB_REPO), "show", JOB_REF], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    spec = importlib.util.spec_from_loader("job_ai_contract", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "job_ai_contract.py", "exec"), module.__dict__)  # noqa: S102 - código del equipo, leído de su repo
    return module


@pytest.mark.parametrize("scenario", list(SUCCESS))
def test_job_detect_response_accepts_real(real, media, scenario):
    job = load_job_contract()
    if job is None:
        pytest.skip("no está el repo del back (BACKWWF) con la rama del spike")
    r = real_post(real, media, SUCCESS[scenario][0])
    job.DetectResponse.model_validate(r.json()["data"])


# ---------------------------------------------------------------------------
# CA-5 · Rápidos y sin PyTorch
# ---------------------------------------------------------------------------
def test_unit_suite_does_not_import_torch():
    code = "import api.main, fake.app, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
