"""HU-DS-13 · Modelos: verificar, descargar con reanudación y hash, descomprimir seguro, dónde se buscan."""

import hashlib
import io
import json
import random
import threading
import zipfile
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from desktop import model_store as ms
from desktop import paths

DiskUsage = namedtuple("DiskUsage", "total used free")
BIG = 5 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Server:
    """Sirve archivos en memoria con Range. ``cut[name] = n`` corta la conexión tras n bytes en el primer pedido;
    ``corrupt[name] = k`` sirve k veces el contenido dañado."""

    def __init__(self):
        self.files, self.cut, self.corrupt, self.requests = {}, {}, {}, []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                name = self.path.lstrip("/")
                server.requests.append((name, self.headers.get("Range")))
                data = server.files.get(name)
                if data is None:
                    self.send_error(404)
                    return
                if server.corrupt.get(name):
                    server.corrupt[name] -= 1
                    data = bytes(b ^ 0xFF for b in data[:16]) + data[16:]
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                limit = server.cut.pop(name, None)
                if limit is not None:
                    self.wfile.write(body[: max(0, limit - start)])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()


def payload(n=BIG, seed=1) -> bytes:
    return random.Random(seed).randbytes(n)


def zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def manifest(server):
    md = payload()
    sn_files = {"info.json": b'{"version": "4.0.2a"}', "clasificador.pt": payload(2 * 1024 * 1024, seed=2)}
    sn = zip_bytes(sn_files)
    server.files = {"md.pt": md, "sn.zip": sn}
    artifacts = [
        ms.Artifact("megadetector", "MegaDetector", "5.0", f"{server.url}/md.pt", len(md), sha(md), "md.pt"),
        ms.Artifact(
            "speciesnet",
            "SpeciesNet",
            "4.0.2a",
            f"{server.url}/sn.zip",
            len(sn),
            sha(sn),
            "speciesnet/",
            unpack=True,
            files={k: {"size": len(v), "sha256": sha(v)} for k, v in sn_files.items()},
        ),
    ]
    return ms.Manifest(1, artifacts)


def no_sleep(seconds):
    pass


# ---------------------------------------------------------------------------
# CA-1 · Instalar todo + models.ok
# ---------------------------------------------------------------------------
def test_install_writes_models_ok(manifest, tmp_path):
    root = tmp_path / "models"
    seen = []
    result = ms.install(root, manifest, on_progress=seen.append, sleep=no_sleep)
    assert result.downloaded == ["megadetector", "speciesnet"]
    assert (root / "md.pt").stat().st_size == BIG
    assert (root / "speciesnet" / "info.json").is_file() and (root / "speciesnet" / "clasificador.pt").is_file()
    assert not list(root.glob("*.part")) and not (root / "sn.zip").exists()
    ok = json.loads((root / ms.MODELS_OK).read_text(encoding="utf-8"))
    assert set(ok["artifacts"]["speciesnet"]["files"]) == {"speciesnet/info.json", "speciesnet/clasificador.pt"}
    assert seen and seen[-1].done == seen[-1].total


# ---------------------------------------------------------------------------
# CA-2 · Reanudar con Range
# ---------------------------------------------------------------------------
def test_download_resumes_with_range(server, manifest, tmp_path):
    artifact = manifest.artifacts[0]
    server.cut["md.pt"] = int(BIG * 0.6)
    part = tmp_path / "md.pt.part"
    ms.download(artifact, part, sleep=no_sleep)
    assert part.read_bytes() == server.files["md.pt"]
    ranges = [r for name, r in server.requests if name == "md.pt"]
    assert ranges[0] is None and ranges[1].startswith("bytes=") and int(ranges[1][6:-1]) >= int(BIG * 0.5)


def test_server_without_range_restarts(server, manifest, tmp_path):
    """Si el servidor responde 200 en lugar de 206, se empieza de 0 (no se pega al final del .part)."""
    artifact = manifest.artifacts[0]
    part = tmp_path / "md.pt.part"
    part.write_bytes(b"basura previa")
    original = server.httpd.RequestHandlerClass.do_GET

    def ignore_range(handler):
        del handler.headers["Range"]
        original(handler)

    server.httpd.RequestHandlerClass.do_GET = ignore_range
    ms.download(artifact, part, sleep=no_sleep)
    assert part.read_bytes() == server.files["md.pt"]


# ---------------------------------------------------------------------------
# CA-3 · Hash incorrecto: un reintento, después DOWNLOAD_INTEGRITY_FAILED
# ---------------------------------------------------------------------------
def test_hash_mismatch_retries_then_fails(server, manifest, tmp_path):
    root = tmp_path / "models"
    server.corrupt["md.pt"] = 1  # la primera vez dañado, la segunda bien
    ms.install(root, manifest, sleep=no_sleep)
    assert (root / "md.pt").read_bytes() == server.files["md.pt"]

    root2 = tmp_path / "models2"
    server.corrupt["md.pt"] = 2
    with pytest.raises(ms.ModelStoreError) as err:
        ms.install(root2, manifest, sleep=no_sleep)
    assert err.value.code == "DOWNLOAD_INTEGRITY_FAILED"
    assert not (root2 / "md.pt").exists() and not list(root2.glob("*.part"))


# ---------------------------------------------------------------------------
# CA-4 · Sin red
# ---------------------------------------------------------------------------
def test_no_network_reports_error(tmp_path):
    waits = []
    artifact = ms.Artifact("x", "X", "1", "http://127.0.0.1:9/x.pt", 10, "0" * 64, "x.pt")
    with pytest.raises(ms.ModelStoreError) as err:
        ms.download(artifact, tmp_path / "x.part", sleep=waits.append)
    assert err.value.code == "NETWORK_UNAVAILABLE" and waits == []  # sin esperar reintentos


def test_cut_retries_with_waits_then_fails(server, manifest, tmp_path, monkeypatch):
    """Cortes a mitad: reintenta con espera; si nunca termina, NETWORK_UNAVAILABLE (y el .part queda)."""
    monkeypatch.setattr(ms, "RETRY_WAITS_S", (0.1, 0.1))
    artifact = manifest.artifacts[0]
    part = tmp_path / "md.pt.part"

    class AlwaysCut(dict):  # cada pedido se corta tras 1 MB más
        def pop(self, key, default=None):
            return (part.stat().st_size if part.exists() else 0) + 1024 * 1024 if key == "md.pt" else default

    server.cut = AlwaysCut()
    with pytest.raises(ms.ModelStoreError) as err:
        ms.download(artifact, part, sleep=no_sleep)
    assert err.value.code == "NETWORK_UNAVAILABLE" and 0 < part.stat().st_size < BIG


# ---------------------------------------------------------------------------
# CA-5 · Cancelar conserva el .part
# ---------------------------------------------------------------------------
def test_cancel_keeps_part(manifest, tmp_path):
    artifact = manifest.artifacts[0]
    part = tmp_path / "md.pt.part"
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(ms.Cancelled):
        ms.download(artifact, part, should_stop=stop, sleep=no_sleep)
    assert 0 < part.stat().st_size < BIG
    ms.download(artifact, part, sleep=no_sleep)  # la próxima vez sigue desde ahí
    assert part.stat().st_size == BIG


# ---------------------------------------------------------------------------
# CA-6 · Inicio normal: solo tamaños; si uno no coincide, se re-descarga solo ese
# ---------------------------------------------------------------------------
def test_quick_check_sizes_only(manifest, tmp_path, monkeypatch):
    root = tmp_path / "models"
    ms.install(root, manifest, sleep=no_sleep)
    monkeypatch.setattr(ms, "sha256_file", lambda p: pytest.fail("no debía leer los archivos"))
    assert ms.quick_check(root, manifest) == [] and ms.needs_install(root, manifest) == []


def test_size_mismatch_redownloads_only_that(server, manifest, tmp_path):
    root = tmp_path / "models"
    ms.install(root, manifest, sleep=no_sleep)
    (root / "speciesnet" / "clasificador.pt").write_bytes(b"truncado")
    server.requests.clear()
    result = ms.install(root, manifest, sleep=no_sleep)
    assert result.downloaded == ["speciesnet"] and result.reused == ["megadetector"]
    assert [name for name, _ in server.requests] == ["sn.zip"]


def test_existing_files_reused_without_download(server, manifest, tmp_path):
    """Sin models.ok, lo que ya está y coincide con el manifiesto (un modelo migrado, esta laptop) se reusa."""
    root = tmp_path / "models"
    (root / "speciesnet").mkdir(parents=True)
    (root / "md.pt").write_bytes(server.files["md.pt"])
    with zipfile.ZipFile(io.BytesIO(server.files["sn.zip"])) as zf:
        zf.extractall(root / "speciesnet")
    result = ms.install(root, manifest, sleep=no_sleep)
    assert result.downloaded == [] and server.requests == []
    assert (root / ms.MODELS_OK).is_file()


def test_disk_full_before_download(manifest, tmp_path):
    with pytest.raises(ms.ModelStoreError) as err:
        ms.install(tmp_path / "m", manifest, sleep=no_sleep, disk_usage=lambda p: DiskUsage(1, 1, 1024))
    assert err.value.code == "DISK_SPACE_LOW"


# ---------------------------------------------------------------------------
# CA-7 · Variante completa vs estándar
# ---------------------------------------------------------------------------
def test_models_dir_resolution_full_vs_standard(tmp_path):
    install_dir, user_dir = tmp_path / "Program Files" / "models", tmp_path / "AppData" / "models"
    assert ms.resolve_models_dir(install_dir, user_dir) == user_dir  # estándar: no hay models.ok junto al exe
    install_dir.mkdir(parents=True)
    (install_dir / ms.MODELS_OK).write_text("{}", encoding="utf-8")
    assert ms.resolve_models_dir(install_dir, user_dir) == install_dir  # completa
    assert ms.resolve_models_dir(None, user_dir) == user_dir  # desde el código (no congelado)


# ---------------------------------------------------------------------------
# CA-8 · Zip seguro
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../fuera.txt", "/absoluto.txt", "C:/windows.txt", "a/../../b.txt"])
def test_zip_slip_rejected(server, manifest, tmp_path, bad):
    sn = zip_bytes({"info.json": b"{}", bad: b"x"})
    server.files["sn.zip"] = sn
    artifact = ms.Artifact("speciesnet", "SN", "1", f"{server.url}/sn.zip", len(sn), sha(sn), "speciesnet/", True)
    with pytest.raises(ms.ModelStoreError) as err:
        ms.install(tmp_path / "models", ms.Manifest(1, [artifact]), sleep=no_sleep)
    assert err.value.code == "DOWNLOAD_INTEGRITY_FAILED"
    assert not (tmp_path / "fuera.txt").exists() and not (tmp_path / "models" / "speciesnet").exists()


# ---------------------------------------------------------------------------
# El manifiesto que va con la app
# ---------------------------------------------------------------------------
def test_shipped_manifest_matches_app_paths():
    manifest = ms.load_manifest()
    by_id = {a.id: a for a in manifest.artifacts}
    assert by_id["megadetector"].dest == paths.MEGADETECTOR_FILE
    assert by_id["speciesnet"].dest.rstrip("/") == paths.SPECIESNET_DIR and by_id["speciesnet"].unpack
    assert "info.json" in by_id["speciesnet"].files
    for a in manifest.artifacts:
        assert a.url.startswith("https://github.com/DAVV422/ia-clasificacion-fauna/releases/download/models-v1/")
        assert len(a.sha256) == 64 and a.size > 100 * 1024 * 1024


def test_this_laptop_models_match_manifest_if_present(monkeypatch):
    """Si esta PC ya tiene los modelos preparados (prepare_models.py), coinciden con el manifiesto: no se bajan."""
    monkeypatch.delenv("WWM_DATA_DIR", raising=False)  # la carpeta real de la app, no la temporal del fixture
    root = paths.models_dir()
    manifest = ms.load_manifest()
    if not (root / paths.SPECIESNET_DIR / "info.json").is_file():
        pytest.skip("esta PC no tiene los modelos preparados")
    assert all(ms.existing_matches(root, a) for a in manifest.artifacts)


def test_relative_path_helper():
    assert Path(ms.manifest_path()).name == ms.MANIFEST_NAME
