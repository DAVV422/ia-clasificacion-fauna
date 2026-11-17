"""HU-IA-19 · CA-4 · GET /api/v1/catalog: con API key, la envoltura, ETag y 304."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from tests.fakes import FakeModels

CATALOG = Path(__file__).resolve().parents[2] / "catalog" / "speciesnet_catalog.json"


def client_for(tmp_path, catalog_path=CATALOG):
    settings = Settings(models_dir=tmp_path / "m", ai_api_key="k", catalog_path=catalog_path)
    return TestClient(create_app(settings, models=FakeModels()), raise_server_exceptions=False)


def test_catalog_endpoint_etag_and_auth(tmp_path):
    expected = json.loads(CATALOG.read_text(encoding="utf-8"))
    with client_for(tmp_path) as client:
        assert client.get("/api/v1/catalog").status_code == 401  # sin API key
        r = client.get("/api/v1/catalog", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success" and body["data"]["count"] == expected["count"]
        assert body["data"]["sha256"] == expected["sha256"]
        assert r.headers["ETag"] == f'"{expected["sha256"]}"'
        again = client.get("/api/v1/catalog", headers={"X-API-Key": "k", "If-None-Match": r.headers["ETag"]})
        assert again.status_code == 304 and not again.content


def test_missing_catalog_is_internal_error(tmp_path):
    with client_for(tmp_path, catalog_path=tmp_path / "no_hay.json") as client:
        r = client.get("/api/v1/catalog", headers={"X-API-Key": "k"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "IA_INTERNAL"


def test_same_shape_as_ia_fake(tmp_path):
    from fake.app import create_app as create_fake

    with client_for(tmp_path) as client:
        real = client.get("/api/v1/catalog", headers={"X-API-Key": "k"}).json()
    fake = TestClient(create_fake()).get("/api/v1/catalog", headers={"X-API-Key": "dev-key"}).json()
    assert set(real) == set(fake) and set(real["data"]) == set(fake["data"])
    assert set(real["data"]["taxa"][0]) == set(fake["data"]["taxa"][0])
