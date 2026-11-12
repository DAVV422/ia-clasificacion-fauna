"""Prepara los modelos para el *release* ``models-v1`` y lo verifica después de subirlo (HU-DS-13, F4-D2).

    python tools/publish_models.py            # arma dist/models-release/ y desktop/models/model_manifest.json
    python tools/publish_models.py --verify   # después de subirlos: tamaños y Range contra las URLs
    python tools/publish_models.py --verify --full   # además descarga todo y compara el sha256 (~510 MB)

``dist/models-release/`` queda con:
  - ``md_v5a.0.0.pt``          (hardlink al de ``models/``: no ocupa disco extra)
  - ``speciesnet-4.0.2a.zip``  (la carpeta que arma ``prepare_models.py``, con su ``info.json`` local)
  - ``SHA256SUMS.txt``

La subida la hace Diego desde la web de GitHub: ``docs-interno/app-escritorio/fase4/GUIA_SUBIR_MODELOS.md``.
El zip es determinístico (orden fijo, fecha fija), así que volver a correrlo da el mismo ``sha256``.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_models import MD_NAME, SPECIESNET_NAME, kaggle_speciesnet_dir, prepare, sha256_of  # noqa: E402

RELEASE_TAG = "models-v1"
BASE_URL = f"https://github.com/DAVV422/ia-clasificacion-fauna/releases/download/{RELEASE_TAG}"
OUT = ROOT / "dist" / "models-release"
MANIFEST = ROOT / "desktop" / "models" / "model_manifest.json"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def _deterministic_zip(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(folder).as_posix(), date_time=ZIP_DATE)
            # Los .pt casi no se comprimen: se guardan tal cual (más rápido de descomprimir).
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, "rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)


def build(md: Path, speciesnet_source: Path) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT, prefix="_staging_") as tmp:
        staging = Path(tmp)
        prepare(staging, md, speciesnet_source)  # hardlinks: no ocupa disco extra
        md_out = OUT / MD_NAME
        md_out.unlink(missing_ok=True)
        try:
            os.link(staging / MD_NAME, md_out)
        except OSError:
            shutil.copyfile(staging / MD_NAME, md_out)
        sn_folder = staging / SPECIESNET_NAME
        files = {
            p.relative_to(sn_folder).as_posix(): {"size": p.stat().st_size, "sha256": sha256_of(p)}
            for p in sorted(sn_folder.rglob("*"))
            if p.is_file()
        }
        zip_path = OUT / f"{SPECIESNET_NAME}.zip"
        _deterministic_zip(sn_folder, zip_path)

    info = json.loads(json.dumps(files))
    artifacts = [
        {
            "id": "megadetector",
            "name": "MegaDetector v5a",
            "version": "5.0",
            "url": f"{BASE_URL}/{MD_NAME}",
            "size": md_out.stat().st_size,
            "sha256": sha256_of(md_out),
            "dest": MD_NAME,
        },
        {
            "id": "speciesnet",
            "name": "SpeciesNet 4.0.2a",
            "version": "4.0.2a",
            "url": f"{BASE_URL}/{zip_path.name}",
            "size": zip_path.stat().st_size,
            "sha256": sha256_of(zip_path),
            "dest": f"{SPECIESNET_NAME}/",
            "unpack": True,
            "files": info,
        },
    ]
    manifest = {"manifest_version": 1, "release": RELEASE_TAG, "artifacts": artifacts}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    sums = "".join(f"{a['sha256']}  {Path(a['url']).name}  ({a['size']} bytes)\n" for a in artifacts)
    (OUT / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return manifest


def verify(full: bool = False) -> int:
    import requests

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = 0
    for a in manifest["artifacts"]:
        name = Path(a["url"]).name
        try:
            head = requests.head(a["url"], allow_redirects=True, timeout=30)
            size = int(head.headers.get("Content-Length", -1))
            rng = requests.get(a["url"], headers={"Range": "bytes=0-1023"}, timeout=30, stream=True)
            range_ok = rng.status_code == 206
            rng.close()
        except requests.RequestException as exc:
            print(f"✗ {name}: no se pudo consultar ({exc})")
            problems += 1
            continue
        ok = head.status_code == 200 and size == a["size"] and range_ok
        print(
            f"{'✓' if ok else '✗'} {name}: HTTP {head.status_code}, {size} bytes "
            f"(esperado {a['size']}), Range {'sí' if range_ok else 'NO'}"
        )
        problems += 0 if ok else 1
        if ok and full:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / name
                with requests.get(a["url"], stream=True, timeout=(10, 120)) as resp, open(target, "wb") as fh:
                    for chunk in resp.iter_content(1024 * 1024):
                        fh.write(chunk)
                match = sha256_of(target) == a["sha256"]
                print(f"  {'✓' if match else '✗'} sha256 {'coincide' if match else 'NO coincide'}")
                problems += 0 if match else 1
    print("Todo bien." if not problems else f"{problems} problema(s): ver la guía, paso 4.")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--md", type=Path, default=ROOT / "models" / MD_NAME)
    parser.add_argument("--speciesnet", type=Path, default=None)
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify(args.full))
    manifest = build(args.md, args.speciesnet or kaggle_speciesnet_dir())
    print(json.dumps({"out": str(OUT), "manifest": str(MANIFEST), "artifacts": manifest["artifacts"]}, indent=1))


if __name__ == "__main__":
    main()
