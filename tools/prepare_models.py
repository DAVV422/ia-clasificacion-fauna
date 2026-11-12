"""Arma la carpeta de modelos que usa la app sin conexión (HU-DS-11).

    models/
      md_v5a.0.0.pt                      MegaDetector v5a
      speciesnet-4.0.2a/
        info.json                        reescrito: "detector" apunta a ../md_v5a.0.0.pt (no a una URL)
        always_crop_…_epoch_00148.pt     el clasificador
        always_crop_…labels.….txt        las etiquetas
        taxonomy_release.….txt
        geofence_release.….json

Fuentes por defecto: ``models/md_v5a.0.0.pt`` del repo y la caché de Kaggle de esta PC
(``~/.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1``). **No** se copia la segunda copia de
MegaDetector que trae el paquete de Kaggle (268 MB).

Si ``info.json`` tuviera una URL en "detector", SpeciesNet bajaría MegaDetector al inicializarse: por eso se
reescribe con la ruta local.

Por defecto usa *hardlinks* (no ocupa disco extra si la fuente está en la misma unidad); con ``--copy`` copia.
Lo usan HU-DS-13 (``publish_models.py``) y HU-DS-12 (la variante completa).

    python tools/prepare_models.py                       # → %LOCALAPPDATA%\\WWM\\models
    python tools/prepare_models.py --out dist/models --copy
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MD_NAME = "md_v5a.0.0.pt"
SPECIESNET_NAME = "speciesnet-4.0.2a"
SPECIESNET_VERSION = "4.0.2a"
KAGGLE_SUBDIR = Path("models/google/speciesnet/pyTorch/v4.0.2a/1")
SPECIESNET_KEYS = ("classifier", "classifier_labels", "taxonomy", "geofence")


def kaggle_speciesnet_dir() -> Path:
    base = Path(os.getenv("KAGGLEHUB_CACHE") or Path.home() / ".cache" / "kagglehub")
    return base / KAGGLE_SUBDIR


def default_out() -> Path:
    from desktop import paths

    return paths.models_dir()


def _place(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "ya estaba"
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass  # otra unidad o sin permiso: se copia
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "copia"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    out: Path, md_source: Path, speciesnet_source: Path, copy: bool = False, hashes: bool = False
) -> Dict[str, dict]:
    if not md_source.is_file():
        raise SystemExit(f"No está MegaDetector en {md_source}")
    info_path = speciesnet_source / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"No está SpeciesNet en {speciesnet_source} (falta info.json)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("version") != SPECIESNET_VERSION:
        raise SystemExit(f"Se esperaba SpeciesNet {SPECIESNET_VERSION}, la fuente es {info.get('version')}")

    report: Dict[str, dict] = {}
    md_dst = out / MD_NAME
    report[MD_NAME] = {"how": _place(md_source, md_dst, copy)}
    sn_dir = out / SPECIESNET_NAME
    for key in SPECIESNET_KEYS:
        name = info[key]
        report[f"{SPECIESNET_NAME}/{name}"] = {"how": _place(speciesnet_source / name, sn_dir / name, copy)}
    local_info = dict(info, detector=f"../{MD_NAME}")
    (sn_dir / "info.json").write_text(json.dumps(local_info, indent=4) + "\n", encoding="utf-8")
    report[f"{SPECIESNET_NAME}/info.json"] = {"how": "reescrito"}
    for rel, entry in report.items():
        path = out / rel
        entry["size"] = path.stat().st_size
        if hashes:
            entry["sha256"] = sha256_of(path)
    return report


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="carpeta destino (por defecto, la de la app)")
    parser.add_argument("--md", type=Path, default=ROOT / "models" / MD_NAME)
    parser.add_argument("--speciesnet", type=Path, default=None, help="carpeta de SpeciesNet con info.json")
    parser.add_argument("--copy", action="store_true", help="copiar en lugar de crear hardlinks")
    parser.add_argument("--hashes", action="store_true", help="calcular el sha256 de cada archivo")
    args = parser.parse_args(argv)
    out = args.out or default_out()
    report = prepare(out, args.md, args.speciesnet or kaggle_speciesnet_dir(), args.copy, args.hashes)
    print(json.dumps({"out": str(out), "files": report}, indent=1))


if __name__ == "__main__":
    main()
