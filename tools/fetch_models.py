"""Deja los modelos de la IA en ``MODELS_DIR`` (HU-IA-06, E2-D7): MegaDetector y SpeciesNet del *release* ``models-v1``.

Usa ``desktop/model_store.py`` (HU-DS-13): descarga con reanudación, verifica el ``sha256`` contra
``desktop/models/model_manifest.json`` y reusa lo que ya esté bien. Se corre **una vez** por servidor (el contenedor
de la API nunca descarga nada al arrancar):

    docker compose run --rm ia python tools/fetch_models.py          # en el volumen "models" (/models)
    python tools/fetch_models.py --dir C:\\ruta\\a\\models               # fuera de Docker
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from desktop import model_store as ms

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path(os.getenv("MODELS_DIR", "/models")))
    args = parser.parse_args(argv)
    manifest = ms.load_manifest()
    last = {}

    def progress(p):
        pct = int(p.done * 100 / p.total) if p.total else 0
        if last.get(p.artifact_id) != pct and pct % 10 == 0:
            last[p.artifact_id] = pct
            print(f"  {p.artifact_id}: {pct} %", flush=True)

    try:
        result = ms.install(args.dir, manifest, on_progress=progress)
    except ms.ModelStoreError as exc:
        print(f"ERROR {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    print(
        f"Modelos listos en {result.root} (descargados: {result.downloaded or 'ninguno'}; ya estaban: {result.reused})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
