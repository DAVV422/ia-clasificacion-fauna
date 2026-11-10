"""Ayuda para etiquetar: recorre EVAL_DATA_DIR, abre cada archivo y pregunta la etiqueta.

Uso:
    set EVAL_DATA_DIR=D:\\wwm-eval\\videos
    python -m eval.label_helper

- Saltea los archivos que ya están en labels.csv (por sha256), así se puede cortar y seguir.
- Abre el archivo con la aplicación predeterminada de Windows (os.startfile).
- Respuestas: "b" = vacío, "p" = persona, "v" = vehículo, "a" = animal sin identificar,
  o el nombre científico (por ejemplo "Panthera onca" o "Dasyprocta"), que se valida contra
  las etiquetas de SpeciesNet. Enter en blanco = saltar; "q" = salir.
Ver eval/LABELING.md para los criterios.
"""

import os
import sys
from pathlib import Path

from eval.labels import LABELS_CSV, Label, file_sha256, load_labels, split_for, write_labels
from eval.taxonomy_index import TaxonomyIndex

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".3gp",
    ".mts",
    ".m2ts",
    ".webm",
}
VIDEO = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".wmv", ".3gp", ".mts", ".m2ts", ".webm"}
SHORTCUTS = {"b": "blank", "p": "person", "v": "vehicle"}


def main() -> int:
    data_dir = Path(os.getenv("EVAL_DATA_DIR", ""))
    if not data_dir.is_dir():
        print("Define EVAL_DATA_DIR con la carpeta de los archivos de prueba.", file=sys.stderr)
        return 2
    labels = load_labels() if LABELS_CSV.exists() and LABELS_CSV.read_text(encoding="utf-8").strip() else []
    known = {l.sha256 for l in labels}
    index = TaxonomyIndex()
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    for n, path in enumerate(files, start=1):
        sha = file_sha256(path)
        if sha in known:
            continue
        print(f"\n[{n}/{len(files)}] {path.name}")
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
        while True:
            answer = input("Etiqueta (b/p/v/a/nombre científico, Enter=saltar, q=salir): ").strip()
            if answer.lower() == "q":
                write_labels(labels)
                return 0
            if not answer:
                break
            notes = input("Notas (opcional): ").strip()
            media = "video" if path.suffix.lower() in VIDEO else "image"
            if answer.lower() in SHORTCUTS:
                lab = Label(path.name, sha, media, SHORTCUTS[answer.lower()], "", "", split_for(sha), notes)
            elif answer.lower() == "a":
                lab = Label(path.name, sha, media, "animal", "", "", split_for(sha), notes or "animal sin identificar")
            else:
                taxon = index.lookup_name(answer)
                if taxon is None:
                    print(
                        "  No está en las etiquetas de SpeciesNet; usa el nombre científico (p. ej. 'Panthera onca')."
                    )
                    continue
                lab = Label(
                    path.name, sha, media, "animal", taxon.scientific_name, taxon.rank.value, split_for(sha), notes
                )
            labels.append(lab)
            known.add(sha)
            write_labels(labels)  # se guarda después de cada archivo
            break
    write_labels(labels)
    print(f"\nListo: {len(labels)} archivos etiquetados en {LABELS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
