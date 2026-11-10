"""Etiquetas del set de validación (``eval/labels.csv``) y su partición calib/test.

Columnas: file_name, sha256, media_type, label_category, label_scientific_name,
label_rank, split, notes.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LABELS_CSV = Path(__file__).resolve().parent / "labels.csv"
COLUMNS = [
    "file_name",
    "sha256",
    "media_type",
    "label_category",
    "label_scientific_name",
    "label_rank",
    "split",
    "notes",
]
CATEGORIES = {"animal", "person", "vehicle", "blank"}
RANKS = {"species", "genus", "family", "order", "class", ""}
MEDIA = {"image", "video"}
CALIB_PERCENT = 30


@dataclass
class Label:
    file_name: str
    sha256: str
    media_type: str
    label_category: str
    label_scientific_name: str
    label_rank: str
    split: str
    notes: str = ""


class LabelError(ValueError):
    pass


def split_for(sha256: str) -> str:
    """Partición estable: depende solo del sha256, no del orden ni de los otros archivos."""
    bucket = int(hashlib.sha256(f"wwm-split:{sha256}".encode()).hexdigest()[:8], 16) % 100
    return "calib" if bucket < CALIB_PERCENT else "test"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_row(row: Dict[str, str], line: int) -> Label:
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise LabelError(f"línea {line}: faltan columnas {missing}")
    lab = Label(**{c: (row.get(c) or "").strip() for c in COLUMNS})
    if lab.media_type not in MEDIA:
        raise LabelError(f"línea {line}: media_type inválido {lab.media_type!r}")
    if lab.label_category not in CATEGORIES:
        raise LabelError(f"línea {line}: label_category inválido {lab.label_category!r}")
    if lab.label_rank not in RANKS:
        raise LabelError(f"línea {line}: label_rank inválido {lab.label_rank!r}")
    if lab.label_category == "animal" and not (lab.label_scientific_name and lab.label_rank):
        if lab.label_rank or lab.label_scientific_name:
            raise LabelError(f"línea {line}: con animal, nombre y rank van juntos")
    if lab.label_category != "animal" and (lab.label_scientific_name or lab.label_rank):
        raise LabelError(f"línea {line}: solo los animales llevan nombre científico")
    if len(lab.sha256) != 64:
        raise LabelError(f"línea {line}: sha256 inválido")
    if lab.split not in {"calib", "test"}:
        raise LabelError(f"línea {line}: split debe ser calib o test")
    if lab.split != split_for(lab.sha256):
        raise LabelError(f"línea {line}: split no coincide con split_for(sha256); no se edita a mano")
    return lab


def load_labels(path: Path = LABELS_CSV) -> List[Label]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [validate_row(r, i) for i, r in enumerate(rows, start=2)]
    seen: Dict[str, int] = {}
    for i, lab in enumerate(labels, start=2):
        if lab.sha256 in seen:
            raise LabelError(f"línea {i}: archivo duplicado (mismo sha256 que la línea {seen[lab.sha256]})")
        seen[lab.sha256] = i
    return labels


def write_labels(labels: List[Label], path: Path = LABELS_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for lab in labels:
            writer.writerow(lab.__dict__)


def by_split(labels: List[Label], split: Optional[str]) -> List[Label]:
    return [l for l in labels if split in (None, "all") or l.split == split]
