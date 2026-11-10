"""Métricas de precisión (HU-IA-10 CA-5, CA-6).

Reglas por nivel (CA-6):
- El acierto de **especie** se mide solo en los archivos etiquetados a nivel especie.
  Si la predicción se quedó en un nivel superior y ese nivel es correcto, cuenta como
  "no llegó a especie" (se informa aparte), no como error.
- El acierto de **género** se mide en los etiquetados a nivel especie o género.
- El acierto de **familia** se mide en los etiquetados a nivel especie, género o familia.
- **Falso negativo:** etiqueta animal y predicción no animal.
- **Falso positivo:** etiqueta blank y predicción animal.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple

from eval.labels import Label
from eval.prediction import Prediction

RANK_DEPTH = {"class": 0, "order": 1, "family": 2, "genus": 3, "species": 4}


@dataclass
class TruthTaxon:
    scientific_name: Optional[str]
    rank: Optional[str]
    genus: Optional[str]
    family: Optional[str]


@dataclass
class LevelScore:
    evaluated: int = 0
    correct: int = 0
    wrong: int = 0
    not_reached: int = 0  # la predicción quedó en un nivel superior (correcto)

    @property
    def accuracy(self) -> Optional[float]:
        return round(self.correct / self.evaluated, 4) if self.evaluated else None


@dataclass
class Metrics:
    files: int = 0
    category_accuracy: Optional[float] = None
    species: LevelScore = field(default_factory=LevelScore)
    genus: LevelScore = field(default_factory=LevelScore)
    family: LevelScore = field(default_factory=LevelScore)
    false_negatives: List[Tuple[str, str]] = field(default_factory=list)  # (archivo, lo que dio)
    false_positives: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    mean_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    mean_frames: Optional[float] = None

    def to_dict(self) -> dict:
        def level(s: LevelScore) -> dict:
            return {"evaluated": s.evaluated, "correct": s.correct, "wrong": s.wrong,
                    "not_reached": s.not_reached, "accuracy": s.accuracy}
        return {
            "files": self.files, "category_accuracy": self.category_accuracy,
            "species": level(self.species), "genus": level(self.genus), "family": level(self.family),
            "false_negatives": len(self.false_negatives), "false_positives": len(self.false_positives),
            "errors": len(self.errors), "mean_ms": self.mean_ms, "p90_ms": self.p90_ms,
            "mean_frames": self.mean_frames,
        }


def _norm(v: Optional[str]) -> Optional[str]:
    return v.strip().lower() if v else None


def _score_level(score: LevelScore, truth_value: Optional[str], pred_value: Optional[str], pred_reached: bool) -> None:
    score.evaluated += 1
    if pred_value and _norm(pred_value) == _norm(truth_value):
        score.correct += 1
    elif not pred_reached:
        score.not_reached += 1
    else:
        score.wrong += 1


def compute(labels: List[Label], predictions: Dict[str, Prediction], truth_taxa: Dict[str, TruthTaxon],
            top_n_confusion: int = 10) -> Metrics:
    """``truth_taxa``: por sha256, la taxonomía completa de la etiqueta (género y familia)."""
    m = Metrics()
    cat_ok = 0
    times: List[int] = []
    frames: List[int] = []
    pairs: List[Tuple[str, str]] = []
    for lab in labels:
        pred = predictions.get(lab.sha256)
        if pred is None:
            continue
        m.files += 1
        if pred.category == "error":
            m.errors.append((lab.file_name, pred.error or ""))
            continue
        times.append(pred.processing_ms)
        frames.append(pred.frames_analyzed)
        if pred.category == lab.label_category:
            cat_ok += 1
        if lab.label_category == "animal" and pred.category != "animal":
            m.false_negatives.append((lab.file_name, pred.category))
        if lab.label_category == "blank" and pred.category == "animal":
            m.false_positives.append((lab.file_name, pred.scientific_name or pred.raw_label or "animal"))
        if lab.label_category != "animal" or not lab.label_rank:
            continue

        truth = truth_taxa.get(lab.sha256) or TruthTaxon(lab.label_scientific_name, lab.label_rank, None, None)
        depth_truth = RANK_DEPTH[lab.label_rank]
        depth_pred = RANK_DEPTH.get(pred.rank or "", -1) if pred.category == "animal" else -1
        if depth_truth >= RANK_DEPTH["species"]:
            pred_species = pred.scientific_name if pred.rank == "species" else None
            _score_level(m.species, truth.scientific_name, pred_species, depth_pred >= RANK_DEPTH["species"])
            pairs.append((truth.scientific_name or "?", pred_species or (pred.scientific_name or pred.category)))
        if depth_truth >= RANK_DEPTH["genus"]:
            _score_level(m.genus, truth.genus, pred.genus, depth_pred >= RANK_DEPTH["genus"])
        if depth_truth >= RANK_DEPTH["family"]:
            _score_level(m.family, truth.family, pred.family, depth_pred >= RANK_DEPTH["family"])

    evaluated = m.files - len(m.errors)
    m.category_accuracy = round(cat_ok / evaluated, 4) if evaluated else None
    if times:
        ordered = sorted(times)
        m.mean_ms = round(mean(times), 1)
        m.p90_ms = float(ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))])
        m.mean_frames = round(mean(frames), 2)
    top = [name for name, _ in Counter(t for t, _ in pairs).most_common(top_n_confusion)]
    for truth_name, pred_name in pairs:
        if truth_name in top:
            row = m.confusion.setdefault(truth_name, {})
            row[pred_name] = row.get(pred_name, 0) + 1
    return m
