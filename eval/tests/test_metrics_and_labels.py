"""HU-IA-10 · CA-1, CA-3, CA-5, CA-6, CA-7 — sin modelos (corren en CI)."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from eval.labels import LABELS_CSV, Label, LabelError, file_sha256, load_labels, split_for, validate_row, write_labels
from eval.metrics import TruthTaxon, compute
from eval.prediction import Prediction
from eval.report import write_report


def sha(n: str) -> str:
    return hashlib.sha256(n.encode()).hexdigest()


def lab(name, category, sci="", rank="", media="video") -> Label:
    s = sha(name)
    return Label(name, s, media, category, sci, rank, split_for(s), "")


def pred(name, category, sci=None, rank=None, genus=None, family=None, ms=1000, frames=3) -> Prediction:
    return Prediction(file_name=name, category=category, confidence=0.9, scientific_name=sci, rank=rank,
                      genus=genus, family=family, processing_ms=ms, frames_analyzed=frames)


JAGUAR = TruthTaxon("Panthera onca", "species", "panthera", "felidae")


def test_genus_prediction_counts_as_not_reached_not_error():
    """CA-6: etiqueta Panthera onca, predicción Panthera (género)."""
    labels = [lab("a.mp4", "animal", "Panthera onca", "species")]
    m = compute(labels, {labels[0].sha256: pred("a.mp4", "animal", "Panthera", "genus", "panthera", "felidae")},
                {labels[0].sha256: JAGUAR})
    assert (m.species.correct, m.species.wrong, m.species.not_reached) == (0, 0, 1)
    assert m.genus.correct == 1 and m.family.correct == 1


def test_wrong_species_same_family():
    """CA-6: etiqueta Panthera onca, predicción Puma concolor."""
    labels = [lab("a.mp4", "animal", "Panthera onca", "species")]
    m = compute(labels, {labels[0].sha256: pred("a.mp4", "animal", "Puma concolor", "species", "puma", "felidae")},
                {labels[0].sha256: JAGUAR})
    assert m.species.wrong == 1 and m.genus.wrong == 1 and m.family.correct == 1


def test_false_negative_and_positive():
    labels = [lab("animal.mp4", "animal", "Panthera onca", "species"), lab("vacio.mp4", "blank")]
    preds = {labels[0].sha256: pred("animal.mp4", "blank"),
             labels[1].sha256: pred("vacio.mp4", "animal", "Puma concolor", "species", "puma", "felidae")}
    m = compute(labels, preds, {labels[0].sha256: JAGUAR})
    assert m.false_negatives == [("animal.mp4", "blank")]
    assert m.false_positives == [("vacio.mp4", "Puma concolor")]
    assert m.category_accuracy == 0.0


def test_genus_label_not_scored_for_species():
    labels = [lab("agouti.mp4", "animal", "Dasyprocta", "genus")]
    truth = {labels[0].sha256: TruthTaxon("Dasyprocta", "genus", "dasyprocta", "dasyproctidae")}
    m = compute(labels, {labels[0].sha256: pred("agouti.mp4", "animal", "Dasyprocta leporina", "species",
                                                "dasyprocta", "dasyproctidae")}, truth)
    assert m.species.evaluated == 0 and m.genus.correct == 1 and m.family.correct == 1


def test_errors_excluded_from_accuracy_and_times():
    labels = [lab("a.mp4", "animal", "Panthera onca", "species"), lab("b.mp4", "blank")]
    preds = {labels[0].sha256: Prediction("a.mp4", "error", 0.0, error="decode"),
             labels[1].sha256: pred("b.mp4", "blank", ms=500, frames=1)}
    m = compute(labels, preds, {})
    assert m.errors == [("a.mp4", "decode")] and m.category_accuracy == 1.0 and m.mean_ms == 500


def test_timing_and_confusion():
    labels = [lab(f"{i}.mp4", "animal", "Panthera onca", "species") for i in range(10)]
    preds = {l.sha256: pred(l.file_name, "animal", "Panthera onca", "species", "panthera", "felidae", ms=(i + 1) * 100)
             for i, l in enumerate(labels)}
    preds[labels[0].sha256] = pred("0.mp4", "animal", "Puma concolor", "species", "puma", "felidae", ms=100)
    m = compute(labels, preds, {l.sha256: JAGUAR for l in labels})
    assert m.species.correct == 9 and m.species.accuracy == 0.9
    assert m.confusion["Panthera onca"] == {"Panthera onca": 9, "Puma concolor": 1}
    assert m.p90_ms == 900.0


def test_split_is_stable_and_about_30_percent():
    """CA-3"""
    hashes = [sha(f"archivo-{i}") for i in range(2000)]
    splits = [split_for(h) for h in hashes]
    assert splits == [split_for(h) for h in hashes]  # determinista
    ratio = splits.count("calib") / len(splits)
    assert 0.26 <= ratio <= 0.34
    for category in range(4):  # aproximadamente estratificado: ninguna categoría queda desbalanceada
        subset = splits[category::4]
        assert 0.22 <= subset.count("calib") / len(subset) <= 0.38


def test_label_row_validation():
    good = lab("a.mp4", "animal", "Panthera onca", "species").__dict__
    validate_row({k: str(v) for k, v in good.items()}, 2)
    for bad in ({"label_category": "gato"}, {"label_rank": "subespecie"}, {"split": "test" if good["split"] == "calib" else "calib"},
                {"label_category": "blank"}, {"sha256": "abc"}):
        with pytest.raises(LabelError):
            validate_row({**{k: str(v) for k, v in good.items()}, **bad}, 2)


def test_duplicates_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels([lab("a.mp4", "blank"), lab("a.mp4", "blank")], path)
    with pytest.raises(LabelError, match="duplicado"):
        load_labels(path)


def test_versioned_labels_csv_is_consistent():
    """CA-1: el labels.csv versionado siempre es válido (vacío al inicio)."""
    labels = load_labels(LABELS_CSV)
    data_dir = os.getenv("EVAL_DATA_DIR")
    if data_dir:
        for l in labels:
            assert file_sha256(Path(data_dir) / l.file_name) == l.sha256, l.file_name


def test_report_files(tmp_path):
    """CA-7: el reporte se escribe en .md y .json con las mismas cifras."""
    labels = [lab("a.mp4", "animal", "Panthera onca", "species"), lab("b.mp4", "blank")]
    preds = {labels[0].sha256: pred("a.mp4", "animal", "Panthera onca", "species", "panthera", "felidae"),
             labels[1].sha256: pred("b.mp4", "blank")}
    m = compute(labels, preds, {labels[0].sha256: JAGUAR})
    md = write_report("baseline", "test", labels, m, {"detector": "MDv5a"}, out_dir=tmp_path, day="2026-11-10")
    data = json.loads((tmp_path / "2026-11-10_baseline.json").read_text(encoding="utf-8"))
    assert data["metrics"]["species"]["accuracy"] == 1.0 and data["set"]["files"] == 2
    assert "Acierto de especie | 100.0 %" in md.read_text(encoding="utf-8")
