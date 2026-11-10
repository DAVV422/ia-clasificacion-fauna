"""Evalúa una estrategia sobre el set etiquetado y escribe el reporte (HU-IA-10).

Uso:
    set EVAL_DATA_DIR=D:\\wwm-eval\\videos
    python -m eval.eval --config baseline --split test
    python -m eval.eval --config baseline --split test --resume     # continúa un run interrumpido

Salida:
    eval/runs/<fecha>_<config>_<split>.jsonl    una predicción por archivo (sin imágenes; no se versiona)
    eval/reports/<fecha>_<config>.md / .json    métricas
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict

from eval.configs import STRATEGIES
from eval.labels import by_split, load_labels
from eval.metrics import TruthTaxon, compute
from eval.prediction import Prediction
from eval.report import write_report
from eval.taxonomy_index import DEFAULT_LABELS, TaxonomyIndex

RUNS_DIR = Path(__file__).resolve().parent / "runs"
MODELS = {"detector": "MegaDetector v5a", "classifier": "SpeciesNet 4.0.2a"}


def truth_taxa_for(labels, index: TaxonomyIndex) -> Dict[str, TruthTaxon]:
    out = {}
    for lab in labels:
        if lab.label_category != "animal" or not lab.label_scientific_name:
            continue
        taxon = index.lookup_name(lab.label_scientific_name)
        out[lab.sha256] = TruthTaxon(
            lab.label_scientific_name, lab.label_rank, taxon.genus if taxon else None, taxon.family if taxon else None
        )
    return out


def run(
    config: str,
    split: str,
    data_dir: Path,
    resume: bool,
    labels_path: Path = DEFAULT_LABELS,
    run_file: Path = None,
    labels_csv: Path = None,
    reports_dir: Path = None,
) -> Path:
    labels = by_split(load_labels(labels_csv) if labels_csv else load_labels(), split)
    index = TaxonomyIndex(labels_path)
    strategy = STRATEGIES[config](index)
    run_file = run_file or RUNS_DIR / f"{date.today().isoformat()}_{config}_{split}.jsonl"
    run_file.parent.mkdir(parents=True, exist_ok=True)

    done: Dict[str, Prediction] = {}
    if resume and run_file.exists():
        for line in run_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["sha256"]] = Prediction.from_dict(rec["prediction"])

    with open(run_file, "a", encoding="utf-8") as out:
        for i, lab in enumerate(labels, start=1):
            if lab.sha256 in done:
                continue
            path = data_dir / lab.file_name
            try:
                pred = strategy(path)
            except Exception as exc:  # un archivo que falla no detiene la evaluación
                pred = Prediction(file_name=lab.file_name, category="error", confidence=0.0, error=str(exc)[:300])
            done[lab.sha256] = pred
            out.write(json.dumps({"sha256": lab.sha256, "prediction": pred.to_dict()}, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{i}/{len(labels)}] {lab.file_name}: {pred.category} {pred.scientific_name or ''} ({pred.processing_ms} ms)"
            )

    metrics = compute(labels, done, truth_taxa_for(labels, index))
    kwargs = {"out_dir": reports_dir} if reports_dir else {}
    return write_report(config, split, labels, metrics, MODELS, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--split", default="test", choices=["calib", "test", "all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", default=os.getenv("EVAL_DATA_DIR"))
    parser.add_argument("--labels-file", default=str(DEFAULT_LABELS), help="etiquetas de SpeciesNet (.labels.txt)")
    args = parser.parse_args(argv)
    if not args.data_dir:
        print("Falta EVAL_DATA_DIR (o --data-dir) con los archivos del set de validación.", file=sys.stderr)
        return 2
    report = run(args.config, args.split, Path(args.data_dir), args.resume, Path(args.labels_file))
    print(f"Reporte: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
