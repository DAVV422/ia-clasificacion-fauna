"""Reporte de una evaluación: Markdown legible + JSON para comparar (HU-IA-10 CA-7)."""

import json
import platform
from datetime import date
from pathlib import Path
from typing import Dict, List

from eval.labels import Label
from eval.metrics import Metrics

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f} %"


def system_info() -> Dict[str, str]:
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor() or "?"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["threads"] = str(torch.get_num_threads())
    except Exception:  # pragma: no cover - depende del entorno
        pass
    try:
        from importlib.metadata import version

        info["speciesnet"] = version("speciesnet")
    except Exception:  # pragma: no cover
        pass
    return info


def write_report(
    config: str,
    split: str,
    labels: List[Label],
    metrics: Metrics,
    models: Dict[str, str],
    observations: str = "",
    out_dir: Path = REPORTS_DIR,
    day: str = "",
) -> Path:
    day = day or date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{day}_{config}"
    sysinfo = system_info()
    by_cat: Dict[str, int] = {}
    for lab in labels:
        by_cat[lab.label_category] = by_cat.get(lab.label_category, 0) + 1

    payload = {
        "config": config,
        "split": split,
        "date": day,
        "models": models,
        "system": sysinfo,
        "set": {"files": len(labels), "by_category": by_cat},
        "metrics": metrics.to_dict(),
        "false_negatives": metrics.false_negatives,
        "false_positives": metrics.false_positives,
        "errors": metrics.errors,
        "confusion": metrics.confusion,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s, g, f = metrics.species, metrics.genus, metrics.family
    lines = [
        f"# Evaluación · `{config}` · {day}",
        "",
        f"- **Split:** `{split}` · **Archivos:** {len(labels)} ({', '.join(f'{k}: {v}' for k, v in sorted(by_cat.items()))})",
        f"- **Modelos:** {', '.join(f'{k} {v}' for k, v in models.items())}",
        f"- **Equipo:** {', '.join(f'{k} {v}' for k, v in sysinfo.items())}",
        "",
        "## Métricas",
        "",
        "| Métrica | Valor | Detalle |",
        "|---|---|---|",
        f"| Acierto de categoría (animal, persona, vehículo, vacío) | {_pct(metrics.category_accuracy)} | sobre {metrics.files - len(metrics.errors)} archivos |",
        f"| Acierto de especie | {_pct(s.accuracy)} | {s.correct} de {s.evaluated} · {s.wrong} errores · {s.not_reached} no llegaron a especie |",
        f"| Acierto de género | {_pct(g.accuracy)} | {g.correct} de {g.evaluated} · {g.wrong} errores · {g.not_reached} no llegaron |",
        f"| Acierto de familia | {_pct(f.accuracy)} | {f.correct} de {f.evaluated} · {f.wrong} errores · {f.not_reached} no llegaron |",
        f"| Falsos negativos (había animal y no lo detectó) | {len(metrics.false_negatives)} | |",
        f"| Falsos positivos (vacío clasificado como animal) | {len(metrics.false_positives)} | |",
        f"| Archivos con error | {len(metrics.errors)} | |",
        f"| Tiempo por archivo | media {metrics.mean_ms} ms · p90 {metrics.p90_ms} ms | |",
        f"| Fotogramas analizados por archivo | media {metrics.mean_frames} | |",
        "",
    ]
    if metrics.confusion:
        lines += [
            "## Matriz de confusión (las 10 especies más frecuentes)",
            "",
            "| Etiqueta | Predicciones |",
            "|---|---|",
        ]
        for truth, preds in metrics.confusion.items():
            lines.append(
                f"| {truth} | " + ", ".join(f"{p} ({n})" for p, n in sorted(preds.items(), key=lambda x: -x[1])) + " |"
            )
        lines.append("")
    if metrics.false_negatives:
        lines += ["## Falsos negativos", "", "| Archivo | Lo que dio |", "|---|---|"]
        lines += [f"| {a} | {b} |" for a, b in metrics.false_negatives] + [""]
    if metrics.false_positives:
        lines += ["## Falsos positivos", "", "| Archivo | Lo que dio |", "|---|---|"]
        lines += [f"| {a} | {b} |" for a, b in metrics.false_positives] + [""]
    if metrics.errors:
        lines += ["## Errores", "", "| Archivo | Error |", "|---|---|"]
        lines += [f"| {a} | {b} |" for a, b in metrics.errors] + [""]
    lines += ["## Observaciones", "", observations or "_(completar a mano)_", ""]
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
