"""Textos de la ventana para cada evento del análisis (E-06, E-09). Sin ``tkinter``: se prueban solos."""

from typing import List, Tuple

from desktop.messages import message_for
from wwm_pipeline.evidence import mmss

Line = Tuple[str, str]  # (texto, etiqueta de color: ok | warn | error | dim | head)


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s"


def _verb(action) -> str:
    kind = getattr(getattr(action, "type", None), "value", None)
    return {"copy": "COPIADO", "move": "MOVIDO"}.get(kind or "", "")


def file_done_lines(event, total: int) -> List[Line]:
    """Las líneas del log de pantalla para un ``FileDone`` (tabla de E-06)."""
    lines: List[Line] = [
        (f"🔍 [{event.file_index}/{total}]  {event.file_name}  ({event.elapsed_ms / 1000:.0f} s)", "dim")
    ]
    if event.status == "cancelled":
        lines.append(("   ⏹  Detenido a mitad del archivo · queda en el origen", "warn"))
        return lines
    if event.status == "error":
        lines.append(
            (f"   ⚠  {message_for(event.error_code or 'INTERNAL_ERROR', ref=event.error_code or '-')}", "error")
        )
        return lines
    result = event.result
    category = result.decision.category
    if category == "animal":
        tax = result.taxonomy
        if tax is None or getattr(event.action, "destination_folder", None) is None:
            lines.append(("   🐾  Animal sin especie identificada · queda en el origen", "warn"))
            return lines
        name = f"{tax.scientific_name} ({tax.common_name})" if tax.common_name else tax.scientific_name
        level = {"genus": " (solo género)", "family": " (solo familia)"}.get(tax.rank.value, "")
        when = ""
        if result.media_type == "video" and result.evidence is not None:
            when = f" · {mmss(result.evidence.timestamp_seconds)}"
        review = "  ⚠️ revisar" if result.low_evidence else ""
        conf = f"{round(result.decision.confidence * 100)} %"
        dest = event.action.destination_folder
        photo = " (+ fotograma)" if result.evidence is not None else ""
        lines.append((f"   🐾  {name}{level} {conf}{when} · {_verb(event.action)} a {dest}{photo}{review}", "ok"))
    elif category == "person":
        lines.append(("   🧍  Persona · queda en el origen", "dim"))
    elif category == "vehicle":
        lines.append(("   🚙  Vehículo · queda en el origen", "dim"))
    else:
        lines.append(("   ⬜  Sin animal · queda en el origen", "dim"))
    return lines


def summary_lines(summary) -> List[Line]:
    """El resumen final del log (E-09: contadores, sin estadísticas por especie)."""
    title = "⏹  ANÁLISIS DETENIDO" if summary.stopped else "✅  ANÁLISIS COMPLETADO"
    return [
        (f"\n{'─' * 54}", "head"),
        (f"  {title}", "head"),
        (f"  Total en la carpeta     : {summary.total}", "head"),
        (f"  Analizados              : {summary.processed}", "head"),
        (f"  Clasificados (copiados o movidos) : {summary.relocated}  🐾", "ok"),
        (f"  Quedaron en el origen   : {summary.left_in_place}", "dim"),
        (f"     de ellos, con error  : {summary.errors}", "error" if summary.errors else "dim"),
        (f"  Tiempo                  : {format_duration(summary.elapsed_ms / 1000)}", "dim"),
        (f"{'─' * 54}\n", "head"),
    ]


def summary_dialog_text(summary, output_dir: str) -> str:
    return (
        f"Proceso {'detenido' if summary.stopped else 'terminado'}.\n\n"
        f"  Clasificados            : {summary.relocated}\n"
        f"  Quedaron en el origen   : {summary.left_in_place}\n"
        f"     de ellos, con error  : {summary.errors}\n\n"
        f"Resultados guardados en:\n{output_dir}"
    )


def job_error_text(event) -> str:
    return message_for(event.code, ref=event.code) if event.code else (event.message or "")
