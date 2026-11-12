"""Textos de la ventana para cada evento del análisis (E-06, E-09). Sin ``tkinter``: se prueban solos."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

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
        if event.error_code:  # DISK_SPACE_LOW (F3-D7): no se escribió nada de este archivo
            lines.append((f"   ⏹  {message_for(event.error_code, ref=event.error_code)}", "error"))
        else:
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
        if getattr(event, "warning_code", None):
            lines.append((f"   ⚠  {message_for(event.warning_code)}", "warn"))
    elif category == "person":
        lines.append(("   🧍  Persona · queda en el origen", "dim"))
    elif category == "vehicle":
        lines.append(("   🚙  Vehículo · queda en el origen", "dim"))
    else:
        lines.append(("   ⬜  Sin animal · queda en el origen", "dim"))
    return lines


CLOSE_PROMPT = (
    "Análisis en curso",
    "Hay un análisis en curso. ¿Detener y salir?\n\n"
    "Lo analizado queda guardado y podrás continuar después con el mismo destino.",
)
RESUME_HINT = "Para seguir, vuelve a pulsar Iniciar con el mismo destino."


def stopping_text(stage, file_name) -> str:
    """HU-DS-31 (F3-D5): clasificando, se cancela; guardando, termina."""
    if stage == "save" and file_name:
        return f"⏹  Deteniendo… terminando de guardar {file_name}"
    return "⏹  Deteniendo… se cancela el análisis del archivo actual (queda en el origen y se repite al continuar)"


def stopped_title(summary) -> str:
    """E-09: "Detenido: N de M"."""
    return f"Detenido: {summary.processed} de {summary.total}"


def summary_lines(summary) -> List[Line]:
    """El resumen final del log (E-09: contadores, sin estadísticas por especie)."""
    title = f"⏹  {stopped_title(summary).upper()}" if summary.stopped else "✅  ANÁLISIS COMPLETADO"
    hint = [(f"  {RESUME_HINT}", "warn")] if summary.stopped and summary.processed < summary.total else []
    return [
        (f"\n{'─' * 54}", "head"),
        (f"  {title}", "head"),
        *hint,
        (f"  Total en la carpeta     : {summary.total}", "head"),
        (f"  Analizados              : {summary.processed}", "head"),
        (f"  Clasificados (copiados o movidos) : {summary.relocated}  🐾", "ok"),
        (f"  Quedaron en el origen   : {summary.left_in_place}", "dim"),
        (f"     de ellos, con error  : {summary.errors}", "error" if summary.errors else "dim"),
        *_not_deleted_lines(summary),
        (f"  Tiempo                  : {format_duration(summary.elapsed_ms / 1000)}", "dim"),
        (f"{'─' * 54}\n", "head"),
    ]


def _not_deleted_lines(summary) -> List[Line]:
    n = getattr(summary, "not_deleted", 0)
    if not n:
        return []
    if n == 1:
        return [("  ⚠  1 no se pudo borrar del origen (en uso): quedó como copia", "warn")]
    return [(f"  ⚠  {n} no se pudieron borrar del origen (en uso): quedaron como copia", "warn")]


def summary_dialog_text(summary, output_dir: str) -> str:
    return (
        (f"{stopped_title(summary)}. {RESUME_HINT}\n\n" if summary.stopped else "Proceso terminado.\n\n")
        + f"  Clasificados            : {summary.relocated}\n"
        f"  Quedaron en el origen   : {summary.left_in_place}\n"
        f"     de ellos, con error  : {summary.errors}\n"
        + (
            f"  No se pudieron borrar del origen: {summary.not_deleted} (quedaron como copia)\n"
            if getattr(summary, "not_deleted", 0)
            else ""
        )
        + f"\nResultados guardados en:\n{output_dir}"
    )


def job_error_text(event) -> str:
    return message_for(event.code, ref=event.code) if event.code else (event.message or "")


def _same_folder(a: Optional[str], b: str) -> bool:
    if not a:
        return True
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return os.path.normcase(a) == os.path.normcase(b)


def resume_prompt(info, source: str, remaining: int) -> Tuple[str, str]:
    """El diálogo de E-08 (HU-DS-14) para ``messagebox.askyesnocancel``: Sí = Continuar, No = Empezar de nuevo."""
    when = f", {info.last_at:%d/%m/%Y %H:%M}" if info.last_at else ""
    noun = "archivo analizado" if info.done == 1 else "archivos analizados"
    lines = [f"Esta carpeta de destino ya tiene resultados de un análisis anterior ({info.done} {noun}{when})."]
    if not _same_folder(info.source, source):  # F3-D6: se avisa y se deja continuar
        lines.append(f"Este registro es de la carpeta {info.source}.")
    if info.errors:
        lines.append(
            "1 con error se volverá a intentar."
            if info.errors == 1
            else f"{info.errors} con error se volverán a intentar."
        )
    pending = "el 1 que falta" if remaining == 1 else f"los {remaining} que faltan"
    lines += [
        "",
        "¿Qué quieres hacer?",
        f"   Sí  →  Continuar (analizar solo {pending})",
        "   No  →  Empezar de nuevo (analizar todo; el registro anterior se guarda como respaldo)",
        "   Cancelar  →  no hacer nada",
    ]
    return "Análisis anterior encontrado", "\n".join(lines)
