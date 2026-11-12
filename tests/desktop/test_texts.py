"""HU-DS-02 · Los textos de la ventana siguen E-06 (lógica sin tkinter)."""

from types import SimpleNamespace

from desktop.job.events import FileDone, JobSummary
from desktop.ui.texts import file_done_lines, format_duration, summary_dialog_text, summary_lines
from wwm_pipeline.package import DesktopAction


def result(category="animal", rank="species", name="Panthera onca", common="jaguar", t=7.4, media="video", low=False):
    tax = None
    if category == "animal":
        tax = SimpleNamespace(scientific_name=name, common_name=common, rank=SimpleNamespace(value=rank))
    return SimpleNamespace(
        decision=SimpleNamespace(category=category, confidence=0.94),
        taxonomy=tax,
        media_type=media,
        evidence=SimpleNamespace(timestamp_seconds=t),
        low_evidence=low,
    )


def done(res, action=None, status="done", code=None):
    return FileDone(3, "CAM01.mp4", status, 12_000, result=res, action=action, error_code=code)


COPY = DesktopAction(type="copy", destination_folder="panthera_onca_jaguar")
LEFT = DesktopAction(type="left_in_place")


def text(event):
    return file_done_lines(event, 27)[-1][0]


def test_species_line():
    assert (
        text(done(result(), COPY)).strip()
        == "🐾  Panthera onca (jaguar) 94 % · 00:07 · COPIADO a panthera_onca_jaguar (+ fotograma)"
    )
    move = DesktopAction(type="move", destination_folder="panthera_sp")
    genus = text(done(result(rank="genus", name="Panthera", common="panthera species"), move))
    assert "Panthera (panthera species) (solo género)" in genus and "MOVIDO a panthera_sp" in genus
    assert "00:07" not in text(done(result(media="image"), COPY))
    assert text(done(result(low=True), COPY)).endswith("⚠️ revisar")


def test_left_in_place_lines():
    assert "Animal sin especie identificada · queda en el origen" in text(done(result(rank="order"), LEFT))
    assert "Persona · queda en el origen" in text(done(result("person"), LEFT))
    assert "Vehículo · queda en el origen" in text(done(result("vehicle"), LEFT))
    assert "Sin animal · queda en el origen" in text(done(result("blank"), LEFT))
    err = text(done(None, LEFT, status="error", code="IA_DECODE_FAILED"))
    assert "No se pudo leer el archivo" in err and "Queda en la carpeta de origen" in err
    assert "queda en el origen" in text(done(None, None, status="cancelled"))


def test_summary():
    summary = JobSummary(total=11, processed=10, relocated=4, left_in_place=6, errors=1, elapsed_ms=125_000)
    joined = "\n".join(t for t, _ in summary_lines(summary))
    assert "Clasificados (copiados o movidos) : 4" in joined and "Quedaron en el origen   : 6" in joined
    assert "2 min 05 s" in joined
    assert "Quedaron en el origen   : 6" in summary_dialog_text(summary, "D:\\salida")
    assert format_duration(3725) == "1 h 02 min"


# ---------------------------------------------------------------------------
# HU-DS-30 · Disco lleno y original en uso
# ---------------------------------------------------------------------------
def test_disk_full_and_not_deleted_texts():
    stopped = file_done_lines(done(None, status="cancelled", code="DISK_SPACE_LOW"), 10)
    assert "Espacio insuficiente" in stopped[1][0] and stopped[1][1] == "error"
    event = FileDone(3, "CAM01.mp4", "done", 12_000, result=result(), action=COPY, warning_code="SOURCE_NOT_DELETED")
    lines = file_done_lines(event, 10)
    assert "COPIADO a panthera_onca_jaguar" in lines[1][0]
    assert "no se pudo borrar el original" in lines[2][0]
    summary = JobSummary(total=5, processed=5, relocated=3, left_in_place=2, not_deleted=2)
    assert any("2 no se pudieron borrar del origen" in t for t, _ in summary_lines(summary))
    assert "No se pudieron borrar del origen: 2" in summary_dialog_text(summary, "D:\salida")
    assert not any("borrar" in t for t, _ in summary_lines(JobSummary(total=1, processed=1)))
