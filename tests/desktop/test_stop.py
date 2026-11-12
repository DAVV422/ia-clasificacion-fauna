"""HU-DS-31 · Detener de forma segura: cancela mientras clasifica, termina mientras guarda, y cerrar la ventana."""

import contextlib
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop import paths
from desktop.job.events import FileDone, JobDone, JobSummary, Progress
from desktop.job.job_log import JobLog
from desktop.job.transfer import journal_path
from desktop.ui import texts
from tests.desktop.fixtures import RELOCATED, SetModels, write_f_set_sint
from tests.desktop.test_job_log import raw_lines, run
from wwm_pipeline.pipeline import classify_file

pytest.importorskip("cv2")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()


@pytest.fixture
def sd(tmp_path):
    src = tmp_path / "SD"
    write_f_set_sint(src)
    return src


def file_events(events):
    return [e for e in events if isinstance(e, FileDone)]


# ---------------------------------------------------------------------------
# CA-1 · Detener mientras clasifica: se cancela y no se escribe nada
# ---------------------------------------------------------------------------
def test_stop_during_classify_writes_nothing(sd, tmp_path):
    dest = tmp_path / "out"
    stop = threading.Event()

    def classify(path, *args, **kwargs):
        if Path(path).name == "familia_d.mp4":
            stop.set()  # Detener llega mientras se clasifica este video (las imágenes son un solo frame)
        return classify_file(path, *args, **kwargs)

    summary, events = run(sd, dest, action="move", stop=stop, classify=classify)
    done = file_events(events)
    assert [e.file_name for e in done][-1] == "familia_d.mp4" and done[-1].status == "cancelled"
    assert (sd / "familia_d.mp4").exists()  # quedó en el origen, intacto
    assert not list(dest.rglob("familia_d*")) and not journal_path(dest).exists()
    assert "familia_d.mp4" not in {x["file_name"] for x in raw_lines(dest)}
    assert summary.stopped and summary.processed == 3  # danado_j, especie_a, especie_b; no empezó genero_c


# ---------------------------------------------------------------------------
# CA-2 · Detener mientras guarda: la copia o el movimiento termina, con su línea
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", ["copy", "move"])
def test_stop_during_save_finishes_file(sd, tmp_path, action):
    dest = tmp_path / "out"
    stop = threading.Event()

    def fault(step):
        if step == "C3":
            stop.set()  # Detener llega en medio de la copia de especie_a
            time.sleep(0.05)

    summary, events = run(sd, dest, action=action, stop=stop, fault=fault)
    done = file_events(events)
    assert [e.file_name for e in done] == ["danado_j.mp4", "especie_a.mp4"]
    assert done[-1].status == "done" and done[-1].action.type.value == action
    assert (dest / "panthera_onca_jaguar" / "especie_a.mp4").exists()
    assert (dest / "panthera_onca_jaguar" / "especie_a_fotograma.jpg").exists()
    assert (sd / "especie_a.mp4").exists() == (action == "copy")
    assert "especie_a.mp4" in {x["file_name"] for x in raw_lines(dest)}
    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    assert summary.stopped and summary.processed == 2


def test_save_stage_is_reported_before_saving(sd, tmp_path):
    _, events = run(sd, tmp_path / "out")
    for i, e in enumerate(events):
        if isinstance(e, FileDone) and e.status == "done":
            stages = [x.stage for x in events[:i] if isinstance(x, Progress) and x.file_name == e.file_name]
            assert stages[-1] == "save"


# ---------------------------------------------------------------------------
# CA-3 · Resumen
# ---------------------------------------------------------------------------
def test_stopped_summary_text():
    summary = JobSummary(total=340, processed=120, relocated=98, left_in_place=22, errors=1, stopped=True)
    lines = [t for t, _ in texts.summary_lines(summary)]
    assert any("DETENIDO: 120 DE 340" in t for t in lines)
    assert any(texts.RESUME_HINT in t for t in lines)
    dialog = texts.summary_dialog_text(summary, "D:\\salida")
    assert dialog.startswith("Detenido: 120 de 340. Para seguir") and "Clasificados            : 98" in dialog
    done = texts.summary_lines(JobSummary(total=3, processed=3))
    assert not any(texts.RESUME_HINT in t for t, _ in done)


def test_stopping_text():
    assert "terminando de guardar V01.mp4" in texts.stopping_text("save", "V01.mp4")
    assert "se cancela" in texts.stopping_text("detection", "V01.mp4")
    assert "se cancela" in texts.stopping_text(None, None)


# ---------------------------------------------------------------------------
# CA-5 · Nada a medias: detener y continuar analiza exactamente lo que falta
# ---------------------------------------------------------------------------
def test_stop_then_continue_processes_rest(sd, tmp_path):
    dest = tmp_path / "out"
    stop = threading.Event()

    def classify(path, *args, **kwargs):
        if Path(path).name == "genero_c.mp4":
            stop.set()
        return classify_file(path, *args, **kwargs)

    first, _ = run(sd, dest, action="move", stop=stop, classify=classify)
    assert first.stopped
    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    _, events = run(sd, dest, action="move")
    assert "genero_c.mp4" in {e.file_name for e in file_events(events)}
    done = Counter(x["file_name"] for x in raw_lines(dest) if x["status"] == "done")
    assert set(done.values()) == {1}
    for name, folder in RELOCATED.items():
        assert (dest / folder / name).exists() and not (sd / name).exists()
    assert len(list(dest.rglob("*_fotograma.jpg"))) == len(RELOCATED)


# ---------------------------------------------------------------------------
# CA-4 · Cerrar la ventana durante el análisis
# ---------------------------------------------------------------------------
@pytest.fixture
def window(monkeypatch):
    tk = pytest.importorskip("tkinter")
    import desktop.ui.legacy_window as lw

    monkeypatch.setattr(lw.WildlifeDesktopApp, "_init_models_async", lambda self: None)
    monkeypatch.setattr(lw.messagebox, "showinfo", lambda *a: None)
    # Tcl en Windows a veces no encuentra init.tcl al crear muchas ventanas Tk() en el mismo proceso de pytest
    # (la app real crea una sola): se reintenta antes de saltear.
    for attempt in range(3):
        try:
            app = lw.WildlifeDesktopApp()
            break
        except tk.TclError as exc:  # pragma: no cover - sin pantalla o falla intermitente de Tcl
            if attempt == 2:
                pytest.skip(f"sin pantalla: {exc}")
            time.sleep(0.2)
    app.withdraw()
    app.detector = SimpleNamespace(model=object(), models=SetModels())
    yield app, lw
    with contextlib.suppress(tk.TclError):
        app.destroy()


def pump(app, until, timeout=60):
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline, "la ventana no llegó al estado esperado"
        try:
            app.update()
        except Exception:  # la ventana ya se destruyó
            return
        time.sleep(0.01)


@pytest.mark.parametrize("answer", [True, False], ids=["si-detener-y-salir", "no-seguir"])
def test_close_window_during_job(window, sd, tmp_path, monkeypatch, answer):
    app, lw = window
    asked = []
    monkeypatch.setattr(lw.messagebox, "askyesno", lambda title, msg: asked.append(msg) or answer)
    destroyed = []
    real_destroy = app.destroy
    monkeypatch.setattr(app, "destroy", lambda: destroyed.append(True) or real_destroy())
    dest = tmp_path / "out"
    app.input_var.set(str(sd))
    app.output_var.set(str(dest))
    app.action_var.set("move")
    seen = []
    real_on_event = app._on_job_event

    def on_event(event):
        seen.append(event)
        real_on_event(event)
        if isinstance(event, FileDone) and len([e for e in seen if isinstance(e, FileDone)]) == 2:
            app._on_close()  # la investigadora cierra la ventana a mitad del análisis

    app._on_job_event = on_event
    callback_errors = []  # QA-09 real: cerrar dentro del sondeo dejaba un TclError en el log
    app.report_callback_exception = lambda *exc: callback_errors.append(exc)
    app._on_start()
    pump(app, lambda: destroyed or any(isinstance(e, JobDone) for e in seen) and not app.is_running)
    assert not callback_errors, callback_errors

    assert asked and "¿Detener y salir?" in asked[0]
    app._job_thread.join(timeout=30)
    assert not app._job_thread.is_alive() and not app._job_thread.daemon
    lines = raw_lines(dest)
    total = len([e for e in seen if isinstance(e, FileDone)])
    if answer:
        assert destroyed  # se cerró sola al llegar JobDone
        assert total < 10 and len(lines) == len(
            [e for e in seen if isinstance(e, FileDone) and e.status != "cancelled"]
        )
    else:
        assert not destroyed and total == 10  # siguió hasta el final
    assert not journal_path(dest).exists() and not list(dest.rglob("*.part"))
    JobLog(dest).read()  # el registro quedó legible


def test_close_window_when_idle_just_closes(window, monkeypatch):
    app, lw = window
    monkeypatch.setattr(lw.messagebox, "askyesno", lambda *a: pytest.fail("no debía preguntar"))
    destroyed = []
    monkeypatch.setattr(app, "destroy", lambda: destroyed.append(True))
    app._on_close()
    assert destroyed
