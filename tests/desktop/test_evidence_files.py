"""HU-DS-10 · El fotograma junto a cada archivo clasificado: nombres, orden con copiar o mover, fallas."""

import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from desktop import paths
from desktop.job.actions import make_species_action
from desktop.job.events import FileDone
from desktop.job.file_ops import evidence_name, unique_pair
from desktop.job.folder_job import FolderJob
from tests.fakes import FakeModels
from wwm_pipeline.evidence import crop_strip, render_evidence

cv2 = pytest.importorskip("cv2")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WWM_DATA_DIR", str(tmp_path / "wwm"))
    paths.ensure_dirs()
    return tmp_path / "wwm"


def frame(i, value=210):
    img = np.full((180, 320, 3), value, np.uint8)
    img[60:120, (i * 9) % 260 : (i * 9) % 260 + 60] = value - 35
    return img


def write_video(path: Path, value=210) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(30):
        writer.write(frame(i, value))
    writer.release()
    return path


def run(src: Path, dest: Path, action="copy", models=None):
    events: queue.Queue = queue.Queue()
    FolderJob(
        src, models or FakeModels(), events, threading.Event(), make_species_action(dest, action), dest=dest
    ).run()
    out = {}
    while not events.empty():
        e = events.get_nowait()
        if isinstance(e, FileDone):
            out[e.file_name] = e
    return out


@pytest.fixture
def one_video(tmp_path):
    src = tmp_path / "SD"
    src.mkdir()
    write_video(src / "CAM01_20260315_042210.mp4")
    return src


def decode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# CA-1 · Junto al archivo · CA-2 · El JPEG
# ---------------------------------------------------------------------------
def test_evidence_saved_next_to_file(one_video, tmp_path):
    dest = tmp_path / "out"
    done = run(one_video, dest)["CAM01_20260315_042210.mp4"]
    folder = dest / "panthera_onca_jaguar"
    assert (folder / "CAM01_20260315_042210.mp4").exists()
    assert (folder / "CAM01_20260315_042210_fotograma.jpg").exists()
    assert done.line.evidence.file == f"evidence/{done.result.sha256}.jpg"


def test_evidence_no_exif_and_bbox(one_video, tmp_path):
    dest = tmp_path / "out"
    done = run(one_video, dest)["CAM01_20260315_042210.mp4"]
    jpg = dest / "panthera_onca_jaguar" / "CAM01_20260315_042210_fotograma.jpg"
    data = jpg.read_bytes()
    assert b"Exif" not in data[:4096] and b"\xff\xe1" not in data[:4096]
    img = decode(jpg)
    ev = done.line.evidence
    assert max(img.shape[:2]) <= 960
    assert (img.shape[1], ev.width) == (320, 320) and ev.height == 180 < img.shape[0]  # la línea: sin la franja
    x1, y1, x2, y2 = ev.bbox
    edge = img[int((y1 + y2) / 2 * (ev.height - 1)), int(round(x1 * (ev.width - 1)))].astype(int)
    assert np.abs(edge - np.array([0, 220, 255])).max() < 60  # el recuadro amarillo cae sobre el bbox


# ---------------------------------------------------------------------------
# CA-4 · Nunca se sobrescribe
# ---------------------------------------------------------------------------
def test_unique_pair(tmp_path):
    assert unique_pair(tmp_path, "A.mp4") == (tmp_path / "A.mp4", tmp_path / "A_fotograma.jpg")
    (tmp_path / "A.mp4").write_bytes(b"x")
    assert unique_pair(tmp_path, "A.mp4") == (tmp_path / "A (2).mp4", tmp_path / "A (2)_fotograma.jpg")
    (tmp_path / "A (2)_fotograma.jpg").write_bytes(b"x")  # solo el fotograma: el par igual está ocupado
    assert unique_pair(tmp_path, "A.mp4")[0] == tmp_path / "A (3).mp4"
    assert evidence_name("IMG_0042.JPG") == "IMG_0042_fotograma.jpg"


def test_unique_names_for_file_and_frame(one_video, tmp_path):
    dest = tmp_path / "out"
    folder = dest / "panthera_onca_jaguar"
    folder.mkdir(parents=True)
    (folder / "CAM01_20260315_042210.mp4").write_bytes(b"otro")
    (folder / "CAM01_20260315_042210_fotograma.jpg").write_bytes(b"otro")
    run(one_video, dest)
    assert (folder / "CAM01_20260315_042210 (2).mp4").exists()
    assert (folder / "CAM01_20260315_042210 (2)_fotograma.jpg").exists()
    assert (folder / "CAM01_20260315_042210.mp4").read_bytes() == b"otro"  # nunca se sobrescribe


# ---------------------------------------------------------------------------
# CA-5 · Imágenes
# ---------------------------------------------------------------------------
def test_image_evidence_without_time(tmp_path):
    src = tmp_path / "SD"
    src.mkdir()
    cv2.imwrite(str(src / "IMG_0042.jpg"), frame(0))
    dest = tmp_path / "out"
    done = run(src, dest)["IMG_0042.jpg"]
    assert (dest / "panthera_onca_jaguar" / "IMG_0042_fotograma.jpg").exists()
    assert (done.line.evidence.frame_index, done.line.evidence.timestamp_seconds) == (0, 0.0)


# ---------------------------------------------------------------------------
# CA-6 · Falla del fotograma · CA-7 · Falla de la copia
# ---------------------------------------------------------------------------
def test_move_aborted_if_evidence_fails(one_video, tmp_path, monkeypatch):
    import desktop.job.actions as actions

    def fail(path, data):
        raise PermissionError("solo lectura")

    monkeypatch.setattr(actions, "write_atomic", fail)
    dest = tmp_path / "out"
    done = run(one_video, dest, action="move")["CAM01_20260315_042210.mp4"]
    assert done.status == "error" and done.error_code == "EVIDENCE_WRITE_FAILED"
    assert done.line.status.value == "error" and done.line.desktop_action.type.value == "left_in_place"
    assert (one_video / "CAM01_20260315_042210.mp4").exists()
    assert not [p for p in dest.rglob("*") if p.is_file()]


def test_evidence_removed_if_copy_fails(one_video, tmp_path, monkeypatch):
    import desktop.job.actions as actions

    def fail(src, dst):
        raise OSError("disco lleno")

    monkeypatch.setattr(actions.shutil, "copy2", fail)
    dest = tmp_path / "out"
    done = run(one_video, dest)["CAM01_20260315_042210.mp4"]
    assert done.status == "error" and done.error_code == "DESKTOP_ACTION_FAILED"
    assert not [p for p in dest.rglob("*") if p.is_file()]  # ni el fotograma huérfano ni un .part


# ---------------------------------------------------------------------------
# CA-8 · Persona y vehículo, sin fotograma (F2-D2, F2-D9)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category_id", [1, 2])
def test_person_evidence_never_written(tmp_path, data_dir, category_id):
    src = tmp_path / "SD"
    src.mkdir()
    write_video(src / "persona.mp4")
    dest = tmp_path / "out"
    done = run(src, dest, action="move", models=FakeModels(category_id=category_id))["persona.mp4"]
    assert done.line.evidence is None and (src / "persona.mp4").exists()
    jpgs = [p for root in (dest, data_dir, src) if root.exists() for p in root.rglob("*.jpg")]
    assert jpgs == []


# ---------------------------------------------------------------------------
# CA-9 · El recorte para la web
# ---------------------------------------------------------------------------
def test_crop_strip_matches_clean_frame():
    img = np.full((540, 960, 3), 90, np.uint8)
    img[100:400, 200:700] = (40, 160, 90)
    bbox = (0.2, 0.18, 0.73, 0.74)
    labeled = render_evidence(img, bbox, label="Panthera onca (jaguar) · 94 %", label_tail="CAM01.mp4", quality=85)
    clean = render_evidence(img, bbox, quality=85)
    assert labeled.height > labeled.frame_height == clean.height == 540
    cropped = crop_strip(labeled.jpeg, labeled.frame_height, quality=85)
    a = cv2.imdecode(np.frombuffer(cropped, np.uint8), cv2.IMREAD_COLOR).astype(int)
    b = cv2.imdecode(np.frombuffer(clean.jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)
    assert a.shape == b.shape == (540, 960, 3)
    assert np.abs(a - b).mean() < 1.5  # solo la pérdida de una recodificación JPEG


# ---------------------------------------------------------------------------
# La franja termina en el nombre del archivo (CA-3 en el pipeline)
# ---------------------------------------------------------------------------
def test_strip_has_file_name(one_video, tmp_path, monkeypatch):
    import wwm_pipeline.evidence as evidence

    seen = []
    original = evidence.fit_tail
    monkeypatch.setattr(evidence, "fit_tail", lambda *a, **k: seen.append(original(*a, **k)) or seen[-1])
    run(one_video, tmp_path / "out")
    assert seen and seen[0].startswith("Panthera onca (jaguar) · 100 % · 00:0")
    assert seen[0].endswith(".mp4") and "CAM0" in seen[0]  # en 320 px el nombre se acorta por el medio


def test_long_file_name_is_shortened():
    """CA-3: con un nombre de 120 caracteres, el nombre científico y el mm:ss quedan enteros."""
    from wwm_pipeline.evidence import fit_tail

    text = "Panthera onca (jaguar) · 94 % · 00:07"
    name = "CAM01_" + "x" * 110 + "_end.mp4"
    out = fit_tail(text, name, len, 80)
    assert out.startswith(text + " · CAM01") and out.endswith("_end.mp4") and "…" in out and len(out) <= 80
    assert fit_tail(text, "corto.mp4", len, 200) == text + " · corto.mp4"
    assert fit_tail(text, name, len, 10) == text  # si ni siquiera entra el texto, no se agrega el nombre
