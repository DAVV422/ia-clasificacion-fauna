"""HU-IA-15 · Fotograma de evidencia: elección del frame, dibujo, JPEG sin EXIF y contrato."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeModels
from tools._fixture_taxa import FIXTURE_LABELS, JAGUAR, PUMA, TAPIR
from wwm_pipeline.aggregation import FrameVote, vote
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.contract import Evidence
from wwm_pipeline.evidence import BOX_COLOR_BGR, label_text, mmss, render_evidence, xyxy_from_prototype
from wwm_pipeline.pipeline import FrameResult, _evidence_target, classify_file
from wwm_pipeline.taxonomy import TaxonomyIndex

cv2 = pytest.importorskip("cv2")


def scene(i: int, size=(180, 320), bright: bool = True):
    h, w = size
    base = 170 if bright else 40
    img = np.full((h, w, 3), base, np.uint8)
    step = max(16, w // 20)
    img[::step, :] = 255 - base
    img[:, ::step] = 255 - base
    bw = w // 5
    x = (i * (w // 35)) % (w - bw)
    img[h // 4 : 3 * h // 4, x : x + bw] = 20 if bright else 90
    return img


def write_video(path: Path, n: int, fps: float, size=(180, 320), bright=True) -> Path:
    h, w = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        writer.write(scene(i, size, bright))
    writer.release()
    return path


def decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# CA-1 · Frame completo, liviano y sin EXIF
# ---------------------------------------------------------------------------
def test_evidence_jpeg_size_and_no_exif(tmp_path):
    video = write_video(tmp_path / "v.mp4", 30, 10, size=(1080, 1920))
    result = classify_file(video, None, PipelineConfig.web(), FakeModels())
    ev = result.evidence
    assert result.taxonomy.scientific_name == "Panthera onca"
    assert (ev.width, ev.height) == (960, 540)
    assert decode(ev.jpeg).shape == (540, 960, 3)
    assert b"\xff\xe1" not in ev.jpeg[:4096] and b"Exif" not in ev.jpeg[:4096]  # sin APP1/EXIF
    assert len(ev.jpeg) < 250 * 1024
    contract = ev.to_contract()
    Evidence.model_validate(contract.model_dump())
    assert contract.image_mime == "image/jpeg"


def test_small_frames_are_not_enlarged(tmp_path):
    video = write_video(tmp_path / "v.mp4", 20, 10)
    ev = classify_file(video, None, PipelineConfig.web(), FakeModels()).evidence
    assert (ev.width, ev.height) == (320, 180)


# ---------------------------------------------------------------------------
# CA-2 · El frame correcto
# ---------------------------------------------------------------------------
def test_evidence_picks_best_frame_of_winner():
    index = TaxonomyIndex(FIXTURE_LABELS, common_names_es={})
    box = (0.2, 0.2, 0.7, 0.6)
    votes = [FrameVote(i, i / 10, JAGUAR, 0.9, 0.5, box) for i in range(10, 50)]
    votes[30] = FrameVote(40, 4.0, JAGUAR, 0.9, 0.8, box)  # el mejor del ganador
    votes.append(FrameVote(5, 0.5, PUMA, 0.99, 0.99, box))  # más alto, pero de OTRO taxón
    result = vote(votes, index)
    assert result.taxonomy.scientific_name == "Panthera onca"
    target = _evidence_target(result, [], votes, FakeModels())
    assert target.frame_index == 40 and target.timestamp_seconds == 4.0


def test_evidence_after_rank_up_uses_any_taxon_of_the_group():
    from tools._fixture_taxa import OCELOT

    index = TaxonomyIndex(FIXTURE_LABELS, common_names_es={})
    box = (0.2, 0.2, 0.7, 0.6)
    votes = [FrameVote(i, i, JAGUAR, 0.9, 0.5, box) for i in range(4)]
    votes += [FrameVote(10 + i, 10 + i, PUMA, 0.9, 0.5, box) for i in range(3)]
    votes += [FrameVote(20, 20, OCELOT, 0.9, 0.9, box), FrameVote(21, 21, OCELOT, 0.9, 0.4, box)]
    result = vote(votes, index)
    assert result.taxonomy.scientific_name == "Felidae"
    assert _evidence_target(result, [], votes, FakeModels()).frame_index == 20


def test_timestamp_uses_decoder_position(tmp_path):
    """El tiempo de la evidencia es el del frame según el decodificador (el mismo de FrameResult)."""
    video = write_video(tmp_path / "v.mp4", 60, 15)
    result = classify_file(video, None, PipelineConfig.desktop(), FakeModels())
    by_index = {f.frame_index: f.timestamp_seconds for f in result.frames}
    assert result.evidence.timestamp_seconds == by_index[result.evidence.frame_index]
    assert result.evidence.timestamp_seconds == pytest.approx(result.evidence.frame_index / 15, abs=0.01)


# ---------------------------------------------------------------------------
# CA-4 · Recuadro dentro de la imagen y dibujado
# ---------------------------------------------------------------------------
def test_bbox_normalized_and_drawn():
    assert xyxy_from_prototype([0.1, 0.2, 0.9, 0.8]) == (0.2, 0.1, 0.8, 0.9)
    assert xyxy_from_prototype([-0.01, 0.0, 1.2, 1.0]) == (0.0, 0.0, 1.0, 1.0)
    frame = np.full((540, 960, 3), 60, np.uint8)
    bbox = (0.25, 0.2, 0.75, 0.8)
    jpeg, w, h, _ = render_evidence(frame, bbox)
    img = decode(jpeg).astype(int)
    x1, y1 = round(0.25 * (w - 1)), round(0.2 * (h - 1))
    x2 = round(0.75 * (w - 1))
    edge = img[y1 + (round(0.8 * (h - 1)) - y1) // 2, x1]  # mitad del lado izquierdo
    assert np.abs(edge - np.array(BOX_COLOR_BGR)).max() < 40
    assert np.abs(img[h // 2, (x1 + x2) // 2] - 60).max() < 10  # el interior no se toca
    assert np.abs(img[5, 5] - 60).max() < 10


# ---------------------------------------------------------------------------
# CA-5 · Etiqueta solo en escritorio
# ---------------------------------------------------------------------------
def test_label_text():
    assert label_text("animal", 0.94, 7.4, "Panthera onca", "jaguar") == "Panthera onca (jaguar) · 94 % · 00:07"
    assert label_text("animal", 0.94, None, "Panthera onca", "jaguar") == "Panthera onca (jaguar) · 94 %"
    assert label_text("person", 0.881, 65.0, lang="es") == "persona · 88 % · 01:05"
    assert mmss(3723.9) == "1:02:03"


def test_label_strip_desktop_only(tmp_path):
    """La franja va DEBAJO del fotograma (no tapa el sello de fecha y hora de la cámara); la web no la lleva."""
    video = write_video(tmp_path / "v.mp4", 30, 10)
    desk = classify_file(video, None, PipelineConfig.desktop(), FakeModels()).evidence
    web = classify_file(video, None, PipelineConfig.web(), FakeModels()).evidence
    assert (web.width, web.height) == (320, 180)
    assert desk.width == 320 and desk.height > 180
    d, w = decode(desk.jpeg).astype(int), decode(web.jpeg).astype(int)
    assert np.abs(d[:170] - w[:170]).mean() < 3  # el fotograma, incluido su pie, queda igual
    strip = d[182:]
    assert strip.mean() < 80 and (strip > 200).all(axis=2).sum() > 20  # franja negra con texto blanco


def test_strip_keeps_max_size_on_portrait_video():
    frame = np.full((960, 540, 3), 90, np.uint8)
    jpeg, w, h, _ = render_evidence(frame, (0.1, 0.1, 0.5, 0.5), label="Tapirus terrestris (anta) · 88 % · 00:12")
    assert max(w, h) <= 960 and decode(jpeg).shape[:2] == (h, w)


def test_label_strip_on_image_has_no_time(tmp_path, monkeypatch):
    import wwm_pipeline.pipeline as pipeline_mod

    seen = []
    original = pipeline_mod.label_text
    monkeypatch.setattr(pipeline_mod, "label_text", lambda *a, **k: seen.append(a) or original(*a, **k))
    image = tmp_path / "a.jpg"
    cv2.imwrite(str(image), scene(0))
    ev = classify_file(image, None, PipelineConfig.desktop(), FakeModels()).evidence
    assert (ev.frame_index, ev.timestamp_seconds) == (0, 0.0)
    assert seen and seen[0][2] is None  # sin mm:ss
    assert "·" in original(*seen[0]) and ":" not in original(*seen[0])


def test_label_in_spanish(tmp_path, monkeypatch):
    import wwm_pipeline.pipeline as pipeline_mod

    seen = []
    original = pipeline_mod.label_text
    monkeypatch.setattr(pipeline_mod, "label_text", lambda *a, **k: seen.append(original(*a, **k)) or seen[-1])
    video = write_video(tmp_path / "v.mp4", 20, 10)
    config = replace(PipelineConfig.desktop(), common_name_lang="es")
    classify_file(video, None, config, FakeModels(labels=[TAPIR, JAGUAR], scores=[0.9, 0.05]))
    assert seen[0].startswith("Tapirus terrestris (anta) · ")


# ---------------------------------------------------------------------------
# CA-6 · Sin evidencia cuando no corresponde
# ---------------------------------------------------------------------------
def test_no_evidence_for_blank(tmp_path):
    video = write_video(tmp_path / "v.mp4", 20, 10, bright=False)
    result = classify_file(video, None, PipelineConfig.desktop(), FakeModels(bright_threshold=200))
    assert result.decision.category == "blank" and result.evidence is None


def test_person_evidence_has_person_box(tmp_path):
    video = write_video(tmp_path / "v.mp4", 20, 10)
    result = classify_file(video, None, PipelineConfig.web(), FakeModels(category_id=1))
    assert result.decision.category == "person" and result.evidence is not None
    assert result.evidence.bbox == pytest.approx((0.2, 0.2, 0.6, 0.7), abs=0.01)  # el recuadro de FakeModels


def test_person_from_classifier_vote():
    """SpeciesNet dice humano sobre un recuadro de "animal": la evidencia es ese recuadro."""
    human = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
    votes = [FrameVote(3, 0.3, human, 0.8, 0.9, (0.1, 0.1, 0.5, 0.5))]
    models = FakeModels(labels=[human])
    result = vote(votes, models.taxonomy)
    target = _evidence_target(result, [FrameResult(3, 0.3, [])], votes, models)
    assert result.category.value == "person" and target.frame_index == 3


def test_legacy_presets_have_no_evidence(tmp_path):
    video = write_video(tmp_path / "v.mp4", 20, 10)
    assert classify_file(video, None, PipelineConfig.legacy_desktop(), FakeModels()).evidence is None
