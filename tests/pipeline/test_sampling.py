"""HU-IA-11 · Muestreo denso: plan por tiempo, filtros, tope, recorte a 15 s y presupuesto de clasificación.

Los videos sintéticos se generan en ``tmp_path`` (unos KB) y se borran solos.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeModels
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.pipeline import _AnimalCandidate, choose_for_classification, classify_file
from wwm_pipeline.sampling import PlannedFrame, plan_times, sample_video, select_frames

cv2 = pytest.importorskip("cv2")

DESKTOP = PipelineConfig.desktop().sampling_params
WEB = PipelineConfig.web().sampling_params


def times(seconds: float, fps: float):
    return [i / fps for i in range(int(round(seconds * fps)))]


def write_video(path: Path, n: int, fps: float, frame_fn, size=(320, 180)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        writer.write(frame_fn(i))
    writer.release()
    return path


def textured(seed: int, size=(180, 320)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (*size, 3), dtype=np.uint8)


def scene(i: int, bright: bool = True):
    """Escena sintética con detalle (cuadrícula) y un bloque grande que se mueve en cada frame.

    A diferencia del ruido aleatorio, sobrevive a la compresión del códec mp4v.
    """
    base = 170 if bright else 60
    img = np.full((180, 320, 3), base, np.uint8)
    img[::16, :] = 255 - base
    img[:, ::16] = 255 - base
    x = (i * 9) % 260
    img[50:130, x : x + 60] = 20 if bright else 220
    return img


# ---------------------------------------------------------------------------
# CA-1, CA-2, CA-3 · El plan
# ---------------------------------------------------------------------------
def test_plan_desktop_15s_30fps():
    t = times(15, 30)
    chosen, truncated = plan_times(t, DESKTOP)
    head = [p for p in chosen if t[p] < 2]
    assert len(head) == 30 and len(chosen) - len(head) == 78 and len(chosen) == 108
    assert head == list(range(0, 60, 2))
    assert not truncated
    assert DESKTOP.label == "15fps@0-2s+6fps/max120"
    assert PipelineConfig.desktop().sampling_label == "15fps@0-2s+6fps/max120"


@pytest.mark.parametrize(
    "seconds,fps,expected",
    [(9.0, 3.57, 32), (1.5, 30, 23)],  # FPS bajo: todos los frames · video corto: todo a 15 fps
)
def test_plan_low_fps_and_short_videos(seconds, fps, expected):
    chosen, _ = plan_times(times(seconds, fps), DESKTOP)
    assert len(chosen) == expected
    assert len(set(chosen)) == len(chosen)  # nunca se repite un frame


def test_plan_irregular_timestamps_use_real_time():
    """CA-7: el plan sigue el tiempo real aunque los frames lleguen con huecos (FPS variable)."""
    t = [0.0, 0.07, 0.13, 0.2, 1.5, 1.55, 2.0, 2.05, 2.4, 9.0]
    chosen, _ = plan_times(t, DESKTOP)
    # 0,13 cae antes del siguiente punto de la grilla de 15 fps (0,133) y 2,05 antes del de 6 fps (2,167)
    assert [t[p] for p in chosen] == [0.0, 0.07, 0.2, 1.5, 1.55, 2.0, 2.4, 9.0]


def _planned(t, seed=0, sharp=100.0):
    return [PlannedFrame(i, ti, sharp, textured(seed + i, (54, 96))[:, :, 0]) for i, ti in enumerate(t)]


def test_cap_keeps_head_window():
    """CA-3: 52 s a 15 fps → 330 en el plan → 120 (los 30 de la ventana + 90 repartidos)."""
    t = times(52, 15)
    chosen, _ = plan_times(t, DESKTOP)
    assert len(chosen) == 330
    sel = select_frames(_planned([t[p] for p in chosen]), DESKTOP)
    assert len(sel.kept) == 120 and sel.discarded["cap"] == 210
    head = [f for f in sel.kept if f.t < 2]
    assert len(head) == 30
    rest = [f.t for f in sel.kept if f.t >= 2]
    assert rest[0] == 2.0 and rest[-1] > 51
    gaps = np.diff(rest)
    assert gaps.max() - gaps.min() < 0.25  # repartidos uniformemente


# ---------------------------------------------------------------------------
# CA-8 · Web: recorte a 15 s (F1-D10)
# ---------------------------------------------------------------------------
def test_web_truncates_at_15s():
    t = times(52, 15)
    chosen, truncated = plan_times(t, WEB)
    assert truncated and max(t[p] for p in chosen) <= 15.0
    _, truncated_desktop = plan_times(t, DESKTOP)
    assert not truncated_desktop
    _, truncated_short = plan_times(times(10, 15), WEB)
    assert not truncated_short
    _, truncated_tolerance = plan_times(times(15.3, 15), WEB)  # 15,3 s no se considera recortado
    assert not truncated_tolerance


def test_web_truncation_on_a_real_file(tmp_path):
    video = write_video(tmp_path / "largo.mp4", 20 * 10, 10, scene)
    sampled = sample_video(str(video), WEB)
    assert sampled.truncated_at_seconds == 15.0
    assert max(f.t for f in sampled.frames) <= 15.0


# ---------------------------------------------------------------------------
# CA-4, CA-5 · Filtros
# ---------------------------------------------------------------------------
def test_static_video_keeps_one_frame_per_second(tmp_path):
    still = textured(7)
    video = write_video(tmp_path / "quieto.mp4", 10 * 15, 15, lambda i: still)
    sampled = sample_video(str(video), DESKTOP)
    seconds = {int(f.t) for f in sampled.frames}
    assert seconds == set(range(10))  # al menos un frame por segundo (max_gap_seconds)
    assert sampled.discarded["duplicate"] > 50


def test_small_animal_crossing_is_kept(tmp_path):
    still = textured(7)

    def frame(i):
        img = still.copy()
        t = i / 15
        if 6.0 <= t < 6.5:  # un cuadrado de 20×20 px cruza durante medio segundo
            x = int(40 + (t - 6.0) * 400)
            img[80:100, x : x + 20] = 255
        return img

    video = write_video(tmp_path / "cruce.mp4", 10 * 15, 15, frame)
    sampled = sample_video(str(video), DESKTOP)
    assert any(6.0 <= f.t < 6.5 for f in sampled.frames)


def test_night_video_is_not_emptied_by_blur_filter(tmp_path):
    rng = np.random.default_rng(3)

    def frame(i):  # poco detalle: ruido suave y oscuro, distinto en cada frame
        return cv2.GaussianBlur(rng.integers(0, 40, (180, 320, 3), dtype=np.uint8), (9, 9), 0)

    video = write_video(tmp_path / "noche.mp4", 8 * 15, 15, frame)
    sampled = sample_video(str(video), DESKTOP)
    assert sampled.discarded["blurry"] <= 0.2 * sampled.frames_planned


def test_strongly_blurred_frame_is_discarded():
    t = [i / 15 for i in range(15)]
    frames = _planned(t)
    frames[5].sharpness = 5.0  # desenfoque fuerte en un video nítido
    sel = select_frames(frames, DESKTOP)
    assert sel.discarded["blurry"] == 1 and all(f.index != 5 for f in sel.kept)


# ---------------------------------------------------------------------------
# Presupuesto de clasificación
# ---------------------------------------------------------------------------
def _cand(i, conf):
    return _AnimalCandidate(i, PlannedFrame(i, i / 10, 1.0, None), {"confidence": conf}, None)


def test_budget_half_by_confidence_half_spread():
    cands = [_cand(i, 0.5) for i in range(100)]
    cands[97].detection["confidence"] = 0.99  # el más seguro está al final
    chosen = choose_for_classification(cands, 10)
    assert len(chosen) == 10
    assert cands[97] in chosen
    assert [c.frame.t for c in chosen] == sorted(c.frame.t for c in chosen)
    assert chosen[0].frame.t < 2 and chosen[-1].frame.t > 7  # cubre todo el video


def test_budget_not_applied_when_few_candidates():
    cands = [_cand(i, 0.5) for i in range(5)]
    assert choose_for_classification(cands, 48) == cands


# ---------------------------------------------------------------------------
# El pipeline con el preset denso (FakeModels)
# ---------------------------------------------------------------------------
def test_dense_pipeline_no_early_exit_and_budget(tmp_path):
    """El animal está en todos los frames: no hay salida temprana y SpeciesNet respeta el presupuesto."""
    video = write_video(tmp_path / "CAM01_20260315_042210.mp4", 30 * 15, 15, scene)
    models = FakeModels(bright_threshold=100)
    result = classify_file(video, None, PipelineConfig.desktop(), models)
    assert result.frames_sampled > 3
    assert result.frames_classified == models.classifier_calls == PipelineConfig.desktop().max_classified_frames
    assert models.detector_calls == result.frames_sampled
    assert result.sampling == "15fps@0-2s+6fps/max120"
    assert set(result.frames_discarded) == {"duplicate", "blurry", "cap"}
    assert result.truncated_at_seconds is None
    assert result.decision.category == "animal" and result.decision.label
    assert result.frames_with_animal == result.frames_sampled


def test_dense_pipeline_web_truncates(tmp_path):
    video = write_video(tmp_path / "largo.mp4", 20 * 10, 10, scene)
    result = classify_file(video, None, PipelineConfig.web(), FakeModels())
    assert result.truncated_at_seconds == 15.0
    assert max(f.timestamp_seconds for f in result.frames) <= 15.0


def test_dense_pipeline_reports_progress_and_stops(tmp_path):
    from wwm_pipeline.errors import PipelineCancelled

    video = write_video(tmp_path / "v.mp4", 6 * 15, 15, scene)
    stages = []
    classify_file(video, None, PipelineConfig.desktop(), FakeModels(), on_progress=lambda s, i, n: stages.append(s))
    assert stages.index("detection") < stages.index("classification")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(PipelineCancelled):
        classify_file(video, None, PipelineConfig.desktop(), FakeModels(), should_stop=stop)


def test_legacy_presets_unchanged():
    """CA-9: los presets legacy siguen con el muestreo de 3 frames."""
    assert PipelineConfig.legacy_desktop().sampling == "legacy3"
    assert PipelineConfig.legacy_api().sampling == "legacy3"
    assert PipelineConfig.legacy_desktop().sampling_label == "legacy3"
