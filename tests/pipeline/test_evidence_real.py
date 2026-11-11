"""HU-IA-15 · CA-3 con el material real de WWF (``estacion_1``, fuera del repo).

En lugar de abrir los videos a mano, el tiempo de cada frame se compara con el que fija el propio
contenedor MP4: la tabla ``stts`` (duración de cada muestra) + ``ctts`` (desfase de presentación)
de la pista de video. Es una fuente independiente del decodificador de OpenCV.
"""

import os
import struct
from pathlib import Path

import pytest

from wwm_pipeline.metadata import _boxes

ROOT = Path(__file__).resolve().parents[2]
ESTACION_1 = Path(os.getenv("ESTACION_1_DIR", str(ROOT.parent / "docs-interno" / "estacion_1")))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ESTACION_1.is_dir(), reason="requiere la carpeta estacion_1"),
]


def _child(fh, body, end, kind):
    for k, b, e in _boxes(fh, body, end):
        if k == kind:
            return b, e
    return None


def presentation_times(path: Path):
    """Los tiempos de presentación (s) de cada frame de la pista de video, en orden, desde 0."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        moov = _child(fh, 0, fh.tell(), b"moov")
        for kind, body, end in _boxes(fh, *moov):
            if kind != b"trak":
                continue
            mdia = _child(fh, body, end, b"mdia")
            hdlr = _child(fh, *mdia, b"hdlr")
            fh.seek(hdlr[0] + 8)
            if fh.read(4) != b"vide":
                continue
            mdhd = _child(fh, *mdia, b"mdhd")
            fh.seek(mdhd[0])
            version = fh.read(4)[0]
            fh.seek(mdhd[0] + (20 if version == 1 else 12))
            timescale = struct.unpack(">I", fh.read(4))[0]
            stbl = _child(fh, *_child(fh, *mdia, b"minf"), b"stbl")
            fh.seek(_child(fh, *stbl, b"stts")[0] + 4)
            dts, t = [], 0
            for _ in range(struct.unpack(">I", fh.read(4))[0]):
                count, delta = struct.unpack(">II", fh.read(8))
                for _ in range(count):
                    dts.append(t)
                    t += delta
            offsets = [0] * len(dts)
            ctts = _child(fh, *stbl, b"ctts")
            if ctts:
                fh.seek(ctts[0])
                ver = fh.read(4)[0]
                i = 0
                for _ in range(struct.unpack(">I", fh.read(4))[0]):
                    count, off = struct.unpack(">Ii" if ver == 1 else ">II", fh.read(8))
                    for _ in range(count):
                        offsets[i] = off
                        i += 1
            pts = sorted(d + o for d, o in zip(dts, offsets))
            return [(p - pts[0]) / timescale for p in pts]
    raise AssertionError(f"sin pista de video: {path.name}")


def _videos():
    import cv2

    chosen, constant = [], None
    for v in sorted(ESTACION_1.glob("*.mp4")):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps % 1:
            chosen.append(v)  # FPS variable (3,57 · 8,11 · 9,81 · 13,32 · 14,54 · 30,002)
        elif constant is None:
            constant = v
    return chosen + [constant]


def test_frame_times_match_container():
    """CA-3: en 7 videos (los 6 con FPS no entero, incluidos 3,57 y 8,11 fps, y uno constante), el tiempo
    de cada frame conservado
    por el muestreo —que es el que va a ``evidence.timestamp_seconds``— coincide con el del contenedor."""
    from wwm_pipeline.config import PipelineConfig
    from wwm_pipeline.sampling import sample_video

    videos = _videos()
    assert len(videos) >= 6
    worst = 0.0
    for video in videos:
        pts = presentation_times(video)
        sampled = sample_video(str(video), PipelineConfig.desktop().sampling_params)
        assert sampled.frames
        for frame in sampled.frames:
            worst = max(worst, abs(frame.t - pts[frame.index]))
    # La spec pide ±0,5 s. Medido: exacto salvo el frame 0 de los reenviados (0,067 s); índice / fps llega a 1,76 s.
    assert worst < 0.1
