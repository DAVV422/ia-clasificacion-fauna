"""Selección de frames.

- ``legacy_frames``: el muestreo del prototipo (3 frames al 25, 50 y 75 %), para los presets ``legacy_*``.
- Muestreo denso (HU-IA-11, presets ``web()`` y ``desktop()``):
    1. **Por tiempo real**: se elige un frame cada ``1/head_fps`` s en la ventana inicial
       ``[0, head_window_seconds)`` y uno cada ``1/sampling_fps`` s después. El tiempo sale de la
       posición del decodificador, no de ``índice / fps`` (en videos de FPS variable esa división
       se equivoca en más de 1 s: medido en ``estacion_1``).
    2. **Duplicados**: se descarta un frame casi igual al último conservado (fracción de píxeles
       que cambian, sobre una miniatura), salvo que hayan pasado ``max_gap_seconds`` desde el último.
    3. **Borrosos**: varianza del Laplaciano menor que ``blur_ratio`` × la mediana del video
       (relativa, para no vaciar los videos nocturnos), con la misma salvaguarda.
    4. **Tope** ``max_frames``: se ralea uniformemente el tramo posterior a la ventana inicial.
    5. **Límite de duración** (web, F1-D10): no se rechaza; se analiza hasta ``max_video_seconds``
       y se informa ``truncated_at_seconds``.

Las funciones de selección son puras (reciben tiempos y medidas), para probarlas sin videos.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Tolerancia del límite de duración: un video de 15,3 s no se considera recortado.
DURATION_TOLERANCE_SECONDS = 0.5
# Miniaturas para los filtros (baratas: no dependen de la resolución del video).
THUMB_SIZE = (96, 54)
BLUR_WIDTH = 320
PIXEL_CHANGE_DELTA = 20  # diferencia de gris (0–255) a partir de la cual un píxel "cambió"
_EPS = 1e-4


def legacy_frames(frame_count: int, mode: str = "desktop") -> List[int]:
    """Índices de los frames que analiza el prototipo.

    - ``desktop`` (``desktop_app._process_file``): ``min(3, max(1, frame_count))`` frames.
    - ``api`` (``app/main.py``): siempre 3 frames si hay frames; si no, el frame 0.
    """
    if mode == "api":
        if frame_count <= 0:
            return [0]
        interval = frame_count / 4
        return [int(interval * i) for i in range(1, 4)]
    n_samples = min(3, max(1, frame_count))
    interval = frame_count / (n_samples + 1)
    return [int(interval * i) for i in range(1, n_samples + 1)]


# ---------------------------------------------------------------------------
# 1. Plan por tiempo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingParams:
    head_window_seconds: float
    head_fps: float
    sampling_fps: float
    max_frames: int
    max_video_seconds: Optional[float] = None
    dup_threshold: float = 0.001  # fracción de píxeles que cambian por debajo de la cual es duplicado
    blur_ratio: float = 0.35
    max_gap_seconds: float = 1.0

    @property
    def label(self) -> str:
        """Texto legible para ``model.sampling``: ``15fps@0-2s+6fps/max120``."""
        label = f"{_num(self.head_fps)}fps@0-{_num(self.head_window_seconds)}s+{_num(self.sampling_fps)}fps/max{self.max_frames}"
        if self.max_video_seconds is not None:
            label += f"/hasta{_num(self.max_video_seconds)}s"
        return label


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


class TimePlanner:
    """Decide, frame a frame y en orden, cuáles entran al plan según su tiempo real."""

    def __init__(self, params: SamplingParams):
        self.p = params
        self._next_t = 0.0
        self.truncated = False
        self.stopped = False

    def offer(self, t: float) -> bool:
        """``True`` si el frame de tiempo ``t`` (s) entra al plan.

        Pasado ``max_video_seconds`` ya no entra ningún frame (``stopped``); si el video sigue más allá
        de la tolerancia, se marca ``truncated`` y quien lee puede dejar de decodificar.
        """
        limit = self.p.max_video_seconds
        if limit is not None and t > limit + _EPS:
            self.stopped = True
            if t > limit + DURATION_TOLERANCE_SECONDS:
                self.truncated = True
            return False
        if t + _EPS < self._next_t:
            return False
        head = self.p.head_window_seconds
        if t + _EPS < head:
            step = 1.0 / self.p.head_fps
            self._next_t = (math.floor((t + _EPS) / step) + 1) * step
        else:
            step = 1.0 / self.p.sampling_fps
            self._next_t = head + (math.floor((t - head + _EPS) / step) + 1) * step
        return True


def plan_times(times: Iterable[float], params: SamplingParams) -> Tuple[List[int], bool]:
    """Posiciones (en ``times``) que entran al plan y si el video se recortó. Útil para los tests."""
    planner = TimePlanner(params)
    chosen = []
    for pos, t in enumerate(times):
        if planner.offer(t):
            chosen.append(pos)
        elif planner.truncated:
            break
    return chosen, planner.truncated


# ---------------------------------------------------------------------------
# 2–4. Filtros y tope sobre los frames del plan
# ---------------------------------------------------------------------------
@dataclass
class PlannedFrame:
    index: int  # índice del frame en el video
    t: float  # segundos, posición real
    sharpness: float  # varianza del Laplaciano
    thumb: object  # miniatura en gris (np.ndarray uint8) para comparar duplicados


@dataclass
class Selection:
    kept: List[PlannedFrame]
    discarded: Dict[str, int] = field(default_factory=lambda: {"duplicate": 0, "blurry": 0, "cap": 0})


def changed_fraction(a, b) -> float:
    """Fracción de píxeles cuya diferencia de gris supera ``PIXEL_CHANGE_DELTA``."""
    import numpy as np

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float((diff > PIXEL_CHANGE_DELTA).mean())


def select_frames(planned: Sequence[PlannedFrame], params: SamplingParams) -> Selection:
    """Aplica los filtros de borrosos y duplicados (con salvaguarda) y después el tope."""
    sel = Selection(kept=[])
    if not planned:
        return sel
    sharp_values = sorted(f.sharpness for f in planned)
    median = sharp_values[len(sharp_values) // 2]
    blur_cut = params.blur_ratio * median if median > 0 else 0.0
    last: Optional[PlannedFrame] = None
    for frame in planned:
        if last is not None and frame.t - last.t < params.max_gap_seconds - _EPS:
            if frame.sharpness < blur_cut:
                sel.discarded["blurry"] += 1
                continue
            if changed_fraction(frame.thumb, last.thumb) < params.dup_threshold:
                sel.discarded["duplicate"] += 1
                continue
        sel.kept.append(frame)
        last = frame
    sel.kept, sel.discarded["cap"] = cap_frames(sel.kept, params.max_frames, params.head_window_seconds)
    return sel


def cap_frames(
    frames: List[PlannedFrame], max_frames: int, head_window_seconds: float
) -> Tuple[List[PlannedFrame], int]:
    """Deja como mucho ``max_frames``: conserva la ventana inicial y ralea uniformemente el resto."""
    if len(frames) <= max_frames:
        return frames, 0
    head = [f for f in frames if f.t + _EPS < head_window_seconds]
    rest = [f for f in frames if f.t + _EPS >= head_window_seconds]
    if len(head) >= max_frames:  # caso raro: ventana más densa que el tope
        return _uniform(head, max_frames), len(frames) - max_frames
    return head + _uniform(rest, max_frames - len(head)), len(frames) - max_frames


def _uniform(items: List[PlannedFrame], k: int) -> List[PlannedFrame]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[0]]
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


# ---------------------------------------------------------------------------
# Lectura del video (OpenCV): dos pasadas secuenciales, sin saltos
# ---------------------------------------------------------------------------
@dataclass
class SampledVideo:
    frames: List[PlannedFrame]
    discarded: Dict[str, int]
    frames_planned: int
    truncated_at_seconds: Optional[float]
    fps: float
    frame_count: int


def measure(frame_bgr) -> Tuple[float, object]:
    """Nitidez (varianza del Laplaciano a 320 px de ancho) y miniatura gris para duplicados."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    small = cv2.resize(gray, (BLUR_WIDTH, max(1, round(h * BLUR_WIDTH / w))), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    thumb = cv2.GaussianBlur(cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA), (3, 3), 0)
    return sharpness, thumb


def _frame_time(cap, index: int, fps: float) -> float:
    import cv2

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if t <= 0 and index > 0:  # el decodificador no informa la posición: se usa la nominal
        t = index / fps
    return t


def sample_video(path: str, params: SamplingParams, should_stop=None, on_progress=None) -> SampledVideo:
    """Primera pasada: arma el plan por tiempo, mide cada frame del plan y aplica filtros y tope."""
    import cv2

    from .errors import PipelineCancelled, PipelineError

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineError("IA_DECODE_FAILED", "No se pudo abrir el video.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        planner = TimePlanner(params)
        planned: List[PlannedFrame] = []
        index = 0
        while cap.grab():
            t = _frame_time(cap, index, fps)
            if planner.offer(t):
                ok, frame = cap.retrieve()
                if ok:
                    sharpness, thumb = measure(frame)
                    planned.append(PlannedFrame(index, round(t, 3), sharpness, thumb))
            elif planner.truncated:  # pasado el límite y la tolerancia: no hace falta seguir leyendo
                break
            index += 1
            if index % 30 == 0:
                if should_stop is not None and should_stop():
                    raise PipelineCancelled(str(path))
                if on_progress and frame_count:
                    on_progress("sampling", min(index, frame_count), frame_count)
    finally:
        cap.release()
    if not planned:
        raise PipelineError("IA_DECODE_FAILED", "No se pudo leer ningún frame del video.")
    selection = select_frames(planned, params)
    return SampledVideo(
        frames=selection.kept,
        discarded=selection.discarded,
        frames_planned=len(planned),
        truncated_at_seconds=params.max_video_seconds if planner.truncated else None,
        fps=fps,
        frame_count=frame_count,
    )


def iter_selected(path: str, frames: Sequence[PlannedFrame]) -> Iterator[Tuple[PlannedFrame, Any]]:
    """Segunda pasada: entrega (frame del plan, imagen BGR) de los frames conservados, en orden."""
    import cv2

    wanted = {f.index: f for f in frames}
    if not wanted:
        return
    last = max(wanted)
    cap = cv2.VideoCapture(str(path))
    try:
        index = 0
        while index <= last and cap.grab():
            if index in wanted:
                ok, image = cap.retrieve()
                if ok:
                    yield wanted[index], image
            index += 1
    finally:
        cap.release()
