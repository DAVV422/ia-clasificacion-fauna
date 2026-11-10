"""Selección de frames. Fase 0: el muestreo del prototipo (3 frames al 25, 50 y 75 %).

La fase 1 (IA-11) agrega la ventana inicial densa y el muestreo a N fps.
"""

from typing import List


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
