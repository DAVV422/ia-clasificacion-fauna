"""Punto de entrada único de la clasificación: ``classify_file`` (IA-02).

Lo usan la API (``app/main.py``) y la app de escritorio. No conoce FastAPI ni Tkinter.

- Presets ``legacy_*``: reproducen exactamente el prototipo v1 (verificado contra
  ``eval/legacy/expected.json``). Ese camino no se toca.
- Presets ``web()``/``desktop()`` (``sampling="dense"``, HU-IA-11): muestreo por tiempo real en dos
  pasadas secuenciales; MegaDetector en todos los frames conservados y SpeciesNet sobre el mejor
  recuadro de animal de una selección de ``max_classified_frames`` frames (presupuesto, F1-D2).
  La decisión sale de la votación ponderada entre frames (``aggregation.vote``, HU-IA-13). Las
  imágenes con esos presets usan la misma votación, con un solo voto.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .aggregation import Decision, FrameVote, VoteResult, legacy_best, vote
from .classification import classify_crop, classify_crop_raw, to_species_list
from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PipelineConfig
from .detection import detect
from .errors import PipelineCancelled, PipelineError
from .evidence import new_crop_name, save_crop
from .models import Models
from .sampling import PlannedFrame, iter_selected, legacy_frames, sample_video

ProgressFn = Callable[[str, int, int], None]
StopFn = Callable[[], bool]


@dataclass
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    detections: List[Dict[str, Any]]  # formato del prototipo: box, class_name, confidence, species, crop_path, crop_url


@dataclass
class FileResult:
    file_name: str
    media_type: str  # image | video
    decision: Decision
    frames: List[FrameResult] = field(default_factory=list)
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    processing_ms: int = 0
    # --- muestreo denso (HU-IA-11) ---
    sampling: str = "legacy3"  # texto de model.sampling
    frames_discarded: Dict[str, int] = field(default_factory=dict)
    frames_classified: int = 0
    truncated_at_seconds: Optional[float] = None
    # --- votación (HU-IA-13) y taxonomía completa (HU-IA-14); solo en los presets densos ---
    vote: Optional[VoteResult] = None

    @property
    def taxonomy(self):
        """``contract.Taxonomy`` del resultado, o ``None`` si no es fauna (o con los presets legacy)."""
        return self.vote.taxonomy if self.vote is not None else None

    @property
    def low_evidence(self) -> bool:
        return bool(self.vote and self.vote.low_evidence)

    @property
    def frames_sampled(self) -> int:
        return len(self.frames)

    @property
    def frames_with_animal(self) -> int:
        return sum(1 for f in self.frames if any(d.get("class_name") == "animal" for d in f.detections))

    @property
    def frames_analyzed(self) -> List[int]:
        return [f.frame_index for f in self.frames]

    @property
    def all_detections(self) -> List[Dict[str, Any]]:
        return [d for f in self.frames for d in f.detections]


def analyze_frame(frame_bgr, models: Models, config: PipelineConfig, run_name: str) -> List[Dict[str, Any]]:
    """Detección + recorte (opcional) + clasificación de un frame, con la salida del prototipo."""
    detections: List[Dict[str, Any]] = []
    for box in detect(frame_bgr, models, config.md_threshold, config.correct_class_map):
        x1, y1, x2, y2 = box.pixels
        crop = frame_bgr[y1:y2, x1:x2]
        crop_name = new_crop_name()
        crop_path = crop_url = None
        if config.save_crops and config.crops_dir is not None:
            crop_path, crop_url = save_crop(
                crop, config.crops_dir, run_name, box.category, crop_name, config.crops_url_prefix
            )
        species = classify_crop(crop, models, crop_name) if box.category == "animal" else []
        detections.append(
            {
                "box": list(box.box_norm),
                "class_name": box.category,
                "confidence": round(box.confidence, 4),
                "species": species,
                "crop_path": crop_path,
                "crop_url": crop_url,
            }
        )
    return detections


def media_type_of(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise PipelineError(
        "IA_INVALID_FORMAT", f"Extensión no soportada: {ext}", accepted=sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    )


def classify_file(
    path,
    file_name: Optional[str],
    config: PipelineConfig,
    models: Models,
    on_progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
    run_name: Optional[str] = None,
) -> FileResult:
    """Clasifica un archivo (imagen o video) y devuelve una decisión por archivo.

    ``file_name``: el nombre original (para la extensión y, en la fase 1, la fecha); por defecto, el de ``path``.
    ``run_name``: carpeta de los recortes cuando ``config.save_crops`` (por defecto, el nombre sin extensión).
    """
    import cv2

    path = Path(path)
    file_name = file_name or path.name
    run_name = run_name or Path(file_name).stem
    media = media_type_of(file_name)
    t0 = time.perf_counter()

    def check_stop() -> None:
        if should_stop is not None and should_stop():
            raise PipelineCancelled(file_name)

    if media == "image":
        frame = cv2.imread(str(path))
        if frame is None:
            raise PipelineError("IA_DECODE_FAILED", "No se pudo leer la imagen (formato no válido o archivo corrupto).")
        if on_progress:
            on_progress("detection", 1, 1)
        if config.sampling == "dense":
            result = _classify_image_dense(frame, file_name, config, models)
        else:
            detections = analyze_frame(frame, models, config, run_name)
            result = FileResult(file_name, "image", legacy_best(detections), [FrameResult(0, 0.0, detections)])
    elif config.sampling == "dense":
        result = _classify_video_dense(path, file_name, config, models, on_progress, should_stop)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise PipelineError("IA_DECODE_FAILED", "No se pudo abrir el video.")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = legacy_frames(frame_count, config.legacy_mode)
            frames: List[FrameResult] = []
            for i, frame_idx in enumerate(indices, start=1):
                check_stop()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if on_progress:
                    on_progress("detection", i, len(indices))
                detections = analyze_frame(frame, models, config, run_name)
                frames.append(FrameResult(frame_idx, round(frame_idx / fps, 2), detections))
                if config.early_exit and any(d.get("class_name") == "animal" for d in detections):
                    break
        finally:
            cap.release()
        all_dets = [d for f in frames for d in f.detections]
        result = FileResult(
            file_name,
            "video",
            legacy_best(all_dets),
            frames,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=round(frame_count / fps, 2),
        )
    result.processing_ms = int((time.perf_counter() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# Muestreo denso (HU-IA-11)
# ---------------------------------------------------------------------------
@dataclass
class _AnimalCandidate:
    position: int  # posición en la lista de frames conservados
    frame: PlannedFrame
    detection: Dict[str, Any]  # el dict del recuadro de animal de mayor confianza de ese frame
    crop: Any  # recorte BGR (reducido a classify_crop_max_px)


def _shrink(crop_bgr, max_px: int):
    import cv2

    h, w = crop_bgr.shape[:2]
    scale = max_px / max(h, w)
    if scale >= 1:
        return crop_bgr.copy()
    return cv2.resize(crop_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def choose_for_classification(candidates: List[_AnimalCandidate], budget: int) -> List[_AnimalCandidate]:
    """Presupuesto de SpeciesNet: la mitad por mayor confianza de detección y la otra mitad repartida en el
    tiempo (para que un segundo animal que aparece más tarde también se clasifique). Devuelve en orden temporal."""
    if len(candidates) <= budget:
        return list(candidates)
    if budget <= 0:
        return []
    by_conf = sorted(candidates, key=lambda c: (-c.detection["confidence"], c.frame.t))
    chosen = {id(c): c for c in by_conf[: (budget + 1) // 2]}
    rest = [c for c in candidates if id(c) not in chosen]
    need = budget - len(chosen)
    if need > 0 and rest:
        step = len(rest) / need
        for i in range(need):
            c = rest[min(len(rest) - 1, int(i * step + step / 2))]
            chosen[id(c)] = c
    return sorted(chosen.values(), key=lambda c: c.frame.t)


def _classify_video_dense(
    path: Path,
    file_name: str,
    config: PipelineConfig,
    models: Models,
    on_progress: Optional[ProgressFn],
    should_stop: Optional[StopFn],
) -> FileResult:
    params = config.sampling_params
    sampled = sample_video(str(path), params, should_stop=should_stop, on_progress=on_progress)
    frames: List[FrameResult] = []
    candidates: List[_AnimalCandidate] = []
    total = len(sampled.frames)
    for i, (planned, image) in enumerate(iter_selected(str(path), sampled.frames), start=1):
        if should_stop is not None and should_stop():
            raise PipelineCancelled(file_name)
        if on_progress:
            on_progress("detection", i, total)
        detections, cand = _detect_dense(image, planned, len(frames), models, config)
        if cand is not None:
            candidates.append(cand)
        frames.append(FrameResult(planned.index, round(planned.t, 2), detections))

    chosen = choose_for_classification(candidates, config.max_classified_frames)
    votes = _classify_candidates(chosen, file_name, models, on_progress, should_stop)
    candidates.clear()  # libera los recortes

    result_vote = _vote_file(frames, votes, models, config)
    fps = sampled.fps
    return FileResult(
        file_name,
        "video",
        result_vote.to_decision(),
        frames,
        fps=fps,
        frame_count=sampled.frame_count,
        duration_seconds=round(sampled.frame_count / fps, 2) if fps else None,
        sampling=config.sampling_label,
        frames_discarded=dict(sampled.discarded),
        frames_classified=len(chosen),
        truncated_at_seconds=sampled.truncated_at_seconds,
        vote=result_vote,
    )


def _detect_dense(image, planned: PlannedFrame, position: int, models: Models, config: PipelineConfig):
    """MegaDetector sobre un frame → (detecciones, candidato con el recuadro de animal más confiable)."""
    detections: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_crop = None
    for box in detect(image, models, config.md_threshold, config.correct_class_map):
        det = {
            "box": list(box.box_norm),
            "class_name": box.category,
            "confidence": round(box.confidence, 4),
            "species": [],
            "crop_path": None,
            "crop_url": None,
        }
        detections.append(det)
        if box.category == "animal" and (best is None or det["confidence"] > best["confidence"]):
            x1, y1, x2, y2 = box.pixels
            best, best_crop = det, image[y1:y2, x1:x2]
    if best is None:
        return detections, None
    return detections, _AnimalCandidate(position, planned, best, _shrink(best_crop, config.classify_crop_max_px))


def _classify_candidates(
    chosen: List[_AnimalCandidate],
    file_name: str,
    models: Models,
    on_progress: Optional[ProgressFn],
    should_stop: Optional[StopFn],
) -> List[FrameVote]:
    """SpeciesNet sobre cada candidato elegido; cada uno vota con su etiqueta top-1."""
    votes: List[FrameVote] = []
    for i, cand in enumerate(chosen, start=1):
        if should_stop is not None and should_stop():
            raise PipelineCancelled(file_name)
        if on_progress:
            on_progress("classification", i, len(chosen))
        labels = classify_crop_raw(cand.crop, models, f"{file_name}#{cand.frame.index}")
        cand.detection["species"] = to_species_list(labels)
        cand.detection["labels_raw"] = labels
        if labels:
            raw, score = labels[0]
            votes.append(
                FrameVote(
                    cand.frame.index,
                    round(cand.frame.t, 2),
                    raw,
                    cand.detection["confidence"],
                    score,
                    tuple(cand.detection["box"]),
                )
            )
    return votes


def _vote_file(frames: List[FrameResult], votes: List[FrameVote], models: Models, config: PipelineConfig):
    md_confidence: Dict[str, float] = {}
    for d in (d for f in frames for d in f.detections):
        if d["class_name"] in ("person", "vehicle"):
            md_confidence[d["class_name"]] = max(md_confidence.get(d["class_name"], 0.0), d["confidence"])
    return vote(
        votes,
        taxonomy_index_of(models),
        species_threshold=config.species_threshold,
        genus_threshold=config.genus_threshold,
        min_single_vote=config.min_single_vote,
        md_confidence=md_confidence,
        frames_without_detections=sum(1 for f in frames if not f.detections),
        frames_total=len(frames),
        lang=config.common_name_lang,
    )


def _classify_image_dense(image, file_name: str, config: PipelineConfig, models: Models) -> FileResult:
    planned = PlannedFrame(0, 0.0, 0.0, None)
    detections, cand = _detect_dense(image, planned, 0, models, config)
    votes = _classify_candidates([cand] if cand is not None else [], file_name, models, None, None)
    frames = [FrameResult(0, 0.0, detections)]
    result_vote = _vote_file(frames, votes, models, config)
    return FileResult(
        file_name,
        "image",
        result_vote.to_decision(),
        frames,
        sampling=config.sampling_label,
        frames_classified=len(votes),
        vote=result_vote,
    )


def taxonomy_index_of(models: Models):
    """El índice del modelo cargado; si el modelo no lo expone, uno vacío que interpreta cada etiqueta."""
    from .taxonomy import TaxonomyIndex

    index = getattr(models, "taxonomy", None)
    return index if index is not None else TaxonomyIndex([], common_names_es=None)
