"""Genera los fotogramas sintéticos y las respuestas de ejemplo de ``POST /detect``.

Salida:
    tests/fixtures/frames/*.jpg           fotogramas sintéticos (rectángulo + etiqueta), ≤ 960 px
    tests/fixtures/ai/*.json              envolturas completas de la IA (éxito y error)

Uso:
    python -m tools.make_fixtures

Los fotogramas son dibujos sintéticos (no fotos reales) para que los fixtures
sean livianos, reproducibles y sin derechos de imagen.
"""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw

from tools import _fixture_taxa as taxa
from wwm_pipeline.catalog_schema import parse_label
from wwm_pipeline.contract import DetectEnvelope

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "tests/fixtures/frames"
AI_DIR = ROOT / "tests/fixtures/ai"
TIMESTAMP = "2026-11-09T12:00:00Z"
MODEL_VIDEO = {
    "detector": "MDv5a",
    "classifier": "speciesnet-4.0.2a",
    "sampling": "10fps@0-2s+4fps/max60/hasta15s",
    "pipeline_version": "1.0.0",
}
MODEL_IMAGE = {**MODEL_VIDEO, "sampling": "image"}


def fake_sha(name: str) -> str:
    return hashlib.sha256(f"wwm-fixture:{name}".encode()).hexdigest()


def make_frame(
    name: str, text: str, bbox: Tuple[float, float, float, float], size=(960, 540), color=(46, 84, 52)
) -> bytes:
    """Dibuja un fotograma sintético con un recuadro y una franja de texto y lo guarda."""
    w, h = size
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    for y in range(0, h, 24):  # textura simple para que no sea un color plano
        draw.line([(0, y), (w, y + 12)], fill=(color[0] + 10, color[1] + 10, color[2] + 10), width=2)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 214, 0), width=4)
    draw.rectangle([0, h - 34, w, h], fill=(0, 0, 0))
    draw.text((10, h - 28), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    data = buf.getvalue()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    (FRAMES_DIR / f"{name}.jpg").write_bytes(data)
    return data


def taxonomy_of(label: str) -> dict:
    t = parse_label(label)
    assert t is not None
    d = t.model_dump(by_alias=True)
    d["label_raw"] = label
    return d


def alternative_of(label: str, score: float) -> dict:
    t = parse_label(label)
    assert t is not None
    return {
        "label_id": t.label_id,
        "scientific_name": t.scientific_name,
        "common_name": t.common_name,
        "rank": t.rank.value,
        "score": score,
    }


def evidence(frame: bytes, frame_index: int, ts: float, bbox, conf: float, size=(960, 540)) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_seconds": ts,
        "bbox": list(bbox),
        "detection_confidence": conf,
        "image_base64": base64.b64encode(frame).decode("ascii"),
        "image_mime": "image/jpeg",
        "width": size[0],
        "height": size[1],
    }


def envelope_ok(data: dict, request_id: str, message: str = "Archivo clasificado") -> dict:
    return {
        "status": "success",
        "data": data,
        "message": message,
        "error": None,
        "meta": {"request_id": request_id, "timestamp": TIMESTAMP},
    }


def envelope_error(code: str, message: str, request_id: str, details: Optional[list] = None) -> dict:
    return {
        "status": "error",
        "data": None,
        "message": message,
        "error": {"code": code, "flow": "FL-08", "details": details or []},
        "meta": {"request_id": request_id, "timestamp": TIMESTAMP},
    }


def video_media(file_name: str, duration=14.8, fps=30.0) -> dict:
    return {
        "file_name": file_name,
        "media_type": "video",
        "sha256": fake_sha(file_name),
        "size_bytes": 18234567,
        "duration_seconds": duration,
        "fps": fps,
        "width": 1920,
        "height": 1080,
    }


def image_media(file_name: str) -> dict:
    return {
        "file_name": file_name,
        "media_type": "image",
        "sha256": fake_sha(file_name),
        "size_bytes": 2400000,
        "duration_seconds": None,
        "fps": None,
        "width": 4000,
        "height": 3000,
    }


def analysis(sampled: int, with_animal: int, votes: list, ms: int, dup=0, blur=0, cap=0, truncated=None) -> dict:
    return {
        "frames_sampled": sampled,
        "frames_with_animal": with_animal,
        "frames_discarded": {"duplicate": dup, "blurry": blur, "cap": cap},
        "votes": votes,
        "truncated_at_seconds": truncated,
        "processing_ms": ms,
    }


def build_all() -> dict:
    jaguar = parse_label(taxa.JAGUAR)
    puma = parse_label(taxa.PUMA)
    fixtures = {}

    bbox = (0.31, 0.12, 0.78, 0.66)
    frame = make_frame("jaguar", "Panthera onca (jaguar) · 94 % · 00:07", bbox)
    fixtures["animal_species_video"] = envelope_ok(
        {
            "media": video_media("CAM01_20260315_042210.mp4"),
            "capture": {"captured_at": "2026-03-15T04:22:10", "source": "filename"},
            "result": {
                "category": "animal",
                "taxonomy": taxonomy_of(taxa.JAGUAR),
                "confidence": 0.94,
                "alternatives": [alternative_of(taxa.PUMA, 0.03), alternative_of(taxa.OCELOT, 0.01)],
            },
            "evidence": evidence(frame, 212, 7.07, bbox, 0.91),
            "analysis": analysis(
                59,
                21,
                [{"label_id": jaguar.label_id, "scientific_name": jaguar.scientific_name, "frames": 17, "score": 0.94}],
                48210,
                dup=18,
                blur=3,
            ),
            "model": MODEL_VIDEO,
        },
        "fixture-animal-species-video",
    )

    bbox = (0.10, 0.40, 0.42, 0.88)
    frame = make_frame("dasyprocta", "Dasyprocta (solo género) · 71 % · 00:04", bbox, color=(70, 60, 40))
    fixtures["animal_genus_video"] = envelope_ok(
        {
            "media": video_media("CAM02_20260318_181502.mp4", duration=10.2),
            "capture": {"captured_at": "2026-03-18T18:15:02", "source": "filename"},
            "result": {
                "category": "animal",
                "taxonomy": taxonomy_of(taxa.DASYPROCTA_GENUS),
                "confidence": 0.71,
                "alternatives": [alternative_of(taxa.AGOUTI, 0.12)],
            },
            "evidence": evidence(frame, 128, 4.27, bbox, 0.83),
            "analysis": analysis(41, 9, [], 30115, dup=10),
            "model": MODEL_VIDEO,
        },
        "fixture-animal-genus-video",
    )

    fixtures["blank_image"] = envelope_ok(
        {
            "media": image_media("IMG_0042.JPG"),
            "capture": {"captured_at": "2026-03-15T11:02:00", "source": "exif"},
            "result": {"category": "blank", "taxonomy": None, "confidence": 0.97, "alternatives": []},
            "evidence": None,
            "analysis": analysis(1, 0, [], 2100),
            "model": MODEL_IMAGE,
        },
        "fixture-blank-image",
    )

    bbox = (0.40, 0.10, 0.60, 0.90)
    frame = make_frame("person", "Persona · 88 % · 00:01", bbox, color=(60, 60, 70))
    fixtures["person_video"] = envelope_ok(
        {
            "media": video_media("CAM01_20260316_090000.mp4", duration=8.0),
            "capture": {"captured_at": "2026-03-16T09:00:00", "source": "filename"},
            "result": {"category": "person", "taxonomy": None, "confidence": 0.88, "alternatives": []},
            "evidence": evidence(frame, 30, 1.0, bbox, 0.88),
            "analysis": analysis(33, 0, [], 21000),
            "model": MODEL_VIDEO,
        },
        "fixture-person-video",
    )

    bbox = (0.05, 0.35, 0.70, 0.95)
    frame = make_frame("vehicle", "Vehículo · 92 %", bbox, color=(80, 80, 80))
    fixtures["vehicle_image"] = envelope_ok(
        {
            "media": image_media("IMG_0107.JPG"),
            "capture": {"captured_at": None, "source": "none"},
            "result": {"category": "vehicle", "taxonomy": None, "confidence": 0.92, "alternatives": []},
            "evidence": evidence(frame, 0, 0.0, bbox, 0.92),
            "analysis": analysis(1, 0, [], 2600),
            "model": MODEL_IMAGE,
        },
        "fixture-vehicle-image",
    )

    bbox = (0.22, 0.30, 0.81, 0.92)
    frame = make_frame("tapir", "Tapirus terrestris (lowland tapir) · 89 %", bbox, color=(40, 70, 45))
    fixtures["animal_image"] = envelope_ok(
        {
            "media": image_media("IMG_0215.JPG"),
            "capture": {"captured_at": "2026-03-20T02:41:13", "source": "exif"},
            "result": {
                "category": "animal",
                "taxonomy": taxonomy_of(taxa.TAPIR),
                "confidence": 0.89,
                "alternatives": [],
            },
            "evidence": evidence(frame, 0, 0.0, bbox, 0.95),
            "analysis": analysis(1, 1, [], 3400),
            "model": MODEL_IMAGE,
        },
        "fixture-animal-image",
    )

    fixtures["error_decode"] = envelope_error(
        "IA_DECODE_FAILED",
        "No se pudo leer el archivo (puede estar dañado).",
        "fixture-error-decode",
        [{"code": "IA_DECODE_FAILED", "context": {"reason": "moov atom not found"}}],
    )
    # F1-D10: un video de más de 15 s no se rechaza; se analiza hasta el segundo 15 y se avisa.
    bbox = (0.28, 0.20, 0.70, 0.74)
    frame = make_frame("puma_truncated", "Puma concolor (puma) · 97 % · 00:03", bbox, color=(55, 65, 45))
    fixtures["animal_video_truncated"] = envelope_ok(
        {
            "media": video_media("CAM03_20260322_051210.mp4", duration=46.6, fps=15.0),
            "capture": {"captured_at": "2026-03-22T05:12:10", "source": "filename"},
            "result": {
                "category": "animal",
                "taxonomy": taxonomy_of(taxa.PUMA),
                "confidence": 0.97,
                "alternatives": [alternative_of(taxa.JAGUAR, 0.02)],
            },
            "evidence": evidence(frame, 47, 3.13, bbox, 0.93),
            "analysis": analysis(
                60,
                52,
                [{"label_id": puma.label_id, "scientific_name": puma.scientific_name, "frames": 23, "score": 0.97}],
                61840,
                dup=4,
                cap=8,
                truncated=15.0,
            ),
            "model": MODEL_VIDEO,
        },
        "fixture-animal-video-truncated",
    )
    fixtures["error_unauthorized"] = envelope_error(
        "IA_UNAUTHORIZED", "API key ausente o incorrecta.", "fixture-error-unauthorized"
    )
    fixtures["error_busy"] = envelope_error(
        "IA_BUSY",
        "La IA está ocupada; reintenta.",
        "fixture-error-busy",
        [{"code": "IA_BUSY", "context": {"queue_size": 2}}],
    )
    return fixtures


def main() -> None:
    AI_DIR.mkdir(parents=True, exist_ok=True)
    for name, env in build_all().items():
        DetectEnvelope.model_validate(env)  # nunca se escribe un fixture inválido
        (AI_DIR / f"{name}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"ok  {name}.json")


if __name__ == "__main__":
    main()
