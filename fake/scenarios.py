"""Escenarios de ``ia-fake`` según el nombre del archivo recibido.

El nombre se compara en minúsculas. El primer escenario que coincide gana.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests/fixtures/ai"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


@dataclass(frozen=True)
class Scenario:
    name: str
    fixture: Optional[str] = None       # fixture de éxito (tests/fixtures/ai/<fixture>.json)
    error_code: Optional[str] = None    # código de error IA_*
    error_message: str = ""
    error_context: Optional[dict] = None
    wait: str = "normal"                # normal | slow | timeout
    bad_contract: bool = False


# (subcadena del nombre, escenario), en orden de prioridad
RULES = [
    ("_error_decode", Scenario("error_decode", error_code="IA_DECODE_FAILED",
                               error_message="No se pudo leer el archivo (puede estar dañado).",
                               error_context={"reason": "simulado por ia-fake"})),
    ("_toolong", Scenario("toolong", error_code="IA_DURATION_EXCEEDED",
                          error_message="El video dura 16,2 s y el máximo es 15 s.",
                          error_context={"duration_seconds": 16.2, "max_seconds": 15})),
    ("_badformat", Scenario("badformat", error_code="IA_INVALID_FORMAT", error_message="Formato no compatible.",
                            error_context={"accepted": ["jpg", "png", "mp4", "avi", "mov"]})),
    ("_busy", Scenario("busy", error_code="IA_BUSY", error_message="La IA está ocupada; reintenta.",
                       error_context={"queue_size": 2})),
    ("_notready", Scenario("notready", error_code="IA_MODEL_NOT_READY", error_message="Los modelos se están cargando.")),
    ("_timeout", Scenario("timeout", error_code="IA_PROCESSING_TIMEOUT", wait="timeout",
                          error_message="El procesamiento superó el tiempo máximo.")),
    ("_crash", Scenario("crash", error_code="IA_INTERNAL", error_message="Error interno de la IA.")),
    ("_badcontract", Scenario("badcontract", fixture="animal_species_video", bad_contract=True)),
    ("_slow", Scenario("slow", fixture="animal_species_video", wait="slow")),
    ("_blank", Scenario("blank", fixture="blank_image")),
    ("_person", Scenario("person", fixture="person_video")),
    ("_vehicle", Scenario("vehicle", fixture="vehicle_image")),
    ("_genus", Scenario("genus", fixture="animal_genus_video")),
]
DEFAULT_VIDEO = Scenario("default", fixture="animal_species_video")
DEFAULT_IMAGE = Scenario("default_image", fixture="animal_image")


def pick(file_name: str) -> Scenario:
    lowered = file_name.lower()
    for needle, scenario in RULES:
        if needle in lowered:
            return scenario
    if "_image" in lowered or Path(lowered).suffix in IMAGE_EXTENSIONS:
        return DEFAULT_IMAGE
    return DEFAULT_VIDEO


_cache = {}


def fixture_data(name: str) -> dict:
    """Copia del ``data`` de un fixture de éxito."""
    if name not in _cache:
        _cache[name] = json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))["data"]
    return copy.deepcopy(_cache[name])
