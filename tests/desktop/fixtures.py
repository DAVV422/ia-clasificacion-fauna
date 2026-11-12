"""``F-SET-SINT``: un set sintético que cubre cada fila de E-06 (reemplaza a F-SET-A, que no existe; fase 2).

Los archivos se generan en ``tmp_path`` (unos KB cada uno). ``SetModels`` decide por el contenido y el nombre:

- MegaDetector, por el brillo medio del frame: > 180 animal · 130–180 persona · 100–130 vehículo · < 100 nada;
- SpeciesNet, por el nombre del archivo (que llega en ``filepath``): ``especie_*`` → jaguar, ``genero_*`` →
  *Dasyprocta* (etiqueta de género del modelo), ``familia_*`` → Felidae, ``orden_*`` → Galliformes.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

from tests.fakes import FakeModels
from tools._fixture_taxa import DASYPROCTA_GENUS, JAGUAR
from wwm_pipeline.models import RawBox

FELIDAE = "ad6a1c7b-1b0c-4c8b-8d1e-2a8b0f1f9a01;mammalia;carnivora;felidae;;;felidae family"
ORDER = "837e12c2-f30c-4875-afeb-32d51938f139;aves;galliformes;;;;galliformes order"
BRIGHTNESS = {"animal": 210, "person": 155, "vehicle": 115, "blank": 40}
LABEL_BY_PREFIX = {"especie": JAGUAR, "genero": DASYPROCTA_GENUS, "familia": FELIDAE, "orden": ORDER}

# nombre → (tipo de medio, lo que "contiene")
F_SET_SINT: Dict[str, str] = {
    "especie_a.mp4": "animal",
    "especie_b.jpg": "animal",
    "genero_c.mp4": "animal",
    "familia_d.mp4": "animal",
    "orden_e.mp4": "animal",
    "vacio_f.mp4": "blank",
    "vacio_g.jpg": "blank",
    "persona_h.mp4": "person",
    "vehiculo_i.jpg": "vehicle",
    "danado_j.mp4": "broken",
    "raw_k.cr2": "unsupported",
}
STAYS_IN_PLACE = {
    "orden_e.mp4",
    "vacio_f.mp4",
    "vacio_g.jpg",
    "persona_h.mp4",
    "vehiculo_i.jpg",
    "danado_j.mp4",
    "raw_k.cr2",
}
RELOCATED = {
    "especie_a.mp4": "panthera_onca_jaguar",
    "especie_b.jpg": "panthera_onca_jaguar",
    "genero_c.mp4": "dasyprocta_sp",
    "familia_d.mp4": "felidae_fam",
}


def frame(i: int, value: int):
    img = np.full((180, 320, 3), value, np.uint8)
    x = (i * 9) % 260
    img[60:120, x : x + 60] = max(0, value - 35)  # un bloque que se mueve: no son duplicados
    return img


def write_f_set_sint(folder: Path) -> List[Path]:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, kind in F_SET_SINT.items():
        path = folder / name
        if kind == "broken":
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 8)
        elif kind == "unsupported":
            path.write_bytes(b"II*\x00 raw de una camara")
        elif path.suffix == ".jpg":
            cv2.imwrite(str(path), frame(0, BRIGHTNESS[kind]))
        else:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
            for i in range(30):
                writer.write(frame(i, BRIGHTNESS[kind]))
            writer.release()
        out.append(path)
    return out


class SetModels(FakeModels):
    def __init__(self):
        super().__init__(labels=[JAGUAR, DASYPROCTA_GENUS, FELIDAE, ORDER], scores=[0.9, 0.05, 0.03, 0.02])

    def run_detector(self, frame_rgb):
        self.detector_calls += 1
        h, w = frame_rgb.shape[:2]
        mean = float(frame_rgb.mean())
        if mean > 180:
            category = (0, "animal")
        elif mean > 130:
            category = (1, "person")
        elif mean > 100:
            category = (2, "vehicle")
        else:
            return []
        return [RawBox(w * 0.2, h * 0.2, w * 0.6, h * 0.7, 0.88, category[0], category[1])]

    def run_classifier(self, crop_pil, filepath: str):
        self.classifier_calls += 1
        prefix = Path(filepath.split("#")[0]).name.split("_")[0]
        label = LABEL_BY_PREFIX.get(prefix, JAGUAR)
        return {"filepath": filepath, "classifications": {"classes": [label], "scores": [0.92]}}
