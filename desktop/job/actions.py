"""Qué se hace con cada archivo clasificado (HU-DS-01, regla **actual** del prototipo).

Copiar o mover a ``<destino>/<carpeta de especie>/`` los archivos con un **animal**; los demás no se
tocan. La regla nueva (dónde va cada categoría, nombres de carpeta, fotograma) llega en la fase 2
(HU-DS-02/03/10/30).
"""

import shutil
from pathlib import Path
from typing import Callable, Optional

UNKNOWN_FOLDER = "animal_desconocido"


def species_folder_name(label) -> str:
    """Nombre de carpeta del prototipo: "género;especie;común" → "genero_especie_comun" (HU-DS-03 lo revisa)."""
    if not label:
        return UNKNOWN_FOLDER
    folder = label.replace(";", "_").replace(" ", "_")
    folder = "".join(c for c in folder if c.isalnum() or c in "_-")
    return folder or UNKNOWN_FOLDER


def prototype_label(taxonomy) -> Optional[str]:
    """La etiqueta corta del prototipo (``panthera;onca;jaguar``) desde la taxonomía nueva.

    Es ``short_label(label_raw)``, la misma función que usaba el prototipo, así que los nombres de carpeta
    no cambian. Usa el nombre común **del modelo**, no el del idioma configurado. Si el resultado subió
    de nivel da ``panthera;;panthera species`` (carpeta ``panthera__panthera_species``), igual que el
    prototipo con las etiquetas de género del modelo; HU-DS-03 lo cambia por ``panthera_sp``.
    """
    if taxonomy is None:
        return None
    from wwm_pipeline.classification import short_label

    return short_label(taxonomy.label_raw)


def unique_dest(dest: Path) -> Path:
    """Si el destino ya existe, añade un sufijo numérico."""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while True:
        candidate = dest.parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def make_species_action(dest_root: Path, action: str) -> Callable[[object, Path], Optional[str]]:
    """``apply_action(result, path)`` para ``FolderJob``: copia (``"copy"``) o mueve (``"move"``) los archivos con animal."""
    if action not in ("copy", "move"):
        raise ValueError(f"acción desconocida: {action!r}")
    dest_root = Path(dest_root)

    def apply_action(result, path: Path) -> Optional[str]:
        if result.decision.category != "animal":
            return None
        folder = species_folder_name(prototype_label(result.taxonomy))
        target_dir = dest_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(target_dir / path.name)
        if action == "copy":
            shutil.copy2(str(path), str(dest))
            return f"copiado a {folder}"
        shutil.move(str(path), str(dest))
        return f"movido a {folder}"

    return apply_action
