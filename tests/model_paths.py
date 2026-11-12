"""Dónde están los modelos reales para los tests de integración (HU-DS-11: sin Kaggle ni torch.hub).

- MegaDetector: ``models/md_v5a.0.0.pt`` del repo.
- SpeciesNet: ``WWM_SPECIESNET_DIR`` si está definida; si no, la carpeta de la app
  (``%LOCALAPPDATA%\\WWM\\models\\speciesnet-4.0.2a``, que arma ``tools/prepare_models.py``); si no, la caché de
  Kaggle de esta PC (también es una carpeta local con su ``info.json``).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "models" / "md_v5a.0.0.pt"


def speciesnet_dir() -> Path:
    env = os.getenv("WWM_SPECIESNET_DIR")
    if env:
        return Path(env)
    local = (
        Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "WWM" / "models" / "speciesnet-4.0.2a"
    )
    if (local / "info.json").is_file():
        return local
    return Path.home() / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"


SPECIESNET = speciesnet_dir()
HAVE_MODELS = MD.is_file() and (SPECIESNET / "info.json").is_file()


def registry(**kwargs):
    from wwm_pipeline.models import ModelRegistry

    return ModelRegistry(MD, SPECIESNET, **kwargs)
