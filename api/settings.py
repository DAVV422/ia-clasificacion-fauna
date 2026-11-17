"""Configuración de la API desde el entorno (ESTÁNDARES §8). Nada se lee en el import de los demás módulos."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MD_FILE = "md_v5a.0.0.pt"
SPECIESNET_DIR = "speciesnet-4.0.2a"


def _default_models_dir() -> Path:
    """``/models`` en el contenedor (E2-D7); en desarrollo, la carpeta de modelos de la app de escritorio."""
    docker = Path("/models")
    if docker.is_dir():
        return docker
    base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "WWM" / "models"


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "dev"))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR") or _default_models_dir()))
    tmp_dir: Path = field(default_factory=lambda: Path(os.getenv("TMP_DIR") or tempfile.gettempdir()))
    max_video_seconds: float = field(default_factory=lambda: _env_float("MAX_VIDEO_SECONDS", 15.0))

    @property
    def detector_path(self) -> Path:
        return self.models_dir / MD_FILE

    @property
    def classifier_dir(self) -> Path:
        return self.models_dir / SPECIESNET_DIR

    def models_present(self) -> bool:
        return self.detector_path.is_file() and (self.classifier_dir / "info.json").is_file()
