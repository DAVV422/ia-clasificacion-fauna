"""La app FastAPI de la IA (Épica 2). ``uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 1``.

- ``lifespan`` carga los modelos **una vez**, en un hilo, desde ``MODELS_DIR`` (E2-D7): ``md_v5a.0.0.pt`` y
  ``speciesnet-4.0.2a/``, sin ``torch.hub`` remoto ni Kaggle (HU-DS-11). Mientras cargan, ``/detect`` responde
  ``503 IA_MODEL_NOT_READY``. **Nunca descarga nada.**
- Correlación (IA §1): ``X-Request-ID`` (o uno nuevo) y ``X-Flow`` van a ``meta`` y vuelven en la respuesta.
- Sin CORS: la IA es interna (IA §1, IA-31).

``create_app(settings, models)``: con ``models`` inyectado (``FakeModels`` en los tests) no carga nada.
"""

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request

from .errors import install_handlers
from .routes import router
from .settings import Settings

logger = logging.getLogger("ia")


def _configure_env(settings: Settings) -> None:
    """Las cachés de las librerías dentro de MODELS_DIR y nada de internet (las mismas variables que el escritorio)."""
    cache = settings.models_dir / "cache"
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")


def _load_models(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    if not settings.models_present():
        app.state.models_status = "missing"
        logger.error(
            "ia.models.missing: faltan los modelos en %s (md_v5a.0.0.pt y speciesnet-4.0.2a/). "
            "Prepáralos con tools/fetch_models.py",
            settings.models_dir,
        )
        return
    try:
        from wwm_pipeline.models import ModelRegistry

        registry = ModelRegistry(settings.detector_path, settings.classifier_dir)
        registry.load()
        app.state.models = registry
        app.state.model_names = registry.describe()
        app.state.models_status = "ready"
        logger.info("ia.models.ready %s", app.state.model_names)
    except Exception:
        app.state.models_status = "error"
        logger.exception("ia.models.failed")


def create_app(settings: Optional[Settings] = None, models: Any = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if models is None:
            _configure_env(settings)
            logger.info("ia.models.loading desde %s", settings.models_dir)
            threading.Thread(target=_load_models, args=(app,), daemon=True, name="ModelLoader").start()
        yield

    app = FastAPI(title="WWM IA", version=settings.app_version, lifespan=lifespan, docs_url="/docs")
    app.state.settings = settings
    app.state.models = models
    app.state.model_names = models.describe() if models is not None else {"detector": "MDv5a", "classifier": "?"}
    app.state.models_status = "ready" if models is not None else "loading"

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.flow = request.headers.get("X-Flow")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    install_handlers(app)
    app.include_router(router)
    return app


app = create_app()
