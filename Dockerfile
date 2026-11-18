# IA de la web (HU-IA-06): uvicorn api.main:app en el puerto 8001 (E2-D6), según contratos/API_IA_BACK.md.
#
# - Dependencias solo del lock con hashes (requirements-api.txt: las del escritorio + fastapi/uvicorn).
# - Los modelos NO van en la imagen: van en el volumen /models (E2-D7) y se preparan UNA vez con
#     docker compose run --rm ia python tools/fetch_models.py
#   El contenedor nunca descarga nada al arrancar: sin modelos, /api/v1/health responde 503 (models_missing).
# - Sin internet en tiempo de ejecución: YOLOv5 local (third_party/), SpeciesNet desde /models, YOLO_OFFLINE.

# ---------------------------------------------------------------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --require-hashes --prefix=/install -r requirements-api.txt

# ---------------------------------------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local

RUN useradd --create-home --uid 1000 ia \
 && mkdir -p /models /tmp/ia \
 && chown ia:ia /models /tmp/ia
WORKDIR /app

# El código de la API y del pipeline (nada de la app de escritorio, salvo model_store para fetch_models.py).
COPY --chown=ia:ia api/ api/
COPY --chown=ia:ia wwm_pipeline/ wwm_pipeline/
COPY --chown=ia:ia third_party/ third_party/
COPY --chown=ia:ia catalog/ catalog/
COPY --chown=ia:ia desktop/__init__.py desktop/paths.py desktop/model_store.py desktop/
COPY --chown=ia:ia desktop/models/model_manifest.json desktop/models/
COPY --chown=ia:ia tools/__init__.py tools/fetch_models.py tools/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_FORMAT=json \
    MODELS_DIR=/models \
    TMP_DIR=/tmp/ia \
    YOLO_OFFLINE=true \
    YOLOv5_AUTOINSTALL=false \
    YOLO_AUTOINSTALL=false

USER ia
VOLUME ["/models"]
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/api/v1/health', timeout=4).status == 200 else 1)"
# Un solo worker de uvicorn: la concurrencia la da la cola de inferencia (HU-IA-04); dos workers = dos copias de
# los modelos en RAM.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
