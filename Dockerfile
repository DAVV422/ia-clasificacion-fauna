# ============================================================
# Stage 1: Builder — Instala dependencias en un entorno limpio
# ============================================================
FROM python:3.10-slim AS builder

# Evitar prompts interactivos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema necesarias para OpenCV, PyTorch y compilación de paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiamos solo requirements para aprovechar el cache de layers de Docker
COPY requirements.txt .

# Instalamos en un directorio de destino separado para la copia al stage final
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

RUN pip install ultralytics
RUN pip install seaborn

# Pre-clonamos el repo de YOLOv5 en el caché de PyTorch Hub para evitar
# el prompt interactivo de confianza ("Do you trust this repo?") en runtime.
# Esto resuelve el error "EOF when reading a line" en Docker (sin TTY).
# Nota: los paquetes están en /install (--prefix), hay que agregar PYTHONPATH.
ENV TORCH_HOME=/build/torch_hub
ENV PYTHONPATH=/install/lib/python3.10/site-packages
RUN python -c "import torch, os; os.makedirs('/build/torch_hub/hub', exist_ok=True); torch.hub.set_dir('/build/torch_hub/hub'); torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, trust_repo=True); print('YOLOv5 hub cache pre-populated')"

# ============================================================
# Stage 2: Runtime — Imagen final compacta
# ============================================================
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Solo librerías de runtime (sin gcc ni herramientas de compilación)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos los paquetes instalados desde el stage builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copiamos el código fuente
COPY app/ ./app/

# Creamos directorios de datos que serán montados como volúmenes
RUN mkdir -p models detections temp

# Copiamos el caché de PyTorch Hub pre-poblado desde el builder
# Esto incluye el repo de ultralytics/yolov5 ya clonado y confiado
COPY --from=builder /build/torch_hub /app/models/torch_hub

# Variables de entorno configurables en tiempo de ejecución
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Directorio de cache de PyTorch Hub (apunta al caché pre-poblado)
    TORCH_HOME=/app/models/torch_hub \
    # Directorio de cache de Kaggle Hub (donde SpeciesNet guarda los pesos)
    KAGGLE_CACHE_DIR=/app/models/kaggle_cache

# Puerto expuesto por FastAPI / Uvicorn
EXPOSE 8000

# Health check para docker-compose y orquestadores (Kubernetes, etc.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Comando de arranque (producción: sin --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
