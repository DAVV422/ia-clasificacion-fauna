# WWF Wildlife Detection & Species Classification API (MegaDetector + SpeciesNet)

Este es un backend de producción desarrollado con **FastAPI** que utiliza un pipeline de dos etapas para procesar imágenes y videos de cámaras trampa:
1. **MegaDetector v5a (YOLOv5):** Detecta la presencia de animales (así como personas y vehículos) y extrae sus coordenadas.
2. **SpeciesNet de Google (EfficientNetV2-M):** Cuando se detecta un `animal`, recorta su área y clasifica automáticamente la **especie exacta** (ej. `"Panthera onca"`, `"Odocoileus virginianus"`, etc.) de entre más de 2,000 categorías globales de la plataforma *Wildlife Insights*.

Cuando se detecta un animal, el backend recorta el área exacta de la imagen/frame y lo almacena localmente en carpetas organizadas por categoría: `detections/{categoria}/{imagen_uuid}.jpg`. Para videos, analiza cuadros clave (muestreo configurable) y reporta la marca de tiempo (segundos) exacta de su aparición junto con la especie y confianza.

---

## 🚀 Características

- **Pipeline de Inferencia en 2 Etapas:**
  - Inferencia rápida de MegaDetector v5a para localizar fauna (`animal`, `person`, `vehicle`).
  - Clasificación de especies global con **SpeciesNet de Google** en los recortes de animales.
- **Autodescarga Automática de Modelos:**
  - Descarga los pesos del detector (`md_v5a.0.0.pt` - ~144MB).
  - Descarga automáticamente el clasificador SpeciesNet desde **Kaggle Hub** (~200MB) en el primer arranque. **No requiere** credenciales ni cuentas API de Kaggle.
- **Muestreo de Frames de Video (Fast-Inference):** Permite procesar videos de ~10 segundos de manera ágil (muestreo por defecto de 2 frames por segundo), evitando timeouts en HTTP.
- **Recorte Automático y Guardado Local:** Almacena los recortes de las detecciones en `detections/{categoria}/{uuid}.jpg` para auditorías o integraciones.
- **Servidor de Archivos Estáticos:** La carpeta de detecciones se expone en `/detections/` para que los recortes puedan previsualizarse directamente desde la web.
- **Limpieza de Temporales Integrada:** Utiliza `BackgroundTasks` de FastAPI para garantizar la eliminación de archivos temporales de video del servidor tras procesarse la petición.

---

## 📁 Estructura del Proyecto

```
ia-deteccion/
├── app/
│   ├── __init__.py
│   ├── main.py            # Inicialización de FastAPI y endpoints
│   ├── config.py          # Variables de entorno y configuraciones de thresholds/rutas
│   ├── detector.py        # Inferencia conjunta con MegaDetector v5a y Google SpeciesNet
│   ├── utils.py           # Descarga de pesos y gestión de archivos temporales
│   └── schemas.py         # Modelos Pydantic para validación de datos
├── models/                # Pesos de MegaDetector
├── detections/            # Recortes organizados localmente por categoría (animal, person, vehicle)
├── temp/                  # Almacenamiento temporal para videos en procesamiento
├── requirements.txt       # Listado de dependencias de Python (incluye speciesnet)
└── README.md              # Guía de uso y documentación (este archivo)
```

---

## 🐳 Levantar con Docker (Recomendado para Producción)

Esta es la forma más sencilla y reproducible de ejecutar la API. No necesitas instalar Python, PyTorch ni ninguna dependencia manualmente.

### Requisitos Previos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) o Docker Engine (Linux).
- Al menos **6 GB de RAM** libres para el contenedor (MegaDetector + SpeciesNet en CPU).

---

### Opción A: Docker Compose (Recomendado)

```bash
# 1. Construir la imagen (solo la primera vez o tras cambios en el código)
docker compose build

# 2. Levantar el servicio en segundo plano
docker compose up -d

# 3. Ver logs en tiempo real
docker compose logs -f wwf-api
```

El servidor estará disponible en **`http://localhost:8000`**.

> **Primera ejecución:** Al arrancar por primera vez, el contenedor descargará automáticamente los pesos de MegaDetector (~144 MB) y el clasificador SpeciesNet desde Kaggle Hub (~200 MB). Esto puede tardar **3-8 minutos** según la velocidad de tu conexión. Los pesos quedan guardados en la carpeta local `./models/` y no se vuelven a descargar.

**Comandos útiles:**

```bash
# Detener el servicio
docker compose down

# Reiniciar el servicio
docker compose restart wwf-api

# Ver el estado del healthcheck
docker compose ps

# Reconstruir y levantar en un solo paso (útil tras cambiar el código)
docker compose up -d --build
```

---

### Opción B: Docker run (Manual)

Si prefieres no usar docker-compose, puedes construir y ejecutar la imagen directamente:

```bash
# 1. Construir la imagen
docker build -t wwf-wildlife-detection:latest .

# 2. Crear carpetas locales para datos persistentes (si no existen)
mkdir -p models detections temp

# 3. Ejecutar el contenedor
docker run -d \
  --name wwf-wildlife-api \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/detections:/app/detections" \
  -v "$(pwd)/temp:/app/temp" \
  -e TORCH_HOME=/app/models/torch_hub \
  -e KAGGLE_CACHE_DIR=/app/models/kaggle_cache \
  --memory="6g" \
  --restart unless-stopped \
  wwf-wildlife-detection:latest
```

> En PowerShell (Windows), reemplaza `$(pwd)` por `${PWD}`.

**Ver logs:**
```bash
docker logs -f wwf-wildlife-api
```

**Detener y eliminar el contenedor:**
```bash
docker stop wwf-wildlife-api && docker rm wwf-wildlife-api
```

---

### Estructura de Archivos Docker

```
ia-deteccion/
├── Dockerfile           # Build multi-stage (builder + runtime)
├── docker-compose.yml   # Orquestación del servicio con volúmenes y healthcheck
└── .dockerignore        # Excluye venv, modelos y caché del contexto de build
```

---

## 🛠️ Instalación Local (Sin Docker)

### Requisitos Previos
- **Python 3.10** (Recomendado)
- **Git** instalado (requerido por PyTorch Hub).

### Paso 1: Activar entorno virtual
En Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
En Linux / macOS:
```bash
source venv/bin/activate
```

### Paso 2: Instalar dependencias actualizadas
```bash
pip install -r requirements.txt
```

---


## ⚙️ Cómo Ejecutar el Servidor

Inicia el servidor de desarrollo:

```bash
uvicorn app.main:app --reload
```

Al iniciar por primera vez:
1. Se descargará el modelo de MegaDetector en `/models` si no está presente.
2. **Kaggle Hub** descargará automáticamente el modelo SpeciesNet (`pyTorch/v4.0.2a/1`). Verás barras de progreso en tu terminal.
3. El servidor cargará ambos modelos en memoria ("Model Warmup").
4. Estará disponible en: **`http://localhost:8000`**

---

## 🧪 Pruebas de la API

### 1. Documentación Interactiva (Swagger UI)
Accede a la UI interactiva:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

Sube fotos o videos directamente desde el navegador en el endpoint `/api/v1/detect`.

### 2. Prueba con cURL
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@c:/ruta/de/tu/test_animal.mp4"
```

---

## 📄 Estructura de la Respuesta JSON (Ejemplo)

La API retornará una respuesta organizada indicando la categoría, la especie clasificada por Google SpeciesNet y los recortes correspondientes:

```json
{
  "filename": "camera_trap_10s.mp4",
  "media_type": "video",
  "duration_seconds": 10.15,
  "total_detections_count": 2,
  "categories_found": [
    "animal"
  ],
  "results": [
    {
      "timestamp_seconds": 2.5,
      "detections": [
        {         
          "confidence": 0.874,
          "species": "Panthera onca",          
          "crop_url": "/detections/animal/4c919a3b-2401-443b-85fa-71d5334de320.jpg"
        }
      ]
    },
    {
      "timestamp_seconds": 3.0,
      "detections": [
        {          
          "confidence": 0.912,
          "species": "Panthera onca",
          "crop_url": "/detections/animal/7f12a64c-83b2-4d2d-94bb-1845bb08a1c9.jpg"
        }
      ]
    }
  ]
}
```

---

## 🛠️ Personalización (Parámetros)

En `app/config.py` puedes ajustar variables de rendimiento clave:
- **`CONFIDENCE_THRESHOLD = 0.30`**: Umbral de detección mínimo de MegaDetector.
- **`SPECIESNET_CONFIDENCE_THRESHOLD = 0.15`**: Umbral mínimo para reportar la clasificación de especie de SpeciesNet.
- **`VIDEO_FPS_SAMPLING = 2.0`**: Cantidad de frames analizados por segundo de video (ej. `1.0` es ideal para CPU, `5.0` para mayor precisión de movimiento).
- **`SPECIESNET_MODEL_NAME`**: Identificador del clasificador en Kaggle Hub (por defecto: `kaggle:google/speciesnet/pyTorch/v4.0.2a/1`).
