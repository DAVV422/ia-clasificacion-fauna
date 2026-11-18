# WWF Wildlife — IA de detección y clasificación de fauna

IA de la plataforma de cámaras trampa de WWF (equipo CETERIS404). Recibe fotos y videos de cámaras trampa, encuentra
los animales, personas y vehículos, y clasifica la especie de cada animal.

Usa dos modelos, en este orden:

1. **MegaDetector v5a** (YOLOv5): encuentra `animal`, `person` y `vehicle` en cada imagen o *frame*.
2. **SpeciesNet 4.0.2a** de Google: clasifica la especie de cada recorte de animal, con la taxonomía de
   *Wildlife Insights* (3538 taxones: clase, orden, familia, género y especie).

En los videos se analizan varios *frames* (los primeros 15 s como máximo), se descartan los repetidos y borrosos, y
el resultado es una **votación** entre los *frames* con animal. Si la evidencia no alcanza para la especie, el
resultado sube a género, familia, etc.

El repositorio tiene tres productos que comparten el mismo *pipeline* (`wwm_pipeline/`):

| Producto | Carpeta | Para qué |
|---|---|---|
| **API de la IA** | `api/` | Servicio HTTP interno que usa el back de la web. Contrato: `docs-interno/plan-implementacion/contratos/API_IA_BACK.md` |
| **ia-fake** | `fake/` | El mismo contrato con respuestas fijas, sin modelos. Para desarrollar el back y el front |
| **App de escritorio** | `desktop/` | Clasifica carpetas enteras sin internet (Windows, con instalador) |

---

## Contenido

- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Levantar la API con Docker](#-levantar-la-api-con-docker)
- [Levantar la API sin Docker](#️-levantar-la-api-sin-docker)
- [Configuración](#️-configuración-variables-de-entorno)
- [Endpoints](#-endpoints)
- [ia-fake](#-ia-fake-desarrollo-sin-modelos)
- [App de escritorio](#️-app-de-escritorio)
- [Tests y calidad](#-tests-y-calidad)
- [Licencia](#licencia)

---

## 📁 Estructura del proyecto

```
ia-deteccion/
├── api/                      # API de la IA (FastAPI)
│   ├── main.py               #   create_app(): middleware (API key, límites, cola), carga de modelos
│   ├── routes.py             #   GET /health, GET /catalog, POST /detect
│   ├── settings.py           #   variables de entorno
│   ├── security.py           #   X-API-Key
│   ├── inference.py          #   cola y timeout (un solo hilo de inferencia)
│   ├── envelope.py           #   respuesta {status, data, message, error, meta}
│   ├── errors.py             #   códigos IA_* → HTTP
│   └── logs.py               #   logs JSON con request_id
├── wwm_pipeline/             # Pipeline compartido (API y escritorio)
│   ├── pipeline.py           #   classify_file(): muestreo → detección → clasificación → votación
│   ├── sampling.py           #   qué frames del video se analizan
│   ├── detection.py          #   MegaDetector
│   ├── classification.py     #   SpeciesNet
│   ├── aggregation.py        #   votación entre frames
│   ├── contract.py           #   modelos Pydantic del contrato (DetectResult, ApiEnvelope, …)  [MIT]
│   ├── response.py           #   resultado del pipeline → DetectResult
│   └── formats.py            #   formatos aceptados                                             [MIT]
├── fake/                     # ia-fake (mismo contrato, sin modelos)                           [MIT]
├── desktop/                  # App de escritorio (Tkinter)
│   └── model_store.py        #   descarga y verificación (sha256) de los modelos
├── catalog/                  # Catálogo de especies de SpeciesNet (GET /catalog)              [MIT]
├── schemas/                  # JSON Schema del contrato (generado desde contract.py)          [MIT]
├── tools/                    # fetch_models.py, build_catalog.py, make_fixtures.py, …
├── third_party/              # Código de YOLOv5 que usa MegaDetector (para no descargarlo)
├── eval/                     # Evaluación de precisión con imágenes etiquetadas
├── tests/                    # pytest (api/, pipeline/, desktop/, contract/, fixtures/)
├── installer/, build.ps1     # Instaladores de la app de escritorio (Inno Setup)
├── deploy/                   # Ejemplo de docker-compose con el back
├── Dockerfile                # Imagen de la API
├── docker-compose.yml        # La API sola
├── requirements-api.txt      # Dependencias de la API, fijadas con hash
└── requirements-desktop.txt  # Dependencias del escritorio, fijadas con hash
```

`app/detector.py` y `app/config.py` son lo que queda del prototipo; los usa todavía la ventana del escritorio.

---

## 📋 Requisitos

| | Mínimo |
|---|---|
| Python | **3.11** (sin Docker) |
| RAM | ~1,5 GB para los modelos en CPU; **6 GB** recomendados para el contenedor (picos de video) |
| Disco | ~505 MB para los modelos; ~3 GB para la imagen de Docker |
| GPU | No hace falta (corre en CPU) |
| Internet | Solo para descargar los modelos **una vez**. La API no se conecta a nada al arrancar ni al clasificar |

---

## 🐳 Levantar la API con Docker

Requisitos: Docker Desktop (Windows/macOS) o Docker Engine (Linux).

```bash
git clone https://github.com/DAVV422/ia-clasificacion-fauna.git ia-deteccion
cd ia-deteccion

# 1. Construir la imagen (~10 min la primera vez)
docker compose build

# 2. Descargar los modelos al volumen "models" (UNA vez; ~505 MB, verificados por sha256)
docker compose run --rm ia python tools/fetch_models.py

# 3. Levantar
docker compose up -d

# 4. Comprobar: "status": "ok" cuando terminó de cargar los modelos (~30 s)
curl http://localhost:8001/api/v1/health
```

Una prueba de clasificación (en desarrollo la clave es `dev-key`):

```bash
curl -H "X-API-Key: dev-key" \
     -F "file=@foto.jpg" -F "file_name=foto.jpg" \
     http://localhost:8001/api/v1/detect
```

Comandos útiles:

```bash
docker compose logs -f ia     # logs (JSON, una línea por evento)
docker compose down           # detener (el volumen con los modelos queda)
docker compose up -d --build  # reconstruir tras cambiar el código
```

**Cómo es la imagen:**
- Instala solo desde `requirements-api.txt` con `--require-hashes` (reproducible).
- **No trae los modelos ni los descarga al arrancar.** Sin modelos, `/health` responde 503 `models_missing` y el
  log dice cómo prepararlos.
- Corre con un usuario sin privilegios, un solo *worker* de uvicorn (cada *worker* cargaría otra copia de los
  modelos) y un *healthcheck* a `/api/v1/health`.
- Con `ENVIRONMENT=production` (el valor de la imagen) **no arranca sin `AI_API_KEY`**. El `docker-compose.yml`
  de la raíz pone `development` y `dev-key` para probar en local.

**Junto al back:** en [deploy/docker-compose.example.yml](deploy/docker-compose.example.yml). Ahí la IA **no
publica puertos**: solo la ve el back, como `http://ia:8001`.

---

## 🛠️ Levantar la API sin Docker

Requiere **Python 3.11**. En Windows (PowerShell):

```powershell
git clone https://github.com/DAVV422/ia-clasificacion-fauna.git ia-deteccion
cd ia-deteccion
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-api.txt

# Los modelos (una vez)
.venv\Scripts\python tools\fetch_models.py --dir models-api

# Levantar
$env:MODELS_DIR = "models-api"
.venv\Scripts\uvicorn api.main:app --port 8001 --workers 1
```

En Linux/macOS:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements-api.txt
.venv/bin/python tools/fetch_models.py --dir models-api
MODELS_DIR=models-api .venv/bin/uvicorn api.main:app --port 8001 --workers 1
```

- La documentación interactiva (Swagger) queda en `http://localhost:8001/docs`.
- Si ya tienes la app de escritorio instalada, la API usa sus modelos (`%LOCALAPPDATA%\WWM\models`) y no hace falta
  `fetch_models.py` ni `MODELS_DIR`.
- `--workers 1` siempre: la concurrencia la maneja la cola de la API.

---

## ⚙️ Configuración (variables de entorno)

Todas tienen un valor por defecto. En producción solo es obligatoria `AI_API_KEY`.

| Variable | Por defecto | Qué hace |
|---|---|---|
| `AI_API_KEY` | `dev-key` (solo si no es `production`) | Clave que debe mandar el back en `X-API-Key`. **La misma** que `AI_API_KEY` del back |
| `ENVIRONMENT` | `development` (la imagen: `production`) | En `production`, sin `AI_API_KEY` la IA no arranca |
| `MODELS_DIR` | `/models` en Docker; si no, `%LOCALAPPDATA%\WWM\models` | Carpeta con `md_v5a.0.0.pt` y `speciesnet-4.0.2a/` |
| `TMP_DIR` | la temporal del sistema (`/tmp/ia` en Docker) | Dónde se guarda el archivo mientras se procesa. Se borra siempre al terminar |
| `MAX_FILE_MB` | `300` | Tamaño máximo del archivo → si se pasa, 413 `IA_FILE_TOO_LARGE` |
| `MAX_IMAGE_PIXELS` | `100000000` | Píxeles máximos de una imagen → 422 `IA_DECODE_FAILED` |
| `MAX_VIDEO_SECONDS` | `15` | Segundos de video que se analizan; el resto se ignora (`analysis.truncated_at_seconds`) |
| `INFERENCE_QUEUE_SIZE` | `2` | Archivos en espera además del que se procesa; con la cola llena → 503 `IA_BUSY` |
| `PROCESS_TIMEOUT_SECONDS` | `150` | Tiempo máximo por archivo → 504 `IA_PROCESSING_TIMEOUT`. **El back debe esperar más (180 s)** |
| `CATALOG_PATH` | `catalog/speciesnet_catalog.json` | Catálogo que sirve `GET /catalog` |
| `APP_VERSION` | `dev` | Aparece en `/health` (`version.app`) |
| `LOG_LEVEL` | `INFO` | Nivel de los logs |
| `LOG_FORMAT` | `json` | `json` (una línea por evento) o `console` |

---

## 🔌 Endpoints

Base: `http://<host>:8001/api/v1`. Todos piden el header **`X-API-Key`**, salvo `/health`.

Headers opcionales de correlación: `X-Request-ID` (si no viene, la IA genera uno) y `X-Flow` (el flujo del back,
p. ej. `FL-08`). Los dos vuelven en la respuesta (`meta.request_id`, `error.flow`) y en los logs; `X-Request-ID`
también como header.

### `GET /health`

Estado de la IA. Sin clave y **sin envoltura**. Responde **200** cuando los modelos están listos y **503** mientras
cargan, si faltan o si fallaron.

```json
{
  "status": "ok",
  "models_loaded": true,
  "queue": { "size": 2, "pending": 0 },
  "config": { "sampling": "10fps@0-2s+4fps/max60/hasta15s", "max_video_seconds": 15.0, "max_file_mb": 300.0 },
  "version": { "pipeline": "1.0.0", "detector": "MDv5a", "classifier": "speciesnet-4.0.2a", "app": "dev" }
}
```

`status`: `ok` · `loading` · `models_missing` · `error`.

### `POST /detect`

Clasifica **un** archivo. `multipart/form-data`:

| Campo | Obligatorio | Qué es |
|---|---|---|
| `file` | sí | La imagen o el video |
| `file_name` | sí | El nombre original (de él se puede leer la fecha de captura, p. ej. `CAM01_20260315_042210.mp4`) |
| `client_mtime` | no | Fecha de modificación en el cliente. La IA no la usa: si `capture.source` es `none`, la aplica el back |

Formatos: imágenes `.jpg .jpeg .png .bmp .webp .tiff`; videos `.mp4 .avi .mov .mkv .wmv .flv .webm`. Se revisa la
extensión **y** el contenido del archivo.

**Respuesta 200** (abreviada; `image_base64` es el *frame* de la evidencia en JPEG):

```json
{
  "status": "success",
  "message": "Archivo clasificado",
  "error": null,
  "meta": { "request_id": "a1b2c3d4e5f6", "timestamp": "2026-11-17T12:00:00+00:00" },
  "data": {
    "media": {
      "file_name": "CAM01_20260315_042210.mp4", "media_type": "video",
      "sha256": "293b6f9f…", "size_bytes": 18234567,
      "duration_seconds": 14.8, "fps": 30.0, "width": 1920, "height": 1080
    },
    "capture": { "captured_at": "2026-03-15T04:22:10", "source": "filename" },
    "result": {
      "category": "animal",
      "taxonomy": {
        "label_id": "9b7fea59-cfa3-499c-a420-bcf36277dcd8", "rank": "species",
        "class": "mammalia", "order": "carnivora", "family": "felidae", "genus": "panthera", "species": "onca",
        "scientific_name": "Panthera onca", "common_name": "jaguar", "label_raw": "9b7fea59-…;mammalia;…;jaguar"
      },
      "confidence": 0.94,
      "alternatives": [
        { "label_id": "9c564562-…", "scientific_name": "Puma concolor", "common_name": "puma", "rank": "species", "score": 0.03 }
      ]
    },
    "evidence": {
      "frame_index": 212, "timestamp_seconds": 7.07, "bbox": [0.31, 0.12, 0.78, 0.66],
      "detection_confidence": 0.91, "image_base64": "/9j/4AAQ…", "image_mime": "image/jpeg", "width": 960, "height": 540
    },
    "analysis": {
      "frames_sampled": 59, "frames_with_animal": 21,
      "frames_discarded": { "duplicate": 18, "blurry": 3, "cap": 0 },
      "votes": [ { "label_id": "9b7fea59-…", "scientific_name": "Panthera onca", "frames": 17, "score": 0.94 } ],
      "truncated_at_seconds": null, "low_evidence": false, "processing_ms": 48210
    },
    "model": { "detector": "MDv5a", "classifier": "speciesnet-4.0.2a", "sampling": "10fps@0-2s+4fps/max60/hasta15s", "pipeline_version": "1.0.0" }
  }
}
```

Cómo leerla:
- **`result.category`**: `animal` · `person` · `vehicle` · `blank` (nada detectado). Solo con `animal` hay
  `taxonomy`; en los otros casos es `null`.
- **`taxonomy.rank`**: hasta dónde llegó la clasificación (`species`, `genus`, `family`, `order`, `class`). Los
  niveles por debajo del `rank` vienen en `null`.
- **`capture.source`**: de dónde salió la fecha: `exif`, `container` (metadatos del video), `filename` o `none`.
- **`bbox`**: `[x1, y1, x2, y2]` normalizado (0 a 1) sobre el *frame* de la evidencia.
- **`analysis.low_evidence`**: `true` si el resultado se apoya en muy pocos *frames*; conviene revisarlo.
- **`analysis.truncated_at_seconds`**: `15.0` si el video era más largo y solo se analizaron los primeros 15 s.
- En las imágenes, `duration_seconds` y `fps` son `null` y `model.sampling` es `"image"`.
- Ejemplos completos de cada caso en [tests/fixtures/ai/](tests/fixtures/ai/) (animal, género, persona, vehículo,
  vacía, video truncado y errores).

**Errores:** misma envoltura, con `status: "error"` y `data: null`:

```json
{
  "status": "error",
  "data": null,
  "message": "La IA está ocupada; reintenta.",
  "error": { "code": "IA_BUSY", "flow": "FL-08", "details": [ { "code": "IA_BUSY", "context": { "queue_size": 2 } } ] },
  "meta": { "request_id": "a1b2c3d4e5f6", "timestamp": "2026-11-17T12:00:00+00:00" }
}
```

| HTTP | `error.code` | Cuándo | ¿Reintentar? |
|---|---|---|---|
| 400 | `IA_INVALID_FORMAT` | Falta `file` o `file_name` | No |
| 401 | `IA_UNAUTHORIZED` | Falta `X-API-Key` o es incorrecta | No (revisar la clave) |
| 413 | `IA_FILE_TOO_LARGE` | Más de `MAX_FILE_MB` (`context`: `max_mb`, `size_mb`) | No |
| 415 | `IA_INVALID_FORMAT` | Extensión o contenido no soportado (`context`: `accepted`, `reason`) | No |
| 422 | `IA_DECODE_FAILED` | Archivo dañado o imagen demasiado grande (`reason: too_many_pixels`) | No |
| 500 | `IA_INTERNAL` | Error inesperado (el detalle queda en el log de la IA, con el `request_id`) | Sí, una vez |
| 503 | `IA_MODEL_NOT_READY` | Los modelos todavía cargan | Sí, en unos segundos |
| 503 | `IA_BUSY` | La cola está llena | Sí, con espera |
| 504 | `IA_PROCESSING_TIMEOUT` | Pasó `PROCESS_TIMEOUT_SECONDS` | Según el caso |

### `GET /catalog`

Todos los taxones que puede devolver SpeciesNet, para que el back los tenga en su base. Responde con un header
`ETag`; si el back manda `If-None-Match` con ese valor y el catálogo no cambió, la respuesta es **304** sin cuerpo.

```json
{
  "status": "success",
  "message": "Catálogo",
  "error": null,
  "meta": { "request_id": "…", "timestamp": "…" },
  "data": {
    "classifier": "speciesnet-4.0.2a",
    "count": 3538,
    "sha256": "d3a7cbde…",
    "taxa": [
      { "label_id": "23a6f03b-…", "rank": "class", "class": "amphibia", "order": null, "family": null,
        "genus": null, "species": null, "scientific_name": "Amphibia", "common_name": "amphibian" }
    ]
  }
}
```

---

## 🧪 ia-fake (desarrollo sin modelos)

Cumple el mismo contrato sin cargar ningún modelo: responde en ~2 s y no necesita una máquina potente. El
escenario se elige con el **nombre del archivo** (`foto_blank.jpg` → vacía, `video_busy.mp4` → `IA_BUSY`, …).

```bash
pip install -r fake/requirements.txt
uvicorn fake.app:app --port 8001
```

Los escenarios, las variables y cómo usarlo con Docker están en [fake/README.md](fake/README.md). **Nunca se usa en
producción.**

---

## 🖥️ App de escritorio

Clasifica carpetas de fotos y videos sin internet y los copia o mueve a subcarpetas por especie. Para usuarios
finales se distribuye con instalador (Windows). Para correrla desde el código:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-desktop.txt
.venv\Scripts\python -m desktop.main              # o doble clic en Iniciar_App.bat
.venv\Scripts\python -m desktop.main --selftest   # verifica la instalación sin cargar los modelos
```

La primera vez descarga los modelos a `%LOCALAPPDATA%\WWM\models`. Los instaladores se arman con
`.\build.ps1 -Variant both -Version <versión>` (necesita PyInstaller e Inno Setup 6): uno **completo** (con los
modelos, funciona sin internet) y uno **estándar** (más liviano, descarga los modelos al primer uso).

---

## ✅ Tests y calidad

```bash
pip install -r requirements-dev.txt
python -m pytest -m "not integration"   # rápidos, sin modelos
python -m pytest                        # también los de integración, con los modelos reales
python -m ruff check . && python -m mypy
```

- `tests/api/test_contract_suite.py` compara la IA real con ia-fake: mismas claves y mismos códigos de error.
- Los tests de integración usan los modelos de `%LOCALAPPDATA%\WWM\models` (o `MODELS_DIR`) y se saltan si no
  están.

---

## Licencia

- **El repositorio** se distribuye bajo la **GNU Affero General Public License v3.0** ([`LICENSE`](LICENSE)). La app de escritorio incluye `ultralytics` (AGPL-3.0) y `yolov5` (GPL-3.0), que llegan como dependencias de SpeciesNet. Quien reciba la app puede pedir este código y reusarlo bajo la misma licencia.
- **El contrato con la web** está además bajo **MIT** ([`LICENSE-CONTRATO-MIT.txt`](LICENSE-CONTRATO-MIT.txt)), para que los repositorios del back y del front puedan copiarlo o ejecutarlo sin quedar sujetos a la AGPL. Son `wwm_pipeline/contract.py`, `package.py`, `enums.py`, `capture_parse.py`, `formats.py`, `schemas/`, `catalog/`, `tests/fixtures/` e `ia-fake`, y llevan `SPDX-License-Identifier: AGPL-3.0-only OR MIT`.
- **La web** (back y front) habla con la IA solo por HTTP y no incluye este código, salvo el contrato MIT. Su licencia la decide el equipo de forma independiente.
- **Los modelos** tienen sus propias licencias: MegaDetector, MIT; SpeciesNet, Apache-2.0.
