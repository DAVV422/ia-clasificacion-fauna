# WWF Wildlife — IA de detección y clasificación (MegaDetector + SpeciesNet)

Este repositorio tiene tres piezas que comparten el mismo pipeline (`wwm_pipeline/`):

1. **API de la IA para la web** (`api/`, Épica 2): FastAPI interna que usa el back. Recibe una foto o un video y
   devuelve las detecciones y la especie según el contrato `API_IA_BACK.md`.
2. **App de escritorio** (`desktop/`): clasifica carpetas de cámaras trampa sin internet.
3. **ia-fake** (`fake/`): servidor con respuestas fijas del mismo contrato, para desarrollar el back sin modelos.

El pipeline usa dos modelos:

- **MegaDetector v5a (YOLOv5):** encuentra animales, personas y vehículos.
- **SpeciesNet de Google:** clasifica la especie de cada animal (taxonomía de *Wildlife Insights*).

---

## 📁 Estructura del proyecto

```
ia-deteccion/
├── api/                  # API de la IA (FastAPI): rutas, seguridad, cola de inferencia, logs
├── wwm_pipeline/         # Pipeline compartido: muestreo, detección, clasificación, contrato
├── desktop/              # App de escritorio (Tkinter)
├── fake/                 # ia-fake (HU-IA-22)
├── catalog/              # Catálogo de especies de SpeciesNet (GET /api/v1/catalog)
├── schemas/              # JSON Schema del contrato
├── tools/                # Scripts: fetch_models.py, build_catalog.py, make_fixtures.py, ...
├── deploy/               # Ejemplo de compose con el back
├── Dockerfile            # Imagen de la IA (HU-IA-06)
├── docker-compose.yml    # La IA sola (desarrollo)
├── requirements-api.txt  # Dependencias fijadas con hash (la imagen usa --require-hashes)
└── tests/
```

---

## 🐳 API de la IA con Docker

Requisitos: Docker Engine o Docker Desktop y unos **6 GB de RAM** para el contenedor.

La imagen **no trae los modelos ni los descarga al arrancar**. Se preparan una vez en el volumen `models`
(~505 MB desde el release del repo, verificados por sha256):

```bash
docker compose build
docker compose run --rm ia python tools/fetch_models.py   # una vez
docker compose up -d
curl http://localhost:8001/api/v1/health                   # "models": "ready" cuando terminó de cargar
```

Prueba de una detección (la clave va en `X-API-Key`):

```bash
curl -H "X-API-Key: dev-key" -F "file=@foto.jpg" http://localhost:8001/api/v1/detect
```

Variables principales (todas tienen valor por defecto, salvo `AI_API_KEY` en producción):

| Variable | Por defecto | Para qué |
|---|---|---|
| `AI_API_KEY` | `dev-key` solo en desarrollo | La misma clave que usa el back. En `production`, sin ella la IA no arranca |
| `ENVIRONMENT` | `production` en la imagen | `development` permite la clave de desarrollo |
| `MODELS_DIR` | `/models` | Volumen con los modelos |
| `MAX_FILE_MB` | `300` | Tamaño máximo del archivo (413 si se pasa) |
| `MAX_VIDEO_SECONDS` | `15` | Segundos de video que se analizan |
| `INFERENCE_QUEUE_SIZE` | `2` | Pedidos en espera; con la cola llena responde `IA_BUSY` (503) |
| `PROCESS_TIMEOUT_SECONDS` | `150` | Tiempo máximo por archivo. El back debe esperar más (180 s) |

En producción la IA **no publica puertos**: solo la ve el back por la red interna de Docker. El ejemplo con el back
está en [deploy/docker-compose.example.yml](deploy/docker-compose.example.yml).

### Endpoints

| Método y ruta | Clave | Qué hace |
|---|---|---|
| `GET /api/v1/health` | no | Estado de los modelos, cola y versiones (sin envoltura) |
| `GET /api/v1/catalog` | sí | Catálogo de especies (`ETag`, responde 304 si no cambió) |
| `POST /api/v1/detect` | sí | `multipart/form-data` con `file`: devuelve `DetectResult` dentro de la envoltura `{status, data, message, error, meta}` |

Los errores tienen código `IA_*` (`IA_INVALID_FORMAT`, `IA_FILE_TOO_LARGE`, `IA_BUSY`, `IA_TIMEOUT`, ...).
El contrato completo está en `docs-interno/plan-implementacion/contratos/API_IA_BACK.md`.

---

## 🛠️ API de la IA sin Docker (desarrollo)

Requiere **Python 3.11**.

```bash
python -m venv .venv
.venv/Scripts/pip install --require-hashes -r requirements-api.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python tools/fetch_models.py --dir models-api            # una vez
MODELS_DIR=models-api .venv/Scripts/uvicorn api.main:app --port 8001 --workers 1
```

Va con `--workers 1`: cada worker cargaría su propia copia de los modelos (~1,5 GB).

---

## 🖥️ Aplicación de Escritorio (GUI)

El proyecto incluye una **interfaz gráfica de escritorio (GUI)** diseñada en blanco y negro, pensada para procesar carpetas enteras de imágenes y videos locales sin necesidad de levantar el servidor web ni usar Docker.

### Cómo ejecutar la App de Escritorio
Si estás en Windows y tienes Python instalado:
1. Simplemente haz doble clic en el archivo **`Iniciar_App.bat`**.
2. O desde la terminal, con tu entorno virtual activo: `python desktop_app.py`

La aplicación te permite:
- Seleccionar una **carpeta de entrada** con múltiples fotos y videos.
- Seleccionar una **carpeta de salida**.
- Elegir si quieres **Copiar** o **Mover** los archivos.
- Al procesar, creará subcarpetas automáticas con los nombres de las especies detectadas (ej. `salida/Panthera_onca/archivo.mp4`).

### 💻 Requisitos Mínimos del Sistema
Para que la aplicación funcione correctamente (ya sea la API o el Escritorio):
- **Sistema Operativo:** Windows 10/11, macOS, o Linux.
- **Procesador (CPU):** Intel Core i5 / AMD Ryzen 5 o superior.
- **Memoria RAM:** Mínimo **6 GB a 8 GB libres** (MegaDetector y SpeciesNet ocupan bastante memoria en la etapa de carga).
- **Almacenamiento:** ~500 MB libres para la descarga automática de los modelos de IA.
- *(Opcional)* **GPU:** Tarjeta gráfica NVIDIA compatible con CUDA para acelerar el procesamiento de video 5x a 10x veces.

---

## 📦 Empaquetar como Ejecutable (.exe) Independiente

Si necesitas distribuir la aplicación a usuarios que no saben usar la consola ni tienen Python instalado, puedes empaquetarla usando **PyInstaller**. Esto generará un programa `.exe` autónomo.

### Limitaciones Importantes:
1. **Tamaño del archivo:** Como empaquetarás PyTorch, OpenCV y los modelos, el tamaño final del programa superará los **2.5 GB a 4 GB**.
2. **Plataforma:** El `.exe` generado en Windows **solo funcionará en Windows**. Para crear una versión de Mac, debes correr el empaquetado desde una Mac.
3. **Falsos Positivos Antivirus:** Windows Defender suele bloquear ejecutables de PyInstaller que no están firmados digitalmente. Podría ser necesario decirle al antivirus que permita la aplicación.
4. **Arranque en frío:** Si decides usar la opción `--onefile` (un solo `.exe`), el programa tardará minutos en abrir porque descomprime gigabytes temporalmente. Por eso se recomienda el modo "carpeta" (`--onedir`).

### Instrucciones para empaquetar en Windows:

1. Activa tu entorno virtual e instala PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Ejecuta el comando de empaquetado (modo carpeta sin consola negra de fondo):
   ```bash
   pyinstaller --noconfirm --onedir --windowed --add-data "public;public" --add-data "app;app" --name "WWF_Clasificador" desktop_app.py
   ```
- Ejecuta el comando de empaquetado (un solo archivo)
   ```bash
   pyinstaller --noconfirm --onefile --windowed --icon="public/logo.ico" --splash="public/splash.png" --collect-all cloudpathlib --collect-all speciesnet --collect-all ultralytics --collect-all onnx2torch desktop_app.py
   ```
3. Al finalizar, ve a la nueva carpeta `dist/WWF_Clasificador/`.
4. El archivo **`WWF_Clasificador.exe`** es tu aplicación. Puedes comprimir toda la carpeta `WWF_Clasificador` en un `.zip` y compartirla.

## Licencia

- **El repositorio** se distribuye bajo la **GNU Affero General Public License v3.0** ([`LICENSE`](LICENSE)). La app de escritorio incluye `ultralytics` (AGPL-3.0) y `yolov5` (GPL-3.0), que llegan como dependencias de SpeciesNet. Quien reciba la app puede pedir este código y reusarlo bajo la misma licencia.
- **El contrato con la web** está además bajo **MIT** ([`LICENSE-CONTRATO-MIT.txt`](LICENSE-CONTRATO-MIT.txt)), para que los repositorios del back y del front puedan copiarlo o ejecutarlo sin quedar sujetos a la AGPL. Son `wwm_pipeline/contract.py`, `package.py`, `enums.py`, `capture_parse.py`, `schemas/`, `tests/fixtures/` e `ia-fake`, y llevan `SPDX-License-Identifier: AGPL-3.0-only OR MIT`.
- **La web** (back y front) habla con la IA solo por HTTP y no incluye este código, salvo el contrato MIT. Su licencia la decide el equipo de forma independiente.
- **Los modelos** tienen sus propias licencias: MegaDetector, MIT; SpeciesNet, Apache-2.0.
