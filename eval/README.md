# Evaluación de precisión (HU-IA-10 → IA-17)

Mide qué tan bien clasifica una estrategia sobre un **set de validación etiquetado a mano**, para demostrar la mejora de la nueva extracción de fotogramas (Producto 1) y calibrarla.

## Estructura
| Archivo | Qué es |
|---|---|
| `labels.csv` | La verdad: una fila por archivo de prueba (ver [LABELING.md](LABELING.md)) |
| `legacy/` | **Copia congelada** del prototipo v1 (3 fotogramas, salida temprana). Es la **línea base**; no se modifica. |
| `legacy/expected.json` | Salida del prototipo sobre unas muestras fijas, para comprobar que la reestructura (HU-IA-40) no cambia el comportamiento |
| `eval.py` | Ejecuta una estrategia sobre el set y escribe el reporte |
| `metrics.py` · `report.py` | Métricas por nivel (especie, género, familia), falsos negativos y positivos, confusión, tiempos |
| `label_helper.py` | Ayuda para etiquetar |
| `reports/` | Los reportes (`<fecha>_<config>.md` y `.json`), versionados |
| `runs/` | Predicciones por archivo (no se versionan) |

## Datos
Los videos e imágenes **no van al repo** (peso y privacidad). Viven en una carpeta local indicada por `EVAL_DATA_DIR`. En el repo solo quedan `labels.csv` (sin rutas) y los reportes.

## Partición calib / test
`split_for(sha256)` asigna cada archivo a `calib` (≈30 %) o `test` (≈70 %) según un hash de su sha256: es **estable** (no cambia al agregar archivos) y no depende del orden. Se **calibra** con `calib` (umbrales, muestreo) y se **reporta** con `test`, para no sobreajustar.

## Uso
```bash
set EVAL_DATA_DIR=D:\wwm-eval\videos
python -m eval.label_helper                              # 1. etiquetar
python -m pytest eval/tests                              # 2. verificar labels.csv y las métricas
python -m eval.eval --config baseline --split test       # 3. línea base (prototipo v1)
# fase 1: python -m eval.eval --config desktop --split test
```
Requiere los modelos (MegaDetector en `models/md_v5a.0.0.pt`, SpeciesNet 4.0.2a en la caché de kagglehub) y las versiones del build del prototipo: **torch 2.8.0, speciesnet 5.0.4, ultralytics 8.4.58, numpy 1.26.4, opencv 4.11, pillow 10.4**.

## Estado del set
| | Archivos |
|---|---|
| Total | _(completar al etiquetar)_ |
| Con animal / vacíos / persona / vehículo | _(completar)_ |
| Especies distintas | _(completar)_ |

## Muestras de `legacy/expected.json`
`expected.json` se generó con 5 muestras (4 recortes de animales de pruebas anteriores y un video sintético de 10 s armado con esos recortes) que están **fuera del repo**, en `wwf/eval-data/legacy_samples/` (`LEGACY_SAMPLES_DIR`). **No borrarlas:** el test de equivalencia de HU-IA-40 las necesita. Si se pierden, regenerar con `python -m eval.legacy.make_expected` **desde un commit anterior a HU-IA-40**.

## ⚠️ Auto-actualización de YOLOv5
Al cargar MegaDetector con `torch.hub`, el código de YOLOv5 (y el paquete `ultralytics`) **instalan paquetes por su cuenta** si creen que faltan (por ejemplo, actualizan `ultralytics`). Para medir se desactiva con `YOLOv5_AUTOINSTALL=false` y `YOLO_AUTOINSTALL=false` (`legacy/config_legacy.py`); no cambia la clasificación (verificado: mismo `expected.json` con y sin la actualización). En la app de escritorio esto se resuelve de fondo en HU-DS-11 (YOLOv5 local, sin red).
