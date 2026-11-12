# YOLOv5 (copia local para cargar MegaDetector sin internet)

- **De dónde:** `ultralytics/yolov5` (rama `master`), tal como lo bajaba `torch.hub.load("ultralytics/yolov5", "custom", ...)` a `~/.cache/torch/hub/ultralytics_yolov5_master`. Es la misma copia que usó la app desde la fase 0 (YOLOv5 informa la fecha 2026-08-17).
- **Licencia:** AGPL-3.0 (`LICENSE` de esta carpeta). El repo `ia-deteccion` también es AGPL-3.0.
- **Por qué está aquí** (HU-DS-11, F4-D3): sin esta copia, `torch.hub` baja el código de GitHub en cada perfil nuevo, y la app no carga sin internet. El paquete `yolov5` de pip no sirve: al cargar un modelo consulta Hugging Face y la API de GitHub.
- **Qué se copió:** solo lo que usa `hubconf.custom`:
  - `hubconf.py`, `export.py` y `requirements.txt`;
  - `models/` y `utils/`;
  - `data/*.yaml` y `data/hyps/`.

  Se dejaron afuera el entrenamiento, la segmentación, la clasificación, las imágenes de ejemplo y los tutoriales.
- **Cómo se usa:** `torch.hub.load(<esta carpeta>, "custom", path=md_v5a.0.0.pt, source="local")`, en `wwm_pipeline/models.py`. Si falta un archivo, `hubconf` devuelve un modelo sin `AutoShape`, y `ModelRegistry` lo detecta y falla con `MODEL_LOAD_FAILED`.
- **No se modifica.** Si hay que actualizarla, se reemplaza la carpeta entera y se corren los tests de equivalencia (`tests/pipeline/test_legacy_equivalence.py`, `tests/desktop/test_desktop_unchanged.py`).
