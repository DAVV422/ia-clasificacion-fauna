"""Motor de clasificación común de WWM (API web + app de escritorio).

Este ``__init__`` no importa nada pesado: ``wwm_pipeline.contract``,
``wwm_pipeline.package`` y ``wwm_pipeline.catalog_schema`` solo dependen de
pydantic y de la biblioteca estándar, para que el back pueda usarlos sin
instalar torch ni OpenCV.
"""

__version__ = "1.0.0"
