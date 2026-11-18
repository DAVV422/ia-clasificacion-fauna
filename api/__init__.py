"""API de la IA para la web (Épica 2): ``uvicorn api.main:app --port 8001``.

Cumple ``docs-interno/plan-implementacion/contratos/API_IA_BACK.md``. La clasificación es la de ``wwm_pipeline`` con el
preset ``web``; aquí solo está la capa HTTP (envoltura, errores, seguridad, cola, logs). Reemplazó a ``app/main.py`` (borrada en HU-IA-06),
la API del prototipo (E2-D1).
"""
