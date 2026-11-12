"""Mensajes al usuario por código de error (app-escritorio/02_FLUJOS E-11).

Un test verifica que cada código de ``ERROR_CODES`` tenga su mensaje.
"""

MESSAGES = {
    "NETWORK_UNAVAILABLE": "No hay conexión. Necesitas internet solo esta primera vez para descargar el modelo.",
    "DOWNLOAD_INTEGRITY_FAILED": "La descarga del modelo está dañada. Revisa tu conexión y vuelve a intentar.",
    "REQUIREMENTS_NOT_MET": "Tu equipo no es compatible con el modelo de IA: {motivo}.",
    "MODEL_LOAD_FAILED": "No se pudo cargar el modelo. Reinstala la aplicación.",
    "DISK_SPACE_LOW": "Espacio insuficiente en el disco de destino. Se detuvo el análisis.",
    "DESTINATION_NOT_WRITABLE": "No se puede escribir en la carpeta de destino. Elige otra.",
    "SAME_OR_NESTED_FOLDER": "Elige una carpeta de destino distinta y fuera de la carpeta de origen.",
    "IA_DECODE_FAILED": "No se pudo leer el archivo (puede estar dañado). Queda en la carpeta de origen.",
    "IA_INVALID_FORMAT": "Formato no compatible. Queda en la carpeta de origen.",
    "FILE_LOCKED": "El archivo está en uso por otro programa. Queda en la carpeta de origen.",
    "FILE_MOVE_VERIFY_FAILED": "No se pudo verificar la copia; el original no se movió.",
    "SOURCE_NOT_DELETED": "Se copió, pero no se pudo borrar el original (está en uso). Bórralo a mano si quieres.",
    "EVIDENCE_WRITE_FAILED": "No se pudo guardar el fotograma; el archivo queda en la carpeta de origen.",
    "DESKTOP_ACTION_FAILED": "No se pudo copiar o mover el archivo. Queda en la carpeta de origen.",
    "IA_INTERNAL": "No se pudo analizar el archivo por un error inesperado. Queda en la carpeta de origen.",
    "OUT_OF_MEMORY": "El equipo se quedó sin memoria. Cierra otras aplicaciones y vuelve a intentar.",
    "EXPORT_INVALID": "No se pudo generar el paquete para la web.",
    "INTERNAL_ERROR": "Ocurrió un error inesperado (código de referencia {ref}).",
}

ERROR_CODES = frozenset(MESSAGES)


def message_for(code: str, **values) -> str:
    template = MESSAGES.get(code, MESSAGES["INTERNAL_ERROR"])
    try:
        return template.format(**values)
    except (KeyError, IndexError):
        return template.replace("{motivo}", "").replace("{ref}", values.get("ref", "-"))
