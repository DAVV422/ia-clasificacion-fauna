"""API key (HU-IA-30, IA §1): ``X-API-Key`` obligatoria en todo ``/api/v1/*`` salvo ``/health``.

Se verifica en el *middleware*, **antes** de leer el cuerpo: un archivo de 300 MB sin clave no se llega a recibir.
La comparación es en tiempo constante (``hmac.compare_digest``). La clave nunca va al log.
"""

import hmac

API_PREFIX = "/api/v1/"
OPEN_PATHS = frozenset({"/api/v1/health"})
DEV_KEY = "dev-key"  # la misma que ia-fake en desarrollo


class MissingApiKey(RuntimeError):
    """``ENVIRONMENT=production`` sin ``AI_API_KEY``: la app no arranca."""


def is_protected(path: str) -> bool:
    return path.startswith(API_PREFIX) and path.rstrip("/") not in OPEN_PATHS


def valid_key(received: str, expected: str) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(received.encode("utf-8"), expected.encode("utf-8"))


def check_settings(environment: str, api_key: str) -> None:
    if environment == "production" and not api_key:
        raise MissingApiKey("AI_API_KEY es obligatoria con ENVIRONMENT=production (la comparte el back, SEC-05)")
