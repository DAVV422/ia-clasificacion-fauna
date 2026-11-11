# ia-fake · IA simulada para desarrollo

Servicio que **cumple el contrato** de la IA (`docs-interno/plan-implementacion/contratos/API_IA_BACK.md`) sin ejecutar ningún modelo. Sirve para desarrollar y probar el back (subida, cola, *worker*) y el front contra el back real **sin esperar la IA nueva** ni necesitar una máquina potente. HU-IA-22.

- Implementa `POST /api/v1/detect`, `GET /api/v1/health` y `GET /api/v1/catalog`, con la envoltura estándar y `X-API-Key` obligatoria (salvo en `/health`).
- Devuelve el `sha256` y el tamaño **reales** del archivo recibido, y la fecha de captura leída del nombre del archivo, así funcionan la detección de duplicados y `captured_at` en el back.
- **Nunca debe usarse en producción.**

## Escenarios según el nombre del archivo
Se compara en minúsculas; gana la primera regla que coincide.

| El nombre contiene | Respuesta |
|---|---|
| `_error_decode` | 422 `IA_DECODE_FAILED` |
| `_toolong` | 200, puma en un video de 46,6 s con `analysis.truncated_at_seconds = 15.0` (la IA analizó solo los primeros 15 s; F1-D10). Ya **no** devuelve `IA_DURATION_EXCEEDED` |
| `_badformat` | 415 `IA_INVALID_FORMAT` |
| `_busy` | 503 `IA_BUSY` |
| `_notready` | 503 `IA_MODEL_NOT_READY` |
| `_timeout` | Espera `FAKE_TIMEOUT_SECONDS` y responde 504 `IA_PROCESSING_TIMEOUT` |
| `_crash` | 500 `IA_INTERNAL` |
| `_badcontract` | 200 con un `data` **fuera de contrato** (sin `result`), para probar `AI_BAD_RESPONSE` |
| `_slow` | Espera `FAKE_SLOW_SECONDS` y responde el jaguar |
| `_blank` | 200, `category=blank` |
| `_person` · `_vehicle` | 200, persona o vehículo |
| `_genus` | 200, *Dasyprocta* (solo hasta género) |
| `_image`, o una extensión de imagen | 200, tapir en una imagen |
| cualquier otro | 200, jaguar en un video (`CAM01_20260315_042210.mp4` → `captured_at=2026-03-15T04:22:10`) |

## Variables
| Variable | Por defecto | Uso |
|---|---|---|
| `AI_API_KEY` | `dev-key` | La misma que `AI_API_KEY` del back |
| `FAKE_DELAY_SECONDS` | `2` | Demora de cada respuesta normal |
| `FAKE_SLOW_SECONDS` | `60` | Demora del escenario `_slow` |
| `FAKE_TIMEOUT_SECONDS` | `200` | Demora del escenario `_timeout` (mayor que el timeout del back, 180 s) |
| `FAKE_CONCURRENCY` / `FAKE_QUEUE_SIZE` | `1` / `2` | Igual que la IA real: con 3 peticiones en curso, la cuarta recibe `IA_BUSY` |
| `FAKE_NOT_READY_SECONDS` | `0` | Durante los primeros N s, `/health` da 503 y `/detect` da `IA_MODEL_NOT_READY` |
| `LOG_LEVEL` | `INFO` | Logs JSON en stdout (`request_id` y `flow` tomados de los headers) |

## Cómo levantarlo
**Sin Docker** (desde la raíz de `ia-deteccion`):
```bash
pip install -r fake/requirements.txt
uvicorn fake.app:app --port 8001
```

**Con Docker:**
```bash
docker build -f fake/Dockerfile -t wwm-ia-fake:1 .
docker run --rm -p 8001:8001 -e AI_API_KEY=dev-key wwm-ia-fake:1
```

**En el `docker-compose.dev.yml` del back (BE-60):**
```yaml
  ia-fake:
    build:
      context: ../ia-deteccion
      dockerfile: fake/Dockerfile
    environment:
      AI_API_KEY: ${AI_API_KEY}
      FAKE_DELAY_SECONDS: "2"
    networks: [internal]
```
En el back: `AI_BASE_URL=http://ia-fake:8001/api/v1`.

## Prueba rápida
```bash
curl -s -H "X-API-Key: dev-key" -F "file=@cualquier.mp4" -F "file_name=CAM01_20260315_042210.mp4" \
     http://localhost:8001/api/v1/detect
```

## Tests
```bash
python -m pytest fake/tests
```
