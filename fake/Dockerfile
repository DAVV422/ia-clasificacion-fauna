# ia-fake · IA simulada para desarrollo (HU-IA-22)
# Construir desde la raíz del repo ia-deteccion:
#   docker build -f fake/Dockerfile -t wwm-ia-fake:1 .
FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY fake/requirements.txt fake/requirements.txt
RUN pip install --no-cache-dir -r fake/requirements.txt

COPY wwm_pipeline/__init__.py wwm_pipeline/enums.py wwm_pipeline/contract.py \
     wwm_pipeline/catalog_schema.py wwm_pipeline/capture_parse.py wwm_pipeline/
COPY tests/fixtures/ai tests/fixtures/ai
COPY tests/fixtures/catalog tests/fixtures/catalog
COPY fake/__init__.py fake/app.py fake/scenarios.py fake/

RUN adduser -D -u 10001 fake
USER fake

EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8001/api/v1/health')"
CMD ["uvicorn", "fake.app:app", "--host", "0.0.0.0", "--port", "8001"]
