"""Exporta el JSON Schema de los modelos compartidos a ``schemas/``.

El back puede validar contra estos archivos (o generar sus modelos con
``datamodel-code-generator``). Un test falla si no están sincronizados.

Uso:
    python -m tools.export_schemas
"""

import json
from pathlib import Path
from typing import Dict

from wwm_pipeline.catalog_schema import CatalogTaxon
from wwm_pipeline.contract import CONTRACT_VERSION, DetectEnvelope, DetectResult
from wwm_pipeline.package import Manifest, PackageLine

OUT = Path(__file__).resolve().parent.parent / "schemas"
MODELS = {
    "detect_result": DetectResult,
    "detect_envelope": DetectEnvelope,
    "package_manifest": Manifest,
    "package_line": PackageLine,
    "catalog_taxon": CatalogTaxon,
}


def render() -> Dict[str, str]:
    out = {}
    for name, model in MODELS.items():
        schema = model.model_json_schema(by_alias=True)
        schema["$comment"] = (
            f"wwm_pipeline contract {CONTRACT_VERSION} — generado por tools/export_schemas.py, no editar"
        )
        out[f"{name}.schema.json"] = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for file_name, text in render().items():
        (OUT / file_name).write_text(text, encoding="utf-8")
        print(f"ok  schemas/{file_name}")


if __name__ == "__main__":
    main()
