# Fixtures compartidos (HU-IA-21)

Datos de prueba para el back (Job) y el front (Miguel). Se generan con herramientas reproducibles; **no se editan a mano**.

| Carpeta | Contenido | Para quién | Cómo se regenera |
|---|---|---|---|
| `ai/` | Envolturas completas de `POST /api/v1/detect` (éxito y error), según `contratos/API_IA_BACK.md` | Job (BE-03, BE-04, tests), Miguel (forma de los datos) | `python -m tools.make_fixtures` |
| `frames/` | Fotogramas sintéticos (dibujos, no fotos) usados por los fixtures y los paquetes | — | ídem |
| `catalog/speciesnet_catalog.sample.json` | 558 taxones **reales** de SpeciesNet 4.0.2a (familias neotropicales) | Job (BE-25), Miguel (autocompletado) | `python -m tools.build_sample_catalog` (requiere el modelo descargado) |
| `package/` | Paquetes `.wwm.zip`: 2 válidos y 6 inválidos, uno por cada regla de `PAQUETE_ESCRITORIO.md §5` | Job (BE-40), Miguel (FE-32) | `python -m tools.make_sample_packages` |

Los **schemas JSON** de todos los modelos están en `../../schemas/` (`python -m tools.export_schemas`).

## Respuestas de `/detect`
| Archivo | Caso |
|---|---|
| `animal_species_video.json` | Jaguar en un video, con 2 alternativas (puma, ocelote), fotograma en 00:07 |
| `animal_genus_video.json` | *Dasyprocta*, identificado solo hasta género |
| `animal_image.json` | Tapir en una imagen (frame 0) |
| `blank_image.json` | Imagen vacía (`evidence=null`) |
| `person_video.json` · `vehicle_image.json` | Clases no animales, con fotograma |
| `error_decode.json` · `error_duration.json` · `error_unauthorized.json` · `error_busy.json` | Errores de la IA con su código |

## Paquetes
| Archivo | Resultado esperado del validador |
|---|---|
| `sample_valid.wwm.zip` | Válido: 12 líneas (7 animal, 2 blank, 1 person, 1 vehicle, 1 error), con datos de cámara |
| `sample_valid_no_camera.wwm.zip` | Válido, sin datos de cámara |
| `invalid_zip_slip.wwm.zip` | `IMPORT_INVALID_PACKAGE` (ruta `../evil.jpg`) |
| `invalid_checksum.wwm.zip` | `IMPORT_CHECKSUM_MISMATCH` |
| `invalid_version_2.wwm.zip` | `IMPORT_UNSUPPORTED_VERSION` |
| `invalid_no_manifest.wwm.zip` | `IMPORT_INVALID_PACKAGE` |
| `invalid_corrupt_jpg.wwm.zip` | *Issue* `EVIDENCE_INVALID` en la línea 1 |
| `invalid_unknown_taxon.wwm.zip` | *Issue* `CATALOG_TAXON_NOT_FOUND` en la línea 4 |

## Nota sobre los nombres comunes
Los nombres comunes vienen **tal cual del modelo SpeciesNet, en inglés** ("jaguar", "ocelot", "lowland tapir"). La traducción al español está pendiente de decisión (ver la HU-IA-21).
