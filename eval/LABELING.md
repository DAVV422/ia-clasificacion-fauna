# Protocolo de etiquetado del set de validación

Objetivo: que cada archivo tenga **la verdad** con la que se compara la IA. Una etiqueta mal puesta ensucia todas las métricas, así que ante la duda **se etiqueta en el nivel seguro**.

## Qué se anota por archivo
| Columna | Qué poner |
|---|---|
| `label_category` | `animal` · `person` · `vehicle` · `blank` |
| `label_scientific_name` | Solo con `animal`. El **nombre científico** del nivel seguro: `Panthera onca` (especie), `Dasyprocta` (género), `Felidae` (familia). Vacío si es un animal que no se puede identificar ni a familia. |
| `label_rank` | `species` · `genus` · `family` · `order` · `class` (el nivel del nombre anotado), o vacío si no hay nombre |
| `notes` | Lo que ayude a entender el caso: "nocturno IR", "solo la cola", "dos especies", "muy lejos" |

`file_name`, `sha256`, `media_type` y `split` los completa `label_helper.py`: **no se editan a mano** (el `split` depende del sha256 y un test lo verifica).

## Reglas
1. **`blank`:** no hay ningún animal, persona ni vehículo visible **en todo** el archivo (revisar el video completo, no solo el inicio). Hojas, sombras o lluvia → `blank`.
2. **Nivel seguro:** si no hay certeza de la especie pero sí del género (por ejemplo, un agutí que no se distingue entre especies), se anota el **género**. Lo mismo con la familia.
3. **Animal no identificable:** si se ve un animal pero no se puede identificar ni a familia, se pone `animal` con nombre y rank vacíos, y una nota. Cuenta para los falsos negativos y para la categoría, pero no para el acierto de especie.
4. **Varias especies en el mismo archivo:** se anota la **más visible** (la que más tiempo o más grande aparece) y la otra en `notes`. (El TDR no contempla más de una especie por archivo.)
5. **Persona o vehículo junto a un animal:** se anota el **animal**.
6. **Nombre científico:** se usa el que tiene SpeciesNet (`label_helper.py` lo valida contra las etiquetas del modelo). Si la especie no está en el catálogo, se anota en el nivel superior que sí está, con una nota.
7. **Ante la duda, nota:** un caso dudoso se marca en `notes` con `revisar` y se resuelve al final con una segunda mirada.

## Fuentes para identificar
- Guías de mamíferos de Bolivia (Wallace et al., *Distribución, ecología y conservación de los mamíferos medianos y grandes de Bolivia*).
- iNaturalist / GBIF para comparar fotos de referencia.
- La consulta con WWF para los casos que el equipo no pueda resolver.

## Cómo se hace
```bash
set EVAL_DATA_DIR=D:\wwm-eval\videos       # la carpeta con los archivos de prueba (fuera del repo)
python -m eval.label_helper
```
Abre cada archivo, pide la etiqueta y guarda `eval/labels.csv` después de cada uno (se puede cortar con `q` y seguir otro día). Al terminar: `python -m pytest eval/tests` verifica que el CSV sea coherente.
