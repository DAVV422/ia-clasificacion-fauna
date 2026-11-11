"""Genera ``wwm_pipeline/data/common_names_es.csv`` (HU-IA-14, F1-D9).

Nombres comunes en español (uso en Bolivia) para ~150 especies frecuentes en cámaras trampa. Solo
se escriben las que existen en las etiquetas de SpeciesNet, con su ``label_id`` real; las que no
están se informan por pantalla. **Lista a revisar por WWF** con el *feedback* de campo: para corregir
un nombre se edita ``NAMES`` y se vuelve a correr.

    python -m tools.build_common_names_es [ruta/al/archivo/de/etiquetas.txt]
"""

import csv
import sys
from pathlib import Path

from wwm_pipeline.catalog_schema import parse_label
from wwm_pipeline.enums import Rank
from wwm_pipeline.taxonomy import COMMON_NAMES_ES

DEFAULT_LABELS = (
    Path.home()
    / ".cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.2a/1"
    / "always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt"
)

NAMES = {
    # --- Felinos ---
    "Panthera onca": "jaguar",
    "Puma concolor": "puma",
    "Leopardus pardalis": "ocelote",
    "Leopardus wiedii": "margay",
    "Leopardus tigrinus": "tigrillo",
    "Leopardus colocolo": "gato del pajonal",
    "Herpailurus yagouaroundi": "gato moro",
    # --- Cánidos ---
    "Speothos venaticus": "perrito de monte",
    "Chrysocyon brachyurus": "borochi",
    "Cerdocyon thous": "zorro de monte",
    "Lycalopex culpaeus": "zorro andino",
    "Atelocynus microtis": "perro de orejas cortas",
    "Canis familiaris": "perro doméstico",
    # --- Otros carnívoros ---
    "Nasua nasua": "tejón",
    "Procyon cancrivorus": "osito lavador",
    "Potos flavus": "martucha",
    "Eira barbara": "melero",
    "Galictis vittata": "hurón grande",
    "Galictis cuja": "hurón menor",
    "Mustela frenata": "comadreja",
    "Lontra longicaudis": "lobito de río",
    "Pteronura brasiliensis": "londra",
    "Conepatus chinga": "zorrino",
    "Conepatus semistriatus": "zorrino",
    "Tremarctos ornatus": "jucumari",
    "Felis catus": "gato doméstico",
    # --- Ungulados ---
    "Tapirus terrestris": "anta",
    "Pecari tajacu": "taitetú",
    "Tayassu pecari": "tropero",
    "Mazama americana": "huaso",
    "Mazama gouazoubira": "urina",
    "Mazama nemorivaga": "huaso amazónico",
    "Mazama chunyi": "huaso andino",
    "Blastocerus dichotomus": "ciervo de los pantanos",
    "Ozotoceros bezoarticus": "venado de las pampas",
    "Odocoileus virginianus": "venado de cola blanca",
    "Hippocamelus antisensis": "taruca",
    "Lama guanicoe": "guanaco",
    "Lama glama": "llama",
    "Capra aegagrus hircus": "cabra",
    "Bos taurus": "vaca",
    "Equus caballus": "caballo",
    "Equus asinus": "burro",
    "Sus scrofa": "cerdo",
    "Ovis aries": "oveja",
    # --- Roedores y lagomorfos ---
    "Hydrochoerus hydrochaeris": "capiguara",
    "Cuniculus paca": "jochi pintado",
    "Dasyprocta punctata": "jochi colorado",
    "Dasyprocta variegata": "jochi colorado",
    "Dasyprocta azarae": "jochi colorado",
    "Dasyprocta fuliginosa": "jochi negro",
    "Myoprocta pratti": "jochi pequeño",
    "Dinomys branickii": "pacarana",
    "Coendou prehensilis": "puercoespín",
    "Coendou bicolor": "puercoespín",
    "Lagidium viscacia": "vizcacha",
    "Sylvilagus brasiliensis": "tapití",
    "Sciurus spadiceus": "ardilla roja",
    "Sciurus ignitus": "ardilla",
    "Sciurus igniventris": "ardilla roja",
    # --- Xenartros ---
    "Myrmecophaga tridactyla": "oso bandera",
    "Tamandua tetradactyla": "oso hormiguero",
    "Priodontes maximus": "pejichi",
    "Dasypus novemcinctus": "tatú",
    "Dasypus kappleri": "tatú grande",
    "Euphractus sexcinctus": "peji",
    "Cabassous unicinctus": "tatú de rabo molle",
    "Tolypeutes matacus": "corechi",
    "Bradypus variegatus": "perezoso de tres dedos",
    "Choloepus hoffmanni": "perezoso de dos dedos",
    # --- Marsupiales ---
    "Didelphis marsupialis": "carachupa",
    "Didelphis albiventris": "carachupa overa",
    "Philander opossum": "carachupa de cuatro ojos",
    "Metachirus nudicaudatus": "carachupa marrón",
    # --- Primates ---
    "Alouatta sara": "manechi colorado",
    "Alouatta caraya": "manechi negro",
    "Ateles chamek": "marimono",
    "Sapajus apella": "mono silbador",
    "Sapajus cay": "mono silbador",
    "Cebus albifrons": "mono capuchino",
    "Saimiri boliviensis": "chichilo",
    "Aotus nigriceps": "mono nocturno",
    "Plecturocebus brunneus": "lucachi",
    "Lagothrix lagotricha": "mono barrigudo",
    "Leontocebus fuscicollis": "chichilo de bigote",
    "Homo sapiens": "persona",
    # --- Aves ---
    "Crax fasciolata": "pava pintada",
    "Mitu tuberosum": "mutún",
    "Penelope jacquacu": "pava de monte",
    "Penelope superciliaris": "pava de monte",
    "Penelope obscura": "pava de monte",
    "Pipile cumanensis": "pava campanilla",
    "Ortalis canicollis": "charata",
    "Ortalis guttata": "guaracachi",
    "Tinamus major": "perdiz grande",
    "Tinamus tao": "perdiz azulada",
    "Tinamus guttatus": "perdiz",
    "Crypturellus undulatus": "perdiz ondulada",
    "Crypturellus obsoletus": "perdiz parda",
    "Crypturellus soui": "perdiz chica",
    "Crypturellus tataupa": "perdiz tataupá",
    "Crypturellus variegatus": "perdiz variada",
    "Rhynchotus rufescens": "perdiz colorada",
    "Nothura maculosa": "perdiz chica",
    "Rhea americana": "piyo",
    "Cariama cristata": "socori",
    "Psophia leucoptera": "pollo de monte",
    "Aramides cajaneus": "pollo de agua",
    "Mycteria americana": "cabeza seca",
    "Jabiru mycteria": "bato",
    "Ciconia maguari": "cigüeña",
    "Cairina moschata": "pato negro",
    "Coragyps atratus": "sucha",
    "Cathartes aura": "peroqui",
    "Sarcoramphus papa": "cóndor de los llanos",
    "Vultur gryphus": "cóndor",
    "Caracara plancus": "carcaña",
    "Ramphastos toco": "tucán",
    "Leptotila verreauxi": "torcaza",
    "Leptotila rufaxilla": "torcaza",
    "Geotrygon montana": "paloma perdiz",
    "Patagioenas picazuro": "paloma picazuró",
    "Tigrisoma lineatum": "garza tigre",
    "Bubulcus ibis": "garza bueyera",
    "Theristicus caudatus": "bandurria",
    "Mesembrinibis cayennensis": "corocoro",
    "Columbina talpacoti": "palomita colorada",
    "Turdus amaurochalinus": "zorzal",
    "Odontophorus stellatus": "codorniz",
    "Gallus gallus": "gallina",
    "Gallus gallus domesticus": "gallina",
    # --- Reptiles ---
    "Caiman crocodilus": "lagarto",
    "Salvator merianae": "peni",
    "Tupinambis teguixin": "peni",
    "Chelonoidis denticulatus": "peta de monte",
}


def main(labels_path: Path = DEFAULT_LABELS) -> int:
    by_name = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        taxon = parse_label(line) if line.strip() else None
        if taxon is not None and taxon.rank is Rank.SPECIES:
            by_name.setdefault(taxon.scientific_name.lower(), taxon)
    rows, missing = [], []
    for name, es in NAMES.items():
        taxon = by_name.get(name.lower())
        if taxon is None:
            missing.append(name)
        else:
            rows.append((taxon.label_id, taxon.scientific_name, es))
    rows.sort(key=lambda r: r[1])
    COMMON_NAMES_ES.parent.mkdir(parents=True, exist_ok=True)
    with COMMON_NAMES_ES.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["label_id", "scientific_name", "common_name_es"])
        writer.writerows(rows)
    print(f"{len(rows)} especies escritas en {COMMON_NAMES_ES}")
    if missing:
        print(f"{len(missing)} no están en las etiquetas de SpeciesNet (se omiten): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LABELS))
