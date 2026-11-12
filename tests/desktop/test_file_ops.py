"""HU-DS-03 · Nombre de la carpeta por especie (F-NAMES generados en el test)."""

from pathlib import Path

import pytest

from desktop.job.file_ops import MAX_FOLDER_CHARS, safe_folder_name
from tools._fixture_taxa import FIXTURE_LABELS, JAGUAR, TAPIR
from wwm_pipeline.contract import Taxonomy
from wwm_pipeline.enums import Rank
from wwm_pipeline.taxonomy import TaxonomyIndex

ROOT = Path(__file__).resolve().parents[2]


def tax(genus="panthera", species="onca", common="jaguar", rank=Rank.SPECIES, family="felidae", order="carnivora"):
    levels = {"class": "mammalia", "order": order, "family": family, "genus": genus, "species": species}
    depth = ["class", "order", "family", "genus", "species"].index(rank.value)
    keep = {k: (v if i <= depth else None) for i, (k, v) in enumerate(levels.items())}
    name = {
        Rank.SPECIES: f"{(genus or '').capitalize()} {species}",
        Rank.GENUS: (genus or "").capitalize(),
        Rank.FAMILY: (family or "").capitalize(),
        Rank.ORDER: (order or "").capitalize(),
        Rank.CLASS: "Mammalia",
    }[rank]
    return Taxonomy.model_validate(
        {"label_id": "x", "label_raw": "x", **keep, "scientific_name": name, "common_name": common, "rank": rank}
    )


# ---------------------------------------------------------------------------
# CA-1 · Por nivel · CA-2 · Nombres comunes difíciles · CA-3 · Windows · CA-4 · Reservados
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "taxonomy,expected",
    [
        (tax(), "panthera_onca_jaguar"),
        (tax(rank=Rank.GENUS, common="panthera species"), "panthera_sp"),
        (tax(rank=Rank.FAMILY, common="felidae family"), "felidae_fam"),
        (tax("tapirus", "terrestris", "tapir amazónico", family="tapiridae"), "tapirus_terrestris_tapir_amazonico"),
        (tax("muntiacus", "reevesi", "reeves' muntjac", family="cervidae"), "muntiacus_reevesi_reeves_muntjac"),
        (tax("rhea", "americana", "ñandú", family="rheidae"), "rhea_americana_nandu"),
        (tax(common=None), "panthera_onca"),
        (tax(common=""), "panthera_onca"),
        (tax(common='ja<g>u:a"r/\\|?*'), "panthera_onca_ja_g_u_a_r"),
        (tax(common="jaguar. "), "panthera_onca_jaguar"),
        (tax(common="  Jaguar   Negro  "), "panthera_onca_jaguar_negro"),
        (tax(common="Gato-Montés (Andino)"), "panthera_onca_gato_montes_andino"),
        (tax(common="cóndor\tde\nlos llanos"), "panthera_onca_condor_de_los_llanos"),
        (tax(genus="con", species=None, common=None, rank=Rank.GENUS), "con_sp"),
        (tax(genus="nul", species=None, common=None, rank=Rank.GENUS), "nul_sp"),
    ],
)
def test_safe_folder_name(taxonomy, expected):
    assert safe_folder_name(taxonomy) == expected


@pytest.mark.parametrize("reserved", ["con", "prn", "aux", "nul", "com1", "com9", "lpt1", "lpt9"])
def test_reserved_names_get_suffix(reserved):
    """CA-4: un nombre igual a uno reservado de Windows recibe "_" (con este formato solo puede pasar si el
    nombre es de una sola parte, pero la protección se aplica siempre)."""
    from desktop.job.file_ops import _windows_safe

    assert _windows_safe(reserved) == f"{reserved}_"
    assert _windows_safe(f"{reserved}_sp") == f"{reserved}_sp"


def test_long_names_are_cut():
    name = safe_folder_name(tax(common="nombre " * 40))
    assert len(name) <= MAX_FOLDER_CHARS and not name.endswith("_")
    assert name.startswith("panthera_onca_nombre")


@pytest.mark.parametrize("rank", [Rank.ORDER, Rank.CLASS])
def test_order_and_class_have_no_folder(rank):
    """CA-1: orden y clase quedan en el origen (HU-DS-02, F2-D8)."""
    with pytest.raises(ValueError):
        safe_folder_name(tax(species=None, genus=None, family=None, rank=rank, common="galliformes order"))


def test_folder_names_are_creatable(tmp_path):
    """CA-3: cada nombre se puede crear como carpeta en Windows y no tiene separadores."""
    cases = [
        tax(),
        tax(common='a<b>c:d"e/f\\g|h?i*j'),
        tax(common="ñandú común."),
        tax(common="x" * 200),
        tax(genus="com1", species=None, common=None, rank=Rank.GENUS),
    ]
    for t in cases:
        name = safe_folder_name(t)
        assert Path(name).name == name
        (tmp_path / name).mkdir()
        assert (tmp_path / name).is_dir()


# ---------------------------------------------------------------------------
# CA-5 · Estable y en el idioma configurado
# ---------------------------------------------------------------------------
def test_folder_name_lang():
    index = TaxonomyIndex(FIXTURE_LABELS)  # con la tabla real en español
    assert safe_folder_name(index.to_taxonomy(TAPIR, "en")) == "tapirus_terrestris_lowland_tapir"
    assert safe_folder_name(index.to_taxonomy(TAPIR, "es")) == "tapirus_terrestris_anta"
    assert safe_folder_name(index.to_taxonomy(JAGUAR, "es")) == "panthera_onca_jaguar"
    assert safe_folder_name(index.to_taxonomy(JAGUAR)) == safe_folder_name(index.to_taxonomy(JAGUAR))


def test_genus_and_family_from_vote_rank_up():
    index = TaxonomyIndex(FIXTURE_LABELS, common_names_es={})
    jaguar = index.lookup(JAGUAR)
    assert safe_folder_name(index.taxonomy_of(index.ancestor(jaguar, Rank.GENUS))) == "panthera_sp"
    assert safe_folder_name(index.taxonomy_of(index.ancestor(jaguar, Rank.FAMILY))) == "felidae_fam"


# ---------------------------------------------------------------------------
# CA-6 · Sin la regla vieja
# ---------------------------------------------------------------------------
def test_no_prototype_folder_rule():
    for py in (ROOT / "desktop").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for banned in ("animal_desconocido", "species_folder_name", "prototype_label"):
            assert banned not in text, (py.name, banned)
