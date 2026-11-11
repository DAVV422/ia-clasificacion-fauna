"""HU-IA-13 · Votación ponderada por archivo y subida de nivel (votos sintéticos)."""

import random

import pytest

from tools._fixture_taxa import FIXTURE_LABELS, JAGUAR, OCELOT, PUMA
from wwm_pipeline.aggregation import FrameVote, vote
from wwm_pipeline.contract import Analysis, DetectResult
from wwm_pipeline.enums import Category, Rank
from wwm_pipeline.taxonomy import TaxonomyIndex

# Formato de 7 partes de SpeciesNet; los label_id de estas son inventados para el test.
RED_BROCKET = "3a6fe9a8-8e8b-4a4f-8e2c-6d1d5b0e0a01;mammalia;cetartiodactyla;cervidae;mazama;americana;red brocket"
GRAY_BROCKET = "5c0c9c0b-2c48-4b55-9d0e-2f6b7c3e1a02;mammalia;cetartiodactyla;cervidae;mazama;gouazoubira;gray brocket"
PECCARY = "f6e1e8f5-6a4b-4d9e-9a5c-8b7d6e5f4a03;mammalia;cetartiodactyla;tayassuidae;pecari;tajacu;collared peccary"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"


@pytest.fixture
def index():
    labels = FIXTURE_LABELS + [RED_BROCKET, GRAY_BROCKET, PECCARY, BLANK, VEHICLE, HUMAN]
    return TaxonomyIndex(labels, common_names_es={})


def votes_of(*groups):
    """``(etiqueta, n_frames, peso)`` → votos con det_conf=1 y cls_conf=peso, en frames consecutivos."""
    out, i = [], 0
    for label, n, w in groups:
        for _ in range(n):
            out.append(FrameVote(i, i / 10, label, 1.0, w))
            i += 1
    return out


def test_vote_majority_beats_single_frame(index):
    """CA-1."""
    res = vote(votes_of((JAGUAR, 8, 0.7), (PUMA, 1, 0.95)), index)
    assert res.category is Category.ANIMAL
    assert res.taxonomy.scientific_name == "Panthera onca" and res.rank is Rank.SPECIES
    assert res.confidence == pytest.approx(5.6 / 6.55, abs=1e-3)
    assert [a.scientific_name for a in res.alternatives] == ["Puma concolor"]
    assert res.votes[0].frames == 8 and res.votes[1].scientific_name == "Puma concolor"
    assert len(res.support) == 8 and not res.low_evidence


def test_vote_goes_up_to_genus(index):
    """CA-2."""
    res = vote(votes_of((RED_BROCKET, 9, 0.5), (GRAY_BROCKET, 8, 0.5), (PECCARY, 3, 0.5)), index)
    assert res.rank is Rank.GENUS
    assert res.taxonomy.scientific_name == "Mazama" and res.taxonomy.family == "cervidae"
    assert res.taxonomy.species is None
    assert res.confidence == pytest.approx(0.85)
    assert [a.scientific_name for a in res.alternatives] == ["Pecari"]  # mismo nivel que el ganador
    assert {v.scientific_name for v in res.votes} == {"Mazama americana", "Mazama gouazoubira", "Pecari tajacu"}


def test_vote_goes_up_to_family(index):
    """CA-3: tres géneros de Felidae sin mayoría."""
    res = vote(votes_of((JAGUAR, 4, 0.6), (PUMA, 3, 0.6), (OCELOT, 3, 0.6)), index)
    assert res.rank is Rank.FAMILY and res.taxonomy.scientific_name == "Felidae"
    assert res.confidence == pytest.approx(1.0)
    assert res.alternatives == []


def test_vote_family_reports_real_confidence(index):
    """Si ni la familia tiene mayoría, se informa la familia ganadora con su proporción real."""
    res = vote(votes_of((JAGUAR, 4, 0.5), (RED_BROCKET, 3, 0.5), (PECCARY, 2, 0.5)), index)
    assert res.rank is Rank.FAMILY and res.taxonomy.scientific_name == "Felidae"
    assert res.confidence == pytest.approx(4 / 9, abs=1e-3)
    assert [a.scientific_name for a in res.alternatives] == ["Cervidae", "Tayassuidae"]


def test_vote_without_fauna(index):
    """CA-4."""
    blank = vote([], index, frames_without_detections=10, frames_total=10)
    assert (blank.category, blank.taxonomy, blank.alternatives, blank.confidence) == (Category.BLANK, None, [], 1.0)
    person = vote([], index, md_confidence={"person": 0.88})
    assert person.category is Category.PERSON and person.confidence == 0.88
    vehicle = vote([], index, md_confidence={"vehicle": 0.7, "person": 0.4})
    assert vehicle.category is Category.VEHICLE
    animal = vote(votes_of((JAGUAR, 2, 0.8)), index, md_confidence={"person": 0.99})
    assert animal.category is Category.ANIMAL  # F1-D4: gana el animal


def test_vote_classifier_says_human_or_vehicle(index):
    """Recuadros de "animal" que SpeciesNet llama humano o vehículo → persona o vehículo, no fauna."""
    assert vote(votes_of((HUMAN, 3, 0.9)), index).category is Category.PERSON
    assert vote(votes_of((VEHICLE, 2, 0.9)), index).category is Category.VEHICLE
    assert vote(votes_of((BLANK, 2, 0.9)), index).category is Category.BLANK


def test_vote_non_fauna_lowers_confidence(index):
    """CA-5: los votos blank bajan la confianza, pero no fuerzan una subida de nivel (F1-D13)."""
    res = vote(votes_of((BLANK, 6, 0.8), (JAGUAR, 4, 0.8)), index)
    assert res.taxonomy.scientific_name == "Panthera onca" and res.rank is Rank.SPECIES
    assert res.confidence == pytest.approx(0.4)


def test_vote_is_deterministic(index):
    """CA-6."""
    votes = votes_of((JAGUAR, 5, 0.7), (PUMA, 3, 0.6), (OCELOT, 2, 0.4))
    expected = vote(votes, index)
    for seed in range(5):
        shuffled = votes[:]
        random.Random(seed).shuffle(shuffled)
        assert vote(shuffled, index) == expected
    tie = vote(votes_of((PUMA, 1, 0.5), (JAGUAR, 1, 0.5)), index, species_threshold=0.5)
    assert tie.taxonomy.label_id == min(JAGUAR.split(";")[0], PUMA.split(";")[0])


def test_vote_low_evidence(index):
    weak = vote([FrameVote(3, 0.2, JAGUAR, 0.4, 0.5)], index)
    assert weak.category is Category.ANIMAL and weak.low_evidence  # no se descarta
    strong = vote([FrameVote(3, 0.2, JAGUAR, 0.9, 0.9)], index)
    assert not strong.low_evidence


def test_vote_common_name_lang():
    index = TaxonomyIndex(FIXTURE_LABELS)
    assert vote(votes_of((JAGUAR, 2, 0.9)), index, lang="es").taxonomy.common_name == "jaguar"


@pytest.mark.parametrize(
    "groups,md",
    [
        ([(JAGUAR, 8, 0.7), (PUMA, 1, 0.95)], {}),
        ([(RED_BROCKET, 9, 0.5), (GRAY_BROCKET, 8, 0.5), (PECCARY, 3, 0.5)], {}),
        ([(JAGUAR, 4, 0.6), (PUMA, 3, 0.6), (OCELOT, 3, 0.6)], {}),
        ([], {}),
        ([], {"person": 0.9}),
        ([(BLANK, 6, 0.8), (JAGUAR, 4, 0.8)], {}),
    ],
)
def test_decision_converts_to_contract(index, groups, md):
    """CA-8: el resultado entra en ``DetectResult`` (taxonomy null ⇔ category ≠ animal)."""
    votes = votes_of(*groups)
    res = vote(votes, index, md_confidence=md, frames_without_detections=3, frames_total=max(3, len(votes)))
    evidence = None
    if res.category is not Category.BLANK:
        evidence = {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "bbox": [0.1, 0.1, 0.5, 0.5],
            "detection_confidence": 0.9,
            "image_base64": "AAAA",
            "width": 64,
            "height": 36,
        }
    data = {
        "media": {
            "file_name": "v.mp4",
            "media_type": "video",
            "sha256": "0" * 64,
            "size_bytes": 10,
            "duration_seconds": 1.0,
            "fps": 10.0,
            "width": 64,
            "height": 36,
        },
        "capture": {"captured_at": None, "source": "none"},
        "result": {
            "category": res.category.value,
            "taxonomy": res.taxonomy.model_dump(by_alias=True) if res.taxonomy else None,
            "confidence": res.confidence,
            "alternatives": [a.model_dump() for a in res.alternatives],
        },
        "evidence": evidence,
        "analysis": Analysis(
            frames_sampled=max(3, len(votes)),
            frames_with_animal=len(votes),
            votes=res.votes,
            low_evidence=res.low_evidence,
            processing_ms=1,
        ).model_dump(),
        "model": {"detector": "md", "classifier": "sn", "sampling": "x", "pipeline_version": "1"},
    }
    DetectResult.model_validate(data)


def test_vote_alternatives_skip_negligible_scores(index):
    res = vote(votes_of((JAGUAR, 9, 0.9), (PUMA, 1, 0.01)), index)
    assert res.alternatives == []  # 0,0012 < 0,005
