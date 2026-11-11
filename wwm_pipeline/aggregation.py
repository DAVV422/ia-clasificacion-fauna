"""Agregación por archivo.

- ``legacy_best`` (fase 0): la regla del prototipo (``_resolve_species``); la usan los presets ``legacy_*``.
- ``vote`` (fase 1, HU-IA-13): votación ponderada entre frames con subida a género o familia; la usan
  los presets ``web()`` y ``desktop()``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract import Alternative, Taxonomy, Vote
from .enums import Category, Rank


@dataclass(frozen=True)
class Decision:
    category: str  # animal | person | vehicle | blank
    label: Optional[str]  # "género;especie;común" (formato del prototipo) o None
    confidence: float


def legacy_best(detections: Iterable[Dict[str, Any]]) -> Decision:
    """La especie de mayor confianza entre las detecciones de animal (primera especie de cada una)."""
    dets = list(detections)
    animal = [d for d in dets if d.get("class_name") == "animal"]
    if not animal:
        classes = {d.get("class_name") for d in dets}
        category = "person" if "person" in classes else "vehicle" if "vehicle" in classes else "blank"
        return Decision(category, None, 0.0)
    best: Optional[str] = None
    best_conf = 0.0
    for det in animal:
        species_list = det.get("species", [])
        if species_list:
            top = species_list[0]
            if top["confidence"] > best_conf:
                best_conf = top["confidence"]
                best = top["species"]
    return Decision("animal", best, best_conf)


# ---------------------------------------------------------------------------
# Fase 1 · Votación ponderada entre frames y subida de nivel (HU-IA-13)
# ---------------------------------------------------------------------------
_LEVELS = (Rank.SPECIES, Rank.GENUS, Rank.FAMILY)
MAX_ALTERNATIVES = 3
MAX_VOTES = 5
MIN_ALTERNATIVE_SCORE = 0.005  # API_IA_BACK §2.3


@dataclass(frozen=True)
class FrameVote:
    """El voto de un frame: la etiqueta top-1 de SpeciesNet sobre su recuadro de animal más confiable."""

    frame_index: int
    timestamp_seconds: float
    label_raw: str
    det_conf: float
    cls_conf: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # formato del prototipo, para HU-IA-15

    @property
    def weight(self) -> float:
        return self.det_conf * self.cls_conf


@dataclass(frozen=True)
class VoteResult:
    category: Category
    taxonomy: Optional[Taxonomy]
    confidence: float
    alternatives: List[Alternative] = field(default_factory=list)
    votes: List[Vote] = field(default_factory=list)
    low_evidence: bool = False
    # Los votos que apoyan al ganador, de mayor a menor peso (HU-IA-15 elige el fotograma de ahí).
    support: Tuple[FrameVote, ...] = ()

    @property
    def rank(self) -> Optional[Rank]:
        return self.taxonomy.rank if self.taxonomy is not None else None

    def to_decision(self) -> Decision:
        """El ``Decision`` corto (categoría, nombre científico, confianza) para el log y la interfaz."""
        label = self.taxonomy.scientific_name if self.taxonomy is not None else None
        return Decision(self.category.value, label, self.confidence)


@dataclass
class _Group:
    taxon: Any  # CatalogTaxon
    weight: float = 0.0
    best_single: float = 0.0
    votes: List[FrameVote] = field(default_factory=list)

    def add(self, v: FrameVote) -> None:
        self.votes.append(v)
        self.weight += v.weight
        self.best_single = max(self.best_single, v.weight)


def _ranked(groups: Mapping[str, _Group]) -> List[_Group]:
    """Desempate determinista (CA-6): mayor peso → mayor voto individual → ``label_id`` alfabético."""
    return sorted(groups.values(), key=lambda g: (-round(g.weight, 9), -round(g.best_single, 9), g.taxon.label_id))


def _group(fauna: Sequence[Tuple[FrameVote, Any]], index, rank: Rank) -> Dict[str, _Group]:
    groups: Dict[str, _Group] = {}
    for v, taxon in fauna:
        key_taxon = taxon if rank is Rank.SPECIES else index.ancestor(taxon, rank)
        groups.setdefault(key_taxon.label_id, _Group(key_taxon)).add(v)
    return groups


def _is_vehicle_label(label_raw: str) -> bool:
    return label_raw.rsplit(";", 1)[-1].strip().lower() == "vehicle"


def vote(
    votes: Sequence[FrameVote],
    index,
    *,
    species_threshold: float = 0.60,
    genus_threshold: float = 0.60,
    min_single_vote: float = 0.25,
    md_confidence: Optional[Mapping[str, float]] = None,
    frames_without_detections: int = 0,
    frames_total: int = 0,
    lang: str = "en",
) -> VoteResult:
    """Una decisión por archivo a partir de los votos de sus frames (HU-IA-13).

    - ``votes``: un voto por frame clasificado (su recuadro de animal más confiable).
    - ``index``: el ``TaxonomyIndex`` del modelo (HU-IA-14).
    - ``md_confidence``: la confianza máxima de MegaDetector por categoría (``person``, ``vehicle``),
      para decidir los archivos **sin fauna**. Si hay fauna, gana la fauna (F1-D4).
    - ``frames_without_detections`` / ``frames_total``: la confianza de un ``blank`` (proporción de
      frames sin ninguna detección).

    **Subida de nivel (F1-D13):** el nivel se elige con la proporción **entre los votos de fauna**,
    porque subir de especie a género solo resuelve la duda taxonómica. La ``confidence`` que se
    informa es la proporción sobre **todos** los votos de recuadros de animal: los votos ``blank`` o
    ``vehicle`` del clasificador la bajan (CA-5), pero no fuerzan una subida de nivel.
    """
    md = dict(md_confidence or {})
    fauna: List[Tuple[FrameVote, Any]] = []
    total = 0.0
    for v in votes:
        total += v.weight
        if index.is_non_taxon(v.label_raw):
            if _is_vehicle_label(v.label_raw):
                md["vehicle"] = max(md.get("vehicle", 0.0), v.det_conf)
            continue
        if index.is_person(v.label_raw):  # Homo sapiens sobre un recuadro de "animal" → voto de persona
            md["person"] = max(md.get("person", 0.0), v.det_conf)
            continue
        taxon = index.lookup(v.label_raw)
        if taxon is None:  # respuesta sin taxonomía (la alternativa del prototipo)
            continue
        fauna.append((v, taxon))

    if not fauna:
        return _without_fauna(md, frames_without_detections, frames_total)

    fauna_total = sum(v.weight for v, _ in fauna)
    thresholds = {Rank.SPECIES: species_threshold, Rank.GENUS: genus_threshold}
    ranked: List[_Group] = []
    for rank in _LEVELS:
        ranked = _ranked(_group(fauna, index, rank))
        share = ranked[0].weight / fauna_total if fauna_total > 0 else 1.0
        if rank is Rank.FAMILY or share >= thresholds[rank]:
            break
    winner = ranked[0]

    def score(weight: float) -> float:
        return round(min(1.0, weight / total), 4) if total > 0 else 0.0

    leaf = _ranked(_group(fauna, index, Rank.SPECIES))
    top_votes = [
        Vote(
            label_id=g.taxon.label_id,
            scientific_name=g.taxon.scientific_name,
            frames=len(g.votes),
            score=score(g.weight),
        )
        for g in leaf[:MAX_VOTES]
    ]
    alternatives = [
        index.to_alternative(g.taxon, score(g.weight), lang)
        for g in ranked[1:]
        if score(g.weight) > MIN_ALTERNATIVE_SCORE
    ][:MAX_ALTERNATIVES]
    support = tuple(sorted(winner.votes, key=lambda v: (-v.weight, v.timestamp_seconds)))
    low_evidence = len(fauna) == 1 and fauna[0][0].weight < min_single_vote
    return VoteResult(
        Category.ANIMAL,
        index.taxonomy_of(winner.taxon, lang),
        score(winner.weight),
        alternatives,
        top_votes,
        low_evidence,
        support,
    )


def _without_fauna(md: Mapping[str, float], empty_frames: int, frames_total: int) -> VoteResult:
    person, vehicle = md.get("person", 0.0), md.get("vehicle", 0.0)
    if person > 0 and person >= vehicle:
        return VoteResult(Category.PERSON, None, round(person, 4))
    if vehicle > 0:
        return VoteResult(Category.VEHICLE, None, round(vehicle, 4))
    confidence = empty_frames / frames_total if frames_total > 0 else 1.0
    return VoteResult(Category.BLANK, None, round(confidence, 4))
