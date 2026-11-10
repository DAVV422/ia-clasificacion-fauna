"""Predicción normalizada: la misma forma para todas las estrategias (legacy, web, desktop)."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Prediction:
    file_name: str
    category: str  # animal | person | vehicle | blank | error
    confidence: float
    raw_label: Optional[str] = None  # etiqueta tal como la dio la estrategia
    scientific_name: Optional[str] = None
    rank: Optional[str] = None  # species | genus | family | order | class
    genus: Optional[str] = None
    family: Optional[str] = None
    frames_analyzed: int = 0
    frames: List[int] = field(default_factory=list)
    processing_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
