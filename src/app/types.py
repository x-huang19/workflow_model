from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Track:
    track_id: str
    confidence: float
    points: list[tuple[float, float]]
    summary: str = ""


@dataclass(slots=True)
class BandTracks:
    band_id: str
    tracks: list[Track] = field(default_factory=list)
