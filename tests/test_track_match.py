from __future__ import annotations

from app.track_match import find_cross_band_tracks
from app.types import BandTracks, Track


def test_find_cross_band_tracks_filters_2_to_3_bands() -> None:
    t1 = Track(track_id="t1", confidence=0.9, points=[(0, 0), (10, 10)])
    t2 = Track(track_id="t2", confidence=0.8, points=[(1, 1), (11, 11)])
    t3 = Track(track_id="t3", confidence=0.7, points=[(100, 100), (110, 110)])

    per_band = [
        BandTracks(band_id="10k", tracks=[t1]),
        BandTracks(band_id="12k", tracks=[t2]),
        BandTracks(band_id="15k", tracks=[t3]),
    ]

    clusters = find_cross_band_tracks(
        per_band_tracks=per_band,
        spatial_tolerance_px=5.0,
        min_shared_bands=2,
        max_shared_bands=3,
    )

    assert len(clusters) == 1
    assert clusters[0]["band_count"] == 2
    assert set(clusters[0]["band_ids"]) == {"10k", "12k"}
