from __future__ import annotations

from app.infer import align_band_ids, parse_model_output


def test_parse_model_output_from_code_fence_and_trailing_comma() -> None:
    raw = """
```json
{
  "bands": [
    {
      "band_id": "10k",
      "tracks": [
        {
          "track_id": "t1",
          "confidence": 0.8,
          "points": [[1,2],[3,4]],
          "summary": "track",
        }
      ]
    }
  ]
}
```
"""

    bands = parse_model_output(raw)
    assert len(bands) == 1
    assert bands[0].band_id == "10k"
    assert len(bands[0].tracks) == 1
    assert bands[0].tracks[0].track_id == "t1"


def test_align_band_ids_by_expected_order() -> None:
    raw = """
{
  "bands": [
    {
      "band_id": "unknown",
      "tracks": [
        {
          "track_id": "t1",
          "confidence": 0.8,
          "points": [[1,2],[3,4]],
          "summary": "track"
        }
      ]
    }
  ]
}
"""
    bands = parse_model_output(raw)
    aligned = align_band_ids(bands, ["10k", "12k"])
    assert aligned[0].band_id == "10k"
    assert aligned[1].band_id == "12k"
