from __future__ import annotations

import json
import re
from typing import Any

from app.config import RuntimeConfig
from app.types import BandTracks, Track

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def build_inference_prompt(cfg: RuntimeConfig, template: str) -> str:
    band_lines = [
        f"{idx + 1}. band_id={item.band_id}, image_path={item.path.name}"
        for idx, item in enumerate(cfg.input.images)
    ]

    instruction = [
        "输入图像顺序与频带映射如�?",
        *band_lines,
        "请严格输�?JSON，格式为:",
        '{"bands":[{"band_id":"...","tracks":[{"track_id":"...","confidence":0.0,"points":[[x,y],[x,y]],"summary":"..."}]}]}',
    ]

    return f"{template}\n\n" + "\n".join(instruction)


def parse_model_output(raw_text: str) -> list[BandTracks]:
    payload = _parse_json_payload(raw_text)

    if isinstance(payload, dict) and isinstance(payload.get("bands"), list):
        bands_raw = payload["bands"]
    elif isinstance(payload, list):
        bands_raw = payload
    else:
        raise ValueError("Parsed JSON does not contain 'bands' list")

    result: list[BandTracks] = []
    for band_idx, band_item in enumerate(bands_raw):
        if not isinstance(band_item, dict):
            continue

        band_id = str(band_item.get("band_id", f"band_{band_idx + 1}")).strip()
        tracks_raw = band_item.get("tracks", [])
        if not isinstance(tracks_raw, list):
            tracks_raw = []

        tracks: list[Track] = []
        for track_idx, track_item in enumerate(tracks_raw):
            if not isinstance(track_item, dict):
                continue

            track_id = str(track_item.get("track_id", f"t{track_idx + 1}")).strip()
            confidence = _to_float(track_item.get("confidence", 0.0))
            summary = str(track_item.get("summary", ""))
            points = _parse_points(track_item.get("points", []))

            if len(points) >= 2:
                tracks.append(
                    Track(
                        track_id=track_id,
                        confidence=confidence,
                        points=points,
                        summary=summary,
                    )
                )

        result.append(BandTracks(band_id=band_id, tracks=tracks))

    return result


def align_band_ids(
    parsed_bands: list[BandTracks],
    expected_band_ids: list[str],
) -> list[BandTracks]:
    if not parsed_bands:
        return [BandTracks(band_id=band_id, tracks=[]) for band_id in expected_band_ids]

    expected_set = set(expected_band_ids)
    output: list[BandTracks] = []

    for idx, band in enumerate(parsed_bands):
        if band.band_id not in expected_set and idx < len(expected_band_ids):
            output.append(BandTracks(band_id=expected_band_ids[idx], tracks=band.tracks))
        else:
            output.append(band)

    existing = {b.band_id for b in output}
    for band_id in expected_band_ids:
        if band_id not in existing:
            output.append(BandTracks(band_id=band_id, tracks=[]))

    return output


def _parse_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()

    match = _CODE_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    text = _TRAILING_COMMA_RE.sub(r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model JSON output: {exc}") from exc


def _parse_points(points_raw: Any) -> list[tuple[float, float]]:
    if not isinstance(points_raw, list):
        return []

    points: list[tuple[float, float]] = []
    for item in points_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x = _to_float(item[0])
        y = _to_float(item[1])
        points.append((x, y))

    return points


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
