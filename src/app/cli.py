from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .infer import align_band_ids, build_inference_prompt, parse_model_output
from .io_utils import ensure_output_dir, load_prompt_template, setup_logging, write_csv, write_json
from .model_loader import VLMEngine
from .track_match import clusters_to_csv_rows, find_cross_band_tracks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-band beam track analyzer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to runtime YAML config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output.dir from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load config: {exc}", file=sys.stderr)
        return 1

    if args.output_dir:
        cfg.output.dir = Path(args.output_dir).resolve()

    output_dir = ensure_output_dir(cfg.output.dir)
    logger = setup_logging(output_dir)

    if args.dry_run:
        logger.info("Config validation successful. dry-run enabled, exiting.")
        return 0

    try:
        prompt_template = load_prompt_template(
            cfg.prompt.template_file,
            cfg.prompt.extra_instruction,
        )
        final_prompt = build_inference_prompt(cfg, prompt_template)

        logger.info("Loading local model from %s", cfg.model.local_model_dir)
        engine = VLMEngine.from_config(cfg.model)
        logger.info("Model loaded on device: %s", engine.device)

        image_paths = [item.path for item in cfg.input.images]
        raw_output = engine.generate_from_images(
            prompt_text=final_prompt,
            image_paths=image_paths,
            max_new_tokens=cfg.model.max_new_tokens,
            temperature=cfg.model.temperature,
        )

        write_json(output_dir / "model_raw_output.json", {"raw_output": raw_output})

        per_band_tracks = parse_model_output(raw_output)
        per_band_tracks = align_band_ids(
            per_band_tracks,
            [item.band_id for item in cfg.input.images],
        )
        per_band_payload = [
            {
                "band_id": band.band_id,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "confidence": t.confidence,
                        "points": [[x, y] for x, y in t.points],
                        "summary": t.summary,
                    }
                    for t in band.tracks
                ],
            }
            for band in per_band_tracks
        ]

        if cfg.output.save_intermediate:
            write_json(output_dir / "all_tracks_by_band.json", {"bands": per_band_payload})

        clusters = find_cross_band_tracks(
            per_band_tracks=per_band_tracks,
            spatial_tolerance_px=cfg.matching.spatial_tolerance_px,
            min_shared_bands=cfg.matching.min_shared_bands,
            max_shared_bands=cfg.matching.max_shared_bands,
        )

        if "json" in cfg.output.formats:
            write_json(output_dir / "cross_band_tracks_2_to_3.json", clusters)

        if "csv" in cfg.output.formats:
            write_csv(
                output_dir / "cross_band_tracks_2_to_3.csv",
                clusters_to_csv_rows(clusters),
                fieldnames=[
                    "cluster_id",
                    "band_count",
                    "band_ids",
                    "track_id",
                    "band_id",
                    "confidence",
                    "summary",
                ],
            )

        logger.info("Completed. Found %d cross-band clusters.", len(clusters))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Run failed: %s", exc)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
