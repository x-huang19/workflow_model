from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("beam_track_analyzer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_prompt_template(template_file: Path, extra_instruction: str) -> str:
    template = template_file.read_text(encoding="utf-8").strip()
    if extra_instruction.strip():
        return f"{template}\n\n附加要求:\n{extra_instruction.strip()}"
    return template


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
