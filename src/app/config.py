from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_DEVICES = {"auto", "cuda", "cpu"}
SUPPORTED_FORMATS = {"json", "csv"}


@dataclass(slots=True)
class ModelConfig:
    local_model_dir: Path
    trust_remote_code: bool = True
    dtype: str = "float16"
    device: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.1


@dataclass(slots=True)
class InputImageConfig:
    path: Path
    band_id: str


@dataclass(slots=True)
class InputConfig:
    images: list[InputImageConfig] = field(default_factory=list)


@dataclass(slots=True)
class PromptConfig:
    template_file: Path
    extra_instruction: str = ""


@dataclass(slots=True)
class MatchingConfig:
    spatial_tolerance_px: float = 20.0
    min_shared_bands: int = 2
    max_shared_bands: int = 3


@dataclass(slots=True)
class OutputConfig:
    dir: Path
    save_intermediate: bool = True
    formats: list[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass(slots=True)
class RuntimeConfig:
    model: ModelConfig
    input: InputConfig
    prompt: PromptConfig
    matching: MatchingConfig
    output: OutputConfig
    config_file: Path


def _to_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _require_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return section


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid value for: {key}")
    return value.strip()


def load_config(config_file: str | Path) -> RuntimeConfig:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    base_dir = config_path.parent

    model_raw = _require_section(raw, "model")
    input_raw = _require_section(raw, "input")
    prompt_raw = _require_section(raw, "prompt")
    matching_raw = _require_section(raw, "matching")
    output_raw = _require_section(raw, "output")

    model_cfg = ModelConfig(
        local_model_dir=_to_path(base_dir, _require_string(model_raw, "local_model_dir")),
        trust_remote_code=bool(model_raw.get("trust_remote_code", True)),
        dtype=str(model_raw.get("dtype", "float16")),
        device=str(model_raw.get("device", "auto")),
        max_new_tokens=int(model_raw.get("max_new_tokens", 1024)),
        temperature=float(model_raw.get("temperature", 0.1)),
    )

    images_raw = input_raw.get("images")
    if not isinstance(images_raw, list) or not images_raw:
        raise ValueError("input.images must be a non-empty list")

    images: list[InputImageConfig] = []
    for idx, item in enumerate(images_raw):
        if not isinstance(item, dict):
            raise ValueError(f"input.images[{idx}] must be an object")
        image_path = _to_path(base_dir, _require_string(item, "path"))
        band_id = _require_string(item, "band_id")
        images.append(InputImageConfig(path=image_path, band_id=band_id))

    prompt_cfg = PromptConfig(
        template_file=_to_path(base_dir, _require_string(prompt_raw, "template_file")),
        extra_instruction=str(prompt_raw.get("extra_instruction", "")),
    )

    matching_cfg = MatchingConfig(
        spatial_tolerance_px=float(matching_raw.get("spatial_tolerance_px", 20.0)),
        min_shared_bands=int(matching_raw.get("min_shared_bands", 2)),
        max_shared_bands=int(matching_raw.get("max_shared_bands", 3)),
    )

    formats_raw = output_raw.get("formats", ["json", "csv"])
    if not isinstance(formats_raw, list) or not formats_raw:
        raise ValueError("output.formats must be a non-empty list")

    output_cfg = OutputConfig(
        dir=_to_path(base_dir, _require_string(output_raw, "dir")),
        save_intermediate=bool(output_raw.get("save_intermediate", True)),
        formats=[str(x).strip().lower() for x in formats_raw],
    )

    cfg = RuntimeConfig(
        model=model_cfg,
        input=InputConfig(images=images),
        prompt=prompt_cfg,
        matching=matching_cfg,
        output=output_cfg,
        config_file=config_path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: RuntimeConfig) -> None:
    if cfg.model.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(SUPPORTED_DTYPES)}")

    if cfg.model.device not in SUPPORTED_DEVICES:
        raise ValueError(f"model.device must be one of {sorted(SUPPORTED_DEVICES)}")

    if cfg.model.max_new_tokens <= 0:
        raise ValueError("model.max_new_tokens must be > 0")

    if cfg.model.temperature < 0:
        raise ValueError("model.temperature must be >= 0")

    if not cfg.model.local_model_dir.exists() or not cfg.model.local_model_dir.is_dir():
        raise ValueError(f"model.local_model_dir must exist: {cfg.model.local_model_dir}")

    if not cfg.prompt.template_file.exists() or not cfg.prompt.template_file.is_file():
        raise ValueError(f"prompt.template_file must exist: {cfg.prompt.template_file}")

    seen_bands: set[str] = set()
    for image_cfg in cfg.input.images:
        if not image_cfg.path.exists() or not image_cfg.path.is_file():
            raise ValueError(f"Image path does not exist: {image_cfg.path}")
        if image_cfg.band_id in seen_bands:
            raise ValueError(f"Duplicated band_id: {image_cfg.band_id}")
        seen_bands.add(image_cfg.band_id)

    if cfg.matching.spatial_tolerance_px <= 0:
        raise ValueError("matching.spatial_tolerance_px must be > 0")

    if cfg.matching.min_shared_bands < 1:
        raise ValueError("matching.min_shared_bands must be >= 1")

    if cfg.matching.max_shared_bands < cfg.matching.min_shared_bands:
        raise ValueError("matching.max_shared_bands must be >= matching.min_shared_bands")

    unsupported = [fmt for fmt in cfg.output.formats if fmt not in SUPPORTED_FORMATS]
    if unsupported:
        raise ValueError(f"Unsupported output formats: {unsupported}")
