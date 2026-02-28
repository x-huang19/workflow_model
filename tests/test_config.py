from __future__ import annotations

from pathlib import Path

import pytest

from app.config import load_config


def test_load_config_success(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    images_dir = tmp_path / "data"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"x")
    (images_dir / "b.png").write_bytes(b"y")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "prompt.txt").write_text("test", encoding="utf-8")

    config_file = tmp_path / "runtime.yaml"
    config_file.write_text(
        """
model:
  local_model_dir: "model"
  dtype: "float16"
  device: "cpu"
input:
  images:
    - path: "data/a.png"
      band_id: "10k"
    - path: "data/b.png"
      band_id: "12k"
prompt:
  template_file: "prompts/prompt.txt"
matching:
  spatial_tolerance_px: 20
  min_shared_bands: 2
  max_shared_bands: 3
output:
  dir: "outputs"
  save_intermediate: true
  formats: ["json", "csv"]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.model.local_model_dir == model_dir.resolve()
    assert len(cfg.input.images) == 2
    assert cfg.output.dir == (tmp_path / "outputs").resolve()


def test_load_config_invalid_matching_bounds(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test", encoding="utf-8")

    config_file = tmp_path / "runtime.yaml"
    config_file.write_text(
        f"""
model:
  local_model_dir: "{model_dir.as_posix()}"
input:
  images:
    - path: "{img.as_posix()}"
      band_id: "10k"
prompt:
  template_file: "{prompt.as_posix()}"
matching:
  spatial_tolerance_px: 20
  min_shared_bands: 3
  max_shared_bands: 2
output:
  dir: "outputs"
  save_intermediate: true
  formats: ["json"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_shared_bands"):
        load_config(config_file)
