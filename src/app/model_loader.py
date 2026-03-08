from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import ModelConfig


def _import_runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required at runtime. Please install torch.") from exc

    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required at runtime. Please install transformers."
        ) from exc

    return torch, transformers


def _resolve_device(torch: Any, preferred: str) -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if preferred == "cpu":
        return "cpu"

    if preferred == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "Configured device='cuda' but CUDA is not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return "cuda"

    if preferred.startswith("cuda:"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"Configured device='{preferred}' but CUDA is not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        try:
            gpu_idx = int(preferred.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device string: {preferred}") from exc

        if gpu_idx < 0 or gpu_idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {gpu_idx}, but available count is {torch.cuda.device_count()}"
            )
        return preferred

    raise RuntimeError(f"Unsupported device setting: {preferred}")


def _resolve_dtype(torch: Any, dtype_name: str, resolved_device: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved = mapping[dtype_name]
    if dtype_name == "bfloat16" and resolved_device.startswith("cuda"):
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_supported) and not is_supported():
            warnings.warn(
                "Configured dtype='bfloat16' but current CUDA device does not support BF16. Falling back to float16.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.float16
    return resolved


def _pick_input_device(model: Any, fallback_device: str) -> str:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        device_str = str(model_device)
        if device_str and device_str != "meta":
            return device_str

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, str) and value.startswith("cuda"):
                return value
        return "cpu"

    return fallback_device


def _build_messages(prompt_text: str, image_paths: list[Path], use_uri: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        abs_path = image_path.resolve()
        image_ref = abs_path.as_uri() if use_uri else str(abs_path)
        content.append({"type": "image", "image": image_ref})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _normalize_device_map(device_map: str) -> Any:
    accelerate_literals = {"auto", "balanced", "balanced_low_0", "sequential"}
    if device_map in accelerate_literals:
        return device_map

    if device_map == "cpu" or device_map == "cuda" or device_map.startswith("cuda:"):
        return {"": device_map}

    raise RuntimeError(
        "Unsupported model.device_map value. Use one of auto/balanced/balanced_low_0/sequential/cpu/cuda/cuda:<index>."
    )


def _normalize_device_map_for_runtime(device_map: str, resolved_device: str) -> str:
    if resolved_device == "cpu" and (device_map == "cuda" or device_map.startswith("cuda:")):
        return "cpu"
    return device_map


@dataclass(slots=True)
class VLMEngine:
    model: Any
    processor: Any
    device: str
    torch: Any

    @classmethod
    def from_config(cls, model_cfg: ModelConfig) -> "VLMEngine":
        torch, transformers = _import_runtime_modules()
        preferred_device = _resolve_device(torch, model_cfg.device)
        torch_dtype = _resolve_dtype(torch, model_cfg.dtype, preferred_device)

        processor = transformers.AutoProcessor.from_pretrained(
            str(model_cfg.local_model_dir),
            local_files_only=True,
            trust_remote_code=model_cfg.trust_remote_code,
        )

        model_load_errors: list[str] = []
        model = None
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_cfg.trust_remote_code,
        }
        if model_cfg.device_map is not None:
            runtime_device_map = _normalize_device_map_for_runtime(
                model_cfg.device_map,
                preferred_device,
            )
            load_kwargs["device_map"] = _normalize_device_map(runtime_device_map)
        if model_cfg.attn_implementation is not None:
            load_kwargs["attn_implementation"] = model_cfg.attn_implementation

        candidate_names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
        for name in candidate_names:
            model_cls = getattr(transformers, name, None)
            if model_cls is None:
                continue
            try:
                model = model_cls.from_pretrained(
                    str(model_cfg.local_model_dir),
                    **load_kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                model_load_errors.append(f"{name}: {exc}")

        if model is None:
            joined = "\n".join(model_load_errors)
            raise RuntimeError(
                f"Failed to load model from {model_cfg.local_model_dir}. "
                "If this is a Qwen3.5 family model, upgrade transformers to latest compatible release.\n"
                f"{joined}"
            )

        if not hasattr(model, "hf_device_map"):
            model.to(preferred_device)

        model.eval()
        input_device = _pick_input_device(model, preferred_device)

        return cls(model=model, processor=processor, device=input_device, torch=torch)

    def generate_from_images(
        self,
        prompt_text: str,
        image_paths: list[Path],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        return self._generate(prompt_text, image_paths, max_new_tokens, temperature)

    def _generate(
        self,
        prompt_text: str,
        image_paths: list[Path],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        inputs = (
            self._build_inputs_with_chat_template(prompt_text, image_paths, use_uri=True)
            or self._build_inputs_with_chat_template(prompt_text, image_paths, use_uri=False)
            or self._build_legacy_inputs(prompt_text, image_paths)
        )

        inputs.pop("token_type_ids", None)
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)

        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        if "input_ids" in inputs and len(generated_ids) == len(inputs["input_ids"]):
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
        else:
            trimmed = generated_ids

        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise RuntimeError("Model returned empty output")
        return decoded[0].strip()

    def _build_inputs_with_chat_template(
        self,
        prompt_text: str,
        image_paths: list[Path],
        use_uri: bool,
    ) -> Any | None:
        if not hasattr(self.processor, "apply_chat_template"):
            return None

        messages = _build_messages(prompt_text, image_paths, use_uri=use_uri)
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:  # noqa: BLE001
            return None

    def _build_legacy_inputs(self, prompt_text: str, image_paths: list[Path]) -> Any:
        messages = [{
            "role": "user",
            "content": ([{"type": "image"} for _ in image_paths] + [{"type": "text", "text": prompt_text}]),
        }]

        rendered_prompt = prompt_text
        if hasattr(self.processor, "apply_chat_template"):
            try:
                rendered_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                rendered_prompt = prompt_text

        images = [Image.open(path).convert("RGB") for path in image_paths]
        try:
            return self.processor(
                text=[rendered_prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in images:
                image.close()

