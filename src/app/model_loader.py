from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ModelConfig


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

    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is not available")

    return preferred


def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


@dataclass(slots=True)
class VLMEngine:
    model: Any
    processor: Any
    device: str
    torch: Any

    @classmethod
    def from_config(cls, model_cfg: ModelConfig) -> "VLMEngine":
        torch, transformers = _import_runtime_modules()
        device = _resolve_device(torch, model_cfg.device)
        torch_dtype = _resolve_dtype(torch, model_cfg.dtype)

        processor = transformers.AutoProcessor.from_pretrained(
            str(model_cfg.local_model_dir),
            local_files_only=True,
            trust_remote_code=model_cfg.trust_remote_code,
        )

        model_load_errors: list[str] = []
        model = None

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
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                    trust_remote_code=model_cfg.trust_remote_code,
                )
                break
            except Exception as exc:  # noqa: BLE001
                model_load_errors.append(f"{name}: {exc}")

        if model is None:
            joined = "\n".join(model_load_errors)
            raise RuntimeError(f"Failed to load model from {model_cfg.local_model_dir}\n{joined}")

        model.to(device)
        model.eval()

        return cls(model=model, processor=processor, device=device, torch=torch)

    def generate_from_images(
        self,
        prompt_text: str,
        image_paths: list[Path],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        try:
            return self._generate(prompt_text, images, max_new_tokens, temperature)
        finally:
            for image in images:
                image.close()

    def _generate(
        self,
        prompt_text: str,
        images: list[Image.Image],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        content: list[dict[str, Any]] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        inputs = None

        if hasattr(self.processor, "apply_chat_template"):
            try:
                rendered_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=[rendered_prompt],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
            except Exception:  # noqa: BLE001
                inputs = None

        if inputs is None:
            inputs = self.processor(
                text=[prompt_text],
                images=images,
                return_tensors="pt",
                padding=True,
            )

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
