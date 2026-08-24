from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HFImageTextToText, ImageRef
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class MLXVisionLanguage(BaseNode):
    """
    Describe or answer questions about an image using MLX vision-language models.
    mlx, vlm, image-to-text, vision, captioning, vqa, apple-silicon

    Use cases:
    - Generate captions or detailed descriptions for an image
    - Visual question answering (VQA) about the contents of an image
    - OCR-style reading and document understanding without external APIs
    - Local multimodal understanding on Apple Silicon (Qwen3-VL, Gemma 4, etc.)
    """

    _body: ClassVar[str] = "content_card"
    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="Describe this image in detail.",
        description="Instruction or question about the image.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Image to analyze.",
    )
    model: HFImageTextToText = Field(
        default=HFImageTextToText(repo_id="mlx-community/Qwen3-VL-4B-Instruct-4bit"),
        description="MLX vision-language model to load from the local Hugging Face cache.",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum number of tokens the model should generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Set to 0 for deterministic output.",
    )

    _vlm: Any | None = None

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image", "model", "max_tokens", "temperature"]

    @classmethod
    def get_title(cls) -> str:
        return "MLX Vision Language"

    def required_inputs(self):
        return ["image", "prompt"]

    @classmethod
    def get_input_fields(cls):
        return ["prompt", "image"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "prompt"]

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        # MLX runs on Apple Silicon accelerators but does not require an external GPU.
        return False

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "MLX vision-language generation requires macOS (Apple Silicon / MLX)."
            )

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()

        model_id = self.model.repo_id
        if not model_id:
            raise ValueError("Select an MLX vision-language model before running.")

        cache_key = f"{model_id}_mlx-vlm"

        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            self._vlm = cached
            return

        from huggingface_hub import try_to_load_from_cache

        if not try_to_load_from_cache(model_id, "config.json"):
            raise ValueError(
                f"Model {model_id} must be downloaded first, check recommended models"
            )

        loop = asyncio.get_running_loop()

        def _load_model() -> tuple[Any, Any, Any]:
            import mlx_vlm

            log.info("Loading MLX-VLM model %s", model_id)
            mdl, proc = mlx_vlm.load(model_id)
            cfg = getattr(mdl, "config", None)
            if cfg is None:
                cfg = mlx_vlm.utils.load_config(model_id)
            return mdl, proc, cfg

        self._vlm = await loop.run_in_executor(None, _load_model)
        ModelManager.set_model(self.id, cache_key, self._vlm)

    async def process(self, context: ProcessingContext) -> str:
        self._ensure_supported_platform()

        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty for vision-language generation.")
        if not self.image.is_set():
            raise ValueError(
                "An input image is required for vision-language generation."
            )

        if self._vlm is None:
            await self.preload_model(context)
        assert self._vlm is not None

        pil_image = await context.image_to_pil(self.image)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_path = Path(tmp.name)
        tmp.close()
        pil_image.convert("RGB").save(image_path)

        loop = asyncio.get_running_loop()

        def _generate() -> str:
            import mlx_vlm

            mdl, proc, cfg = self._vlm
            formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
                proc,
                cfg,
                self.prompt,
                num_images=1,
            )

            generate_fn = mlx_vlm.generate.generate
            sig_params = set(inspect.signature(generate_fn).parameters)
            gen_kwargs: dict[str, Any] = {"verbose": False}
            if "max_tokens" in sig_params:
                gen_kwargs["max_tokens"] = self.max_tokens
            # mlx-vlm has used both "temperature" and "temp" across versions.
            if "temperature" in sig_params:
                gen_kwargs["temperature"] = self.temperature
            elif "temp" in sig_params:
                gen_kwargs["temp"] = self.temperature

            result = generate_fn(
                mdl,
                proc,
                formatted_prompt,
                image=[str(image_path)],
                **gen_kwargs,
            )
            # mlx-vlm may return either a result object exposing ``.text`` or a string.
            return getattr(result, "text", result)

        try:
            text = await loop.run_in_executor(None, _generate)
        finally:
            with contextlib.suppress(FileNotFoundError):
                image_path.unlink()

        return text if isinstance(text, str) else str(text)

    @classmethod
    def get_recommended_models(cls) -> list[HFImageTextToText]:
        return [
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-2B-Instruct-4bit"),
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-4B-Instruct-4bit"),
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-4B-Instruct-8bit"),
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-8B-Instruct-4bit"),
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-8B-Instruct-8bit"),
            HFImageTextToText(repo_id="mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"),
            HFImageTextToText(repo_id="mlx-community/gemma-4-e4b-it-4bit"),
            HFImageTextToText(repo_id="mlx-community/gemma-4-12B-it-4bit"),
        ]
