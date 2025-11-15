import asyncio
import contextlib
import random
import sys
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar
import PIL.Image
from mflux.config.config import Config
import PIL.Image
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Flux1
from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.flux_tools.depth.flux_depth import Flux1Depth
from mflux.flux_tools.redux.flux_redux import Flux1Redux
from mflux.kontext.flux_kontext import Flux1Kontext
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.callback import InLoopCallback
import PIL.Image
import numpy as np
from mflux.config.config import Config
from mflux.post_processing.image_util import ImageUtil
from mflux.ui.box_values import BoxValues, parse_box_value
from mflux.config.model_config import ModelConfig
from nodetool.ml.core.model_manager import ModelManager
from pydantic import Field
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    HFFlux,
    HFControlNet,
    HFControlNetFlux,
    HFDepthGeneration,
    HFReduxGeneration,
    HFKontextGeneration,
    HFInpainting,
    HFOutpainting,
    HuggingFaceModel,
    ImageRef,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

log = get_logger(__name__)


class QuantizationLevel(IntEnum):
    BITS_3 = 3
    BITS_4 = 4
    BITS_5 = 5
    BITS_6 = 6
    BITS_8 = 8


class BaseMFluxNode(BaseNode):
    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseMFluxNode

    @staticmethod
    def _ensure_supported_platform(message: str) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(message)

    def _ensure_seed(self) -> None:
        if hasattr(self, "seed") and getattr(self, "seed") == 0:
            self.seed = random.randint(0, 2**32 - 1)

    @staticmethod
    def _require_prompt(prompt: str, message: str) -> None:
        if not prompt.strip():
            raise ValueError(message)

    def _register_progress_callback(
        self,
        context: ProcessingContext,
        total_steps: int,
    ) -> InLoopCallback:
        node_id = self.id

        class Callback(InLoopCallback):
            def call_in_loop(
                self,
                t: int,
                seed: int,
                prompt: str,
                latents,
                config,
                time_steps,
            ):
                context.post_message(
                    NodeProgress(
                        node_id=node_id,
                        progress=t,
                        total=total_steps,
                    )
                )

        callback = Callback()
        CallbackRegistry.register_in_loop(callback)
        return callback

    @staticmethod
    def _remove_progress_callback(callback: InLoopCallback) -> None:
        with contextlib.suppress(ValueError):
            CallbackRegistry.in_loop_callbacks().remove(callback)


class MFlux(BaseMFluxNode):
    """
    Generate images locally using the MFLUX MLX implementation of FLUX.1.
    mlx, flux, image generation, apple-silicon

    Use cases:
    - Create high quality images on Apple Silicon without external APIs
    - Prototype prompts locally before running on cloud inference providers
    - Experiment with quantized FLUX models (schnell/dev/krea-dev variants)

    Recommended models:
    - schnell: Fastest model, good for quick generations (2-4 steps)
    - dev: More powerful model, higher quality (20-25 steps)
    - krea-dev: Enhanced photorealism with distinctive aesthetics
    - Quantized 4-bit models: Reduced memory usage versions of the official models
    """

    prompt: str = Field(
        default="A vivid concept art piece of a futuristic city at sunset",
        description="The text prompt describing the image to generate.",
    )
    model: HFFlux = Field(
        default=HFFlux(
            repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
        ),
        description="MFLUX model variant to load"
    )
    quantize: QuantizationLevel = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of denoising steps for the generation run.",
    )
    guidance: float | None = Field(
        default=3.5,
        ge=0.0,
        description="Classifier-free guidance scale. Used by dev/krea-dev models.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Height of the generated image in pixels.",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Width of the generated image in pixels.",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    _flux_model: Flux1 | None = None

    @classmethod
    def get_title(cls):
        return "MFlux"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1:
            log.info(
                "Loading MFlux model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1.from_name(
                model_name=self.model.repo_id,
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, self.model.repo_id, "flux", model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for image generation."
        )
        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            config_kwargs: dict[str, Any] = {
                "num_inference_steps": self.steps,
                "height": self.height,
                "width": self.width,
            }
            if self.guidance is not None:
                config_kwargs["guidance"] = self.guidance

            dataclass_fields = getattr(Config, "__dataclass_fields__", None)
            if isinstance(dataclass_fields, dict):
                allowed = set(dataclass_fields.keys())
                config_kwargs = {
                    key: value for key, value in config_kwargs.items() if key in allowed
                }

            config = Config(**config_kwargs)

            assert self._flux_model is not None
            assert isinstance(self._flux_model, Flux1)

            generated_image = self._flux_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                config=config,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)
        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"),
            HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
            HFFlux(repo_id="filipstrand/FLUX.1-Krea-dev-mflux-4bit"),
            HFFlux(repo_id="akx/FLUX.1-Kontext-dev-mflux-4bit"),
        ]
