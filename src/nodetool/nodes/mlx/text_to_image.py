from __future__ import annotations

import asyncio
import random
import sys
from enum import IntEnum
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HFFlux, ImageRef
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    import PIL.Image
    from mflux.models.flux.variants.txt2img.flux import Flux1

log = get_logger(__name__)


class QuantizationLevel(IntEnum):
    BITS_3 = 3
    BITS_4 = 4
    BITS_5 = 5
    BITS_6 = 6
    BITS_8 = 8


class BaseMFluxNode(BaseNode):
    _expose_as_tool: ClassVar[bool] = True
    _body: ClassVar[str] = "content_card"

    # Handles match HuggingFace image nodes: primary text plus asset inputs.
    _DATA_INPUT_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"prompt", "negative_prompt"}
    )
    _INLINE_NAMES: ClassVar[frozenset[str]] = frozenset({"model", "prompt"})
    _BASIC_FIELD_ORDER: ClassVar[tuple[str, ...]] = (
        "model",
        "controlnet_model",
        "prompt",
        "image",
        "images",
        "mask",
        "control_image",
        "depth_image",
        "redux_image",
        "reference_image",
        "width",
        "height",
        "steps",
        "resolution",
        "style",
    )

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseMFluxNode

    @classmethod
    def get_input_fields(cls) -> list[str]:
        names: list[str] = []
        for prop in cls.properties():
            hint = cls._expose_hint(prop)
            if hint in ("inline", "none"):
                continue
            if hint in ("handle", "both"):
                names.append(prop.name)
                continue
            if prop.type.is_asset_type(recursive=True):
                names.append(prop.name)
            elif prop.name in cls._DATA_INPUT_NAMES:
                names.append(prop.name)
        return names

    @classmethod
    def get_inline_fields(cls) -> list[str]:
        names: list[str] = []
        for prop in cls.properties():
            hint = cls._expose_hint(prop)
            if hint in ("inline", "both"):
                names.append(prop.name)
                continue
            if hint in ("handle", "none"):
                continue
            if prop.name in cls._INLINE_NAMES:
                names.append(prop.name)
        return names

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        available = {prop.name for prop in cls.properties()}
        return [name for name in cls._BASIC_FIELD_ORDER if name in available]

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

    # Private attributes that may hold the loaded MFLUX model across the various
    # node families. Each model instance owns its own ``CallbackRegistry`` at
    # ``model.callbacks`` (mflux >= 0.17), which is the registry the generation
    # loop iterates.
    _MODEL_ATTRS: ClassVar[tuple[str, ...]] = (
        "_flux_model",
        "_flux2_model",
        "_fibo_model",
        "_qwen_model",
        "_zimage_model",
        "_seedvr2_model",
        "_krea2_model",
    )

    def _active_model(self) -> Any | None:
        for attr in self._MODEL_ATTRS:
            model = getattr(self, attr, None)
            if model is not None:
                return model
        return None

    def _register_progress_callback(
        self,
        context: ProcessingContext,
        total_steps: int,
    ) -> Any:
        # mflux registers callbacks per-model on ``model.callbacks``; a callback
        # is any object exposing ``call_in_loop`` (duck-typed by the registry).
        model = self._active_model()
        registry = getattr(model, "callbacks", None) if model is not None else None
        if registry is None:
            return None

        node_id = self.id

        class Callback:
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
        registry.register(callback)
        return callback

    def _remove_progress_callback(self, callback: Any) -> None:
        # Called from a ``finally`` block, so it must never raise: the model is
        # cached in the ModelManager and outlives the run, and letting an
        # unregister failure propagate would discard the generated image.
        if callback is None:
            return
        model = self._active_model()
        registry = getattr(model, "callbacks", None) if model is not None else None
        if registry is None:
            return
        try:
            registry.in_loop_callbacks().remove(callback)
        except Exception:  # pragma: no cover - depends on the mflux version
            log.debug(
                "Could not unregister the MFlux progress callback; it will be "
                "dropped on the next model load.",
                exc_info=True,
            )


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
            repo_id="mflux-community/flux-1-schnell-mflux-q4",
        ),
        description="MFLUX model variant to load",
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux"

    def required_inputs(self):
        return ["prompt"]

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux_q{quantize_value}"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1":
            from mflux.models.flux.variants.txt2img.flux import Flux1

            log.info(
                "Loading MFlux model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1.from_name(
                model_name=self.model.repo_id,
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, cache_key, model)
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

        def _generate() -> "PIL.Image.Image":
            from mflux.models.flux.variants.txt2img.flux import Flux1

            assert self._flux_model is not None
            assert isinstance(self._flux_model, Flux1)

            generated_image = self._flux_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
                guidance=self.guidance,
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
            HFFlux(repo_id="mflux-community/flux-1-schnell-mflux-q4"),
            HFFlux(repo_id="mflux-community/flux-1-schnell-mflux-q8"),
            HFFlux(repo_id="mflux-community/flux-1-dev-mflux-q4"),
            HFFlux(repo_id="mflux-community/flux-1-dev-mflux-q8"),
            HFFlux(repo_id="mflux-community/flux-1-krea-dev-mflux-q4"),
            HFFlux(repo_id="mflux-community/flux-1-krea-dev-mflux-q8"),
        ]
