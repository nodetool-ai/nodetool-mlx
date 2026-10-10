from __future__ import annotations
from nodetool.nodes.mlx._mlx_thread import MLX_EXECUTOR

import asyncio
import random
import sys
from enum import Enum, IntEnum
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HFFlux, HFTextToImage, ImageRef
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    import PIL.Image
    from mflux.models.ernie_image import ErnieImage
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.ideogram4 import Ideogram4

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
        "_ernie_model",
        "_ideogram_model",
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

        self._flux_model = await loop.run_in_executor(MLX_EXECUTOR, _load_model)

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
            pil_image = await loop.run_in_executor(MLX_EXECUTOR, _generate)
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


class MFluxErnieImage(BaseMFluxNode):
    """
    Generate images with Baidu's ERNIE-Image via MFLUX.
    mlx, ernie, ernie-image, text-to-image, apple-silicon

    Use cases:
    - Fast 8-step generation with ERNIE-Image-Turbo
    - Higher-fidelity generation with the non-distilled base model and guidance
    - Local 8B diffusion transformer generation on Apple Silicon
    """

    prompt: str = Field(
        default="Close-up portrait of a barn owl perched on a mossy branch, detailed feathers, soft forest bokeh",
        description="Text prompt describing the image to generate.",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt. Used only when guidance is above 1.0.",
    )
    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="mflux-community/ernie-image-turbo-mflux-q4"),
        description="ERNIE-Image checkpoint. Turbo runs in 8 steps, the base model needs about 50.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Quantization level for model weights. Pre-quantized checkpoints keep their stored level.",
    )
    steps: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Number of denoising steps. Use 8 for Turbo and about 50 for the base model.",
    )
    guidance: float = Field(
        default=1.0,
        ge=0.0,
        description="Guidance scale. Turbo uses 1.0, the base model about 4.0.",
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
    lora_path: str | None = Field(
        default=None,
        description="Optional path or HuggingFace repo ID for an ERNIE-Image LoRA adapter.",
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scale factor for the LoRA adapter.",
    )

    _ernie_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux ERNIE-Image"

    def required_inputs(self):
        return ["prompt"]

    def _is_turbo(self) -> bool:
        return "turbo" in self.model.repo_id.lower()

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux ERNIE-Image requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        lora_key = self.lora_path or "none"
        cache_key = f"{self.model.repo_id}_{lora_key}_ernie-image_q{quantize_value}"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._ernie_model = model
            return

        loop = asyncio.get_running_loop()
        is_turbo = self._is_turbo()

        def _load_model() -> "ErnieImage":
            from mflux.models.common.config import ModelConfig
            from mflux.models.ernie_image import ErnieImage

            log.info(
                "Loading MFlux ERNIE-Image model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            # The named configs carry the sigma shift and rope overrides that a
            # config inferred from the repo id (ModelConfig.from_name) drops.
            model_config = (
                ModelConfig.ernie_image_turbo()
                if is_turbo
                else ModelConfig.ernie_image()
            )
            lora_paths = [self.lora_path] if self.lora_path else None
            lora_scales = [self.lora_scale] if self.lora_path else None
            model = ErnieImage(
                quantize=quantize_value,
                model_path=self.model.repo_id,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                model_config=model_config,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._ernie_model = await loop.run_in_executor(MLX_EXECUTOR, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux ERNIE-Image requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for ERNIE-Image generation."
        )
        self._ensure_seed()

        assert self._ernie_model is not None

        loop = asyncio.get_running_loop()
        progress_callback = self._register_progress_callback(context, self.steps)
        negative_prompt = self.negative_prompt.strip() or None

        def _generate() -> "PIL.Image.Image":
            assert self._ernie_model is not None
            generated_image = self._ernie_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps,
                height=16 * (self.height // 16),
                width=16 * (self.width // 16),
                guidance=self.guidance,
                negative_prompt=negative_prompt,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(MLX_EXECUTOR, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(repo_id="mflux-community/ernie-image-turbo-mflux-q4"),
            HFTextToImage(repo_id="mflux-community/ernie-image-turbo-mflux-q6"),
            HFTextToImage(repo_id="mflux-community/ernie-image-turbo-mflux-q8"),
            HFTextToImage(repo_id="mflux-community/ernie-image-base-mflux-q4"),
            HFTextToImage(repo_id="mflux-community/ernie-image-base-mflux-q8"),
        ]


class Ideogram4Preset(str, Enum):
    """Sampler presets shipped with Ideogram 4 (steps and guidance schedule)."""

    TURBO_12 = "V4_TURBO_12"
    DEFAULT_20 = "V4_DEFAULT_20"
    QUALITY_48 = "V4_QUALITY_48"


class MFluxIdeogram4(BaseMFluxNode):
    """
    Generate typography-heavy images with Ideogram 4 via MFLUX.
    mlx, ideogram, ideogram4, text-to-image, typography, apple-silicon

    Use cases:
    - Posters, labels and layouts with legible rendered text
    - Structured JSON captions with bounding boxes and color palettes
    - Local Apple Silicon generation without the Ideogram API

    The prompt accepts plain text or an Ideogram JSON caption. JSON captions
    give much better results; see Ideogram's prompting guide for the schema.
    """

    prompt: str = Field(
        default="A vintage travel poster for the city of Lisbon with the headline 'LISBOA' in bold art deco letters",
        description="Plain text prompt or an Ideogram 4 JSON caption.",
    )
    model: HFTextToImage = Field(
        default=HFTextToImage(repo_id="mflux-community/ideogram-4-mflux-q4"),
        description="Ideogram 4 checkpoint to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Quantization level for model weights. Pre-quantized checkpoints keep their stored level.",
    )
    preset: Ideogram4Preset = Field(
        default=Ideogram4Preset.DEFAULT_20,
        description="Sampler preset. Sets the step count and guidance schedule unless steps is set.",
    )
    steps: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Override the preset step count. 0 uses the preset.",
    )
    guidance: float = Field(
        default=7.0,
        ge=0.0,
        description="Constant guidance scale. Only used when steps overrides the preset.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Height of the generated image in pixels (rounded down to a multiple of 16).",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Width of the generated image in pixels (rounded down to a multiple of 16).",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    _ideogram_model: Any | None = None

    _PRESET_STEPS: ClassVar[dict[str, int]] = {
        Ideogram4Preset.TURBO_12.value: 12,
        Ideogram4Preset.DEFAULT_20.value: 20,
        Ideogram4Preset.QUALITY_48.value: 48,
    }

    @classmethod
    def get_title(cls):
        return "MFlux Ideogram 4"

    def required_inputs(self):
        return ["prompt"]

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Ideogram 4 requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_ideogram4_q{quantize_value}"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._ideogram_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Ideogram4":
            from mflux.models.common.config import ModelConfig
            from mflux.models.ideogram4 import Ideogram4

            log.info(
                "Loading MFlux Ideogram 4 model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Ideogram4(
                quantize=quantize_value,
                model_path=self.model.repo_id,
                model_config=ModelConfig.ideogram4_fp8(),
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._ideogram_model = await loop.run_in_executor(MLX_EXECUTOR, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Ideogram 4 requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Ideogram 4 generation."
        )
        self._ensure_seed()

        assert self._ideogram_model is not None

        loop = asyncio.get_running_loop()
        preset = Ideogram4Preset(self.preset).value
        total_steps = self.steps or self._PRESET_STEPS[preset]
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> "PIL.Image.Image":
            assert self._ideogram_model is not None
            generated_image = self._ideogram_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps or None,
                height=16 * (self.height // 16),
                width=16 * (self.width // 16),
                guidance=self.guidance if self.steps else None,
                preset=preset,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(MLX_EXECUTOR, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(repo_id="mflux-community/ideogram-4-mflux-q4"),
            HFTextToImage(repo_id="mflux-community/ideogram-4-mflux-q6"),
            HFTextToImage(repo_id="mflux-community/ideogram-4-mflux-q8"),
        ]
