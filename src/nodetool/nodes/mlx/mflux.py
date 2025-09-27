from __future__ import annotations

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
    - Freepik/flux.1-lite-8B-alpha: Lighter version of FLUX
    - Quantized 4-bit models: Reduced memory usage versions of the official models
    """

    prompt: str = Field(
        default="A vivid concept art piece of a futuristic city at sunset",
        description="The text prompt describing the image to generate.",
    )
    model: HFFlux = Field(
        default=HFFlux(
            repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
            path=None,
        ),
        description="MFLUX model variant to load. Options include official models (schnell, dev, krea-dev), third-party community models (Freepik/flux.1-lite-8B-alpha), and quantized 4-bit versions for reduced memory usage.",
    )
    quantize: QuantizationLevel | None = Field(
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
            HFFlux(repo_id="Freepik/flux.1-lite-8B-alpha"),
            HFFlux(repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"),
            HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
            HFFlux(repo_id="filipstrand/FLUX.1-Krea-dev-mflux-4bit"),
            HFFlux(repo_id="akx/FLUX.1-Kontext-dev-mflux-4bit"),
        ]


class MFluxImageToImage(BaseMFluxNode):
    """
    Transform an existing image using the MFLUX MLX implementation of FLUX.1.
    mlx, flux, image-to-image, apple-silicon

    Use cases:
    - Apply prompt-based edits to an existing image without relying on external APIs
    - Experiment with strength-controlled transformations locally
    """

    prompt: str = Field(
        default="Refine this image with cinematic lighting",
        description="Text prompt describing how to transform the input image.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Base image that will be transformed.",
    )
    model: HFFlux = Field(
        default=HFFlux(
            repo_id="dhairyashil/FLUX.1-dev-mflux-4bit",
            path=None,
        ),
        description="MFLUX model variant to load for image-to-image generation.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of denoising steps for the transformation.",
    )
    guidance: float | None = Field(
        default=3.5,
        ge=0.0,
        description="Classifier-free guidance scale. Used by dev/krea-dev models.",
    )
    image_strength: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Blend factor between the original image and the generation (0 keeps original).",
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
        return "MFlux ImageToImage"

    async def preload_model(self, context: ProcessingContext) -> None:
        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1:
            log.info(
                "Loading MFlux image-to-image model %s (quantize=%s)",
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
            "MFlux image-to-image requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for image-to-image generation."
        )

        base_image = await context.image_to_pil(self.image)

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            working_image = base_image.convert("RGB")
            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)

            if working_image.size != (target_width, target_height):
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = Path(tmp.name)
                working_image.save(image_path)

            try:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "image_strength": float(self.image_strength),
                    "image_path": image_path,
                }
                if self.guidance is not None:
                    config_kwargs["guidance"] = self.guidance

                dataclass_fields = getattr(Config, "__dataclass_fields__", None)
                if isinstance(dataclass_fields, dict):
                    allowed = set(dataclass_fields.keys())
                    config_kwargs = {
                        key: value
                        for key, value in config_kwargs.items()
                        if key in allowed
                    }

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated_image = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated_image.image
            finally:
                with contextlib.suppress(FileNotFoundError):
                    image_path.unlink()

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)
        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
            HFFlux(repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"),
            HFFlux(repo_id="filipstrand/FLUX.1-Krea-dev-mflux-4bit"),
        ]


class MFluxControlNet(BaseMFluxNode):
    """
    Generate images with MFlux ControlNet guidance using local MLX acceleration.
    mlx, flux, controlnet, conditioning, edge-detection

    Use cases:
    - Apply edge-aware guidance via ControlNet canny models
    - Leverage local Apple Silicon acceleration for conditioned generations
    - Upscale images using ControlNet upscaler weights
    """

    prompt: str = Field(
        default="Highly detailed cinematic portrait",
        description="Primary text prompt for image generation.",
    )
    control_image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image used by ControlNet for conditioning.",
    )
    model: HFFlux = Field(
        default=HFFlux(
            repo_id="dhairyashil/FLUX.1-dev-mflux-4bit",
            path=None,
        ),
        description="Base Flux model to load for conditioned generation.",
    )
    controlnet_model: HFControlNetFlux = Field(
        default=HFControlNetFlux(repo_id="InstantX/FLUX.1-dev-Controlnet-Canny"),
        description="ControlNet weights that match the selected Flux base model.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of denoising steps for diffusion.",
    )
    guidance: float | None = Field(
        default=3.5,
        ge=0.0,
        description="Classifier-free guidance scale when supported by the selected model.",
    )
    controlnet_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Blend factor between ControlNet conditioning and base model prior.",
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
        description="Seed for deterministic generation. Leave 0 for random.",
    )

    _flux_model: Flux1Controlnet | None = None

    @classmethod
    def get_title(cls):
        return "MFlux ControlNet"

    async def preload_model(self, context: ProcessingContext) -> None:
        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Controlnet":
            log.info(
                "Loading MFlux ControlNet model %s with controlnet %s (quantize=%s)",
                self.model.repo_id,
                self.controlnet_model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )

            model_config = ModelConfig.from_name(self.model.repo_id)
            model_config.controlnet_model = self.controlnet_model.repo_id

            model = Flux1Controlnet(
                model_config=model_config,
                quantize=quantize_value,
            )
            ModelManager.set_model(
                self.id,
                f"{self.model.repo_id}:{self.controlnet_model.repo_id}",
                "flux-controlnet",
                model,
            )
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux ControlNet requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for ControlNet generation."
        )

        control_image = await context.image_to_pil(self.control_image)

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            from mflux.config.config import Config

            config_kwargs: dict[str, Any] = {
                "num_inference_steps": self.steps,
                "height": self.height,
                "width": self.width,
                "controlnet_strength": float(self.controlnet_strength),
            }
            if self.guidance is not None:
                config_kwargs["guidance"] = self.guidance

            config = Config(**config_kwargs)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                control_path = Path(tmp.name)
                control_image.save(control_path)

            try:
                assert self._flux_model is not None
                generated = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    controlnet_image_path=str(control_path),
                    config=config,
                )
            finally:
                with contextlib.suppress(FileNotFoundError):
                    control_path.unlink()

            return generated.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
            HFControlNetFlux(repo_id="InstantX/FLUX.1-dev-Controlnet-Canny"),
            HFControlNetFlux(repo_id="jasperai/Flux.1-dev-Controlnet-Upscaler"),
        ]


class MFluxInpaint(BaseMFluxNode):
    """
    Inpaint portions of an image locally using the MFLUX MLX implementation of FLUX.1 Fill.
    mlx, flux, inpainting, mask editing

    Use cases:
    - Restore masked regions with prompt-guided content
    - Blend new elements into an existing composition while preserving unmasked areas
    """

    prompt: str = Field(
        default="Refine the masked region with additional details",
        description="Text prompt describing what to generate inside the mask.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Base image that will stay fixed outside the masked regions.",
    )
    mask: ImageRef = Field(
        default=ImageRef(),
        description="Mask image: white areas will be regenerated, black areas remain untouched.",
    )
    model: HFInpainting = Field(
        default=HFInpainting(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
        description="Inpainting model to load. Defaults to FLUX.1 Fill dev weights.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the inpainting run.",
    )
    guidance: float | None = Field(
        default=30.0,
        ge=0.0,
        description="Classifier-free guidance scale. Higher values tend to better respect the prompt in Fill mode.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Target output height in pixels.",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Target output width in pixels.",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave 0 for random seed.",
    )

    _flux_model: Flux1Fill | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Inpaint"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux inpainting requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux-fill")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1Fill:
            log.info(
                "Loading MFlux Fill model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Fill(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, self.model.repo_id, "flux-fill", model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux inpainting requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(self.prompt, "Prompt cannot be empty for inpainting.")

        base_image = await context.image_to_pil(self.image)
        mask_image = await context.image_to_pil(self.mask)

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)

            working_image = base_image.convert("RGB")
            if working_image.size != (target_width, target_height):
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )

            working_mask = mask_image.convert("L")
            if working_mask.size != (target_width, target_height):
                working_mask = working_mask.resize(
                    (target_width, target_height), PIL.Image.Resampling.NEAREST
                )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_tmp:
                image_path = Path(image_tmp.name)
                working_image.save(image_path)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as mask_tmp:
                mask_path = Path(mask_tmp.name)
                working_mask.save(mask_path)

            try:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "guidance": self.guidance,
                    "image_path": image_path,
                    "masked_image_path": mask_path,
                }

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated_image = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated_image.image
            finally:
                with contextlib.suppress(FileNotFoundError):
                    image_path.unlink()
                with contextlib.suppress(FileNotFoundError):
                    mask_path.unlink()

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFInpainting]:
        return [
            HFInpainting(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
        ]


class MFluxOutpaint(BaseMFluxNode):
    """
    Outpaint an existing image by extending the canvas using the MFLUX Fill pipeline.
    mlx, flux, outpainting, canvas extension

    Use cases:
    - Expand scene borders while maintaining continuity with the original image
    - Add sky, foreground elements, or contextual scenery around a provided image
    """

    prompt: str = Field(
        default="Expand the scene with complementary surroundings",
        description="Prompt guiding what to generate in the newly added canvas regions.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Base image that will remain visible inside the padded region.",
    )
    mask: ImageRef = Field(
        default=ImageRef(),
        description="Mask defining areas to regenerate (white) after padding. If blank, generated automatically.",
    )
    model: HFOutpainting = Field(
        default=HFOutpainting(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
        description="Outpainting model to load. Defaults to FLUX.1 Fill dev weights.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the outpainting run.",
    )
    guidance: float | None = Field(
        default=30.0,
        ge=0.0,
        description="Classifier-free guidance scale. Higher values tend to better respect the prompt in Fill mode.",
    )
    padding: str | None = Field(
        default=None,
        description="CSS-style padding string (e.g. '128', '96,64', '10%,5%') describing additional canvas to create before generation.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Target output height after padding.",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Target output width after padding.",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave 0 for random seed.",
    )

    _flux_model: Flux1Fill | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Outpaint"

    async def preload_model(self, context: ProcessingContext):
        self._ensure_supported_platform(
            "MFlux outpainting requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(self.prompt, "Prompt cannot be empty for outpainting.")

        base_image = await context.image_to_pil(self.image)
        existing_mask = await context.image_to_pil(self.mask)

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            working_image = base_image.convert("RGB")

            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)

            if working_image.size != (target_width, target_height):
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )

            # Prepare mask: if empty, generate from padding
            mask_candidate = existing_mask
            if mask_candidate.size != (target_width, target_height):
                mask_candidate = mask_candidate.resize(
                    (target_width, target_height), PIL.Image.Resampling.NEAREST
                )

            mask_array = np.array(mask_candidate.convert("L"))
            if not mask_array.any():
                if not self.padding:
                    raise ValueError(
                        "Outpainting requires either a mask or padding to expand the canvas."
                    )
                padding_values: BoxValues = parse_box_value(self.padding)
                abs_padding = padding_values.normalize_to_dimensions(
                    target_width, target_height
                )
                expanded = ImageUtil.expand_image(
                    image=working_image,
                    top=abs_padding.top,
                    right=abs_padding.right,
                    bottom=abs_padding.bottom,
                    left=abs_padding.left,
                )
                canvas_width, canvas_height = expanded.size
                mask_candidate = ImageUtil.create_outpaint_mask_image(
                    orig_width=working_image.width,
                    orig_height=working_image.height,
                    top=abs_padding.top,
                    right=abs_padding.right,
                    bottom=abs_padding.bottom,
                    left=abs_padding.left,
                )
                mask_candidate = mask_candidate.resize(
                    (canvas_width, canvas_height), PIL.Image.Resampling.NEAREST
                )
                working_image_resized = expanded
                target_width, target_height = canvas_width, canvas_height
            else:
                working_image_resized = working_image

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_tmp:
                image_path = Path(image_tmp.name)
                working_image_resized.save(image_path)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as mask_tmp:
                mask_path = Path(mask_tmp.name)
                mask_candidate.save(mask_path)

            try:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "guidance": self.guidance,
                    "image_path": image_path,
                    "masked_image_path": mask_path,
                }

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated_image = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated_image.image
            finally:
                with contextlib.suppress(FileNotFoundError):
                    image_path.unlink()
                with contextlib.suppress(FileNotFoundError):
                    mask_path.unlink()

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFOutpainting]:
        return [
            HFOutpainting(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
        ]


class MFluxDepth(BaseMFluxNode):
    """
    Generate images with depth guidance via the MFlux depth pipeline using local MLX acceleration.
    mlx, flux, depth, conditioning, structure-preserving

    Use cases:
    - Use a depth map to control structural composition while keeping prompt-driven appearance
    - Provide both source image and depth map to transfer scene layout to a new generation
    - Generate depth-guided outputs when only a depth map is available (source image optional)
    """

    prompt: str = Field(
        default="Highly detailed cinematic portrait with depth cues",
        description="Primary text prompt for the depth-guided generation.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional reference image used for depth extraction or as a content guide.",
    )
    depth_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional depth map to guide geometry. If omitted, depth is inferred from the image when provided.",
    )
    model: HFDepthGeneration = Field(
        default=HFDepthGeneration(repo_id="black-forest-labs/FLUX.1-Depth-dev"),
        description="Depth model weights compatible with the Flux depth pipeline.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the generation run.",
    )
    guidance: float | None = Field(
        default=10.0,
        ge=0.0,
        description="Classifier-free guidance scale. Defaults higher to encourage prompt adherence in depth mode.",
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
        description="Seed for deterministic generation. Leave 0 for random seed.",
    )

    _flux_model: Flux1Depth | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Depth"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux depth generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux-depth")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1Depth:
            log.info(
                "Loading MFlux depth model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Depth(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, self.model.repo_id, "flux-depth", model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux depth generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for depth-guided generation."
        )

        base_image = (
            await context.image_to_pil(self.image) if self.image.is_set() else None
        )
        depth_image = (
            await context.image_to_pil(self.depth_image)
            if self.depth_image.is_set()
            else None
        )

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()

        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)

            working_image_path: Path | None = None
            working_depth_path: Path | None = None

            try:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "guidance": self.guidance,
                    "image_path": None,
                    "depth_image_path": None,
                }

                if base_image is not None:
                    working_image = base_image.convert("RGB")
                    if working_image.size != (target_width, target_height):
                        working_image = working_image.resize(
                            (target_width, target_height), PIL.Image.Resampling.LANCZOS
                        )
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        working_image_path = Path(tmp.name)
                        working_image.save(working_image_path)
                    config_kwargs["image_path"] = working_image_path

                if depth_image is not None:
                    working_depth = depth_image.convert("L")
                    if working_depth.size != (target_width, target_height):
                        working_depth = working_depth.resize(
                            (target_width, target_height), PIL.Image.Resampling.NEAREST
                        )
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        working_depth_path = Path(tmp.name)
                        working_depth.save(working_depth_path)
                    config_kwargs["depth_image_path"] = working_depth_path

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated.image
            finally:
                if working_image_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        working_image_path.unlink()
                if working_depth_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        working_depth_path.unlink()

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFDepthGeneration]:
        return [
            HFDepthGeneration(repo_id="black-forest-labs/FLUX.1-Depth-dev"),
        ]


class MFluxRedux(BaseMFluxNode):
    """
    Generate images using reference images with Flux Redux guidance on Apple Silicon.
    mlx, flux, redux, reference fusion

    Use cases:
    - Blend multiple reference images with a text prompt to steer style and content
    - Reinterpret a photo collection into a coherent output while keeping structure from the references
    - Experiment locally with the Flux Redux pipeline without external APIs
    """

    prompt: str = Field(
        default="Create a cinematic composition inspired by the reference images",
        description="Primary text prompt for the Redux generation.",
    )
    redux_image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image that will guide the generation.",
    )
    redux_image_strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional strength value (0-1) for the reference image.",
    )
    model: HFReduxGeneration = Field(
        default=HFReduxGeneration(repo_id="black-forest-labs/FLUX.1-Redux-dev"),
        description="Redux model variant to load. Defaults to FLUX.1 Redux dev weights.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the generation run.",
    )
    guidance: float | None = Field(
        default=7.0,
        ge=0.0,
        description="Classifier-free guidance scale. A moderate default balances prompt adherence and references.",
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
        description="Seed for deterministic generation. Leave 0 for random seed.",
    )

    _flux_model: Flux1Redux | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Redux"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Redux generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux-redux")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1Redux:
            log.info(
                "Loading MFlux Redux model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )

            model_config = ModelConfig.dev_redux()
            model = Flux1Redux(
                model_config=model_config,
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, self.model.repo_id, "flux-redux", model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Redux generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Redux generation."
        )
        if not self.redux_image.is_set():
            raise ValueError("A reference image is required for Redux generation.")

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()

        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)
        try:
            temp_paths: list[Path] = []
            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)
            pil_image = await context.image_to_pil(self.redux_image)
            working_image = pil_image.convert("RGB")
            if working_image.size != (target_width, target_height):
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = Path(tmp.name)
            tmp.close()
            working_image.save(temp_path)
            temp_paths.append(temp_path)
            redux_path = str(temp_path)
            strength = (
                [float(self.redux_image_strength)]
                if self.redux_image_strength is not None
                else None
            )

            def _generate() -> PIL.Image.Image:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "guidance": self.guidance,
                    "redux_image_paths": [redux_path],
                    "redux_image_strengths": strength,
                }

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated.image

            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)
            for path in temp_paths:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFReduxGeneration]:
        return [
            HFReduxGeneration(repo_id="black-forest-labs/FLUX.1-Redux-dev"),
        ]


class MFluxKontext(BaseMFluxNode):
    """
    Generate images using Kontext reference image fusion on Apple Silicon.
    mlx, flux, kontext, reference guidance

    Use cases:
    - Leverage a reference image and prompt to produce stylistically consistent outputs
    - Perform context-aware edits without external services
    - Prototype Kontext-driven workflows locally
    """

    prompt: str = Field(
        default="Create an atmospheric scene based on the reference image",
        description="Primary text prompt for Kontext-guided generation.",
    )
    reference_image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image that will guide the Kontext generation.",
    )
    model: HFKontextGeneration = Field(
        default=HFKontextGeneration(repo_id="black-forest-labs/FLUX.1-Kontext-dev"),
        description="Kontext model weights compatible with the Flux Kontext pipeline.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the generation run.",
    )
    guidance: float | None = Field(
        default=2.5,
        ge=0.0,
        description="Classifier-free guidance scale. Kontext often works best with moderate values.",
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
        description="Seed for deterministic generation. Leave 0 for random seed.",
    )

    _flux_model: Flux1Kontext | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Kontext"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Kontext generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None

        model = ModelManager.get_model(self.model.repo_id, "flux-kontext")
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> Flux1Kontext:
            log.info(
                "Loading MFlux Kontext model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Kontext(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, self.model.repo_id, "flux-kontext", model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Kontext generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Kontext generation."
        )

        reference_image = await context.image_to_pil(self.reference_image)

        self._ensure_seed()

        assert self._flux_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> PIL.Image.Image:
            target_width = 16 * (self.width // 16)
            target_height = 16 * (self.height // 16)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = Path(tmp.name)
                working_image = reference_image.convert("RGB")
                if working_image.size != (target_width, target_height):
                    working_image = working_image.resize(
                        (target_width, target_height), PIL.Image.Resampling.LANCZOS
                    )
                working_image.save(image_path)

            try:
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": self.steps,
                    "height": target_height,
                    "width": target_width,
                    "guidance": self.guidance,
                    "image_path": image_path,
                }

                config = Config(**config_kwargs)

                assert self._flux_model is not None
                generated = self._flux_model.generate_image(
                    seed=self.seed,
                    prompt=self.prompt,
                    config=config,
                )
                return generated.image
            finally:
                with contextlib.suppress(FileNotFoundError):
                    image_path.unlink()

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFKontextGeneration]:
        return [
            HFKontextGeneration(repo_id="black-forest-labs/FLUX.1-Kontext-dev", 
        ]
