from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    HFControlNet,
    HFControlNetFlux,
    HFFlux,
    HFFluxDepth,
    HFFluxFill,
    HFFluxKontext,
    HFFluxRedux,
    HuggingFaceModel,
    ImageRef,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.nodes.mlx.text_to_image import BaseMFluxNode, QuantizationLevel
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
    from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
    from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
    from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
    from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
    from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    from mflux.utils.image_util import ImageUtil
    from mflux.ui.box_values import BoxValues

log = get_logger(__name__)


class InContextLoRAStyle(str, Enum):
    """Available styles for In-Context LoRA generation."""
    COUPLE = "couple"
    STORYBOARD = "storyboard"
    FONT = "font"
    HOME = "home"
    ILLUSTRATION = "illustration"
    PORTRAIT = "portrait"
    PPT = "ppt"
    SANDSTORM = "sandstorm"
    SPARKLERS = "sparklers"
    IDENTITY = "identity"


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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux ImageToImage"

    async def preload_model(self, context: ProcessingContext) -> None:
        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1":
            from mflux.generate import Flux1

            log.info(
                "Loading MFlux image-to-image model %s (quantize=%s)",
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            from mflux.config.config import Config

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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux ControlNet"

    async def preload_model(self, context: ProcessingContext) -> None:
        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}:{self.controlnet_model.repo_id}_flux-controlnet"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Controlnet":
            from mflux.config.model_config import ModelConfig
            from mflux.generate_controlnet import Flux1Controlnet

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
            ModelManager.set_model(self.id, cache_key, model)
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
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
    model: HFFluxFill = Field(
        default=HFFluxFill(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Inpaint"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux inpainting requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux-fill"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Fill":
            from mflux.generate_fill import Flux1Fill

            log.info(
                "Loading MFlux Fill model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Fill(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, cache_key, model)
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            from mflux.config.config import Config

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
    def get_recommended_models(cls) -> list[HFFluxFill]:
        return [
            HFFluxFill(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
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
    model: HFFluxFill = Field(
        default=HFFluxFill(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Outpaint"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux outpainting requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux-fill"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Fill":
            from mflux.generate_fill import Flux1Fill

            log.info(
                "Loading MFlux Fill model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Fill(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            import numpy as np
            from mflux.config.config import Config
            from mflux.post_processing.image_util import ImageUtil
            from mflux.ui.box_values import BoxValues, parse_box_value

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
    def get_recommended_models(cls) -> list[HFFluxFill]:
        return [
            HFFluxFill(repo_id="black-forest-labs/FLUX.1-Fill-dev"),
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
    model: HFFluxDepth = Field(
        default=HFFluxDepth(repo_id="black-forest-labs/FLUX.1-Depth-dev"),
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Depth"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux depth generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux-depth"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Depth":
            from mflux.generate_depth import Flux1Depth

            log.info(
                "Loading MFlux depth model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Depth(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, cache_key, model)
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            from mflux.config.config import Config

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
    def get_recommended_models(cls) -> list[HFFluxDepth]:
        return [
            HFFluxDepth(repo_id="black-forest-labs/FLUX.1-Depth-dev"),
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
    model: HFFluxRedux = Field(
        default=HFFluxRedux(repo_id="black-forest-labs/FLUX.1-Redux-dev"),
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Redux"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Redux generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux-redux"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Redux":
            from mflux.config.model_config import ModelConfig
            from mflux.generate_redux import Flux1Redux

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
            ModelManager.set_model(self.id, cache_key, model)
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
            import PIL.Image
            from mflux.config.config import Config

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
    def get_recommended_models(cls) -> list[HFFluxRedux]:
        return [
            HFFluxRedux(repo_id="black-forest-labs/FLUX.1-Redux-dev"),
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
    model: HFFluxKontext = Field(
        default=HFFluxKontext(repo_id="black-forest-labs/FLUX.1-Kontext-dev"),
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

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Kontext"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Kontext generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_flux-kontext"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1Kontext":
            from mflux.generate_kontext import Flux1Kontext

            log.info(
                "Loading MFlux Kontext model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = Flux1Kontext(
                quantize=quantize_value,
            )
            ModelManager.set_model(self.id, cache_key, model)
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

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            from mflux.config.config import Config

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
    def get_recommended_models(cls) -> list[HFFluxKontext]:
        return [
            HFFluxKontext(repo_id="black-forest-labs/FLUX.1-Kontext-dev"),
        ]


class MFluxQwenImage(BaseMFluxNode):
    """
    Generate images using the Qwen Image model via MFLUX on Apple Silicon.
    mlx, qwen, text-to-image, apple-silicon

    Use cases:
    - Generate high-quality images with multilingual prompt support including Chinese
    - Create images with rendered Chinese text content (signs, calligraphy, menus)
    - Leverage Qwen's 20B parameter vision-language model for detailed generation
    """

    prompt: str = Field(
        default="A beautiful landscape with mountains",
        description="Text prompt describing the image to generate. Supports multilingual prompts including Chinese.",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt describing what to avoid in the generation.",
    )
    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="filipstrand/Qwen-Image-mflux-6bit"),
        description="Qwen Image model to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_6,
        description="Quantization level for model weights. 6-bit recommended for Qwen.",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for the generation.",
    )
    guidance: float = Field(
        default=3.5,
        ge=0.0,
        description="Guidance scale for the generation.",
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

    _qwen_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Qwen Image"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Qwen Image generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_qwen-image"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._qwen_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "QwenImage":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

            log.info(
                "Loading MFlux Qwen Image model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = QwenImage(
                quantize=quantize_value,
                model_path=self.model.repo_id if "/" in self.model.repo_id else None,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._qwen_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Qwen Image generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Qwen Image generation."
        )
        self._ensure_seed()

        assert self._qwen_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> "PIL.Image.Image":
            import PIL.Image

            assert self._qwen_model is not None
            generated_image = self._qwen_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
                guidance=self.guidance,
                negative_prompt=self.negative_prompt if self.negative_prompt else None,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id="filipstrand/Qwen-Image-mflux-6bit"),
            HuggingFaceModel(repo_id="Qwen/Qwen-Image"),
        ]


class MFluxQwenImageEdit(BaseMFluxNode):
    """
    Edit images using natural language instructions with the Qwen Image Edit model.
    mlx, qwen, image-editing, apple-silicon

    Use cases:
    - Modify images using natural language descriptions
    - Combine elements from multiple reference images
    - Apply transformations while maintaining original poses and positions
    """

    prompt: str = Field(
        default="Make the sky more dramatic",
        description="Text instruction describing the desired edit.",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt describing what to avoid.",
    )
    images: list[ImageRef] = Field(
        default_factory=list,
        description="Reference images for the edit. The last image dimensions determine output size.",
    )
    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="Qwen/Qwen-Image-Edit-2509"),
        description="Qwen Image Edit model to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_8,
        description="Quantization level. 8-bit recommended for editing quality.",
    )
    steps: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Number of denoising steps.",
    )
    guidance: float = Field(
        default=2.5,
        ge=0.0,
        description="Guidance scale. 2.5 works well for Qwen Edit.",
    )
    height: int | None = Field(
        default=None,
        ge=256,
        le=2048,
        description="Optional output height. If not set, computed from input images.",
    )
    width: int | None = Field(
        default=None,
        ge=256,
        le=2048,
        description="Optional output width. If not set, computed from input images.",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    _qwen_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Qwen Image Edit"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Qwen Image Edit requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_qwen-image-edit"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._qwen_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "QwenImageEdit":
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

            log.info(
                "Loading MFlux Qwen Image Edit model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = QwenImageEdit(
                quantize=quantize_value,
                model_path=self.model.repo_id if "/" in self.model.repo_id else None,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._qwen_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Qwen Image Edit requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Qwen Image Edit."
        )

        if not self.images:
            raise ValueError("At least one reference image is required for Qwen Image Edit.")

        self._ensure_seed()

        assert self._qwen_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        # Save reference images to temp files
        temp_paths: list[Path] = []
        for img_ref in self.images:
            pil_img = await context.image_to_pil(img_ref)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = Path(tmp.name)
            tmp.close()
            pil_img.convert("RGB").save(temp_path)
            temp_paths.append(temp_path)

        def _generate() -> "PIL.Image.Image":
            import PIL.Image

            assert self._qwen_model is not None
            image_paths = [str(p) for p in temp_paths]
            generated_image = self._qwen_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                image_paths=image_paths,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
                guidance=self.guidance,
                negative_prompt=self.negative_prompt if self.negative_prompt else None,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)
            for path in temp_paths:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id="Qwen/Qwen-Image-Edit-2509"),
        ]


class MFluxZImageTurbo(BaseMFluxNode):
    """
    Generate images quickly using the Z-Image Turbo model via MFLUX.
    mlx, z-image, text-to-image, fast-generation, apple-silicon

    Use cases:
    - Fast image generation (typically 9 steps)
    - Efficient 6B parameter model for quick iterations
    - Support for LoRA adapters for style customization
    """

    prompt: str = Field(
        default="A beautiful sunset over mountains",
        description="Text prompt describing the image to generate.",
    )
    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="filipstrand/Z-Image-Turbo-mflux-4bit"),
        description="Z-Image Turbo model to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Quantization level for model weights.",
    )
    steps: int = Field(
        default=9,
        ge=1,
        le=20,
        description="Number of denoising steps. Z-Image Turbo works best with 9 steps.",
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
        description="Optional path or HuggingFace repo ID for a LoRA adapter.",
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scale factor for the LoRA adapter.",
    )

    _zimage_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux Z-Image Turbo"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux Z-Image Turbo requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        lora_key = self.lora_path or "none"
        cache_key = f"{self.model.repo_id}_{lora_key}_z-image-turbo"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._zimage_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "ZImageTurbo":
            from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

            log.info(
                "Loading MFlux Z-Image Turbo model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            lora_paths = [self.lora_path] if self.lora_path else None
            lora_scales = [self.lora_scale] if self.lora_path else None

            model = ZImageTurbo(
                quantize=quantize_value,
                model_path=self.model.repo_id if "/" in self.model.repo_id else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._zimage_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux Z-Image Turbo requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for Z-Image Turbo generation."
        )
        self._ensure_seed()

        assert self._zimage_model is not None

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> "PIL.Image.Image":
            import PIL.Image

            assert self._zimage_model is not None
            generated_image = self._zimage_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id="filipstrand/Z-Image-Turbo-mflux-4bit"),
            HuggingFaceModel(repo_id="Tongyi-MAI/Z-Image-Turbo"),
        ]


class MFluxSeedVR2Upscale(BaseMFluxNode):
    """
    Upscale images using the SeedVR2 super-resolution model via MFLUX.
    mlx, seedvr2, upscaling, super-resolution, apple-silicon

    Use cases:
    - Fast and faithful image upscaling (typically 1 step)
    - High-fidelity super-resolution without text prompts
    - Upscale by target resolution or scale factor
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="Image to upscale.",
    )
    resolution: int = Field(
        default=1800,
        ge=256,
        le=4096,
        description="Target resolution for the shortest edge in pixels.",
    )
    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="numz/SeedVR2-3B"),
        description="SeedVR2 model to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_8,
        description="Quantization level (4 or 8 bit supported).",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    _seedvr2_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux SeedVR2 Upscale"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux SeedVR2 upscaling requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        cache_key = f"{self.model.repo_id}_seedvr2"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._seedvr2_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "SeedVR2":
            from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

            log.info(
                "Loading MFlux SeedVR2 model %s (quantize=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
            )
            model = SeedVR2(
                quantize=quantize_value,
                model_path=self.model.repo_id if "/" in self.model.repo_id else None,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._seedvr2_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux SeedVR2 upscaling requires macOS (Apple Silicon / MLX)."
        )

        if not self.image.is_set():
            raise ValueError("An input image is required for SeedVR2 upscaling.")

        self._ensure_seed()

        assert self._seedvr2_model is not None

        # Save input image to temp file
        pil_img = await context.image_to_pil(self.image)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(tmp.name)
        tmp.close()
        pil_img.convert("RGB").save(temp_path)

        loop = asyncio.get_running_loop()

        def _generate() -> "PIL.Image.Image":
            import PIL.Image

            assert self._seedvr2_model is not None
            generated_image = self._seedvr2_model.generate_image(
                seed=self.seed,
                image_path=str(temp_path),
                resolution=self.resolution,
            )
            return generated_image.image

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id="numz/SeedVR2-3B"),
        ]


class MFluxInContext(BaseMFluxNode):
    """
    Generate style-consistent images using In-Context LoRA via MFLUX.
    mlx, flux, in-context, style-transfer, apple-silicon

    Use cases:
    - Generate images following a specific style from a reference image
    - Apply various pre-defined style LoRAs (illustration, portrait, storyboard, etc.)
    - Create consistent visual identity across multiple generations
    """

    prompt: str = Field(
        default="A portrait in the same style",
        description="Text prompt describing the image to generate. Should describe both reference and target.",
    )
    reference_image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image that provides the style context.",
    )
    style: InContextLoRAStyle | None = Field(
        default=None,
        description="Optional pre-defined style LoRA to apply.",
    )
    model: HFFlux = Field(
        default=HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
        description="Base Flux dev model to load.",
    )
    quantize: QuantizationLevel | None = Field(
        default=QuantizationLevel.BITS_4,
        description="Quantization level for model weights.",
    )
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps.",
    )
    guidance: float = Field(
        default=4.0,
        ge=0.0,
        description="Guidance scale for the generation.",
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

    # Mapping of style names to LoRA paths
    STYLE_LORA_MAP: ClassVar[dict[str, str]] = {
        "couple": "ali-vilab/In-Context-LoRA:couple-profile.safetensors",
        "storyboard": "ali-vilab/In-Context-LoRA:film-storyboard.safetensors",
        "font": "ali-vilab/In-Context-LoRA:font-design.safetensors",
        "home": "ali-vilab/In-Context-LoRA:home-decoration.safetensors",
        "illustration": "ali-vilab/In-Context-LoRA:portrait-illustration.safetensors",
        "portrait": "ali-vilab/In-Context-LoRA:portrait-photography.safetensors",
        "ppt": "ali-vilab/In-Context-LoRA:ppt-templates.safetensors",
        "sandstorm": "ali-vilab/In-Context-LoRA:sandstorm.safetensors",
        "sparklers": "ali-vilab/In-Context-LoRA:sparklers.safetensors",
        "identity": "ali-vilab/In-Context-LoRA:visual-identity-design.safetensors",
    }

    @classmethod
    def get_title(cls):
        return "MFlux In-Context"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux In-Context generation requires macOS (Apple Silicon / MLX)."
        )

        quantize_value = int(self.quantize) if self.quantize is not None else None
        style_key = self.style.value if self.style else "none"
        cache_key = f"{self.model.repo_id}_{style_key}_flux-in-context"

        model = ModelManager.get_model(cache_key)
        if model is not None:
            self._flux_model = model
            return

        loop = asyncio.get_running_loop()

        def _load_model() -> "Flux1InContextDev":
            from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev

            log.info(
                "Loading MFlux In-Context model %s (quantize=%s, style=%s)",
                self.model.repo_id,
                quantize_value if quantize_value is not None else "none",
                style_key,
            )

            lora_paths = None
            lora_scales = None
            if self.style:
                lora_path = self.STYLE_LORA_MAP.get(self.style.value)
                if lora_path:
                    lora_paths = [lora_path]
                    lora_scales = [1.0]

            model = Flux1InContextDev(
                quantize=quantize_value,
                model_path=self.model.repo_id if "/" in self.model.repo_id else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            ModelManager.set_model(self.id, cache_key, model)
            return model

        self._flux_model = await loop.run_in_executor(None, _load_model)

    async def process(self, context: ProcessingContext) -> ImageRef:
        self._ensure_supported_platform(
            "MFlux In-Context generation requires macOS (Apple Silicon / MLX)."
        )
        self._require_prompt(
            self.prompt, "Prompt cannot be empty for In-Context generation."
        )

        if not self.reference_image.is_set():
            raise ValueError("A reference image is required for In-Context generation.")

        self._ensure_seed()

        assert self._flux_model is not None

        # Save reference image to temp file
        import PIL.Image
        pil_img = await context.image_to_pil(self.reference_image)
        working_image = pil_img.convert("RGB")

        # Resize to match output dimensions if needed
        target_width = 16 * (self.width // 16)
        target_height = 16 * (self.height // 16)
        if working_image.size != (target_width, target_height):
            working_image = working_image.resize(
                (target_width, target_height), PIL.Image.Resampling.LANCZOS
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(tmp.name)
        tmp.close()
        working_image.save(temp_path)

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> "PIL.Image.Image":
            assert self._flux_model is not None
            generated_image = self._flux_model.generate_image(
                seed=self.seed,
                prompt=self.prompt,
                num_inference_steps=self.steps,
                height=target_height,
                width=target_width,
                guidance=self.guidance,
                image_path=str(temp_path),
            )
            # The In-Context model generates side-by-side images
            # We need to crop to get only the right half (generated image)
            full_image = generated_image.image
            output_width = full_image.width // 2
            cropped = full_image.crop((output_width, 0, full_image.width, full_image.height))
            return cropped

        try:
            pil_image = await loop.run_in_executor(None, _generate)
        finally:
            self._remove_progress_callback(progress_callback)
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()

        return await context.image_from_pil(pil_image)

    @classmethod
    def get_recommended_models(cls) -> list[HFFlux]:
        return [
            HFFlux(repo_id="dhairyashil/FLUX.1-dev-mflux-4bit"),
            HFFlux(repo_id="black-forest-labs/FLUX.1-dev"),
        ]
