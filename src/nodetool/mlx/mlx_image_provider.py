"""
MLX local image provider implementation.

This module implements the BaseProvider interface for local MLX models using mflux.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, List, Set
from io import BytesIO
import PIL.Image

from nodetool.chat.providers.base import BaseProvider, ProviderCapability
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageModel, Provider
from nodetool.config.logging_config import get_logger
from nodetool.mlx.flux_model_loader import (
    load_flux_model,
    ensure_mflux_available,
    FluxModelNotAvailableError,
)

try:
    from mflux.config.config import Config

    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False

log = get_logger(__name__)


class MlxLocalImageProvider(BaseProvider):
    """Local image provider for MLX models using mflux.

    This provider runs FLUX models locally on Apple Silicon using MLX acceleration.
    Requires macOS and the mflux library. Models must be downloaded via the Model Manager
    before use.
    """

    provider_name = "mlx"

    def __init__(self):
        super().__init__()

    def get_capabilities(self) -> Set[ProviderCapability]:
        """MLX provider supports text-to-image generation only."""
        return {
            ProviderCapability.TEXT_TO_IMAGE,
        }

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {}

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using MLX.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
            FluxModelNotAvailableError: If model is not cached locally
        """
        ensure_mflux_available()

        if not params.prompt:
            raise ValueError("Prompt cannot be empty for image generation.")

        self._log_api_request("text_to_image", params)

        try:
            # Load model using centralized loader
            model = await load_flux_model(
                model_id=params.model.id,
                quantize=4,  # Default to 4-bit quantization
                task="flux",
            )

            loop = asyncio.get_running_loop()

            def _generate() -> PIL.Image.Image:
                # Prepare config
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": params.num_inference_steps or 4,
                    "height": params.height or 1024,
                    "width": params.width or 1024,
                }

                if params.guidance_scale is not None:
                    config_kwargs["guidance"] = params.guidance_scale

                # Filter to only allowed config fields
                dataclass_fields = getattr(Config, "__dataclass_fields__", None)
                if isinstance(dataclass_fields, dict):
                    allowed = set(dataclass_fields.keys())
                    config_kwargs = {
                        key: value
                        for key, value in config_kwargs.items()
                        if key in allowed
                    }

                config = Config(**config_kwargs)

                # Generate image
                generated = model.generate_image(
                    seed=params.seed if params.seed is not None else 0,
                    prompt=params.prompt,
                    config=config,
                )
                return generated.image

            # Run generation in executor
            pil_image = await loop.run_in_executor(None, _generate)

            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("text_to_image", 1)

            return image_bytes

        except FluxModelNotAvailableError:
            # Re-raise with the helpful error message
            raise
        except Exception as e:
            log.error(f"MLX text-to-image generation failed: {e}")
            raise RuntimeError(f"MLX text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using MLX.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
            FluxModelNotAvailableError: If model is not cached locally
        """
        ensure_mflux_available()

        if not params.prompt:
            raise ValueError("Prompt cannot be empty for image-to-image generation.")

        self._log_api_request("image_to_image", params)

        try:
            # Load model using centralized loader
            model = await load_flux_model(
                model_id=params.model.id,
                quantize=4,  # Default to 4-bit quantization
                task="flux",
            )

            # Load input image
            base_image = PIL.Image.open(BytesIO(image))

            loop = asyncio.get_running_loop()

            def _generate() -> PIL.Image.Image:
                # Prepare image
                working_image = base_image.convert("RGB")
                target_width = 16 * ((params.target_width or 1024) // 16)
                target_height = 16 * ((params.target_height or 1024) // 16)

                if working_image.size != (target_width, target_height):
                    working_image = working_image.resize(
                        (target_width, target_height), PIL.Image.Resampling.LANCZOS
                    )

                # Save to temp file (mflux requires file paths)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image_path = Path(tmp.name)
                    working_image.save(image_path)

                try:
                    # Prepare config
                    config_kwargs: dict[str, Any] = {
                        "num_inference_steps": params.num_inference_steps or 8,
                        "height": target_height,
                        "width": target_width,
                        "image_strength": params.strength or 0.4,
                        "image_path": image_path,
                    }

                    if params.guidance_scale is not None:
                        config_kwargs["guidance"] = params.guidance_scale

                    # Filter to only allowed config fields
                    dataclass_fields = getattr(Config, "__dataclass_fields__", None)
                    if isinstance(dataclass_fields, dict):
                        allowed = set(dataclass_fields.keys())
                        config_kwargs = {
                            key: value
                            for key, value in config_kwargs.items()
                            if key in allowed
                        }

                    config = Config(**config_kwargs)

                    # Generate image
                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        prompt=params.prompt,
                        config=config,
                    )
                    return generated.image
                finally:
                    # Clean up temp file
                    try:
                        image_path.unlink()
                    except FileNotFoundError:
                        pass

            # Run generation in executor
            pil_image = await loop.run_in_executor(None, _generate)

            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("image_to_image", 1)

            return image_bytes

        except FluxModelNotAvailableError:
            # Re-raise with the helpful error message
            raise
        except Exception as e:
            log.error(f"MLX image-to-image generation failed: {e}")
            raise RuntimeError(f"MLX image-to-image generation failed: {e}")

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available MLX image models.

        Returns recommended FLUX models from MFlux nodes that can be used
        for local image generation on Apple Silicon. Always returns models
        (does not check if MLX is available or models are cached).

        Returns:
            List of ImageModel instances for MLX
        """
        # Recommended FLUX models from MFlux nodes (mflux.py)
        models = [
            # From MFlux.get_recommended_models()
            ImageModel(
                id="Freepik/flux.1-lite-8B-alpha",
                name="FLUX.1 Lite 8B Alpha",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
                name="FLUX.1 Schnell 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="dhairyashil/FLUX.1-dev-mflux-4bit",
                name="FLUX.1 Dev 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="filipstrand/FLUX.1-Krea-dev-mflux-4bit",
                name="FLUX.1 Krea Dev 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="akx/FLUX.1-Kontext-dev-mflux-4bit",
                name="FLUX.1 Kontext Dev 4-bit",
                provider=Provider.MLX,
            ),
        ]

        return models
