from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HFFlux, ImageRef
from nodetool.ml.core.model_manager import ModelManager
from nodetool.mlx.memory_estimator import (
    estimate_mflux_memory_bytes,
    format_memory_error,
)
from nodetool.mlx.system_memory import check_memory_headroom
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    # Type-only imports - actual imports happen at runtime in method bodies
    # This allows compatibility with multiple mflux versions
    import numpy as np
    import PIL.Image
    from mflux.callbacks.callback import InLoopCallback
    from mflux.models.common.config.config import Config  # mflux >= 0.11
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.generate import Flux1
    from mflux.post_processing.image_util import ImageUtil
    from mflux.ui.box_values import BoxValues

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
    ) -> "InLoopCallback":
        from mflux.callbacks.callback import InLoopCallback
        from mflux.callbacks.callback_registry import CallbackRegistry

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
    def _remove_progress_callback(callback: "InLoopCallback") -> None:
        from mflux.callbacks.callback_registry import CallbackRegistry

        with contextlib.suppress(ValueError):
            CallbackRegistry.in_loop_callbacks().remove(callback)

    def _check_memory_safety(
        self,
        width: int,
        height: int,
        steps: int,
        quantize_bits: int | None,
        low_memory: bool,
        model_type: str = "flux",
    ) -> None:
        """
        Perform preflight memory check to prevent system freezes.

        This checks if there's sufficient system memory headroom before
        running the MFlux generation. Fails fast if memory is insufficient.

        Args:
            width: Output image width
            height: Output image height
            steps: Number of inference steps
            quantize_bits: Quantization level or None
            low_memory: Whether VAE tiling is enabled
            model_type: Model family for auxiliary memory estimation

        Raises:
            RuntimeError: If insufficient memory headroom is available
        """
        # Only perform check on macOS
        if sys.platform != "darwin":
            return

        try:
            # Estimate memory needed for this job
            estimated_bytes = estimate_mflux_memory_bytes(
                width=width,
                height=height,
                steps=steps,
                quantize_bits=quantize_bits,
                low_memory=low_memory,
                model_type=model_type,
            )

            # Check if we have sufficient headroom (10% of total memory)
            has_sufficient, snapshot, required_headroom = check_memory_headroom(
                required_bytes=estimated_bytes,
                min_headroom_fraction=0.10,
            )

            log.info(
                f"Memory preflight check: estimated={estimated_bytes / (1024**3):.1f}GB, "
                f"available={snapshot.available_gb:.1f}GB, "
                f"headroom={required_headroom / (1024**3):.1f}GB, "
                f"sufficient={has_sufficient}"
            )

            if not has_sufficient:
                error_msg = format_memory_error(
                    estimated_bytes=estimated_bytes,
                    available_bytes=snapshot.available_bytes,
                    required_headroom_bytes=required_headroom,
                    width=width,
                    height=height,
                    steps=steps,
                    low_memory=low_memory,
                )
                log.error(f"Memory preflight check failed:\n{error_msg}")
                raise RuntimeError(error_msg)

        except RuntimeError:
            # Re-raise memory check failures
            raise
        except Exception as e:
            # Log but don't fail on memory check errors (e.g., on non-macOS or permission issues)
            log.warning(
                f"Memory preflight check could not be performed (non-fatal): {e}"
            )

    def _configure_vae_tiling(
        self,
        flux_model: Any,
        low_memory: bool,
        vae_tiling_split: str = "horizontal",
    ) -> None:
        """
        Configure VAE tiling on the model if low-memory mode is enabled.

        This sets the VAE decoder to use tiling, which reduces peak memory
        usage at the cost of potential seams in the output image.

        Args:
            flux_model: The loaded MFlux model
            low_memory: Whether to enable VAE tiling
            vae_tiling_split: Direction to split ("horizontal" or "vertical")
        """
        try:
            if low_memory and hasattr(flux_model, "vae"):
                if hasattr(flux_model.vae, "decoder"):
                    flux_model.vae.decoder.enable_tiling = True
                    flux_model.vae.decoder.split_direction = vae_tiling_split
                    log.info(
                        f"VAE tiling enabled (split={vae_tiling_split}). "
                        "This reduces memory usage but may cause visible seams."
                    )
                else:
                    log.warning(
                        "Model does not have vae.decoder attribute; VAE tiling not configured"
                    )
            elif low_memory:
                log.warning(
                    "Model does not have vae attribute; VAE tiling not configured"
                )
        except Exception as e:
            log.warning(f"Failed to configure VAE tiling: {e}")

    def _prepare_config_kwargs(
        self,
        config_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prepare config kwargs by adding model_config if available.

        This helper reduces code duplication across different MFlux node types.

        Args:
            config_kwargs: Base configuration kwargs

        Returns:
            Updated config kwargs with model_config if available
        """
        # Get model_config from the flux model if available
        if self._flux_model is not None and hasattr(self._flux_model, "model_config"):
            config_kwargs["model_config"] = self._flux_model.model_config
        return config_kwargs


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
    low_memory: bool = Field(
        default=False,
        description="Enable low-memory mode using VAE tiling. Slower, but safer on low-RAM Macs.",
    )
    vae_tiling_split: Literal["horizontal", "vertical"] = Field(
        default="horizontal",
        description="VAE tiling direction used during decode (if low-memory is enabled).",
    )

    _flux_model: Any | None = None

    @classmethod
    def get_title(cls):
        return "MFlux"

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform(
            "MFlux generation requires macOS (Apple Silicon / MLX)."
        )

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

        # Perform memory safety check before generation
        quantize_value = int(self.quantize) if self.quantize is not None else None
        self._check_memory_safety(
            width=self.width,
            height=self.height,
            steps=self.steps,
            quantize_bits=quantize_value,
            low_memory=self.low_memory,
            model_type="flux",
        )

        assert self._flux_model is not None

        # Configure VAE tiling if low-memory mode is enabled
        self._configure_vae_tiling(
            flux_model=self._flux_model,
            low_memory=self.low_memory,
            vae_tiling_split=self.vae_tiling_split,
        )

        loop = asyncio.get_running_loop()
        total_steps = self.steps
        progress_callback = self._register_progress_callback(context, total_steps)

        def _generate() -> "PIL.Image.Image":
            import PIL.Image
            from mflux.models.common.config.config import Config
            from mflux.generate import Flux1

            config_kwargs: dict[str, Any] = {
                "num_inference_steps": self.steps,
                "height": self.height,
                "width": self.width,
            }
            if self.guidance is not None:
                config_kwargs["guidance"] = self.guidance

            # Add model_config if available
            config_kwargs = self._prepare_config_kwargs(config_kwargs)

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
            HFFlux(repo_id="filipstrand/Qwen-Image-mflux-6bit"),
        ]
