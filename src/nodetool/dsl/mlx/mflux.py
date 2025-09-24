from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ImageGeneration(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="A vivid concept art piece of a futuristic city at sunset",
        description="The text prompt describing the image to generate.",
    )
    model: types.HFFlux | GraphNode | tuple[GraphNode, str] = Field(
        default=types.HFFlux(
            type="hf.flux",
            repo_id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
            path=None,
            variant=None,
            allow_patterns=None,
            ignore_patterns=None,
        ),
        description="MFLUX model variant to load. Options include official models (schnell, dev, krea-dev), third-party community models (Freepik/flux.1-lite-8B-alpha), and quantized 4-bit versions for reduced memory usage.",
    )
    quantize: (
        nodetool.nodes.mlx.mflux.QuantizationLevel
        | None
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(
        default=nodetool.nodes.mlx.mflux.QuantizationLevel.BITS_4,
        description="Optional quantization level for model weights (reduces memory usage).",
    )
    steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4, description="Number of denoising steps for the generation run."
    )
    guidance: float | None | GraphNode | tuple[GraphNode, str] = Field(
        default=3.5,
        description="Classifier-free guidance scale. Used by dev/krea-dev models.",
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Height of the generated image in pixels."
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1024, description="Width of the generated image in pixels."
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    @classmethod
    def get_node_type(cls):
        return "mlx.mflux.ImageGeneration"
