"""MLX integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.mlx.mlx_image_provider import MlxImageProvider

register_image_provider("mlx", lambda: MlxImageProvider())

__all__ = ["MlxImageProvider"]
