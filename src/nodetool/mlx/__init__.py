"""MLX integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.mlx.mlx_image_provider import MlxLocalImageProvider

register_image_provider("mlx", lambda: MlxLocalImageProvider())

__all__ = ["MlxLocalImageProvider"]
