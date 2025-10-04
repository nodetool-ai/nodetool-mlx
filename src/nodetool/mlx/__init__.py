"""MLX integration for NodeTool."""

# MLXProvider is automatically registered via @register_provider decorator
from nodetool.mlx.mlx_provider import MLXProvider

__all__ = ["MLXProvider"]
