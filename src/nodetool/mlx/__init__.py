"""MLX integration for NodeTool.

``MLXProvider`` is imported on first access, so importing a light submodule
(``nodetool.mlx.stable_audio_3``, ``nodetool.mlx.threads``) does not pull in
mlx-lm, mlx-vlm and mlx-audio through the provider.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nodetool.mlx.mlx_provider import MLXProvider

__all__ = ["MLXProvider"]


def __getattr__(name: str) -> Any:
    if name == "MLXProvider":
        from nodetool.mlx.mlx_provider import MLXProvider

        return MLXProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
