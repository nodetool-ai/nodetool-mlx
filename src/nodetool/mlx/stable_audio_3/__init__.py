"""Stable Audio 3 — Apple-Silicon-native (MLX) inference runtime.

Vendored from Stability AI's official optimized MLX implementation (MIT,
see ``LICENSE`` / ``NOTICE.md``) and wrapped for NodeTool.
"""

from nodetool.mlx.stable_audio_3 import pipeline, weights
from nodetool.mlx.stable_audio_3.pipeline import (
    DIT_CHOICES,
    SAMPLE_RATE,
    default_decoder,
    generate,
    weight_keys,
)

__all__ = [
    "pipeline",
    "weights",
    "generate",
    "default_decoder",
    "weight_keys",
    "DIT_CHOICES",
    "SAMPLE_RATE",
]
