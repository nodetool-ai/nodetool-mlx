from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.mlx.tts


class TTS(GraphNode):
    """
    Generate speech locally using mlx-audio TTS models on Apple Silicon.
    tts, audio, speech, mlx, kokoro, sesame

    Use cases:
    - Fast offline TTS generation on Apple Silicon devices
    - Experiment with Kokoro/Sesame voices without GPU dependencies
    - Produce narration or assistant voices with low-latency local inference
    """

    Model: typing.ClassVar[type] = nodetool.nodes.mlx.tts.TTS.Model
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Hello from MLX TTS.",
        description="Text content to synthesize into speech.",
    )
    model: nodetool.nodes.mlx.tts.TTS.Model = Field(
        default=nodetool.nodes.mlx.tts.TTS.Model.KOKORO_82M,
        description="MLX TTS model repo ID or local path.",
    )
    voice: str | GraphNode | tuple[GraphNode, str] = Field(
        default="af_heart",
        description="Voice preset supported by the selected model. Kokoro exposes af_*/am_*/bf_* voices; Sesame expects reference-audio-assisted voices.",
    )
    speed: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="Playback speed multiplier supported by Kokoro models."
    )
    lang_code: str | GraphNode | tuple[GraphNode, str] = Field(
        default="a",
        description="Language code when supported (Kokoro: a=American English, b=British English, j=Japanese, etc.).",
    )
    temperature: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.7,
        description="Sampling temperature passed to the underlying generator when supported.",
    )

    @classmethod
    def get_node_type(cls):
        return "mlx.tts.TTS"
