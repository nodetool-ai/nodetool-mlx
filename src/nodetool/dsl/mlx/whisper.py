from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.mlx.whisper


class MLXWhisper(GraphNode):
    """
    Transcribe an audio asset using MLX Whisper.
    whisper, mlx, asr, speech-to-text

    - Uses MLX for efficient Apple Silicon acceleration
    - Emits final transcript on `text` and segments on `segments`
    """

    Model: typing.ClassVar[type] = nodetool.nodes.mlx.whisper.MLXWhisper.Model
    model: nodetool.nodes.mlx.whisper.MLXWhisper.Model = Field(
        default=nodetool.nodes.mlx.whisper.MLXWhisper.Model.TINY_EN,
        description="Model to use for transcription",
    )
    length_ms: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5000, description="Chunk length in milliseconds before decode"
    )
    compression_ratio_threshold: float | None | GraphNode | tuple[GraphNode, str] = (
        Field(
            default=2.4,
            description="Threshold for gzip compression ratio; above this, the result is treated as failed.",
        )
    )
    logprob_threshold: float | None | GraphNode | tuple[GraphNode, str] = Field(
        default=-1.0,
        description="Average log probability threshold; below this, the result is treated as failed.",
    )
    no_speech_threshold: float | None | GraphNode | tuple[GraphNode, str] = Field(
        default=0.6,
        description="Threshold for no-speech probability; if exceeded and logprob is low, the segment is considered silent.",
    )
    condition_on_previous_text: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="If True, the previous output is used as a prompt for the next window, improving consistency.",
    )
    word_timestamps: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="If True, extracts word-level timestamps using cross-attention and dynamic time warping.",
    )
    chunk: types.Chunk | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Chunk(
            type="chunk",
            node_id=None,
            content_type="text",
            content="",
            content_metadata={},
            done=False,
        ),
        description="Streaming input chunk. Provide audio chunks (PCM16 base64 or bytes).",
    )

    @classmethod
    def get_node_type(cls):
        return "mlx.whisper.MLXWhisper"
