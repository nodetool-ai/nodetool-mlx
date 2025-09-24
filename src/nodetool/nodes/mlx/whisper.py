from __future__ import annotations

import asyncio
import logging
from enum import Enum
import sys
from typing import Any, Optional, cast

from huggingface_hub import try_to_load_from_cache
from nodetool.metadata.types import HuggingFaceModel
import numpy as np
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.io import NodeInputs, NodeOutputs

# Optional: used for streaming string chunks
from nodetool.chat.providers import Chunk

log = logging.getLogger(__name__)

# Text utils
from nodetool.nodes.lib.text_utils import compute_incremental_suffix


class MLXWhisper(BaseNode):
    """
    Transcribe an audio asset using MLX Whisper.
    whisper, mlx, asr, speech-to-text

    - Uses MLX for efficient Apple Silicon acceleration
    - Emits final transcript on `text` and segments on `segments`
    """

    class Model(str, Enum):
        TINY = "mlx-community/whisper-tiny-mlx"
        TINY_EN = "mlx-community/whisper-tiny.en-mlx"
        BASE = "mlx-community/whisper-base-mlx"
        BASE_EN = "mlx-community/whisper-base.en-mlx"
        SMALL = "mlx-community/whisper-small-mlx"
        SMALL_EN = "mlx-community/whisper-small.en-mlx"
        MEDIUM = "mlx-community/whisper-medium-mlx"
        MEDIUM_EN = "mlx-community/whisper-medium.en-mlx"
        LARGE_V3 = "mlx-community/whisper-large-v3-mlx"

    model: Model = Field(
        default=Model.TINY_EN,
        description="Model to use for transcription",
    )

    # Pseudo-streaming control: minimum window before triggering a decode
    length_ms: int = Field(
        default=5000, description="Chunk length in milliseconds before decode"
    )

    compression_ratio_threshold: Optional[float] = Field(
        default=2.4,
        description="Threshold for gzip compression ratio; above this, the result is treated as failed.",
    )
    logprob_threshold: Optional[float] = Field(
        default=-1.0,
        description="Average log probability threshold; below this, the result is treated as failed.",
    )
    no_speech_threshold: Optional[float] = Field(
        default=0.6,
        description="Threshold for no-speech probability; if exceeded and logprob is low, the segment is considered silent.",
    )
    condition_on_previous_text: bool = Field(
        default=True,
        description="If True, the previous output is used as a prompt for the next window, improving consistency.",
    )
    word_timestamps: bool = Field(
        default=False,
        description="If True, extracts word-level timestamps using cross-attention and dynamic time warping.",
    )

    chunk: Chunk = Field(
        default=Chunk(),
        description="Streaming input chunk. Provide audio chunks (PCM16 base64 or bytes).",
    )

    @classmethod
    def get_title(cls):
        return "MLX Whisper"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
            "segments": list,
        }

    async def preload_model(self, context: ProcessingContext):
        local_path = try_to_load_from_cache(self.model.value, "config.json")
        if not local_path:
            raise ValueError(
                f"Model {self.model.value} must be downloaded first, check recommended models"
            )

    # Note: This node is streaming-only; use `run` with `chunk` input.

    async def run(
        self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs
    ) -> None:
        """Stream transcription by accumulating audio context and emitting deltas.

        Accepts either streaming `chunk` inputs with audio payloads or a single
        `audio` input. Maintains an audio buffer so decoding incorporates prior
        context. Emits text deltas on `chunk` and final `text` on completion.
        """
        # Queues for streaming audio samples
        input_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        done_flag = {"done": False}

        if sys.platform != "darwin":
            raise RuntimeError("MLX Whisper is only supported on macOS")

        import mlx_whisper

        async def producer() -> None:
            async for handle, item in inputs.any():
                if handle == "chunk" and isinstance(item, Chunk):
                    if item.content_type == "audio" and item.content:
                        raw: bytes | None = None
                        if isinstance(item.content, str):
                            # base64-encoded PCM16
                            try:
                                import base64  # local import to avoid overhead if unused

                                raw = base64.b64decode(item.content)
                            except Exception:
                                raw = None
                        elif isinstance(item.content, (bytes, bytearray)):
                            raw = bytes(item.content)
                        if raw:
                            pcm16 = np.frombuffer(raw, dtype=np.int16)
                            arr = (pcm16.astype(np.float32) / 32768.0).flatten()
                            if arr.size:
                                await input_q.put(arr)
                    if getattr(item, "done", False):
                        # Boundary hint – trigger a decode
                        await input_q.put(np.array([], dtype=np.float32))
                else:
                    log.debug(f"Ignoring unsupported handle: {handle}")
            done_flag["done"] = True

        async def consumer() -> None:
            full_text = ""

            loop = asyncio.get_running_loop()

            while not (done_flag["done"] and input_q.empty()):
                try:
                    chunk = await asyncio.wait_for(input_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    chunk = None  # type: ignore

                # Trigger decode on each non-empty chunk or explicit boundary
                if chunk is None:
                    await asyncio.sleep(0)
                    continue

                # Empty array indicates boundary; skip decoding if no new audio
                if chunk.size == 0:
                    await asyncio.sleep(0)
                    continue

                arr = chunk.astype(np.float32, copy=False)

                def _do_transcribe(a: np.ndarray) -> dict[str, Any]:
                    return mlx_whisper.transcribe(
                        a,
                        path_or_hf_repo=self.model.value,
                        compression_ratio_threshold=self.compression_ratio_threshold,
                        logprob_threshold=self.logprob_threshold,
                        no_speech_threshold=self.no_speech_threshold,
                        condition_on_previous_text=self.condition_on_previous_text,
                        word_timestamps=self.word_timestamps,
                    )

                try:
                    result: dict[str, Any] = await loop.run_in_executor(
                        None, _do_transcribe, arr
                    )
                except Exception as e:
                    raise RuntimeError(f"MLX Whisper transcription failed: {e}") from e

                new_text = result.get("text", "") or ""
                delta_text = compute_incremental_suffix(full_text, new_text)
                if delta_text:
                    await outputs.emit("chunk", Chunk(content=delta_text, done=False))
                    full_text = full_text + delta_text
                await outputs.emit("segments", result.get("segments", []) or [])

                await asyncio.sleep(0)

            await outputs.emit("text", full_text)
            outputs.complete("chunk")
            outputs.complete("text")
            outputs.complete("segments")

        await asyncio.gather(producer(), consumer())

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        """Recommend ggml Whisper models from ggerganov/whisper.cpp for local cache use.

        These correspond to files listed on the HF repo page and are suitable
        for whisper.cpp bindings.
        """
        return [HuggingFaceModel(repo_id=p.value, path=None) for p in cls.Model]
