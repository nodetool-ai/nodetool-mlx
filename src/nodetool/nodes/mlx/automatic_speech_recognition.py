from __future__ import annotations

import asyncio
import logging
from enum import Enum
import sys
from typing import TYPE_CHECKING, Any, Optional, TypedDict
from pydantic import Field

from nodetool.metadata.types import AudioRef, HuggingFaceModel
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import numpy as np
    from huggingface_hub import try_to_load_from_cache

log = logging.getLogger(__name__)


class Whisper(BaseNode):
    """
    Transcribe an audio asset using MLX Whisper.
    whisper, mlx, asr, speech-to-text

    - Uses MLX for efficient Apple Silicon acceleration
    - Returns transcript and segments with optional word-level timestamps
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
        LARGE_V2 = "mlx-community/whisper-large-v2-mlx"
        LARGE_V3 = "mlx-community/whisper-large-v3-mlx"
        LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
        LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"
        DISTIL_LARGE_V3 = "mlx-community/distil-whisper-large-v3"
        DISTIL_MEDIUM_EN = "mlx-community/distil-whisper-medium.en"
        DISTIL_SMALL_EN = "mlx-community/distil-whisper-small.en"

    model: Model = Field(
        default=Model.TINY_EN,
        description="Model to use for transcription",
    )

    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio Input",
        description="The input audio to transcribe.",
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

    @classmethod
    def get_basic_fields(cls):
        return ["model", "audio"]

    @classmethod
    def get_title(cls):
        return "MLX Whisper"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_output(cls) -> bool:
        return False

    @classmethod
    def is_streaming_input(cls) -> bool:
        return False

    class OutputType(TypedDict):
        text: str
        segments: list

    async def preload_model(self, context: ProcessingContext):
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        found = any(r.repo_id == self.model.value for r in cache.repos)
        if not found:
            raise ValueError(
                f"Model {self.model.value} must be downloaded first, check recommended models"
            )

    async def process(self, context: ProcessingContext) -> OutputType:
        """Transcribe audio using MLX Whisper.

        Converts the input audio to the appropriate format and runs transcription
        in a dedicated thread to avoid blocking the asyncio event loop.
        """
        if sys.platform != "darwin":
            raise RuntimeError("MLX Whisper is only supported on macOS")

        import numpy as np
        import mlx_whisper

        log.info("Starting audio processing...")

        # Convert audio to numpy array (16kHz sample rate for Whisper)
        samples, _, _ = await context.audio_to_numpy(self.audio, sample_rate=16_000)

        # Flatten to 1D array and ensure float32
        arr = samples.flatten().astype(np.float32)

        # Define transcription function to run in thread
        def _do_transcribe(audio: np.ndarray) -> dict[str, Any]:
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model.value,
                compression_ratio_threshold=self.compression_ratio_threshold,
                logprob_threshold=self.logprob_threshold,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
                word_timestamps=self.word_timestamps,
            )

        # Run transcription in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        result: dict[str, Any] = await loop.run_in_executor(None, _do_transcribe, arr)

        text = result.get("text", "") or ""
        segments = result.get("segments", []) or []

        log.info("Audio processing completed successfully.")
        return {
            "text": text,
            "segments": segments,
        }

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        """Recommend ggml Whisper models from ggerganov/whisper.cpp for local cache use.

        These correspond to files listed on the HF repo page and are suitable
        for whisper.cpp bindings.
        """
        return [HuggingFaceModel(repo_id=p.value, path=None) for p in cls.Model]
