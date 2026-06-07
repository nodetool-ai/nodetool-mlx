from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import tempfile
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast

from pydantic import Field, PrivateAttr

from nodetool.metadata.types import AudioRef, HuggingFaceModel, Provider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


class BaseMLXSpeechToText(BaseNode):
    """Shared functionality for MLX speech-to-text (ASR) nodes backed by ``mlx-audio``.

    Subclasses pick a concrete model family (Parakeet, Qwen3-ASR, ...) and may add
    model specific generation parameters via :meth:`_build_generate_kwargs`.
    """

    _body: ClassVar[str] = "content_card"

    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio Input",
        description="The input audio to transcribe.",
    )

    _provider: ClassVar[Provider] = Provider.MLX
    _stt_model: Any | None = PrivateAttr(default=None)
    _model_id_loaded: str | None = PrivateAttr(default=None)

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseMLXSpeechToText

    def required_inputs(self):
        return ["audio"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio"]

    @classmethod
    def get_input_fields(cls):
        return ["audio"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "MLX speech-to-text requires macOS (Apple Silicon / MLX)."
            )

    def _get_model_id(self) -> str:
        model = getattr(self, "model", None)
        if model is None:
            raise ValueError("Model must be selected before loading MLX STT.")
        if isinstance(model, Enum):
            return cast(str, model.value)
        return str(model)

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        model_id = self._get_model_id()

        if self._stt_model is not None and self._model_id_loaded == model_id:
            return

        from huggingface_hub import try_to_load_from_cache

        model_path = try_to_load_from_cache(model_id, "config.json")
        if not model_path:
            raise ValueError(
                f"Model {model_id} must be downloaded first, check recommended models"
            )

        load_target = Path(model_path).parent
        loop = asyncio.get_running_loop()

        def _load_model() -> Any:
            from mlx_audio.stt.utils import load_model

            log.info("Loading MLX STT model %s", model_id)
            return load_model(load_target)

        self._stt_model = await loop.run_in_executor(None, _load_model)
        self._model_id_loaded = model_id

    class OutputType(TypedDict):
        text: str
        segments: list
        language: str

    def _build_generate_kwargs(self) -> dict[str, Any]:
        """Model specific keyword arguments forwarded to ``model.generate``."""
        return {}

    async def _export_audio(self, context: ProcessingContext) -> str:
        """Write the input audio to a temporary 16 kHz mono WAV file."""
        segment = await context.audio_to_audio_segment(self.audio)
        segment = segment.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
        segment.export(temp_path, format="wav")
        return temp_path

    @staticmethod
    def _normalize_segments(result: Any) -> list[dict[str, Any]]:
        """Coerce model segments / sentences into a list of plain dicts."""
        segments = getattr(result, "segments", None)
        if segments is None:
            segments = getattr(result, "sentences", None)
        if not segments:
            return []

        normalized: list[dict[str, Any]] = []
        for seg in segments:
            if isinstance(seg, dict):
                normalized.append(seg)
                continue
            normalized.append(
                {
                    "text": getattr(seg, "text", ""),
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                }
            )
        return normalized

    async def process(self, context: ProcessingContext) -> OutputType:
        self._ensure_supported_platform()

        if self.audio is None or not self.audio.is_set():
            raise ValueError("An input audio asset is required for transcription.")

        if self._stt_model is None or self._model_id_loaded != self._get_model_id():
            await self.preload_model(context)

        assert self._stt_model is not None

        audio_path = await self._export_audio(context)
        kwargs = self._build_generate_kwargs()

        def _run_transcription() -> Any:
            generate = self._stt_model.generate
            # Only forward kwargs accepted by this model's generate method.
            sig = inspect.signature(generate)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_var_kw:
                filtered = kwargs
            else:
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return generate(audio_path, **filtered)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _run_transcription)
        finally:
            with suppress(FileNotFoundError):
                os.remove(audio_path)

        text = str(getattr(result, "text", "") or "")
        segments = self._normalize_segments(result)
        language = str(getattr(result, "language", "") or "")

        return {"text": text, "segments": segments, "language": language}


class Parakeet(BaseMLXSpeechToText):
    """
    Transcribe audio using NVIDIA Parakeet via MLX.
    parakeet, mlx, asr, speech-to-text

    - High-accuracy multilingual transcription with word/sentence alignment
    - Runs locally on Apple Silicon through mlx-audio
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        PARAKEET_TDT_0_6B_V2 = "mlx-community/parakeet-tdt-0.6b-v2"
        PARAKEET_TDT_0_6B_V3 = "mlx-community/parakeet-tdt-0.6b-v3"

    model: Model = Field(
        default=Model.PARAKEET_TDT_0_6B_V3,
        description="Parakeet model variant to use for transcription.",
    )
    chunk_duration: float = Field(
        default=0.0,
        ge=0.0,
        le=1800.0,
        description="Chunk length in seconds for long audio. 0 disables chunking.",
    )

    @classmethod
    def get_title(cls):
        return "Parakeet"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HuggingFaceModel(repo_id=m.value) for m in cls.Model]

    def _build_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.chunk_duration > 0:
            kwargs["chunk_duration"] = self.chunk_duration
        return kwargs


class Qwen3ASR(BaseMLXSpeechToText):
    """
    Transcribe audio using Qwen3-ASR via MLX.
    qwen3, mlx, asr, speech-to-text

    - Alibaba's multilingual ASR with long-form chunking
    - Runs locally on Apple Silicon through mlx-audio
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        QWEN3_ASR_0_6B_8BIT = "mlx-community/Qwen3-ASR-0.6B-8bit"
        QWEN3_ASR_1_7B_8BIT = "mlx-community/Qwen3-ASR-1.7B-8bit"

    model: Model = Field(
        default=Model.QWEN3_ASR_0_6B_8BIT,
        description="Qwen3-ASR model variant to use for transcription.",
    )
    language: str = Field(
        default="",
        description="Optional language hint (e.g. English, Chinese). Empty auto-detects.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 is greedy / deterministic).",
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio", "language"]

    @classmethod
    def get_title(cls):
        return "Qwen3 ASR"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HuggingFaceModel(repo_id=m.value) for m in cls.Model]

    def _build_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.language.strip():
            kwargs["language"] = self.language.strip()
        return kwargs


class Qwen3ForcedAligner(BaseMLXSpeechToText):
    """
    Align a known transcript to audio using Qwen3-ForcedAligner via MLX.
    qwen3, mlx, alignment, timestamps

    - Produces word-level start/end timestamps for the provided text
    - Runs locally on Apple Silicon through mlx-audio
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        QWEN3_FORCED_ALIGNER_0_6B_8BIT = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    model: Model = Field(
        default=Model.QWEN3_FORCED_ALIGNER_0_6B_8BIT,
        description="Qwen3-ForcedAligner model variant to use.",
    )
    text: str = Field(
        default="",
        description="Transcript to align against the audio.",
    )
    language: str = Field(
        default="",
        description="Optional language hint (e.g. English). Empty auto-detects.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio", "text"]

    def required_inputs(self):
        return ["audio", "text"]

    @classmethod
    def get_input_fields(cls):
        return ["audio", "text"]

    @classmethod
    def get_title(cls):
        return "Qwen3 Forced Aligner"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HuggingFaceModel(repo_id=m.value) for m in cls.Model]

    def _build_generate_kwargs(self) -> dict[str, Any]:
        if not self.text.strip():
            raise ValueError("Text is required for forced alignment.")
        kwargs: dict[str, Any] = {"text": self.text.strip()}
        if self.language.strip():
            kwargs["language"] = self.language.strip()
        return kwargs


class MLXSpeechToText(BaseMLXSpeechToText):
    """
    Generic MLX speech-to-text node.
    mlx, asr, speech-to-text, transcription

    Run any speech-to-text model supported by the ``mlx-audio`` library by entering
    its Hugging Face repository id. Useful for models without a dedicated typed node.
    """

    _expose_as_tool: ClassVar[bool] = True

    model: str = Field(
        default="mlx-community/parakeet-tdt-0.6b-v3",
        description="Hugging Face repository id of any mlx-audio STT model.",
    )
    language: str = Field(
        default="",
        description="Optional language hint (model dependent). Empty auto-detects.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate (for autoregressive models).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio", "language"]

    @classmethod
    def get_title(cls):
        return "MLX Speech To Text"

    def _build_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.language.strip():
            kwargs["language"] = self.language.strip()
        return kwargs
