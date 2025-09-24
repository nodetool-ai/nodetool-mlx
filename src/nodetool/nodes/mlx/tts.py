from __future__ import annotations

import asyncio
import logging
import base64
from pathlib import Path
import sys
from enum import Enum
from typing import Any, ClassVar, Generator, Literal, Optional

from huggingface_hub import try_to_load_from_cache
import mlx.core as mx
from nodetool.workflows.types import Chunk
import numpy as np
from pydantic import Field, PrivateAttr

from nodetool.metadata.types import AudioRef, HuggingFaceModel, Provider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


class TTS(BaseNode):
    """
    Generate speech locally using mlx-audio TTS models on Apple Silicon.
    tts, audio, speech, mlx, kokoro, sesame

    Use cases:
    - Fast offline TTS generation on Apple Silicon devices
    - Experiment with Kokoro/Sesame voices without GPU dependencies
    - Produce narration or assistant voices with low-latency local inference
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        KOKORO_82M = "prince-canuma/Kokoro-82M"
        KOKORO_82M_BF16 = "mlx-community/Kokoro-82M-bf16"
        SESAME_1B = "mlx-community/csm-1b"

    text: str = Field(
        default="Hello from MLX TTS.",
        description="Text content to synthesize into speech.",
    )
    model: Model = Field(
        default=Model.KOKORO_82M,
        description="MLX TTS model repo ID or local path.",
    )
    voice: str = Field(
        default="af_heart",
        description=(
            "Voice preset supported by the selected model. Kokoro exposes af_*/am_*/bf_* voices; "
            "Sesame expects reference-audio-assisted voices."
        ),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Playback speed multiplier supported by Kokoro models.",
    )
    lang_code: str = Field(
        default="a",
        description=(
            "Language code when supported (Kokoro: a=American English, b=British English, j=Japanese, etc.)."
        ),
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.5,
        description="Sampling temperature passed to the underlying generator when supported.",
    )
    _provider: ClassVar[Provider] = Provider.MLX
    _tts_model: Any | None = PrivateAttr(default=None)
    _model_id_loaded: str | None = PrivateAttr(default=None)

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        model_id = self.model.value

        if self._tts_model is not None and self._model_id_loaded == model_id:
            return

        model_path = try_to_load_from_cache(model_id, "config.json")
        if not model_path:
            raise ValueError(
                f"Model {model_id} must be downloaded first, check recommended models"
            )

        loop = asyncio.get_running_loop()

        def _load_model() -> Any:
            from mlx_audio.tts.utils import load_model

            log.info("Loading MLX TTS model %s", model_path)
            return load_model(Path(model_path).parent)

        self._tts_model = await loop.run_in_executor(None, _load_model)
        self._model_id_loaded = model_id

    @classmethod
    def get_title(cls):
        return "MLX Text To Speech"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M.value),
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M_BF16.value),
            HuggingFaceModel(repo_id=cls.Model.SESAME_1B.value),
        ]

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @classmethod
    def _ensure_supported_platform(cls) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("MLX TTS requires macOS (Apple Silicon / MLX).")

    @classmethod
    def return_type(cls) -> dict[str, Any]:
        return {"audio": AudioRef, "chunk": Chunk}

    async def gen_process(self, context: ProcessingContext):
        self._ensure_supported_platform()

        if self._tts_model is None or self._model_id_loaded != self.model.value:
            await self.preload_model(context)

        assert self._tts_model is not None

        loop = asyncio.get_running_loop()

        async def _generate():
            model = self._tts_model
            assert model is not None, "Model not loaded"

            log.debug(
                "Running MLX TTS generation: voice=%s speed=%.2f lang=%s",
                self.voice,
                self.speed,
                self.lang_code,
            )

            results = model.generate(
                text=self.text,
                voice=self.voice,
                speed=self.speed,
                lang_code=self.lang_code,
                temperature=self.temperature,
                stream=False,
            )

            sample_rate: Optional[int] = getattr(model, "sample_rate", None)

            for idx, result in enumerate(results):
                sample_rate = getattr(result, "sample_rate", sample_rate)
                audio = getattr(result, "audio", None)
                chunk = self._mx_array_to_numpy(audio)

                if chunk.size == 0:
                    log.debug("Skipping empty MLX audio segment %d", idx)
                    continue

                yield chunk
                log.debug(
                    "Collected MLX audio segment %d with %d samples",
                    idx,
                    chunk.shape[0],
                )

        yield "chunk", Chunk(
            content="",
            done=True,
        )

        chunks = []
        async for chunk in _generate():
            # Convert to int16 for audio output
            if chunk.dtype != np.int16:
                # Assume audio is float32 in [-1, 1], scale to int16
                audio_int16 = np.clip(chunk, -1.0, 1.0)
                audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
            else:
                audio_int16 = chunk.astype(np.int16, copy=False)
            yield "chunk", Chunk(
                content=base64.b64encode(audio_int16.tobytes()).decode("utf-8"),
                content_type="audio",
                content_metadata={
                    "sample_rate": 24_000,
                    "channels": 1,
                    "dtype": "int16",
                },
                done=False,
            )
            chunks.append(audio_int16)

        combined = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        sr = 24_000
        yield "audio", await context.audio_from_numpy(combined, sr)

    @staticmethod
    def _mx_array_to_numpy(audio: Any) -> np.ndarray:
        if audio is None:
            return np.array([], dtype=np.float32)
        if hasattr(audio, "astype") and hasattr(audio, "numpy"):
            try:
                return audio.astype(mx.float32).numpy()
            except Exception:  # pragma: no cover - fallback for non-MLX arrays
                pass
        return np.asarray(audio, dtype=np.float32)
