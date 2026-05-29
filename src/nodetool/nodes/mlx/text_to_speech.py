from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import tempfile
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    ClassVar,
    Optional,
    TypedDict,
    cast,
)
from pydantic import Field, PrivateAttr

from nodetool.metadata.types import AudioRef, HuggingFaceModel, Provider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import try_to_load_from_cache

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class BaseMLXTTS(BaseNode):
    """Shared functionality for MLX-based text to speech nodes."""

    _body: ClassVar[str] = "content_card"

    text: str = Field(
        default="Hello from MLX TTS.",
        description="Text content to synthesize into speech.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Playback speed multiplier.",
    )

    _provider: ClassVar[Provider] = Provider.MLX
    _tts_model: Any | None = PrivateAttr(default=None)
    _model_id_loaded: str | None = PrivateAttr(default=None)

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseMLXTTS

    def required_inputs(self):
        return ["text"]

    @classmethod
    def get_input_fields(cls):
        return ["text"]

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
            raise RuntimeError("MLX TTS requires macOS (Apple Silicon / MLX).")

    def _get_model_id(self) -> str:
        model = getattr(self, "model", None)
        if model is None:
            raise ValueError("Model must be selected before loading MLX TTS.")
        if isinstance(model, Enum):
            return cast(str, model.value)
        return str(model)

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        model_id = self._get_model_id()

        if self._tts_model is not None and self._model_id_loaded == model_id:
            return

        from huggingface_hub import try_to_load_from_cache

        model_path = try_to_load_from_cache(model_id, "config.json")
        if not model_path:
            raise ValueError(f"Model {model_id} must be downloaded first")

        load_target = Path(model_path).parent
        loop = asyncio.get_running_loop()

        def _load_model() -> Any:
            from mlx_audio.tts.utils import load_model

            log.info("Loading MLX TTS model %s", model_id)
            return load_model(load_target)

        self._tts_model = await loop.run_in_executor(None, _load_model)
        self._model_id_loaded = model_id

    class OutputType(TypedDict):
        audio: AudioRef | None
        chunk: Chunk

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        self._ensure_supported_platform()
        import numpy as np

        if self._tts_model is None or self._model_id_loaded != self._get_model_id():
            await self.preload_model(context)

        assert self._tts_model is not None

        # Split text by newlines and process each line separately
        lines = [line.strip() for line in self.text.split("\n") if line.strip()]

        if not lines:
            raise ValueError("No text content to synthesize")

        params, cleanup_path = await self._build_generation_params(context)

        all_chunks: list[np.ndarray] = []
        try:
            for line_idx, line in enumerate(lines):
                # Update params with current line
                line_params = params.copy()
                line_params["text"] = line

                log.debug(
                    f"Generating audio for line {line_idx + 1}/{len(lines)}: {line[:50]}..."
                )

                for idx, result in enumerate(self._tts_model.generate(**line_params)):
                    audio = getattr(result, "audio", None)
                    chunk = self._mx_array_to_numpy(audio)
                    if chunk.size == 0:
                        log.debug("Skipping empty MLX audio segment %d", idx)
                        continue
                    chunk_msg, audio_int16 = self._encode_chunk(chunk)
                    yield {"chunk": chunk_msg, "audio": None}
                    all_chunks.append(audio_int16)
        finally:
            if cleanup_path:
                with suppress(FileNotFoundError):
                    os.remove(cleanup_path)

        if not all_chunks:
            raise ValueError("MLX TTS did not produce any audio")

        combined = all_chunks[0] if len(all_chunks) == 1 else np.concatenate(all_chunks)
        yield {
            "audio": await context.audio_from_numpy(combined, 24_000),
            "chunk": Chunk(content="", done=True, content_type="audio"),
        }

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Return generation params and optional cleanup path."""
        return {
            "text": self.text,
            "speed": self._normalize_speed(),
            "stream": False,
            "verbose": False,
        }, None

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.0 <= value <= 2.0:
            raise ValueError("Speed must be between 0.0 and 2.0 for this model.")
        return value

    @staticmethod
    def _mx_array_to_numpy(audio: Any) -> np.ndarray:
        import mlx.core as mx
        import numpy as np

        if audio is None:
            return np.array([], dtype=np.float32)
        if hasattr(audio, "astype") and hasattr(audio, "numpy"):
            try:
                return audio.astype(mx.float32).numpy()
            except Exception:  # pragma: no cover
                pass
        return np.asarray(audio, dtype=np.float32)

    def _encode_chunk(self, chunk: np.ndarray) -> tuple[Chunk, np.ndarray]:
        import numpy as np

        audio_int16 = self._to_int16(chunk)
        chunk_msg = Chunk(
            content=base64.b64encode(audio_int16.tobytes()).decode("utf-8"),
            content_type="audio",
            content_metadata={
                "sample_rate": 24_000,
                "channels": 1,
                "dtype": "int16",
            },
            done=False,
        )
        return chunk_msg, audio_int16

    @staticmethod
    def _to_int16(chunk: np.ndarray) -> np.ndarray:
        import numpy as np

        if chunk.dtype != np.int16:
            audio = np.clip(chunk, -1.0, 1.0)
            return (audio * 32767.0).astype(np.int16)
        return chunk.astype(np.int16, copy=False)

    async def _export_reference_audio(
        self, context: ProcessingContext, audio_ref: AudioRef
    ) -> str:
        segment = await context.audio_to_audio_segment(audio_ref)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
        segment.export(temp_path, format="wav")
        return temp_path


class KokoroTTS(BaseMLXTTS):
    """MLX Kokoro text-to-speech."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        KOKORO_82M = "prince-canuma/Kokoro-82M"
        KOKORO_82M_BF16 = "mlx-community/Kokoro-82M-bf16"
        KOKORO_82M_4BIT = "mlx-community/Kokoro-82M-4bit"
        KOKORO_82M_6BIT = "mlx-community/Kokoro-82M-6bit"
        KOKORO_82M_8BIT = "mlx-community/Kokoro-82M-8bit"

    class LanguageCode(str, Enum):
        AMERICAN_ENGLISH = "a"
        BRITISH_ENGLISH = "b"
        ESPANOL = "e"
        FRENCH = "f"
        HINDI = "h"
        ITALIAN = "i"
        PORTUGUESE = "p"
        JAPANESE = "j"
        CHINESE = "z"
        KOREAN = "k"
        RUSSIAN = "r"
        TURKISH = "t"
        VIETNAMESE = "v"
        ARABIC = "a"
        GERMAN = "g"
        POLISH = "p"
        ROMANIAN = "r"
        UKRAINIAN = "u"

    class Voice(str, Enum):
        # af_*
        AF_ALLOY = "af_alloy"
        AF_AOEDE = "af_aoede"
        AF_BELLA = "af_bella"
        AF_HEART = "af_heart"
        AF_JESSICA = "af_jessica"
        AF_KORE = "af_kore"
        AF_NICOLE = "af_nicole"
        AF_NOVA = "af_nova"
        AF_RIVER = "af_river"
        AF_SARAH = "af_sarah"
        AF_SKY = "af_sky"
        # am_*
        AM_ADAM = "am_adam"
        AM_ECHO = "am_echo"
        AM_ERIC = "am_eric"
        AM_FENRIR = "am_fenrir"
        AM_LIAM = "am_liam"
        AM_MICHAEL = "am_michael"
        AM_ONYX = "am_onyx"
        AM_PUCK = "am_puck"
        AM_SANTA = "am_santa"
        # bf_*
        BF_ALICE = "bf_alice"
        BF_EMMA = "bf_emma"
        BF_ISABELLA = "bf_isabella"
        BF_LILY = "bf_lily"
        # bm_*
        BM_DANIEL = "bm_daniel"
        BM_FABLE = "bm_fable"
        BM_GEORGE = "bm_george"
        BM_LEWIS = "bm_lewis"
        # ef_*
        EF_DORA = "ef_dora"
        # em_*
        EM_ALEX = "em_alex"
        EM_SANTA = "em_santa"
        # ff_*
        FF_SIWIS = "ff_siwis"
        # hf_*
        HF_ALPHA = "hf_alpha"
        HF_BETA = "hf_beta"
        # hm_*
        HM_OMEGA = "hm_omega"
        HM_PSI = "hm_psi"
        # if_*
        IF_SARA = "if_sara"
        # im_*
        IM_NICOLA = "im_nicola"
        # jf_*
        JF_ALPHA = "jf_alpha"
        JF_GONGITSUNE = "jf_gongitsune"
        JF_NEZUMI = "jf_nezumi"
        JF_TEBUKURO = "jf_tebukuro"
        # jm_*
        JM_KUMO = "jm_kumo"
        # pf_*
        PF_DORA = "pf_dora"
        # pm_*
        PM_ALEX = "pm_alex"
        PM_SANTA = "pm_santa"
        # zf_*
        ZF_XIAOBEI = "zf_xiaobei"
        ZF_XIAONI = "zf_xiaoni"
        ZF_XIAOXIAO = "zf_xiaoxiao"
        ZF_XIAOYI = "zf_xiaoyi"

    model: Model = Field(
        default=Model.KOKORO_82M,
        description="Kokoro model variant to load.",
    )
    voice: Voice = Field(
        default=Voice.AF_HEART,
        description="Voice preset supported by Kokoro (e.g. af_heart, am_adam, bf_emma).",
    )
    language: LanguageCode = Field(
        default=LanguageCode.AMERICAN_ENGLISH,
        description=("Language code"),
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.5,
        description="Sampling temperature passed to the Kokoro generator.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier for Kokoro (0.5–2.0).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "language", "temperature", "speed"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "Kokoro TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M.value),
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M_BF16.value),
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M_4BIT.value),
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M_6BIT.value),
            HuggingFaceModel(repo_id=cls.Model.KOKORO_82M_8BIT.value),
        ]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("Kokoro speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "voice": self.voice.value,
                "lang_code": self.language.value,
                "temperature": self.temperature,
            }
        )
        return params, cleanup


class SesameTTS(BaseMLXTTS):
    """MLX Sesame / CSM text-to-speech with reference audio cloning."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        SESAME_1B = "mlx-community/csm-1b"
        SESAME_1B_8BIT = "mlx-community/csm-1b-8bit"

    model: Model = Field(
        default=Model.SESAME_1B,
        description="Sesame/CSM model variant to load.",
    )
    reference_audio: AudioRef = Field(
        description="Reference audio clip used for voice cloning.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier for Sesame (0.5–2.0).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "reference_audio", "speed"]

    def required_inputs(self):
        return ["text", "reference_audio"]

    @classmethod
    def get_input_fields(cls):
        return ["text", "reference_audio"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "Sesame TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=cls.Model.SESAME_1B.value),
            HuggingFaceModel(repo_id=cls.Model.SESAME_1B_8BIT.value),
        ]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("Sesame speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, _ = await super()._build_generation_params(context)
        if self.reference_audio is None:
            raise ValueError("Reference audio is required for Sesame TTS.")
        ref_path = await self._export_reference_audio(context, self.reference_audio)
        params["ref_audio"] = ref_path
        return params, ref_path


class SparkTTS(BaseMLXTTS):
    """MLX Spark text-to-speech."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        SPARK_TTS_0_5B_BF16 = "mlx-community/Spark-TTS-0.5B-bf16"
        SPARK_TTS_0_5B_8BIT = "mlx-community/Spark-TTS-0.5B-8bit"

    class Speed(str, Enum):
        VERY_LOW = "very_low"
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"
        VERY_HIGH = "very_high"

    class Pitch(str, Enum):
        VERY_LOW = "very_low"
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"
        VERY_HIGH = "very_high"

    class Gender(str, Enum):
        FEMALE = "female"
        MALE = "male"

    model: Model = Field(
        default=Model.SPARK_TTS_0_5B_BF16,
        description="Spark model variant to load.",
    )
    speed: Speed = Field(
        default=Speed.MODERATE,
        description="Spark speed preset (very_low, low, moderate, high, very_high).",
    )
    voice: str | None = Field(
        default=None,
        description="Optional Spark voice preset.",
    )
    pitch: Pitch = Field(
        default=Pitch.MODERATE,
        description="Spark pitch preset.",
    )
    gender: Gender = Field(
        default=Gender.FEMALE,
        description="Spark voice gender.",
    )

    _SPEED_VALUES: ClassVar[dict[Speed, float]] = {
        Speed.VERY_LOW: 0.0,
        Speed.LOW: 0.5,
        Speed.MODERATE: 1.0,
        Speed.HIGH: 1.5,
        Speed.VERY_HIGH: 2.0,
    }
    _PITCH_VALUES: ClassVar[dict[Pitch, float]] = {
        Pitch.VERY_LOW: 0.0,
        Pitch.LOW: 0.5,
        Pitch.MODERATE: 1.0,
        Pitch.HIGH: 1.5,
        Pitch.VERY_HIGH: 2.0,
    }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "speed", "voice", "pitch", "gender"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice", "gender"]

    @classmethod
    def get_title(cls):
        return "Spark TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=cls.Model.SPARK_TTS_0_5B_BF16.value),
            HuggingFaceModel(repo_id=cls.Model.SPARK_TTS_0_5B_8BIT.value),
        ]

    def _normalize_speed(self) -> float:
        return self._SPEED_VALUES[self.speed]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "pitch": self._PITCH_VALUES[self.pitch],
                "gender": self.gender.value,
            }
        )
        if self.voice:
            params["voice"] = self.voice
        return params, cleanup
