from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import logging
import os
import sys
import tempfile
from contextlib import suppress
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    ClassVar,
    Optional,
    TypedDict,
)
from pydantic import Field, PrivateAttr

from nodetool.metadata.types import AudioRef, HFTextToSpeech, HuggingFaceModel, Provider
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# MLX binds a Metal stream per thread, so a model loaded on one thread cannot
# be used from another: mlx-audio then raises "There is no Stream(gpu, N) in
# current thread." Loading with `run_in_executor(None, ...)` and iterating the
# generator on the event loop thread put the two on different threads. Pin
# every MLX call in this module to one thread. A single worker also serializes
# Metal access, which these models want anyway.
_MLX_AUDIO_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-audio")
_GENERATOR_EXHAUSTED = object()
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
    # When True the input text is split on newlines and each line is synthesized
    # separately. Dialogue / long-form models (Dia, OmniVoice, ...) override this
    # to False so the model receives the full text and can manage its own chunking.
    _split_text_by_lines: ClassVar[bool] = True
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

    @staticmethod
    def _configure_espeak() -> None:
        """Point phonemizer at the bundled espeak-ng library.

        Many mlx-audio TTS models (Kokoro, Kitten, Melo, ...) phonemize text via
        phonemizer's espeak backend, which raises "espeak not installed on your
        system" unless it can locate a library. ``espeakng_loader`` ships one, so
        wiring it here makes those models work without a system espeak-ng install.
        No-op when the optional phonemizer stack is unavailable, so models that
        don't need espeak still load.
        """
        try:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError:
            return
        with suppress(Exception):
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
        with suppress(Exception):
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    def _get_model_id(self) -> str:
        model = getattr(self, "model", None)
        repo_id = getattr(model, "repo_id", "") if model is not None else ""
        if not repo_id:
            raise ValueError("Model must be selected before loading MLX TTS.")
        return repo_id

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        self._configure_espeak()
        model_id = self._get_model_id()

        if self._tts_model is not None and self._model_id_loaded == model_id:
            return

        from nodetool.nodes.mlx._hf_cache import find_cached_snapshot

        load_target = find_cached_snapshot(model_id, "config.json")
        if load_target is None:
            raise ValueError(
                f"Model {model_id} must be downloaded first, check recommended models"
            )

        loop = asyncio.get_running_loop()

        def _load_model() -> Any:
            from mlx_audio.tts.utils import load_model

            log.info("Loading MLX TTS model %s", model_id)
            return load_model(load_target)

        self._tts_model = await loop.run_in_executor(_MLX_AUDIO_THREAD, _load_model)
        self._model_id_loaded = model_id

    class OutputType(TypedDict):
        audio: AudioRef | None
        chunk: Chunk

    def _iter_text_segments(self) -> list[str]:
        """Split the input text into the segments fed to ``model.generate``."""
        if self._split_text_by_lines:
            return [line.strip() for line in self.text.split("\n") if line.strip()]
        text = self.text.strip()
        return [text] if text else []

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        self._ensure_supported_platform()
        import numpy as np

        if self._tts_model is None or self._model_id_loaded != self._get_model_id():
            await self.preload_model(context)

        assert self._tts_model is not None

        segments = self._iter_text_segments()

        if not segments:
            raise ValueError("No text content to synthesize")

        params, cleanup_path = await self._build_generation_params(context)

        sample_rate = 24_000
        sample_rate_locked = False
        all_chunks: list[np.ndarray] = []
        try:
            for seg_idx, segment in enumerate(segments):
                # Update params with current text segment
                seg_params = params.copy()
                seg_params["text"] = segment

                log.debug(
                    f"Generating audio for segment {seg_idx + 1}/{len(segments)}: {segment[:50]}..."
                )

                loop = asyncio.get_running_loop()
                model = self._tts_model

                def _start() -> Any:
                    return model.generate(**seg_params)

                def _pull(iterator: Any) -> Any:
                    """Advance the generator and leave MLX behind on this thread."""
                    try:
                        item = next(iterator)
                    except StopIteration:
                        return _GENERATOR_EXHAUSTED
                    audio = getattr(item, "audio", None)
                    return (
                        None if audio is None else self._mx_array_to_numpy(audio),
                        getattr(item, "sample_rate", None),
                    )

                iterator = await loop.run_in_executor(_MLX_AUDIO_THREAD, _start)
                idx = -1
                while True:
                    pulled = await loop.run_in_executor(_MLX_AUDIO_THREAD, _pull, iterator)
                    if pulled is _GENERATOR_EXHAUSTED:
                        break
                    idx += 1
                    audio, result_sr = pulled
                    if result_sr:
                        new_sr = int(result_sr)
                        # All collected chunks share a single sample rate. If the
                        # model changes rate after audio has been emitted, the
                        # concatenation below would be played back at the wrong
                        # rate, so fail loudly instead of corrupting the output.
                        if sample_rate_locked and new_sr != sample_rate:
                            raise ValueError(
                                f"{type(self).__name__} emitted multiple sample rates "
                                f"({sample_rate} and {new_sr}); cannot concatenate audio safely."
                            )
                        sample_rate = new_sr
                    chunk = audio
                    if chunk is None or chunk.size == 0:
                        log.debug("Skipping empty MLX audio segment %d", idx)
                        continue
                    chunk_msg, audio_int16 = self._encode_chunk(chunk, sample_rate)
                    yield {"chunk": chunk_msg, "audio": None}
                    all_chunks.append(audio_int16)
                    sample_rate_locked = True
        finally:
            if cleanup_path:
                with suppress(FileNotFoundError):
                    os.remove(cleanup_path)

        if not all_chunks:
            raise ValueError("MLX TTS did not produce any audio")

        combined = all_chunks[0] if len(all_chunks) == 1 else np.concatenate(all_chunks)
        yield {
            "audio": await context.audio_from_numpy(combined, sample_rate),
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

    def _encode_chunk(
        self, chunk: np.ndarray, sample_rate: int = 24_000
    ) -> tuple[Chunk, np.ndarray]:

        audio_int16 = self._to_int16(chunk)
        chunk_msg = Chunk(
            content=base64.b64encode(audio_int16.tobytes()).decode("utf-8"),
            content_type="audio",
            content_metadata={
                "sample_rate": sample_rate,
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

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.KOKORO_82M.value),
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
            HFTextToSpeech(repo_id=cls.Model.KOKORO_82M.value),
            HFTextToSpeech(repo_id=cls.Model.KOKORO_82M_BF16.value),
            HFTextToSpeech(repo_id=cls.Model.KOKORO_82M_4BIT.value),
            HFTextToSpeech(repo_id=cls.Model.KOKORO_82M_6BIT.value),
            HFTextToSpeech(repo_id=cls.Model.KOKORO_82M_8BIT.value),
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

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.SESAME_1B.value),
        description="Sesame/CSM model variant to load.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
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
            HFTextToSpeech(repo_id=cls.Model.SESAME_1B.value),
            HFTextToSpeech(repo_id=cls.Model.SESAME_1B_8BIT.value),
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
        if self.reference_audio is None or not self.reference_audio.is_set():
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

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.SPARK_TTS_0_5B_BF16.value),
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
            HFTextToSpeech(repo_id=cls.Model.SPARK_TTS_0_5B_BF16.value),
            HFTextToSpeech(repo_id=cls.Model.SPARK_TTS_0_5B_8BIT.value),
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


class Qwen3TTS(BaseMLXTTS):
    """MLX Qwen3-TTS multilingual text-to-speech with speaker voices and voice design."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        QWEN3_TTS_06B_BASE = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
        QWEN3_TTS_06B_CUSTOM = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
        QWEN3_TTS_17B_BASE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        QWEN3_TTS_17B_BASE_8BIT = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
        QWEN3_TTS_17B_CUSTOM_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        QWEN3_TTS_17B_CUSTOM_6BIT = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
        QWEN3_TTS_17B_VOICE_DESIGN = (
            "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
        )

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.QWEN3_TTS_17B_BASE_8BIT.value),
        description="Qwen3-TTS model variant to load.",
    )
    voice: str = Field(
        default="Chelsie",
        description="Speaker name for multi-speaker models (e.g. Chelsie, Ethan, Vivian).",
    )
    language: str = Field(
        default="auto",
        description="Language code (e.g. auto, English, Chinese, Japanese, Korean).",
    )
    instruct: str = Field(
        default="",
        description="Optional voice-design instruction describing the target voice (VoiceDesign models).",
    )
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the talker model.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "language", "temperature", "speed"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "Qwen3 TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("Qwen3-TTS speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "lang_code": self.language or "auto",
                "temperature": self.temperature,
            }
        )
        # Omit the key entirely when unset: passing voice=None overrides the
        # pipeline's default speaker and breaks the speaker lookup.
        if self.voice:
            params["voice"] = self.voice
        if self.instruct.strip():
            params["instruct"] = self.instruct.strip()
        return params, cleanup


class KittenTTS(BaseMLXTTS):
    """MLX KittenTTS compact, edge-friendly English text-to-speech."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        KITTEN_NANO = "mlx-community/kitten-tts-nano-0.8"
        KITTEN_MICRO = "mlx-community/kitten-tts-micro-0.8"
        KITTEN_MINI = "mlx-community/kitten-tts-mini-0.8"

    class Voice(str, Enum):
        EXPR_VOICE_2_M = "expr-voice-2-m"
        EXPR_VOICE_2_F = "expr-voice-2-f"
        EXPR_VOICE_3_M = "expr-voice-3-m"
        EXPR_VOICE_3_F = "expr-voice-3-f"
        EXPR_VOICE_4_M = "expr-voice-4-m"
        EXPR_VOICE_4_F = "expr-voice-4-f"
        EXPR_VOICE_5_M = "expr-voice-5-m"
        EXPR_VOICE_5_F = "expr-voice-5-f"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.KITTEN_NANO.value),
        description="KittenTTS model variant to load.",
    )
    voice: Voice = Field(
        default=Voice.EXPR_VOICE_5_M,
        description="Voice preset supported by KittenTTS.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "speed"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "Kitten TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("KittenTTS speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params["voice"] = self.voice.value
        return params, cleanup


class DiaTTS(BaseMLXTTS):
    """MLX Dia dialogue text-to-speech.

    Use the ``[S1]`` and ``[S2]`` speaker tags inside the text to script multi-turn
    dialogue. Optionally provide reference audio to clone a voice.
    """

    _expose_as_tool: ClassVar[bool] = True
    # Dia interprets the whole script (including newlines / speaker tags) at once.
    _split_text_by_lines: ClassVar[bool] = False

    class Model(str, Enum):
        DIA_1_6B = "mlx-community/Dia-1.6B"
        DIA_1_6B_FP16 = "mlx-community/Dia-1.6B-fp16"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.DIA_1_6B_FP16.value),
        description="Dia model variant to load.",
    )
    text: str = Field(
        default="[S1] Hello, how are you? [S2] I'm doing great, thanks for asking!",
        description="Dialogue script. Use [S1] and [S2] to tag speaker turns.",
    )
    temperature: float = Field(
        default=1.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for voice cloning.",
    )
    reference_text: str = Field(
        default="",
        description="Transcript of the reference audio (required when cloning a voice).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "temperature", "top_p"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "Dia TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            if self.reference_text.strip():
                params["ref_text"] = self.reference_text.strip()
            cleanup = ref_path
        return params, cleanup


class OuteTTS(BaseMLXTTS):
    """MLX OuteTTS efficient multilingual text-to-speech with optional voice cloning."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        OUTETTS_1_0_0_6B = "mlx-community/OuteTTS-1.0-0.6B-fp16"
        OUTETTS_1_0_0_6B_8BIT = "mlx-community/OuteTTS-1.0-0.6B-8bit"
        LLAMA_OUTETTS_1_0_1B = "mlx-community/Llama-OuteTTS-1.0-1B-fp16"
        LLAMA_OUTETTS_1_0_1B_8BIT = "mlx-community/Llama-OuteTTS-1.0-1B-8bit"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.OUTETTS_1_0_0_6B.value),
        description="OuteTTS model variant to load.",
    )
    voice: str = Field(
        default="",
        description="Optional built-in speaker name. Leave empty to use a default voice.",
    )
    temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for voice cloning.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "temperature", "top_p"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "Oute TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )
        if self.voice.strip():
            params["voice"] = self.voice.strip()
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            cleanup = ref_path
        return params, cleanup


class OmniVoiceTTS(BaseMLXTTS):
    """MLX OmniVoice zero-shot multilingual text-to-speech (646+ languages)."""

    _expose_as_tool: ClassVar[bool] = True
    _split_text_by_lines: ClassVar[bool] = False

    class Model(str, Enum):
        OMNIVOICE_BF16 = "mlx-community/OmniVoice-bf16"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.OMNIVOICE_BF16.value),
        description="OmniVoice model variant to load.",
    )
    language: str = Field(
        default="english",
        description="Target language name (e.g. english, spanish, french, chinese).",
    )
    duration_s: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Target duration in seconds. 0 lets the model decide automatically.",
    )
    num_steps: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Number of diffusion sampling steps.",
    )
    guidance_scale: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance scale.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for voice cloning.",
    )
    reference_text: str = Field(
        default="",
        description="Transcript of the reference audio (used when cloning a voice).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "language", "duration_s", "num_steps", "guidance_scale"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "OmniVoice TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "language": self.language or "english",
                "num_steps": self.num_steps,
                "guidance_scale": self.guidance_scale,
            }
        )
        if self.duration_s > 0:
            params["duration_s"] = self.duration_s
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            if self.reference_text.strip():
                params["ref_text"] = self.reference_text.strip()
            cleanup = ref_path
        return params, cleanup


class MeloTTS(BaseMLXTTS):
    """MLX MeloTTS lightweight VITS2-based multilingual text-to-speech."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        MELOTTS_ENGLISH = "mlx-community/MeloTTS-English-MLX"

    class LanguageCode(str, Enum):
        EN_US = "EN-US"
        EN_BR = "EN-BR"
        EN_INDIA = "EN_INDIA"
        EN_AU = "EN-AU"
        EN_DEFAULT = "EN-Default"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.MELOTTS_ENGLISH.value),
        description="MeloTTS model variant to load.",
    )
    language: LanguageCode = Field(
        default=LanguageCode.EN_US,
        description="Accent / language code for the speaker.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "language", "speed"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "language"]

    @classmethod
    def get_title(cls):
        return "Melo TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("MeloTTS speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params["lang_code"] = self.language.value
        return params, cleanup


class VoxtralTTS(BaseMLXTTS):
    """MLX Voxtral TTS — Mistral's multilingual text-to-speech with voice presets."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        VOXTRAL_4B_TTS = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"

    class Voice(str, Enum):
        NEUTRAL_FEMALE = "neutral_female"
        NEUTRAL_MALE = "neutral_male"
        CASUAL_FEMALE = "casual_female"
        CASUAL_MALE = "casual_male"
        CHEERFUL_FEMALE = "cheerful_female"
        AR_MALE = "ar_male"
        DE_FEMALE = "de_female"
        DE_MALE = "de_male"
        ES_FEMALE = "es_female"
        ES_MALE = "es_male"
        FR_FEMALE = "fr_female"
        FR_MALE = "fr_male"
        HI_FEMALE = "hi_female"
        HI_MALE = "hi_male"
        IT_FEMALE = "it_female"
        IT_MALE = "it_male"
        NL_FEMALE = "nl_female"
        NL_MALE = "nl_male"
        PT_FEMALE = "pt_female"
        PT_MALE = "pt_male"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.VOXTRAL_4B_TTS.value),
        description="Voxtral TTS model variant to load.",
    )
    voice: Voice = Field(
        default=Voice.NEUTRAL_FEMALE,
        description="Voice preset supported by Voxtral TTS.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "temperature"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "Voxtral TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "voice": self.voice.value,
                "temperature": self.temperature,
            }
        )
        return params, cleanup


class ChatterboxTTS(BaseMLXTTS):
    """MLX Chatterbox expressive multilingual text-to-speech with voice cloning."""

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        CHATTERBOX_FP16 = "mlx-community/chatterbox-fp16"
        CHATTERBOX_TTS_FP16 = "mlx-community/Chatterbox-TTS-fp16"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.CHATTERBOX_FP16.value),
        description="Chatterbox model variant to load.",
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotional exaggeration intensity.",
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance weight (pacing / fidelity).",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for voice cloning.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "exaggeration", "cfg_weight", "temperature"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "Chatterbox TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight,
                "temperature": self.temperature,
            }
        )
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            cleanup = ref_path
        return params, cleanup


class HiggsAudioTTS(BaseMLXTTS):
    """MLX Higgs Audio conversational text-to-speech with zero-shot voice cloning."""

    _expose_as_tool: ClassVar[bool] = True
    _split_text_by_lines: ClassVar[bool] = False

    class Model(str, Enum):
        HIGGS_V2_3B_Q6 = "mlx-community/higgs-audio-v2-3B-mlx-q6"
        HIGGS_V2_3B_Q8 = "mlx-community/higgs-audio-v2-3B-mlx-q8"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.HIGGS_V2_3B_Q8.value),
        description="Higgs Audio model variant to load.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for zero-shot voice cloning.",
    )
    reference_text: str = Field(
        default="",
        description="Transcript of the reference audio (used when cloning a voice).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "temperature"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "Higgs Audio TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params["temperature"] = self.temperature
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            if self.reference_text.strip():
                params["ref_text"] = self.reference_text.strip()
            cleanup = ref_path
        return params, cleanup


class LongCatAudioTTS(BaseMLXTTS):
    """MLX LongCat-AudioDiT diffusion text-to-speech with zero-shot voice cloning."""

    _expose_as_tool: ClassVar[bool] = True
    _split_text_by_lines: ClassVar[bool] = False

    class Model(str, Enum):
        LONGCAT_1B_BF16 = "mlx-community/LongCat-AudioDiT-1B-bf16"
        LONGCAT_1B_4BIT = "mlx-community/LongCat-AudioDiT-1B-4bit"
        LONGCAT_1B_8BIT = "mlx-community/LongCat-AudioDiT-1B-8bit"
        LONGCAT_3_5B_BF16 = "mlx-community/LongCat-AudioDiT-3.5B-bf16"
        LONGCAT_3_5B_4BIT = "mlx-community/LongCat-AudioDiT-3.5B-4bit"
        LONGCAT_3_5B_8BIT = "mlx-community/LongCat-AudioDiT-3.5B-8bit"

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id=Model.LONGCAT_1B_4BIT.value),
        description="LongCat-AudioDiT model variant to load.",
    )
    language: str = Field(
        default="en",
        description="Language code (e.g. en, zh).",
    )
    steps: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Number of diffusion sampling steps.",
    )
    cfg_strength: float = Field(
        default=4.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance strength.",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for zero-shot voice cloning.",
    )
    reference_text: str = Field(
        default="",
        description="Transcript of the reference audio (used when cloning a voice).",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "language", "steps", "cfg_strength"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def get_title(cls):
        return "LongCat Audio TTS"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFTextToSpeech(repo_id=m.value) for m in cls.Model]

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        params.update(
            {
                "lang_code": self.language or "en",
                "steps": self.steps,
                "cfg_strength": self.cfg_strength,
            }
        )
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            if self.reference_text.strip():
                params["ref_text"] = self.reference_text.strip()
            cleanup = ref_path
        return params, cleanup


class MLXTextToSpeech(BaseMLXTTS):
    """Generic MLX text-to-speech node.

    Run any text-to-speech model supported by the ``mlx-audio`` library by entering
    its Hugging Face repository id. Useful for models that do not have a dedicated
    typed node yet.
    """

    _expose_as_tool: ClassVar[bool] = True
    _split_text_by_lines: ClassVar[bool] = False

    model: HFTextToSpeech = Field(
        default=HFTextToSpeech(repo_id="mlx-community/Kokoro-82M-bf16"),
        description="Any mlx-audio TTS model. Pick a recommended model or enter a Hugging Face repo id.",
    )
    voice: str = Field(
        default="",
        description="Optional voice / speaker preset (model dependent).",
    )
    language: str = Field(
        default="",
        description="Optional language code (model dependent, e.g. 'a', 'en', 'auto').",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature. 0 uses the model default.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )
    reference_audio: AudioRef = Field(
        default=AudioRef(),
        description="Optional reference audio for models that support voice cloning.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "voice", "language", "speed"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "voice"]

    @classmethod
    def get_title(cls):
        return "MLX Text To Speech"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        # A representative cross-section of mlx-audio TTS families. The model
        # select still lets users enter any other Hugging Face repo id.
        return [
            HFTextToSpeech(repo_id="mlx-community/Kokoro-82M-bf16"),
            HFTextToSpeech(repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"),
            HFTextToSpeech(repo_id="mlx-community/kitten-tts-nano-0.8"),
            HFTextToSpeech(repo_id="mlx-community/Dia-1.6B-fp16"),
            HFTextToSpeech(repo_id="mlx-community/csm-1b"),
            HFTextToSpeech(repo_id="mlx-community/OuteTTS-1.0-0.6B-fp16"),
        ]

    def _normalize_speed(self) -> float:
        value = float(self.speed)
        if not 0.5 <= value <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0.")
        return value

    async def _build_generation_params(
        self, context: ProcessingContext
    ) -> tuple[dict[str, Any], Optional[str]]:
        params, cleanup = await super()._build_generation_params(context)
        if self.voice.strip():
            params["voice"] = self.voice.strip()
        if self.language.strip():
            params["lang_code"] = self.language.strip()
        if self.temperature > 0:
            params["temperature"] = self.temperature
        if self.reference_audio is not None and self.reference_audio.is_set():
            ref_path = await self._export_reference_audio(context, self.reference_audio)
            params["ref_audio"] = ref_path
            cleanup = ref_path
        return params, cleanup
