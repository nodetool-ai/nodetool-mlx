from __future__ import annotations

import asyncio
import io
import random
import sys
import wave
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, HuggingFaceModel
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)

SA3_REPO_ID = "stabilityai/stable-audio-3-optimized"
SAMPLE_RATE = 44100


class DiTModel(str, Enum):
    """Stable Audio 3 DiT variants."""

    SM_MUSIC = "sm-music"  # 50M, fast music
    SM_SFX = "sm-sfx"  # 50M, sound effects
    MEDIUM = "medium"  # 1.4B, higher fidelity music


class BaseStableAudio3(BaseNode):
    """Shared machinery for the MLX-native Stable Audio 3 nodes.

    Generate high quality 44.1 kHz stereo audio locally on Apple Silicon using
    Stability AI's optimized MLX implementation — no PyTorch at runtime. The
    underlying runtime is vendored under ``nodetool.mlx.stable_audio_3``.
    """

    _expose_as_tool: ClassVar[bool] = True
    _body: ClassVar[str] = "content_card"

    model: DiTModel = Field(
        default=DiTModel.SM_MUSIC,
        description=(
            "DiT model: 'sm-music' (50M, fast music), 'sm-sfx' (50M, sound "
            "effects) or 'medium' (1.4B, higher fidelity). The matching SAME "
            "decoder is selected automatically."
        ),
    )
    prompt: str = Field(
        default="A driving lofi house loop with warm vinyl crackle",
        description="Text prompt describing the audio to generate.",
    )
    seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Output length in seconds (max 300; trimmed exactly).",
    )
    steps: int = Field(
        default=8,
        ge=1,
        le=16,
        description=(
            "Ping-pong sampling steps. The distilled model targets 8 (the sweet "
            "spot); 1-4 is faster but lower quality."
        ),
    )
    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=15.0,
        description=(
            "Classifier-free guidance scale. 1.0 = off (fastest). >1.0 pushes "
            "toward the prompt; costs ~2x per step."
        ),
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt for CFG's unconditional branch (only used when cfg != 1.0).",
    )
    seed: int = Field(
        default=0,
        description="Seed for deterministic generation. Leave as 0 for random.",
    )

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseStableAudio3

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    def required_inputs(self):
        return ["prompt"]

    @classmethod
    def get_input_fields(cls):
        return ["prompt"]

    @classmethod
    def get_inline_fields(cls):
        return ["model", "prompt"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt", "seconds", "steps", "cfg", "negative_prompt", "seed"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=SA3_REPO_ID, allow_patterns=["MLX/*"]),
        ]

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Stable Audio 3 requires macOS (Apple Silicon / MLX).")

    def _needs_encoder(self) -> bool:
        """Whether this node consumes an input clip (needs the SAME encoder)."""
        return False

    async def preload_model(self, context: ProcessingContext) -> None:
        """Ensure the weight files for the selected model are downloaded."""
        self._ensure_supported_platform()
        from nodetool.mlx.stable_audio_3 import weight_keys, weights

        keys = weight_keys(self.model.value, with_encoder=self._needs_encoder())

        def _download() -> None:
            for key in keys:
                weights.ensure_local(key)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _download)

    def _ensure_seed(self) -> None:
        if self.seed == 0:
            self.seed = random.randint(1, 2**31 - 1)

    async def _load_init_audio(
        self, context: ProcessingContext, audio: AudioRef
    ) -> "np.ndarray":
        """Decode an AudioRef to a (2, T) float32 array at 44.1 kHz stereo."""
        import numpy as np

        segment = await context.audio_to_audio_segment(audio)
        segment = (
            segment.set_frame_rate(SAMPLE_RATE).set_channels(2).set_sample_width(2)
        )
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        return samples.reshape(-1, 2).T / 32768.0

    async def _generate(
        self,
        context: ProcessingContext,
        *,
        init_audio: "np.ndarray | None" = None,
        init_noise_level: float = 1.0,
        inpaint_range: tuple[float, float] | None = None,
    ) -> AudioRef:
        self._ensure_supported_platform()
        if not self.prompt.strip() and self.cfg == 1.0:
            # An empty prompt is valid (unconditional), but warn-free generation
            # of pure silence is rarely intended — still allow it.
            log.debug("Stable Audio 3 running with an empty prompt (unconditional).")
        self._ensure_seed()

        node_id = self.id

        def on_step(i: int, total: int) -> None:
            context.post_message(NodeProgress(node_id=node_id, progress=i, total=total))

        negative = self.negative_prompt.strip() or None

        def _run() -> tuple["np.ndarray", int]:
            from nodetool.mlx.stable_audio_3 import generate

            return generate(
                prompt=self.prompt,
                dit=self.model.value,
                seconds=float(self.seconds),
                steps=int(self.steps),
                seed=int(self.seed),
                cfg=float(self.cfg),
                negative_prompt=negative,
                init_audio=init_audio,
                init_noise_level=float(init_noise_level),
                inpaint_range=inpaint_range,
                on_step=on_step,
                node_id=node_id,
            )

        loop = asyncio.get_running_loop()
        audio_np, sample_rate = await loop.run_in_executor(None, _run)
        wav_bytes = _audio_to_wav_bytes(audio_np, sample_rate)
        return await context.audio_from_bytes(wav_bytes)


def _audio_to_wav_bytes(audio: "np.ndarray", sample_rate: int) -> bytes:
    """Encode a (channels, T) float32 array in [-1, 1] to 16-bit PCM WAV bytes."""
    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).T  # (T, channels) interleaved
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(clipped.shape[0])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue()


class StableAudio3(BaseStableAudio3):
    """
    Generate music and sound effects from text with Stable Audio 3 on Apple Silicon.
    mlx, audio, music, sound-effects, text-to-audio, apple-silicon, stable-audio

    Use cases:
    - Generate music loops and beds locally without cloud APIs
    - Create sound effects from a text description
    - Prototype audio ideas on-device with low latency
    """

    @classmethod
    def get_title(cls):
        return "Stable Audio 3"

    async def process(self, context: ProcessingContext) -> AudioRef:
        return await self._generate(context)


class StableAudio3AudioToAudio(BaseStableAudio3):
    """
    Transform an input clip into a variation guided by a text prompt (Stable Audio 3).
    mlx, audio, audio-to-audio, style-transfer, variation, apple-silicon, stable-audio

    Use cases:
    - Create prompt-guided variations of an existing loop
    - Restyle a melody or texture while keeping its broad structure
    - Explore alternatives seeded from a reference recording
    """

    audio: AudioRef = Field(
        description="Input audio used as the starting point (resampled to 44.1 kHz stereo).",
    )
    init_noise_level: float = Field(
        default=0.7,
        ge=0.01,
        le=1.0,
        description=(
            "How much noise to add to the input (σmax). ~0.4-0.8 keeps structure "
            "while varying; 1.0 ignores the input entirely."
        ),
    )

    @classmethod
    def get_title(cls):
        return "Stable Audio 3 Audio to Audio"

    def _needs_encoder(self) -> bool:
        return True

    def required_inputs(self):
        return ["prompt", "audio"]

    @classmethod
    def get_input_fields(cls):
        return ["prompt", "audio"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "audio",
            "init_noise_level",
            "seconds",
            "steps",
            "cfg",
            "seed",
        ]

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self.audio.is_empty():
            raise ValueError("An input audio clip is required for audio-to-audio.")
        init_audio = await self._load_init_audio(context, self.audio)
        return await self._generate(
            context, init_audio=init_audio, init_noise_level=self.init_noise_level
        )


class StableAudio3Inpaint(BaseStableAudio3):
    """
    Regenerate a time range inside an audio clip from a text prompt (Stable Audio 3).
    mlx, audio, inpainting, edit, repair, apple-silicon, stable-audio

    Use cases:
    - Replace a section of a track while preserving the rest exactly
    - Fix or restyle a few seconds of a recording
    - Fill a gap with prompt-guided content that matches the surroundings
    """

    audio: AudioRef = Field(
        description="Input audio to edit (resampled to 44.1 kHz stereo).",
    )
    inpaint_start: float = Field(
        default=2.0,
        ge=0.0,
        le=300.0,
        description="Start of the range to regenerate, in seconds.",
    )
    inpaint_end: float = Field(
        default=5.0,
        ge=0.0,
        le=300.0,
        description="End of the range to regenerate, in seconds (must be > start and <= seconds).",
    )

    @classmethod
    def get_title(cls):
        return "Stable Audio 3 Inpaint"

    def _needs_encoder(self) -> bool:
        return True

    def required_inputs(self):
        return ["prompt", "audio"]

    @classmethod
    def get_input_fields(cls):
        return ["prompt", "audio"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "audio",
            "inpaint_start",
            "inpaint_end",
            "seconds",
            "steps",
            "cfg",
            "seed",
        ]

    async def process(self, context: ProcessingContext) -> AudioRef:
        if self.audio.is_empty():
            raise ValueError("An input audio clip is required for inpainting.")
        if not (0 <= self.inpaint_start < self.inpaint_end <= self.seconds):
            raise ValueError(
                f"Invalid inpaint range {self.inpaint_start}-{self.inpaint_end}s "
                f"(need 0 <= start < end <= seconds={self.seconds})."
            )
        init_audio = await self._load_init_audio(context, self.audio)
        return await self._generate(
            context,
            init_audio=init_audio,
            inpaint_range=(self.inpaint_start, self.inpaint_end),
        )
