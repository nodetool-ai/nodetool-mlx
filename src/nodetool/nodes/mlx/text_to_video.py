"""LTX-2.5 video with sound on Apple Silicon, through ltx-2-mlx.

The engine is ``ltx-pipelines-mlx`` from https://github.com/dgrauet/ltx-2-mlx.
It is not on PyPI, so it is imported lazily: this module loads, and its nodes
are listed, on any platform and without the engine. Every node runs the
distilled two-stage flow (8 + 3 steps) and returns an MP4 with its soundtrack.

Weights are dgrauet's pre-converted LTX-2.5 packs. A pack holds the dev and
the distilled transformer, a distilled LoRA and the Gemma 4 text encoder.
These nodes load only the distilled transformer, so the recommended download
skips the other two.
"""

from __future__ import annotations

import asyncio
import random
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioRef,
    HFTextToVideo,
    ImageRef,
    Provider,
    VideoRef,
)
from nodetool.nodes.mlx._hf_cache import find_cached_snapshot
from nodetool.nodes.mlx._mlx_thread import MLX_EXECUTOR
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

LTX_MLX_VERSION = "v0.18.0"
LTX_MLX_REPO = "https://github.com/dgrauet/ltx-2-mlx"
LTX_MLX_INSTALL = (
    f'uv pip install "ltx-core-mlx @ git+{LTX_MLX_REPO}@{LTX_MLX_VERSION}'
    '#subdirectory=packages/ltx-core-mlx" '
    f'"ltx-pipelines-mlx @ git+{LTX_MLX_REPO}@{LTX_MLX_VERSION}'
    '#subdirectory=packages/ltx-pipelines-mlx"'
)

# The file every distilled pipeline loads; its presence marks a usable pack.
LTX_MLX_MARKER = "transformer-distilled.safetensors"
# Files the distilled flow never reads: the dev transformer and the LoRA that
# distills it. Together they are 20 GB (q4) to 29 GB (q8).
LTX_MLX_IGNORE_PATTERNS = [
    "transformer-dev.safetensors",
    "ltx-2.5-22b-distilled-lora-*.safetensors",
]

LTX_MLX_MIN_FRAMES = 9
LTX_MLX_MAX_FRAMES = 257
LTX_MLX_MAX_KEYFRAMES = 8
# The extend pipeline carries the last 25 pixel frames of the source.
LTX_MLX_EXTEND_CARRY_FRAMES = 25


class LTXQuantization(str, Enum):
    """Weight precision of the LTX-2.5 pack."""

    Q4 = "q4"
    Q8 = "q8"


LTX_MLX_PACKS = {
    LTXQuantization.Q4: "dgrauet/ltx-2.5-mlx-q4",
    LTXQuantization.Q8: "dgrauet/ltx-2.5-mlx-q8",
}


def snap_frames(num_frames: int) -> int:
    """Round a frame count down onto LTX's 8n+1 grid, within 9 to 257 frames."""
    clamped = min(max(int(num_frames), LTX_MLX_MIN_FRAMES), LTX_MLX_MAX_FRAMES)
    return 1 + 8 * ((clamped - 1) // 8)


def snap_size(value: int) -> int:
    """Round a side down to a multiple of 64, as the two-stage flow needs."""
    return max(64, (int(value) // 64) * 64)


def spread_frames(count: int, num_frames: int) -> list[int]:
    """Pixel frames for ``count`` keyframes spread from the first frame to the last."""
    if count < 2:
        raise ValueError("Keyframes need at least two images.")
    last = num_frames - 1
    return [round(index * last / (count - 1)) for index in range(count)]


def extend_latent_frames(seconds: float, frame_rate: float) -> int:
    """Latent frames (8 pixel frames each) that cover ``seconds`` of new video."""
    return max(1, -(-round(seconds * frame_rate) // 8))


class _LTXMLXBase(BaseNode):
    """Shared plumbing for the LTX-2.5 MLX nodes."""

    _provider: ClassVar[Provider] = Provider.MLX
    _body: ClassVar[str] = "content_card"

    prompt: str = Field(
        default="",
        description="What happens in the clip, including the sound.",
    )
    quantization: LTXQuantization = Field(
        default=LTXQuantization.Q8,
        description="q8 (about 46 GB to download, recommended) or q4 (about 27 GB, lower quality).",
    )
    seed: int = Field(default=-1, description="Random seed. -1 picks one at random.")
    low_ram: bool = Field(
        default=False,
        description="Stream transformer blocks from disk. Slower, but fits a 32 GB Mac.",
    )

    @classmethod
    def is_visible(cls) -> bool:
        return not cls.__name__.startswith("_")

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToVideo]:
        return [
            HFTextToVideo(
                repo_id=repo_id,
                ignore_patterns=list(LTX_MLX_IGNORE_PATTERNS),
            )
            for repo_id in LTX_MLX_PACKS.values()
        ]

    # -- environment ---------------------------------------------------------

    @staticmethod
    def _require_engine() -> None:
        if sys.platform != "darwin":
            raise RuntimeError("LTX on MLX needs a Mac with Apple Silicon.")
        try:
            import ltx_pipelines_mlx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The LTX MLX engine (ltx-2-mlx) is not installed. It is not on PyPI. "
                f"Install it into the NodeTool Python environment with: {LTX_MLX_INSTALL}"
            ) from exc

    def _model_dir(self) -> str:
        repo_id = LTX_MLX_PACKS[LTXQuantization(self.quantization)]
        snapshot = find_cached_snapshot(repo_id, LTX_MLX_MARKER)
        if snapshot is None:
            raise RuntimeError(
                f"{repo_id} is not downloaded. Download it from the Models panel first."
            )
        return str(snapshot)

    def _seed(self) -> int:
        if self.seed is not None and self.seed >= 0:
            return int(self.seed)
        return random.randint(0, 2**31 - 1)

    def _pipeline(self, class_name: str) -> Any:
        import ltx_pipelines_mlx

        pipeline_class = getattr(ltx_pipelines_mlx, class_name)
        return pipeline_class(self._model_dir(), low_ram_streaming=bool(self.low_ram))

    @staticmethod
    async def _image_file(
        context: ProcessingContext, image: ImageRef, directory: Path, name: str
    ) -> str:
        path = directory / f"{name}.png"
        (await context.image_to_pil(image)).convert("RGB").save(path)
        return str(path)

    @staticmethod
    async def _asset_file(
        context: ProcessingContext, asset: Any, directory: Path, name: str
    ) -> str:
        path = directory / name
        path.write_bytes(await context.asset_to_bytes(asset))
        return str(path)

    @staticmethod
    def _conditioning(path: str, frame_idx: int, strength: float) -> Any:
        from ltx_pipelines_mlx.utils.types import ImageConditioningInput

        return ImageConditioningInput.at(path, frame_idx, strength)

    async def _render(
        self, context: ProcessingContext, run: Any, output: Path
    ) -> VideoRef:
        """Run ``run()`` on the MLX thread and return the MP4 it wrote to ``output``."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(MLX_EXECUTOR, run)
        if not output.exists():
            raise RuntimeError("ltx-2-mlx did not write an output video.")
        return await context.video_from_bytes(output.read_bytes())


class _LTXMLXGenerate(_LTXMLXBase):
    """Fields of a clip generated from scratch."""

    num_frames: int = Field(
        default=121,
        ge=LTX_MLX_MIN_FRAMES,
        le=LTX_MLX_MAX_FRAMES,
        description="Frames to generate, rounded down to 8n+1. 121 frames at 24 fps is 5 seconds.",
    )
    frame_rate: float = Field(
        default=24.0, ge=1.0, le=60.0, description="Frames per second."
    )
    width: int = Field(
        default=768,
        ge=256,
        le=1536,
        description="Width in pixels, rounded down to a multiple of 64.",
    )
    height: int = Field(
        default=448,
        ge=256,
        le=1536,
        description="Height in pixels, rounded down to a multiple of 64.",
    )

    async def _generate(
        self,
        context: ProcessingContext,
        images: list[tuple[ImageRef, int, float]],
    ) -> VideoRef:
        """Generate with each ``(image, pixel_frame, strength)`` as a conditioning anchor."""
        self._require_engine()
        num_frames = snap_frames(self.num_frames)
        seed = self._seed()
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            anchors = [
                self._conditioning(
                    await self._image_file(
                        context, image, directory, f"anchor_{index}"
                    ),
                    frame,
                    strength,
                )
                for index, (image, frame, strength) in enumerate(images)
            ]
            output = directory / "output.mp4"

            def run() -> None:
                pipeline = self._pipeline("DistilledPipeline")
                pipeline.generate_and_save(
                    prompt=self.prompt,
                    output_path=str(output),
                    height=snap_size(self.height),
                    width=snap_size(self.width),
                    num_frames=num_frames,
                    frame_rate=float(self.frame_rate),
                    seed=seed,
                    images=anchors or None,
                )

            return await self._render(context, run, output)


class LTXVideo(_LTXMLXGenerate):
    """
    Generates a video with sound from a prompt, optionally starting and ending on given images, with LTX-2.5 on Apple Silicon.
    video, text-to-video, image-to-video, ltx, mlx, apple silicon, audio

    Use cases:
    - Generate a short clip and its soundtrack from a description
    - Animate a still image, or move from one image to another
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional first frame. Leave empty for text to video.",
    )
    last_image: ImageRef = Field(default=ImageRef(), description="Optional last frame.")
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How closely the clip holds the given frames.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Video (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image", "last_image", "num_frames", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["prompt"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.prompt.strip():
            raise ValueError("A prompt is required.")
        anchors: list[tuple[ImageRef, int, float]] = []
        if self.image.is_set():
            anchors.append((self.image, 0, float(self.strength)))
        if self.last_image.is_set():
            anchors.append((self.last_image, -1, float(self.strength)))
        return await self._generate(context, anchors)


class LTXKeyframes(_LTXMLXGenerate):
    """
    Generates a video with sound that passes through 2 to 8 images, spread evenly from the first frame to the last, with LTX-2.5 on Apple Silicon.
    video, keyframes, interpolation, ltx, mlx, apple silicon, audio

    Use cases:
    - Turn a storyboard into one continuous shot
    - Interpolate between several stills
    """

    images: list[ImageRef] = Field(
        default=[],
        description="2 to 8 keyframes, in order. The first is frame one, the last is the final frame.",
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How closely the clip holds each keyframe.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Keyframes (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "images", "num_frames", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "images"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        images = [image for image in self.images if image.is_set()]
        if not 2 <= len(images) <= LTX_MLX_MAX_KEYFRAMES:
            raise ValueError(
                f"Keyframes need 2 to {LTX_MLX_MAX_KEYFRAMES} images, got {len(images)}."
            )
        frames = spread_frames(len(images), snap_frames(self.num_frames))
        if len(set(frames)) != len(frames):
            raise ValueError(
                "Too many keyframes for this clip length. Add frames or remove images."
            )
        strength = float(self.strength)
        return await self._generate(
            context, [(image, frame, strength) for image, frame in zip(images, frames)]
        )


class LTXAudioToVideo(_LTXMLXGenerate):
    """
    Generates a video for a given soundtrack, such as speech or music, with LTX-2.5 on Apple Silicon. The output keeps the input audio.
    video, audio-to-video, lip sync, music video, ltx, mlx, apple silicon

    Use cases:
    - Make a talking or singing character from a voice track
    - Cut visuals to a piece of music
    """

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The soundtrack. With music, isolated vocals sync better.",
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional first frame, such as the speaker."
    )
    num_frames: int = Field(
        default=0,
        ge=0,
        le=LTX_MLX_MAX_FRAMES,
        description="Frames to generate, rounded down to 8n+1. 0 covers the audio, up to 257 frames.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Audio to Video (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "audio", "image", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "audio"]

    async def _audio_frames(self, context: ProcessingContext) -> int:
        if self.num_frames:
            return snap_frames(self.num_frames)
        segment = await context.audio_to_audio_segment(self.audio)
        frames = int(len(segment) / 1000.0 * float(self.frame_rate))
        if frames < LTX_MLX_MIN_FRAMES:
            raise ValueError("The audio is shorter than the shortest clip LTX makes.")
        return snap_frames(frames)

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.audio.is_set():
            raise ValueError("An audio clip is required.")
        self._require_engine()
        num_frames = await self._audio_frames(context)
        seed = self._seed()
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            audio_path = await self._asset_file(context, self.audio, directory, "audio")
            image_path = (
                await self._image_file(context, self.image, directory, "image")
                if self.image.is_set()
                else None
            )
            output = directory / "output.mp4"

            def run() -> None:
                pipeline = self._pipeline("A2VidDistilledPipeline")
                pipeline.generate_and_save(
                    prompt=self.prompt,
                    output_path=str(output),
                    audio_path=audio_path,
                    height=snap_size(self.height),
                    width=snap_size(self.width),
                    num_frames=num_frames,
                    frame_rate=float(self.frame_rate),
                    seed=seed,
                    image=image_path,
                )

            return await self._render(context, run, output)


class LTXExtend(_LTXMLXBase):
    """
    Continues a video and its sound with LTX-2.5 on Apple Silicon. Returns the whole clip with the new part appended.
    video, extend, continue, ltx, mlx, apple silicon, audio

    Use cases:
    - Make a generated shot longer
    - Continue a scene in a new direction with a new prompt
    """

    video: VideoRef = Field(
        default=VideoRef(),
        description="Clip to continue. Both sides must be multiples of 64 pixels, as LTX output is.",
    )
    seconds: float = Field(
        default=4.0, ge=0.5, le=10.0, description="Seconds of video to add."
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Extend (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "video", "seconds", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.video.is_set():
            raise ValueError("A video is required.")
        self._require_engine()
        seed = self._seed()
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            video_path = await self._asset_file(
                context, self.video, directory, "source.mp4"
            )
            output = directory / "output.mp4"

            def run() -> None:
                from ltx_core_mlx.utils.ffmpeg import probe_video_info

                frame_rate = probe_video_info(video_path).fps
                pipeline = self._pipeline("ExtendDistilledPipeline")
                video_latent, audio_latent = pipeline.extend_from_video(
                    prompt=self.prompt,
                    video_path=video_path,
                    extend_frames=extend_latent_frames(float(self.seconds), frame_rate),
                    seed=seed,
                )
                # As the ltx-2-mlx CLI does after extend: _decode_and_save_video
                # frees the DiT before decoding.
                pipeline._load_decoders()
                pipeline._decode_and_save_video(
                    video_latent,
                    audio_latent,
                    str(output),
                    frame_rate=pipeline.source_frame_rate,
                )

            return await self._render(context, run, output)
