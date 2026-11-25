"""LTX-2.5 and LTX-2.3 video with sound on Apple Silicon, through ltx-2-mlx.

The engine is ``ltx-pipelines-mlx`` from https://github.com/dgrauet/ltx-2-mlx.
It is not on PyPI, so it is imported lazily: this module loads, and its nodes
are listed, on any platform and without the engine. Every node runs the
distilled two-stage flow (8 + 3 steps) and returns an MP4 with its soundtrack.

Weights are dgrauet's pre-converted LTX-2.5 packs. A pack holds the dev and
the distilled transformer, a distilled LoRA and the Gemma 4 text encoder.
These nodes load only the distilled transformer, so the recommended download
skips the other two.

The IC-LoRA nodes (motion control, colorize, ingredients, upscale) follow the
recipes Phosphene ships. Control, colorize and ingredients run on the LTX-2.3
packs, which the IC-LoRAs were trained on, and upscale on LTX-2.5. Every LoRA is
attached unfused, because fusing a delta into Q4 or Q8 weights rounds most of it
away.
"""

from __future__ import annotations

import asyncio
import math
import random
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioRef,
    BaseType,
    HFTextToVideo,
    HuggingFaceModel,
    ImageRef,
    Provider,
    VideoRef,
)
from nodetool.nodes.mlx._hf_cache import find_cached_snapshot
from nodetool.nodes.mlx._mlx_thread import MLX_EXECUTOR
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

try:
    from nodetool.metadata.types import HFLoraLTX, HFLoraLTXConfig
except ImportError:  # nodetool-core releases before the LTX LoRA types

    class HFLoraLTX(HuggingFaceModel):
        """A LoRA for the LTX-2 video models: a character or style LoRA, or an IC-LoRA."""

        type: Literal["hf.lora_ltx"] = "hf.lora_ltx"

    class HFLoraLTXConfig(BaseType):
        type: Literal["hf.lora_ltx_config"] = "hf.lora_ltx_config"
        lora: HFLoraLTX = Field(
            default=HFLoraLTX(),
            description="The LoRA file: a repo and the .safetensors path in it.",
        )
        strength: float = Field(
            default=1.0, ge=0.0, le=3.0, description="LoRA strength"
        )


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
# The same for an LTX-2.3 pack, plus the retired 1.1 transformer and the
# upscalers the distilled flow does not use.
LTX23_MLX_IGNORE_PATTERNS = [
    "transformer-dev.safetensors",
    "transformer-distilled-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-*.safetensors",
    "spatial_upscaler_x1_5_*.safetensors",
    "temporal_upscaler_*.safetensors",
]
# LTX-2.3 packs do not bundle a text encoder.
LTX23_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"

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
LTX23_MLX_PACKS = {
    LTXQuantization.Q4: "dgrauet/ltx-2.3-mlx-q4",
    LTXQuantization.Q8: "dgrauet/ltx-2.3-mlx-q8",
}


class ICLoraFile:
    """An IC-LoRA file on the Hugging Face Hub, with the strength it was tuned for."""

    def __init__(self, repo_id: str, path: str, strength: float) -> None:
        self.repo_id = repo_id
        self.path = path
        self.strength = strength

    def model(self) -> HFLoraLTX:
        return HFLoraLTX(repo_id=self.repo_id, path=self.path)


IC_LORA_MOTION = ICLoraFile(
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    1.0,
)
IC_LORA_COLORIZE = ICLoraFile(
    "DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer",
    "LTX-2.3-22b-IC-LoRA-Colorizer-0.9.safetensors",
    1.0,
)
# The repo holds about 700 GB of files: download only this one.
IC_LORA_INGREDIENTS = ICLoraFile(
    "DeepBeepMeep/LTX-2",
    "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
    1.4,
)
# Gated: accept the license on Hugging Face and set HF_TOKEN.
IC_LORA_UPSCALE = ICLoraFile(
    "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
    1.0,
)

INGREDIENTS_SHEET_SIZE = (1536, 896)
INGREDIENTS_GUTTER = 16
INGREDIENTS_MAX_IMAGES = 8
# Longest side of the motion control and colorize canvas.
LTX_MLX_CONTROL_MAX_SIDE = 1280


def snap_frames(num_frames: int) -> int:
    """Round a frame count down onto LTX's 8n+1 grid, within 9 to 257 frames."""
    clamped = min(max(int(num_frames), LTX_MLX_MIN_FRAMES), LTX_MLX_MAX_FRAMES)
    return 1 + 8 * ((clamped - 1) // 8)


def snap_size(value: int) -> int:
    """Round a side down to a multiple of 64, as the two-stage flow needs."""
    return max(64, (int(value) // 64) * 64)


def pad_frames(num_frames: int) -> int:
    """Round a frame count up onto LTX's 8n+1 grid, from 9 frames."""
    return max(LTX_MLX_MIN_FRAMES, 1 + 8 * math.ceil((int(num_frames) - 1) / 8))


def control_canvas(width: int, height: int) -> tuple[int, int]:
    """The source size, scaled to fit 1280 pixels and floored to multiples of 64."""
    scale = min(1.0, LTX_MLX_CONTROL_MAX_SIDE / max(width, height))
    return snap_size(round(width * scale)), snap_size(round(height * scale))


def ingredients_prompt(sheet: str, action: str) -> str:
    """The prompt format the ingredients LoRA was trained on."""
    sheet, action = sheet.strip(), action.strip()
    if sheet and action:
        return f"Reference sheet: {sheet}\n\nGenerated video: {action}"
    if action:
        return f"Generated video: {action}"
    return sheet or "the subjects from the reference sheet, cinematic"


def compose_sheet(images: list[Any]) -> Any:
    """Tile PIL images on a black 1536x896 sheet, each fitted into its cell.

    One image is returned unchanged, as Phosphene does.
    """
    from PIL import Image

    if len(images) == 1:
        return images[0].convert("RGB")
    sheet_w, sheet_h = INGREDIENTS_SHEET_SIZE
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    cell_w = (sheet_w - INGREDIENTS_GUTTER * (cols + 1)) // cols
    cell_h = (sheet_h - INGREDIENTS_GUTTER * (rows + 1)) // rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        fitted = image.convert("RGB")
        scale = min(cell_w / fitted.width, cell_h / fitted.height)
        fitted = fitted.resize(
            (max(1, round(fitted.width * scale)), max(1, round(fitted.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = INGREDIENTS_GUTTER + col * (cell_w + INGREDIENTS_GUTTER)
        top = INGREDIENTS_GUTTER + row * (cell_h + INGREDIENTS_GUTTER)
        sheet.paste(
            fitted,
            (left + (cell_w - fitted.width) // 2, top + (cell_h - fitted.height) // 2),
        )
    return sheet


def _ffmpeg(*args: str) -> None:
    """Run ffmpeg, which ltx-2-mlx also needs, and raise with its log on failure."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install it with: brew install ffmpeg"
        )
    result = subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-2000:]}")


def mux_source_audio(video: str, source: str, output: str, frames: int | None) -> None:
    """Copy ``video`` to ``output`` with the audio of ``source``, cut to ``frames``."""
    if frames:
        codec = ["-frames:v", str(frames), "-c:v", "libx264", "-crf", "12"]
        codec += ["-pix_fmt", "yuv420p"]
    else:
        codec = ["-c:v", "copy"]
    _ffmpeg(
        "-i", video, "-i", source, "-map", "0:v:0", "-map", "1:a:0?",
        *codec, "-c:a", "aac", "-shortest", output,
    )  # fmt: skip


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
    loras: list[HFLoraLTXConfig] = Field(
        default=[],
        description="Character or style LoRAs trained for the LTX version this node runs.",
    )

    # LTX packs this node loads, by quantization, and the files to skip.
    _packs: ClassVar[dict[LTXQuantization, str]] = LTX_MLX_PACKS
    _pack_ignore: ClassVar[list[str]] = LTX_MLX_IGNORE_PATTERNS

    @classmethod
    def is_visible(cls) -> bool:
        return not cls.__name__.startswith("_")

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        models: list[HuggingFaceModel] = [
            HFTextToVideo(repo_id=repo_id, ignore_patterns=list(cls._pack_ignore))
            for repo_id in cls._packs.values()
        ]
        if cls._packs is LTX23_MLX_PACKS:
            models.append(HFTextToVideo(repo_id=LTX23_GEMMA_REPO))
        return models

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
        repo_id = self._packs[LTXQuantization(self.quantization)]
        snapshot = find_cached_snapshot(repo_id, LTX_MLX_MARKER)
        if snapshot is None:
            raise RuntimeError(
                f"{repo_id} is not downloaded. Download it from the Models panel first."
            )
        return str(snapshot)

    async def _lora_files(
        self, context: ProcessingContext, task: list[ICLoraFile] | None = None
    ) -> list[tuple[str, float]]:
        """Download ``task`` and the node's LoRAs; return ``(path, strength)`` pairs."""
        wanted = [(lora.model(), lora.strength) for lora in task or []]
        for config in self.loras:
            if not config.lora.is_set():
                continue
            if not config.lora.path:
                raise ValueError(
                    f"LoRA {config.lora.repo_id} needs the path of its .safetensors file."
                )
            wanted.append((config.lora, float(config.strength)))
        if not wanted:
            return []
        token = await context.get_secret("HF_TOKEN")

        def download() -> list[tuple[str, float]]:
            from huggingface_hub import hf_hub_download

            return [
                (
                    hf_hub_download(lora.repo_id, str(lora.path), token=token),
                    strength,
                )
                for lora, strength in wanted
            ]

        # Off the event loop, on the one thread the MLX nodes share.
        return await asyncio.get_running_loop().run_in_executor(MLX_EXECUTOR, download)

    def _seed(self) -> int:
        if self.seed is not None and self.seed >= 0:
            return int(self.seed)
        return random.randint(0, 2**31 - 1)

    def _pipeline(
        self,
        class_name: str,
        loras: list[tuple[str, float]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a pipeline with ``loras`` attached to its transformer when it loads."""
        import ltx_pipelines_mlx

        pipeline_class = getattr(ltx_pipelines_mlx, class_name)
        pipeline = pipeline_class(
            self._model_dir(), low_ram_streaming=bool(self.low_ram), **kwargs
        )
        # Attach LoRAs as run-time adapters: fused into Q4/Q8 weights, most of
        # a LoRA's delta rounds away. Read when the transformer loads.
        pipeline.lora_mode = "unfused"
        if loras:
            pipeline._pending_loras = list(loras)
        return pipeline

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
        loras = await self._lora_files(context)
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
                pipeline = self._pipeline("DistilledPipeline", loras)
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
        loras = await self._lora_files(context)
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
                pipeline = self._pipeline("A2VidDistilledPipeline", loras)
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
        loras = await self._lora_files(context)
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            video_path = await self._asset_file(
                context, self.video, directory, "source.mp4"
            )
            output = directory / "output.mp4"

            def run() -> None:
                from ltx_core_mlx.utils.ffmpeg import probe_video_info

                frame_rate = probe_video_info(video_path).fps
                pipeline = self._pipeline("ExtendDistilledPipeline", loras)
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


class _LTXMLXICLora(_LTXMLXBase):
    """Shared plumbing for the IC-LoRA nodes, which run ``ICLoraPipeline``."""

    _packs: ClassVar[dict[LTXQuantization, str]] = LTX23_MLX_PACKS
    _pack_ignore: ClassVar[list[str]] = LTX23_MLX_IGNORE_PATTERNS

    quantization: LTXQuantization = Field(
        default=LTXQuantization.Q4,
        description="LTX-2.3 pack: q4 (about 12 GB, what the IC-LoRAs were tuned on) or q8 (about 26 GB).",
    )

    def _ic_pipeline(self, loras: list[tuple[str, float]]) -> Any:
        kwargs: dict[str, Any] = {"lora_paths": loras}
        if self._packs is LTX23_MLX_PACKS:
            gemma = find_cached_snapshot(LTX23_GEMMA_REPO)
            kwargs["gemma_model_id"] = str(gemma) if gemma else LTX23_GEMMA_REPO
        return self._pipeline("ICLoraPipeline", **kwargs)

    @staticmethod
    def _probe(path: str) -> Any:
        from ltx_core_mlx.utils.ffmpeg import probe_video_info

        return probe_video_info(path)


class LTXControlMode(str, Enum):
    """What the IC-LoRA takes from the source video."""

    MOTION = "motion"
    COLORIZE = "colorize"


class LTXControl(_LTXMLXICLora):
    """
    Renders a new video that follows the motion and layout of a source video, or colorizes a black-and-white video, with LTX-2.3 IC-LoRAs on Apple Silicon.
    video, video-to-video, motion control, colorize, ic-lora, ltx, mlx, apple silicon

    Use cases:
    - Restyle a clip while keeping its camera move and choreography
    - Drive a new character with the motion of a reference performance
    - Add color to old black-and-white footage
    """

    mode: LTXControlMode = Field(
        default=LTXControlMode.MOTION,
        description="motion: a new video that follows the source (Union Control IC-LoRA). colorize: the source in color, with its own sound.",
    )
    video: VideoRef = Field(default=VideoRef(), description="The source video.")
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional first frame for motion control, such as the new character.",
    )
    num_frames: int = Field(
        default=121,
        ge=LTX_MLX_MIN_FRAMES,
        le=LTX_MLX_MAX_FRAMES,
        description="Frames to generate, rounded down to 8n+1 and to the source length.",
    )
    reference_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How closely the output follows the source. Lower is looser.",
    )
    frame_rate: float = Field(
        default=24.0, ge=1.0, le=60.0, description="Frames per second."
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.3 Motion Control and Colorize (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["mode", "prompt", "video", "image", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.video.is_set():
            raise ValueError("A source video is required.")
        colorize = LTXControlMode(self.mode) == LTXControlMode.COLORIZE
        if not colorize and not self.prompt.strip():
            raise ValueError("Motion control needs a prompt for the new video.")
        self._require_engine()
        seed = self._seed()
        task = IC_LORA_COLORIZE if colorize else IC_LORA_MOTION
        loras = await self._lora_files(context, [task])
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            source = await self._asset_file(
                context, self.video, directory, "source.mp4"
            )
            image_path = (
                await self._image_file(context, self.image, directory, "image")
                if self.image.is_set() and not colorize
                else None
            )
            generated = directory / "generated.mp4"
            output = directory / "output.mp4" if colorize else generated
            prompt = self.prompt.strip() or "the same scene in natural color"

            def run() -> None:
                info = self._probe(source)
                width, height = control_canvas(info.width, info.height)
                num_frames = snap_frames(min(self.num_frames, info.num_frames))
                pipeline = self._ic_pipeline(loras)
                pipeline.generate_and_save(
                    prompt=prompt,
                    output_path=str(generated),
                    video_conditioning=[(source, float(self.reference_strength))],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(self.frame_rate),
                    seed=seed,
                    stage1_steps=8,
                    stage2_steps=3,
                    images=(
                        [self._conditioning(image_path, 0, 1.0)] if image_path else None
                    ),
                )
                if colorize:
                    mux_source_audio(str(generated), source, str(output), None)

            return await self._render(context, run, output)


class LTXIngredients(_LTXMLXICLora):
    """
    Generates a video with sound that features the people, objects and places from 1 to 8 reference pictures, with the LTX-2.3 ingredients IC-LoRA on Apple Silicon.
    video, references, ingredients, character, product, ic-lora, ltx, mlx, apple silicon

    Use cases:
    - Put a character and a product into one scene
    - Keep a cast consistent across shots
    """

    images: list[ImageRef] = Field(
        default=[],
        description=f"1 to {INGREDIENTS_MAX_IMAGES} reference pictures. They are tiled onto one reference sheet.",
    )
    sheet_description: str = Field(
        default="",
        description="What the reference pictures show, such as 'a red-haired woman, a vintage motorbike'.",
    )
    prompt: str = Field(
        default="",
        description="What happens in the clip, including the sound.",
    )
    num_frames: int = Field(
        default=121,
        ge=LTX_MLX_MIN_FRAMES,
        le=LTX_MLX_MAX_FRAMES,
        description="Frames to generate, rounded down to 8n+1.",
    )
    width: int = Field(
        default=1536,
        ge=256,
        le=1536,
        description="Width in pixels, rounded down to a multiple of 64.",
    )
    height: int = Field(
        default=896,
        ge=256,
        le=1536,
        description="Height in pixels, rounded down to a multiple of 64.",
    )
    quick: bool = Field(
        default=False,
        description="Skip the refine stage: faster, at half the size, with a dreamier look.",
    )
    frame_rate: float = Field(
        default=24.0, ge=1.0, le=60.0, description="Frames per second."
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.3 Ingredients (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["images", "sheet_description", "prompt", "num_frames", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["images", "prompt"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        images = [image for image in self.images if image.is_set()]
        if not 1 <= len(images) <= INGREDIENTS_MAX_IMAGES:
            raise ValueError(
                f"Ingredients need 1 to {INGREDIENTS_MAX_IMAGES} pictures, got {len(images)}."
            )
        self._require_engine()
        seed = self._seed()
        num_frames = snap_frames(self.num_frames)
        width, height = snap_size(self.width), snap_size(self.height)
        # The IC encoder reads the reference at half the output size.
        cond_w, cond_h = max(32, width // 2 // 32 * 32), max(32, height // 2 // 32 * 32)
        loras = await self._lora_files(context, [IC_LORA_INGREDIENTS])
        pictures = [await context.image_to_pil(image) for image in images]
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            sheet = directory / "sheet.png"
            compose_sheet(pictures).save(sheet)
            reference = directory / "reference.mp4"
            output = directory / "output.mp4"
            fps = f"{float(self.frame_rate):g}"
            prompt = ingredients_prompt(self.sheet_description, self.prompt)

            def run() -> None:
                _ffmpeg(
                    "-loop", "1", "-framerate", fps, "-i", str(sheet),
                    "-frames:v", str(num_frames), "-vf", f"scale={cond_w}:{cond_h}",
                    "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0", str(reference),
                )  # fmt: skip
                pipeline = self._ic_pipeline(loras)
                pipeline.generate_and_save(
                    prompt=prompt,
                    output_path=str(output),
                    video_conditioning=[(str(reference), 1.0)],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(self.frame_rate),
                    seed=seed,
                    stage1_steps=8,
                    stage2_steps=4,
                    skip_stage_2=bool(self.quick),
                )

            return await self._render(context, run, output)


class LTXUpscale(_LTXMLXICLora):
    """
    Doubles the width and height of a video with the LTX-2.5 pixel upscaler IC-LoRA on Apple Silicon. Keeps the source sound and length.
    video, upscale, super resolution, x2, ic-lora, ltx, mlx, apple silicon

    Use cases:
    - Bring a 768x448 LTX draft to 1536x896
    - Sharpen a low-resolution clip before delivery
    """

    _packs: ClassVar[dict[LTXQuantization, str]] = LTX_MLX_PACKS
    _pack_ignore: ClassVar[list[str]] = LTX_MLX_IGNORE_PATTERNS

    quantization: LTXQuantization = Field(
        default=LTXQuantization.Q8,
        description="LTX-2.5 pack: q8 (about 46 GB, recommended) or q4 (about 27 GB).",
    )
    video: VideoRef = Field(
        default=VideoRef(),
        description="The video to upscale. Its sides should be multiples of 32 pixels.",
    )
    keep: float = Field(
        default=1.0,
        ge=0.3,
        le=1.0,
        description="How much of the source to keep. Lower lets the model invent more detail.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Upscale x2 (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["video", "keep", "quantization"]

    def required_inputs(self) -> list[str]:
        return ["video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.video.is_set():
            raise ValueError("A video is required.")
        self._require_engine()
        seed = self._seed()
        loras = await self._lora_files(context, [IC_LORA_UPSCALE])
        keep = float(self.keep)
        with tempfile.TemporaryDirectory(prefix="ltx_mlx_") as tmp:
            directory = Path(tmp)
            source = await self._asset_file(
                context, self.video, directory, "source.mp4"
            )
            padded = directory / "padded.mp4"
            generated = directory / "generated.mp4"
            output = directory / "output.mp4"
            prompt = self.prompt.strip() or "high detail, sharp, clean"

            def run() -> None:
                info = self._probe(source)
                if info.num_frames > LTX_MLX_MAX_FRAMES:
                    raise ValueError(
                        f"The video has {info.num_frames} frames. Upscale takes up to {LTX_MLX_MAX_FRAMES}."
                    )
                num_frames = pad_frames(info.num_frames)
                # LTX needs 8n+1 frames: hold the last frame, then cut it again.
                _ffmpeg(
                    "-i", source,
                    "-vf", f"tpad=stop_mode=clone:stop={num_frames - info.num_frames}",
                    "-an", "-c:v", "libx264", "-crf", "0", str(padded),
                )  # fmt: skip
                pipeline = self._ic_pipeline(loras)
                pipeline.generate_and_save(
                    prompt=prompt,
                    output_path=str(generated),
                    video_conditioning=[(str(padded), keep)],
                    height=2 * (info.height // 32 * 32),
                    width=2 * (info.width // 32 * 32),
                    num_frames=num_frames,
                    frame_rate=float(info.fps),
                    seed=seed,
                    stage1_steps=4,
                    conditioning_attention_strength=keep,
                    single_stage=True,
                )
                mux_source_audio(str(generated), source, str(output), info.num_frames)

            return await self._render(context, run, output)
