"""LTX-2.5 MLX nodes, run against a stub of the ltx-2-mlx engine."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pydub import AudioSegment

import nodetool.nodes.mlx.text_to_video as ltx
from nodetool.metadata.types import AudioRef, ImageRef, VideoRef

VIDEO = VideoRef(uri="memory://out")


@dataclass
class Anchor:
    path: str
    frame_idx: int
    strength: float

    @classmethod
    def at(cls, path, frame_idx, strength):
        return cls(path, frame_idx, strength)


class FakePipeline:
    instances: list["FakePipeline"] = []

    def __init__(self, model_dir, low_ram_streaming=False, **kwargs):
        self.model_dir = model_dir
        self.low_ram_streaming = low_ram_streaming
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict]] = []
        self.source_frame_rate = None
        FakePipeline.instances.append(self)

    def generate_and_save(self, **kwargs):
        self.calls.append(("generate_and_save", kwargs))
        self.anchor_files = [Path(a.path).exists() for a in kwargs.get("images") or []]
        Path(kwargs["output_path"]).write_bytes(b"mp4")
        return kwargs["output_path"]

    def extend_from_video(self, **kwargs):
        self.calls.append(("extend_from_video", kwargs))
        self.source_frame_rate = 25.0
        return "video-latent", "audio-latent"

    def _load_decoders(self):
        self.calls.append(("_load_decoders", {}))

    def _decode_and_save_video(self, video, audio, output_path, *, frame_rate):
        self.calls.append(
            ("decode", {"latents": (video, audio), "frame_rate": frame_rate})
        )
        Path(output_path).write_bytes(b"mp4")


@pytest.fixture
def engine(monkeypatch, tmp_path):
    FakePipeline.instances.clear()
    package = types.ModuleType("ltx_pipelines_mlx")
    for name in (
        "DistilledPipeline",
        "A2VidDistilledPipeline",
        "ExtendDistilledPipeline",
        "ICLoraPipeline",
    ):
        setattr(package, name, type(name, (FakePipeline,), {}))
    utils = types.ModuleType("ltx_pipelines_mlx.utils")
    utils_types = types.ModuleType("ltx_pipelines_mlx.utils.types")
    utils_types.ImageConditioningInput = Anchor
    ffmpeg = types.ModuleType("ltx_core_mlx.utils.ffmpeg")
    ffmpeg.probe_video_info = lambda path: types.SimpleNamespace(
        fps=25.0, width=1920, height=1080, num_frames=100
    )
    for name, module in {
        "ltx_pipelines_mlx": package,
        "ltx_pipelines_mlx.utils": utils,
        "ltx_pipelines_mlx.utils.types": utils_types,
        "ltx_core_mlx": types.ModuleType("ltx_core_mlx"),
        "ltx_core_mlx.utils": types.ModuleType("ltx_core_mlx.utils"),
        "ltx_core_mlx.utils.ffmpeg": ffmpeg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ltx.sys, "platform", "darwin")
    snapshots = {}

    def find(repo_id, marker="config.json"):
        if repo_id.startswith("dgrauet/"):
            assert marker == ltx.LTX_MLX_MARKER
        snapshots[repo_id] = tmp_path / repo_id
        return snapshots[repo_id]

    monkeypatch.setattr(ltx, "find_cached_snapshot", find)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, token=None: f"/hub/{repo_id}/{filename}",
    )
    return FakePipeline.instances


@pytest.fixture
def ffmpeg_calls(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def run(*args):
        calls.append(args)
        Path(args[-1]).write_bytes(b"mp4")

    monkeypatch.setattr(ltx, "_ffmpeg", run)
    return calls


def _context(audio_ms: int = 4000) -> MagicMock:
    context = MagicMock()

    async def image_to_pil(ref):
        return Image.new("RGB", (8, 8))

    async def asset_to_bytes(ref):
        return b"bytes"

    async def audio_to_audio_segment(ref):
        return AudioSegment.silent(duration=audio_ms)

    async def video_from_bytes(data):
        assert data == b"mp4"
        return VIDEO

    context.image_to_pil = image_to_pil
    context.asset_to_bytes = asset_to_bytes
    context.audio_to_audio_segment = audio_to_audio_segment

    async def get_secret(key):
        return None

    context.video_from_bytes = video_from_bytes
    context.get_secret = get_secret
    return context


def test_grid_helpers():
    assert ltx.snap_frames(100) == 97
    assert ltx.snap_frames(1) == 9
    assert ltx.snap_frames(1000) == 257
    assert ltx.snap_size(500) == 448
    assert ltx.spread_frames(3, 121) == [0, 60, 120]
    assert ltx.extend_latent_frames(4.0, 24.0) == 12
    assert ltx.extend_latent_frames(0.1, 24.0) == 1


def test_recommended_packs_skip_the_dev_transformer():
    models = ltx.LTXVideo.get_recommended_models()
    assert [m.repo_id for m in models] == [
        "dgrauet/ltx-2.5-mlx-q4",
        "dgrauet/ltx-2.5-mlx-q8",
    ]
    assert all("transformer-dev.safetensors" in m.ignore_patterns for m in models)


async def test_text_to_video_runs_the_distilled_pipeline(engine):
    node = ltx.LTXVideo(
        prompt="a door creaks", num_frames=100, width=500, seed=3, low_ram=True
    )
    assert await node.process(_context()) is VIDEO
    [pipeline] = engine
    assert type(pipeline).__name__ == "DistilledPipeline"
    assert pipeline.model_dir.endswith("dgrauet/ltx-2.5-mlx-q8")
    assert pipeline.low_ram_streaming is True
    [(method, kwargs)] = pipeline.calls
    assert method == "generate_and_save"
    assert kwargs["num_frames"] == 97
    assert (kwargs["width"], kwargs["height"]) == (448, 448)
    assert kwargs["seed"] == 3
    assert kwargs["images"] is None


async def test_first_and_last_frame_become_anchors(engine):
    node = ltx.LTXVideo(
        prompt="x",
        image=ImageRef(data=b"a"),
        last_image=ImageRef(data=b"b"),
        strength=0.8,
        quantization=ltx.LTXQuantization.Q4,
    )
    await node.process(_context())
    [pipeline] = engine
    assert pipeline.model_dir.endswith("dgrauet/ltx-2.5-mlx-q4")
    anchors = pipeline.calls[0][1]["images"]
    assert [(a.frame_idx, a.strength) for a in anchors] == [(0, 0.8), (-1, 0.8)]
    assert pipeline.anchor_files == [True, True]


async def test_keyframes_spread_over_the_clip(engine):
    node = ltx.LTXKeyframes(
        prompt="x", images=[ImageRef(data=b"k")] * 3, num_frames=121
    )
    await node.process(_context())
    anchors = engine[0].calls[0][1]["images"]
    assert [a.frame_idx for a in anchors] == [0, 60, 120]


async def test_keyframes_need_two_to_eight_images(engine):
    with pytest.raises(ValueError, match="2 to 8"):
        await ltx.LTXKeyframes(prompt="x", images=[ImageRef(data=b"k")]).process(
            _context()
        )


async def test_audio_to_video_covers_the_audio(engine):
    node = ltx.LTXAudioToVideo(
        prompt="x", audio=AudioRef(data=b"a"), image=ImageRef(data=b"i")
    )
    assert await node.process(_context(audio_ms=4000)) is VIDEO
    [pipeline] = engine
    assert type(pipeline).__name__ == "A2VidDistilledPipeline"
    kwargs = pipeline.calls[0][1]
    assert kwargs["num_frames"] == 89
    assert kwargs["audio_path"].endswith("audio")
    assert kwargs["image"].endswith("image.png")


async def test_extend_decodes_at_the_source_frame_rate(engine):
    node = ltx.LTXExtend(prompt="x", video=VideoRef(data=b"v"), seconds=4.0, seed=1)
    assert await node.process(_context()) is VIDEO
    [pipeline] = engine
    methods = [method for method, _ in pipeline.calls]
    assert methods == ["extend_from_video", "_load_decoders", "decode"]
    assert pipeline.calls[0][1]["extend_frames"] == 13
    assert pipeline.calls[2][1]["frame_rate"] == 25.0


async def test_missing_engine_names_the_install_command(monkeypatch):
    monkeypatch.setattr(ltx.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", None)
    with pytest.raises(
        RuntimeError, match="git\\+https://github.com/dgrauet/ltx-2-mlx@v0.18.0"
    ):
        await ltx.LTXVideo(prompt="x").process(_context())


async def test_missing_pack_asks_for_a_download(engine, monkeypatch):
    monkeypatch.setattr(ltx, "find_cached_snapshot", lambda repo, marker: None)
    with pytest.raises(RuntimeError, match="is not downloaded"):
        await ltx.LTXVideo(prompt="x").process(_context())


def test_ic_lora_helpers():
    assert ltx.pad_frames(100) == 105
    assert ltx.pad_frames(97) == 97
    assert ltx.pad_frames(3) == 9
    assert ltx.control_canvas(1920, 1080) == (1280, 704)
    assert ltx.control_canvas(768, 416) == (768, 384)
    assert ltx.ingredients_prompt(" two cats ", "they dance") == (
        "Reference sheet: two cats\n\nGenerated video: they dance"
    )
    assert ltx.ingredients_prompt("", "they dance") == "Generated video: they dance"


def test_sheet_tiles_images_in_a_grid():
    one = Image.new("RGB", (40, 30), (255, 0, 0))
    assert ltx.compose_sheet([one]).size == (40, 30)
    sheet = ltx.compose_sheet([one] * 3)
    assert sheet.size == ltx.INGREDIENTS_SHEET_SIZE
    # Two columns, two rows: the first cell holds red, the empty fourth is black.
    assert sheet.getpixel((384, 224)) == (255, 0, 0)
    assert sheet.getpixel((1152, 672)) == (0, 0, 0)


async def test_character_loras_attach_unfused(engine):
    lora = ltx.HFLoraLTXConfig(
        lora=ltx.HFLoraLTX(repo_id="me/hero", path="hero.safetensors"), strength=0.7
    )
    await ltx.LTXVideo(prompt="x", loras=[lora]).process(_context())
    [pipeline] = engine
    assert pipeline.lora_mode == "unfused"
    assert pipeline._pending_loras == [("/hub/me/hero/hero.safetensors", 0.7)]


async def test_lora_without_a_file_path_is_refused(engine):
    lora = ltx.HFLoraLTXConfig(lora=ltx.HFLoraLTX(repo_id="me/hero"))
    with pytest.raises(ValueError, match="path of its .safetensors"):
        await ltx.LTXVideo(prompt="x", loras=[lora]).process(_context())


async def test_motion_control_follows_the_source(engine, ffmpeg_calls):
    node = ltx.LTXControl(
        prompt="a robot dances", video=VideoRef(data=b"v"), image=ImageRef(data=b"i")
    )
    assert await node.process(_context()) is VIDEO
    [pipeline] = engine
    assert type(pipeline).__name__ == "ICLoraPipeline"
    assert pipeline.model_dir.endswith("dgrauet/ltx-2.3-mlx-q4")
    assert pipeline.kwargs["lora_paths"] == [
        (f"/hub/{ltx.IC_LORA_MOTION.repo_id}/{ltx.IC_LORA_MOTION.path}", 1.0)
    ]
    assert pipeline.kwargs["gemma_model_id"].endswith(ltx.LTX23_GEMMA_REPO)
    kwargs = pipeline.calls[0][1]
    assert (kwargs["width"], kwargs["height"]) == (1280, 704)
    assert kwargs["num_frames"] == 97
    assert kwargs["video_conditioning"][0][0].endswith("source.mp4")
    assert [a.frame_idx for a in kwargs["images"]] == [0]
    assert ffmpeg_calls == []


async def test_colorize_keeps_the_source_sound(engine, ffmpeg_calls):
    node = ltx.LTXControl(mode=ltx.LTXControlMode.COLORIZE, video=VideoRef(data=b"v"))
    assert await node.process(_context()) is VIDEO
    [pipeline] = engine
    assert ltx.IC_LORA_COLORIZE.path in pipeline.kwargs["lora_paths"][0][0]
    [mux] = ffmpeg_calls
    assert mux[1].endswith("generated.mp4") and mux[3].endswith("source.mp4")
    assert "1:a:0?" in mux


async def test_ingredients_build_a_reference_clip(engine, ffmpeg_calls):
    node = ltx.LTXIngredients(
        images=[ImageRef(data=b"a"), ImageRef(data=b"b")],
        sheet_description="a cat, a hat",
        prompt="the cat puts on the hat",
        quick=True,
    )
    assert await node.process(_context()) is VIDEO
    [clip] = ffmpeg_calls
    assert "scale=768:448" in clip and clip[clip.index("-frames:v") + 1] == "121"
    [pipeline] = engine
    assert pipeline.kwargs["lora_paths"][0][1] == 1.4
    kwargs = pipeline.calls[0][1]
    assert kwargs["prompt"].startswith("Reference sheet: a cat, a hat")
    assert (kwargs["width"], kwargs["height"]) == (1536, 896)
    assert kwargs["video_conditioning"] == [(clip[-1], 1.0)]
    assert kwargs["skip_stage_2"] is True


async def test_upscale_doubles_the_size_and_keeps_the_length(engine, ffmpeg_calls):
    node = ltx.LTXUpscale(video=VideoRef(data=b"v"), keep=0.6)
    assert await node.process(_context()) is VIDEO
    [pipeline] = engine
    assert pipeline.model_dir.endswith("dgrauet/ltx-2.5-mlx-q8")
    assert "gemma_model_id" not in pipeline.kwargs
    kwargs = pipeline.calls[0][1]
    assert (kwargs["width"], kwargs["height"]) == (3840, 2112)
    assert kwargs["num_frames"] == 105
    assert kwargs["single_stage"] is True
    assert kwargs["conditioning_attention_strength"] == 0.6
    pad, mux = ffmpeg_calls
    assert "tpad=stop_mode=clone:stop=5" in pad
    assert mux[mux.index("-frames:v") + 1] == "100"


def test_ic_lora_nodes_recommend_the_23_packs_and_gemma():
    repos = [m.repo_id for m in ltx.LTXControl.get_recommended_models()]
    assert repos == [
        "dgrauet/ltx-2.3-mlx-q4",
        "dgrauet/ltx-2.3-mlx-q8",
        ltx.LTX23_GEMMA_REPO,
    ]
