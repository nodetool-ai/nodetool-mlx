"""The MLX provider's video models: listing, canvas choice and the node each call runs."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import PIL.Image
import pytest

import nodetool.mlx.video as video
from nodetool.metadata.types import Provider, VideoModel, VideoRef
from nodetool.nodes.mlx.minimax_h3 import H3Resolution, H3Speed
from nodetool.nodes.mlx.text_to_video import LTXQuantization
from nodetool.providers.types import ImageToVideoParams, TextToVideoParams

PROVIDER_PATH = Path(__file__).parents[1] / "src/nodetool/mlx/mlx_provider.py"
VIDEO = VideoRef(uri="memory://out")


def _png(width: int, height: int) -> bytes:
    buffer = BytesIO()
    PIL.Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def nodes(monkeypatch):
    """Record each node the provider builds instead of rendering."""
    built: list = []

    async def process(self, context):
        built.append(self)
        return VIDEO

    monkeypatch.setattr(video.LTXVideo, "process", process)
    monkeypatch.setattr(video.MiniMaxH3Video, "process", process)
    return built


def _model(model_id: str) -> VideoModel:
    return VideoModel(id=model_id, provider=Provider.MLX)


def test_lists_only_downloaded_models(monkeypatch):
    cached = {"dgrauet/ltx-2.5-mlx-q4"}
    monkeypatch.setattr(
        video,
        "find_cached_snapshot",
        lambda repo_id, marker: Path("/hub") if repo_id in cached else None,
    )
    assert [m.id for m in video.mlx_video_models()] == ["dgrauet/ltx-2.5-mlx-q4"]
    cached |= {video.H3_DIT_REPO, video.H3_COMPACT_REPO, video.H3_META_REPO}
    models = video.mlx_video_models()
    assert [m.id for m in models][1:] == [
        "MiniMaxAI/MiniMax-H3",
        "MiniMaxAI/MiniMax-H3:fast",
    ]
    assert all(m.provider == Provider.MLX for m in models)
    assert all(m.supported_tasks == ["text_to_video", "image_to_video"] for m in models)


def test_ltx_canvas_follows_resolution_and_aspect():
    assert video.ltx_canvas(None, None) == (768, 448)
    assert video.ltx_canvas("720p", "16:9") == (1216, 704)
    assert video.ltx_canvas("720p", "9:16") == (704, 1216)
    assert video.ltx_canvas("1080p", "16:9") == (1536, 832)
    assert video.ltx_canvas("480p", None, image_size=(1000, 1000)) == (448, 448)


def test_h3_canvas_picks_the_nearest_tier():
    assert video.h3_canvas(None, None) is H3Resolution.STANDARD
    assert video.h3_canvas("360p", "16:9") is H3Resolution.DRAFT
    assert video.h3_canvas("720p", "9:16") is H3Resolution.HIGH_PORTRAIT
    assert video.h3_canvas("480p", None, image_size=(400, 800)) is (
        H3Resolution.STANDARD_PORTRAIT
    )


async def test_text_to_video_runs_ltx_with_the_pack_quantization(nodes):
    params = TextToVideoParams(
        model=_model("dgrauet/ltx-2.5-mlx-q8"),
        prompt="A boat.",
        resolution="720p",
        aspect_ratio="16:9",
        seed=3,
        duration_seconds=4,
    )
    assert await video.generate_video(MagicMock(), params) is VIDEO
    node = nodes[0]
    assert isinstance(node, video.LTXVideo)
    assert node.quantization == LTXQuantization.Q8
    assert (node.width, node.height, node.num_frames, node.seed) == (1216, 704, 97, 3)
    assert not node.image.is_set()


async def test_image_to_video_runs_h3_fast_with_both_frames(nodes):
    params = ImageToVideoParams(
        model=_model("MiniMaxAI/MiniMax-H3:fast"), prompt="She turns.", num_frames=240
    )
    start, end = _png(300, 600), _png(300, 600)
    await video.generate_video(MagicMock(), params, image=start, end_image=end)
    node = nodes[0]
    assert isinstance(node, video.MiniMaxH3Video)
    assert node.speed is H3Speed.FAST
    assert node.resolution is H3Resolution.STANDARD_PORTRAIT
    assert node.seconds == 10.0 and node.seed == -1
    assert node.image.data == start and node.last_image.data == end


async def test_h3_best_takes_the_step_count_and_clamps_length(nodes):
    params = TextToVideoParams(
        model=_model("MiniMaxAI/MiniMax-H3"), prompt="x", num_inference_steps=12,
        duration_seconds=60,
    )  # fmt: skip
    await video.generate_video(MagicMock(), params)
    node = nodes[0]
    assert (node.speed, node.steps, node.seconds) == (H3Speed.BEST, 12, 15.0)


async def test_unknown_model_is_refused(nodes):
    params = TextToVideoParams(model=_model("other/model"), prompt="x")
    with pytest.raises(ValueError, match="not an MLX video model"):
        await video.generate_video(MagicMock(), params)


def test_provider_overrides_the_video_methods():
    """The worker lists a capability only when the provider class overrides it."""
    tree = ast.parse(PROVIDER_PATH.read_text())
    provider = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXProvider"
    )  # fmt: skip
    methods = {
        node.name: node
        for node in provider.body
        if isinstance(node, ast.AsyncFunctionDef)
    }
    for name in ("get_available_video_models", "text_to_video", "image_to_video"):
        assert name in methods
    # The worker passes a typed params object when the signature names ``params``.
    assert [a.arg for a in methods["text_to_video"].args.args][:2] == ["self", "params"]
    assert [a.arg for a in methods["image_to_video"].args.args][:3] == [
        "self",
        "image",
        "params",
    ]
