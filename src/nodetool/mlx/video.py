"""Video models of the MLX provider: LTX-2.5 and MiniMax H3.

The provider runs the same nodes the graph uses (``mlx.text_to_video.LTXVideo``
and ``mlx.minimax_h3.MiniMaxH3Video``), so a provider call and a node render
the same clip. This module does not import the MLX stack, so it is tested on
any platform.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from nodetool.metadata.types import ImageRef, Provider, VideoModel, VideoRef
from nodetool.nodes.mlx._hf_cache import find_cached_snapshot
from nodetool.nodes.mlx.minimax_h3 import (
    H3_COMPACT_REPO,
    H3_DIT_FILE,
    H3_DIT_REPO,
    H3_FPS,
    H3_META_REPO,
    H3_TEXT_CONFIG,
    H3Resolution,
    H3Speed,
    MiniMaxH3Video,
)
from nodetool.nodes.mlx.text_to_video import (
    LTX_MLX_MARKER,
    LTX_MLX_MAX_FRAMES,
    LTX_MLX_MIN_FRAMES,
    LTX_MLX_PACKS,
    LTXVideo,
)
from nodetool.workflows.processing_context import ProcessingContext

H3_MODEL_ID = "MiniMaxAI/MiniMax-H3"
H3_FAST_SUFFIX = ":fast"
VIDEO_TASKS = ["text_to_video", "image_to_video"]
VIDEO_CAPABILITIES = ["text_to_video", "image_to_video", "audio"]

LTX_FPS = 24.0
LTX_DEFAULT_FRAMES = 121
LTX_DEFAULT_SHORT_SIDE = 448
LTX_MAX_SHORT_SIDE = 1024
LTX_MAX_LONG_SIDE = 1536
H3_DEFAULT_SECONDS = 5.0
H3_MIN_SECONDS = 1.0
H3_MAX_SECONDS = 15.0


def _h3_weights_cached() -> bool:
    return all(
        find_cached_snapshot(repo_id, marker) is not None
        for repo_id, marker in (
            (H3_DIT_REPO, H3_DIT_FILE),
            (H3_COMPACT_REPO, "text_encoder.safetensors"),
            (H3_META_REPO, H3_TEXT_CONFIG),
        )
    )


def mlx_video_models() -> list[VideoModel]:
    """The LTX-2.5 packs and MiniMax H3 (Best and Fast) whose weights are downloaded."""
    models = [
        VideoModel(
            id=repo_id,
            name=f"LTX-2.5 MLX {quantization.value}",
            provider=Provider.MLX,
            supported_tasks=list(VIDEO_TASKS),
            capabilities=list(VIDEO_CAPABILITIES),
        )
        for quantization, repo_id in LTX_MLX_PACKS.items()
        if find_cached_snapshot(repo_id, LTX_MLX_MARKER) is not None
    ]
    if _h3_weights_cached():
        for model_id, name in (
            (H3_MODEL_ID, "MiniMax H3 MLX"),
            (H3_MODEL_ID + H3_FAST_SUFFIX, "MiniMax H3 MLX Fast"),
        ):
            models.append(
                VideoModel(
                    id=model_id,
                    name=name,
                    provider=Provider.MLX,
                    supported_tasks=list(VIDEO_TASKS),
                    capabilities=list(VIDEO_CAPABILITIES),
                )
            )
    return models


def _aspect(aspect_ratio: str | None, image_size: tuple[int, int] | None) -> float:
    """Width over height: from the image when there is one, else ``"W:H"``, else 16:9."""
    if image_size and image_size[0] > 0 and image_size[1] > 0:
        return image_size[0] / image_size[1]
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)\s*", aspect_ratio or ""
    )
    if match and float(match.group(2)) > 0:
        return float(match.group(1)) / float(match.group(2))
    return 16 / 9


def _short_side(resolution: str | None) -> int | None:
    match = re.search(r"\d+", resolution or "")
    return int(match.group()) if match else None


def _snap64(value: float) -> int:
    return max(256, int(value) // 64 * 64)


def ltx_canvas(
    resolution: str | None,
    aspect_ratio: str | None,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """``(width, height)`` for LTX, multiples of 64. 448p at 16:9 is the node's 768x448."""
    short = _snap64(
        min(_short_side(resolution) or LTX_DEFAULT_SHORT_SIDE, LTX_MAX_SHORT_SIDE)
    )
    ratio = _aspect(aspect_ratio, image_size)
    stretch = max(ratio, 1 / ratio)
    long = _snap64(short * stretch)
    if long > LTX_MAX_LONG_SIDE:
        long, short = LTX_MAX_LONG_SIDE, _snap64(LTX_MAX_LONG_SIDE / stretch)
    return (long, short) if ratio >= 1 else (short, long)


def h3_canvas(
    resolution: str | None,
    aspect_ratio: str | None,
    image_size: tuple[int, int] | None = None,
) -> H3Resolution:
    """The H3 canvas nearest to ``resolution``, in the orientation of the aspect ratio."""
    short = _short_side(resolution)
    if short is None or 400 < short <= 500:
        tier = "STANDARD"
    elif short <= 400:
        tier = "DRAFT"
    else:
        tier = "HIGH"
    if _aspect(aspect_ratio, image_size) < 1:
        tier += "_PORTRAIT"
    return H3Resolution[tier]


def _image_size(data: bytes | None) -> tuple[int, int] | None:
    if not data:
        return None
    import PIL.Image

    with PIL.Image.open(BytesIO(data)) as image:
        return image.size


def _extra(params: Any) -> dict[str, Any]:
    return dict(getattr(params, "model_extra", None) or {})


def _duration(params: Any) -> float | None:
    extra = _extra(params)
    duration = extra.get("duration_seconds") or extra.get("duration")
    return float(duration) if duration else None


async def generate_video(
    context: ProcessingContext,
    params: Any,
    image: bytes | None = None,
    end_image: bytes | None = None,
) -> VideoRef:
    """Render ``params`` (``TextToVideoParams`` or ``ImageToVideoParams``) with the MLX nodes."""
    model_id = params.model.id
    prompt = params.prompt or ""
    seed = params.seed if params.seed is not None else -1
    start = ImageRef(data=image) if image else ImageRef()
    last = ImageRef(data=end_image) if end_image else ImageRef()
    size = _image_size(image)

    if model_id in (H3_MODEL_ID, H3_MODEL_ID + H3_FAST_SUFFIX):
        if params.num_frames:
            seconds = int(params.num_frames) / H3_FPS
        else:
            seconds = _duration(params) or H3_DEFAULT_SECONDS
        node = MiniMaxH3Video(
            prompt=prompt,
            image=start,
            last_image=last,
            seconds=min(max(seconds, H3_MIN_SECONDS), H3_MAX_SECONDS),
            resolution=h3_canvas(params.resolution, params.aspect_ratio, size),
            speed=H3Speed.FAST if model_id.endswith(H3_FAST_SUFFIX) else H3Speed.BEST,
            steps=(
                int(params.num_inference_steps or 0)
                if not model_id.endswith(H3_FAST_SUFFIX)
                else 0
            ),
            seed=seed,
        )
        return await node.process(context)

    quantization = next(
        (quant for quant, repo_id in LTX_MLX_PACKS.items() if repo_id == model_id), None
    )
    if quantization is None:
        raise ValueError(
            f"{model_id} is not an MLX video model. Use an LTX-2.5 MLX pack or MiniMax H3."
        )
    width, height = ltx_canvas(params.resolution, params.aspect_ratio, size)
    duration = _duration(params)
    if params.num_frames:
        num_frames = int(params.num_frames)
    elif duration:
        num_frames = round(duration * LTX_FPS) + 1
    else:
        num_frames = LTX_DEFAULT_FRAMES
    node = LTXVideo(
        prompt=prompt,
        image=start,
        last_image=last,
        quantization=quantization,
        num_frames=min(max(num_frames, LTX_MLX_MIN_FRAMES), LTX_MLX_MAX_FRAMES),
        width=width,
        height=height,
        seed=seed,
    )
    return await node.process(context)
