"""Tests for the MLX MFLUX image nodes.

These exercise the pure-Python orchestration in ``process`` with a mocked MFLUX
model, so they run on any platform without loading real weights. They guard
against the kind of copy-paste regression where a node forwards the wrong (or
undefined) arguments to ``generate_image``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

import nodetool.nodes.mlx.image_to_image as i2i
from nodetool.metadata.types import ImageRef


def _mock_flux_model() -> MagicMock:
    model = MagicMock()
    model.generate_image.return_value = MagicMock(image=PILImage.new("RGB", (16, 16)))
    return model


def _mock_context() -> MagicMock:
    ctx = MagicMock()
    pil = PILImage.new("RGB", (32, 32))

    async def _to_pil(_ref):
        return pil

    async def _from_pil(img):
        return "image-ref"

    ctx.image_to_pil = _to_pil
    ctx.image_from_pil = _from_pil
    return ctx


@pytest.fixture(autouse=True)
def _force_darwin(monkeypatch):
    # The MFLUX nodes refuse to run off macOS; pretend we're on Apple Silicon.
    monkeypatch.setattr(i2i.sys, "platform", "darwin")


async def test_redux_forwards_redux_image_paths():
    node = i2i.MFluxRedux(
        prompt="a portrait",
        redux_image=ImageRef(uri="memory://ref"),
        redux_image_strength=0.7,
    )
    node._flux_model = _mock_flux_model()

    result = await node.process(_mock_context())

    assert result == "image-ref"
    kwargs = node._flux_model.generate_image.call_args.kwargs
    # Redux is guided by reference images, not an image_path/depth_image_path.
    assert kwargs["redux_image_paths"], "redux image path must be forwarded"
    assert kwargs["redux_image_strengths"] == [0.7]
    assert "image_path" not in kwargs
    assert "depth_image_path" not in kwargs


class _FakeModelWithRegistry:
    """A stand-in MFLUX model carrying a real per-model CallbackRegistry."""

    def __init__(self):
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.callbacks = CallbackRegistry()


def test_progress_callback_registers_reports_and_removes():
    node = i2i.MFluxKontext(prompt="x")
    model = _FakeModelWithRegistry()
    node._flux_model = model

    ctx = MagicMock()
    callback = node._register_progress_callback(ctx, total_steps=10)

    # The callback must land in the model's own in-loop registry (this is the
    # list the MFLUX generation loop iterates).
    assert callback in model.callbacks.in_loop_callbacks()

    # Simulate one MFLUX denoising step driving the callback.
    callback.call_in_loop(
        t=3, seed=0, prompt="x", latents=None, config=None, time_steps=None
    )
    msg = ctx.post_message.call_args.args[0]
    assert msg.node_id == node.id
    assert msg.progress == 3
    assert msg.total == 10

    # Cleanup must unregister so cached models don't accumulate stale callbacks.
    node._remove_progress_callback(callback)
    assert callback not in model.callbacks.in_loop_callbacks()


def test_progress_callback_uses_variant_model_attribute():
    # FLUX.2 nodes store the model on a different attribute; the base helper must
    # still find it and register on its registry.
    node = i2i.MFluxFlux2(prompt="x")
    model = _FakeModelWithRegistry()
    node._flux2_model = model

    callback = node._register_progress_callback(MagicMock(), total_steps=4)
    assert callback in model.callbacks.in_loop_callbacks()


async def test_kontext_forwards_image_path():
    node = i2i.MFluxKontext(
        prompt="an atmospheric scene",
        reference_image=ImageRef(uri="memory://ref"),
    )
    node._flux_model = _mock_flux_model()

    result = await node.process(_mock_context())

    assert result == "image-ref"
    kwargs = node._flux_model.generate_image.call_args.kwargs
    # Kontext is conditioned by a single reference image_path.
    assert kwargs["image_path"] is not None
    assert "depth_image_path" not in kwargs
