"""Tests for the MLX MFLUX image nodes.

These exercise the pure-Python orchestration in ``process`` with a mocked MFLUX
model, so they run on any platform without loading real weights. They guard
against the kind of copy-paste regression where a node forwards the wrong (or
undefined) arguments to ``generate_image``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import sys
from pathlib import Path
from types import SimpleNamespace

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
    # The guard lives on the shared base in ``text_to_image``, so patch the
    # stdlib module rather than a re-export that the node module may not keep.
    monkeypatch.setattr(sys, "platform", "darwin")


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
        registry_mod = pytest.importorskip("mflux.callbacks.callback_registry")

        self.callbacks = registry_mod.CallbackRegistry()


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


def test_progress_callback_uses_krea2_model_attribute():
    node = i2i.MFluxKrea2(prompt="x")
    model = _FakeModelWithRegistry()
    node._krea2_model = model

    callback = node._register_progress_callback(MagicMock(), total_steps=8)
    assert callback in model.callbacks.in_loop_callbacks()


_SETTING_FIELDS = {
    "quantize",
    "steps",
    "guidance",
    "height",
    "width",
    "seed",
    "lora_path",
    "lora_scale",
    "lora_paths",
    "lora_scales",
    "softness",
}


def _mflux_node_classes():
    return [
        cls
        for cls in i2i.__dict__.values()
        if isinstance(cls, type)
        and issubclass(cls, i2i.BaseMFluxNode)
        and cls is not i2i.BaseMFluxNode
    ] + [__import__("nodetool.nodes.mlx.text_to_image", fromlist=["MFlux"]).MFlux]


def test_mflux_input_fields_are_primary_only():
    # HuggingFace image nodes only expose prompt/assets as handles. Settings
    # such as steps and seed stay in the inspector.
    for cls in _mflux_node_classes():
        inputs = set(cls.get_input_fields())
        leaked = inputs & _SETTING_FIELDS
        assert not leaked, f"{cls.__name__} exposes setting handles: {sorted(leaked)}"
        assert inputs, f"{cls.__name__} has no input handles"


def test_mflux_basic_fields_stay_short():
    for cls in _mflux_node_classes():
        basic = cls.get_basic_fields()
        leaked = set(basic) & {
            "quantize",
            "guidance",
            "seed",
            "lora_path",
            "lora_scale",
            "softness",
        }
        assert not leaked, f"{cls.__name__} lists settings as basic: {sorted(leaked)}"
        assert "model" in basic, f"{cls.__name__} is missing model in basic fields"


async def test_krea2_forwards_generate_args():
    node = i2i.MFluxKrea2(
        prompt="a red fox",
        negative_prompt="blurry",
        steps=8,
        guidance=1.0,
        height=1024,
        width=1024,
        seed=42,
    )
    node._krea2_model = _mock_flux_model()

    result = await node.process(_mock_context())

    assert result == "image-ref"
    kwargs = node._krea2_model.generate_image.call_args.kwargs
    assert kwargs["prompt"] == "a red fox"
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance"] == 1.0
    assert kwargs["height"] == 1024
    assert kwargs["width"] == 1024
    assert kwargs["seed"] == 42
    assert "image_path" not in kwargs


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


def _install_fake_mflux_utils(monkeypatch) -> list[str]:
    """Fake the two mflux utility modules the outpaint padding path imports.

    Both mflux 0.18.1 and 0.22.0 expose ``mflux.utils.box_values.BoxValues``
    with a ``parse`` static method; there is no ``mflux.ui`` package.
    """
    import types

    parsed: list[str] = []

    class BoxValues:
        def __init__(self, value: int):
            self.value = value

        @staticmethod
        def parse(value: str) -> "BoxValues":
            parsed.append(value)
            return BoxValues(int(value))

        def normalize_to_dimensions(self, width, height):
            v = self.value
            return types.SimpleNamespace(top=v, right=v, bottom=v, left=v)

    class ImageUtil:
        @staticmethod
        def expand_image(image, top, right, bottom, left):
            return PILImage.new(
                "RGB", (image.width + left + right, image.height + top + bottom)
            )

        @staticmethod
        def create_outpaint_mask_image(
            orig_width, orig_height, top, right, bottom, left
        ):
            return PILImage.new(
                "RGB", (orig_width + left + right, orig_height + top + bottom)
            )

    mflux = types.ModuleType("mflux")
    utils = types.ModuleType("mflux.utils")
    box_values = types.ModuleType("mflux.utils.box_values")
    box_values.BoxValues = BoxValues
    image_util = types.ModuleType("mflux.utils.image_util")
    image_util.ImageUtil = ImageUtil
    for name, mod in {
        "mflux": mflux,
        "mflux.utils": utils,
        "mflux.utils.box_values": box_values,
        "mflux.utils.image_util": image_util,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return parsed


async def test_outpaint_blank_mask_uses_padding(monkeypatch):
    parsed = _install_fake_mflux_utils(monkeypatch)
    node = i2i.MFluxOutpaint(
        prompt="sky",
        image=ImageRef(uri="memory://base"),
        padding="64",
        width=256,
        height=256,
    )
    node._flux_model = _mock_flux_model()
    ctx = _mock_context()
    loaded: list = []
    original = ctx.image_to_pil

    async def _tracking_to_pil(ref):
        loaded.append(ref)
        return await original(ref)

    ctx.image_to_pil = _tracking_to_pil

    assert "mask" not in node.required_inputs()
    result = await node.process(ctx)

    assert result == "image-ref"
    assert parsed == ["64"]
    # The blank mask is never decoded; only the base image is.
    assert loaded == [node.image]
    kwargs = node._flux_model.generate_image.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (256 + 128, 256 + 128)


def test_outpaint_box_values_import_matches_mflux():
    box_values = pytest.importorskip("mflux.utils.box_values")
    assert hasattr(box_values.BoxValues, "parse")


class _ListRegistry:
    """A minimal stand-in for mflux's per-model CallbackRegistry."""

    def __init__(self):
        self.loop: list = []

    def register(self, callback):
        self.loop.append(callback)

    def in_loop_callbacks(self):
        return self.loop


@pytest.mark.parametrize(
    ("node_cls", "model_attr"),
    [(i2i.MFluxFlux2Edit, "_flux2_model"), (i2i.MFluxQwenImageEdit, "_qwen_model")],
)
async def test_edit_nodes_clean_up_when_an_input_image_fails(
    node_cls, model_attr, monkeypatch
):
    import tempfile as tempfile_mod

    written: list[str] = []
    real_ntf = tempfile_mod.NamedTemporaryFile

    def tracking_ntf(*args, **kwargs):
        handle = real_ntf(*args, **kwargs)
        written.append(handle.name)
        return handle

    monkeypatch.setattr(i2i.tempfile, "NamedTemporaryFile", tracking_ntf)

    model = _mock_flux_model()
    model.callbacks = _ListRegistry()
    node = node_cls(
        prompt="edit",
        images=[ImageRef(uri="memory://ok"), ImageRef(uri="memory://bad")],
    )
    setattr(node, model_attr, model)

    ctx = _mock_context()
    pil = PILImage.new("RGB", (8, 8))

    async def to_pil(ref):
        if ref.uri == "memory://bad":
            raise ValueError("cannot decode image")
        return pil

    ctx.image_to_pil = to_pil

    with pytest.raises(ValueError, match="cannot decode"):
        await node.process(ctx)

    # No callback left on the cached model, and no temp PNG left on disk.
    assert model.callbacks.loop == []
    assert written, "the first image should have been staged"
    assert not any(Path(name).exists() for name in written)
    model.generate_image.assert_not_called()


async def test_vlm_node_loads_from_revision_only_cache_on_one_thread(
    monkeypatch, tmp_path
):
    import threading
    import types

    from nodetool.ml.core.model_manager import ModelManager
    from nodetool.nodes.mlx import _hf_cache
    from nodetool.nodes.mlx import image_to_text as i2t

    monkeypatch.setattr(ModelManager, "_models", {})
    monkeypatch.setattr(ModelManager, "_models_by_node", {})
    # try_to_load_from_cache misses a snapshot fetched without refs/main;
    # find_cached_snapshot finds it.
    monkeypatch.setattr(_hf_cache, "find_cached_snapshot", lambda *a, **k: tmp_path)

    threads: list[str] = []
    load_targets: list[str] = []

    def load(target):
        threads.append(threading.current_thread().name)
        load_targets.append(target)
        return SimpleNamespace(config={"model_type": "qwen3_vl"}), object()

    def generate(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return SimpleNamespace(text="a fox")

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.load = load
    mlx_vlm.generate = generate
    mlx_vlm.prompt_utils = SimpleNamespace(
        apply_chat_template=lambda proc, cfg, prompt, num_images: prompt
    )
    mlx_vlm.utils = SimpleNamespace(load_config=lambda target: {})
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)

    node = i2t.MLXVisionLanguage(image=ImageRef(uri="memory://img"))
    assert await node.process(_mock_context()) == "a fox"

    assert load_targets == [str(tmp_path)]
    assert len(set(threads)) == 1 and threads[0].startswith("mlx-vlm")
