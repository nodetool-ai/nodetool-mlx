"""Regression tests for the MLX provider's text-to-image argument building.

The provider module imports the MLX stack at import time, so the method is
compiled out of the source file with fakes for mflux and the model cache.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import sys
import types
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import PIL.Image
import pytest

PROVIDER_PATH = Path(__file__).parents[1] / "src/nodetool/mlx/mlx_provider.py"


class _Lock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _ModelManager:
    models: dict[str, Any] = {}

    @staticmethod
    def lock_model(key: str) -> _Lock:
        return _Lock()

    @classmethod
    def get_model(cls, key: str) -> Any:
        return cls.models.get(key)

    @classmethod
    def set_model(cls, node_id: Any, key: str, model: Any) -> None:
        cls.models[key] = model


def _image() -> SimpleNamespace:
    return SimpleNamespace(image=PIL.Image.new("RGB", (4, 4)))


class Flux2Klein:
    """Mirrors mflux's signature: no negative_prompt and no **kwargs."""

    calls: list[dict[str, Any]] = []

    def __init__(self, quantize, model_config):
        self.model_config = model_config

    def generate_image(
        self, seed, prompt, num_inference_steps=4, height=1024, width=1024, guidance=1.0
    ):
        Flux2Klein.calls.append(
            dict(seed=seed, prompt=prompt, guidance=guidance, steps=num_inference_steps)
        )
        return _image()


class ZImage:
    instances: list["ZImage"] = []

    def __init__(self, quantize, model_config):
        self.model_config = model_config
        ZImage.instances.append(self)

    def generate_image(self, **kwargs):
        return _image()


class _Flux1:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return _image()


@pytest.fixture
def provider(monkeypatch):
    tree = ast.parse(PROVIDER_PATH.read_text())
    provider_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXProvider"
    )
    method = next(
        node
        for node in provider_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "text_to_image"
    )
    flux1 = _Flux1()

    async def load_flux_model(model_id, quantize, task):
        return flux1

    namespace: dict[str, Any] = {
        "Any": Any,
        "BytesIO": BytesIO,
        "ImageBytes": bytes,
        "ModelManager": _ModelManager,
        "PIL": PIL,
        "ProcessingContext": Any,
        "TextToImageParams": Any,
        "asyncio": asyncio,
        "json": json,
        "load_flux_model": load_flux_model,
        "log": logging.getLogger(__name__),
        "mflux_model_config": lambda repo_id: f"config:{repo_id}",
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(PROVIDER_PATH), "exec"), namespace)

    fakes = {
        "mflux.models.common.config": dict(ModelConfig=object()),
        "mflux.models.flux2.variants": dict(Flux2Klein=Flux2Klein),
        "mflux.models.z_image.variants": dict(ZImage=ZImage),
    }
    for name, attrs in fakes.items():
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_ModelManager, "models", {})
    Flux2Klein.calls.clear()
    ZImage.instances.clear()

    class Provider:
        text_to_image = namespace["text_to_image"]

    return SimpleNamespace(provider=Provider(), flux1=flux1)


def _params(model_id: str, **overrides) -> SimpleNamespace:
    fields = dict(
        prompt="a fox",
        negative_prompt=None,
        model=SimpleNamespace(id=model_id),
        seed=1,
        num_inference_steps=None,
        height=None,
        width=None,
        guidance_scale=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


async def test_flux2_does_not_receive_negative_prompt(provider):
    image = await provider.provider.text_to_image(
        _params("black-forest-labs/FLUX.2-klein-4B", negative_prompt="blurry")
    )

    assert image.startswith(b"\x89PNG")
    # Unset guidance keeps Flux2Klein's own default.
    assert Flux2Klein.calls[0]["guidance"] == 1.0


async def test_unset_guidance_keeps_the_flux1_default(provider):
    await provider.provider.text_to_image(_params("black-forest-labs/FLUX.1-dev"))
    await provider.provider.text_to_image(
        _params("black-forest-labs/FLUX.1-dev", guidance_scale=3.5)
    )

    assert "guidance" not in provider.flux1.calls[0]
    assert "negative_prompt" not in provider.flux1.calls[0]
    assert provider.flux1.calls[1]["guidance"] == 3.5


async def test_z_image_uses_the_requested_repo(provider):
    repo = "mflux-community/z-image-turbo-mflux-q4"
    await provider.provider.text_to_image(_params(repo))

    assert ZImage.instances[0].model_config == f"config:{repo}"
