"""Tests for mflux model-config resolution, the Flux loader cache, and imports.

mflux needs Apple Silicon, so the config tests fake its two config modules with
the shape both mflux 0.18.1 and 0.22.0 share.
"""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace

import pytest


class _FakeModelConfig:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _base(**overrides):
    fields = dict(
        priority=15,
        aliases=["qwen-image"],
        model_name="Qwen/Qwen-Image",
        base_model=None,
        controlnet_model=None,
        sigma_max_shift=0.9,
        sigma_shift_terminal=0.02,
        text_encoder_overrides={},
    )
    fields.update(overrides)
    return _FakeModelConfig(**fields)


@pytest.fixture
def fake_mflux(monkeypatch):
    qwen = _base()
    registry = {"qwen-image": qwen}

    def from_name(name):
        if name in ("Qwen/Qwen-Image", "qwen-image"):
            return qwen  # exact match returns the registry singleton
        # mflux 0.18.1's _create_config: a fixed field list, so the sigma
        # schedule falls back to the constructor defaults.
        return _FakeModelConfig(
            priority=qwen.priority,
            aliases=qwen.aliases,
            model_name=name,
            base_model=qwen.model_name,
            controlnet_model=qwen.controlnet_model,
            sigma_max_shift=1.15,
            sigma_shift_terminal=None,
            text_encoder_overrides={},
        )

    model_config_cls = SimpleNamespace(from_name=from_name)
    config_pkg = types.ModuleType("mflux.models.common.config")
    config_pkg.ModelConfig = model_config_cls
    config_mod = types.ModuleType("mflux.models.common.config.model_config")
    config_mod.ModelConfig = model_config_cls
    config_mod.AVAILABLE_MODELS = registry
    for name in ("mflux", "mflux.models", "mflux.models.common"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mflux.models.common.config", config_pkg)
    monkeypatch.setitem(
        sys.modules, "mflux.models.common.config.model_config", config_mod
    )
    return qwen


def test_community_repo_keeps_base_sigma_schedule(fake_mflux):
    from nodetool.mlx.mflux_config import mflux_model_config

    config = mflux_model_config("mflux-community/qwen-image-mflux-q4")

    assert config.model_name == "mflux-community/qwen-image-mflux-q4"
    assert config.base_model == "Qwen/Qwen-Image"
    assert config.sigma_max_shift == 0.9
    assert config.sigma_shift_terminal == 0.02


def test_config_is_a_private_copy(fake_mflux):
    from nodetool.mlx.mflux_config import mflux_model_config

    config = mflux_model_config("Qwen/Qwen-Image")
    config.controlnet_model = "some/controlnet"

    assert config is not fake_mflux
    assert fake_mflux.controlnet_model is None


async def test_load_flux_model_caches_without_node_id(monkeypatch):
    from nodetool.ml.core.model_manager import ModelManager
    from nodetool.mlx import flux_model_loader as loader

    monkeypatch.setattr(ModelManager, "_models", {})
    monkeypatch.setattr(ModelManager, "_models_by_node", {})
    monkeypatch.setattr(loader, "has_cached_files", lambda model_id: True)
    monkeypatch.setattr(loader, "check_memory_availability", lambda gb: None)

    loads: list[str] = []

    class Flux1:
        @staticmethod
        def from_name(model_name, quantize):
            loads.append(model_name)
            return object()

    for name in (
        "mflux",
        "mflux.models",
        "mflux.models.flux",
        "mflux.models.flux.variants",
        "mflux.models.flux.variants.txt2img",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    flux_mod = types.ModuleType("mflux.models.flux.variants.txt2img.flux")
    flux_mod.Flux1 = Flux1
    monkeypatch.setitem(sys.modules, flux_mod.__name__, flux_mod)

    # The provider calls without a node_id; the second call must hit the cache.
    first = await loader.load_flux_model("black-forest-labs/FLUX.1-schnell")
    second = await loader.load_flux_model("black-forest-labs/FLUX.1-schnell")

    assert first is second
    assert loads == ["black-forest-labs/FLUX.1-schnell"]


def test_light_submodules_do_not_import_the_provider():
    # Stable Audio 3 and the node modules import nodetool.mlx submodules; that
    # must not pull in mlx-lm, mlx-vlm and mlx-audio through the provider.
    code = (
        "import sys\n"
        "import nodetool.mlx.threads, nodetool.mlx.mflux_config\n"
        "assert 'nodetool.mlx.mlx_provider' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
