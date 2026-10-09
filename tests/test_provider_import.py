"""Verify that the declared MLX dependencies support a clean provider import."""

import importlib
import importlib.util
import platform
import sys

import pytest

from nodetool.metadata.types import Provider
from nodetool.providers.base import _PROVIDER_REGISTRY

PROVIDER_MODULE = "nodetool.mlx.mlx_provider"


@pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("mlx") is None,
    reason="Requires the installed Apple Silicon MLX runtime",
)
def test_provider_imports_with_installed_runtime():
    module = importlib.import_module(PROVIDER_MODULE)
    assert module.MLXProvider.provider.value == "mlx"


@pytest.mark.parametrize(
    ("os_name", "machine"),
    [("linux", "x86_64"), ("win32", "AMD64"), ("darwin", "x86_64")],
)
def test_provider_does_not_register_off_apple_silicon(monkeypatch, os_name, machine):
    monkeypatch.setattr(sys, "platform", os_name)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    for name in [
        m for m in sys.modules if m == "nodetool.mlx" or m.startswith("nodetool.mlx.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(_PROVIDER_REGISTRY, Provider.MLX, raising=False)

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(PROVIDER_MODULE)

    assert excinfo.value.name == PROVIDER_MODULE
    assert Provider.MLX not in _PROVIDER_REGISTRY
