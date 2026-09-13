"""Verify that the declared MLX dependencies support a clean provider import."""

import importlib
import importlib.util
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("mlx") is None,
    reason="Requires the installed Apple Silicon MLX runtime",
)
def test_provider_imports_with_installed_runtime():
    module = importlib.import_module("nodetool.mlx.mlx_provider")
    assert module.MLXProvider.provider.value == "mlx"
