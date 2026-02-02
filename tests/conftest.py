"""
Pytest configuration for nodetool-mlx integration tests.
"""

import sys
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring MLX hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Add integration marker to all tests in integration directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
