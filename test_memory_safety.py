"""
Basic tests for memory safety utilities.

These tests validate the core functionality without requiring
actual MFlux models or macOS-specific features.
"""

import sys
import pytest


def test_memory_estimator_basic():
    """Test that memory estimator returns reasonable values."""
    from nodetool.mlx.memory_estimator import estimate_mflux_memory_bytes, format_bytes

    # Test basic estimation
    memory = estimate_mflux_memory_bytes(
        width=1024,
        height=1024,
        steps=4,
        quantize_bits=4,
        low_memory=False,
        model_type="flux",
    )
    
    # Should be a positive integer
    assert isinstance(memory, int)
    assert memory > 0
    
    # Should be in a reasonable range (e.g., 1-50 GB for this config)
    assert 1 * (1024**3) < memory < 50 * (1024**3)
    
    # Low-memory mode should reduce estimate
    low_mem = estimate_mflux_memory_bytes(
        width=1024,
        height=1024,
        steps=4,
        quantize_bits=4,
        low_memory=True,
        model_type="flux",
    )
    assert low_mem < memory
    
    # Higher quantization should reduce memory
    quant_8bit = estimate_mflux_memory_bytes(
        width=1024,
        height=1024,
        steps=4,
        quantize_bits=8,
        low_memory=False,
        model_type="flux",
    )
    assert quant_8bit > memory  # 8-bit uses more than 4-bit
    
    # Test format_bytes
    formatted = format_bytes(memory)
    assert isinstance(formatted, str)
    assert "GB" in formatted or "MB" in formatted


def test_memory_estimator_model_types():
    """Test that different model types affect memory estimates."""
    from nodetool.mlx.memory_estimator import estimate_mflux_memory_bytes
    
    base_params = {
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "quantize_bits": 4,
        "low_memory": False,
    }
    
    # Different model types should have different base memory
    flux_mem = estimate_mflux_memory_bytes(**base_params, model_type="flux")
    redux_mem = estimate_mflux_memory_bytes(**base_params, model_type="flux-redux")
    schnell_mem = estimate_mflux_memory_bytes(**base_params, model_type="flux-schnell")
    
    # Redux should use more than base flux
    assert redux_mem > flux_mem
    
    # Schnell should use less than dev
    assert schnell_mem < flux_mem


def test_memory_error_message():
    """Test that memory error messages are formatted correctly."""
    from nodetool.mlx.memory_estimator import format_memory_error
    
    error_msg = format_memory_error(
        estimated_bytes=20 * (1024**3),
        available_bytes=15 * (1024**3),
        required_headroom_bytes=2 * (1024**3),
        width=2048,
        height=2048,
        steps=20,
        low_memory=False,
    )
    
    assert isinstance(error_msg, str)
    assert "freeze macOS" in error_msg
    assert "20.0 GB" in error_msg or "20.1 GB" in error_msg  # estimated
    assert "Low-Memory mode" in error_msg or "Low-memory mode" in error_msg
    assert "2048x2048" in error_msg


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only test")
def test_system_memory_macos():
    """Test system memory utilities on macOS."""
    from nodetool.mlx.system_memory import (
        get_system_memory_snapshot,
        check_memory_headroom,
    )
    
    # Get memory snapshot
    snapshot = get_system_memory_snapshot()
    
    # Validate snapshot properties
    assert snapshot.total_bytes > 0
    assert snapshot.available_bytes > 0
    assert snapshot.total_bytes > snapshot.available_bytes
    assert 0 <= snapshot.available_bytes <= snapshot.total_bytes
    
    # Check memory_pressure is valid if present
    if snapshot.memory_pressure is not None:
        assert snapshot.memory_pressure in ["normal", "warn", "critical"]
    
    # Test check_memory_headroom
    has_mem, snap, headroom = check_memory_headroom(
        required_bytes=1024,  # Just 1 KB
        min_headroom_fraction=0.10,
    )
    
    # Should have memory for such a small request
    assert has_mem is True
    assert headroom > 0
    
    # Test insufficient memory scenario
    huge_request = snapshot.total_bytes * 2  # Request more than total RAM
    has_mem, snap, headroom = check_memory_headroom(
        required_bytes=huge_request,
        min_headroom_fraction=0.10,
    )
    
    # Should not have sufficient memory
    assert has_mem is False


def test_system_memory_non_macos():
    """Test that system memory raises on non-macOS platforms."""
    if sys.platform == "darwin":
        pytest.skip("Test only runs on non-macOS platforms")
    
    from nodetool.mlx.system_memory import get_system_memory_snapshot
    
    with pytest.raises(RuntimeError, match="only supported on macOS"):
        get_system_memory_snapshot()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
