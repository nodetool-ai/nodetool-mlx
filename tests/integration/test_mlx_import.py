"""
Integration tests for nodetool-mlx package.

These tests verify that the MLX nodes can be imported and basic functionality works.
Since MLX requires Apple Silicon, these tests are designed to run on macOS runners.
"""

import pytest
import sys


def test_python_version():
    """Verify Python version meets requirements."""
    assert sys.version_info >= (3, 10, 13), "Python 3.10.13+ is required"


def test_mlx_available():
    """Test that MLX is available on the system."""
    try:
        import mlx.core as mx
        assert mx is not None, "MLX core module should be importable"
    except ImportError as e:
        pytest.fail(f"MLX is not available: {e}")


def test_import_whisper_node():
    """Test that Whisper node can be imported."""
    try:
        from nodetool.nodes.mlx.automatic_speech_recognition import Whisper
        assert Whisper is not None, "Whisper node should be importable"
    except ImportError as e:
        pytest.fail(f"Failed to import Whisper node: {e}")


def test_import_tts_nodes():
    """Test that TTS nodes can be imported."""
    try:
        from nodetool.nodes.mlx.text_to_speech import (
            BaseMLXTTS,
        )
        assert BaseMLXTTS is not None, "TTS nodes should be importable"
    except ImportError as e:
        pytest.fail(f"Failed to import TTS nodes: {e}")


def test_import_image_generation_nodes():
    """Test that image generation nodes can be imported."""
    try:
        from nodetool.nodes.mlx.text_to_image import Flux1Dev, Flux1Schnell
        assert Flux1Dev is not None, "Flux1Dev should be importable"
        assert Flux1Schnell is not None, "Flux1Schnell should be importable"
    except ImportError as e:
        pytest.fail(f"Failed to import image generation nodes: {e}")


def test_import_text_generation_nodes():
    """Test that text generation nodes can be imported."""
    try:
        from nodetool.nodes.mlx.text_generation import TextGeneration
        assert TextGeneration is not None, "TextGeneration should be importable"
    except ImportError as e:
        pytest.fail(f"Failed to import text generation nodes: {e}")


def test_mlx_whisper_available():
    """Test that mlx-whisper package is available."""
    try:
        import mlx_whisper
        assert mlx_whisper is not None, "mlx-whisper should be importable"
    except ImportError as e:
        pytest.fail(f"mlx-whisper is not available: {e}")


def test_mflux_available():
    """Test that mflux package is available."""
    try:
        import mflux
        assert mflux is not None, "mflux should be importable"
    except ImportError as e:
        pytest.fail(f"mflux is not available: {e}")


def test_mlx_audio_available():
    """Test that mlx-audio package is available."""
    try:
        import mlx_audio
        assert mlx_audio is not None, "mlx-audio should be importable"
    except ImportError as e:
        pytest.fail(f"mlx-audio is not available: {e}")
