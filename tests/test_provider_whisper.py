"""Regression tests for the MLX provider's Whisper transcription adapter."""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_transcription_method():
    """Compile the production method without importing MLX or registering a provider."""

    provider_path = Path(__file__).parents[1] / "src/nodetool/mlx/mlx_provider.py"
    tree = ast.parse(provider_path.read_text())
    provider_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXProvider"
    )
    method = next(
        node
        for node in provider_class.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == "automatic_speech_recognition"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {
        "Any": Any,
        "asyncio": asyncio,
        "convert_audio_to_standard_format": None,
        "inspect": inspect,
        "log": logging.getLogger(__name__),
        "np": np,
        "sys": sys,
    }
    exec(compile(module, str(provider_path), "exec"), namespace)
    return namespace["automatic_speech_recognition"]


@pytest.fixture
def transcription_method():
    return _load_transcription_method()


@pytest.fixture
def configured_provider(transcription_method, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    converted_audio = np.array([0, 16384, -16384], dtype=np.int16)

    def convert_audio(audio: bytes, *, target_sample_rate: int) -> np.ndarray:
        assert audio == b"audio"
        assert target_sample_rate == 16_000
        return converted_audio

    monkeypatch.setitem(
        transcription_method.__globals__,
        "convert_audio_to_standard_format",
        convert_audio,
    )

    async def resolve_cached_repo_path(self: Any, model: str) -> Path:
        assert model == "mlx-community/whisper-base-mlx"
        return Path("/cached/whisper")

    class ProviderHarness:
        automatic_speech_recognition = transcription_method

        _resolve_cached_repo_path = resolve_cached_repo_path

    return ProviderHarness()


async def test_whisper_forwards_decode_options_to_var_keyword_signature(
    configured_provider, monkeypatch: pytest.MonkeyPatch
):
    received: dict[str, Any] = {}

    def transcribe(audio: np.ndarray, path_or_hf_repo: str, **decode_options: Any):
        received["audio"] = audio
        received["path_or_hf_repo"] = path_or_hf_repo
        received.update(decode_options)
        return {"text": "translated"}

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe)
    )

    result = await configured_provider.automatic_speech_recognition(
        b"audio",
        model="mlx-community/whisper-base-mlx",
        language="de",
        temperature=0.25,
        task="translate",
        beam_size=5,
    )

    assert result == "translated"
    assert received["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"
    np.testing.assert_array_equal(received["audio"], np.array([0.0, 0.5, -0.5]))
    assert received["language"] == "de"
    assert received["task"] == "translate"
    assert received["beam_size"] == 5
    assert received["temperature"] == 0.25


async def test_whisper_keeps_filtering_strict_signature(
    configured_provider, monkeypatch: pytest.MonkeyPatch
):
    received: dict[str, Any] = {}

    def transcribe(
        audio: np.ndarray,
        path_or_hf_repo: str,
        language: str | None = None,
        temperature: float = 0.0,
    ):
        received.update(
            audio=audio,
            path_or_hf_repo=path_or_hf_repo,
            language=language,
            temperature=temperature,
        )
        return "strict"

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe)
    )

    result = await configured_provider.automatic_speech_recognition(
        b"audio",
        model="mlx-community/whisper-base-mlx",
        language="de",
        temperature=0.25,
        task="translate",
        beam_size=5,
    )

    assert result == "strict"
    assert received["language"] == "de"
    assert received["temperature"] == 0.25
    assert "task" not in received
    assert "beam_size" not in received
