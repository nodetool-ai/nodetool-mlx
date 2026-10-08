"""Regression tests for the MLX provider's text-to-speech path.

The provider module imports the MLX stack at import time, so these tests
compile the production method and helpers out of the source file, as the other
provider tests do.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import numpy as np
import pytest

PROVIDER_PATH = Path(__file__).parents[1] / "src/nodetool/mlx/mlx_provider.py"
MODULE_NAMES = {"_GENERATOR_EXHAUSTED", "_KOKORO_LANG_CODES", "_tts_lang_code"}
METHOD_NAMES = {"text_to_speech", "_load_tts_model"}


class _FakeSegment:
    def export(self, path: str, format: str) -> None:
        Path(path).write_bytes(b"RIFF")


class _FakeAudioSegment:
    @staticmethod
    def from_file(data: BytesIO) -> _FakeSegment:
        assert data.read() == b"reference-bytes"
        return _FakeSegment()


def _compile(executor: ThreadPoolExecutor) -> dict[str, Any]:
    tree = ast.parse(PROVIDER_PATH.read_text())
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in MODULE_NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in MODULE_NAMES for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MLXProvider":
            nodes.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name in METHOD_NAMES
            )
    namespace: dict[str, Any] = {
        "Any": Any,
        "AsyncGenerator": AsyncGenerator,
        "AudioSegment": _FakeAudioSegment,
        "BytesIO": BytesIO,
        "MLX_AUDIO_THREAD": executor,
        "Path": Path,
        "asyncio": asyncio,
        "log": logging.getLogger(__name__),
        "np": np,
        "os": os,
        "sys": sys,
        "tempfile": tempfile,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(PROVIDER_PATH), "exec"), namespace)
    return namespace


class _FakeTTSModel:
    def __init__(self, chunks: int = 3) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []
        self.threads: list[str] = []
        self.closed = False
        self.pulled = 0
        self.ref_audio_existed: bool | None = None

    def generate(self, **params: Any):
        self.calls.append(params)
        self.threads.append(threading.current_thread().name)
        if "ref_audio" in params:
            self.ref_audio_existed = Path(params["ref_audio"]).exists()
        try:
            for _ in range(self.chunks):
                self.pulled += 1
                self.threads.append(threading.current_thread().name)
                yield SimpleNamespace(
                    audio=np.full(4, 0.5, dtype=np.float32), sample_rate=24_000
                )
        finally:
            self.closed = True


@pytest.fixture
def tts():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-test")
    namespace = _compile(executor)
    model = _FakeTTSModel()

    class Provider:
        text_to_speech = namespace["text_to_speech"]

        async def _load_tts_model(self, model_id: str) -> Any:
            return model

    try:
        yield SimpleNamespace(
            provider=Provider(), model=model, executor=executor, ns=namespace
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture(autouse=True)
def _darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


async def _collect(stream) -> list[np.ndarray]:
    return [chunk async for chunk in stream]


async def test_none_options_use_defaults_and_kokoro_language_is_mapped(tts):
    chunks = await _collect(
        tts.provider.text_to_speech(
            text="hello",
            model="mlx-community/Kokoro-82M-bf16",
            voice=None,
            speed=None,
            language="en",
            temperature=None,
            reference_audio=None,
            reference_text=None,
            instructions=None,
        )
    )

    assert len(chunks) == 3
    assert chunks[0].dtype == np.int16
    params = tts.model.calls[0]
    assert params["speed"] == 1.0
    assert params["lang_code"] == "a"
    assert params["voice"] == "af_heart"
    assert "temperature" not in params
    assert not {"ref_audio", "ref_text", "instruct"} & params.keys()


@pytest.mark.parametrize(
    ("language", "is_kokoro", "expected"),
    [
        (None, True, None),
        ("en-GB", True, "b"),
        ("ja", True, "j"),
        ("b", True, "b"),
        ("xx", True, None),
        ("english", False, "english"),
        (None, False, None),
    ],
)
def test_tts_lang_code(tts, language, is_kokoro, expected):
    assert tts.ns["_tts_lang_code"](language, is_kokoro) == expected


async def test_model_generation_runs_on_the_audio_thread(tts):
    await _collect(
        tts.provider.text_to_speech(text="hello", model="mlx-community/kitten-tts")
    )

    assert tts.model.threads
    assert all(name.startswith("audio-test") for name in tts.model.threads)


async def test_reference_audio_and_instructions_are_forwarded(tts):
    await _collect(
        tts.provider.text_to_speech(
            text="hello",
            model="mlx-community/Qwen3-TTS",
            reference_audio=b"reference-bytes",
            reference_text="the words",
            instructions="calm voice",
        )
    )

    params = tts.model.calls[0]
    assert tts.model.ref_audio_existed is True
    assert params["ref_text"] == "the words"
    assert params["instruct"] == "calm voice"
    # The staged reference file is removed once the stream ends.
    assert not Path(params["ref_audio"]).exists()


async def test_abandoned_stream_stops_generating(tts):
    tts.model.chunks = 1_000
    stream = tts.provider.text_to_speech(text="hello", model="mlx-community/kitten")
    await stream.__anext__()
    await stream.aclose()
    await asyncio.wrap_future(tts.executor.submit(lambda: None))

    assert tts.model.closed
    assert tts.model.pulled < 1_000


async def test_load_tts_model_loads_on_the_audio_thread(tts, monkeypatch):
    loaded_on: list[str] = []
    cached: dict[str, Any] = {}

    class _Lock:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc):
            return False

    class ModelManager:
        @staticmethod
        def lock_model(key: str) -> _Lock:
            return _Lock()

        @staticmethod
        def get_model(key: str) -> Any:
            return cached.get(key)

        @staticmethod
        def set_model(node_id: str, key: str, model: Any) -> None:
            cached[key] = model

    def load_model(path: Path) -> str:
        loaded_on.append(threading.current_thread().name)
        return "model"

    tts.ns["ModelManager"] = ModelManager
    tts.ns["mlx_audio"] = SimpleNamespace(
        tts=SimpleNamespace(utils=SimpleNamespace(load_model=load_model))
    )

    class Provider:
        _load_tts_model = tts.ns["_load_tts_model"]

        async def _resolve_cached_repo_path(self, repo_id: str) -> Path:
            return Path("/cached")

    assert await Provider()._load_tts_model("repo/tts") == "model"
    assert loaded_on and loaded_on[0].startswith("audio-test")
