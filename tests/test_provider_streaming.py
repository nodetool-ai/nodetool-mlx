"""Regression tests for the MLX provider's response text streaming."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Sequence

import pytest

from nodetool.metadata.types import Chunk, Message, ToolCall

PROVIDER_PATH = Path(__file__).parents[1] / "src/nodetool/mlx/mlx_provider.py"
STREAM_METHODS = (
    "_stream_chat",
    "_strip_terminal_special_tokens",
    "_process_response_text",
    "_parse_tool_call",
    "_convert_message",
    "_convert_tools",
    "_normalize_content",
    "_build_stream_kwargs",
    "_build_sampler",
)


def _extract_stream_methods() -> dict[str, Any]:
    """Extract provider methods without importing optional MLX packages."""
    tree = ast.parse(PROVIDER_PATH.read_text())
    methods = {}
    namespace = {
        "Any": Any,
        "AsyncIterator": AsyncIterator,
        "Sequence": Sequence,
        "Tool": object,
        "Chunk": Chunk,
        "Message": Message,
        "ToolCall": ToolCall,
        "TokenizerWrapper": Any,
        "json": json,
        "asyncio": asyncio,
        "logging": logging,
        "os": os,
        "threading": __import__("threading"),
        "time": __import__("time"),
        "log": logging.getLogger(__name__),
    }
    provider_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXProvider"
    )
    for method_name in STREAM_METHODS:
        method = next(
            node
            for node in provider_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(PROVIDER_PATH),
                "exec",
            ),
            namespace,
        )
        methods[method_name] = namespace[method_name]
    methods["_globals"] = namespace
    return methods


@pytest.fixture
def mlx_provider():
    """Build a provider from real streaming methods and local runtime globals."""
    methods = _extract_stream_methods()
    runtime_globals = methods.pop("_globals")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-test")
    runtime_globals["_MLX_LM_THREAD"] = executor
    runtime_globals["stream_generate"] = lambda *args, **kwargs: iter(())

    class StreamProvider:
        def __init__(self) -> None:
            self._generation_lock = asyncio.Lock()
            self.usage: dict[str, int] = {}

        def _extract_image_parts(self, messages):
            return []

        def _extract_audio_parts(self, messages):
            return []

        def _is_fibo_vlm_model(self, model):
            return False

        def _is_vision_model(self, model):
            return False

        def _is_audio_model(self, model):
            return False

    for name, method in methods.items():
        setattr(StreamProvider, name, method)

    try:
        yield SimpleNamespace(
            MLXProvider=StreamProvider,
            executor=executor,
            globals=runtime_globals,
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class Response:
    text: str
    token: int
    finish_reason: str | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    model: str = "test-model"


class FakeTokenizer:
    has_tool_calling = False

    def __init__(self, decoded: str = "") -> None:
        self.decoded = decoded
        self.decode_calls: list[list[int]] = []

    def apply_chat_template(
        self, messages: Any, tools: Any, add_generation_prompt: bool
    ) -> str:
        return "formatted prompt"

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        self.decode_calls.append(tokens)
        return self.decoded


async def _stream(harness, tokenizer, responses, monkeypatch, tools=()):
    provider = harness.MLXProvider()

    async def load_model(model: str):
        return object(), tokenizer

    monkeypatch.setattr(provider, "_load_model", load_model, raising=False)
    monkeypatch.setattr(provider, "_build_sampler", lambda kwargs: None, raising=False)
    monkeypatch.setattr(provider, "_update_usage", lambda response: None, raising=False)
    harness.globals["stream_generate"] = lambda *args, **kwargs: iter(responses)
    items = [
        item
        async for item in provider._stream_chat(
            [Message(role="user", content="hello")],
            "test-model",
            tools,
            max_tokens=16,
            context_window=128,
            response_format=None,
        )
    ]
    # Let the producer finish its final queue.put before fixture teardown shuts
    # down the single-worker executor. This barrier runs while the loop is alive.
    worker_barrier = harness.executor.submit(lambda: None)
    await asyncio.wrap_future(worker_barrier)
    return items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decoded", "emitted_text"),
    [("�", "é"), (" ", " word")],
)
async def test_empty_detokenizer_buffer_does_not_decode_raw_token(
    mlx_provider, monkeypatch, decoded, emitted_text
):
    tokenizer = FakeTokenizer(decoded=decoded)
    responses = [
        # The detokenizer has buffered this token, so response.text is empty.
        Response(text="", token=11),
        Response(text=emitted_text, token=12),
        Response(text="", token=13, finish_reason="stop"),
    ]

    items = await _stream(mlx_provider, tokenizer, responses, monkeypatch)

    assert [(item.content, item.done) for item in items if isinstance(item, Chunk)] == [
        (emitted_text, False),
        ("", True),
    ]
    assert tokenizer.decode_calls == []


class NativeToolTokenizer(FakeTokenizer):
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"


class LookupTool:
    name = "lookup"
    description = "Look up a value."
    input_schema = {"type": "object"}

    def tool_param(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@pytest.mark.asyncio
async def test_native_tool_call_text_is_still_parsed(mlx_provider, monkeypatch):
    tokenizer = NativeToolTokenizer(decoded='{"name":"wrong"}')
    responses = [
        Response(text='<tool_call>{"name":"lookup",', token=21),
        Response(text='"arguments":{"q":"x"}}</tool_call>', token=22),
        Response(text="", token=23, finish_reason="stop"),
    ]

    items = await _stream(
        mlx_provider,
        tokenizer,
        responses,
        monkeypatch,
        tools=(LookupTool(),),
    )

    calls = [item for item in items if isinstance(item, ToolCall)]
    assert [(call.name, call.args) for call in calls] == [("lookup", {"q": "x"})]
    assert tokenizer.decode_calls == []


def test_build_sampler_treats_none_as_default(mlx_provider):
    """The TS bridge sends unset options as None (msgpack ``undefined``)."""
    captured: dict[str, Any] = {}

    def make_sampler(**params: Any) -> str:
        captured.update(params)
        return "sampler"

    mlx_provider.globals["mlx_lm"] = SimpleNamespace(
        sample_utils=SimpleNamespace(make_sampler=make_sampler)
    )
    provider = mlx_provider.MLXProvider()
    kwargs = {"temperature": None, "top_p": None, "top_k": None, "seed": 1}

    assert provider._build_sampler(kwargs) == "sampler"
    assert captured["temp"] == 0.5
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 50
    assert kwargs == {"seed": 1}


@pytest.mark.asyncio
async def test_stream_chat_treats_none_options_as_unset(mlx_provider, monkeypatch):
    provider = mlx_provider.MLXProvider()
    calls: list[dict[str, Any]] = []
    sampler_inputs: list[dict[str, Any]] = []

    async def load_model(model: str):
        return object(), FakeTokenizer()

    def build_sampler(kwargs: dict[str, Any]) -> None:
        sampler_inputs.append(dict(kwargs))
        return None

    def stream_generate(*args: Any, **kwargs: Any):
        calls.append(kwargs)
        return iter([Response(text="hi", token=1, finish_reason="stop")])

    monkeypatch.setattr(provider, "_load_model", load_model, raising=False)
    monkeypatch.setattr(provider, "_build_sampler", build_sampler, raising=False)
    monkeypatch.setattr(provider, "_update_usage", lambda response: None, raising=False)
    mlx_provider.globals["stream_generate"] = stream_generate

    items = [
        item
        async for item in provider._stream_chat(
            [Message(role="user", content="hello")],
            "test-model",
            (),
            max_tokens=None,
            context_window=128,
            response_format=None,
            temperature=None,
            top_p=None,
        )
    ]
    await asyncio.wrap_future(mlx_provider.executor.submit(lambda: None))

    assert [item.content for item in items if isinstance(item, Chunk)] == ["hi"]
    assert calls[0]["max_tokens"] == 8192
    assert None not in calls[0].values()
    assert "temperature" not in sampler_inputs[0]
    assert "top_p" not in sampler_inputs[0]


@pytest.mark.asyncio
async def test_abandoned_stream_stops_the_generation_thread(mlx_provider, monkeypatch):
    """A consumer that stops early must not leave the mlx-lm thread generating."""
    import threading
    import time

    provider = mlx_provider.MLXProvider()
    produced = 0
    finished = threading.Event()

    async def load_model(model: str):
        return object(), FakeTokenizer()

    def endless_stream(*args: Any, **kwargs: Any):
        nonlocal produced
        try:
            # Bounded so a regression fails the timeout below instead of
            # hanging the executor shutdown forever.
            while produced < 20_000:
                produced += 1
                yield Response(text="x", token=produced)
                time.sleep(0.001)
        finally:
            finished.set()

    monkeypatch.setattr(provider, "_load_model", load_model, raising=False)
    monkeypatch.setattr(provider, "_build_sampler", lambda kwargs: None, raising=False)
    monkeypatch.setattr(provider, "_update_usage", lambda response: None, raising=False)
    mlx_provider.globals["stream_generate"] = endless_stream

    stream = provider._stream_chat(
        [Message(role="user", content="hello")],
        "test-model",
        (),
        max_tokens=None,
        context_window=128,
        response_format=None,
    )
    first = await stream.__anext__()
    assert first.content == "x"
    await stream.aclose()

    # The producer thread notices the stop flag on its next token and exits.
    await asyncio.wait_for(
        asyncio.wrap_future(mlx_provider.executor.submit(lambda: None)), timeout=5
    )
    assert finished.is_set()
