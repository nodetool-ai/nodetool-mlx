"""Regression tests for the MLX provider's response text streaming."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from nodetool.metadata.types import Chunk, Message, ToolCall

_MISSING = object()
_MLX_MODULES = (
    "mlx",
    "mlx.nn",
    "mlx_lm",
    "mlx_lm.generate",
    "mlx_lm.tokenizer_utils",
    "mlx_lm.sample_utils",
    "mlx_lm.utils",
    "mlx_vlm",
    "mlx_vlm.prompt_utils",
    "mlx_vlm.utils",
    "mlx_vlm.generate",
    "mlx_audio",
    "mlx_audio.tts",
    "mlx_audio.tts.utils",
    "mlx_audio.tts.generate",
)


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_mlx_stubs() -> None:
    """Make the provider importable on test hosts without the MLX packages."""
    mlx = _module("mlx")
    nn = _module("mlx.nn")
    nn.Module = type("Module", (), {})
    mlx.nn = nn

    mlx_lm = _module("mlx_lm")
    mlx_lm.generate = _module("mlx_lm.generate")
    mlx_lm.generate.stream_generate = lambda *args, **kwargs: iter(())
    mlx_lm.tokenizer_utils = _module("mlx_lm.tokenizer_utils")
    mlx_lm.tokenizer_utils.TokenizerWrapper = object
    mlx_lm.sample_utils = _module("mlx_lm.sample_utils")
    mlx_lm.sample_utils.make_sampler = None
    mlx_lm.utils = _module("mlx_lm.utils")
    mlx_lm.utils.load = lambda *args, **kwargs: (None, None)

    mlx_vlm = _module("mlx_vlm")
    mlx_vlm.prompt_utils = _module("mlx_vlm.prompt_utils")
    mlx_vlm.utils = _module("mlx_vlm.utils")
    mlx_vlm.utils.load_config = None
    mlx_vlm.generate = _module("mlx_vlm.generate")

    mlx_audio = _module("mlx_audio")
    mlx_audio.tts = _module("mlx_audio.tts")
    mlx_audio.tts.utils = _module("mlx_audio.tts.utils")
    mlx_audio.tts.utils.load_model = lambda *args, **kwargs: None
    mlx_audio.tts.generate = _module("mlx_audio.tts.generate")


@pytest.fixture
def mlx_provider(monkeypatch: pytest.MonkeyPatch):
    """Import one isolated provider instance and clean it up after the test."""
    import nodetool
    from nodetool.providers import base

    module_names = (
        "nodetool.mlx",
        "nodetool.mlx.mlx_provider",
        "nodetool.mlx.flux_model_loader",
    )
    previous_mlx_modules = {
        name: sys.modules.get(name, _MISSING) for name in _MLX_MODULES
    }
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in module_names}
    previous_mlx_attr = getattr(nodetool, "mlx", _MISSING)
    previous_registry = base._PROVIDER_REGISTRY.get(base.Provider.MLX, _MISSING)

    try:
        importlib.import_module("mlx")
        importlib.import_module("mlx_lm")
        importlib.import_module("mlx_vlm")
        importlib.import_module("mlx_audio")
    except ModuleNotFoundError:
        _install_mlx_stubs()
    provider_module = importlib.import_module("nodetool.mlx.mlx_provider")
    try:
        yield provider_module
    finally:
        if previous_modules["nodetool.mlx.mlx_provider"] is _MISSING:
            provider_module._MLX_LM_THREAD.shutdown(wait=True, cancel_futures=True)
        if previous_registry is _MISSING:
            base._PROVIDER_REGISTRY.pop(base.Provider.MLX, None)
        else:
            base._PROVIDER_REGISTRY[base.Provider.MLX] = previous_registry
        for name, previous in previous_modules.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, previous in previous_mlx_modules.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if previous_mlx_attr is _MISSING:
            if hasattr(nodetool, "mlx"):
                del nodetool.mlx
        else:
            nodetool.mlx = previous_mlx_attr


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
    ):
        return "formatted prompt"

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        self.decode_calls.append(tokens)
        return self.decoded


async def _stream(provider_module, tokenizer, responses, monkeypatch, tools=()):
    provider = provider_module.MLXProvider()

    async def load_model(model: str):
        return object(), tokenizer

    monkeypatch.setattr(provider, "_load_model", load_model)
    monkeypatch.setattr(provider, "_build_sampler", lambda kwargs: None)
    monkeypatch.setattr(
        provider_module,
        "stream_generate",
        lambda *args, **kwargs: iter(responses),
    )
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
    worker_barrier = provider_module._MLX_LM_THREAD.submit(lambda: None)
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
