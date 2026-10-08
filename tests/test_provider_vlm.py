"""Regression tests for the provider's mlx-vlm integration.

The provider imports Metal-only packages at module import time, so these tests
extract the small pure-Python helpers and exercise them with a contract-faithful
mlx-vlm prompt utility double.
"""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Stands in for the provider's single mlx-vlm thread.
_VLM_TEST_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-test")

PROVIDER_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "nodetool" / "mlx" / "mlx_provider.py"
)


class _Text:
    def __init__(self, text: str):
        self.text = text


class _Image:
    type = "image_url"


class _Audio:
    type = "audio"


class _PromptUtils:
    """Small model of mlx-vlm 0.4.4's single-message/list split.

    The real list path moves side-channel media to the last user turn. The
    provider avoids that path by calling apply_chat_template once per message
    with return_messages=True, then calling get_chat_template for the full list.
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        processor: Any,
        config: Any,
        prompt: dict[str, Any],
        *,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
    ) -> list[dict[str, Any]]:
        assert isinstance(prompt, dict)
        self.calls.append(
            {
                "role": prompt["role"],
                "num_images": num_images,
                "num_audios": num_audios,
                "return_messages": return_messages,
            }
        )
        raw_content = prompt.get("content", "")
        if isinstance(raw_content, str):
            text = raw_content
        else:
            text = "".join(
                str(part.get("text", ""))
                for part in raw_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        role = prompt["role"]
        if role == "user":
            content: Any = (
                [{"type": "image"}] * num_images
                + ([{"type": "text", "text": text}] if text else [])
                + [{"type": "audio"}] * num_audios
            )
        else:
            content = text
        return [{"role": role, "content": content}]

    def get_chat_template(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        lines = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                rendered = "".join(
                    (
                        "<image>"
                        if item.get("type") == "image"
                        else (
                            "<audio>"
                            if item.get("type") == "audio"
                            else item.get("text", "")
                        )
                    )
                    for item in content
                )
            else:
                rendered = content
            lines.append(f"{message['role']}: {rendered}")
        if add_generation_prompt:
            lines.append("assistant:")
        return "\n".join(lines)


def _helpers() -> tuple[type[Any], Any, type[Any], type[Any], type[Any]]:
    tree = ast.parse(PROVIDER_PATH.read_text())
    provider = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXProvider"
    )
    names = {
        "_normalize_content",
        "_convert_message",
        "_convert_vlm_message",
        "_format_vlm_messages",
        "_stream_vlm_chat",
    }
    selected = [
        node
        for node in provider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_vlm_generate"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "json": json,
        "MessageTextContent": _Text,
        "MessageImageContent": _Image,
        "MessageAudioContent": _Audio,
        "mlx_vlm": SimpleNamespace(prompt_utils=None),
        "asyncio": asyncio,
        "MLX_VLM_THREAD": _VLM_TEST_THREAD,
        "os": SimpleNamespace(remove=lambda path: None),
        "Chunk": SimpleNamespace,
        "log": SimpleNamespace(debug=lambda *args: None, exception=lambda *args: None),
    }
    module = ast.Module(body=[resolver, *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PROVIDER_PATH), "exec"), namespace)
    helper_type = type(
        "ProviderHelpers",
        (),
        {name: namespace[name] for name in names},
    )
    return (
        helper_type,
        namespace["_resolve_vlm_generate"],
        _Text,
        _Image,
        _Audio,
    )


def _message(role: str, content: Any, **extra: Any) -> SimpleNamespace:
    values = {
        "role": role,
        "content": content,
        "name": None,
        "tool_call_id": None,
        "tool_calls": None,
    }
    values.update(extra)
    return SimpleNamespace(**values)


class _AsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _stream_provider(
    prompt_utils: Any,
    generate_export: Any,
    removed_paths: list[str],
    *,
    force_assets: bool = False,
    audio_failure: bool = False,
):
    provider_type, _resolve, _text, _image, _audio = _helpers()
    provider = provider_type()
    provider._vlm_generation_lock = _AsyncLock()
    provider._ensure_vlm_processor_ready = lambda proc, cfg: None

    async def load_model(model: str):
        return object(), object(), {"model_type": "qwen3_vl"}

    async def prepare_images(parts: Any, *, include_parts: bool = False):
        return (["image.tmp"] if (parts or force_assets) else [], list(parts))

    async def prepare_audio(parts: Any, *, include_parts: bool = False):
        if audio_failure:
            raise RuntimeError("audio preparation failed")
        return (["audio.tmp"] if (parts or force_assets) else [], list(parts))

    provider._load_vlm_model = load_model
    provider._prepare_vlm_images = prepare_images
    provider._prepare_vlm_audio = prepare_audio
    globals_namespace = provider_type._stream_vlm_chat.__globals__
    globals_namespace["mlx_vlm"] = SimpleNamespace(
        prompt_utils=prompt_utils,
        generate=generate_export,
    )
    globals_namespace["os"] = SimpleNamespace(
        remove=lambda path: removed_paths.append(path)
    )
    return provider


def test_generate_export_supports_function_and_legacy_module():
    _provider, resolve, _text, _image, _audio = _helpers()

    def function() -> str:
        return "function"

    assert resolve(function) is function

    legacy = SimpleNamespace(generate=function)
    assert resolve(legacy) is function


@pytest.mark.parametrize("legacy_export", [False, True])
async def test_stream_uses_generate_export_and_full_conversation(legacy_export: bool):
    prompt_utils = _PromptUtils()
    generated: dict[str, Any] = {}

    def generate(*args: Any, **kwargs: Any):
        generated["args"] = args
        generated["kwargs"] = kwargs
        generated["thread"] = threading.current_thread().name
        return SimpleNamespace(text="generated")

    export = SimpleNamespace(generate=generate) if legacy_export else generate
    removed_paths: list[str] = []
    provider = _stream_provider(prompt_utils, export, removed_paths)
    image = _Image()
    messages = [
        _message("system", "rules"),
        _message("user", [_Text("first"), image]),
        _message("assistant", "earlier answer"),
        _message("user", "follow up"),
    ]

    chunks = [
        chunk
        async for chunk in provider._stream_vlm_chat(
            messages,
            "mlx-community/qwen3-vl",
            [image],
            [],
            max_tokens=32,
        )
    ]

    assert chunks[0].content == "generated"
    assert generated["kwargs"]["image"] == ["image.tmp"]
    assert generated["kwargs"]["audio"] is None
    # Generation runs on the dedicated thread the model was loaded on.
    assert generated["thread"].startswith("vlm-test")
    assert "system: rules" in generated["args"][2]
    assert "user: <image>first" in generated["args"][2]
    assert "assistant: earlier answer" in generated["args"][2]
    assert "user: follow up" in generated["args"][2]
    assert generated["args"][2].endswith("assistant:")
    assert removed_paths == ["image.tmp"]


async def test_stream_cleans_assets_when_prompt_formatting_fails():
    class FailingPromptUtils(_PromptUtils):
        def apply_chat_template(self, *args: Any, **kwargs: Any):
            raise ValueError("template failed")

    removed_paths: list[str] = []
    provider = _stream_provider(
        FailingPromptUtils(),
        lambda *args, **kwargs: "unused",
        removed_paths,
        force_assets=True,
    )

    with pytest.raises(ValueError, match="template failed"):
        async for _chunk in provider._stream_vlm_chat(
            [_message("user", "prompt")],
            "mlx-community/qwen3-vl",
            [],
            [],
            max_tokens=32,
        ):
            pass

    assert removed_paths == ["audio.tmp", "image.tmp"]


async def test_stream_cleans_images_when_audio_preparation_fails():
    removed_paths: list[str] = []
    provider = _stream_provider(
        _PromptUtils(),
        lambda *args, **kwargs: "unused",
        removed_paths,
        audio_failure=True,
    )
    image = _Image()
    audio = _Audio()

    with pytest.raises(RuntimeError, match="audio preparation failed"):
        async for _chunk in provider._stream_vlm_chat(
            [_message("user", [_Text("prompt"), image, audio])],
            "mlx-community/qwen3-vl",
            [image],
            [audio],
            max_tokens=32,
        ):
            pass

    assert removed_paths == ["image.tmp"]


def test_vlm_message_conversion_keeps_typed_content_and_metadata():
    provider_type, _resolve, _text, _image, _audio = _helpers()
    provider = provider_type()
    image = _Image()
    audio = _Audio()
    message = _message(
        "user",
        [_Text("look"), image, _Text(" here"), audio],
        name="human",
        tool_call_id="tool-1",
    )

    converted = provider._convert_vlm_message(message, 0)

    assert converted["role"] == "user"
    assert converted["name"] == "human"
    assert converted["tool_call_id"] == "tool-1"
    assert converted["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image"},
        {"type": "text", "text": " here"},
        {"type": "audio"},
    ]


def test_vlm_formatter_templates_each_turn_and_keeps_media_on_its_turn():
    provider_type, _resolve, _text, _image, _audio = _helpers()
    prompt_utils = _PromptUtils()
    provider = provider_type()
    provider_namespace = provider_type._format_vlm_messages.__globals__
    provider_namespace["mlx_vlm"] = SimpleNamespace(prompt_utils=prompt_utils)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "rules"}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "first"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": "second"}, {"type": "audio"}],
        },
    ]

    rendered = provider._format_vlm_messages(
        object(), {"model_type": "qwen3_vl"}, messages
    )

    assert [call["role"] for call in prompt_utils.calls] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [call["num_images"] for call in prompt_utils.calls] == [0, 1, 0, 0]
    assert [call["num_audios"] for call in prompt_utils.calls] == [0, 0, 0, 1]
    assert all(call["return_messages"] for call in prompt_utils.calls)
    assert "system: rules" in rendered
    assert "user: <image>first" in rendered
    assert "assistant: answer" in rendered
    assert "user: second<audio>" in rendered
    assert rendered.endswith("assistant:")


def test_failed_media_is_removed_from_only_its_originating_turn():
    provider_type, _resolve, _text, _image, _audio = _helpers()
    provider = provider_type()
    first_image = _Image()
    second_image = _Image()
    first = _message("user", [first_image, _Text("first")])
    second = _message("user", [second_image, _Text("second")])

    converted = [
        provider._convert_vlm_message(
            first, 0, image_ids={id(second_image)}, audio_ids=set()
        ),
        provider._convert_vlm_message(
            second, 1, image_ids={id(second_image)}, audio_ids=set()
        ),
    ]

    assert converted[0]["content"] == [{"type": "text", "text": "first"}]
    assert converted[1]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "second"},
    ]


def test_prompt_only_models_keep_final_formatted_message():
    provider_type, _resolve, _text, _image, _audio = _helpers()
    prompt_utils = _PromptUtils()
    provider = provider_type()
    provider_type._format_vlm_messages.__globals__["mlx_vlm"] = SimpleNamespace(
        prompt_utils=prompt_utils
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "old"}]},
        {"role": "user", "content": [{"type": "text", "text": "latest"}]},
    ]

    rendered = provider._format_vlm_messages(
        object(), {"model_type": "paligemma"}, messages
    )

    assert rendered == {"role": "user", "content": [{"type": "text", "text": "latest"}]}
