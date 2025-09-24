"""MLX chat provider implementation.

This provider integrates the `mlx-lm` runtime so models exported for MLX can be
used through Nodetool's unified chat provider interface. The implementation
keeps a lazy reference to the MLX runtime because importing `mlx_lm` requires a
Metal capable environment. When the provider is first used we import the
library, load the configured model, and then stream generations through
``stream_generate``.

The provider supports the chat history format used by other providers, basic
streaming, and MLX's tool calling conventions (``<tool_call>`` markers around a
JSON payload). Tool definitions are passed through ``tokenizer.apply_chat_template``
whenever the tokenizer advertises tool calling support.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import dataclass
import time
from typing import Any, AsyncIterator, Callable, Iterable, Sequence
from io import BytesIO
from urllib.parse import urlparse, unquote
import os
import tempfile

from nodetool.chat.providers.base import ChatProvider, register_chat_provider
from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    ImageRef,
    AudioRef,
)
from nodetool.workflows.types import Chunk
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime

import PIL.Image
from pydub import AudioSegment  # type: ignore

log = get_logger(__name__)


DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


@dataclass(slots=True)
class _MLXRuntime:
    """Small container bundling the callables we need from mlx-lm."""

    load: Callable[..., tuple[Any, Any]]
    stream_generate: Callable[..., Iterable[Any]]
    make_sampler: Callable[..., Any] | None = None


@dataclass(slots=True)
class _MLXVLMRuntime:
    """Container bundling callables we need from mlx-vlm for vision models."""

    load: Callable[..., tuple[Any, Any]]
    generate: Callable[..., Any]
    apply_chat_template: Callable[..., str]
    load_config: Callable[..., Any] | None = None


# Simple in-memory TTL cache for loaded MLX models: 5 minutes
_CACHE_TTL_SECONDS: int = 300
_MODEL_CACHE: dict[str, tuple[Any, Any, float]] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Separate cache for VLM models (mlx-vlm)
_VLM_MODEL_CACHE: dict[str, tuple[Any, Any, Any, float]] = {}
_VLM_MODEL_CACHE_LOCK = threading.Lock()


@register_chat_provider(Provider.MLX)
class MLXProvider(ChatProvider):
    """Chat provider backed by the ``mlx-lm`` runtime.

    This provider exposes a standard chat interface that uses the MLX runtime
    to generate responses. It lazily imports the underlying `mlx_lm` library to
    allow running in environments where the library is not available at import
    time. The provider supports token streaming and tool calling conventions used
    by other Nodetool chat providers.
    """

    provider: Provider = Provider.MLX

    def __init__(
        self,
        adapter_path: str | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        sampler_defaults: dict[str, Any] | None = None,
        lazy_load: bool | None = None,
        runtime: _MLXRuntime | None = None,
    ) -> None:
        super().__init__()

        env = Environment.get_environment()
        self.adapter_path = adapter_path or env.get("MLX_ADAPTER_PATH")

        self.lazy_load = _coerce_bool(
            lazy_load if lazy_load is not None else env.get("MLX_LAZY_LOAD", "0")
        )

        self._tokenizer_config = dict(tokenizer_config or {})
        self._sampler_defaults = dict(sampler_defaults or {})

        self._runtime = runtime
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

        # mlx-vlm runtime + cache holders
        self._vlm_runtime: _MLXVLMRuntime | None = None
        self._vlm_model: Any | None = None
        self._vlm_processor: Any | None = None
        self._vlm_config: Any | None = None
        self._vlm_load_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = (),
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Stream a single assistant message, collecting content and tool calls."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async for item in self._stream_chat(
            messages,
            model,
            tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            **kwargs,
        ):
            if isinstance(item, ToolCall):
                tool_calls.append(item)
            elif isinstance(item, Chunk):
                if item.content:
                    content_parts.append(item.content)
                if item.done:
                    break

        message = Message(
            role="assistant",
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls or None,
            provider=self.provider,
            model=model,
        )
        return message

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = (),
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        async for item in self._stream_chat(
            messages,
            model,
            tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            **kwargs,
        ):
            yield item

    def get_context_length(self, model: str) -> int:
        # Most MLX converted instruct models default to 8k context unless
        # otherwise specified.
        return 8192

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool],
        max_tokens: int,
        context_window: int,
        response_format: dict | None,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream chat responses for a given message sequence.

        Internal helper that orchestrates prompting, token streaming, and
        extraction of tool calls from the MLX runtime. Yields `Chunk` and
        `ToolCall` items to the caller.
        """
        # Route to mlx-vlm if the model appears vision-capable and images are present
        image_parts = self._extract_image_parts(messages)
        audio_parts = self._extract_audio_parts(messages)
        if (image_parts and self._is_vision_model(model)) or (
            audio_parts and self._is_audio_model(model)
        ):
            async for item in self._stream_vlm_chat(
                messages,
                model,
                image_parts,
                audio_parts,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield item
            return

        log.debug(
            "MLX _stream_chat start | model=%s messages=%d tools=%d kwargs_keys=%s",
            model,
            len(messages),
            len(tools),
            sorted(list(kwargs.keys())),
        )

        await self._ensure_model_loaded(model)
        assert self._tokenizer is not None
        assert self._model is not None

        converted_messages = [
            self._convert_message(msg, index) for index, msg in enumerate(messages)
        ]
        tool_defs = self._convert_tools(tools)

        prompt = await asyncio.to_thread(
            self._tokenizer.apply_chat_template,
            converted_messages,
            tool_defs or None,
            add_generation_prompt=True,
        )

        runtime = await self._get_runtime()
        runtime_overrides = dict(kwargs)
        sampler = self._build_sampler(runtime, runtime_overrides)
        stream_kwargs = self._build_stream_kwargs(runtime_overrides)
        if sampler is not None:
            stream_kwargs["sampler"] = sampler

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_stream() -> None:
            log.debug(
                "MLX _run_stream thread start | model=%s max_tokens=%s stream_kwargs=%s",
                model,
                max_tokens,
                {
                    k: ("<callable>" if callable(v) else v)
                    for k, v in stream_kwargs.items()
                },
            )
            try:
                for response in runtime.stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    **stream_kwargs,
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("response", response)), loop
                    ).result()
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("MLX _run_stream thread error: %s", exc)
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", exc)), loop
                ).result()
            finally:
                log.debug("MLX _run_stream thread done")
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", None)), loop
                ).result()

        threading.Thread(target=_run_stream, daemon=True).start()

        tool_state = {
            "buffer": "",
            "in_tool_call": False,
            "counter": 0,
        }
        done_emitted = False

        # Watchdog to log if no items arrive for too long
        last_activity = time.monotonic()
        watchdog_seconds = int(os.getenv("MLX_WATCHDOG_SECS", "60"))

        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=5.0)
                last_activity = time.monotonic()
            except asyncio.TimeoutError:
                if (
                    watchdog_seconds > 0
                    and (time.monotonic() - last_activity) > watchdog_seconds
                ):
                    log.warning(
                        "MLX _stream_chat watchdog tripped | no-activity-for=%ss model=%s in_tool_call=%s",
                        int(time.monotonic() - last_activity),
                        model,
                        tool_state["in_tool_call"],
                    )
                    # Continue waiting; we only log to surface the hang
                continue
            if kind == "response":
                response = payload
                is_final = getattr(response, "finish_reason", None) is not None
                if is_final:
                    self._update_usage(response)

                segments, parsed_calls = self._process_response_text(
                    response.text, tool_state
                )
                for tool_call in parsed_calls:
                    log.debug(
                        "MLX parsed tool_call name=%s id=%s",
                        tool_call.name,
                        tool_call.id,
                    )
                    yield tool_call

                for i, segment in enumerate(segments):
                    if not segment:
                        continue
                    done_flag = (
                        is_final
                        and i == len(segments) - 1
                        and not tool_state["in_tool_call"]
                    )
                    if done_flag:
                        log.debug("MLX emitting final chunk (done=True)")
                    yield Chunk(content=segment, done=done_flag)
                    done_emitted = done_emitted or done_flag

                if is_final:
                    if not done_emitted:
                        log.debug("MLX emitting trailing done chunk")
                        yield Chunk(content="", done=True)
                    break
            elif kind == "error":
                log.error("MLX _stream_chat received error from thread: %s", payload)
                raise payload
            elif kind == "done":
                if not done_emitted:
                    log.debug(
                        "MLX done without explicit final token; emitting done chunk"
                    )
                    yield Chunk(content="", done=True)
                break
        log.debug("MLX _stream_chat end | model=%s", model)

    async def _ensure_model_loaded(self, model: str) -> None:
        async with self._load_lock:
            cached = self._get_cached_model(model)
            if cached is not None:
                self._model, self._tokenizer = cached
                return

            runtime = await self._get_runtime()

            def _load() -> tuple[Any, Any]:
                return runtime.load(
                    model,
                    tokenizer_config=self._tokenizer_config,
                    adapter_path=self.adapter_path,
                    lazy=self.lazy_load,
                )

            self._model, self._tokenizer = await asyncio.to_thread(_load)
            self._set_cached_model(model, self._model, self._tokenizer)
            log.info("Loaded MLX model %s", model)

    async def _get_runtime(self) -> _MLXRuntime:
        if self._runtime is not None:
            return self._runtime

        def _import_runtime() -> _MLXRuntime:
            try:
                import importlib

                mlx_module = importlib.import_module("mlx_lm")
                sample_utils = importlib.import_module("mlx_lm.sample_utils")
                return _MLXRuntime(
                    load=mlx_module.load,
                    stream_generate=mlx_module.stream_generate,
                    make_sampler=getattr(sample_utils, "make_sampler", None),
                )
            except Exception as exc:  # pragma: no cover - import failure
                raise RuntimeError(
                    "Install the nodetool huggingface pack using the nodetool package manager."
                ) from exc

        self._runtime = await asyncio.to_thread(_import_runtime)
        return self._runtime

    # ------------------------------------------------------------------
    # mlx-vlm runtime and flow (vision models)
    # ------------------------------------------------------------------
    def _cache_key_vlm(self, model: str) -> str:
        adapter = self.adapter_path or ""
        lazy_flag = "1" if self.lazy_load else "0"
        return f"vlm|{model}|{adapter}|lazy={lazy_flag}"

    def _get_cached_vlm_model(self, model: str) -> tuple[Any, Any, Any] | None:
        key = self._cache_key_vlm(model)
        now = time.monotonic()
        with _VLM_MODEL_CACHE_LOCK:
            entry = _VLM_MODEL_CACHE.get(key)
            if not entry:
                return None
            mdl, proc, cfg, expires_at = entry
            if expires_at < now:
                _VLM_MODEL_CACHE.pop(key, None)
                return None
            return mdl, proc, cfg

    def _set_cached_vlm_model(self, model: str, mdl: Any, proc: Any, cfg: Any) -> None:
        key = self._cache_key_vlm(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _VLM_MODEL_CACHE_LOCK:
            _VLM_MODEL_CACHE[key] = (mdl, proc, cfg, expires_at)

    async def _get_vlm_runtime(self) -> _MLXVLMRuntime:
        if self._vlm_runtime is not None:
            return self._vlm_runtime

        def _import_vlm_runtime() -> _MLXVLMRuntime:
            try:
                import importlib

                vlm_module = importlib.import_module("mlx_vlm")
                prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")
                utils_module = importlib.import_module("mlx_vlm.utils")

                return _MLXVLMRuntime(
                    load=getattr(vlm_module, "load"),
                    generate=getattr(vlm_module, "generate"),
                    apply_chat_template=getattr(prompt_utils, "apply_chat_template"),
                    load_config=getattr(utils_module, "load_config", None),
                )
            except Exception as exc:  # pragma: no cover - import failure
                raise RuntimeError(
                    "Install the nodetool huggingface pack using the nodetool package manager."
                ) from exc

        self._vlm_runtime = await asyncio.to_thread(_import_vlm_runtime)
        return self._vlm_runtime

    async def _ensure_vlm_model_loaded(self, model: str) -> None:
        async with self._vlm_load_lock:
            cached = self._get_cached_vlm_model(model)
            if cached is not None:
                self._vlm_model, self._vlm_processor, self._vlm_config = cached
                return

            runtime = await self._get_vlm_runtime()

            def _load() -> tuple[Any, Any, Any]:
                mdl, proc = runtime.load(model)
                cfg = getattr(mdl, "config", None)
                if cfg is None and runtime.load_config is not None:
                    cfg = runtime.load_config(model)
                # proc.image_processor.patch_size = 14
                return mdl, proc, cfg

            self._vlm_model, self._vlm_processor, self._vlm_config = (
                await asyncio.to_thread(_load)
            )
            self._set_cached_vlm_model(
                model, self._vlm_model, self._vlm_processor, self._vlm_config
            )
            log.info("Loaded MLX-VLM model %s", model)

    def _is_vision_model(self, model: str) -> bool:
        name = (model or "").lower()
        keywords = (
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen-vl",
            "llava",
            "idefics",
            "vl-",
            "gemma-3n",
        )
        return any(k in name for k in keywords)

    def _is_audio_model(self, model: str) -> bool:
        name = (model or "").lower()
        keywords = (
            "gemma-3n-e2b",
            "e2b-it",
            "audio",
        )
        # be permissive if both image/audio present and model is vision
        return any(k in name for k in keywords) or self._is_vision_model(model)

    def _ensure_vlm_processor_ready(self) -> None:
        """Ensure mlx-vlm processor has necessary attributes set.

        Some processors (e.g., LLaVA) expect a non-None patch_size. If missing,
        attempt to infer from model config or fall back to a sane default (14).
        """
        proc = self._vlm_processor
        cfg = self._vlm_config
        if proc is None:
            return
        image_proc = getattr(proc, "image_processor", None)
        if image_proc is None:
            return
        patch_size = getattr(image_proc, "patch_size", None)
        if patch_size is None:
            inferred = None
            if cfg is not None:
                vision_cfg = getattr(cfg, "vision_config", None)
                inferred = getattr(vision_cfg, "patch_size", None)
            try:
                if inferred is None:
                    inferred = 14
                setattr(image_proc, "patch_size", int(inferred))
            except Exception:
                # Last resort default
                try:
                    setattr(image_proc, "patch_size", 14)
                except Exception:
                    pass

    def _extract_text_from_content(
        self, content: str | list[MessageContent] | None
    ) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        texts: list[str] = []
        for part in content:
            if isinstance(part, MessageTextContent):
                if part.text:
                    texts.append(part.text)
        return "".join(texts)

    def _extract_last_user_prompt(self, messages: Sequence[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                text = self._extract_text_from_content(msg.content)
                if text:
                    return text
        # Fallback: concatenate all text parts
        parts: list[str] = []
        for msg in messages:
            t = self._extract_text_from_content(msg.content)
            if t:
                parts.append(t)
        return "\n".join(parts)

    def _extract_image_parts(
        self, messages: Sequence[Message]
    ) -> list[MessageImageContent]:
        images: list[MessageImageContent] = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, MessageImageContent):
                        images.append(part)
        return images

    def _extract_audio_parts(
        self, messages: Sequence[Message]
    ) -> list[MessageAudioContent]:
        audios: list[MessageAudioContent] = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, MessageAudioContent):
                        audios.append(part)
        return audios

    async def _prepare_vlm_images(self, parts: list[MessageImageContent]) -> list[str]:
        """Load images as PIL.Image objects in memory (no temp files)."""
        prepared_images: list[str] = []
        for part in parts:
            image_ref: ImageRef = part.image
            uri = image_ref.uri or ""
            data: bytes | None = None
            if uri:
                try:
                    _mime, fetched = await fetch_uri_bytes_and_mime(uri)
                    data = fetched
                except Exception:
                    try:
                        parsed = urlparse(uri)
                        local_path = (
                            unquote(parsed.path) if parsed.scheme == "file" else uri
                        )
                        with open(local_path, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None
            if data is None and image_ref.data:
                data = image_ref.data

            if data is None:
                continue

            try:
                bytes_io = BytesIO()
                img = PIL.Image.open(BytesIO(data))
                img = img.convert("RGB")
                img = img.resize((224, 224))
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp, format="PNG")
                prepared_images.append(tmp.name)
            except Exception:
                pass

        return prepared_images

    async def _prepare_vlm_audio(self, parts: list[MessageAudioContent]) -> list[str]:
        """Download/convert audios and persist as temporary WAV files.

        Returns (audio_paths, temp_files) where both lists contain filesystem paths.
        All returned paths in audio_paths are WAV files when possible.
        """
        prepared_paths: list[str] = []
        for part in parts:
            audio_ref: AudioRef = part.audio
            uri = audio_ref.uri or ""
            data: bytes | None = None
            if uri:
                try:
                    _mime, fetched = await fetch_uri_bytes_and_mime(uri)
                    data = fetched
                except Exception:
                    try:
                        parsed = urlparse(uri)
                        local_path = (
                            unquote(parsed.path) if parsed.scheme == "file" else uri
                        )
                        with open(local_path, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None
            if data is None and audio_ref.data:
                data = audio_ref.data

            if data is None:
                continue

            audio_seg = AudioSegment.from_file(BytesIO(data))
            audio_seg = (
                audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_seg.export(tmp, format="wav")
            prepared_paths.append(tmp.name)

        return prepared_paths

    async def _stream_vlm_chat(
        self,
        messages: Sequence[Message],
        model: str,
        image_parts: list[MessageImageContent],
        audio_parts: list[MessageAudioContent],
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        log.debug(
            "MLX-VLM _stream_vlm_chat start | model=%s images=%d audios=%d",
            model,
            len(image_parts),
            len(audio_parts),
        )
        await self._ensure_vlm_model_loaded(model)
        assert self._vlm_model is not None
        assert self._vlm_processor is not None
        assert self._vlm_config is not None

        prompt_text = self._extract_last_user_prompt(messages)
        images = await self._prepare_vlm_images(image_parts)
        audios = await self._prepare_vlm_audio(audio_parts)
        log.debug(
            "MLX-VLM prepared assets | images=%d audios=%d", len(images), len(audios)
        )

        runtime = await self._get_vlm_runtime()

        # Defensive processor normalization
        self._ensure_vlm_processor_ready()

        formatted_prompt = await asyncio.to_thread(
            runtime.apply_chat_template,
            self._vlm_processor,
            self._vlm_config,
            prompt_text,
            num_images=len(images),
            num_audios=len(audios),
        )
        log.debug(
            "MLX-VLM prompt prepared | prompt_len=%d",
            len(formatted_prompt) if isinstance(formatted_prompt, str) else -1,
        )

        def _run_generate() -> str:
            # Keep params conservative; mlx-vlm's generate may accept more kwargs
            result = runtime.generate(
                self._vlm_model,
                self._vlm_processor,
                formatted_prompt,
                image=images if images else None,
                audio=audios if audios else None,
                verbose=False,
                max_tokens=max_tokens,
            )
            # Some versions may return objects; coerce to string
            return result.text

        try:
            output: str = await asyncio.to_thread(_run_generate)
        except Exception as exc:
            log.exception("MLX-VLM generation error: %s", exc)
            raise RuntimeError(f"mlx-vlm generation failed: {exc}")

        log.debug(
            "MLX-VLM generation ok | output_len=%d",
            len(output) if isinstance(output, str) else -1,
        )
        yield Chunk(content=output, done=True)

        for audio_path in audios:
            try:
                os.remove(audio_path)
            except Exception:
                pass

        for image_path in images:
            try:
                os.remove(image_path)
            except Exception:
                pass
        log.debug("MLX-VLM _stream_vlm_chat end | model=%s", model)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------
    def _cache_key(self, model: str) -> str:
        adapter = self.adapter_path or ""
        lazy_flag = "1" if self.lazy_load else "0"
        return f"{model}|{adapter}|lazy={lazy_flag}"

    def _get_cached_model(self, model: str) -> tuple[Any, Any] | None:
        key = self._cache_key(model)
        now = time.monotonic()
        with _MODEL_CACHE_LOCK:
            entry = _MODEL_CACHE.get(key)
            if not entry:
                return None
            mdl, tok, expires_at = entry
            if expires_at < now:
                # Expired; evict
                _MODEL_CACHE.pop(key, None)
                return None
            return mdl, tok

    def _set_cached_model(self, model: str, mdl: Any, tok: Any) -> None:
        key = self._cache_key(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = (mdl, tok, expires_at)

    def _build_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        stream_kwargs = dict(kwargs)
        for key in (
            "temperature",
            "temp",
            "top_p",
            "top_k",
            "min_p",
            "min_tokens_to_keep",
            "xtc_probability",
            "xtc_threshold",
        ):
            stream_kwargs.pop(key, None)
        return stream_kwargs

    def _build_sampler(
        self, runtime: _MLXRuntime, kwargs: dict[str, Any]
    ) -> Any | None:
        if runtime.make_sampler is None:
            return None

        sampler_params = {
            "temp": self._sampler_defaults.get("temp", 0.0),
            "top_p": self._sampler_defaults.get("top_p", 0.0),
            "min_p": self._sampler_defaults.get("min_p", 0.0),
            "min_tokens_to_keep": self._sampler_defaults.get("min_tokens_to_keep", 1),
            "top_k": self._sampler_defaults.get("top_k", 0),
            "xtc_probability": self._sampler_defaults.get("xtc_probability", 0.0),
            "xtc_threshold": self._sampler_defaults.get("xtc_threshold", 0.0),
        }

        overrides = {
            "temp": kwargs.pop("temperature", None),
            "top_p": kwargs.pop("top_p", None),
            "top_k": kwargs.pop("top_k", None),
            "min_p": kwargs.pop("min_p", None),
            "min_tokens_to_keep": kwargs.pop("min_tokens_to_keep", None),
            "xtc_probability": kwargs.pop("xtc_probability", None),
            "xtc_threshold": kwargs.pop("xtc_threshold", None),
        }
        for key, value in overrides.items():
            if value is not None:
                sampler_params[key] = value

        # When temperature is explicitly zero we avoid allocating a sampler.
        if (
            sampler_params.get("temp", 0.0) == 0
            and sampler_params.get("top_p", 0.0) == 0.0
        ):
            return None

        return runtime.make_sampler(**sampler_params)

    def _update_usage(self, response: Any) -> None:
        prompt_tokens = int(getattr(response, "prompt_tokens", 0))
        completion_tokens = int(getattr(response, "generation_tokens", 0))
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens

    def _convert_message(self, message: Message, index: int) -> dict[str, Any]:
        content = self._normalize_content(message.content)
        payload: dict[str, Any] = {
            "role": message.role or "user",
            "content": content,
        }

        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id or f"call_{index}_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args or {}),
                    },
                }
                for i, tool_call in enumerate(message.tool_calls)
            ]

        return payload

    def _convert_tools(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        tool_defs: list[dict[str, Any]] = []
        for tool in tools:
            try:
                tool_defs.append(tool.tool_param())
            except Exception as exc:
                log.warning(
                    "Failed to convert tool %s: %s", getattr(tool, "name", tool), exc
                )
        return tool_defs

    def _normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content)

    def _process_response_text(
        self,
        text: str,
        tool_state: dict[str, Any],
    ) -> tuple[list[str], list[ToolCall]]:
        if not text:
            return [], []

        if not getattr(self._tokenizer, "has_tool_calling", False):
            return [text], []

        start_token = getattr(self._tokenizer, "tool_call_start", None)
        end_token = getattr(self._tokenizer, "tool_call_end", None)
        if not start_token or not end_token:
            return [text], []
        segments: list[str] = []
        tool_calls: list[ToolCall] = []
        remaining = text

        while remaining:
            if tool_state["in_tool_call"]:
                end_index = remaining.find(end_token)
                if end_index == -1:
                    tool_state["buffer"] += remaining
                    remaining = ""
                else:
                    tool_state["buffer"] += remaining[:end_index]
                    call = self._parse_tool_call(tool_state)
                    if call is not None:
                        tool_calls.append(call)
                    tool_state["in_tool_call"] = False
                    tool_state["buffer"] = ""
                    remaining = remaining[end_index + len(end_token) :]
            else:
                start_index = remaining.find(start_token)
                if start_index == -1:
                    segments.append(remaining)
                    remaining = ""
                else:
                    prefix = remaining[:start_index]
                    if prefix:
                        segments.append(prefix)
                    tool_state["in_tool_call"] = True
                    remaining = remaining[start_index + len(start_token) :]

        return segments, tool_calls

    def _parse_tool_call(self, tool_state: dict[str, Any]) -> ToolCall | None:
        payload = tool_state["buffer"].strip()
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Failed to decode tool call payload: %s", payload)
            return None

        function = parsed.get("function") or {}
        name = parsed.get("name") or function.get("name") or ""
        if not name:
            log.warning("Tool call missing name: %s", payload)
            return None

        args = parsed.get("arguments") or function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        call_id = parsed.get("id") or parsed.get("tool_call_id")
        if not call_id:
            call_id = f"mlx_tool_{tool_state['counter']}"
        tool_state["counter"] += 1

        return ToolCall(
            id=call_id,
            name=name,
            args=args if isinstance(args, dict) else {"value": args},
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


async def main() -> None:
    """Run a simple demo of the MLX provider.

    - Basic generation: ask the model to say hello
    - Optional tool-call round trip with a trivial echo tool
    """
    from nodetool.workflows.processing_context import ProcessingContext

    class EchoTool(Tool):
        name = "echo"
        description = "Echo back the provided text."
        input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        async def process(  # type: ignore[override]
            self, context: ProcessingContext, params: dict[str, Any]
        ) -> Any:
            return {"echo": params.get("text", "")}

    provider = MLXProvider()
    model_name = "Qwen/Qwen3-4B-MLX-4bit"
    content_parts = []

    # Basic, non-tool generation
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello from MLX' in one short sentence."),
    ]

    async for item in provider.generate_messages(
        messages=messages, model=model_name, max_tokens=4096
    ):
        if isinstance(item, Chunk):
            if item.content:
                content_parts.append(item.content)
            if item.done:
                break
    print("".join(content_parts))
    # Optional: demonstrate a simple tool call round trip
    tools: list[Tool] = [EchoTool()]
    context = ProcessingContext()
    messages = [
        Message(
            role="user",
            content="Use the echo tool with text 'mlx', then say done.",
        ),
    ]

    try:
        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        async for item in provider.generate_messages(
            messages=messages, model=model_name, tools=tools, max_tokens=4096
        ):
            if isinstance(item, ToolCall):
                tool_calls.append(item)
            elif isinstance(item, Chunk):
                if item.content:
                    content_parts.append(item.content)
                if item.done:
                    break
        if tool_calls:
            for tc in tool_calls:
                tool = next((t for t in tools if t.name == tc.name), None)
                if tool is None:
                    continue
                print(f"Processing tool call: {tc.name}")
                print(tc.args)
                result = await tool.process(context, tc.args or {})
                print(f"Tool result: {result}")
                messages.append(
                    Message(
                        role="tool",
                        name=tool.name,
                        tool_call_id=tc.id,
                        content=json.dumps(result),
                    )
                )

            print(messages)

            async for item in provider.generate_messages(
                messages=messages, model=model_name, tools=tools, max_tokens=4096
            ):
                if isinstance(item, Chunk):
                    if item.content:
                        print(item.content, end="", flush=True)
                    if item.done:
                        print()
                        break
        else:
            print("No tool calls returned by the model.")
    except Exception as e:  # pragma: no cover - demo convenience
        print("Tool call demo failed:", e)


if __name__ == "__main__":
    asyncio.run(main())
