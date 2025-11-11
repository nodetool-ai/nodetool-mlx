"""MLX unified provider implementation.

This provider integrates multiple MLX libraries to provide a complete multi-modal
AI provider for Apple Silicon:

- mlx-lm: Language models for chat and text generation
- mlx-vlm: Vision-language models for multimodal understanding
- mlx-audio: Text-to-speech audio generation
- mflux: FLUX models for image generation

The implementation keeps lazy references to the MLX runtimes because importing these
libraries requires a Metal-capable environment. When a capability is first used, we
import the required library, load the configured model, and execute the task.

The provider supports:
- Chat with tool calling conventions
- Vision and audio understanding
- Text-to-speech generation
- Text-to-image and image-to-image generation
"""

import ast
import asyncio
import base64
import json
import logging
import inspect
from pathlib import Path
import threading
from dataclasses import dataclass
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterable,
    List,
    Sequence,
)
from io import BytesIO
from urllib.parse import urlparse, unquote
import os
import sys
import tempfile
from nodetool.media.audio.audio_helpers import convert_audio_to_standard_format
import numpy as np
from huggingface_hub import CacheNotFound, scan_cache_dir
from mlx import nn
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx_vlm
import mlx_audio
import mlx_lm.sample_utils
import mlx_vlm.prompt_utils
import mlx_vlm.utils
import mlx_audio.tts.utils
import mlx_audio.tts.generate
# mflux imports are lazy - only imported when image generation is used
# This allows the MLX provider to be registered even if mflux isn't installed

from nodetool.providers.base import BaseProvider, register_provider
from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.metadata.types import (
    ASRModel,
    Message,
    Provider,
    ToolCall,
    TTSModel,
    ASRModel,
    MessageContent,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    ImageRef,
    AudioRef,
    LanguageModel,
    ImageModel,
)
from nodetool.workflows.types import Chunk
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.workflows.processing_context import ProcessingContext

import PIL.Image
from pydub import AudioSegment  # type: ignore
from nodetool.integrations.huggingface.huggingface_models import (
    get_mlx_language_models_from_hf_cache,
)
from nodetool.mlx.flux_model_loader import (
    load_flux_model,
)
from mflux.config.config import Config


log = get_logger(__name__)
log.setLevel(logging.DEBUG)


DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


# Simple in-memory TTL cache for loaded MLX models: 5 minutes
_CACHE_TTL_SECONDS: int = 300
_MODEL_CACHE: dict[str, tuple[nn.Module, TokenizerWrapper, float]] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Separate cache for VLM models (mlx-vlm)
_VLM_MODEL_CACHE: dict[str, tuple[nn.Module, mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, Any, float]] = {}
_VLM_MODEL_CACHE_LOCK = threading.Lock()

# Separate cache for TTS models (mlx-audio)
_TTS_MODEL_CACHE: dict[str, tuple[Any, float]] = {}
_TTS_MODEL_CACHE_LOCK = threading.Lock()


@register_provider(Provider.MLX)
class MLXProvider(BaseProvider):
    """Unified multi-modal provider for Apple Silicon using MLX.

    This provider exposes a complete AI interface leveraging multiple MLX libraries:
    - Language models (mlx-lm) for chat and text generation
    - Vision models (mlx-vlm) for multimodal understanding
    - Audio models (mlx-audio) for text-to-speech
    - Image models (mflux) for FLUX-based image generation

    All libraries are lazily imported to allow running in environments where they
    are not available at import time. The provider supports streaming, tool calling,
    and all standard Nodetool provider capabilities.
    """

    provider: Provider = Provider.MLX

    def __init__(
        self,
        secrets: dict[str, str] = {},
        adapter_path: str | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        sampler_defaults: dict[str, Any] | None = None,
        lazy_load: bool | None = None,
    ) -> None:
        super().__init__(secrets=secrets)

        env = Environment.get_environment()
        self.adapter_path = adapter_path or env.get("MLX_ADAPTER_PATH")

        self._load_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()  # Serialize generation calls

        self._vlm_load_lock = asyncio.Lock()
        self._vlm_generation_lock = asyncio.Lock()  # Serialize VLM generation calls

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


    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available MLX models.

        Returns MLX-converted models available in the local HuggingFace cache.
        Always returns models (doesn't check if MLX is available).

        Returns:
            List of LanguageModel instances for MLX
        """
        models = await get_mlx_language_models_from_hf_cache()
        log.debug(f"Found {len(models)} MLX models in HF cache")
        return models

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available MLX image models.

        Returns recommended FLUX models from MFlux nodes that can be used
        for local image generation on Apple Silicon. Also discovers mflux models
        from the HuggingFace cache.

        Returns:
            List of ImageModel instances for MLX
        """
        # Start with hardcoded recommended FLUX models from MFlux nodes (mflux.py)
        models = [
            # From MFlux.get_recommended_models()
            ImageModel(
                id="Freepik/flux.1-lite-8B-alpha",
                name="FLUX.1 Lite 8B Alpha",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
                name="FLUX.1 Schnell 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="dhairyashil/FLUX.1-dev-mflux-4bit",
                name="FLUX.1 Dev 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="filipstrand/FLUX.1-Krea-dev-mflux-4bit",
                name="FLUX.1 Krea Dev 4-bit",
                provider=Provider.MLX,
            ),
            ImageModel(
                id="akx/FLUX.1-Kontext-dev-mflux-4bit",
                name="FLUX.1 Kontext Dev 4-bit",
                provider=Provider.MLX,
            ),
        ]

        # Also discover mflux models from HuggingFace cache
        try:
            from nodetool.integrations.huggingface.huggingface_models import (
                get_mlx_image_models_from_hf_cache,
            )
            cached_models = await get_mlx_image_models_from_hf_cache()
            # Add cached models, avoiding duplicates
            seen_ids = {m.id for m in models}
            for cached_model in cached_models:
                if cached_model.id not in seen_ids:
                    models.append(cached_model)
                    seen_ids.add(cached_model.id)
            log.debug(f"Discovered {len(cached_models)} mflux models from HF cache")
        except Exception as e:
            log.debug(f"Could not discover mflux models from cache: {e}")

        return models

    async def get_available_asr_models(self) -> List[ASRModel]:
        """
        Get available MLX ASR models.
        """
        models = [
            ASRModel(
                id="mlx-community/whisper-tiny-mlx",
                name="Whisper Tiny",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-base-mlx",
                name="Whisper Base",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-base.en-mlx",
                name="Whisper Base EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-small-mlx",
                name="Whisper Small",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-small.en-mlx",
                name="Whisper Small EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-medium-mlx",
                name="Whisper Medium",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-medium.en-mlx",
                name="Whisper Medium EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-large-v3-mlx",
                name="Whisper Large V3",
                provider=Provider.MLX,
            ),
        ]
        return models

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using MLX Whisper checkpoints."""
        _ = context  # Context not currently used but kept for interface compatibility.
        if sys.platform != "darwin":
            raise RuntimeError("MLX automatic speech recognition requires macOS")

        if not audio:
            raise ValueError("audio must not be empty")

        # Whisper expects 16 kHz mono audio normalized to [-1, 1]
        samples = convert_audio_to_standard_format(audio, target_sample_rate=16_000)
        if samples.size == 0:
            return ""
        audio_float = (samples.astype(np.float32) / 32768.0).flatten()
        audio_float = np.ascontiguousarray(audio_float, dtype=np.float32)

        # Ensure the model is present in the local Hugging Face cache
        if self._resolve_cached_repo_path(model) is None:
            raise ValueError(
                f"Model {model} must be downloaded first (not found in cache)"
            )

        log.debug(
            "Transcribing audio with MLX Whisper model=%s language=%s temperature=%s",
            model,
            language,
            temperature,
        )

        def _run_transcription() -> dict[str, Any] | str | None:
            import mlx_whisper

            transcribe_fn = mlx_whisper.transcribe
            sig_params = set(inspect.signature(transcribe_fn).parameters)

            call_kwargs: dict[str, Any] = {"path_or_hf_repo": model}

            # Standard Whisper options with sensible defaults
            default_options: dict[str, Any] = {
                "compression_ratio_threshold": kwargs.get(
                    "compression_ratio_threshold", 2.4
                ),
                "logprob_threshold": kwargs.get("logprob_threshold", -1.0),
                "no_speech_threshold": kwargs.get("no_speech_threshold", 0.6),
                "condition_on_previous_text": kwargs.get(
                    "condition_on_previous_text", True
                ),
                "word_timestamps": kwargs.get("word_timestamps", False),
                "temperature": kwargs.get("temperature", temperature),
            }

            optional_options: dict[str, Any] = {
                "language": kwargs.get("language", language),
                "initial_prompt": kwargs.get("initial_prompt", prompt),
            }

            # Include any additional keyword arguments provided by the caller
            extra_options = {
                key: value
                for key, value in kwargs.items()
                if key not in default_options and key not in optional_options
            }

            for options in (default_options, optional_options, extra_options):
                for key, value in options.items():
                    if value is not None and key in sig_params:
                        call_kwargs[key] = value

            return transcribe_fn(audio_float, **call_kwargs)

        transcribe_task = asyncio.to_thread(_run_transcription)
        try:
            if timeout_s is not None:
                result = await asyncio.wait_for(transcribe_task, timeout=timeout_s)
            else:
                result = await transcribe_task
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MLX automatic speech recognition timed out after {timeout_s} seconds"
            ) from exc
        except Exception as exc:
            log.error("MLX Whisper transcription failed: %s", exc)
            raise RuntimeError(f"MLX Whisper transcription failed: {exc}") from exc

        text: str
        if isinstance(result, dict):
            text = str(result.get("text", "") or "")
        elif isinstance(result, str):
            text = result
        else:
            text = "" if result is None else str(result)

        log.debug("MLX Whisper transcription complete", extra={"length": len(text)})
        return text

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available MLX TTS models.

        Returns TTS models from mlx-audio nodes that can be used for
        local text-to-speech generation on Apple Silicon.

        Returns:
            List of TTSModel instances for MLX TTS
        """
        # Kokoro TTS models (from KokoroTTS.Model enum)
        kokoro_models = [
            TTSModel(
                id="prince-canuma/Kokoro-82M",
                name="Kokoro TTS 82M",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-bf16",
                name="Kokoro TTS 82M BF16",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-4bit",
                name="Kokoro TTS 82M 4-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-6bit",
                name="Kokoro TTS 82M 6-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-8bit",
                name="Kokoro TTS 82M 8-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
        ]

        # Sesame TTS models (from SesameTTS.Model enum) - voice cloning models
        sesame_models = [
            TTSModel(
                id="mlx-community/csm-1b",
                name="Sesame TTS 1B",
                provider=Provider.MLX,
                voices=[],  # Voice cloning - uses reference audio
            ),
            TTSModel(
                id="mlx-community/csm-1b-8bit",
                name="Sesame TTS 1B 8-bit",
                provider=Provider.MLX,
                voices=[],  # Voice cloning - uses reference audio
            ),
        ]

        # Spark TTS models (from SparkTTS.Model enum)
        spark_models = [
            TTSModel(
                id="mlx-community/Spark-TTS-0.5B-bf16",
                name="Spark TTS 0.5B BF16",
                provider=Provider.MLX,
                voices=[],  # Spark uses speed/pitch/gender presets instead of voices
            ),
            TTSModel(
                id="mlx-community/Spark-TTS-0.5B-8bit",
                name="Spark TTS 0.5B 8-bit",
                provider=Provider.MLX,
                voices=[],  # Spark uses speed/pitch/gender presets instead of voices
            ),
        ]

        return kokoro_models + sesame_models + spark_models

    # ------------------------------------------------------------------
    # Tool emulation helpers
    # ------------------------------------------------------------------
    def _format_tools_as_python(self, tools: Sequence[Tool]) -> str:
        """Format tools as Python function definitions for emulation.

        Args:
            tools: Sequence of tools to format.

        Returns:
            String containing Python function definitions.
        """
        log.debug(f"Formatting {len(tools)} tools as Python functions for emulation")

        function_defs = []
        for tool in tools:
            tool_param = tool.tool_param()
            func = tool_param.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # Build simplified function signature
            params = []
            if "properties" in parameters:
                for param_name in parameters["properties"].keys():
                    params.append(param_name)

            params_str = ", ".join(params) if params else ""
            func_def = f"# {name}({params_str})\n# {description}"
            if params:
                func_def += f"\n# Example: {name}({params[0]}='...')"
            function_defs.append(func_def)

        return "\n\n".join(function_defs)

    def _parse_function_calls(
        self, text: str, tools: Sequence[Tool] | None = None
    ) -> list[ToolCall]:
        """Parse Python function calls from text using AST parsing.

        Args:
            text: Text containing potential function calls.
            tools: Optional list of tools to map positional args to named parameters.

        Returns:
            List of parsed ToolCall objects.
        """
        if not text or not text.strip():
            return []

        tool_calls: list[ToolCall] = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try to parse as a function call
            try:
                tree = ast.parse(line, mode="eval")
                if isinstance(tree.body, ast.Call):
                    call_node = tree.body
                    func_name = self._extract_function_name(call_node.func)

                    if func_name:
                        args = self._extract_call_arguments(call_node, func_name, tools)
                        tool_call = ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=func_name,
                            args=args,
                        )
                        tool_calls.append(tool_call)
                        log.debug(f"Parsed tool call: {func_name} with args {args}")
            except (SyntaxError, ValueError):
                # Not a valid function call, skip
                continue

        return tool_calls

    def _extract_function_name(self, node: ast.expr) -> str | None:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_call_arguments(
        self, call_node: ast.Call, func_name: str, tools: Sequence[Tool] | None
    ) -> dict[str, Any]:
        """Extract arguments from a function call AST node."""
        args: dict[str, Any] = {}

        # Extract keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg:
                value = self._ast_to_value(keyword.value)
                args[keyword.arg] = value

        # Handle positional arguments by mapping to parameter names
        if call_node.args and tools:
            matching_tool = next((t for t in tools if t.name == func_name), None)
            if matching_tool:
                tool_param = matching_tool.tool_param()
                func_schema = tool_param.get("function", {})
                parameters = func_schema.get("parameters", {})
                properties = parameters.get("properties", {})
                param_names = list(properties.keys())

                for i, arg in enumerate(call_node.args):
                    if i < len(param_names):
                        param_name = param_names[i]
                        value = self._ast_to_value(arg)
                        args[param_name] = value

        return args

    def _ast_to_value(self, node: ast.expr) -> Any:
        """Convert AST node to Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.List):
            return [self._ast_to_value(el) for el in node.elts]
        elif isinstance(node, ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                if k is not None:  # Skip None keys
                    key = self._ast_to_value(k)
                    value = self._ast_to_value(v)
                    if key is not None:
                        result[key] = value
            return result
        elif isinstance(node, ast.NameConstant):  # Python 3.7 compatibility
            return node.value
        elif isinstance(node, (ast.Name, ast.Attribute)):
            # For variable references, try to return the name as a string
            return ast.unparse(node) if hasattr(ast, "unparse") else str(node)
        return None

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

        mdl, tokenizer = await self._load_model(model)

        # Determine if we need tool emulation
        use_tool_emulation = len(tools) > 0 and not tokenizer.has_tool_calling
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        # Prepare messages with tool emulation if needed
        if use_tool_emulation and len(tools) > 0:
            tool_definitions = self._format_tools_as_python(tools)
            tool_instruction = (
                "\n\n=== AVAILABLE FUNCTIONS ===\n"
                "You can call these functions by writing a function call on a single line.\n"
                "DO NOT write function definitions - only write function CALLS.\n\n"
                f"{tool_definitions}\n\n"
                "=== INSTRUCTIONS ===\n"
                "When you need to use a function:\n"
                "1. Write ONLY the function call, nothing else\n"
                "2. Use this exact format: function_name(param='value')\n"
                "3. Do NOT write 'def', 'return', or any other Python keywords\n"
                "4. After calling a function, wait for the result\n"
                "5. Once you receive a function result, use it in your final answer\n"
                "6. Do NOT call the same function twice"
            )

            # Inject tool instructions into system message or prepend to first user message
            messages_list = list(messages)
            if messages_list and messages_list[0].role == "system":
                # Append to existing system message
                existing_content = messages_list[0].content or ""
                messages_list[0] = Message(
                    role="system", content=f"{existing_content}{tool_instruction}"
                )
            else:
                # Prepend new system message
                messages_list.insert(
                    0, Message(role="system", content=tool_instruction)
                )

            converted_messages = [
                self._convert_message(msg, index)
                for index, msg in enumerate(messages_list)
            ]
            # Don't pass tool_defs when using emulation
            tool_defs = None
        else:
            converted_messages = [
                self._convert_message(msg, index) for index, msg in enumerate(messages)
            ]
            tool_defs = self._convert_tools(tools)

        prompt = tokenizer.apply_chat_template(  # pyright: ignore[reportCallIssue]
            converted_messages,
            tool_defs or None,
            add_generation_prompt=True,
        )

        runtime_overrides = dict(kwargs)
        sampler = self._build_sampler(runtime_overrides)
        stream_kwargs = self._build_stream_kwargs(runtime_overrides)
        if sampler is not None:
            stream_kwargs["sampler"] = sampler

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Acquire generation lock to serialize MLX model usage across concurrent requests
        await self._generation_lock.acquire()

        try:

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
                    for response in stream_generate(
                        mdl,
                        tokenizer,
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
            accumulated_content = ""  # For emulation parsing

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

                    # Process native tool calls if supported
                    segments, parsed_calls = self._process_response_text(
                        response.text, tool_state, tokenizer
                    )

                    # Accumulate content for emulation
                    if use_tool_emulation:
                        accumulated_content += response.text

                    # Yield native tool calls
                    for tool_call in parsed_calls:
                        log.debug(
                            "MLX parsed tool_call name=%s id=%s",
                            tool_call.name,
                            tool_call.id,
                        )
                        yield tool_call

                    # Parse emulated tool calls on final response
                    if is_final and use_tool_emulation and accumulated_content:
                        log.debug(
                            "Parsing emulated tool calls from accumulated content"
                        )
                        emulated_calls = self._parse_function_calls(
                            accumulated_content, tools
                        )
                        for tool_call in emulated_calls:
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call
                        # Don't emit segments if we found emulated tool calls
                        if emulated_calls:
                            segments = []

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
                    log.error(
                        "MLX _stream_chat received error from thread: %s", payload
                    )
                    raise payload
                elif kind == "done":
                    if not done_emitted:
                        log.debug(
                            "MLX done without explicit final token; emitting done chunk"
                        )
                        yield Chunk(content="", done=True)
                    break
        finally:
            # Always release generation lock, even on error
            self._generation_lock.release()
            log.debug("MLX _stream_chat end | model=%s", model)

    async def _load_model(self, model: str) -> tuple[nn.Module, TokenizerWrapper]:
        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(model, "language_model", self.adapter_path):
            async with self._load_lock:
                # Try to get cached model from ModelManager
                cached = ModelManager.get_model(model, "language_model", self.adapter_path)
                if cached is not None:
                    return cached

                def _load() -> tuple[nn.Module, TokenizerWrapper]:
                    return mlx_lm.utils.load(  # pyright: ignore[reportReturnType]
                        model,
                        adapter_path=self.adapter_path,
                    )

                mdl, tokenizer = await asyncio.to_thread(_load)

                # Cache in ModelManager (requires a node_id, use a synthetic one for provider-level caching)
                node_id = f"mlx_provider_{model}_{self.adapter_path or 'default'}"
                ModelManager.set_model(node_id, model, "language_model", (mdl, tokenizer), self.adapter_path)

                log.info("Loaded MLX model %s", model)
                return mdl, tokenizer

    # ------------------------------------------------------------------
    # mlx-vlm runtime and flow (vision models)
    # ------------------------------------------------------------------
    def _cache_key_vlm(self, model: str) -> str:
        adapter = self.adapter_path or ""
        return f"vlm|{model}|{adapter}"

    def _get_cached_vlm_model(self, model: str) -> tuple[nn.Module, mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, Any] | None:
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

    def _set_cached_vlm_model(self, model: str, mdl: nn.Module, proc: mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, cfg: Any) -> None:
        key = self._cache_key_vlm(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _VLM_MODEL_CACHE_LOCK:
            _VLM_MODEL_CACHE[key] = (mdl, proc, cfg, expires_at)

    async def _load_vlm_model(self, model: str) -> tuple[nn.Module, mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, Any]:
        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(model, "vision_model", self.adapter_path):
            async with self._vlm_load_lock:
                # Try to get cached model from ModelManager
                cached = ModelManager.get_model(model, "vision_model", self.adapter_path)
                if cached is not None:
                    return cached

                def _load() -> tuple[nn.Module, mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, Any]:
                    mdl, proc = mlx_vlm.load(model)
                    cfg = getattr(mdl, "config", None)
                    if cfg is None and mlx_vlm.utils.load_config is not None:
                        cfg = mlx_vlm.utils.load_config(model)
                    return mdl, proc, cfg

                mdl, proc, cfg = await asyncio.to_thread(_load)

                # Cache in ModelManager
                node_id = f"mlx_vlm_provider_{model}_{self.adapter_path or 'default'}"
                ModelManager.set_model(node_id, model, "vision_model", (mdl, proc, cfg), self.adapter_path)

                log.info("Loaded MLX-VLM model %s", model)
                return mdl, proc, cfg

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

    def _ensure_vlm_processor_ready(self, proc: mlx_vlm.utils.PreTrainedTokenizer | mlx_vlm.utils.PreTrainedTokenizerFast, cfg: Any) -> None:
        """Ensure mlx-vlm processor has necessary attributes set.

        Some processors (e.g., LLaVA) expect a non-None patch_size. If missing,
        attempt to infer from model config or fall back to a sane default (14).
        """
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
        mdl, proc, cfg = await self._load_vlm_model(model)

        prompt_text = self._extract_last_user_prompt(messages)
        images = await self._prepare_vlm_images(image_parts)
        audios = await self._prepare_vlm_audio(audio_parts)
        log.debug(
            "MLX-VLM prepared assets | images=%d audios=%d", len(images), len(audios)
        )

        # Defensive processor normalization
        self._ensure_vlm_processor_ready(proc, cfg)

        formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
            proc,
            cfg,
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
            result = mlx_vlm.generate.generate(
                mdl,
                proc,
                formatted_prompt,
                image=images if images else None,
                audio=audios if audios else None,
                verbose=False,
                max_tokens=max_tokens,
            )
            # Some versions may return objects; coerce to string
            return result.text

        # Serialize VLM generation to avoid concurrent Metal access
        async with self._vlm_generation_lock:
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
        return f"{model}|{adapter}"

    def _get_cached_model(self, model: str) -> tuple[nn.Module, TokenizerWrapper] | None:
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

    def _set_cached_model(self, model: str, mdl: nn.Module, tok: TokenizerWrapper) -> None:
        key = self._cache_key(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = (mdl, tok, expires_at)

    # ------------------------------------------------------------------
    # TTS model caching helpers
    # ------------------------------------------------------------------
    def _get_cached_tts_model(self, model: str) -> Any | None:
        """Get cached TTS model if available and not expired."""
        now = time.monotonic()
        with _TTS_MODEL_CACHE_LOCK:
            entry = _TTS_MODEL_CACHE.get(model)
            if not entry:
                return None
            tts_model, expires_at = entry
            if expires_at < now:
                # Expired; evict
                _TTS_MODEL_CACHE.pop(model, None)
                return None
            return tts_model

    def _set_cached_tts_model(self, model: str, tts_model: Any) -> None:
        """Cache a TTS model with TTL expiration."""
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _TTS_MODEL_CACHE_LOCK:
            _TTS_MODEL_CACHE[model] = (tts_model, expires_at)

    async def _load_tts_model(self, model: str) -> Any:
        """Load TTS model from cache or filesystem.

        Args:
            model: Model repository ID (e.g., "mlx-community/Kokoro-82M-bf16")

        Returns:
            Loaded TTS model

        Raises:
            ValueError: If model is not found in HuggingFace cache
        """
        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(model, "tts_model", None):
            # Try to get cached model from ModelManager
            tts_model = ModelManager.get_model(model, "tts_model", None)

            if tts_model is None:
                # Model not in cache, load it
                repo_path = self._resolve_cached_repo_path(model)
                if repo_path is None:
                    raise ValueError(
                        f"Model {model} must be downloaded first (not found in cache)"
                    )

                load_target = repo_path

                # Load the TTS model
                def _load():
                    log.info(f"Loading MLX TTS model {model}")
                    return mlx_audio.tts.utils.load_model(Path(load_target))

                tts_model = await asyncio.to_thread(_load)

                # Cache in ModelManager
                node_id = f"mlx_tts_provider_{model}"
                ModelManager.set_model(node_id, model, "tts_model", tts_model, None)
                log.debug(f"Cached TTS model {model}")
            else:
                log.debug(f"Using cached TTS model {model}")

            return tts_model

    def _resolve_cached_repo_path(self, repo_id: str) -> Path | None:
        """
        Locate the snapshot directory for a cached Hugging Face repo.

        Returns:
            Path to the cached repo snapshot, or None if not found.
        """
        try:
            cache_info = scan_cache_dir()
        except CacheNotFound:
            return None

        for repo in cache_info.repos:
            if repo.repo_type != "model" or repo.repo_id != repo_id:
                continue

            for revision in repo.revisions:
                snapshot_path = getattr(revision, "snapshot_path", None)
                if snapshot_path and os.path.isdir(snapshot_path):
                    return Path(snapshot_path)

            repo_path = getattr(repo, "repo_path", None)
            if repo_path and os.path.isdir(repo_path):
                return Path(repo_path)

        return None

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

    def _build_sampler(self, kwargs: dict[str, Any]) -> Any | None:
        if mlx_lm.sample_utils.make_sampler is None:
            return None

        sampler_params = {
            "temp": kwargs.pop("temperature", 0.5),
            "top_p": kwargs.pop("top_p", 0.95),
            "top_k": kwargs.pop("top_k", 50),
            "min_p": kwargs.pop("min_p", 0.0),
            "min_tokens_to_keep": kwargs.pop("min_tokens_to_keep", 1),
            "xtc_probability": kwargs.pop("xtc_probability", 0.0),
            "xtc_threshold": kwargs.pop("xtc_threshold", 0.0),
        }

        # When temperature is explicitly zero we avoid allocating a sampler.
        if (
            sampler_params.get("temp", 0.0) == 0
            and sampler_params.get("top_p", 0.0) == 0.0
        ):
            return None

        return mlx_lm.sample_utils.make_sampler(**sampler_params)

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
        tokenizer: TokenizerWrapper,
    ) -> tuple[list[str], list[ToolCall]]:
        if not text:
            return [], []

        if not tokenizer.has_tool_calling:
            return [text], []

        start_token = getattr(tokenizer, "tool_call_start", None)
        end_token = getattr(tokenizer, "tool_call_end", None)
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

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:  # Returns np.ndarray[np.int16]
        """Generate speech audio from text using MLX text-to-speech models with streaming.

        MLX TTS models support true streaming, yielding audio chunks as they're generated.

        Args:
            text: Input text to convert to speech
            model: Model repository ID (e.g., "mlx-community/Kokoro-82M-bf16")
            voice: Voice preset (e.g., "af_heart", "am_adam") - defaults to "af_heart"
            speed: Speech speed multiplier (0.5 to 2.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional MLX TTS parameters (e.g., temperature, language)

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails or platform is unsupported
        """
        log.debug(
            f"Generating streaming speech with MLX model: {model}, voice: {voice}, speed: {speed}"
        )

        if sys.platform != "darwin":
            raise RuntimeError("MLX TTS requires macOS (Apple Silicon / MLX)")

        if not text:
            raise ValueError("text must not be empty")

        # Default voice and parameters
        voice = voice or "af_heart"
        speed = max(0.5, min(2.0, speed))

        try:
            # Load TTS model (from cache or filesystem)
            tts_model = await self._load_tts_model(model)

            # Prepare generation parameters - enable streaming
            gen_params = {
                "text": text,
                "speed": speed,
                "stream": True,  # Enable streaming
                "verbose": False,
            }

            # Add voice if the model supports it (Kokoro-style models)
            if voice:
                gen_params["voice"] = voice

            # Add additional kwargs
            if "temperature" in kwargs:
                gen_params["temperature"] = kwargs["temperature"]
            if "lang_code" in kwargs:
                gen_params["lang_code"] = kwargs["lang_code"]
            if "language" in kwargs:
                gen_params["lang_code"] = kwargs["language"]

            # Stream audio chunks
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _stream_audio():
                """Stream audio chunks in background thread."""
                try:
                    assert tts_model.generate is not None
                    for result in tts_model.generate(**gen_params):
                        audio = result.audio
                        if audio is None:
                            continue

                        # Convert MLX array to numpy float32
                        if hasattr(audio, "astype") and hasattr(audio, "numpy"):
                            chunk_float = audio.astype(np.float32).numpy()
                        else:
                            chunk_float = np.asarray(audio, dtype=np.float32)

                        chunk_int16 = np.clip(chunk_float, -1.0, 1.0)
                        chunk_int16 = (chunk_int16 * 32767.0).astype(np.int16)

                        if chunk_float.size > 0:
                            # Put float chunk in queue
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("chunk", chunk_int16)), loop
                            ).result()

                    # Signal completion
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("done", None)), loop
                    ).result()

                except Exception as exc:
                    log.exception(f"MLX TTS streaming error: {exc}")
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("error", exc)), loop
                    ).result()

            # Start streaming thread
            threading.Thread(target=_stream_audio, daemon=True).start()

            while True:
                kind, payload = await queue.get()

                if kind == "chunk":
                    yield payload
                elif kind == "error":
                    raise payload
                elif kind == "done":
                    break

        except Exception as e:
            log.error(f"MLX TTS generation failed: {e}")
            raise RuntimeError(f"MLX TTS generation failed: {str(e)}")

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using MLX.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
        """
        if not params.prompt:
            raise ValueError("Prompt cannot be empty for image generation.")

        try:
            # Use ModelManager lock for thread-safe model loading
            async with ModelManager.lock_model(params.model.id, "flux_model", None):
                # Load model using centralized loader
                model = await load_flux_model(
                    model_id=params.model.id,
                    quantize=4,  # Default to 4-bit quantization
                    task="flux",
                )

                loop = asyncio.get_running_loop()

                def _generate() -> PIL.Image.Image:
                    # Default to 4 steps for text-to-image (FLUX-schnell models are optimized for 4 steps)
                    default_steps = 4
                    
                    # Prepare config
                    config_kwargs: dict[str, Any] = {
                        "num_inference_steps": params.num_inference_steps or default_steps,
                        "height": params.height or 1024,
                        "width": params.width or 1024,
                    }

                    if params.guidance_scale is not None:
                        config_kwargs["guidance"] = params.guidance_scale

                    # Filter to only allowed config fields
                    dataclass_fields = getattr(Config, "__dataclass_fields__", None)
                    if isinstance(dataclass_fields, dict):
                        allowed = set(dataclass_fields.keys())
                        config_kwargs = {
                            key: value
                            for key, value in config_kwargs.items()
                            if key in allowed
                        }

                    config = Config(**config_kwargs)

                    # Generate image
                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        prompt=params.prompt,
                        config=config,
                    )
                    return generated.image

                # Run generation in executor
                pil_image = await loop.run_in_executor(None, _generate)

            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            return image_bytes

        except Exception as e:
            log.error(f"MLX text-to-image generation failed: {e}")
            raise RuntimeError(f"MLX text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using MLX.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
        """

        if not params.prompt:
            raise ValueError("Prompt cannot be empty for image-to-image generation.")

        # Load model using centralized loader
        model = await load_flux_model(
            model_id=params.model.id,
            quantize=4,  # Default to 4-bit quantization
            task="flux",
        )

        # Load input image
        base_image = PIL.Image.open(BytesIO(image))

        loop = asyncio.get_running_loop()

        def _generate() -> PIL.Image.Image:
            # Prepare image
            working_image = base_image.convert("RGB")
            target_width = 16 * ((params.target_width or 1024) // 16)
            target_height = 16 * ((params.target_height or 1024) // 16)

            if working_image.size != (target_width, target_height):
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = Path(tmp.name)
                working_image.save(image_path)

            try:
                # Auto-detect inference steps based on model name
                # FLUX-schnell models are optimized for 4 steps
                model_name_lower = params.model.id.lower()
                default_steps = 4 if "schnell" in model_name_lower else 8
                
                # Prepare config
                config_kwargs: dict[str, Any] = {
                    "num_inference_steps": params.num_inference_steps or default_steps,
                    "height": target_height,
                    "width": target_width,
                    "image_strength": params.strength or 0.4,
                    "image_path": image_path,
                }

                if params.guidance_scale is not None:
                    config_kwargs["guidance"] = params.guidance_scale

                # Filter to only allowed config fields
                dataclass_fields = getattr(Config, "__dataclass_fields__", None)
                if isinstance(dataclass_fields, dict):
                    allowed = set(dataclass_fields.keys())
                    config_kwargs = {
                        key: value
                        for key, value in config_kwargs.items()
                        if key in allowed
                    }

                config = Config(**config_kwargs)

                # Generate image
                generated = model.generate_image(
                    seed=params.seed if params.seed is not None else 0,
                    prompt=params.prompt,
                    config=config,
                )
                return generated.image
            finally:
                # Clean up temp file
                try:
                    image_path.unlink()
                except FileNotFoundError:
                    pass

        # Run generation in executor
        pil_image = await loop.run_in_executor(None, _generate)

        # Convert to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        self.usage["total_requests"] += 1
        self.usage["total_images"] += 1

        return image_bytes


async def main() -> None:
    """Run a comprehensive demo of the MLX provider.

    - Basic generation: ask the model to say hello
    - Multiple concurrent generations to test serialization lock
    - Optional tool-call round trip with a calculator tool
    """
    from nodetool.workflows.processing_context import ProcessingContext

    class CalculatorTool(Tool):
        name = "calculator"
        description = (
            "Perform basic arithmetic operations (add, subtract, multiply, divide)."
        )
        input_schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        }

        async def process(  # type: ignore[override]
            self, context: ProcessingContext, params: dict[str, Any]
        ) -> Any:
            operation = params.get("operation")
            a = float(params.get("a", 0))
            b = float(params.get("b", 0))

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"error": "Cannot divide by zero"}
                result = a / b
            else:
                return {"error": f"Unknown operation: {operation}"}

            return {
                "operation": operation,
                "a": a,
                "b": b,
                "result": result,
            }

    provider = MLXProvider()
    model_name = "ibm-granite/granite-4.0-micro"

    print("=" * 70)
    print("TEST 1: Basic single generation")
    print("=" * 70)

    content_parts = []
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
    print("Response:", "".join(content_parts))
    print()

    print("=" * 70)
    print("TEST 2: Multiple concurrent generations (serialization test)")
    print("=" * 70)

    async def generate_response(prompt: str, index: int) -> str:
        """Helper to generate a single response."""
        messages = [Message(role="user", content=prompt)]
        parts = []
        print(f"[{index}] Starting generation...")
        async for item in provider.generate_messages(
            messages=messages, model=model_name, max_tokens=100
        ):
            if isinstance(item, Chunk):
                if item.content:
                    parts.append(item.content)
                if item.done:
                    break
        result = "".join(parts)
        print(f"[{index}] Completed: {result[:50]}...")
        return result

    # Launch multiple concurrent requests
    prompts = [
        "What is 2+2? Answer in one sentence.",
        "Name a color. Just one word.",
        "Count from 1 to 3. Just the numbers.",
        "What is the capital of France? One word answer.",
        "Is water wet? Answer yes or no.",
    ]

    print(f"\nLaunching {len(prompts)} concurrent requests...")
    tasks = [generate_response(prompt, i) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    print("\n✅ All concurrent requests completed successfully!")
    for i, result in enumerate(results):
        print(f"  [{i}]: {result[:60]}...")
    print()

    print("=" * 70)
    print("TEST 3: Tool calling with multiple rounds")
    print("=" * 70)

    tools: list[Tool] = [CalculatorTool()]
    context = ProcessingContext()
    messages = [
        Message(
            role="user",
            content="Use the calculator to multiply 23 by 17, then tell me the result.",
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
                print(f"  Args: {tc.args}")
                result = await tool.process(context, tc.args or {})
                print(f"  Result: {result}")
                messages.append(
                    Message(
                        role="tool",
                        name=tool.name,
                        tool_call_id=tc.id,
                        content=json.dumps(result),
                    )
                )

            print("\nFinal response:")
            async for item in provider.generate_messages(
                messages=messages, model=model_name, tools=tools, max_tokens=4096
            ):
                if isinstance(item, Chunk):
                    if item.content:
                        print(item.content, end="", flush=True)
                    if item.done:
                        print("\n")
                        break
        else:
            print("No tool calls returned by the model.")
    except Exception as e:  # pragma: no cover - demo convenience
        print(f"Tool call demo failed: {e}")

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
