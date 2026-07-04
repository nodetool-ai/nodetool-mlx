from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, TYPE_CHECKING, TypedDict

from pydantic import Field, PrivateAttr

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import AudioRef, HuggingFaceModel, Provider
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

log = get_logger(__name__)

# HuggingFace organisation hosting the ACE-Step 1.5 checkpoints.
ACE_STEP_ORG = "ACE-Step"
# Bundle download containing the VAE, text encoder, the turbo DiT and the 1.7B LM.
ACE_STEP_MAIN_REPO = f"{ACE_STEP_ORG}/Ace-Step1.5"


class DiTModel(str, Enum):
    """ACE-Step 1.5 Diffusion Transformer configurations.

    The value of each member matches the ``config_path`` understood by
    :meth:`AceStepHandler.initialize_service`.
    """

    TURBO = "acestep-v15-turbo"
    BASE = "acestep-v15-base"
    SFT = "acestep-v15-sft"
    TURBO_SHIFT1 = "acestep-v15-turbo-shift1"
    TURBO_SHIFT3 = "acestep-v15-turbo-shift3"
    TURBO_CONTINUOUS = "acestep-v15-turbo-continuous"
    XL_BASE = "acestep-v15-xl-base"
    XL_SFT = "acestep-v15-xl-sft"
    XL_TURBO = "acestep-v15-xl-turbo"


class LanguageModel(str, Enum):
    """Optional 5Hz language-model "planner" used for chain-of-thought.

    ``DISABLED`` runs ACE-Step in DiT-only mode (lighter, faster) where the
    caption and metadata fields are used verbatim. Selecting a model enables
    the planner so the LM can expand the caption, infer metadata and clean up
    lyrics before synthesis.
    """

    DISABLED = ""
    LM_0_6B = "acestep-5Hz-lm-0.6B"
    LM_1_7B = "acestep-5Hz-lm-1.7B"
    LM_4B = "acestep-5Hz-lm-4B"


class VocalLanguage(str, Enum):
    """A subset of the languages ACE-Step understands for vocal synthesis."""

    UNKNOWN = "unknown"
    ENGLISH = "English"
    MANDARIN = "Mandarin"
    CANTONESE = "Cantonese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    PORTUGUESE = "Portuguese"
    RUSSIAN = "Russian"
    ARABIC = "Arabic"
    HINDI = "Hindi"
    TURKISH = "Turkish"
    VIETNAMESE = "Vietnamese"
    INDONESIAN = "Indonesian"
    THAI = "Thai"


class ACEStepBaseNode(BaseNode):
    """Shared plumbing for ACE-Step 1.5 nodes running on the MLX backend.

    The heavy ``acestep`` package is imported lazily so that this module can be
    imported (and scanned for metadata) on any platform. Generation itself only
    works on Apple Silicon, where ACE-Step uses MLX for the DiT, VAE and LM.
    """

    _provider: ClassVar[Provider] = Provider.MLX
    _body: ClassVar[str] = "content_card"

    # ACE-Step resolves the device internally; "auto" maps to MPS + MLX on
    # Apple Silicon.
    _DEVICE: ClassVar[str] = "auto"

    _dit_handler: Any | None = PrivateAttr(default=None)
    _llm_handler: Any | None = PrivateAttr(default=None)

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not ACEStepBaseNode

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    # -- environment helpers -------------------------------------------------

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "ACE-Step 1.5 generation requires macOS (Apple Silicon / MLX)."
            )

    @staticmethod
    def _require_acestep() -> None:
        try:
            import acestep  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on host install
            raise RuntimeError(
                "ACE-Step 1.5 is not installed. Install it from "
                "https://github.com/ace-step/ACE-Step-1.5 (note: the PyPI "
                "'ace-step' package is the older 1.0 release and is not "
                "compatible with these nodes)."
            ) from exc

    @staticmethod
    def _setup_env() -> None:
        # Force the Apple Silicon MLX backend for the language model and keep
        # the tokenizers library quiet inside the worker threads.
        os.environ.setdefault("ACESTEP_LM_BACKEND", "mlx")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @staticmethod
    def _resolve_paths() -> tuple[str, str]:
        """Return ``(project_root, checkpoints_dir)`` for ACE-Step.

        ACE-Step downloads its checkpoints into ``<project_root>/checkpoints``
        (or ``ACESTEP_CHECKPOINTS_DIR`` when set) on first use. We default to a
        stable cache directory so models persist between runs.
        """

        env_ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
        if env_ckpt:
            checkpoints_dir = os.path.abspath(os.path.expanduser(env_ckpt))
            project_root = os.path.dirname(checkpoints_dir)
        else:
            project_root = os.path.abspath(
                os.path.expanduser(
                    os.environ.get("ACESTEP_PROJECT_ROOT", "~/.cache/nodetool/acestep")
                )
            )
            checkpoints_dir = os.path.join(project_root, "checkpoints")
            os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", checkpoints_dir)

        os.makedirs(checkpoints_dir, exist_ok=True)
        return project_root, checkpoints_dir

    # -- defensive call helpers ---------------------------------------------

    @staticmethod
    def _call_filtered(func: Callable[..., Any], **kwargs: Any) -> Any:
        """Call ``func`` passing only the keyword arguments it accepts.

        ACE-Step's handler signatures have evolved between releases; filtering
        keeps these nodes resilient to minor signature drift.
        """

        import inspect

        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - builtins, etc.
            return func(**kwargs)

        params = sig.parameters.values()
        if any(p.kind == p.VAR_KEYWORD for p in params):
            return func(**kwargs)

        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        dropped = sorted(set(kwargs) - set(accepted))
        if dropped:
            log.debug(
                "Dropping unsupported kwargs for %s: %s",
                getattr(func, "__qualname__", func),
                dropped,
            )
        return func(**accepted)

    @staticmethod
    def _dataclass_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
        """Filter ``kwargs`` to the fields defined on a dataclass ``cls``."""

        fields = getattr(cls, "__dataclass_fields__", None)
        if not fields:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in fields}

    # -- model loading -------------------------------------------------------

    def _load_dit_handler(
        self, project_root: str, config_path: str, device: str
    ) -> "AceStepHandler":
        from acestep.handler import AceStepHandler

        cache_key = f"acestep_dit::{config_path}::{device}"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            return cached

        log.info("Loading ACE-Step DiT handler (config=%s)", config_path)
        handler = AceStepHandler()
        self._call_filtered(
            handler.initialize_service,
            project_root=project_root,
            config_path=config_path,
            device=device,
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        ModelManager.set_model(self.id, cache_key, handler)
        return handler

    def _load_llm_handler(
        self, checkpoints_dir: str, lm_model_path: str, device: str
    ) -> "LLMHandler":
        from acestep.llm_inference import LLMHandler

        cache_key = f"acestep_llm::{lm_model_path}::{device}"
        cached = ModelManager.get_model(cache_key)
        if cached is not None:
            return cached

        log.info("Loading ACE-Step 5Hz LM handler (model=%s)", lm_model_path)
        handler = LLMHandler()
        self._call_filtered(
            handler.initialize,
            checkpoint_dir=checkpoints_dir,
            lm_model_path=lm_model_path,
            backend="mlx",
            device=device,
            offload_to_cpu=False,
            dtype=None,
        )
        ModelManager.set_model(self.id, cache_key, handler)
        return handler

    def _new_llm_handler(self) -> "LLMHandler":
        """An uninitialised LLM handler for DiT-only (planner disabled) runs."""

        from acestep.llm_inference import LLMHandler

        return LLMHandler()

    def _post_progress(
        self, context: ProcessingContext, progress: int, total: int
    ) -> None:
        context.post_message(
            NodeProgress(node_id=self.id, progress=progress, total=max(total, 1))
        )


class ACEStepMusicGeneration(ACEStepBaseNode):
    """
    Generate full songs locally with ACE-Step 1.5 on Apple Silicon (MLX).
    mlx, music, audio, text-to-music, ace-step, apple-silicon

    Use cases:
    - Turn a text description (and optional lyrics) into a complete track
    - Produce instrumental beds, demos or background music without cloud APIs
    - Iterate on prompts locally using the fast turbo DiT, then switch to the
      higher quality base / XL checkpoints

    ACE-Step downloads its checkpoints from HuggingFace into
    ``~/.cache/nodetool/acestep/checkpoints`` (configurable via
    ``ACESTEP_CHECKPOINTS_DIR``) the first time a model is used.
    """

    _expose_as_tool: ClassVar[bool] = True

    prompt: str = Field(
        default="A dreamy lo-fi hip hop beat with warm piano and vinyl crackle",
        title="Prompt",
        description="Text description (caption) of the music to generate.",
    )
    lyrics: str = Field(
        default="",
        title="Lyrics",
        description=(
            "Optional lyrics. Use structure tags like [verse], [chorus] and "
            "[outro] on their own lines. Leave empty for instrumental output."
        ),
    )
    instrumental: bool = Field(
        default=False,
        title="Instrumental",
        description="Generate music without vocals, ignoring any lyrics.",
    )
    duration: float = Field(
        default=30.0,
        ge=-1.0,
        le=600.0,
        title="Duration",
        description="Target length in seconds (10-600). Use -1 to let the model decide.",
    )
    dit_model: DiTModel = Field(
        default=DiTModel.TURBO,
        title="DiT Model",
        description="Diffusion Transformer checkpoint used for synthesis.",
    )
    language_model: LanguageModel = Field(
        default=LanguageModel.DISABLED,
        title="Planner LM",
        description=(
            "Optional 5Hz language model. When enabled it expands the prompt "
            "and infers metadata via chain-of-thought before synthesis. "
            "Disabled runs ACE-Step in DiT-only mode."
        ),
    )
    inference_steps: int = Field(
        default=8,
        ge=1,
        le=200,
        title="Inference Steps",
        description="Number of diffusion steps. ~8 for turbo, ~50 for base/sft.",
    )
    guidance_scale: float = Field(
        default=7.0,
        ge=0.0,
        le=30.0,
        title="Guidance Scale",
        description=(
            "Classifier-free guidance strength. Turbo checkpoints force this "
            "to 1.0 internally for stability."
        ),
    )
    bpm: int = Field(
        default=0,
        ge=0,
        le=300,
        title="BPM",
        description="Target tempo in beats per minute. 0 lets the model choose.",
    )
    keyscale: str = Field(
        default="",
        title="Key / Scale",
        description="Optional key and mode, e.g. 'C major' or 'A minor'. Empty = auto.",
    )
    time_signature: str = Field(
        default="",
        title="Time Signature",
        description="Optional time signature, e.g. '4/4'. Empty = auto.",
    )
    vocal_language: VocalLanguage = Field(
        default=VocalLanguage.UNKNOWN,
        title="Vocal Language",
        description="Language hint for vocals.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        title="Seed",
        description="Seed for reproducible generation. 0 picks a random seed.",
    )
    thinking: bool = Field(
        default=True,
        title="Planner Thinking",
        description="Enable chain-of-thought when a planner LM is selected.",
    )
    lm_temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        title="LM Temperature",
        description="Sampling temperature for the planner LM.",
    )
    lm_cfg_scale: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        title="LM Guidance",
        description="Classifier-free guidance scale for the planner LM.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Music Generation"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "lyrics",
            "instrumental",
            "duration",
            "dit_model",
            "inference_steps",
            "guidance_scale",
            "seed",
        ]

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["prompt", "lyrics"]

    @classmethod
    def get_inline_fields(cls) -> list[str]:
        return ["dit_model", "prompt"]

    def required_inputs(self) -> list[str]:
        return ["prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            # Main bundle: VAE + text encoder + turbo DiT + 1.7B planner LM.
            HuggingFaceModel(repo_id=ACE_STEP_MAIN_REPO),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-v15-base"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-v15-sft"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-v15-xl-base"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-v15-xl-turbo"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-5Hz-lm-0.6B"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-5Hz-lm-4B"),
        ]

    def _resolve_seed(self) -> int:
        if self.seed and self.seed > 0:
            return int(self.seed)
        return random.randint(0, 2**31 - 1)

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        self._require_acestep()
        self._setup_env()
        project_root, checkpoints_dir = self._resolve_paths()
        device = self._DEVICE
        config_path = self.dit_model.value
        lm_enabled = self.language_model != LanguageModel.DISABLED
        lm_model_path = self.language_model.value

        loop = asyncio.get_running_loop()

        def _load() -> tuple[Any, Any]:
            dit = self._load_dit_handler(project_root, config_path, device)
            llm = (
                self._load_llm_handler(checkpoints_dir, lm_model_path, device)
                if lm_enabled
                else self._new_llm_handler()
            )
            return dit, llm

        self._dit_handler, self._llm_handler = await loop.run_in_executor(None, _load)

    async def process(self, context: ProcessingContext) -> AudioRef:
        self._ensure_supported_platform()
        self._require_acestep()
        self._setup_env()
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty for music generation.")

        if self._dit_handler is None:
            await self.preload_model(context)

        seed = self._resolve_seed()
        lm_enabled = self.language_model != LanguageModel.DISABLED
        self._post_progress(context, 0, self.inference_steps)

        loop = asyncio.get_running_loop()

        def _generate() -> bytes:
            from acestep.inference import (
                GenerationConfig,
                GenerationParams,
                generate_music,
            )

            dit_handler = self._dit_handler
            llm_handler = self._llm_handler or self._new_llm_handler()
            assert dit_handler is not None

            params = GenerationParams(
                **self._dataclass_kwargs(
                    GenerationParams,
                    task_type="text2music",
                    caption=self.prompt,
                    lyrics="" if self.instrumental else self.lyrics,
                    instrumental=self.instrumental,
                    duration=float(self.duration),
                    bpm=(self.bpm or None),
                    keyscale=self.keyscale.strip(),
                    timesignature=self.time_signature.strip(),
                    vocal_language=self.vocal_language.value,
                    inference_steps=int(self.inference_steps),
                    guidance_scale=float(self.guidance_scale),
                    seed=seed,
                    thinking=(self.thinking and lm_enabled),
                    lm_temperature=float(self.lm_temperature),
                    lm_cfg_scale=float(self.lm_cfg_scale),
                    use_cot_metas=lm_enabled,
                    use_cot_caption=lm_enabled,
                    use_cot_lyrics=False,
                    use_cot_language=lm_enabled,
                )
            )
            config = GenerationConfig(
                **self._dataclass_kwargs(
                    GenerationConfig,
                    batch_size=1,
                    use_random_seed=(self.seed == 0),
                    seeds=(None if self.seed == 0 else [seed]),
                    audio_format="wav",
                )
            )

            with tempfile.TemporaryDirectory(prefix="acestep_") as tmp_dir:
                result = generate_music(
                    dit_handler, llm_handler, params, config, save_dir=tmp_dir
                )
                return self._read_result_audio(result)

        data = await loop.run_in_executor(None, _generate)
        self._post_progress(context, self.inference_steps, self.inference_steps)
        return await context.audio_from_bytes(data)

    @staticmethod
    def _read_result_audio(result: Any) -> bytes:
        if not getattr(result, "success", True):
            message = (
                getattr(result, "error", None)
                or getattr(result, "status_message", None)
                or "ACE-Step generation failed."
            )
            raise RuntimeError(str(message))

        audios = getattr(result, "audios", None) or []
        if not audios:
            raise RuntimeError("ACE-Step generation returned no audio.")

        first = audios[0]
        path = (
            first.get("path")
            if isinstance(first, dict)
            else getattr(first, "path", None)
        )
        if not path or not os.path.exists(path):
            raise RuntimeError("ACE-Step did not write an output audio file.")
        return Path(path).read_bytes()


class ACEStepSongPlanner(ACEStepBaseNode):
    """
    Turn a natural-language idea into a song blueprint using the ACE-Step LM.
    mlx, music, planner, lyrics, ace-step, apple-silicon

    Use cases:
    - Expand a short idea into a caption, lyrics and musical metadata
    - Generate lyrics and a key/BPM suggestion to feed into music generation
    - Prototype song concepts before committing to full audio synthesis

    This node only runs the 5Hz language model (no diffusion), so it is fast
    and lightweight compared to full music generation.
    """

    query: str = Field(
        default="An uplifting indie pop song about chasing the sunrise",
        title="Idea",
        description="Natural-language description of the song you want to plan.",
    )
    language_model: LanguageModel = Field(
        default=LanguageModel.LM_1_7B,
        title="Planner LM",
        description="5Hz language model used to draft the song blueprint.",
    )
    instrumental: bool = Field(
        default=False,
        title="Instrumental",
        description="Plan an instrumental track (no lyrics).",
    )
    vocal_language: VocalLanguage = Field(
        default=VocalLanguage.UNKNOWN,
        title="Vocal Language",
        description="Preferred language for generated lyrics.",
    )
    temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        title="Temperature",
        description="Sampling temperature for the planner LM.",
    )

    class OutputType(TypedDict):
        caption: str
        lyrics: str
        bpm: int | None
        keyscale: str
        time_signature: str
        vocal_language: str

    @classmethod
    def get_title(cls) -> str:
        return "ACE-Step Song Planner"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["query", "language_model", "instrumental", "vocal_language"]

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["query"]

    @classmethod
    def get_inline_fields(cls) -> list[str]:
        return ["language_model", "query"]

    def required_inputs(self) -> list[str]:
        return ["query"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(repo_id=ACE_STEP_MAIN_REPO),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-5Hz-lm-0.6B"),
            HuggingFaceModel(repo_id=f"{ACE_STEP_ORG}/acestep-5Hz-lm-4B"),
        ]

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        self._require_acestep()
        self._setup_env()
        _, checkpoints_dir = self._resolve_paths()
        model = self.language_model
        if model == LanguageModel.DISABLED:
            model = LanguageModel.LM_1_7B
        lm_model_path = model.value
        device = self._DEVICE

        loop = asyncio.get_running_loop()

        def _load() -> Any:
            return self._load_llm_handler(checkpoints_dir, lm_model_path, device)

        self._llm_handler = await loop.run_in_executor(None, _load)

    async def process(self, context: ProcessingContext) -> OutputType:
        self._ensure_supported_platform()
        self._require_acestep()
        self._setup_env()
        if not self.query.strip():
            raise ValueError("Query cannot be empty for the song planner.")

        if self._llm_handler is None:
            await self.preload_model(context)

        vocal_language = (
            None
            if self.vocal_language == VocalLanguage.UNKNOWN
            else self.vocal_language.value
        )

        loop = asyncio.get_running_loop()

        def _plan() -> "ACEStepSongPlanner.OutputType":
            from acestep.inference import create_sample

            handler = self._llm_handler
            assert handler is not None

            result = self._call_filtered(
                create_sample,
                llm_handler=handler,
                query=self.query,
                instrumental=self.instrumental,
                vocal_language=vocal_language,
                temperature=float(self.temperature),
            )
            return self._result_to_output(result)

        return await loop.run_in_executor(None, _plan)

    @staticmethod
    def _result_to_output(result: Any) -> "ACEStepSongPlanner.OutputType":
        if dataclasses.is_dataclass(result) and not isinstance(result, type):
            data: dict[str, Any] = dataclasses.asdict(result)
        elif isinstance(result, dict):
            data = result
        else:
            data = {}

        def _get(name: str, default: Any = "") -> Any:
            if name in data:
                return data[name]
            return getattr(result, name, default)

        if not _get("success", True):
            message = _get("error", None) or _get(
                "status_message", "ACE-Step planner failed."
            )
            raise RuntimeError(str(message))

        bpm_value = _get("bpm", None)
        try:
            bpm: int | None = int(bpm_value) if bpm_value not in (None, "") else None
        except (TypeError, ValueError):
            bpm = None

        return {
            "caption": str(_get("caption", "") or ""),
            "lyrics": str(_get("lyrics", "") or ""),
            "bpm": bpm,
            "keyscale": str(_get("keyscale", "") or ""),
            "time_signature": str(_get("timesignature", "") or ""),
            "vocal_language": str(_get("vocal_language", "unknown") or "unknown"),
        }
