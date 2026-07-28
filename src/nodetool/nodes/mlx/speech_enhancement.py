from __future__ import annotations

import asyncio
import logging
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

from pydantic import Field, PrivateAttr

from nodetool.metadata.types import (
    AudioRef,
    HFAudioToAudio,
    HuggingFaceModel,
    Provider,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class BaseMLXSpeechEnhancement(BaseNode):
    """Shared functionality for MLX speech enhancement (denoising) nodes.

    These speech-to-speech models from ``mlx-audio`` clean up noisy speech and run
    locally on Apple Silicon. All of them operate on 48 kHz mono audio.
    """

    _body: ClassVar[str] = "content_card"
    # All currently supported enhancement models operate at 48 kHz.
    _sample_rate: ClassVar[int] = 48_000

    audio: AudioRef = Field(
        default=AudioRef(),
        title="Audio Input",
        description="The noisy input audio to enhance.",
    )

    _provider: ClassVar[Provider] = Provider.MLX
    _model: Any | None = PrivateAttr(default=None)
    _model_id_loaded: str | None = PrivateAttr(default=None)

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not BaseMLXSpeechEnhancement

    def required_inputs(self):
        return ["audio"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "audio"]

    @classmethod
    def get_input_fields(cls):
        return ["audio"]

    @classmethod
    def get_inline_fields(cls):
        return ["model"]

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "MLX speech enhancement requires macOS (Apple Silicon / MLX)."
            )

    def _get_model_id(self) -> str:
        model = getattr(self, "model", None)
        repo_id = getattr(model, "repo_id", "") if model is not None else ""
        if not repo_id:
            raise ValueError("Model must be selected before loading the MLX model.")
        return repo_id

    def _load_key(self) -> str:
        """Cache key for the loaded model; includes every field that affects loading."""
        return self._get_model_id()

    def _load_model_sync(self) -> Any:
        """Load the enhancement model. Implemented by subclasses."""
        raise NotImplementedError

    def _enhance_sync(self, samples: "np.ndarray") -> "np.ndarray":
        """Run enhancement on a 1-D float32 array. Implemented by subclasses."""
        raise NotImplementedError

    async def preload_model(self, context: ProcessingContext) -> None:
        self._ensure_supported_platform()
        load_key = self._load_key()

        if self._model is not None and self._model_id_loaded == load_key:
            return

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model_sync)
        self._model_id_loaded = load_key

    class OutputType(TypedDict):
        audio: AudioRef

    async def process(self, context: ProcessingContext) -> OutputType:
        self._ensure_supported_platform()
        import numpy as np

        if self.audio is None or not self.audio.is_set():
            raise ValueError("An input audio asset is required for enhancement.")

        if self._model is None or self._model_id_loaded != self._load_key():
            await self.preload_model(context)

        samples, _, _ = await context.audio_to_numpy(
            self.audio, sample_rate=self._sample_rate, mono=True
        )
        mono = samples.flatten().astype(np.float32)

        loop = asyncio.get_running_loop()
        enhanced = await loop.run_in_executor(None, self._enhance_sync, mono)

        enhanced_np = np.asarray(enhanced, dtype=np.float32).flatten()
        audio_ref = await context.audio_from_numpy(enhanced_np, self._sample_rate)
        return {"audio": audio_ref}

    @staticmethod
    def _resolve_cached(model_id: str, filename: str) -> str | None:
        from nodetool.nodes.mlx._hf_cache import find_cached_snapshot

        snapshot = find_cached_snapshot(model_id, filename)
        return str(snapshot) if snapshot is not None else None


class DeepFilterNet(BaseMLXSpeechEnhancement):
    """
    Suppress background noise using DeepFilterNet via MLX.
    deepfilternet, mlx, denoise, speech enhancement

    - Real-time capable noise suppression at 48 kHz
    - Runs locally on Apple Silicon through mlx-audio
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        DEEPFILTERNET = "mlx-community/DeepFilterNet-mlx"

    class Version(str, Enum):
        V1 = "1"
        V2 = "2"
        V3 = "3"

    model: HFAudioToAudio = Field(
        default=HFAudioToAudio(repo_id=Model.DEEPFILTERNET.value),
        description="DeepFilterNet model repository.",
    )
    version: Version = Field(
        default=Version.V3,
        description="DeepFilterNet version to use (v1, v2, or v3).",
    )

    def _load_key(self) -> str:
        # The version selects different weights within the same repo, so it must
        # invalidate the loaded model too.
        return f"{self._get_model_id()}#v{self.version.value}"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "version", "audio"]

    @classmethod
    def get_title(cls):
        return "DeepFilterNet"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFAudioToAudio(repo_id=cls.Model.DEEPFILTERNET.value)]

    def _load_model_sync(self) -> Any:
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

        model_id = self._get_model_id()
        version = int(self.version.value)
        if self._resolve_cached(model_id, f"v{version}/config.json") is None:
            raise ValueError(
                f"Model {model_id} (v{version}) must be downloaded first, "
                "check recommended models"
            )
        log.info("Loading DeepFilterNet model %s v%s", model_id, version)
        return DeepFilterNetModel.from_pretrained(model_id, version=version)

    def _enhance_sync(self, samples: "np.ndarray") -> "np.ndarray":
        return self._model.enhance_array(samples)


class MossFormer2(BaseMLXSpeechEnhancement):
    """
    Enhance speech using MossFormer2 via MLX.
    mossformer2, mlx, denoise, speech enhancement

    - High-quality 48 kHz speech enhancement
    - Runs locally on Apple Silicon through mlx-audio
    """

    _expose_as_tool: ClassVar[bool] = True

    class Model(str, Enum):
        # ``starkdmi/MossFormer2_SE_48K_MLX`` is the legacy layout: a bare bag of
        # ``model_*.safetensors`` with no ``config.json``, which mlx-audio cannot
        # load. The ``MossFormer2-SE*`` repos are the maintained per-precision
        # conversions and each ship ``config.json`` + ``model.safetensors``.
        MOSSFORMER2_SE = "starkdmi/MossFormer2-SE"
        MOSSFORMER2_SE_FP16 = "starkdmi/MossFormer2-SE-fp16"
        MOSSFORMER2_SE_8BIT = "starkdmi/MossFormer2-SE-8bit"
        MOSSFORMER2_SE_6BIT = "starkdmi/MossFormer2-SE-6bit"
        MOSSFORMER2_SE_4BIT = "starkdmi/MossFormer2-SE-4bit"

    model: HFAudioToAudio = Field(
        default=HFAudioToAudio(repo_id=Model.MOSSFORMER2_SE_FP16.value),
        description="MossFormer2 speech enhancement model variant.",
    )

    @classmethod
    def get_title(cls):
        return "MossFormer2"

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [HFAudioToAudio(repo_id=m.value) for m in cls.Model]

    def _load_model_sync(self) -> Any:
        from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

        model_id = self._get_model_id()
        if self._resolve_cached(model_id, "config.json") is None:
            raise ValueError(
                f"Model {model_id} must be downloaded first, check recommended models"
            )
        log.info("Loading MossFormer2 model %s", model_id)
        return MossFormer2SEModel.from_pretrained(model_id)

    def _enhance_sync(self, samples: "np.ndarray") -> "np.ndarray":
        return self._model.enhance(samples)
