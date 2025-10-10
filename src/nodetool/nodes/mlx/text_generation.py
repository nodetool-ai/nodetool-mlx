import sys
from typing import AsyncGenerator, ClassVar, TypedDict

from pydantic import Field

from nodetool.metadata.types import HFTextGeneration, LanguageModel, Message, Provider, ToolCall
from nodetool.providers import get_provider
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

DEFAULT_MLX_MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"


class TextGeneration(BaseNode):
    """
    Generate text using locally cached MLX models on Apple Silicon.
    text, generation, mlx, local-llm

    Use cases:
    - Fast local iteration with MLX-converted instruction models
    - Privacy-preserving summarization, drafting, or Q&A
    - Building multi-node agent workflows without external APIs
    """

    _provider: ClassVar[Provider] = Provider.MLX

    model: LanguageModel = Field(
        default=LanguageModel(
            provider=Provider.MLX,
            id=DEFAULT_MLX_MODEL_ID,
            name="Llama-3.2-3B Instruct (4-bit)",
        ),
        title="Model",
        description="MLX language model to use for generation. The model must be available in the local Hugging Face cache.",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        title="System Prompt",
        description="Optional system prompt used to steer the model.",
    )
    prompt: str = Field(
        default="",
        title="Prompt",
        description="The user prompt used for generation.",
    )
    max_new_tokens: int = Field(
        default=256,
        title="Max New Tokens",
        ge=1,
        le=4096,
        description="Maximum number of tokens the model should generate.",
    )
    temperature: float = Field(
        default=0.7,
        title="Temperature",
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Set to 0 for deterministic output.",
    )
    top_p: float = Field(
        default=0.95,
        title="Top P",
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass.",
    )
    top_k: int = Field(
        default=50,
        title="Top K",
        ge=0,
        le=1000,
        description="Limits sampling to the top K tokens at each step.",
    )
    min_p: float = Field(
        default=0.0,
        title="Min P",
        ge=0.0,
        le=1.0,
        description="Minimum token probability threshold (0 disables).",
    )
    min_tokens_to_keep: int = Field(
        default=1,
        title="Min Tokens To Keep",
        ge=1,
        le=20,
        description="Ensures at least this many tokens remain after filtering.",
    )

    class OutputType(TypedDict):
        text: str | None
        chunk: Chunk | None

    @classmethod
    def get_title(cls) -> str:
        return "MLX Text Generation"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        # MLX runs on Apple Silicon accelerators but does not require an external GPU.
        return False

    @staticmethod
    def _ensure_supported_platform() -> None:
        if sys.platform != "darwin":
            raise RuntimeError("MLX text generation requires macOS (Apple Silicon).")

    def _build_messages(self) -> list[Message]:
        system_prompt = self.system_prompt.strip()
        user_prompt = self.prompt.strip()
        if not user_prompt:
            raise ValueError("Prompt must not be empty.")

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_prompt))
        return messages

    def _model_id(self) -> str:
        if not self.model.id:
            raise ValueError("Select an MLX model before running generation.")
        return self.model.id

    def _generation_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "min_tokens_to_keep": self.min_tokens_to_keep,
        }

    @classmethod
    def get_recommended_models(cls):
        return [
            # Existing Llama model
            HFTextGeneration(
                repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
            ),
            # Gemma 3 1B models - lightweight and fast
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-1b-it-8bit",
            ),
            # Gemma 3 4B models - balanced size/performance
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-4b-it-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-4b-it-8bit",
            ),
            # Gemma 3 12B models - high quality
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-12b-it-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-12b-it-8bit",
            ),
            # Gemma 3 27B models - maximum quality
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-27b-it-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/gemma-3-27b-it-8bit",
            ),
            # Granite 4.0 models - IBM's high-quality multilingual models
            # Tiny/Micro models (1B) - lightweight and fast
            HFTextGeneration(
                repo_id="mlx-community/Granite-4.0-H-Tiny-4bit-DWQ",
            ),
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-h-micro-8bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-tiny-preview-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-micro-8bit",
            ),
            # Medium models (7B) - balanced size/performance
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-h-tiny-8bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-h-tiny-6bit-MLX",
            ),
            # Large models (32B) - high quality
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-h-small-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/granite-4.0-h-small-8bit",
            ),
            # Qwen3 models - Alibaba's latest multilingual models
            # Ultra-small models (0.6B) - extremely lightweight
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-0.6B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-0.6B-8bit",
            ),
            # Small models (1.7B) - good balance of speed and quality
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-1.7B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-1.7B-8bit",
            ),
            # Medium-small models (4B) - versatile performance
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-4B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-4B-8bit",
            ),
            # Medium models (8B) - high quality with good speed
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-8B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-8B-6bit",
            ),
            # Large models (30B) - maximum quality and capability
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-30B-A3B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/Qwen3-30B-A3B-8bit",
            ),
            # LFM2-8B-A1B-MoE models - Best-in-class Mixture of Experts, optimized for Apple Silicon
            # Optimized for smaller devices (sub 16GB M1/M2/M3/M4)
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-3bit-MLX",
            ),
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-4bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-5bit",
            ),
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-6bit-MLX",
            ),
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-8bit-MLX",
            ),
            HFTextGeneration(
                repo_id="mlx-community/LFM2-8B-A1B-fp16",
            ),
            HFTextGeneration(
                repo_id="nightmedia/LFM2-8B-A1B-qx86-hi-mlx",
            ),
        ]

    async def preload_model(self, context: ProcessingContext) -> None:
        # Verify platform early to provide a clear error before generation begins.
        self._ensure_supported_platform()

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        self._ensure_supported_platform()
        model_id = self._model_id()
        messages = self._build_messages()
        provider = get_provider(self._provider)

        context.post_message(
            JobUpdate(
                status="running",
                message=f"Generating text with {model_id}",
            )
        )

        full_text = ""

        async for item in provider.generate_messages(
            messages=messages,
            model=model_id,
            max_tokens=self.max_new_tokens,
            **self._generation_kwargs(),
        ):
            if isinstance(item, ToolCall):
                # The MLX provider can emulate tool calls, but this node only returns text.
                continue

            if isinstance(item, Chunk):
                if item.content:
                    full_text += item.content
                    yield {
                        "text": None,
                        "chunk": Chunk(
                            content=item.content,
                            done=False,
                            content_type=item.content_type or "text",
                            content_metadata=item.content_metadata or {},
                        ),
                    }

                if item.done:
                    break

        yield {
            "text": full_text,
            "chunk": Chunk(content="", done=True, content_type="text"),
        }