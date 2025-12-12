"""
Centralized Flux model loading utilities for MLX.

This module provides a unified way to load Flux models across both nodes and image providers,
with proper caching and availability checks.
"""

import asyncio
from typing import Any
from mflux.generate import Flux1
from mflux.generate_controlnet import Flux1Controlnet
from nodetool.ml.core.model_manager import ModelManager
from nodetool.integrations.huggingface.hf_cache import has_cached_files
from nodetool.config.logging_config import get_logger
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

log = get_logger(__name__)

# Estimated memory requirements in GB for different quantization levels
# These are rough estimates for Flux.1 models (approx 12B params)
# 4-bit: ~12GB (model) + overhead
# 8-bit: ~20GB (model) + overhead
# 16-bit: ~34GB (model) + overhead
FLUX_MEMORY_ESTIMATES = {
    4: 12.0,
    8: 20.0,
    None: 34.0,  # Default/16-bit
}

def estimate_required_memory(quantize: int | None, is_controlnet: bool = False) -> float:
    """Estimate the required memory in GB for loading the model."""
    base_mem = FLUX_MEMORY_ESTIMATES.get(quantize, 34.0)
    
    # If quantization is not one of the standard values, estimate linearly
    if quantize not in FLUX_MEMORY_ESTIMATES and quantize is not None:
        # Rough linear interpolation: ~1.5GB per bit for 12B params + overhead
        base_mem = quantize * 2.5 
        
    if is_controlnet:
        # ControlNet adds extra parameters (approx 2-3GB for Flux ControlNet)
        base_mem += 3.0
        
    return base_mem

def check_memory_availability(required_gb: float):
    """
    Check if there is enough available RAM to load the model.
    Raises MemoryError if insufficient memory.
    """
    import psutil
    vm = psutil.virtual_memory()
    
    # Calculate available memory in GB
    # available includes memory that can be reclaimed (cache/buffers)
    available_gb = vm.available / (1024**3)
    total_gb = vm.total / (1024**3)
    
    # 5% safety buffer
    buffer_gb = total_gb * 0.05
    
    if available_gb < (required_gb + buffer_gb):
        raise MemoryError(
            f"Insufficient memory to load Flux model.\n"
            f"Required: {required_gb:.1f} GB\n"
            f"Available: {available_gb:.1f} GB\n"
            f"Safety Buffer: {buffer_gb:.1f} GB (5% of total)\n"
            f"Total System RAM: {total_gb:.1f} GB\n\n"
            f"Please close other applications or try a higher quantization level (e.g., 4-bit)."
        )
        
    log.info(f"Memory check passed: {available_gb:.1f} GB available, {required_gb:.1f} GB required")
    



class FluxModelNotAvailableError(Exception):
    """Raised when a Flux model is not available in the local cache."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(
            f"Model '{model_id}' is not available in your local cache.\n\n"
            f"To use this model:\n"
            f"1. Open the Model Manager (in the NodeTool UI)\n"
            f"2. Search for '{model_id}'\n"
            f"3. Download the model\n"
            f"4. Try again once the download is complete\n\n"
            f"Note: MLX models are only loaded from local cache and will not be downloaded automatically "
            f"to avoid unexpected network usage and storage consumption."
        )


def is_flux_model_available(model_id: str) -> bool:
    """
    Check if a Flux model is available in the local HuggingFace cache.

    Args:
        model_id: The HuggingFace repo ID (e.g., "apple/FLUX.1-schnell-4bit")

    Returns:
        True if the model is cached locally, False otherwise
    """
    return has_cached_files(model_id)


async def load_flux_model(
    model_id: str,
    quantize: int | None = 4,
    node_id: str | None = None,
    task: str = "flux",
    force_reload: bool = False,
) -> Flux1:
    """
    Load a Flux model with proper caching and availability checks.

    This function:
    1. Checks if the model is available in the HuggingFace cache
    2. Checks the ModelManager cache
    3. Only loads from local files (does not download)
    4. Caches the loaded model for reuse

    Args:
        model_id: The HuggingFace repo ID (e.g., "apple/FLUX.1-schnell-4bit")
        quantize: Quantization level (3, 4, 5, 6, 8, or None)
        node_id: Optional node ID for ModelManager association
        task: Task identifier for ModelManager (default: "flux")
        force_reload: If True, bypass ModelManager cache and reload from disk

    Returns:
        The loaded Flux1 model instance

    Raises:
        RuntimeError: If mflux is not available or not on macOS
        FluxModelNotAvailableError: If the model is not in the local cache
    """
    # Check if model is available in HF cache
    if not is_flux_model_available(model_id):
        raise FluxModelNotAvailableError(model_id)

    # Construct cache key
    cache_key = f"{model_id}_{task}"

    # Check ModelManager cache (unless forcing reload)
    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux model: {model_id}")
            return cached_model

    # Load model from local cache
    required_mem = estimate_required_memory(quantize)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1:
        log.info(
            f"Loading Flux model {model_id} from local cache "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model = Flux1.from_name(
            model_name=model_id,
            quantize=quantize,
        )
        return model

    model = await loop.run_in_executor(None, _load)

    # Cache in ModelManager if node_id provided
    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model


async def load_flux_controlnet_model(
    base_model_id: str,
    controlnet_model_id: str,
    quantize: int | None = 4,
    node_id: str | None = None,
    force_reload: bool = False,
) -> Flux1Controlnet:
    """
    Load a Flux ControlNet model with proper caching and availability checks.

    Args:
        base_model_id: Base Flux model repo ID
        controlnet_model_id: ControlNet model repo ID
        quantize: Quantization level
        node_id: Optional node ID for ModelManager association
        force_reload: If True, bypass ModelManager cache

    Returns:
        The loaded Flux1Controlnet model instance

    Raises:
        RuntimeError: If mflux is not available or not on macOS
        FluxModelNotAvailableError: If either model is not in the local cache
    """
    # Check if both models are available
    if not is_flux_model_available(base_model_id):
        raise FluxModelNotAvailableError(base_model_id)
    if not is_flux_model_available(controlnet_model_id):
        raise FluxModelNotAvailableError(controlnet_model_id)

    # Construct cache key
    cache_key = f"{base_model_id}:{controlnet_model_id}_flux-controlnet"

    # Check ModelManager cache
    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux ControlNet model: {cache_key}")
            return cached_model

    required_mem = estimate_required_memory(quantize, is_controlnet=True)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1Controlnet:
        log.info(
            f"Loading Flux ControlNet model {base_model_id} with controlnet {controlnet_model_id} "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model_config = ModelConfig.from_name(base_model_id)
        model_config.controlnet_model = controlnet_model_id

        model = Flux1Controlnet(
            model_config=model_config,
            quantize=quantize,
        )
        return model

    model = await loop.run_in_executor(None, _load)

    # Cache in ModelManager
    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model


async def load_flux_fill_model(
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
    quantize: int | None = 4,
    node_id: str | None = None,
    force_reload: bool = False,
) -> Flux1Fill:
    """
    Load a Flux Fill (inpainting/outpainting) model.

    Args:
        model_id: Fill model repo ID
        quantize: Quantization level
        node_id: Optional node ID for ModelManager association
        force_reload: If True, bypass ModelManager cache

    Returns:
        The loaded Flux1Fill model instance
    """
    if not is_flux_model_available(model_id):
        raise FluxModelNotAvailableError(model_id)

    # Construct cache key
    cache_key = f"{model_id}_flux-fill"

    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux Fill model: {model_id}")
            return cached_model

    required_mem = estimate_required_memory(quantize)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1Fill:
        log.info(
            f"Loading Flux Fill model {model_id} "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model = Flux1Fill(quantize=quantize)
        return model

    model = await loop.run_in_executor(None, _load)

    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model


async def load_flux_depth_model(
    model_id: str = "black-forest-labs/FLUX.1-Depth-dev",
    quantize: int | None = 4,
    node_id: str | None = None,
    force_reload: bool = False,
) -> Flux1Depth:
    """Load a Flux Depth model."""
    if not is_flux_model_available(model_id):
        raise FluxModelNotAvailableError(model_id)

    # Construct cache key
    cache_key = f"{model_id}_flux-depth"

    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux Depth model: {model_id}")
            return cached_model

    required_mem = estimate_required_memory(quantize)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1Depth:
        log.info(
            f"Loading Flux Depth model {model_id} "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model = Flux1Depth(quantize=quantize)
        return model

    model = await loop.run_in_executor(None, _load)

    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model


async def load_flux_redux_model(
    model_id: str = "black-forest-labs/FLUX.1-Redux-dev",
    quantize: int | None = 4,
    node_id: str | None = None,
    force_reload: bool = False,
) -> Flux1Redux:
    """Load a Flux Redux model."""
    if not is_flux_model_available(model_id):
        raise FluxModelNotAvailableError(model_id)

    # Construct cache key
    cache_key = f"{model_id}_flux-redux"

    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux Redux model: {model_id}")
            return cached_model

    required_mem = estimate_required_memory(quantize)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1Redux:
        log.info(
            f"Loading Flux Redux model {model_id} "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model_config = ModelConfig.dev_redux()
        model = Flux1Redux(
            model_config=model_config,
            quantize=quantize,
        )
        return model

    model = await loop.run_in_executor(None, _load)

    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model


async def load_flux_kontext_model(
    model_id: str = "black-forest-labs/FLUX.1-Kontext-dev",
    quantize: int | None = 4,
    node_id: str | None = None,
    force_reload: bool = False,
) -> Flux1Kontext:
    """Load a Flux Kontext model."""
    if not is_flux_model_available(model_id):
        raise FluxModelNotAvailableError(model_id)

    # Construct cache key
    cache_key = f"{model_id}_flux-kontext"

    if not force_reload:
        cached_model = ModelManager.get_model(cache_key)
        if cached_model is not None:
            log.info(f"Using cached Flux Kontext model: {model_id}")
            return cached_model

    required_mem = estimate_required_memory(quantize)
    check_memory_availability(required_mem)

    loop = asyncio.get_running_loop()

    def _load() -> Flux1Kontext:
        log.info(
            f"Loading Flux Kontext model {model_id} "
            f"(quantize={quantize if quantize is not None else 'none'})"
        )
        model = Flux1Kontext(quantize=quantize)
        return model

    model = await loop.run_in_executor(None, _load)

    if node_id:
        ModelManager.set_model(node_id, cache_key, model)

    return model
