"""
Memory estimation utilities for MFlux image generation.

This module provides conservative, heuristic-based memory estimates
for MFlux operations. Precision is not required - safety is the goal.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def estimate_mflux_memory_bytes(
    width: int,
    height: int,
    steps: int,
    quantize_bits: Optional[int],
    low_memory: bool,
    model_type: str = "flux",
) -> int:
    """
    Estimate memory usage for an MFlux image generation job.

    This is a conservative, heuristic-based estimate that intentionally
    overestimates to prevent memory exhaustion. It accounts for:
    - Model weights (reduced by quantization)
    - Latent activations (pixel count * steps)
    - VAE decode memory (reduced by tiling)
    - Safety margin

    Args:
        width: Output image width in pixels
        height: Output image height in pixels
        steps: Number of inference steps
        quantize_bits: Quantization level (3, 4, 5, 6, 8) or None for fp16
        low_memory: Whether VAE tiling is enabled
        model_type: Model family hint ("flux", "flux-fill", "flux-depth", etc.)

    Returns:
        Estimated memory usage in bytes (conservative overestimate)
    """
    # Base model memory (unquantized FLUX is ~23GB for dev, ~13GB for schnell)
    # We assume dev model as conservative estimate
    base_model_memory_gb = 23.0

    # Quantization reduces model weight memory
    if quantize_bits is not None:
        if quantize_bits == 3:
            quant_factor = 0.25  # ~25% of original
        elif quantize_bits == 4:
            quant_factor = 0.30  # ~30% of original
        elif quantize_bits == 5:
            quant_factor = 0.35  # ~35% of original
        elif quantize_bits == 6:
            quant_factor = 0.40  # ~40% of original
        elif quantize_bits == 8:
            quant_factor = 0.55  # ~55% of original
        else:
            quant_factor = 1.0  # Unknown quantization, assume no reduction
    else:
        quant_factor = 1.0  # fp16

    model_memory_bytes = base_model_memory_gb * quant_factor * (1024**3)

    # Latent activations memory
    # FLUX uses latent space that's 1/8 resolution with 16 channels
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = 16

    # Memory per step (in fp16 = 2 bytes per value)
    # We need to hold multiple intermediate tensors, so multiply by factor
    activation_factor = 4.0  # Conservative multiplier for intermediate tensors
    latent_memory_per_step = (
        latent_height * latent_width * latent_channels * 2 * activation_factor
    )
    activation_memory_bytes = latent_memory_per_step * steps

    # VAE decode memory
    # The VAE decoder needs to hold the full-resolution image tensor
    # Plus intermediate activations in the decoder
    vae_decode_factor = 8.0 if not low_memory else 2.0  # Tiling reduces peak memory
    vae_memory_bytes = height * width * 3 * 2 * vae_decode_factor  # RGB, fp16

    # Additional overhead for auxiliary models (ControlNet, Depth, etc.)
    auxiliary_memory_gb = 0.0
    if "controlnet" in model_type.lower():
        auxiliary_memory_gb = 2.0
    elif "depth" in model_type.lower():
        auxiliary_memory_gb = 2.0
    elif "redux" in model_type.lower():
        auxiliary_memory_gb = 3.0
    elif "fill" in model_type.lower():
        auxiliary_memory_gb = 1.5

    auxiliary_memory_bytes = auxiliary_memory_gb * (1024**3)

    # Total estimate
    total_estimate = (
        model_memory_bytes
        + activation_memory_bytes
        + vae_memory_bytes
        + auxiliary_memory_bytes
    )

    # Add 30% safety margin
    safety_margin = 1.30
    final_estimate = int(total_estimate * safety_margin)

    log.debug(
        f"Memory estimate: model={model_memory_bytes / (1024**3):.1f}GB, "
        f"activations={activation_memory_bytes / (1024**3):.1f}GB, "
        f"vae={vae_memory_bytes / (1024**3):.1f}GB, "
        f"auxiliary={auxiliary_memory_bytes / (1024**3):.1f}GB, "
        f"total={final_estimate / (1024**3):.1f}GB (with {safety_margin}x margin)"
    )

    return final_estimate


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_memory_error(
    estimated_bytes: int,
    available_bytes: int,
    required_headroom_bytes: int,
    width: int,
    height: int,
    steps: int,
    low_memory: bool,
) -> str:
    """
    Format a user-friendly error message for insufficient memory.

    Args:
        estimated_bytes: Estimated memory needed for the job
        available_bytes: Available system memory
        required_headroom_bytes: Required safety headroom
        width: Requested image width
        height: Requested image height
        steps: Number of inference steps
        low_memory: Whether low-memory mode is enabled

    Returns:
        Formatted error message with suggestions
    """
    total_required = estimated_bytes + required_headroom_bytes
    deficit = total_required - available_bytes

    message_parts = [
        "This job is likely to exhaust system memory and freeze macOS.",
        "",
        "Memory Analysis:",
        f"  • Estimated usage: {format_bytes(estimated_bytes)}",
        f"  • Available memory: {format_bytes(available_bytes)}",
        f"  • Required headroom: {format_bytes(required_headroom_bytes)} (10% safety margin)",
        f"  • Total required: {format_bytes(total_required)}",
        f"  • Deficit: {format_bytes(deficit)}",
        "",
        "Configuration:",
        f"  • Resolution: {width}x{height}",
        f"  • Steps: {steps}",
        f"  • Low-memory mode: {'enabled' if low_memory else 'disabled'}",
        "",
        "Suggested Actions:",
    ]

    # Provide specific suggestions based on configuration
    if not low_memory:
        message_parts.append(
            "  1. Enable Low-Memory mode (VAE tiling) - reduces peak memory by ~4x"
        )

    if width * height > 1024 * 1024:
        message_parts.append(
            f"  2. Reduce resolution (try {width // 2}x{height // 2} or 1024x1024)"
        )

    if steps > 10:
        message_parts.append(f"  3. Reduce inference steps (try {max(4, steps // 2)})")

    message_parts.extend(
        [
            "  4. Use higher quantization (4-bit or 3-bit models)",
            "  5. Close other applications to free system memory",
            "",
            "Note: This is a conservative estimate. The actual memory usage may be lower,",
            "but running the job risks freezing your system if memory is exhausted.",
        ]
    )

    return "\n".join(message_parts)
