"""Reusable Stable Audio 3 generation pipeline (pure MLX).

A thin, NodeTool-friendly orchestration layer over the vendored ``defs/``
modules. ``generate()`` mirrors the upstream ``scripts/sa3_mlx.py`` ``main()``
flow — T5Gemma → conditioning → DiT ping-pong sampling → SAME decode → unpatch
— but returns the audio as a numpy array instead of printing/writing a WAV, and
caches loaded models through NodeTool's ``ModelManager`` so repeated runs are
fast.

All heavy imports (mlx, the vendored model defs) are deferred into ``generate``
so this module can be imported on any platform (e.g. for metadata scanning on
Linux) — actual generation only runs on Apple Silicon.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

from . import weights

SAMPLE_RATE = 44100
# PatchedPretransform downsample (×256) × SAME 16× expansion.
SAMPLES_PER_LATENT = 4096

# DiT family → (vendored module name, weight key, default decoder).
DIT_CHOICES: dict[str, dict[str, str]] = {
    "sm-music": {"module": "dit_mlx", "weight": "dit_sm-music", "decoder": "same-s"},
    "sm-sfx": {"module": "dit_mlx", "weight": "dit_sm-sfx", "decoder": "same-s"},
    "medium": {"module": "dit_mlx_medium", "weight": "dit_medium", "decoder": "same-l"},
}
# Decoder → (module, chunk_size, overlap, weight key).
DECODER_CHOICES: dict[str, tuple[str, int, int, str]] = {
    "same-s": ("same_s_decoder", 8, 2, "same_s_decoder"),
    "same-l": ("same_l_decoder", 128, 8, "same_l_decoder"),
}
# Decoder → (encoder module, pad modulo, weight key) for --init-audio modes.
ENCODER_CHOICES: dict[str, tuple[str, int, str]] = {
    "same-s": ("same_s_encoder", 32, "same_s_encoder"),
    "same-l": ("same_l_encoder", 16, "same_l_encoder"),
}

# σmax floor — the rf_denoiser model is undefined at t≈0 and emits NaN below this.
MIN_SIGMA = 0.01


def default_decoder(dit: str) -> str:
    """The decoder paired with a DiT family."""
    return DIT_CHOICES[dit]["decoder"]


def weight_keys(dit: str, with_encoder: bool = False) -> list[str]:
    """Logical weight keys needed to run a given DiT (for pre-download)."""
    decoder = default_decoder(dit)
    keys = ["t5gemma", DIT_CHOICES[dit]["weight"], DECODER_CHOICES[decoder][3]]
    if with_encoder:
        keys.append(ENCODER_CHOICES[decoder][2])
    return keys


def _import_def(module: str):
    import importlib

    return importlib.import_module(f"{__package__}.defs.{module}")


def _load_t5gemma(node_id: str):
    from nodetool.ml.core.model_manager import ModelManager

    key = "sa3_t5gemma"
    model = ModelManager.get_model(key)
    if model is not None:
        return model
    from .defs.t5gemma_mlx import T5Gemma

    model = T5Gemma.from_npz(str(weights.ensure_local("t5gemma")))
    ModelManager.set_model(node_id, key, model)
    return model


def _load_dit(node_id: str, dit: str, t_lat: int, dtype):
    import mlx.core as mx

    from nodetool.ml.core.model_manager import ModelManager

    cfg = DIT_CHOICES[dit]
    dtype_name = "fp16" if dtype == mx.float16 else "fp32"
    key = f"sa3_dit_{dit}_{dtype_name}"
    model = ModelManager.get_model(key)
    if model is None:
        mod = _import_def(cfg["module"])
        model = mod.load_dit(
            str(weights.ensure_local(cfg["weight"])),
            T_lat=t_lat,
            dtype=dtype,
            compile_=False,
        )
        ModelManager.set_model(node_id, key, model)
    # The only T_lat-dependent state is a constant zeros buffer; rebuild it so a
    # cached DiT can serve any requested duration.
    if getattr(model, "T_lat", None) != t_lat:
        model.T_lat = t_lat
        model._local_zeros_1 = mx.zeros((1, t_lat, model._local_zeros_1.shape[-1]))
    return model


def _load_decoder(node_id: str, decoder: str):
    import mlx.core as mx

    from nodetool.ml.core.model_manager import ModelManager

    module, chunk, ovl, weight = DECODER_CHOICES[decoder]
    key = f"sa3_dec_{decoder}"
    model = ModelManager.get_model(key)
    mod = _import_def(module)
    if model is None:
        # SAME decoders always run FP32 (FP16 catastrophically cancels at the
        # differential-attention bottleneck).
        model = mod.load_model(
            weights_path=str(weights.ensure_local(weight)),
            dtype=mx.float32,
            compile_=False,
        )
        ModelManager.set_model(node_id, key, model)
    return model, mod.decode_chunked, (chunk, ovl)


def _load_encoder(node_id: str, decoder: str):
    import mlx.core as mx

    from nodetool.ml.core.model_manager import ModelManager

    module, pad_modulo, weight = ENCODER_CHOICES[decoder]
    key = f"sa3_enc_{decoder}"
    model = ModelManager.get_model(key)
    if model is None:
        mod = _import_def(module)
        model = mod.load_model(
            weights_path=str(weights.ensure_local(weight)),
            dtype=mx.float32,
            compile_=False,
        )
        ModelManager.set_model(node_id, key, model)
    return model, pad_modulo


def _patch_audio(audio: "np.ndarray", patch_size: int = 256) -> "np.ndarray":
    """Patched-pretransform encode: (B, 2, T) → (B, 512, T/patch_size)."""
    B, C, T = audio.shape
    if T % patch_size != 0:
        raise ValueError(f"audio length {T} not a multiple of patch_size {patch_size}")
    L = T // patch_size
    x = audio.reshape(B, C, L, patch_size).transpose(0, 1, 3, 2)
    return x.reshape(B, C * patch_size, L)


def compute_t_lat(seconds: float, decoder: str) -> int:
    """Latent length for a requested duration, honoring SAME-S's even constraint."""
    t_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if decoder == "same-s" and t_lat % 2 != 0:
        t_lat += 1
    return t_lat


def generate(
    *,
    prompt: str,
    dit: str,
    seconds: float = 10.0,
    steps: int = 8,
    seed: int = 0,
    cfg: float = 1.0,
    apg: float = 1.0,
    negative_prompt: Optional[str] = None,
    dit_dtype: str = "fp16",
    init_audio: Optional["np.ndarray"] = None,
    init_noise_level: float = 1.0,
    inpaint_range: Optional[tuple[float, float]] = None,
    decoder: Optional[str] = None,
    on_step: Optional[Callable[[int, int], None]] = None,
    node_id: str = "stable_audio_3",
) -> tuple["np.ndarray", int]:
    """Generate audio with Stable Audio 3.

    Returns ``(audio, sample_rate)`` where ``audio`` is ``(2, num_samples)``
    float32 in ``[-1, 1]`` at 44.1 kHz, trimmed to ``seconds``.

    Modes are selected by the optional args:
      - text-to-audio : ``init_audio=None``
      - audio-to-audio: ``init_audio`` set, ``inpaint_range=None`` (use
        ``init_noise_level`` ∈ ~0.4–0.8 for variation)
      - inpainting    : ``init_audio`` set + ``inpaint_range=(start_s, end_s)``
    """
    import mlx.core as mx
    import numpy as np

    from .defs.sa3_pipeline import (
        apply_prompt_padding,
        build_pingpong_schedule,
        load_conditioner_from_npz,
        patched_decode,
        sample_flow_pingpong,
    )

    if dit not in DIT_CHOICES:
        raise ValueError(f"unknown dit {dit!r}; expected one of {sorted(DIT_CHOICES)}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1 (got {steps})")
    decoder = decoder or default_decoder(dit)

    dtype = mx.float32 if dit_dtype == "fp32" else mx.float16
    t_lat = compute_t_lat(seconds, decoder)

    sigma_max = float(init_noise_level)
    if sigma_max < MIN_SIGMA:
        raise ValueError(
            f"init_noise_level={sigma_max} too low (min {MIN_SIGMA}); the model "
            f"is undefined at t≈0. Use 1.0 for plain text-to-audio."
        )

    # Map an inpaint range in seconds to latent indices.
    inpaint_lat: Optional[tuple[int, int]] = None
    if inpaint_range is not None:
        if init_audio is None:
            raise ValueError("inpaint_range requires init_audio")
        start_s, end_s = inpaint_range
        if not (0 <= start_s < end_s <= seconds):
            raise ValueError(
                f"invalid inpaint range {start_s}-{end_s}s "
                f"(need 0 <= start < end <= {seconds})"
            )
        s0 = max(0, int(round(start_s * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(t_lat, int(round(end_s * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_lat = (s0, s1)

    # ── 1. Text encoder ──
    enc = _load_t5gemma(node_id)
    embeds, mask = enc.encode([prompt], max_len=256)
    mx.eval(embeds, mask)

    # ── 2. Conditioning (conditioner weights baked into the DiT npz) ──
    padding_emb, secs_embedder = load_conditioner_from_npz(
        str(weights.ensure_local(DIT_CHOICES[dit]["weight"])), prefix="cond."
    )
    embeds = embeds.astype(dtype)
    embeds_padded = apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
    seconds_embed = secs_embedder(seconds).astype(dtype)  # (1, 1, 768)
    cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)  # (1, 257, 768)
    global_cond = seconds_embed[:, 0, :]  # (1, 768)

    null_cross_attn = None
    if cfg != 1.0:
        if negative_prompt:
            neg_embeds, neg_mask = enc.encode([negative_prompt], max_len=256)
            mx.eval(neg_embeds, neg_mask)
            neg_padded = apply_prompt_padding(
                neg_embeds.astype(dtype), neg_mask, padding_emb.astype(dtype)
            )
            null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
        else:
            null_cross_attn = mx.zeros_like(cross_attn)
        mx.eval(null_cross_attn)
    mx.eval(cross_attn, global_cond)

    # ── 3a. (audio-to-audio / inpaint) encode init audio → latents ──
    init_latents = None
    if init_audio is not None:
        enc_model, pad_mod = _load_encoder(node_id, decoder)
        audio_in = np.asarray(init_audio, dtype=np.float32)
        if audio_in.ndim == 1:
            audio_in = np.stack([audio_in, audio_in], axis=0)
        if audio_in.shape[0] != 2:
            raise ValueError(f"init_audio must be (2, T); got shape {audio_in.shape}")
        target_samples = t_lat * SAMPLES_PER_LATENT
        if audio_in.shape[-1] >= target_samples:
            audio_in = audio_in[:, :target_samples]
        else:
            pad = target_samples - audio_in.shape[-1]
            audio_in = np.pad(audio_in, ((0, 0), (0, pad)), mode="constant")
        patches_np = _patch_audio(audio_in[None, ...], patch_size=256)
        if patches_np.shape[-1] % pad_mod != 0:
            raise ValueError(
                f"encoded patch length {patches_np.shape[-1]} not divisible by "
                f"{pad_mod} for decoder {decoder}"
            )
        init_latents = enc_model(mx.array(patches_np))
        mx.eval(init_latents)
        init_latents = init_latents.astype(dtype)

    # ── 3b. DiT ping-pong sampling ──
    dit_model = _load_dit(node_id, dit, t_lat, dtype)
    sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)

    key = mx.random.key(seed)
    pure_noise = mx.random.normal((1, 256, t_lat), dtype=dtype, key=key)
    if init_latents is not None and inpaint_lat is None:
        # rf_denoiser init mix: noise = init * (1 - σmax) + pure_noise * σmax
        noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
    else:
        noise = pure_noise
    mx.eval(noise)

    # Inpaint conditioning + paste-back.
    local_add_cond = None
    paste_back = None
    if inpaint_lat is not None:
        s0, s1 = inpaint_lat
        mask_np = np.ones((1, 1, t_lat), dtype=np.float32)
        mask_np[:, :, s0:s1] = 0.0  # 1 = keep, 0 = regenerate
        inpaint_mask = mx.array(mask_np)
        masked_input = init_latents.astype(mx.float32) * inpaint_mask
        local_add_cond = (
            mx.concatenate([inpaint_mask, masked_input], axis=1)
            .transpose(0, 2, 1)
            .astype(dtype)
        )
        paste_back = (init_latents, inpaint_mask)

    def model_fn(x, t):
        if cfg == 1.0:
            return dit_model(
                x, t, cross_attn, global_cond, local_add_cond=local_add_cond
            )

        # Batched classifier-free guidance over cat([cond, uncond]).
        x2 = mx.concatenate([x, x], axis=0)
        t2 = mx.concatenate([t, t], axis=0)
        cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
        global2 = mx.concatenate([global_cond, global_cond], axis=0)
        lac2 = (
            None
            if local_add_cond is None
            else mx.concatenate([local_add_cond, local_add_cond], axis=0)
        )
        v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
        cond_v, uncond_v = mx.split(v_batched, 2, axis=0)

        sigma = t.reshape(-1, 1, 1).astype(mx.float32)
        cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
        uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
        diff = cond_d - uncond_d

        if apg <= 0.0:
            cfg_diff = diff
        else:
            # Adaptive Projected Guidance — project diff orthogonal to cond_d.
            norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
            unit = cond_d / mx.maximum(norm, 1e-8)
            parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
            diff_orth = diff - parallel
            cfg_diff = (
                diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
            )

        cfg_d = cond_d + (cfg - 1.0) * cfg_diff
        cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
        return cfg_v.astype(x.dtype)

    latents = sample_flow_pingpong(
        model_fn, noise, sigmas, seed=seed + 1, paste_back=paste_back, on_step=on_step
    )
    mx.eval(latents)

    # ── 4. Decode latents → audio patches (decoder always FP32) ──
    decoder_model, chunk_fn, (chunk, ovl) = _load_decoder(node_id, decoder)
    latents_fp32 = latents.astype(mx.float32)
    kernel = chunk + 2 * ovl
    if t_lat > kernel:
        patches = chunk_fn(decoder_model, latents_fp32, chunk, ovl)
    elif t_lat % 2 == 0:
        patches = decoder_model(latents_fp32)
    else:
        patches = chunk_fn(decoder_model, latents_fp32, 2, 2)
    mx.eval(patches)

    # ── 5. Unpatch → (2, T) audio ──
    audio = patched_decode(patches, patch_size=256, channels=2)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32))[0]  # (2, T_lat * 4096)
    requested = int(round(seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested:
        audio_np = audio_np[..., :requested]
    return audio_np, SAMPLE_RATE
