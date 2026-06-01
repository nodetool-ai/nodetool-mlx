"""Weight manifest + Hugging Face resolver for the Stable Audio 3 MLX runtime.

Every weight file lives in the ``stabilityai/stable-audio-3-optimized`` repo
under its ``MLX/`` folder. We resolve files through ``hf_hub_download`` which
returns the cached path when present and downloads on demand otherwise — so the
node works whether the user pre-fetched the repo via NodeTool's Models Manager
or not.

Adapted from the upstream ``scripts/weights.py`` (MIT, Stability AI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

REPO_ID = "stabilityai/stable-audio-3-optimized"

# Logical component name -> path inside the Hugging Face repo.
FILES: dict[str, str] = {
    "dit_sm-music": "MLX/dit_sm-music_f16.npz",
    "dit_sm-sfx": "MLX/dit_sm-sfx_f16.npz",
    "dit_medium": "MLX/dit_medium_f16.npz",
    "same_s_decoder": "MLX/same_s_decoder_f32.npz",
    "same_l_decoder": "MLX/same_l_decoder_f32.npz",
    "same_s_encoder": "MLX/same_s_encoder_f32.npz",
    "same_l_encoder": "MLX/same_l_encoder_f32.npz",
    "t5gemma": "MLX/t5gemma_f16.npz",
}

# Only the MLX weights are needed at runtime — used as ``allow_patterns`` so the
# Models Manager doesn't pull the (much larger) original PyTorch checkpoints.
ALLOW_PATTERNS = ["MLX/*"]


def hf_filename(key: str) -> str:
    """Return the in-repo path for a logical component key."""
    try:
        return FILES[key]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(
            f"unknown Stable Audio 3 weight key {key!r}; "
            f"expected one of {sorted(FILES)}"
        ) from exc


def ensure_local(key: str) -> "Path":
    """Resolve ``key`` to a local path, downloading from Hugging Face if missing."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=REPO_ID, filename=hf_filename(key))
    return Path(cached)


def is_present(key: str) -> bool:
    """True if the weight file is already in the Hugging Face cache (no download)."""
    from huggingface_hub import try_to_load_from_cache

    result = try_to_load_from_cache(REPO_ID, hf_filename(key))
    return isinstance(result, str)
