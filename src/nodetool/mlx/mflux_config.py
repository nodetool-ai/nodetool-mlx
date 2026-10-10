"""Resolve an mflux ``ModelConfig`` for a repo id without losing base settings.

``ModelConfig.from_name`` resolves a repo id that is not a registered name (for
example ``mflux-community/qwen-image-mflux-q4``) by copying the matching base
config under the new name. mflux 0.18.1 copies only a fixed list of fields and
drops the rest: the sigma schedule (Qwen-Image's ``sigma_max_shift=0.9``,
``sigma_shift_terminal=0.02``), FLUX.2 Klein's ``text_encoder_overrides``, and
the LoRA and KV-cache settings. Later mflux releases copy every field, so the
restore below is a no-op there.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mflux.models.common.config import ModelConfig

_IDENTITY_FIELDS = {"model_name", "base_model"}


def mflux_model_config(repo_id: str) -> "ModelConfig":
    """Return a private ``ModelConfig`` for ``repo_id`` carrying every base field.

    The result is always a copy, so callers may set fields such as
    ``controlnet_model`` without changing mflux's process-wide registry.
    """
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    config = copy.copy(ModelConfig.from_name(repo_id))
    if config.base_model is None:
        return config

    # Mirror the resolver's preference: a plain base before a ControlNet
    # derivative that shares its model_name.
    candidates = sorted(
        (
            entry
            for entry in AVAILABLE_MODELS.values()
            if entry.base_model is None and entry.model_name == config.base_model
        ),
        key=lambda entry: (
            entry.controlnet_model != config.controlnet_model,
            entry.priority,
        ),
    )
    if not candidates:
        return config
    for key, value in vars(candidates[0]).items():
        if key.startswith("_") or key in _IDENTITY_FIELDS:
            continue
        setattr(config, key, copy.deepcopy(value))
    return config
