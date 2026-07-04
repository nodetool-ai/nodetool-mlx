# Stable Audio 3 — vendored MLX runtime

The modules under `defs/` are vendored (with light packaging changes) from the
official Stability AI **Stable Audio 3** optimized MLX implementation:

- Upstream: https://github.com/Stability-AI/stable-audio-3/tree/main/optimized/mlx
- License: MIT (see `LICENSE`), Copyright (c) 2026 Stability AI

Vendored files (pure MLX, no PyTorch / transformers at runtime):

- `defs/t5gemma_mlx.py`    — T5Gemma text encoder
- `defs/dit_mlx.py`        — small DiT (sm-music / sm-sfx, 50M)
- `defs/dit_mlx_medium.py` — medium DiT (1.4B)
- `defs/same_s_decoder.py` / `defs/same_s_encoder.py` — SAME-S codec
- `defs/same_l_decoder.py` / `defs/same_l_encoder.py` — SAME-L codec
- `defs/sa3_pipeline.py`   — conditioning, ping-pong sampler, patched (de)patch

`weights.py` and `pipeline.py` are NodeTool-side glue (manifest + orchestration)
adapted from the upstream `scripts/weights.py` and `scripts/sa3_mlx.py`.

The model **weights** are downloaded on demand from the Hugging Face repo
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)
and are governed by their own license terms on that repository.
