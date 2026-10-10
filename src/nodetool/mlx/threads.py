"""Dedicated threads for MLX work that must stay on the thread that loaded it.

MLX binds a Metal stream per thread. A model loaded on one thread and used on
another fails with "There is no Stream(gpu, N) in current thread." Load and run
each library's models on one of these single-worker executors. A single worker
also serializes Metal access, which these models want anyway.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

# mlx-audio (TTS, STT, speech enhancement) and mlx-whisper.
MLX_AUDIO_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-audio")

# mlx-vlm vision-language models.
MLX_VLM_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-vlm")
