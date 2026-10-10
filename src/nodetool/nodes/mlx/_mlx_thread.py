"""Run all MLX work on one dedicated thread.

MLX >= 0.32 only lets the thread that created lazily loaded arrays and streams
evaluate them. asyncio's default executor is a multi-thread pool, so a model
loaded on one worker thread and used on another fails with
"There is no Stream(cpu, N) in current thread". Routing every MLX call through
this single-thread executor keeps model loading and inference on the same thread.
"""

from concurrent.futures import ThreadPoolExecutor

MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
