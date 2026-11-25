"""MiniMax H3 video with sound on Apple Silicon, through Phosphene's pinned engine.

Powered by MiniMax H3. The engine is minimax-h3-mlx by Mr. Bizarro
(https://github.com/mrbizarro/minimax-h3-mlx, Apache-2.0), at the commit
Phosphene pins. It has no package metadata, so the first run downloads that
commit's archive into NodeTool's data folder and runs its driver script,
``scripts/generate_staged.py``, in a subprocess of the NodeTool Python. The
engine needs ``mlx``, ``mlx-vlm``, ``transformers``, ``safetensors``,
``huggingface_hub``, ``numpy`` and ``pillow``, which this pack already
installs, and ``ffmpeg`` on PATH.

The weights (about 75 GB) come from three Hugging Face repos listed as each
node's recommended models. The first run also quantizes the pruned bf16
transformer to the engine's Q8 pack (about 22 GB, about five minutes, one
tensor at a time), which halves the memory a render holds.

The weights are under the MiniMax H3 Community License. It asks that products
show "Powered by MiniMax H3", it needs separate permission for commercial use
above USD 20M in yearly revenue, and it excludes some territories. Read it on
the MiniMaxAI/MiniMax-H3 model card before using the output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from collections import deque
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.config.settings import get_system_data_path
from nodetool.metadata.types import (
    AudioRef,
    HFTextToVideo,
    HuggingFaceModel,
    ImageRef,
    Provider,
    VideoRef,
)
from nodetool.nodes.mlx._hf_cache import find_cached_snapshot
from nodetool.nodes.mlx._mlx_thread import MLX_EXECUTOR
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)

# The engine at the commit Phosphene pins (branch phosphene/v5.0-lowram1).
H3_ENGINE_REPO = "mrbizarro/minimax-h3-mlx"
H3_ENGINE_COMMIT = "26010372e81af2224da97e6ea27e6be3bcb6d957"
H3_ENGINE_URL = (
    f"https://codeload.github.com/{H3_ENGINE_REPO}/tar.gz/{H3_ENGINE_COMMIT}"
)
H3_RUNNER = "scripts/generate_staged.py"
H3_QUANTIZER = "scripts/quantize_stream.py"
H3_DATA_DIR = "minimax-h3"
H3_Q8_DIR = "h3-dit-q8"
H3_Q8_MARKER = ".built_ok"
H3_Q8_FREE_GB = 24.0

# Weights, as the engine's scripts/download_selected.py fetches them.
H3_DIT_REPO = "DeepBeepMeep/MiniMax-H3"
H3_DIT_FILE = "MiniMax-H3-FL2VA-pruned_bf16.safetensors"
H3_COMPACT_REPO = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
H3_COMPACT_FILES = [
    "text_encoder.safetensors",
    "video_vae.safetensors",
    "audio_vae.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "NOTICE",
    "MODIFICATIONS.md",
]
H3_META_REPO = "MiniMaxAI/MiniMax-H3"
H3_TEXT_CONFIG = "FL2VA/text_encoder/config.json"
H3_META_FILES = [H3_TEXT_CONFIG, "FL2VA/model_index.json"]

# Fast: TaoMate's three-step LoRA in Kijai's rank-19 extraction, on the
# TaoMate sigma ladder. Pinned by revision and checked by hash, as Phosphene does.
H3_FAST_REPO = "Kijai/MiniMax-H3_comfy"
H3_FAST_REVISION = "098f8c48fccead9a93191c166ca31a130659d3bd"
H3_FAST_FILE = "loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors"
H3_FAST_SHA256 = "de9663d974a884b477556748239c6f28239f7ca1825be270f98f023ff5dab6a7"
H3_FAST_STEPS = 4
H3_FAST_SIGMA_SUBSET = "50:0,16,33,49"
H3_BEST_STEPS = 9
H3_BEST_STEPS_HIGH = 16

H3_FPS = 24
# One 5 second window on the engine's 17n+5 frame grid.
H3_WINDOW_FRAMES = 124
H3_MIN_FRAMES = 22
# Frames each chained window carries from the one before it.
H3_CONTEXT_FRAMES = 17
H3_SEAM_COLOUR_FRAMES = 24
H3_REFERENCE_SIZE = 1024
H3_MAX_KEYFRAMES = 8
H3_LIPSYNC_MAX_SECONDS = 30.0
H3_LIPSYNC_CONTRACT = (
    "The speaker lip-syncs every syllable to the supplied soundtrack, exactly as it is."
)
# The engine's admission check: the largest pass cannot fit this Mac, or other
# apps leave too little memory free right now.
H3_EXIT_CANNOT_FIT = 76
H3_EXIT_BUSY = 77

# The fused qkv rows of an H3 LoRA: 56 heads of 128 rows for each of q, k, v.
H3_QKV_HEADS = 56
H3_QKV_HEAD_DIM = 128
H3_QKV_LORA_A = ".attn.qkv_proj.lora_A.weight"
H3_QKV_LORA_B = ".attn.qkv_proj.lora_B.weight"
H3_QKV_VOTE = 0.7


class H3Resolution(str, Enum):
    DRAFT = "640x384"
    STANDARD = "768x448"
    HIGH = "1024x576"
    DRAFT_PORTRAIT = "384x640"
    STANDARD_PORTRAIT = "448x768"
    HIGH_PORTRAIT = "576x1024"


class H3Speed(str, Enum):
    FAST = "fast"
    BEST = "best"


def resolution_size(resolution: H3Resolution) -> tuple[int, int]:
    """``(width, height)`` of a canvas."""
    width, height = H3Resolution(resolution).value.split("x")
    return int(width), int(height)


def window_for_short(frames: int) -> int:
    """The smallest 17n+5 window that holds ``frames``."""
    count = max(H3_MIN_FRAMES, int(frames))
    while count % 17 != 5:
        count += 1
    return count


def windows_for(frames: int, window_frames: int = H3_WINDOW_FRAMES) -> int:
    """Chained windows that deliver ``frames``; each window after the first adds window - 1."""
    frames = max(1, int(frames))
    if frames <= window_frames:
        return 1
    return 1 + -(-(frames - window_frames) // (window_frames - 1))


def clip_shape(frames: int) -> list[str]:
    """Runner flags for a clip of ``frames``: one window, or a chain of 5 second windows."""
    if frames <= H3_WINDOW_FRAMES:
        return ["--frames", str(window_for_short(frames))]
    return [
        "--frames", str(H3_WINDOW_FRAMES),
        "--chain-windows", str(windows_for(frames)),
        "--chain-total-frames", str(frames),
        "--chain-context-frames", str(H3_CONTEXT_FRAMES),
        "--seam-colour-frames", str(H3_SEAM_COLOUR_FRAMES),
    ]  # fmt: skip


def memory_budget(total_ram_gb: float) -> tuple[float, float]:
    """``(memory_gb, wired_gb)`` that leave the user room to keep using the Mac.

    The engine's defaults (58 and 50) clamp to all but 1 GB of RAM on any Mac
    below 59 GB. Keep the larger of 6 GB and an eighth of RAM free instead.
    """
    if total_ram_gb <= 0:
        return 0.0, 0.0
    memory_gb = max(8.0, total_ram_gb - max(6.0, total_ram_gb / 8.0))
    wired_gb = max(4.0, memory_gb - max(4.0, total_ram_gb / 16.0))
    return round(memory_gb, 1), round(wired_gb, 1)


def system_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def lipsync_prompt(prompt: str) -> str:
    """The lip-sync prompt with its sync sentence, added once."""
    text = prompt.strip()
    lowered = text.lower()
    if "lip-sync" in lowered or "lip sync" in lowered:
        return text
    marker = "\n\noverall_soundscape:"
    if marker in text:
        head, tail = text.split(marker, 1)
        return f"{head.rstrip()} {H3_LIPSYNC_CONTRACT}{marker}{tail}"
    return f"{text} {H3_LIPSYNC_CONTRACT}".strip()


def keyframe_positions(count: int, frames: int) -> list[int]:
    """Frames for the ``count - 2`` stills between the first and the last."""
    last = frames - 1
    inner = [round(index * last / (count - 1)) for index in range(1, count - 1)]
    return [max(1, min(frames - 2, frame)) for frame in inner]


# -- engine install -------------------------------------------------------------


def h3_home() -> Path:
    return get_system_data_path(H3_DATA_DIR)


def engine_dir() -> Path:
    return h3_home() / f"engine-{H3_ENGINE_COMMIT[:12]}"


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in the H3 engine archive: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"Unexpected entry in the H3 engine archive: {member.name}"
            )
    return members


def ensure_engine(download: Callable[[str, Path], None] | None = None) -> Path:
    """Download and unpack the pinned engine once; return its folder."""
    target = engine_dir()
    if (target / H3_RUNNER).is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fetch = download or _download_url
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".engine-") as tmp:
        archive = Path(tmp) / "engine.tar.gz"
        log.info("Downloading the MiniMax H3 engine at %s", H3_ENGINE_COMMIT)
        fetch(H3_ENGINE_URL, archive)
        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            members = _safe_members(tar)
            # GitHub stamps the commit an archive was made from into its header.
            if tar.pax_headers.get("comment") != H3_ENGINE_COMMIT:
                raise RuntimeError(
                    "The downloaded H3 engine archive is not the pinned commit."
                )
            tar.extractall(unpacked, members=members)
        roots = [path for path in unpacked.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / H3_RUNNER).is_file():
            raise RuntimeError(
                "The H3 engine archive has no scripts/generate_staged.py."
            )
        if not target.exists():
            os.replace(roots[0], target)
    return target


def _download_url(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response:
        with open(destination, "wb") as out:
            shutil.copyfileobj(response, out)


def q8_pack_dir() -> Path:
    return h3_home() / H3_Q8_DIR


def ensure_q8_pack(
    engine: Path, dit_file: Path, run: Callable[[list[str], Path], None]
) -> Path:
    """Quantize the bf16 transformer to the engine's Q8 pack once; return the pack."""
    pack = q8_pack_dir()
    if (pack / H3_Q8_MARKER).is_file():
        return pack
    pack.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(pack.parent).free / 1024**3
    if free_gb < H3_Q8_FREE_GB:
        raise RuntimeError(
            f"Building the MiniMax H3 Q8 engine needs about {H3_Q8_FREE_GB:.0f} GB free "
            f"in {pack.parent}; {free_gb:.0f} GB is free."
        )
    partial = pack.with_name(f"{H3_Q8_DIR}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    log.info("Building the MiniMax H3 Q8 engine in %s (about five minutes)", pack)
    # The quantizer reloads the written pack with load_dit() before it exits 0.
    run(
        [sys.executable, str(engine / H3_QUANTIZER), "--src", str(dit_file), "--out", str(partial)],
        engine,
    )  # fmt: skip
    (partial / H3_Q8_MARKER).write_text(H3_ENGINE_COMMIT)
    shutil.rmtree(pack, ignore_errors=True)
    os.replace(partial, pack)
    return pack


# -- LoRA layout ----------------------------------------------------------------


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if not 0 < size <= 100 * 1024 * 1024:
            raise ValueError(f"{path.name} is not a safetensors file")
        return size, json.loads(handle.read(size))


def _read_f64(handle: Any, header_size: int, info: dict[str, Any]) -> Any:
    import numpy as np

    start, end = info["data_offsets"]
    handle.seek(8 + header_size + start)
    raw = handle.read(end - start)
    dtype = info["dtype"]
    if dtype == "BF16":
        values = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype in ("F16", "F32", "F64"):
        values = np.frombuffer(raw, {"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return values.astype(np.float64).reshape(info["shape"])


def _anova_f(values: Any, labels: Any) -> float:
    groups = [values[labels == group] for group in range(3)]
    grand = values.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / 2
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(values) - 3)
    return float(between / within) if within > 0 else float("inf")


def lora_qkv_layout(path: Path) -> str:
    """Order of a LoRA's fused qkv rows: ``contiguous``, ``per-head``, ``unclear`` or ``none``.

    The H3 checkpoint stores qkv per head, (heads, 3, head_dim). Trainers that
    go through the ComfyUI model write (3, heads, head_dim). Each trained qkv
    module votes for the grouping under which the row norms of B @ A differ
    most between q, k and v.
    """
    import numpy as np

    header_size, header = _safetensors_header(path)
    rows = 3 * H3_QKV_HEADS * H3_QKV_HEAD_DIM
    index = np.arange(rows)
    per_head = (index // H3_QKV_HEAD_DIM) % 3
    contiguous = index // (H3_QKV_HEADS * H3_QKV_HEAD_DIM)
    votes = {"contiguous": 0, "per-head": 0}
    with open(path, "rb") as handle:
        for name in sorted(key for key in header if key.endswith(H3_QKV_LORA_B)):
            a_info = header.get(name[: -len(H3_QKV_LORA_B)] + H3_QKV_LORA_A)
            b_info = header[name]
            if a_info is None or int(b_info["shape"][0]) != rows:
                continue
            b = _read_f64(handle, header_size, b_info)
            a = _read_f64(handle, header_size, a_info)
            norms = np.sqrt(np.clip(((b @ (a @ a.T)) * b).sum(axis=1), 0, None))
            if not norms.any():
                continue
            if _anova_f(norms, per_head) > _anova_f(norms, contiguous):
                votes["per-head"] += 1
            else:
                votes["contiguous"] += 1
    modules = sum(votes.values())
    if not modules:
        return "none"
    for layout, count in votes.items():
        if count >= H3_QKV_VOTE * modules:
            return layout
    return "unclear"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fast_lora_argv(token: str | None) -> list[str]:
    """Download the Fast LoRA once and return its runner flags."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            H3_FAST_REPO, H3_FAST_FILE, revision=H3_FAST_REVISION, token=token
        )
    )
    # The hash and layout of the pinned file are recorded beside it once.
    stamp = h3_home() / "fast-lora.json"
    try:
        known = json.loads(stamp.read_text())
    except (OSError, ValueError):
        known = {}
    if known.get("sha256") != H3_FAST_SHA256 or known.get("path") != str(path):
        if _sha256(path) != H3_FAST_SHA256:
            raise RuntimeError(
                f"{H3_FAST_FILE} does not match the pinned hash. Delete it from the "
                "Hugging Face cache and run again."
            )
        known = {
            "path": str(path),
            "sha256": H3_FAST_SHA256,
            "layout": lora_qkv_layout(path),
        }
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(known))
    argv = ["--steps", str(H3_FAST_STEPS), "--sigma-subset", H3_FAST_SIGMA_SUBSET]
    argv += ["--lora", f"{path}:1.0"]
    # Re-order qkv rows unless the file already uses the checkpoint's order.
    if known["layout"] in ("contiguous", "unclear"):
        argv.append("--lora-qkv-permute")
    return argv


# -- running the engine ---------------------------------------------------------


def run_engine(
    command: list[str], engine: Path, cancelled: Callable[[], bool] = lambda: False
) -> None:
    """Run an engine script, log its output and raise with its tail on failure."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(engine), env.get("PYTHONPATH", "")) if part
    )
    process = subprocess.Popen(
        command,
        cwd=engine,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=30)

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                log.info("[h3] %s", line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while True:
        try:
            code = process.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                reader.join(timeout=5)
                raise RuntimeError("The MiniMax H3 render was cancelled.")
    reader.join(timeout=5)
    if code == 0:
        return
    if code == H3_EXIT_CANNOT_FIT:
        raise RuntimeError(
            "MiniMax H3 cannot fit this Mac at this size. Pick a smaller resolution "
            "or a shorter clip."
        )
    if code == H3_EXIT_BUSY:
        raise RuntimeError(
            "Other apps leave too little memory free for MiniMax H3 right now. "
            "Close some and run again."
        )
    details = "\n".join(tail)
    raise RuntimeError(f"The MiniMax H3 engine failed (exit {code}):\n{details}")


class _MiniMaxH3Base(BaseNode):
    """Shared plumbing for the MiniMax H3 MLX nodes."""

    _provider: ClassVar[Provider] = Provider.MLX
    _body: ClassVar[str] = "content_card"

    prompt: str = Field(
        default="",
        description="What happens in the clip, including dialogue and sound.",
    )
    resolution: H3Resolution = Field(
        default=H3Resolution.STANDARD,
        description="Canvas. Draft is fastest; High is true 16:9 and resolves faces best.",
    )
    speed: H3Speed = Field(
        default=H3Speed.FAST,
        description="Fast runs three steps with TaoMate's LoRA, about four times faster. "
        "Best runs the full sampler.",
    )
    steps: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Sigma points for Best (steps run = points - 1). 0 uses 9, or 16 at High.",
    )
    seed: int = Field(default=-1, description="Random seed. -1 picks one at random.")

    @classmethod
    def is_visible(cls) -> bool:
        return not cls.__name__.startswith("_")

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    def requires_gpu(self) -> bool:
        return False

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(repo_id=H3_DIT_REPO, allow_patterns=[H3_DIT_FILE]),
            HFTextToVideo(
                repo_id=H3_COMPACT_REPO, allow_patterns=list(H3_COMPACT_FILES)
            ),
            HFTextToVideo(repo_id=H3_META_REPO, allow_patterns=list(H3_META_FILES)),
        ]

    # -- inputs --------------------------------------------------------------

    @staticmethod
    def _weights() -> tuple[Path, Path, Path]:
        """``(bf16 transformer, compact root, text config)`` from the HF cache."""
        found = {}
        for repo_id, marker in (
            (H3_DIT_REPO, H3_DIT_FILE),
            (H3_COMPACT_REPO, "text_encoder.safetensors"),
            (H3_META_REPO, H3_TEXT_CONFIG),
        ):
            snapshot = find_cached_snapshot(repo_id, marker)
            if snapshot is None:
                raise RuntimeError(
                    f"{repo_id} is not downloaded. Download the MiniMax H3 models "
                    "(about 75 GB) from the Models panel first."
                )
            found[repo_id] = snapshot
        compact = found[H3_COMPACT_REPO]
        missing = [name for name in H3_COMPACT_FILES if not (compact / name).exists()]
        if missing:
            raise RuntimeError(
                f"{H3_COMPACT_REPO} is missing {', '.join(missing)}. Download it again."
            )
        return (
            found[H3_DIT_REPO] / H3_DIT_FILE,
            compact,
            found[H3_META_REPO] / H3_TEXT_CONFIG,
        )

    def _seed(self) -> int:
        if self.seed is not None and self.seed >= 0:
            return int(self.seed)
        return random.randint(0, 2**31 - 1)

    def _size(self) -> tuple[int, int]:
        return resolution_size(self.resolution)

    async def _image_file(
        self, context: ProcessingContext, image: ImageRef, directory: Path, name: str
    ) -> str:
        """Save ``image`` cover-cropped to the canvas; the engine would stretch it."""
        from PIL import ImageOps

        path = directory / f"{name}.png"
        picture = (await context.image_to_pil(image)).convert("RGB")
        ImageOps.fit(picture, self._size()).save(path)
        return str(path)

    @staticmethod
    async def _reference_file(
        context: ProcessingContext, image: ImageRef, directory: Path
    ) -> str:
        path = directory / "reference.png"
        (await context.image_to_pil(image)).convert("RGB").save(path)
        return str(path)

    @staticmethod
    async def _video_file(
        context: ProcessingContext, video: VideoRef, directory: Path
    ) -> str:
        path = directory / "source.mp4"
        path.write_bytes(await context.asset_to_bytes(video))
        return str(path)

    @staticmethod
    async def _audio_file(
        context: ProcessingContext, audio: AudioRef, directory: Path
    ) -> tuple[str, float]:
        """Save ``audio`` as WAV; return its path and length in seconds."""
        segment = await context.audio_to_audio_segment(audio)
        path = directory / "audio.wav"
        segment.export(str(path), format="wav")
        return str(path), len(segment) / 1000.0

    # -- render --------------------------------------------------------------

    async def _render(
        self, context: ProcessingContext, prompt: str, argv: list[str], directory: Path
    ) -> VideoRef:
        """Run the engine with ``argv`` after the common flags; return its MP4."""
        if sys.platform != "darwin":
            raise RuntimeError("MiniMax H3 on MLX needs a Mac with Apple Silicon.")
        if not prompt.strip():
            raise ValueError("A prompt is required.")
        dit_file, compact, text_config = self._weights()
        token = (
            await context.get_secret("HF_TOKEN") if self.speed == H3Speed.FAST else None
        )
        width, height = self._size()
        seed = self._seed()
        output = directory / "output.mp4"

        def cancelled() -> bool:
            return bool(getattr(context, "is_cancelled", False))

        def run(command: list[str], engine: Path) -> None:
            run_engine(command, engine, cancelled)

        def work() -> None:
            engine = ensure_engine()
            pack = ensure_q8_pack(engine, dit_file, run)
            if self.speed == H3Speed.FAST:
                sampler = fast_lora_argv(token)
            else:
                default = (
                    H3_BEST_STEPS_HIGH if max(width, height) >= 1024 else H3_BEST_STEPS
                )
                sampler = ["--steps", str(self.steps or default)]
            command = [
                sys.executable, str(engine / H3_RUNNER), prompt,
                "--dit", str(pack),
                "--compact-root", str(compact),
                "--text-config", str(text_config),
                "-o", str(output),
                "--metrics", str(directory / "metrics.json"),
                "--width", str(width),
                "--height", str(height),
                "--seed", str(seed),
                *sampler,
                *argv,
                "--low-memory", "auto",
            ]  # fmt: skip
            memory_gb, wired_gb = memory_budget(system_ram_gb())
            if memory_gb > 0:
                command += ["--memory-gb", str(memory_gb), "--wired-gb", str(wired_gb)]
            run(command, engine)

        # On the thread the MLX nodes share, so a render never overlaps them.
        await asyncio.get_running_loop().run_in_executor(MLX_EXECUTOR, work)
        if not output.exists():
            raise RuntimeError("The MiniMax H3 engine did not write an output video.")
        return await context.video_from_bytes(output.read_bytes())


class MiniMaxH3Video(_MiniMaxH3Base):
    """
    Generates a video with sound from a prompt, optionally from a first frame, to a last frame, or keeping a face, with MiniMax H3 on Apple Silicon. Powered by MiniMax H3.
    video, text-to-video, image-to-video, minimax, h3, hailuo, mlx, apple silicon, audio

    Use cases:
    - Generate a clip with dialogue and sound from a description
    - Animate a still, or move from a first frame to a last frame
    - Keep a person's face from a reference photo
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional first frame. Leave empty for text to video.",
    )
    last_image: ImageRef = Field(default=ImageRef(), description="Optional last frame.")
    reference: ImageRef = Field(
        default=ImageRef(),
        description="Optional face to keep: a photo of the person, not a frame of the clip.",
    )
    seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
        description="Clip length. Past 5 seconds the engine chains 5 second windows.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax H3 Video (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "image", "last_image", "seconds", "resolution", "speed"]

    def required_inputs(self) -> list[str]:
        return ["prompt"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        frames = round(float(self.seconds) * H3_FPS)
        with tempfile.TemporaryDirectory(prefix="h3_mlx_") as tmp:
            directory = Path(tmp)
            argv = clip_shape(frames)
            if self.image.is_set():
                path = await self._image_file(context, self.image, directory, "first")
                argv += ["--first-frame", path]
            if self.last_image.is_set():
                path = await self._image_file(
                    context, self.last_image, directory, "last"
                )
                argv += ["--last-frame", path]
            if self.reference.is_set():
                path = await self._reference_file(context, self.reference, directory)
                argv += [
                    "--reference",
                    path,
                    "--reference-size",
                    str(H3_REFERENCE_SIZE),
                ]
            return await self._render(context, self.prompt, argv, directory)


class MiniMaxH3Keyframes(_MiniMaxH3Base):
    """
    Generates a video with sound that passes through a sequence of images, with MiniMax H3 on Apple Silicon. Powered by MiniMax H3.
    video, keyframes, interpolation, minimax, h3, hailuo, mlx, apple silicon, audio

    Use cases:
    - Move a shot through planned story beats
    - Join a first and a last image with stills in between
    """

    images: list[ImageRef] = Field(
        default=[],
        description="Two to eight stills, spread evenly from the first frame to the last.",
    )
    seconds: float = Field(default=5.0, ge=1.0, le=15.0, description="Clip length.")

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax H3 Keyframes (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "images", "seconds", "resolution", "speed"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "images"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        images = [image for image in self.images if image.is_set()]
        if not 2 <= len(images) <= H3_MAX_KEYFRAMES:
            raise ValueError(f"Keyframes need between 2 and {H3_MAX_KEYFRAMES} images.")
        frames = round(float(self.seconds) * H3_FPS)
        delivered = frames if frames > H3_WINDOW_FRAMES else window_for_short(frames)
        with tempfile.TemporaryDirectory(prefix="h3_mlx_") as tmp:
            directory = Path(tmp)
            first = await self._image_file(context, images[0], directory, "first")
            last = await self._image_file(context, images[-1], directory, "last")
            argv = clip_shape(frames) + ["--first-frame", first, "--last-frame", last]
            for index, frame in enumerate(keyframe_positions(len(images), delivered)):
                path = await self._image_file(
                    context, images[index + 1], directory, f"beat_{index}"
                )
                argv += ["--keyframe", f"{path}@{frame}"]
            return await self._render(context, self.prompt, argv, directory)


class MiniMaxH3Extend(_MiniMaxH3Base):
    """
    Continues a video with new frames and sound, carrying its last 17 frames and their sound, with MiniMax H3 on Apple Silicon. Powered by MiniMax H3.
    video, extend, continue, minimax, h3, hailuo, mlx, apple silicon, audio

    Use cases:
    - Make a clip longer without a cut
    - Continue a shot toward an optional end frame
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The clip to continue, up to 60 seconds."
    )
    seconds: float = Field(
        default=5.0, ge=0.5, le=15.0, description="Seconds of new video after the clip."
    )
    last_image: ImageRef = Field(
        default=ImageRef(), description="Optional frame the continuation should end on."
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax H3 Extend (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "video", "seconds", "resolution", "speed"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.video.is_set():
            raise ValueError("A video to continue is required.")
        new_frames = round(float(self.seconds) * H3_FPS)
        with tempfile.TemporaryDirectory(prefix="h3_mlx_") as tmp:
            directory = Path(tmp)
            source = await self._video_file(context, self.video, directory)
            argv = [
                "--frames", str(H3_WINDOW_FRAMES),
                "--extend-from", source,
                "--extend-frames", str(new_frames),
                "--chain-windows", str(windows_for(new_frames + 1)),
                "--chain-context-frames", str(H3_CONTEXT_FRAMES),
                "--seam-colour-frames", str(H3_SEAM_COLOUR_FRAMES),
            ]  # fmt: skip
            if self.last_image.is_set():
                path = await self._image_file(
                    context, self.last_image, directory, "last"
                )
                argv += ["--last-frame", path]
            return await self._render(context, self.prompt, argv, directory)


class MiniMaxH3LipSync(_MiniMaxH3Base):
    """
    Generates a video performing a given soundtrack, with lips synced to speech or singing, with MiniMax H3 on Apple Silicon. Powered by MiniMax H3.
    video, lip sync, audio-to-video, talking head, minimax, h3, hailuo, mlx, apple silicon

    Use cases:
    - Make a speaker or singer perform a recording
    - Animate a portrait to a voice-over
    """

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The soundtrack the clip performs and keeps. Isolated vocals sync best.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Optional first frame, such as the speaker's face.",
    )
    start_time: float = Field(
        default=0.0, ge=0.0, description="Second of the audio the clip starts at."
    )
    seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=H3_LIPSYNC_MAX_SECONDS,
        description="Clip length. 0 covers the audio from the start time, up to 30 seconds.",
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax H3 Lip Sync (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "audio", "image", "resolution", "speed"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "audio"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.audio.is_set():
            raise ValueError("An audio clip is required.")
        with tempfile.TemporaryDirectory(prefix="h3_mlx_") as tmp:
            directory = Path(tmp)
            audio, duration = await self._audio_file(context, self.audio, directory)
            available = duration - float(self.start_time)
            if available <= 0.2:
                raise ValueError(
                    f"The start time {self.start_time:.1f}s is past the end of the audio "
                    f"({duration:.1f}s)."
                )
            seconds = min(
                available, float(self.seconds) or available, H3_LIPSYNC_MAX_SECONDS
            )
            delivered = max(6, round(seconds * H3_FPS))
            window = (
                H3_WINDOW_FRAMES
                if delivered > H3_WINDOW_FRAMES
                else window_for_short(delivered)
            )
            argv = [
                "--frames", str(window),
                "--audio-drive", audio,
                "--audio-drive-offset", f"{float(self.start_time):.3f}",
                "--chain-total-frames", str(delivered),
            ]  # fmt: skip
            if delivered > H3_WINDOW_FRAMES:
                argv += ["--chain-windows", str(windows_for(delivered))]
                argv += ["--chain-context-frames", str(H3_CONTEXT_FRAMES)]
                argv += ["--seam-colour-frames", str(H3_SEAM_COLOUR_FRAMES)]
            if self.image.is_set():
                path = await self._image_file(context, self.image, directory, "face")
                argv += ["--first-frame", path]
            return await self._render(
                context, lipsync_prompt(self.prompt), argv, directory
            )


class MiniMaxH3AddSound(_MiniMaxH3Base):
    """
    Generates a soundtrack for a video and keeps its picture exactly as it is, with MiniMax H3 on Apple Silicon. Powered by MiniMax H3.
    video, audio, video-to-audio, foley, sound effects, minimax, h3, hailuo, mlx, apple silicon

    Use cases:
    - Add sound effects and ambience to a silent clip
    - Replace a clip's soundtrack with one described in the prompt
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The clip to score, up to 60 seconds."
    )

    @classmethod
    def get_title(cls) -> str:
        return "MiniMax H3 Add Sound (MLX)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "video", "resolution", "speed"]

    def required_inputs(self) -> list[str]:
        return ["prompt", "video"]

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not self.video.is_set():
            raise ValueError("A video is required.")
        with tempfile.TemporaryDirectory(prefix="h3_mlx_") as tmp:
            directory = Path(tmp)
            source = await self._video_file(context, self.video, directory)
            argv = [
                "--frames", str(H3_WINDOW_FRAMES),
                "--video-to-audio", source,
                "--chain-context-frames", str(H3_CONTEXT_FRAMES),
            ]  # fmt: skip
            return await self._render(context, self.prompt, argv, directory)
