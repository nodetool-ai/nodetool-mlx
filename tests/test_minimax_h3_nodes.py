"""MiniMax H3 MLX nodes, run against a stub of the pinned engine's subprocess."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from pydub import AudioSegment
from safetensors.numpy import save_file

import nodetool.nodes.mlx.minimax_h3 as h3
from nodetool.metadata.types import AudioRef, ImageRef, VideoRef

VIDEO = VideoRef(uri="memory://out")


def _context(audio_ms: int = 4000) -> MagicMock:
    context = MagicMock()
    context.is_cancelled = False

    async def image_to_pil(ref):
        return Image.new("RGB", (40, 10))

    async def asset_to_bytes(ref):
        return b"source"

    async def audio_to_audio_segment(ref):
        return AudioSegment.silent(duration=audio_ms)

    async def video_from_bytes(data):
        assert data == b"mp4"
        return VIDEO

    async def get_secret(name):
        return "hf-token"

    context.image_to_pil = image_to_pil
    context.asset_to_bytes = asset_to_bytes
    context.audio_to_audio_segment = audio_to_audio_segment
    context.video_from_bytes = video_from_bytes
    context.get_secret = get_secret
    return context


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """Record each engine command and write the MP4 the runner would."""
    calls: list[dict] = []
    monkeypatch.setattr(h3.sys, "platform", "darwin")
    monkeypatch.setattr(
        h3._MiniMaxH3Base,
        "_weights",
        staticmethod(
            lambda: (
                tmp_path / "dit.safetensors",
                tmp_path / "compact",
                tmp_path / "cfg.json",
            )
        ),
    )
    monkeypatch.setattr(h3, "ensure_engine", lambda: tmp_path / "engine")
    monkeypatch.setattr(h3, "ensure_q8_pack", lambda engine, dit, run: tmp_path / "q8")
    monkeypatch.setattr(
        h3, "fast_lora_argv", lambda token: ["--steps", "4", "--fast", token]
    )
    monkeypatch.setattr(h3, "system_ram_gb", lambda: 64.0)

    def run(command, engine_dir, cancelled):
        files = {
            flag: Path(command[index + 1]).exists()
            for index, flag in enumerate(command)
            if flag in ("--first-frame", "--last-frame", "--reference", "--extend-from",
                        "--audio-drive", "--video-to-audio")
        }  # fmt: skip
        calls.append({"command": command, "files": files})
        Path(command[command.index("-o") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(h3, "run_engine", run)
    return calls


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _all(command: list[str], flag: str) -> list[str]:
    return [command[i + 1] for i, value in enumerate(command) if value == flag]


# -- shapes -----------------------------------------------------------------------


def test_clip_shape_uses_one_window_up_to_five_seconds():
    assert h3.clip_shape(24) == ["--frames", "39"]
    assert h3.clip_shape(120) == ["--frames", "124"]


def test_clip_shape_chains_windows_past_five_seconds():
    shape = h3.clip_shape(240)
    assert _flag(shape, "--frames") == "124"
    assert _flag(shape, "--chain-windows") == "2"
    assert _flag(shape, "--chain-total-frames") == "240"
    assert _flag(shape, "--chain-context-frames") == "17"
    assert h3.windows_for(360) == 3


def test_window_sizes_sit_on_the_17n_plus_5_grid():
    for frames in (1, 22, 23, 100, 124):
        assert h3.window_for_short(frames) % 17 == 5
        assert h3.window_for_short(frames) >= frames


def test_memory_budget_leaves_room_on_small_macs():
    assert h3.memory_budget(36.0) == (30.0, 26.0)
    assert h3.memory_budget(128.0) == (112.0, 104.0)
    assert h3.memory_budget(0.0) == (0.0, 0.0)


def test_lipsync_prompt_adds_the_sync_sentence_once():
    assert h3.lipsync_prompt("A singer.").endswith(h3.H3_LIPSYNC_CONTRACT)
    assert h3.lipsync_prompt("She lip-syncs.") == "She lip-syncs."
    fields = "description: a man\n\noverall_soundscape: rain"
    assert h3.lipsync_prompt(fields).endswith("\n\noverall_soundscape: rain")


def test_keyframe_positions_spread_between_the_ends():
    assert h3.keyframe_positions(4, 124) == [41, 82]
    assert h3.keyframe_positions(2, 124) == []


# -- LoRA layout ------------------------------------------------------------------


def _qkv_lora(path: Path, per_head: bool) -> Path:
    rows = 3 * h3.H3_QKV_HEADS * h3.H3_QKV_HEAD_DIM
    index = np.arange(rows)
    group = (index // h3.H3_QKV_HEAD_DIM) % 3 if per_head else index // (rows // 3)
    rng = np.random.default_rng(0)
    b = (rng.normal(size=(rows, 2)) * (1.0 + 4.0 * group[:, None])).astype(np.float32)
    a = rng.normal(size=(2, 8)).astype(np.float32)
    save_file(
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": a,
            "blocks.0.attn.qkv_proj.lora_B.weight": b,
        },
        str(path),
    )
    return path


def test_lora_qkv_layout_reads_the_row_order(tmp_path):
    assert (
        h3.lora_qkv_layout(_qkv_lora(tmp_path / "c.safetensors", False)) == "contiguous"
    )
    assert h3.lora_qkv_layout(_qkv_lora(tmp_path / "p.safetensors", True)) == "per-head"


def test_lora_qkv_layout_without_qkv_modules(tmp_path):
    path = tmp_path / "mlp.safetensors"
    save_file(
        {"blocks.0.mlp.fc1.lora_A.weight": np.zeros((2, 4), np.float32)}, str(path)
    )
    assert h3.lora_qkv_layout(path) == "none"


def test_fast_lora_checks_the_hash_and_permutes_contiguous_files(monkeypatch, tmp_path):
    lora = _qkv_lora(tmp_path / "fast.safetensors", False)
    monkeypatch.setattr(h3, "get_system_data_path", lambda name: tmp_path / name)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(lora))
    monkeypatch.setattr(h3, "_sha256", lambda path: "wrong")
    with pytest.raises(RuntimeError, match="pinned hash"):
        h3.fast_lora_argv(None)
    monkeypatch.setattr(h3, "_sha256", lambda path: h3.H3_FAST_SHA256)
    argv = h3.fast_lora_argv(None)
    assert _flag(argv, "--steps") == "4"
    assert _flag(argv, "--sigma-subset") == h3.H3_FAST_SIGMA_SUBSET
    assert _flag(argv, "--lora") == f"{lora}:1.0"
    assert "--lora-qkv-permute" in argv
    # The second run reads the recorded hash and layout instead of the file.
    monkeypatch.setattr(h3, "_sha256", lambda path: pytest.fail("hashed twice"))
    assert h3.fast_lora_argv(None) == argv


# -- engine install ---------------------------------------------------------------


def _archive(commit: str, names: dict[str, bytes]):
    def download(url, destination):
        assert url == h3.H3_ENGINE_URL
        buffer = io.BytesIO()
        with tarfile.open(
            fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT,
            pax_headers={"comment": commit},
        ) as tar:  # fmt: skip
            for name, data in names.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        destination.write_bytes(buffer.getvalue())

    return download


def test_ensure_engine_unpacks_the_pinned_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(h3, "get_system_data_path", lambda name: tmp_path / name)
    root = f"minimax-h3-mlx-{h3.H3_ENGINE_COMMIT}"
    engine = h3.ensure_engine(
        _archive(h3.H3_ENGINE_COMMIT, {f"{root}/{h3.H3_RUNNER}": b"#"})
    )
    assert (engine / h3.H3_RUNNER).read_bytes() == b"#"
    assert h3.ensure_engine(lambda url, dest: pytest.fail("downloaded twice")) == engine


def test_ensure_engine_refuses_another_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(h3, "get_system_data_path", lambda name: tmp_path / name)
    with pytest.raises(RuntimeError, match="pinned commit"):
        h3.ensure_engine(_archive("0" * 40, {f"x/{h3.H3_RUNNER}": b"#"}))


def test_ensure_engine_refuses_paths_outside_the_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(h3, "get_system_data_path", lambda name: tmp_path / name)
    with pytest.raises(RuntimeError, match="Unsafe path"):
        h3.ensure_engine(_archive(h3.H3_ENGINE_COMMIT, {"../evil.py": b"#"}))


def test_ensure_q8_pack_builds_once(monkeypatch, tmp_path):
    monkeypatch.setattr(h3, "get_system_data_path", lambda name: tmp_path / name)
    runs = []

    def run(command, engine):
        runs.append(command)
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")

    pack = h3.ensure_q8_pack(tmp_path / "engine", tmp_path / "dit.safetensors", run)
    assert (pack / "config.json").exists() and (pack / h3.H3_Q8_MARKER).exists()
    assert runs[0][1].endswith(h3.H3_QUANTIZER)
    assert _flag(runs[0], "--src") == str(tmp_path / "dit.safetensors")
    assert (
        h3.ensure_q8_pack(tmp_path / "engine", tmp_path / "dit.safetensors", run)
        == pack
    )
    assert len(runs) == 1


@pytest.mark.parametrize(
    ("code", "message"),
    [(76, "cannot fit this Mac"), (77, "too little memory free"), (3, "exit 3")],
)
def test_run_engine_explains_exit_codes(tmp_path, code, message):
    command = [sys.executable, "-c", f"print('log line'); raise SystemExit({code})"]
    with pytest.raises(RuntimeError, match=message):
        h3.run_engine(command, tmp_path)


def test_run_engine_stops_a_cancelled_render(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(RuntimeError, match="cancelled"):
        h3.run_engine(command, tmp_path, cancelled=lambda: True)


# -- nodes ------------------------------------------------------------------------


async def test_video_node_passes_frames_reference_and_canvas(engine, tmp_path):
    node = h3.MiniMaxH3Video(
        prompt="A cat sings.",
        image=ImageRef(uri="memory://a"),
        last_image=ImageRef(uri="memory://b"),
        reference=ImageRef(uri="memory://c"),
        resolution=h3.H3Resolution.HIGH,
        speed=h3.H3Speed.BEST,
        seconds=10.0,
        seed=7,
    )
    assert await node.process(_context()) is VIDEO
    command = engine[0]["command"]
    assert command[1].endswith(h3.H3_RUNNER) and command[2] == "A cat sings."
    assert _flag(command, "--dit") == str(tmp_path / "q8")
    assert (_flag(command, "--width"), _flag(command, "--height")) == ("1024", "576")
    assert _flag(command, "--steps") == "16"
    assert _flag(command, "--seed") == "7"
    assert _flag(command, "--chain-windows") == "2"
    assert _flag(command, "--reference-size") == "1024"
    assert _flag(command, "--low-memory") == "auto"
    assert (_flag(command, "--memory-gb"), _flag(command, "--wired-gb")) == (
        "56.0",
        "52.0",
    )
    assert engine[0]["files"] == {
        "--first-frame": True,
        "--last-frame": True,
        "--reference": True,
    }


async def test_video_node_cover_crops_frames_to_the_canvas(monkeypatch, engine):
    sizes = []
    monkeypatch.setattr(
        h3, "run_engine",
        lambda command, *_: (
            sizes.append(Image.open(_flag(command, "--first-frame")).size),
            Path(_flag(command, "-o")).write_bytes(b"mp4"),
        ),
    )  # fmt: skip
    node = h3.MiniMaxH3Video(prompt="x", image=ImageRef(uri="memory://a"))
    await node.process(_context())
    assert sizes == [(768, 448)]


async def test_fast_speed_uses_the_fast_sampler(engine):
    await h3.MiniMaxH3Video(prompt="x").process(_context())
    command = engine[0]["command"]
    assert _all(command, "--steps") == ["4"]
    assert _flag(command, "--fast") == "hf-token"
    assert _flag(command, "--frames") == "124"


async def test_video_node_requires_a_prompt(engine):
    with pytest.raises(ValueError, match="prompt"):
        await h3.MiniMaxH3Video(prompt=" ").process(_context())


async def test_keyframes_node_places_inner_stills(engine):
    images = [ImageRef(uri=f"memory://{i}") for i in range(4)]
    await h3.MiniMaxH3Keyframes(prompt="x", images=images).process(_context())
    command = engine[0]["command"]
    assert engine[0]["files"] == {"--first-frame": True, "--last-frame": True}
    assert [spec.rsplit("@", 1)[1] for spec in _all(command, "--keyframe")] == [
        "41",
        "82",
    ]


async def test_keyframes_node_needs_two_images(engine):
    with pytest.raises(ValueError, match="between 2"):
        await h3.MiniMaxH3Keyframes(prompt="x", images=[ImageRef(uri="m://a")]).process(
            _context()
        )


async def test_extend_node_continues_the_source(engine):
    node = h3.MiniMaxH3Extend(prompt="x", video=VideoRef(uri="memory://v"), seconds=7.5)
    await node.process(_context())
    command = engine[0]["command"]
    assert engine[0]["files"] == {"--extend-from": True}
    assert _flag(command, "--extend-frames") == "180"
    assert _flag(command, "--chain-windows") == "2"
    assert _flag(command, "--chain-context-frames") == "17"


async def test_lipsync_node_covers_the_audio(engine):
    node = h3.MiniMaxH3LipSync(
        prompt="A man talks.", audio=AudioRef(uri="memory://a"), image=ImageRef(uri="m://f"),
        start_time=1.0,
    )  # fmt: skip
    await node.process(_context(audio_ms=4000))
    command = engine[0]["command"]
    assert command[2].endswith(h3.H3_LIPSYNC_CONTRACT)
    assert engine[0]["files"] == {"--audio-drive": True, "--first-frame": True}
    assert _flag(command, "--audio-drive-offset") == "1.000"
    assert _flag(command, "--chain-total-frames") == "72"
    assert _flag(command, "--frames") == "73"
    assert "--chain-windows" not in command


async def test_lipsync_node_chains_long_audio_and_caps_it(engine):
    node = h3.MiniMaxH3LipSync(prompt="x", audio=AudioRef(uri="memory://a"))
    await node.process(_context(audio_ms=45000))
    command = engine[0]["command"]
    assert _flag(command, "--chain-total-frames") == "720"
    assert _flag(command, "--chain-windows") == "6"


async def test_lipsync_node_refuses_a_start_past_the_end(engine):
    node = h3.MiniMaxH3LipSync(prompt="x", audio=AudioRef(uri="m://a"), start_time=9.0)
    with pytest.raises(ValueError, match="past the end"):
        await node.process(_context(audio_ms=4000))


async def test_add_sound_node_holds_the_picture(engine):
    await h3.MiniMaxH3AddSound(prompt="rain", video=VideoRef(uri="m://v")).process(
        _context()
    )
    command = engine[0]["command"]
    assert engine[0]["files"] == {"--video-to-audio": True}
    assert "--first-frame" not in command


def test_nodes_recommend_the_three_weight_repos():
    repos = [model.repo_id for model in h3.MiniMaxH3Video.get_recommended_models()]
    assert repos == [h3.H3_DIT_REPO, h3.H3_COMPACT_REPO, h3.H3_META_REPO]
    assert not h3._MiniMaxH3Base.is_visible()
    assert h3.MiniMaxH3AddSound.is_visible()
