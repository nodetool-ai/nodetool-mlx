"""Tests for the MLX audio nodes (text-to-speech, speech-to-text, enhancement).

These tests exercise the pure-Python configuration logic of the nodes (parameter
building, text segmentation, metadata) without loading any MLX model, so they run
on any platform.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nodetool.nodes.mlx import automatic_speech_recognition as asr
from nodetool.nodes.mlx import speech_enhancement as se
from nodetool.nodes.mlx import speech_to_text as stt
from nodetool.nodes.mlx import text_to_speech as tts

TTS_NODE_CLASSES = [
    tts.KokoroTTS,
    tts.SesameTTS,
    tts.SparkTTS,
    tts.Qwen3TTS,
    tts.KittenTTS,
    tts.DiaTTS,
    tts.OuteTTS,
    tts.OmniVoiceTTS,
    tts.MeloTTS,
    tts.VoxtralTTS,
    tts.ChatterboxTTS,
    tts.HiggsAudioTTS,
    tts.LongCatAudioTTS,
    tts.MLXTextToSpeech,
]

STT_NODE_CLASSES = [
    stt.Parakeet,
    stt.Qwen3ASR,
    stt.Qwen3ForcedAligner,
    stt.MLXSpeechToText,
]

ENHANCEMENT_NODE_CLASSES = [se.DeepFilterNet, se.MossFormer2]

# Every MLX audio node that exposes a ``model`` field. The frontend only renders
# the model-select widget (with downloadable recommended models) when the field's
# metadata type triggers it; see ``_renders_model_select`` below.
MODEL_FIELD_NODE_CLASSES = (
    TTS_NODE_CLASSES + STT_NODE_CLASSES + ENHANCEMENT_NODE_CLASSES + [asr.Whisper]
)


def _model_property_type(node_cls) -> str:
    props = {p.name: p for p in node_cls.properties()}
    assert "model" in props, f"{node_cls.__name__} has no 'model' property"
    return props["model"].type.type


def _renders_model_select(type_str: str) -> bool:
    """Mirror the frontend resolver (PropertyInput.resolver.tsx handleModelTypes).

    A property renders the model-select widget when its type ends with ``_model``
    or starts with ``hf.`` / ``tjs.``.
    """
    return type_str.endswith("_model") or type_str.startswith(("hf.", "tjs."))


# ---------------------------------------------------------------------------
# Model-select widget eligibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_cls", MODEL_FIELD_NODE_CLASSES)
def test_model_field_renders_model_select(node_cls):
    model_type = _model_property_type(node_cls)
    assert _renders_model_select(model_type), (
        f"{node_cls.__name__}.model has type {model_type!r}, which will not render "
        "the model-select widget in the frontend"
    )


@pytest.mark.parametrize("node_cls", MODEL_FIELD_NODE_CLASSES)
def test_model_field_has_recommended_models(node_cls):
    models = node_cls.get_recommended_models()
    assert models, f"{node_cls.__name__} should recommend at least one model"
    model_type = _model_property_type(node_cls)
    for m in models:
        assert m.repo_id, f"{node_cls.__name__} recommended a model with no repo_id"
        assert m.type == model_type, (
            f"{node_cls.__name__} recommends a {m.type!r} model but its field is "
            f"{model_type!r}; they should match"
        )


# ---------------------------------------------------------------------------
# Metadata / registration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_cls", TTS_NODE_CLASSES)
def test_tts_node_metadata(node_cls):
    assert node_cls.is_visible()
    assert isinstance(node_cls.get_title(), str) and node_cls.get_title()
    assert "model" in node_cls.get_basic_fields()


@pytest.mark.parametrize(
    "node_cls",
    [c for c in TTS_NODE_CLASSES if c is not tts.MLXTextToSpeech],
)
def test_tts_recommended_models(node_cls):
    models = node_cls.get_recommended_models()
    assert models, f"{node_cls.__name__} should recommend at least one model"
    for m in models:
        assert m.repo_id.startswith(("mlx-community/", "prince-canuma/", "starkdmi/"))


def test_base_tts_is_hidden():
    assert tts.BaseMLXTTS.is_visible() is False
    assert stt.BaseMLXSpeechToText.is_visible() is False
    assert se.BaseMLXSpeechEnhancement.is_visible() is False


# ---------------------------------------------------------------------------
# Text segmentation
# ---------------------------------------------------------------------------
def test_line_split_default():
    node = tts.KokoroTTS(text="line one\nline two\n\nline three")
    assert node._iter_text_segments() == ["line one", "line two", "line three"]


def test_whole_text_for_dialogue_models():
    script = "[S1] Hello there.\n[S2] General Kenobi."
    assert tts.DiaTTS(text=script)._iter_text_segments() == [script]
    assert tts.OmniVoiceTTS(text=script)._iter_text_segments() == [script]
    assert tts.HiggsAudioTTS(text=script)._iter_text_segments() == [script]


def test_empty_text_yields_no_segments():
    assert tts.KokoroTTS(text="   \n  ")._iter_text_segments() == []


# ---------------------------------------------------------------------------
# TTS generation parameters
# ---------------------------------------------------------------------------
@pytest.fixture
def ctx():
    return MagicMock()


async def test_kokoro_params(ctx):
    node = tts.KokoroTTS(text="hi", voice=tts.KokoroTTS.Voice.AM_ADAM, temperature=0.5)
    params, cleanup = await node._build_generation_params(ctx)
    assert params["voice"] == "am_adam"
    assert params["lang_code"] == "a"
    assert params["temperature"] == 0.5
    assert params["stream"] is False
    assert cleanup is None


async def test_qwen3_instruct_optional(ctx):
    no_instruct, _ = await tts.Qwen3TTS(
        text="hi", voice="Ethan"
    )._build_generation_params(ctx)
    assert "instruct" not in no_instruct
    assert no_instruct["voice"] == "Ethan"

    with_instruct, _ = await tts.Qwen3TTS(
        text="hi", instruct="a calm narrator"
    )._build_generation_params(ctx)
    assert with_instruct["instruct"] == "a calm narrator"


async def test_kitten_voice_enum(ctx):
    node = tts.KittenTTS(text="hi", voice=tts.KittenTTS.Voice.EXPR_VOICE_3_F)
    params, _ = await node._build_generation_params(ctx)
    assert params["voice"] == "expr-voice-3-f"


async def test_voxtral_voice_and_temperature(ctx):
    node = tts.VoxtralTTS(text="hi", voice=tts.VoxtralTTS.Voice.FR_MALE)
    params, _ = await node._build_generation_params(ctx)
    assert params["voice"] == "fr_male"
    assert "temperature" in params


async def test_melotts_language_maps_to_lang_code(ctx):
    node = tts.MeloTTS(text="hi", language=tts.MeloTTS.LanguageCode.EN_AU)
    params, _ = await node._build_generation_params(ctx)
    assert params["lang_code"] == "EN-AU"


async def test_omnivoice_duration_optional(ctx):
    zero, _ = await tts.OmniVoiceTTS(text="hi", duration_s=0)._build_generation_params(
        ctx
    )
    assert "duration_s" not in zero
    five, _ = await tts.OmniVoiceTTS(text="hi", duration_s=5)._build_generation_params(
        ctx
    )
    assert five["duration_s"] == 5


async def test_reference_audio_not_required_when_unset(ctx):
    # Cloning-capable nodes must not export reference audio when none is provided.
    params, cleanup = await tts.HiggsAudioTTS(text="hi")._build_generation_params(ctx)
    assert "ref_audio" not in params
    assert cleanup is None
    ctx.audio_to_audio_segment.assert_not_called()


def test_speed_validation():
    # Pydantic rejects out-of-range speed at construction time.
    with pytest.raises(ValidationError):
        tts.KokoroTTS(text="hi", speed=5.0)
    # The runtime guard in _normalize_speed also rejects invalid values.
    node = tts.KokoroTTS(text="hi")
    object.__setattr__(node, "speed", 5.0)
    with pytest.raises(ValueError):
        node._normalize_speed()


class _FakeResult:
    def __init__(self, audio, sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate


class _FakeTTSModel:
    def __init__(self, results):
        self._results = results

    def generate(self, **kwargs):
        return iter(self._results)


async def test_sample_rate_change_mid_run_raises(monkeypatch):
    """A model that switches sample rate after emitting audio must fail loudly."""
    import sys

    import numpy as np

    # Stub the (macOS-only) mlx.core import used by _mx_array_to_numpy.
    monkeypatch.setitem(sys.modules, "mlx", MagicMock())
    monkeypatch.setitem(sys.modules, "mlx.core", MagicMock())
    monkeypatch.setattr(tts.sys, "platform", "darwin")

    node = tts.MLXTextToSpeech(text="hello", model=tts.HFTextToSpeech(repo_id="repo/x"))
    chunk = np.ones(8, dtype=np.float32)
    node._tts_model = _FakeTTSModel(
        [_FakeResult(chunk, 24_000), _FakeResult(chunk, 48_000)]
    )
    node._model_id_loaded = node._get_model_id()

    with pytest.raises(ValueError, match="multiple sample rates"):
        async for _ in node.gen_process(MagicMock()):
            pass


async def test_consistent_sample_rate_does_not_raise(monkeypatch):
    import sys

    import numpy as np

    monkeypatch.setitem(sys.modules, "mlx", MagicMock())
    monkeypatch.setitem(sys.modules, "mlx.core", MagicMock())
    monkeypatch.setattr(tts.sys, "platform", "darwin")

    node = tts.MLXTextToSpeech(text="hello", model=tts.HFTextToSpeech(repo_id="repo/x"))
    chunk = np.ones(8, dtype=np.float32)
    node._tts_model = _FakeTTSModel(
        [_FakeResult(chunk, 16_000), _FakeResult(chunk, 16_000)]
    )
    node._model_id_loaded = node._get_model_id()

    ctx = MagicMock()

    async def _audio_from_numpy(data, sample_rate, *a, **k):
        assert sample_rate == 16_000
        return "audio-ref"

    ctx.audio_from_numpy = _audio_from_numpy

    outputs = [item async for item in node.gen_process(ctx)]
    # Two streamed chunks plus a final aggregated output.
    assert outputs[-1]["audio"] == "audio-ref"


# ---------------------------------------------------------------------------
# STT parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_cls", STT_NODE_CLASSES)
def test_stt_metadata(node_cls):
    assert node_cls.is_visible()
    assert node_cls.get_title()
    assert "audio" in node_cls().required_inputs()


def test_parakeet_chunk_duration():
    assert stt.Parakeet()._build_generate_kwargs() == {}
    assert stt.Parakeet(chunk_duration=12.5)._build_generate_kwargs() == {
        "chunk_duration": 12.5
    }


def test_qwen3asr_kwargs():
    base = stt.Qwen3ASR()._build_generate_kwargs()
    assert base["temperature"] == 0.0
    assert base["max_tokens"] == 8192
    assert "language" not in base
    assert (
        stt.Qwen3ASR(language="English")._build_generate_kwargs()["language"]
        == "English"
    )


def test_forced_aligner_requires_text():
    with pytest.raises(ValueError):
        stt.Qwen3ForcedAligner(text="")._build_generate_kwargs()
    kwargs = stt.Qwen3ForcedAligner(text="hello world")._build_generate_kwargs()
    assert kwargs["text"] == "hello world"


def test_normalize_segments_handles_dicts_and_objects():
    class _Seg:
        text = "hello"
        start = 1.0
        end = 2.5

    obj_result = type("R", (), {"sentences": [_Seg()]})()
    assert stt.BaseMLXSpeechToText._normalize_segments(obj_result) == [
        {"text": "hello", "start": 1.0, "end": 2.5}
    ]

    dict_result = type("R", (), {"segments": [{"text": "x", "start": 0, "end": 1}]})()
    assert stt.BaseMLXSpeechToText._normalize_segments(dict_result) == [
        {"text": "x", "start": 0, "end": 1}
    ]

    assert stt.BaseMLXSpeechToText._normalize_segments(object()) == []


# ---------------------------------------------------------------------------
# Speech enhancement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_cls", ENHANCEMENT_NODE_CLASSES)
def test_enhancement_metadata(node_cls):
    assert node_cls.is_visible()
    assert node_cls.get_title()
    assert node_cls.get_recommended_models()
    assert node_cls._sample_rate == 48_000


def test_deepfilternet_version_default():
    assert se.DeepFilterNet().version == se.DeepFilterNet.Version.V3


# ---------------------------------------------------------------------------
# Hugging Face cache resolution
#
# Models downloaded by revision/commit (rather than by branch) end up cached
# without a ``refs/main`` pointer, so ``try_to_load_from_cache(repo, "config.json")``
# returns None even though every file is present on disk. The nodes must still
# recognise such models as available instead of demanding a re-download.
# ---------------------------------------------------------------------------
def _fake_hf_cache(repo_id: str, snapshot_dir, files: list[str]):
    rev = SimpleNamespace(
        snapshot_path=snapshot_dir,
        files=[
            SimpleNamespace(file_name=Path(f).name, file_path=snapshot_dir / f)
            for f in files
        ],
    )
    repo = SimpleNamespace(repo_id=repo_id, repo_type="model", revisions=[rev])
    return SimpleNamespace(repos=[repo])


def _patch_revision_only_cache(monkeypatch, repo_id, snapshot_dir, files):
    """Simulate a cache with files present but no ``refs/main`` pointer."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        huggingface_hub,
        "scan_cache_dir",
        lambda *a, **k: _fake_hf_cache(repo_id, snapshot_dir, files),
    )


def test_find_cached_snapshot_falls_back_to_scan(monkeypatch, tmp_path):
    from nodetool.nodes.mlx import _hf_cache

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    _patch_revision_only_cache(
        monkeypatch, "repo/x", snap, ["config.json", "model.safetensors"]
    )

    assert _hf_cache.find_cached_snapshot("repo/x", "config.json") == snap


def test_find_cached_snapshot_returns_none_when_absent(monkeypatch):
    from nodetool.nodes.mlx import _hf_cache

    _patch_revision_only_cache(monkeypatch, "other/repo", None, [])
    assert _hf_cache.find_cached_snapshot("repo/x", "config.json") is None


async def test_tts_preload_accepts_revision_only_cache(monkeypatch, tmp_path):
    import mlx_audio.tts.utils as tts_utils

    rid = "mlx-community/kitten-tts-nano-0.8"
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")

    monkeypatch.setattr(tts.sys, "platform", "darwin")
    _patch_revision_only_cache(
        monkeypatch, rid, snap, ["config.json", "model.safetensors"]
    )

    captured = {}

    def _fake_load(path, *a, **k):
        captured["path"] = Path(path)
        return "kitten-model"

    monkeypatch.setattr(tts_utils, "load_model", _fake_load)

    node = tts.KittenTTS()
    await node.preload_model(MagicMock())

    assert node._tts_model == "kitten-model"
    assert captured["path"] == snap


async def test_stt_preload_accepts_revision_only_cache(monkeypatch, tmp_path):
    import mlx_audio.stt.utils as stt_utils

    rid = "mlx-community/parakeet-tdt-0.6b-v3"
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")

    monkeypatch.setattr(stt.sys, "platform", "darwin")
    _patch_revision_only_cache(
        monkeypatch, rid, snap, ["config.json", "model.safetensors"]
    )

    captured = {}

    def _fake_load(path, *a, **k):
        captured["path"] = Path(path)
        return "parakeet-model"

    monkeypatch.setattr(stt_utils, "load_model", _fake_load)

    node = stt.Parakeet()
    await node.preload_model(MagicMock())

    assert node._stt_model == "parakeet-model"
    assert captured["path"] == snap


# ---------------------------------------------------------------------------
# espeak / phonemizer setup
#
# Many mlx-audio TTS models (Kokoro, Kitten, Melo, ...) phonemize text via
# phonemizer's espeak backend, which raises "espeak not installed on your
# system" unless pointed at a library. ``espeakng_loader`` bundles one; the
# TTS base must wire it so these models work without a system espeak-ng.
# ---------------------------------------------------------------------------
def test_configure_espeak_wires_bundled_library(monkeypatch):
    espeakng_loader = pytest.importorskip("espeakng_loader")
    pytest.importorskip("phonemizer")
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    captured = {}
    monkeypatch.setattr(
        EspeakWrapper,
        "set_library",
        classmethod(lambda cls, p: captured.__setitem__("lib", p)),
    )
    monkeypatch.setattr(
        EspeakWrapper,
        "set_data_path",
        classmethod(lambda cls, p: captured.__setitem__("data", p)),
    )

    tts.BaseMLXTTS._configure_espeak()

    assert captured["lib"] == espeakng_loader.get_library_path()
    assert captured["data"] == espeakng_loader.get_data_path()


def test_configure_espeak_is_noop_without_phonemizer(monkeypatch):
    # Models that don't need espeak must still load when the optional
    # phonemizer stack is absent.
    import builtins

    real_import = builtins.__import__

    def _fail_phonemizer(name, *args, **kwargs):
        if name.startswith("phonemizer") or name == "espeakng_loader":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_phonemizer)
    # Must not raise.
    tts.BaseMLXTTS._configure_espeak()


async def test_tts_preload_configures_espeak(monkeypatch, tmp_path):
    import mlx_audio.tts.utils as tts_utils

    rid = "mlx-community/kitten-tts-nano-0.8"
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")

    monkeypatch.setattr(tts.sys, "platform", "darwin")
    _patch_revision_only_cache(monkeypatch, rid, snap, ["config.json"])
    monkeypatch.setattr(tts_utils, "load_model", lambda *a, **k: "kitten-model")

    called = {}
    monkeypatch.setattr(
        tts.BaseMLXTTS,
        "_configure_espeak",
        staticmethod(lambda: called.__setitem__("configured", True)),
    )

    await tts.KittenTTS().preload_model(MagicMock())
    assert called.get("configured") is True
