"""Tests for the ACE-Step 1.5 music generation nodes.

These tests cover the platform-independent surface of the nodes (metadata,
field wiring and the defensive helpers). The actual generation paths require
macOS, MLX and the ``acestep`` package and are therefore not exercised here.
"""

from __future__ import annotations

import dataclasses

import pytest

from nodetool.nodes.mlx.text_to_music import (
    ACEStepBaseNode,
    ACEStepMusicGeneration,
    ACEStepSongPlanner,
    DiTModel,
    LanguageModel,
    VocalLanguage,
)


def test_node_types_and_titles():
    assert (
        ACEStepMusicGeneration.get_node_type()
        == "mlx.text_to_music.ACEStepMusicGeneration"
    )
    assert ACEStepSongPlanner.get_node_type() == "mlx.text_to_music.ACEStepSongPlanner"
    assert ACEStepMusicGeneration.get_title() == "ACE-Step Music Generation"
    assert ACEStepSongPlanner.get_title() == "ACE-Step Song Planner"


def test_base_node_is_hidden_but_concrete_nodes_visible():
    assert ACEStepBaseNode.is_visible() is False
    assert ACEStepMusicGeneration.is_visible() is True
    assert ACEStepSongPlanner.is_visible() is True


def test_required_inputs():
    assert ACEStepMusicGeneration().required_inputs() == ["prompt"]
    assert ACEStepSongPlanner().required_inputs() == ["query"]


def test_nodes_are_not_cacheable():
    assert ACEStepMusicGeneration.is_cacheable() is False
    assert ACEStepSongPlanner.is_cacheable() is False


def test_recommended_models_use_ace_step_org():
    for cls in (ACEStepMusicGeneration, ACEStepSongPlanner):
        repos = [m.repo_id for m in cls.get_recommended_models()]
        assert repos, f"{cls.__name__} should recommend at least one model"
        assert all(r.startswith("ACE-Step/") for r in repos)
    music_repos = [m.repo_id for m in ACEStepMusicGeneration.get_recommended_models()]
    assert "ACE-Step/Ace-Step1.5" in music_repos


def test_dit_and_lm_enum_values():
    assert DiTModel.TURBO.value == "acestep-v15-turbo"
    assert DiTModel.XL_BASE.value == "acestep-v15-xl-base"
    # DISABLED maps to an empty config so generation can run DiT-only.
    assert LanguageModel.DISABLED.value == ""
    assert LanguageModel.LM_1_7B.value == "acestep-5Hz-lm-1.7B"
    assert VocalLanguage.UNKNOWN.value == "unknown"


def test_default_music_node_runs_dit_only():
    node = ACEStepMusicGeneration()
    assert node.language_model == LanguageModel.DISABLED
    assert node.dit_model == DiTModel.TURBO


def test_dataclass_kwargs_filters_unknown_fields():
    @dataclasses.dataclass
    class Sample:
        a: int = 0
        b: str = ""

    filtered = ACEStepBaseNode._dataclass_kwargs(Sample, a=1, b="x", c=99)
    assert filtered == {"a": 1, "b": "x"}


def test_call_filtered_drops_unsupported_kwargs():
    def only_a(a: int) -> int:
        return a

    # ``b`` is silently dropped because ``only_a`` does not accept it.
    assert ACEStepBaseNode._call_filtered(only_a, a=5, b=6) == 5


def test_call_filtered_passes_all_when_var_keyword():
    def accepts_kwargs(**kwargs):
        return kwargs

    result = ACEStepBaseNode._call_filtered(accepts_kwargs, a=1, b=2)
    assert result == {"a": 1, "b": 2}


def test_resolve_paths_honours_checkpoints_env(tmp_path, monkeypatch):
    ckpt = tmp_path / "my_checkpoints"
    monkeypatch.setenv("ACESTEP_CHECKPOINTS_DIR", str(ckpt))
    monkeypatch.delenv("ACESTEP_PROJECT_ROOT", raising=False)
    project_root, checkpoints_dir = ACEStepBaseNode._resolve_paths()
    assert checkpoints_dir == str(ckpt)
    assert project_root == str(tmp_path)
    assert ckpt.is_dir()


def test_song_planner_result_parsing_from_dict():
    result = {
        "caption": "warm acoustic ballad",
        "lyrics": "[verse]\nhello",
        "bpm": "90",
        "keyscale": "C major",
        "timesignature": "4/4",
        "vocal_language": "English",
        "success": True,
    }
    out = ACEStepSongPlanner._result_to_output(result)
    assert out == {
        "caption": "warm acoustic ballad",
        "lyrics": "[verse]\nhello",
        "bpm": 90,
        "keyscale": "C major",
        "time_signature": "4/4",
        "vocal_language": "English",
    }


def test_song_planner_result_parsing_from_dataclass():
    @dataclasses.dataclass
    class CreateSampleResult:
        caption: str = "lo-fi beat"
        lyrics: str = ""
        bpm: int = 75
        keyscale: str = ""
        timesignature: str = ""
        vocal_language: str = "unknown"
        success: bool = True

    out = ACEStepSongPlanner._result_to_output(CreateSampleResult())
    assert out["caption"] == "lo-fi beat"
    assert out["bpm"] == 75
    assert out["vocal_language"] == "unknown"


def test_song_planner_result_raises_on_failure():
    with pytest.raises(RuntimeError, match="boom"):
        ACEStepSongPlanner._result_to_output({"success": False, "error": "boom"})


def test_song_planner_handles_missing_bpm():
    out = ACEStepSongPlanner._result_to_output({"caption": "x"})
    assert out["bpm"] is None
    assert out["caption"] == "x"
