"""Stable Audio 3 nodes must construct the way the worker builds them."""

from __future__ import annotations

import pytest

from nodetool.nodes.mlx.text_to_audio import (
    StableAudio3AudioToAudio,
    StableAudio3Inpaint,
)


@pytest.mark.parametrize("node_cls", [StableAudio3AudioToAudio, StableAudio3Inpaint])
async def test_node_constructs_with_id_only_and_requires_audio_at_process(node_cls):
    # The worker executor builds nodes as ``node_class(id=...)`` and assigns
    # properties afterwards, so every field needs a default.
    node = node_cls(id="n1")
    assert node.audio.is_empty()

    with pytest.raises(ValueError, match="input audio clip is required"):
        await node.process(context=None)  # type: ignore[arg-type]
