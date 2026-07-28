"""Guards against drift between the node code and the shipped package metadata.

``src/nodetool/package_metadata/nodetool-mlx.json`` is generated (via
``nodetool-pkg scan --write --enrich``) and committed, so it silently goes stale
whenever a node's model list changes without a regeneration. The Nodetool UI
reads that file to populate the model picker and the downloader, so stale
entries surface as models a user cannot select — or, worse, as offers to
download repositories that no longer exist.

These checks compare the committed file against the classes themselves, and
only for the fields the classes own. They deliberately ignore the enriched
Hugging Face fields (sizes, tags, download counts), which change on the Hub
without any change here.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nodetool"
    / "package_metadata"
    / "nodetool-mlx.json"
)


def _metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def _load_node_class(node_type: str):
    """Resolve ``mlx.image_to_text.MLXVisionLanguage`` to the class object."""
    namespace, class_name = node_type.rsplit(".", 1)
    module = importlib.import_module(f"nodetool.nodes.{namespace}")
    return getattr(module, class_name)


def _metadata_nodes() -> list[dict]:
    return _metadata()["nodes"]


def _node_ids() -> list[str]:
    return [node["node_type"] for node in _metadata_nodes()]


def test_metadata_is_valid_json_with_nodes():
    data = _metadata()
    assert data["name"] == "nodetool-mlx"
    assert data["nodes"], "metadata lists no nodes"


@pytest.mark.parametrize("node_type", _node_ids())
def test_metadata_node_class_exists(node_type: str):
    # A node renamed or removed in code but left in the metadata makes the UI
    # offer a node it cannot instantiate.
    assert _load_node_class(node_type) is not None


@pytest.mark.parametrize("node_type", _node_ids())
def test_metadata_recommended_models_match_code(node_type: str):
    node_cls = _load_node_class(node_type)
    metadata_entry = next(
        node for node in _metadata_nodes() if node["node_type"] == node_type
    )

    get_recommended = getattr(node_cls, "get_recommended_models", None)
    if get_recommended is None:
        return

    expected = [model.repo_id for model in get_recommended()]
    actual = [
        model["repo_id"] for model in metadata_entry.get("recommended_models", [])
    ]

    assert actual == expected, (
        f"{node_type}: recommended models in the metadata are out of date. "
        "Regenerate with `nodetool-pkg scan --write --enrich`."
    )
