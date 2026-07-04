"""Helpers for locating already-downloaded models in the Hugging Face cache.

``huggingface_hub.try_to_load_from_cache`` resolves a file relative to a branch
(``revision="main"`` by default). Models fetched by commit/revision — which is
common for the ``mlx-community`` weights pulled by ``mlx-audio`` — are cached
*without* a ``refs/main`` pointer, so that lookup returns ``None`` even though
every file is present on disk. That manifested as a bogus "model must be
downloaded first" error for models that were, in fact, downloaded.

``find_cached_snapshot`` keeps the fast ``try_to_load_from_cache`` path but falls
back to scanning the cache for any revision that actually contains the marker
file, making model detection independent of the ``refs/main`` pointer.
"""

from __future__ import annotations

from pathlib import Path


def find_cached_snapshot(repo_id: str, marker: str = "config.json") -> Path | None:
    """Return the cached snapshot directory containing ``marker``, or ``None``.

    The returned path is the snapshot root (the directory that holds ``marker``
    at its given relative location), suitable for passing to ``mlx_audio``'s
    ``load_model``.
    """
    import huggingface_hub

    # Fast path: works whenever a ``refs/main`` pointer exists.
    cached = huggingface_hub.try_to_load_from_cache(repo_id, marker)
    if isinstance(cached, str):
        root = Path(cached)
        for _ in Path(marker).parts:
            root = root.parent
        return root

    # Fallback: scan every cached revision for one that holds the marker file.
    try:
        cache = huggingface_hub.scan_cache_dir()
    except Exception:
        return None

    for repo in cache.repos:
        if repo.repo_id != repo_id or repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            snapshot_path = Path(revision.snapshot_path)
            for file_info in revision.files:
                try:
                    relative = Path(file_info.file_path).relative_to(snapshot_path)
                except ValueError:
                    continue
                if str(relative) == marker:
                    return snapshot_path
    return None
