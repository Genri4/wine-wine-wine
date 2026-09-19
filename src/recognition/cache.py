"""Validated on-disk embedding cache for catalog experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


CACHE_FORMAT_VERSION = 1


class EmbeddingCacheMismatch(ValueError):
    """Raised when a cache exists but does not match the requested contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_embedding_cache(
    path: str | Path,
    embeddings: Sequence[Sequence[float]],
    metadata: Mapping[str, Any],
) -> None:
    """Save embeddings and their exact cache contract to a torch file."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for embedding cache") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "metadata": dict(metadata),
        "embeddings": torch.tensor(embeddings, dtype=torch.float32),
    }
    torch.save(payload, output_path)


def load_embedding_cache(
    path: str | Path,
    expected_metadata: Mapping[str, Any],
) -> list[list[float]]:
    """Load a cache only when every expected metadata field matches exactly."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for embedding cache") from exc

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    try:
        payload = torch.load(input_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        payload = torch.load(input_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise EmbeddingCacheMismatch("cache payload is not an object")
    if payload.get("format_version") != CACHE_FORMAT_VERSION:
        raise EmbeddingCacheMismatch("cache format version differs")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not cache_metadata_matches(metadata, expected_metadata):
        raise EmbeddingCacheMismatch("cache metadata differs")
    embeddings = payload.get("embeddings")
    if not hasattr(embeddings, "ndim") or embeddings.ndim != 2:
        raise EmbeddingCacheMismatch("cache embeddings have an invalid shape")
    expected_count = expected_metadata.get("catalog_count")
    expected_dim = expected_metadata.get("embedding_dim")
    if expected_count is not None and embeddings.shape[0] != expected_count:
        raise EmbeddingCacheMismatch("cache catalog size differs")
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise EmbeddingCacheMismatch("cache embedding dimension differs")
    return embeddings.detach().cpu().tolist()


def cache_metadata_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare the full nested cache contract, including preprocessing."""

    return all(actual.get(key) == value for key, value in expected.items())
