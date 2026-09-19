"""A small on-disk tensor catalog index and exact cosine search."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

from .data import CatalogItem


@dataclass(frozen=True)
class CatalogEntry:
    item_id: str
    image_path: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    image_path: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "image": self.image_path,
            "score": self.score,
            "rank": self.rank,
        }


class CatalogIndex:
    """PyTorch ``.pt`` index with exact cosine search and nearby metadata."""

    FORMAT_VERSION = 2
    SUPPORTED_SUFFIXES = {".pt", ".pth"}

    def __init__(
        self,
        entries: Sequence[CatalogEntry],
        encoder_name: str,
        embedding_dim: int,
        model_name: str | None = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        for entry in entries:
            if len(entry.embedding) != embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for {entry.item_id}: "
                    f"expected {embedding_dim}, got {len(entry.embedding)}"
                )
        self.entries = tuple(entries)
        self.encoder_name = encoder_name
        self.model_name = model_name or encoder_name
        self.embedding_dim = embedding_dim
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for the catalog index") from exc
        self._embedding_matrix = torch.tensor(
            [entry.embedding for entry in self.entries], dtype=torch.float32
        )
        self._device_matrices: dict[str, Any] = {"cpu": self._embedding_matrix}

    @classmethod
    def from_items(
        cls,
        items: Sequence[CatalogItem],
        embeddings: Sequence[Sequence[float]],
        encoder_name: str,
        model_name: str | None = None,
    ) -> "CatalogIndex":
        if not items:
            raise ValueError("Cannot build an index from an empty catalog")
        if len(items) != len(embeddings):
            raise ValueError("Number of items and embeddings must match")

        embedding_dim = len(embeddings[0])
        if embedding_dim < 1:
            raise ValueError("Embeddings must not be empty")

        entries: list[CatalogEntry] = []
        for item, embedding in zip(items, embeddings):
            if len(embedding) != embedding_dim:
                raise ValueError("All embeddings must have the same dimension")
            entries.append(
                CatalogEntry(
                    item_id=item.item_id,
                    image_path=str(item.image_path),
                    embedding=_normalize(embedding),
                )
            )
        return cls(entries, encoder_name, embedding_dim, model_name=model_name)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        device: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(query_embedding)}"
            )
        if not self.entries:
            return []

        # Exact retrieval is deliberately a plain normalized matrix
        # multiplication. No FAISS/vector database is needed for this catalog.
        import torch

        matrix = self._matrix_for_device(device)
        query = torch.tensor(
            [list(float(value) for value in query_embedding)],
            dtype=torch.float32,
            device=matrix.device,
        )
        query = torch.nn.functional.normalize(query, p=2, dim=1)
        scores = torch.matmul(matrix, query.transpose(0, 1)).squeeze(1)
        scored = list(zip(scores.detach().cpu().tolist(), self.entries))
        scored.sort(key=lambda pair: (-pair[0], pair[1].item_id))
        return [
            SearchResult(
                item_id=entry.item_id,
                image_path=entry.image_path,
                score=float(score),
                rank=rank,
            )
            for rank, (score, entry) in enumerate(scored[:top_k], start=1)
        ]

    def _matrix_for_device(self, device: str | None) -> Any:
        import torch

        resolved = str(torch.device(device or "cpu"))
        matrix = self._device_matrices.get(resolved)
        if matrix is None:
            matrix = self._embedding_matrix.to(resolved)
            self._device_matrices[resolved] = matrix
        return matrix

    def prepare_device(self, device: str | None = None) -> None:
        """Materialize the catalog matrix before measuring per-query latency."""

        self._matrix_for_device(device)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        if output_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError("Catalog index must use a .pt or .pth path")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to save a .pt catalog index") from exc

        embeddings = self._embedding_matrix
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model_name": self.model_name,
            "encoder_name": self.encoder_name,
            "embedding_dim": self.embedding_dim,
            "embeddings": embeddings,
            "items": [
                {"item_id": entry.item_id, "image": entry.image_path}
                for entry in self.entries
            ],
        }
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "CatalogIndex":
        input_path = Path(path)
        if input_path.suffix.lower() not in cls.SUPPORTED_SUFFIXES:
            raise ValueError("Catalog index must use a .pt or .pth path")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to load a .pt catalog index") from exc

        try:
            payload = torch.load(input_path, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older PyTorch versions.
            payload = torch.load(input_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Malformed catalog index")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"Unsupported index format: {payload.get('format_version')}")

        encoder_name = payload.get("encoder_name")
        model_name = payload.get("model_name")
        embedding_dim = payload.get("embedding_dim")
        embeddings = payload.get("embeddings")
        raw_items = payload.get("items")
        if not isinstance(encoder_name, str) or not isinstance(embedding_dim, int):
            raise ValueError("Index metadata is incomplete")
        if model_name is not None and not isinstance(model_name, str):
            raise ValueError("Index model_name must be a string")
        if not isinstance(raw_items, list) or not hasattr(embeddings, "ndim"):
            raise ValueError("Index must contain an embeddings tensor and items list")
        if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
            raise ValueError("Index embedding tensor has an invalid shape")
        if len(raw_items) != embeddings.shape[0]:
            raise ValueError("Index item metadata and embeddings must have the same length")

        entries: list[CatalogEntry] = []
        embedding_rows = embeddings.detach().cpu().tolist()
        for raw_item, embedding in zip(raw_items, embedding_rows):
            if not isinstance(raw_item, dict):
                raise ValueError("Each index item must be an object")
            try:
                entries.append(
                    CatalogEntry(
                        item_id=str(raw_item["item_id"]),
                        image_path=str(raw_item["image"]),
                        embedding=[float(value) for value in embedding],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Malformed index item") from exc
        return cls(entries, encoder_name, embedding_dim, model_name=model_name)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    left_norm = sqrt(sum(float(value) ** 2 for value in left))
    right_norm = sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(float(a) * float(b) for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = sqrt(sum(float(value) ** 2 for value in vector))
    if norm == 0.0:
        raise ValueError("Zero vectors cannot be indexed")
    return [float(value) / norm for value in vector]
