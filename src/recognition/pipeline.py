"""Online recognition pipeline: encode, search, and apply confidence policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .confidence import decide_unknown
from .index import CatalogIndex, SearchResult


@dataclass(frozen=True)
class Prediction:
    query_image: str
    predicted_item_id: str | None
    is_unknown: bool
    confidence_score: float | None
    unknown_threshold: float
    candidates: list[SearchResult]
    latency_ms: float
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "image": self.query_image,
            "prediction": self.predicted_item_id,
            "unknown": self.is_unknown,
            "confidence": self.confidence_score,
            "unknown_threshold": self.unknown_threshold,
            "top_k": [candidate.to_dict() for candidate in self.candidates],
            "latency_ms": self.latency_ms,
            "embedding_latency_ms": self.embedding_latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
        }


class RecognitionPipeline:
    """Compose one encoder with one exact-search catalog index.

    A replacement encoder only needs an ``encode(paths)`` method returning one
    vector per path. No separate plugin or interface layer is needed for this
    baseline.
    """

    def __init__(
        self,
        encoder: Any,
        index: CatalogIndex,
        unknown_threshold: float = 0.5,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be positive")
        # Validate the range once at construction time.
        decide_unknown(None, unknown_threshold)
        self.encoder = encoder
        self.index = index
        self.unknown_threshold = unknown_threshold
        self.default_top_k = default_top_k

    def predict(self, image_path: str | Path, top_k: int | None = None) -> Prediction:
        requested_top_k = self.default_top_k if top_k is None else top_k
        if requested_top_k < 1:
            raise ValueError("top_k must be positive")

        query_path = Path(image_path)
        started = perf_counter()
        embedding_started = perf_counter()
        embeddings = self.encoder.encode([query_path])
        embedding_latency_ms = (perf_counter() - embedding_started) * 1000.0
        if len(embeddings) != 1:
            raise ValueError("Encoder must return exactly one embedding for one image")
        retrieval_started = perf_counter()
        candidates = self.index.search(
            embeddings[0],
            requested_top_k,
            device=getattr(self.encoder, "device", None),
        )
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000.0
        score = candidates[0].score if candidates else None
        decision = decide_unknown(score, self.unknown_threshold)
        predicted_item_id = None if decision.is_unknown else candidates[0].item_id
        latency_ms = (perf_counter() - started) * 1000.0
        return Prediction(
            query_image=str(query_path),
            predicted_item_id=predicted_item_id,
            is_unknown=decision.is_unknown,
            confidence_score=decision.score,
            unknown_threshold=decision.threshold,
            candidates=candidates,
            latency_ms=round(latency_ms, 3),
            embedding_latency_ms=round(embedding_latency_ms, 3),
            retrieval_latency_ms=round(retrieval_latency_ms, 3),
        )
