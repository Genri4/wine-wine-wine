"""Small, replaceable baseline for image-based catalog recognition."""

from .data import CatalogItem, EvaluationExample
from .encoder import SUPPORTED_MODEL_NAMES, VisualEncoder
from .index import CatalogIndex, SearchResult
from .pipeline import Prediction, RecognitionPipeline

__all__ = [
    "CatalogIndex",
    "CatalogItem",
    "EvaluationExample",
    "Prediction",
    "RecognitionPipeline",
    "SearchResult",
    "SUPPORTED_MODEL_NAMES",
    "VisualEncoder",
]
