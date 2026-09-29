"""Deterministic local-patch alignment and small Top-5 score fusion helpers.

This module is experiment-only. It accepts a fixed candidate list and never
uses a target label to create features, choose transforms, or reorder items.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AlignedOverlap:
    query_patch: Image.Image
    reference_patch: Image.Image
    overlap_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    overlap_fraction: float


def local_feature_cache_fingerprint(
    method_name: str, method_version: str, params: dict, reference_records: Sequence[tuple[str, str]]
) -> str:
    """Stable cache identity for the feature/matcher setup and reference bytes."""
    payload = {
        "method_name": method_name,
        "method_version": method_version,
        "params": params,
        "references": sorted((slug, image_sha256) for slug, image_sha256 in reference_records),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_feature_alignment(keypoints: np.ndarray, descriptors: np.ndarray) -> None:
    keypoints = np.asarray(keypoints)
    descriptors = np.asarray(descriptors)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N, 2]")
    if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
        raise ValueError("one descriptor row is required per keypoint")
    if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all():
        raise ValueError("feature cache contains non-finite values")


def validate_normalized_embeddings(embeddings: np.ndarray, atol: float = 2e-3) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite [N, D] matrix")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(np.abs(norms - 1.0) > atol):
        raise ValueError("local visual embeddings must be L2 normalized")


def validate_five_candidate_pairs(candidate_slugs: Sequence[str], pair_count: int) -> None:
    if len(candidate_slugs) != 5 or len(set(candidate_slugs)) != 5 or pair_count != 5:
        raise ValueError("online local reranking must score all five unique frozen candidates")


def resize_rgb_max_side(image_rgb: np.ndarray, max_side: int = 1600) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB HxWx3 image")
    height, width = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image_rgb
    return cv2.resize(
        image_rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def warp_aligned_overlap(
    query_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    homography_ref_to_query: np.ndarray,
    inlier_reference_points: np.ndarray | None = None,
    *,
    max_side: int = 1600,
    keypoint_padding: float = 0.18,
    min_overlap_fraction: float = 0.30,
    min_patch_side: int = 32,
) -> AlignedOverlap | None:
    """Warp the query into reference coordinates and return matched overlap.

    The same deterministic matched-point region is used regardless of which
    reference is the catalog target. Invalid pixels in both crops receive the
    same neutral fill so the encoders compare only visible shared content.
    """
    query = resize_rgb_max_side(query_rgb, max_side)
    reference = resize_rgb_max_side(reference_rgb, max_side)
    h_ref_to_query = np.asarray(homography_ref_to_query, dtype=np.float64)
    if h_ref_to_query.shape != (3, 3) or not np.isfinite(h_ref_to_query).all():
        return None
    if abs(float(np.linalg.det(h_ref_to_query))) < 1e-12:
        return None
    try:
        h_query_to_ref = np.linalg.inv(h_ref_to_query)
    except np.linalg.LinAlgError:
        return None

    ref_h, ref_w = reference.shape[:2]
    q_h, q_w = query.shape[:2]
    warped_query = cv2.warpPerspective(
        query,
        h_query_to_ref,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(127, 127, 127),
    )
    query_mask = np.full((q_h, q_w), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        query_mask,
        h_query_to_ref,
        (ref_w, ref_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    if not np.any(warped_mask):
        return None

    if inlier_reference_points is not None and len(inlier_reference_points) >= 4:
        points = np.asarray(inlier_reference_points, dtype=np.float32).reshape(-1, 2)
        if np.isfinite(points).all():
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            width, height = max(1.0, float(right - left)), max(1.0, float(bottom - top))
            left -= width * keypoint_padding
            right += width * keypoint_padding
            top -= height * keypoint_padding
            bottom += height * keypoint_padding
            x1 = max(0, int(np.floor(left)))
            y1 = max(0, int(np.floor(top)))
            x2 = min(ref_w, int(np.ceil(right)) + 1)
            y2 = min(ref_h, int(np.ceil(bottom)) + 1)
        else:
            x1, y1, x2, y2 = 0, 0, ref_w, ref_h
    else:
        x1, y1, x2, y2 = 0, 0, ref_w, ref_h

    if x2 - x1 < min_patch_side or y2 - y1 < min_patch_side:
        return None
    overlap = warped_mask[y1:y2, x1:x2]
    overlap_fraction = float(overlap.mean()) if overlap.size else 0.0
    if overlap_fraction < min_overlap_fraction:
        return None

    query_crop = warped_query[y1:y2, x1:x2].copy()
    reference_crop = reference[y1:y2, x1:x2].copy()
    invalid = ~overlap
    query_crop[invalid] = (127, 127, 127)
    reference_crop[invalid] = (127, 127, 127)
    return AlignedOverlap(
        query_patch=Image.fromarray(query_crop, mode="RGB"),
        reference_patch=Image.fromarray(reference_crop, mode="RGB"),
        overlap_mask=overlap.copy(),
        bbox_xyxy=(x1, y1, x2, y2),
        overlap_fraction=overlap_fraction,
    )


def normalize_candidate_scores(scores: Sequence[float | None]) -> list[float]:
    """Min-max normalize one query's signal; missing evidence maps to zero."""
    clean = [float(value) if value is not None and np.isfinite(value) else 0.0 for value in scores]
    if not clean:
        return []
    low, high = min(clean), max(clean)
    if high - low <= 1e-12:
        return [0.0] * len(clean)
    return [(value - low) / (high - low) for value in clean]


def fuse_local_signals(
    base_scores: Sequence[float],
    signal_vectors: Sequence[Sequence[float | None]],
    weight: float,
) -> list[float]:
    """Add the mean of fixed local signals to one existing Top-5 score."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    if any(len(values) != len(base_scores) for values in signal_vectors):
        raise ValueError("all Top-5 signal vectors must align with base scores")
    if not signal_vectors:
        return normalize_candidate_scores(base_scores)
    base = normalize_candidate_scores(base_scores)
    normalized = [normalize_candidate_scores(values) for values in signal_vectors]
    signal_mean = [sum(values[index] for values in normalized) / len(normalized) for index in range(len(base))]
    return [(1.0 - weight) * old + weight * new for old, new in zip(base, signal_mean)]


def rank_top5(candidate_slugs: Sequence[str], scores: Sequence[float]) -> list[str]:
    if len(candidate_slugs) != 5 or len(scores) != 5 or len(set(candidate_slugs)) != 5:
        raise ValueError("reranking requires exactly five unique fixed candidates")
    return [slug for _, slug in sorted(enumerate(candidate_slugs), key=lambda item: (-scores[item[0]], item[0]))]


def signal_oracle_wins(
    candidate_slugs: Sequence[str], signal_scores: Sequence[float | None], target_slug: str, incumbent_slug: str
) -> bool:
    """Whether raw candidate-local signal prefers target over incumbent."""
    if len(candidate_slugs) != len(signal_scores):
        raise ValueError("candidate and signal vectors must align")
    try:
        target_index = candidate_slugs.index(target_slug)
        incumbent_index = candidate_slugs.index(incumbent_slug)
    except ValueError:
        return False
    target_score, incumbent_score = signal_scores[target_index], signal_scores[incumbent_index]
    return target_score is not None and incumbent_score is not None and target_score > incumbent_score
