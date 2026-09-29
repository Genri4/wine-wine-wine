"""Deterministic SIFT geometry over an already retrieved candidate set.

This module deliberately has no ground-truth input. It can only score and
reorder the candidate rows supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SIFTConfig:
    nfeatures: int = 4096
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    max_image_side: int = 1600
    ratio_test: float = 0.75
    ransac_reprojection_threshold: float = 4.0
    min_good_matches: int = 12
    min_inliers: int = 8
    min_inlier_ratio: float = 0.18
    min_projected_area_ratio: float = 1.0 / 40.0
    max_projected_area_ratio: float = 40.0
    min_spatial_coverage: float = 0.0


DEFAULT_CONFIG = SIFTConfig()


@dataclass
class SIFTFeatures:
    points_xy: np.ndarray
    descriptors: np.ndarray
    image_width: int
    image_height: int

    @property
    def keypoint_count(self) -> int:
        return int(self.points_xy.shape[0])


@dataclass(frozen=True)
class GeometryScore:
    num_query_keypoints: int
    num_reference_keypoints: int
    raw_matches: int
    good_matches: int
    ransac_inliers: int
    inlier_ratio: float
    homography_valid: bool
    homography_reason: str
    reprojection_error: float | None
    projected_area_ratio: float | None
    spatial_coverage: float
    geometric_score: float
    # Diagnostic alignment payload for downstream local-patch experiments.
    # Coordinates follow the cached feature domain: reference -> query.
    homography_ref_to_query: np.ndarray | None = None
    inlier_reference_points: np.ndarray | None = None


def sift_cache_fingerprint(reference_records: Iterable[tuple[str, str]], config: SIFTConfig = DEFAULT_CONFIG) -> str:
    """Fingerprint the extractor and the ordered slug/path/content manifest."""
    records = []
    for slug, path in reference_records:
        records.append({"slug": slug, "path": path, "sha256": file_sha256(path)})
    payload = {
        "opencv_version": cv2.__version__,
        "sift_config": asdict(config),
        "references": sorted(records, key=lambda row: row["slug"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_filename(slug: str) -> str:
    """Use a collision-safe stable filename for a catalog slug."""
    return hashlib.sha256(slug.encode("utf-8")).hexdigest() + ".npz"


def validate_top5_candidates(candidate_slugs: Sequence[str]) -> list[str]:
    """Validate the inference boundary: exactly five unique retrieved items."""
    candidates = list(candidate_slugs)
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError(f"Expected exactly five unique candidates; got {len(candidates)}")
    return candidates


def _sift(config: SIFTConfig = DEFAULT_CONFIG):
    return cv2.SIFT_create(
        nfeatures=config.nfeatures,
        nOctaveLayers=config.n_octave_layers,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )


def read_rgb(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def extract_sift(image_rgb: np.ndarray, config: SIFTConfig = DEFAULT_CONFIG) -> SIFTFeatures:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("SIFT input must be an RGB HxWx3 image")
    height, width = image_rgb.shape[:2]
    scale = min(1.0, config.max_image_side / max(width, height))
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = _sift(config).detectAndCompute(gray, None)
    points = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors, dtype=np.float32)
    return SIFTFeatures(points, descriptors, int(image_rgb.shape[1]), int(image_rgb.shape[0]))


def save_feature_cache(path: str | Path, features: SIFTFeatures, image_sha256: str, fingerprint: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_xy=features.points_xy.astype(np.float32),
        descriptors=features.descriptors.astype(np.float32),
        image_width=np.int32(features.image_width),
        image_height=np.int32(features.image_height),
        image_sha256=np.asarray(image_sha256),
        fingerprint=np.asarray(fingerprint),
    )


def load_feature_cache(path: str | Path, expected_image_sha256: str, expected_fingerprint: str) -> SIFTFeatures | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["image_sha256"].item()) != expected_image_sha256:
                return None
            if str(data["fingerprint"].item()) != expected_fingerprint:
                return None
            points = np.asarray(data["points_xy"], dtype=np.float32)
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            if len(points) != len(descriptors):
                return None
            return SIFTFeatures(
                points,
                descriptors,
                int(data["image_width"].item()),
                int(data["image_height"].item()),
            )
    except (OSError, KeyError, ValueError):
        return None


def validate_homography(
    homography: np.ndarray | None,
    reference_width: int,
    reference_height: int,
    query_width: int,
    query_height: int,
    config: SIFTConfig = DEFAULT_CONFIG,
) -> tuple[bool, str, float | None]:
    """Reject degenerate transforms and implausible projected reference area."""
    if homography is None:
        return False, "homography_missing", None
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False, "homography_non_finite_or_bad_shape", None
    if abs(float(np.linalg.det(matrix))) < 1e-12 or np.linalg.matrix_rank(matrix) < 3:
        return False, "homography_singular", None

    corners = np.float32(
        [[0, 0], [reference_width - 1, 0], [reference_width - 1, reference_height - 1], [0, reference_height - 1]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False, "projected_corners_non_finite", None
    if not cv2.isContourConvex(projected.astype(np.float32).reshape(-1, 1, 2)):
        return False, "projected_polygon_not_convex", None
    ref_area = float(reference_width * reference_height)
    projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    area_ratio = projected_area / max(ref_area, 1.0)
    if area_ratio < config.min_projected_area_ratio or area_ratio > config.max_projected_area_ratio:
        return False, "projected_area_out_of_range", area_ratio
    # A projection wholly outside the query is almost always a false match.
    qbox = np.float32([[0, 0], [query_width - 1, 0], [query_width - 1, query_height - 1], [0, query_height - 1]])
    overlap_area, _ = cv2.intersectConvexConvex(projected.astype(np.float32), qbox)
    if overlap_area <= 0:
        return False, "projection_outside_query", area_ratio
    return True, "valid", area_ratio


def _spatial_coverage(reference_points: np.ndarray, query_points: np.ndarray, inlier_indices: np.ndarray,
                      reference: SIFTFeatures, query: SIFTFeatures) -> float:
    if len(inlier_indices) < 3:
        return 0.0
    ref_hull = cv2.convexHull(reference_points[inlier_indices].astype(np.float32))
    query_hull = cv2.convexHull(query_points[inlier_indices].astype(np.float32))
    ref_coverage = float(cv2.contourArea(ref_hull)) / max(reference.image_width * reference.image_height, 1)
    query_coverage = float(cv2.contourArea(query_hull)) / max(query.image_width * query.image_height, 1)
    return float(min(ref_coverage, query_coverage))


def match_sift_pair(
    query: SIFTFeatures,
    reference: SIFTFeatures,
    config: SIFTConfig = DEFAULT_CONFIG,
) -> GeometryScore:
    """Match one query-reference pair and validate its RANSAC homography."""
    base = dict(
        num_query_keypoints=query.keypoint_count,
        num_reference_keypoints=reference.keypoint_count,
    )
    if query.descriptors.size == 0 or reference.descriptors.size == 0:
        return GeometryScore(**base, raw_matches=0, good_matches=0, ransac_inliers=0, inlier_ratio=0.0,
                             homography_valid=False, homography_reason="no_descriptors", reprojection_error=None,
                             projected_area_ratio=None, spatial_coverage=0.0, geometric_score=0.0)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(query.descriptors, reference.descriptors, k=2)
    good = [pair[0] for pair in knn_matches if len(pair) == 2 and pair[0].distance < config.ratio_test * pair[1].distance]
    if len(good) < 4:
        return GeometryScore(**base, raw_matches=len(knn_matches), good_matches=len(good), ransac_inliers=0,
                             inlier_ratio=0.0, homography_valid=False, homography_reason="too_few_good_matches",
                             reprojection_error=None, projected_area_ratio=None, spatial_coverage=0.0,
                             geometric_score=0.0)

    query_points = np.asarray([query.points_xy[match.queryIdx] for match in good], dtype=np.float32)
    reference_points = np.asarray([reference.points_xy[match.trainIdx] for match in good], dtype=np.float32)
    cv2.setRNGSeed(0)
    homography, inlier_mask = cv2.findHomography(
        reference_points.reshape(-1, 1, 2), query_points.reshape(-1, 1, 2),
        cv2.RANSAC, config.ransac_reprojection_threshold,
    )
    if homography is None or inlier_mask is None:
        return GeometryScore(**base, raw_matches=len(knn_matches), good_matches=len(good), ransac_inliers=0,
                             inlier_ratio=0.0, homography_valid=False, homography_reason="ransac_failed",
                             reprojection_error=None, projected_area_ratio=None, spatial_coverage=0.0,
                             geometric_score=0.0)

    inlier_indices = np.flatnonzero(inlier_mask.reshape(-1).astype(bool))
    inliers = int(len(inlier_indices))
    ratio = inliers / max(len(good), 1)
    is_valid, reason, area_ratio = validate_homography(
        homography, reference.image_width, reference.image_height,
        query.image_width, query.image_height, config,
    )
    if inliers < config.min_inliers:
        is_valid, reason = False, "too_few_ransac_inliers"
    elif ratio < config.min_inlier_ratio:
        is_valid, reason = False, "low_inlier_ratio"
    elif len(good) < config.min_good_matches:
        is_valid, reason = False, "too_few_good_matches"

    projected_points = cv2.perspectiveTransform(reference_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected_points[inlier_indices] - query_points[inlier_indices], axis=1)
    reprojection_error = float(np.median(errors)) if len(errors) else None
    coverage = _spatial_coverage(reference_points, query_points, inlier_indices, reference, query)
    if coverage < config.min_spatial_coverage:
        is_valid, reason = False, "low_spatial_coverage"
    score = 0.0
    if is_valid and reprojection_error is not None:
        score = float(np.log1p(inliers) * ratio * np.exp(-reprojection_error / 8.0) * np.sqrt(max(coverage, 1e-6)))
    return GeometryScore(
        **base,
        raw_matches=len(knn_matches),
        good_matches=len(good),
        ransac_inliers=inliers,
        inlier_ratio=float(ratio),
        homography_valid=bool(is_valid),
        homography_reason=reason,
        reprojection_error=reprojection_error,
        projected_area_ratio=area_ratio,
        spatial_coverage=coverage,
        geometric_score=score,
        homography_ref_to_query=np.asarray(homography, dtype=np.float64),
        inlier_reference_points=np.asarray(reference_points[inlier_indices], dtype=np.float32),
    )


def normalize_geometry_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize within a query; flat/no evidence maps to all zero."""
    if not scores:
        return []
    clean = [max(float(score), 0.0) for score in scores]
    low, high = min(clean), max(clean)
    if high - low <= 1e-12:
        return [0.0 for _ in clean]
    return [(score - low) / (high - low) for score in clean]


def valid_geometry_scores(geometry: Sequence[GeometryScore]) -> list[float]:
    """Invalid/failed transforms always contribute zero reranking evidence."""
    return [float(item.geometric_score) if item.homography_valid else 0.0 for item in geometry]


def fuse_scores(base_scores: Sequence[float], geometry_scores: Sequence[float], weight: float) -> list[float]:
    if len(base_scores) != len(geometry_scores):
        raise ValueError("base and geometry score arrays must align")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    base = normalize_geometry_scores(base_scores)
    geometry = normalize_geometry_scores(geometry_scores)
    return [(1.0 - weight) * b + weight * g for b, g in zip(base, geometry)]


def rank_candidates(slugs: Sequence[str], scores: Sequence[float]) -> list[str]:
    if len(slugs) != len(scores):
        raise ValueError("candidate slugs and scores must align")
    return [slug for _, slug in sorted(enumerate(slugs), key=lambda item: (-float(scores[item[0]]), item[0]))]


def conservative_geometry_order(
    slugs: Sequence[str], geometry: Sequence[GeometryScore], normalized_scores: Sequence[float],
    min_geo_margin: float = 0.15, min_inliers: int = 12, min_ratio: float = 0.25,
) -> list[str]:
    """Change current Top-1 only when valid, strong geometry has a fixed margin."""
    if len(slugs) != len(geometry) or len(slugs) != len(normalized_scores):
        raise ValueError("candidate, geometry and score arrays must align")
    order = sorted(range(len(slugs)), key=lambda index: (-float(normalized_scores[index]), index))
    best_geo = max(range(len(slugs)), key=lambda index: (float(normalized_scores[index]), -index))
    if best_geo == order[0]:
        return [slugs[index] for index in order]
    evidence = geometry[best_geo]
    current_score = float(normalized_scores[order[0]])
    best_score = float(normalized_scores[best_geo])
    if (evidence.homography_valid and evidence.ransac_inliers >= min_inliers
            and evidence.inlier_ratio >= min_ratio and best_score - current_score >= min_geo_margin):
        order.remove(best_geo)
        order.insert(0, best_geo)
    return [slugs[index] for index in order]


def transition_counts(before: Sequence[str], after: Sequence[str], targets: Sequence[str]) -> dict[str, int]:
    if not (len(before) == len(after) == len(targets)):
        raise ValueError("before, after and targets must align")
    counts = {"wrong_to_correct": 0, "correct_to_wrong": 0, "wrong_to_wrong": 0, "correct_to_correct": 0}
    for old, new, target in zip(before, after, targets):
        old_correct, new_correct = old == target, new == target
        if not old_correct and new_correct:
            counts["wrong_to_correct"] += 1
        elif old_correct and not new_correct:
            counts["correct_to_wrong"] += 1
        elif old_correct:
            counts["correct_to_correct"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return counts
