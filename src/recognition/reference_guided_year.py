"""Reference-guided year crop via SIFT+RANSAC homography alignment.

For every relevant same-family Top-5 candidate:

    candidate reference image
      -> frozen eslav OCR boxes on the reference (cached)
      -> reference year bbox (the box whose recognized year matches the
         candidate's canonical year; provenance kept)
      -> SIFT + descriptor matching + RANSAC homography reference -> query
      -> homography validity gates (Part 17)
      -> project the year bbox into the query
      -> padded perspective/axis-aligned crop, upscale, OCR
      -> candidate-specific year evidence

No target identity is used anywhere: candidates, their references, boxes
and years all come from the catalog side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .vintage_disambiguation import extract_years_with_corrections, normalize_year_token

MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.18
MAX_AREA_RATIO = 40.0
MIN_AREA_RATIO = 1 / 40.0


@dataclass
class AlignmentResult:
    valid: bool
    reason: str
    homography: Any = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0


def compute_homography(
    reference_image: np.ndarray,
    query_image: np.ndarray,
) -> AlignmentResult:
    """SIFT + ratio-test matching + RANSAC homography ref -> query."""

    import cv2

    sift = cv2.SIFT_create(nfeatures=4096)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    gray_query = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_query, descriptors_query = sift.detectAndCompute(gray_query, None)
    if descriptors_ref is None or descriptors_query is None:
        return AlignmentResult(False, "no_descriptors")
    if len(keypoints_ref) < MIN_GOOD_MATCHES or len(keypoints_query) < MIN_GOOD_MATCHES:
        return AlignmentResult(False, "too_few_keypoints")
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors_ref, descriptors_query, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good.append(first)
    if len(good) < MIN_GOOD_MATCHES:
        return AlignmentResult(False, "too_few_good_matches", good_matches=len(good))
    src_points = np.float32([keypoints_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints_query[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        return AlignmentResult(False, "homography_failed", good_matches=len(good))
    inliers = int(inlier_mask.sum())
    ratio = inliers / len(good)
    if inliers < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
        return AlignmentResult(False, "weak_inliers", homography=homography, good_matches=len(good), inliers=inliers, inlier_ratio=ratio)
    return AlignmentResult(
        True,
        "valid",
        homography=homography,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=ratio,
    )


def project_bbox(bbox: Sequence[float], homography: Any) -> np.ndarray | None:
    """Project a reference-axis-aligned bbox (x1,y1,x2,y2) into the query."""

    import cv2

    x1, y1, x2, y2 = (float(value) for value in bbox)
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return projected


def projected_crop_bounds(projected: np.ndarray, query_width: int, query_height: int) -> tuple[int, int, int, int] | None:
    """Axis-aligned bounds of the projected polygon, clipped to the query."""

    xs = projected[:, 0]
    ys = projected[:, 1]
    x1, y1 = int(max(0, xs.min())), int(max(0, ys.min()))
    x2, y2 = int(min(query_width, xs.max())), int(min(query_height, ys.max()))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def crop_bounds_valid(bounds: tuple[int, int, int, int], reference_bbox: Sequence[float]) -> bool:
    """Area-ratio sanity gate (Part 17): projected crop must be comparable
    in size to the reference year box (within 1/40x..40x)."""

    reference_area = max((reference_bbox[2] - reference_bbox[0]) * (reference_bbox[3] - reference_bbox[1]), 1.0)
    crop_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    ratio = crop_area / reference_area
    return MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO


def reference_year_bbox(
    reference_lines: Sequence[Mapping[str, Any]],
    candidate_year: str | None,
) -> tuple[tuple[float, float, float, float], str] | None:
    """Find the reference OCR box whose text resolves to the candidate year.

    Provenance recorded with the box. Returns None when the year is not
    confidently read on the reference (nothing is invented).
    """

    if not candidate_year:
        return None
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for line in reference_lines:
        confidence = float(line.get("confidence", 0.0))
        if confidence < 0.5:
            continue
        year, _ = normalize_year_token(str(line.get("text", "")))
        if year == candidate_year:
            box = line.get("box")
            if box and len(box) == 4:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if best is None or confidence > best[0]:
                    best = (confidence, tuple(float(value) for value in box))
    if best is None:
        return None
    return best[1], f"reference_ocr_conf{best[0]:.2f}"


def read_year_from_crop(
    crop_image: Any,
    recognizer,
    scale: int = 4,
    scratch_dir: str | Path | None = None,
    scratch_name: str = "crop",
) -> tuple[str | None, list[str], list[dict]]:
    """Upscale a projected year crop and OCR it with the frozen recognizer.

    The PaddleOCR pipeline accepts file paths, so the crop is written to a
    scratch PNG first (deterministic naming by ``scratch_name``).
    """

    resized = crop_image.resize((crop_image.width * scale, crop_image.height * scale))
    scratch_path: Path | None = None
    if scratch_dir:
        scratch_path = Path(scratch_dir)
        scratch_path.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_path / f"{scratch_name}_{scale}x.png"
        resized.save(scratch_path)
        lines = recognizer.predict_lines(scratch_path)
    else:
        # In-memory fallback via a temp file.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            resized.save(handle, format="PNG")
            temp_name = handle.name
        try:
            lines = recognizer.predict_lines(temp_name)
        finally:
            os.unlink(temp_name)
    years, corrections = extract_years_with_corrections(
        [{"text": line.text, "confidence": line.confidence} for line in lines]
    )
    decided = None
    if years:
        decided = max(
            years.items(),
            key=lambda item: max(observation["confidence"] for observation in item[1]),
        )[0]
    return decided, corrections, [{"text": line.text, "confidence": round(line.confidence, 4)} for line in lines]
