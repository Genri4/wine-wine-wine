#!/usr/bin/env python3
"""Score the new 100-image set as an open-set retrieval evaluation.

The input review CSV is exported by slug_candidate_review.html. Only human
confirmed catalog matches and no-catalog-match decisions are scored; pending
and uncertain examples are reported separately. The primary metric is a
threshold-free open-set retrieval AUC over max-cosine confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/experiments/new_data_slug_audit_20260924"
PREDICTIONS_PATH = RUN_DIR / "slug_candidates.csv"
OUT_DIR = RUN_DIR / "open_set_metric"
MODELS = {
    "frozen_so400m": ("frozen_top5_json", "frozen_top1_slug_candidate"),
    "selected_r16": ("r16_top5_json", "r16_top1_slug_candidate"),
}
REVIEWED_STATUSES = {"catalog_match", "no_catalog_match"}
ALL_STATUSES = REVIEWED_STATUSES | {"uncertain", "unreviewed"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unique_by_image(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image = row.get("image", "").strip()
        if not image:
            raise ValueError(f"{label} contains an empty image path")
        if image in indexed:
            raise ValueError(f"{label} contains duplicate image path: {image}")
        indexed[image] = row
    return indexed


def osr_curve(known: list[dict[str, Any]], unknown: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]]]:
    """Integrate CCR over unknown false-accept rate, grouped by tied scores.

    Confidence is maximum cosine similarity over the catalog (the Top-1
    score). CCR counts known examples whose Top-1 slug is exactly correct and
    whose score reaches the threshold; FPR counts out-of-catalog examples
    whose score reaches it. The curve is threshold-independent.
    """
    if not known or not unknown:
        return None, []

    groups: dict[float, dict[str, int]] = {}
    for row in known:
        score = float(row["top1_similarity"])
        group = groups.setdefault(score, {"known_correct": 0, "unknown_accepted": 0})
        if row["top1_exact"]:
            group["known_correct"] += 1
    for row in unknown:
        score = float(row["top1_similarity"])
        group = groups.setdefault(score, {"known_correct": 0, "unknown_accepted": 0})
        group["unknown_accepted"] += 1

    known_count, unknown_count = len(known), len(unknown)
    accepted_known_correct = 0
    accepted_unknown = 0
    points: list[dict[str, Any]] = [{"threshold": "above_max", "unknown_false_accept_rate": 0.0, "known_correct_classification_rate": 0.0}]
    for score in sorted(groups, reverse=True):
        accepted_known_correct += groups[score]["known_correct"]
        accepted_unknown += groups[score]["unknown_accepted"]
        points.append({
            "threshold": score,
            "unknown_false_accept_rate": accepted_unknown / unknown_count,
            "known_correct_classification_rate": accepted_known_correct / known_count,
        })

    area = 0.0
    previous_x = previous_y = 0.0
    for point in points[1:]:
        x = float(point["unknown_false_accept_rate"])
        y = float(point["known_correct_classification_rate"])
        area += (x - previous_x) * (y + previous_y) / 2.0
        previous_x, previous_y = x, y
    return area, points


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-set retrieval on the new 100-image dataset")
    parser.add_argument("--review-csv", required=True, help="CSV exported from slug_candidate_review.html")
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH), help="Frozen/R16 candidate predictions CSV")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Directory for metrics and per-image records")
    args = parser.parse_args()

    prediction_rows = read_csv(Path(args.predictions))
    review_rows = read_csv(Path(args.review_csv))
    prediction_by_image = unique_by_image(prediction_rows, "predictions")
    review_by_image = unique_by_image(review_rows, "review CSV")
    missing = sorted(set(prediction_by_image) - set(review_by_image))
    extra = sorted(set(review_by_image) - set(prediction_by_image))
    if missing or extra:
        raise ValueError(f"Review/prediction image mismatch: missing={len(missing)}, extra={len(extra)}")

    slugs = _catalog_slugs()
    labels_by_image: dict[str, dict[str, str]] = {}
    unresolved_counts = {"unreviewed": 0, "uncertain": 0}
    for image, review in review_by_image.items():
        status = review.get("review_status", "").strip()
        target_slug = review.get("verified_slug", "").strip()
        if status not in ALL_STATUSES:
            raise ValueError(f"Invalid review_status for {image}: {status!r}")
        prediction = prediction_by_image[image]
        if review.get("sha256") and prediction.get("sha256") and review["sha256"] != prediction["sha256"]:
            raise ValueError(f"Image checksum changed for {image}")
        if status == "catalog_match":
            if not target_slug:
                raise ValueError(f"catalog_match needs verified_slug: {image}")
            if target_slug not in slugs:
                raise ValueError(f"verified_slug is not in the benchmark catalog: {target_slug}")
        elif status == "no_catalog_match" and target_slug:
            raise ValueError(f"no_catalog_match must not have verified_slug: {image}")
        elif status in {"unreviewed", "uncertain"}:
            unresolved_counts[status] += 1
        labels_by_image[image] = {"status": status, "target_slug": target_slug}

    resolved_count = sum(label["status"] in REVIEWED_STATUSES for label in labels_by_image.values())
    known_count = sum(label["status"] == "catalog_match" for label in labels_by_image.values())
    unknown_count = sum(label["status"] == "no_catalog_match" for label in labels_by_image.values())
    completeness = (
        "complete"
        if resolved_count == len(prediction_by_image) and known_count > 0 and unknown_count > 0
        else "provisional"
    )

    per_image_rows: list[dict[str, Any]] = []
    model_metrics: dict[str, Any] = {}
    curves: list[dict[str, Any]] = []
    for model_name, (top5_field, top1_field) in MODELS.items():
        known: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        for image in sorted(prediction_by_image):
            prediction = prediction_by_image[image]
            label = labels_by_image[image]
            try:
                candidates = json.loads(prediction[top5_field])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid {top5_field} for {image}") from exc
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"No ranked candidates for {image} ({model_name})")
            top1_slug = prediction[top1_field].strip()
            top1_similarity = float(candidates[0]["similarity"])
            target_slug = label["target_slug"]
            exact = label["status"] == "catalog_match" and top1_slug == target_slug
            recall5 = label["status"] == "catalog_match" and target_slug in {str(item["slug"]) for item in candidates[:5]}
            result = {
                "image": image,
                "model": model_name,
                "review_status": label["status"],
                "target_slug": target_slug,
                "top1_slug": top1_slug,
                "top1_similarity": top1_similarity,
                "top1_exact": exact if label["status"] == "catalog_match" else "",
                "recall_at_5": recall5 if label["status"] == "catalog_match" else "",
            }
            per_image_rows.append(result)
            if label["status"] == "catalog_match":
                known.append(result)
            elif label["status"] == "no_catalog_match":
                unknown.append(result)

        known_top1 = sum(bool(row["top1_exact"]) for row in known) / len(known) if known else None
        known_recall5 = sum(bool(row["recall_at_5"]) for row in known) / len(known) if known else None
        auc, points = osr_curve(known, unknown)
        model_metrics[model_name] = {
            "known_top1_accuracy": known_top1,
            "known_recall_at_5": known_recall5,
            "open_set_retrieval_auc": auc,
            "osr_curve_points": len(points),
        }
        curves.extend({"model": model_name, **point} for point in points)

    payload = {
        "benchmark": "new_data_open_set_v1",
        "metric_version": 1,
        "label_completeness": completeness,
        "query_count": len(prediction_by_image),
        "resolved_count": resolved_count,
        "known_in_catalog_count": known_count,
        "out_of_catalog_count": unknown_count,
        "unresolved_counts": unresolved_counts,
        "metric_definition": "Area under the curve of known exact classification rate vs unknown false-accept rate, thresholding each model's max cosine similarity.",
        "scores": model_metrics,
        "note": "Image-only frozen SO400M/R16 candidate scores; this does not measure the full OCR/SIFT production pipeline.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "per_image.csv", ["image", "model", "review_status", "target_slug", "top1_slug", "top1_similarity", "top1_exact", "recall_at_5"], per_image_rows)
    write_csv(output_dir / "osr_curve.csv", ["model", "threshold", "unknown_false_accept_rate", "known_correct_classification_rate"], curves)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _catalog_slugs() -> set[str]:
    path = ROOT / "data/processed/catalog_manifest.csv"
    return {row["slug"].strip() for row in read_csv(path) if row.get("slug", "").strip() and row.get("reference_image_path", "").strip()}


if __name__ == "__main__":
    main()
