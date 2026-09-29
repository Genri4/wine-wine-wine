#!/usr/bin/env python3
"""Vintage disambiguation milestone runner.

Phases:
1. candidate year evidence (provenance audit) -> vintage_metadata_audit.csv
2. vintage challenge slice v1 (hard_v2 + generated hard) -> challenge manifest
3. oracle ceiling
4. detector-box targeted OCR (1x/2x/4x) on challenge queries -> diagnostics
5. conservative vintage rerank evaluation + STOP/GO checkpoint
6. artifacts (config, summaries, per-query diagnostics)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.ocr_engine import OCR_CONFIGS, create_ocr_engine  # noqa: E402
from recognition.ocr_reranker import FusionConfig, rerank_one_query  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
)
from recognition.vintage_disambiguation import (  # noqa: E402
    build_vintage_challenge,
    candidate_year_evidence,
    conservative_vintage_rerank,
    decide_query_year,
    extract_years_with_corrections,
    oracle_ceiling,
    write_challenge_csv,
)
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
)
from scripts.run_so400m_ocr_reranker import hard_v2_family_classification  # noqa: E402

BENCHMARKS = ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
ESLAV_CACHE = OCR_CONFIGS["current_eslav"].cache_key
FROZEN_BLEND = FusionConfig(
    policy="reference_ocr_blend", alpha=0.30, vintage_bonus=0.05, vintage_penalty=0.05, min_text_margin=0.05
)
BASELINE_RUN = PROJECT_ROOT / "artifacts" / "experiments" / "so400m_ocr_reranker_20260920T193925Z"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vintage disambiguation milestone")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--skip-targeted", action="store_true", help="skip detector-box retry (metadata/oracle only)")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, _ = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    catalog_by_slug = {item.item_id: item for item in catalog_items}

    # ------------------------------------------------------------------
    # Phase 1: candidate year evidence with provenance (Part 4)
    # ------------------------------------------------------------------
    ref_ocr_lines: dict[str, list[dict]] = {}
    ref_records = load_jsonl(
        project_root / "artifacts" / "ocr_cache" / ESLAV_CACHE / "catalog_references" / "reference_ocr.jsonl"
    )
    for record in ref_records:
        ref_ocr_lines[record["slug"]] = record["lines"]

    year_evidence: dict[str, dict] = {}
    for item in catalog_items:
        year_evidence[item.item_id] = candidate_year_evidence(
            item.item_id, item.metadata, ref_ocr_lines.get(item.item_id)
        )
    audit_rows = [
        {
            "slug": slug,
            "year": evidence["year"] or "",
            "provenance": evidence["provenance"],
            "sources": json.dumps(evidence.get("sources", {}), ensure_ascii=False),
        }
        for slug, evidence in sorted(year_evidence.items())
    ]
    write_csv(run_dir / "vintage_metadata_audit.csv", ["slug", "year", "provenance", "sources"], audit_rows)
    known = sum(1 for row in audit_rows if row["year"])
    provenance_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        provenance_counts[row["provenance"]] += 1
    print(f"year evidence: {known}/{len(audit_rows)} known; {dict(provenance_counts)}", flush=True)

    candidate_years = {slug: evidence["year"] for slug, evidence in year_evidence.items()}
    family_by_slug = {item.item_id: item.metadata.get("family_key", "") for item in catalog_items}

    # Family maps per benchmark.
    hard_family_type = hard_v2_family_classification()
    manifests = {benchmark: _read_csv(_default_benchmark_manifest(project_root, benchmark, None)) for benchmark in BENCHMARKS}
    # hard_v2 membership comes from the frozen families.csv: it lists every
    # family member, including products that are not the target of any
    # manifest row. Catalog-side information, not ground truth.
    family_by_slug: dict[str, str] = {}
    with (project_root / "data" / "benchmarks" / "hard_near_duplicate_dev_v2" / "families.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            family_by_slug[row["slug"]] = row["family_id"]
    family_types_by_family: dict[str, str] = {}
    for benchmark, rows in manifests.items():
        family_column = "family_id" if benchmark == "hard_near_duplicate_dev_v2" else "target_family_id"
        for row in rows:
            family_id = row.get(family_column, "")
            if family_id:
                family_by_slug[row["target_slug"]] = family_id
        if benchmark == "hard_near_duplicate_dev_v2":
            family_types_by_family.update(hard_family_type)
        else:
            for row in rows:
                if row.get("target_family_id"):
                    family_types_by_family[row["target_family_id"]] = row.get("family_type", "")
    # hard_v2 families.csv gives family type per family id; generated family_type already in rows.

    # ------------------------------------------------------------------
    # Phase 2: challenge slices (Part 3)
    # ------------------------------------------------------------------
    challenge: dict[str, list[dict]] = {}
    for benchmark in BENCHMARKS:
        baseline_rows = {
            row["query_id"]: row
            for row in _read_csv(BASELINE_RUN / "benchmarks" / benchmark / "baseline_predictions.csv")
        }
        family_types_scoped = {
            family_id: (
                hard_family_type.get(family_id, "subtype_or_other")
                if benchmark == "hard_near_duplicate_dev_v2"
                else "vintage"
            )
            for family_id in {row.get("family_id" if benchmark == "hard_near_duplicate_dev_v2" else "target_family_id", "") for row in manifests[benchmark]}
        }
        # For generated, only vintage-type families enter the challenge.
        if benchmark == "generated_stress_dev_pilot32":
            family_types_scoped = {
                family_id: ("vintage" if row.get("family_type", "") == "vintage" else "subtype_or_other")
                for family_id, row in (
                    (row.get("target_family_id", ""), row)
                    for row in manifests[benchmark]
                )
            }
        challenge[benchmark] = build_vintage_challenge(
            benchmark, manifests[benchmark], baseline_rows, family_by_slug, family_types_scoped, candidate_years
        )
        write_challenge_csv(run_dir / f"vintage_challenge_{benchmark}.csv", challenge[benchmark])
        print(f"challenge[{benchmark}]: {len(challenge[benchmark])} queries", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: oracle ceiling (Part 5)
    # ------------------------------------------------------------------
    oracle: dict[str, dict] = {}
    for benchmark in BENCHMARKS:
        # member years: candidates' canonical years for disambiguation check
        for row in challenge[benchmark]:
            member_slugs = [row["target_slug"]] + [entry["slug"] for entry in row["same_family_competitors"]]
            row["member_years"] = {slug: candidate_years.get(slug) for slug in member_slugs}
        oracle[benchmark] = oracle_ceiling(challenge[benchmark])
        print(f"oracle[{benchmark}]: {oracle[benchmark]}", flush=True)
    (run_dir / "oracle_ceiling.json").write_text(
        json.dumps(oracle, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.skip_targeted:
        print("Targeted OCR skipped; metadata/oracle artifacts written.")
        return

    # ------------------------------------------------------------------
    # Phase 4: detector-box targeted OCR (Parts 6-9)
    # ------------------------------------------------------------------
    ocr = create_ocr_engine(OCR_CONFIGS["current_eslav"], device=args.device)
    manifest_by_query = {
        benchmark: {row["query_id"]: row for row in manifests[benchmark]} for benchmark in BENCHMARKS
    }
    targeted_rows: list[dict] = []
    for benchmark in BENCHMARKS:
        for challenge_row in challenge[benchmark]:
            query_id = challenge_row["query_id"]
            manifest_row = manifest_by_query[benchmark][query_id]
            image_path = _path(project_root, manifest_row["query_path"])
            started = time.time()
            full_lines = ocr.predict_lines(image_path)
            full_seconds = time.time() - started

            # Full-image year evidence (existing full OCR baseline)
            full_years, _ = extract_years_with_corrections(
                [{"text": line.text, "confidence": line.confidence} for line in full_lines]
            )
            candidate_years_top5 = {slug: candidate_years.get(slug) for slug in challenge_row["top5_slugs"]}
            full_decision = decide_query_year(full_years, candidate_years_top5)

            # Detector boxes from the same full-image detection pass.
            detection = ocr._pipeline.predict(str(image_path))
            boxes = []
            for page in detection:
                raw_boxes = page.get("rec_boxes", page.get("dt_polys", []))
                for box_index, box in enumerate(raw_boxes):
                    try:
                        if hasattr(box, "shape"):
                            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                        else:
                            points = [(float(point[0]), float(point[1])) for point in box]
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    except (TypeError, ValueError, IndexError):
                        continue
                    boxes.append((x1, y1, x2, y2))

            from PIL import Image

            with Image.open(image_path) as pil_image:
                pil_image.load()
                width, height = pil_image.size
                box_variants: dict[str, list] = {"1x": [], "2x": [], "4x": []}
                for box_index, (x1, y1, x2, y2) in enumerate(boxes):
                    pad_x = (x2 - x1) * 0.10
                    pad_y = (y2 - y1) * 0.40
                    crop_box = (
                        max(0, int(x1 - pad_x)),
                        max(0, int(y1 - pad_y)),
                        min(width, int(x2 + pad_x)),
                        min(height, int(y2 + pad_y)),
                    )
                    if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
                        continue
                    crop = pil_image.crop(crop_box)
                    for scale in (1, 2, 4):
                        resized = crop.resize(
                            (crop.width * scale, crop.height * scale), Image.LANCZOS
                        )
                        temp_path = run_dir / "crops" / f"{benchmark}" / f"{query_id}_box{box_index}_{scale}x.png"
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        resized.save(temp_path)
                        lines = ocr.predict_lines(temp_path)
                        years, corrections = extract_years_with_corrections(
                            [{"text": line.text, "confidence": line.confidence} for line in lines]
                        )
                        box_variants[f"{scale}x"].append(
                            {"box_index": box_index, "years": years, "corrections": corrections}
                        )

            variant_decisions = {}
            for variant, boxes_results in box_variants.items():
                merged_years: dict[str, list] = {}
                merged_corrections: list[str] = []
                for box_result in boxes_results:
                    for year, observations in box_result["years"].items():
                        merged_years.setdefault(year, []).extend(observations)
                    merged_corrections.extend(box_result["corrections"])
                decision = decide_query_year(merged_years, candidate_years_top5)
                decision.years = merged_years
                decision.corrections = merged_corrections
                variant_decisions[variant] = decision.to_dict()

            # Full-image decision as dict for storage
            targeted_rows.append(
                {
                    "benchmark": benchmark,
                    "query_id": query_id,
                    "target_slug": challenge_row["target_slug"],
                    "target_year": challenge_row["target_year"],
                    "full_image_decision": json.dumps(full_decision.to_dict(), ensure_ascii=False),
                    "variants": json.dumps(variant_decisions, ensure_ascii=False),
                    "full_ocr_seconds": round(full_seconds, 3),
                    "box_count": len(boxes),
                }
            )
            if len(targeted_rows) % 20 == 0:
                print(f"  targeted processed {len(targeted_rows)} queries", flush=True)

    write_csv(
        run_dir / "targeted_ocr_per_query.csv",
        ["benchmark", "query_id", "target_slug", "target_year", "full_image_decision", "variants", "full_ocr_seconds", "box_count"],
        targeted_rows,
    )

    # Diagnostics (Part 9): correct-year rate per method per benchmark
    diagnostics = targeted_diagnostics(targeted_rows)
    (run_dir / "targeted_ocr_summary.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=1), flush=True)

    # ------------------------------------------------------------------
    # Phase 5: conservative vintage rerank on challenge slices (Parts 10-13)
    # ------------------------------------------------------------------
    evaluation_rows: list[dict] = []
    for benchmark in BENCHMARKS:
        for challenge_row, targeted_row in zip(
            challenge[benchmark],
            [row for row in targeted_rows if row["benchmark"] == benchmark],
        ):
            query_id = challenge_row["query_id"]
            base_row = (
                _read_csv(BASELINE_RUN / "benchmarks" / benchmark / "baseline_predictions.csv")
                if False
                else None
            )
            # Use the manifest order stored earlier
            base = _baseline_row_cache(benchmark, BASELINE_RUN)[query_id]
            top5 = json.loads(base["top5_slugs"])
            image_scores = json.loads(base["top5_scores"])

            # Method C: current production reranker (frozen blend) baseline order
            record = query_ocr_record(ESLAV_CACHE, benchmark, query_id)
            query_evidence = build_query_text_evidence(
                [(line["text"], line["confidence"]) for line in record["lines"]]
            )
            candidate_indexes = {slug: build_candidate_text_index(catalog_by_slug[slug].metadata) for slug in top5}
            candidate_signals = [
                compute_text_signals_cached(
                    query_evidence, candidate_indexes[slug], ref_ocr_lines.get(slug)
                )
                for slug in top5
            ]
            current_outcome = rerank_one_query(image_scores, candidate_signals, FROZEN_BLEND)
            current_top1 = top5[current_outcome.final_order[0]]

            variants = json.loads(targeted_row["variants"])
            best_variant = max(
                variants.items(),
                key=lambda item: (
                    item[1]["decision"] in ("unique", "family_consistent"),
                    item[1]["confidence"],
                ),
            )
            query_year = best_variant[1]["decided_year"]
            query_confidence = best_variant[1]["confidence"]
            outcome = conservative_vintage_rerank(
                top5, image_scores, family_by_slug, candidate_years, query_year, query_confidence
            )
            final_top1 = top5[outcome["order"][0]]
            evaluation_rows.append(
                {
                    "benchmark": benchmark,
                    "query_id": query_id,
                    "target_slug": challenge_row["target_slug"],
                    "target_year": challenge_row["target_year"],
                    "baseline_top1_correct": challenge_row["baseline_correct_top1"],
                    "current_rerank_top1": current_top1,
                    "current_rerank_correct": current_top1 == challenge_row["target_slug"],
                    "targeted_variant": best_variant[0],
                    "query_year": query_year or "",
                    "query_year_decision": best_variant[1]["decision"],
                    "query_year_confidence": query_confidence,
                    "vintage_action": outcome["action"],
                    "vintage_reason": outcome["reason"],
                    "final_top1": final_top1,
                    "final_correct": final_top1 == challenge_row["target_slug"],
                }
            )
    write_csv(
        run_dir / "vintage_rerank_per_query.csv",
        list(evaluation_rows[0]),
        evaluation_rows,
    )

    # ------------------------------------------------------------------
    # Phase 6: reference-guided year crops (Parts 14-21) via SIFT
    # ------------------------------------------------------------------
    import numpy as np
    from PIL import Image

    from recognition.reference_guided_year import (
        compute_homography,
        crop_bounds_valid,
        projected_crop_bounds,
        project_bbox,
        read_year_from_crop,
        reference_year_bbox,
    )

    recognizer = ocr  # same frozen eslav engine
    rg_rows: list[dict] = []
    alignment_stats: dict[str, list] = defaultdict(lambda: {"success": 0, "total": 0, "good": [], "inliers": [], "ratio": []})
    for benchmark in BENCHMARKS:
        for challenge_row in challenge[benchmark]:
            query_id = challenge_row["query_id"]
            manifest_row = manifest_by_query[benchmark][query_id]
            query_path = _path(project_root, manifest_row["query_path"])
            with Image.open(query_path) as query_pil:
                query_pil.load()
                query_rgb = np.asarray(query_pil.convert("RGB"))
                query_width, query_height = query_pil.size

            candidate_evidence = {}
            for entry in challenge_row["same_family_competitors"] + [
                {"slug": challenge_row["target_slug"], "year": challenge_row["target_year"]}
            ]:
                slug = entry["slug"]
                candidate_year = candidate_years.get(slug)
                reference_lines = ref_ocr_lines.get(slug)
                year_box = reference_year_bbox(reference_lines or [], candidate_year)
                if year_box is None:
                    candidate_evidence[slug] = {
                        "year": candidate_year,
                        "year_box": None,
                        "reason": "no_reference_year_box",
                        "ocr_year": None,
                    }
                    continue
                bbox, provenance = year_box
                reference_path = _path(project_root, catalog_by_slug[slug].image_path)
                with Image.open(reference_path) as reference_pil:
                    reference_pil.load()
                    reference_rgb = np.asarray(reference_pil.convert("RGB"))
                alignment = compute_homography(reference_rgb, query_rgb)
                stats = alignment_stats[benchmark]
                stats["total"] += 1
                stats["good"].append(alignment.good_matches)
                stats["inliers"].append(alignment.inliers)
                stats["ratio"].append(round(alignment.inlier_ratio, 3))
                if not alignment.valid:
                    candidate_evidence[slug] = {
                        "year": candidate_year,
                        "year_box": list(bbox),
                        "reason": f"alignment_invalid:{alignment.reason}",
                        "ocr_year": None,
                    }
                    continue
                stats["success"] += 1
                projected = project_bbox(bbox, alignment.homography)
                if projected is None:
                    candidate_evidence[slug] = {"year": candidate_year, "year_box": list(bbox), "reason": "projection_failed", "ocr_year": None}
                    continue
                bounds = projected_crop_bounds(projected, query_width, query_height)
                if bounds is None or not crop_bounds_valid(bounds, bbox):
                    candidate_evidence[slug] = {"year": candidate_year, "year_box": list(bbox), "reason": "invalid_projected_crop", "ocr_year": None}
                    continue
                pad_x = (bounds[2] - bounds[0]) * 0.25
                pad_y = (bounds[3] - bounds[1]) * 0.25
                crop_box = (
                    max(0, int(bounds[0] - pad_x)),
                    max(0, int(bounds[1] - pad_y)),
                    min(query_width, int(bounds[2] + pad_x)),
                    min(query_height, int(bounds[3] + pad_y)),
                )
                crop = query_pil.crop(crop_box)
                ocr_year, corrections, raw_lines = read_year_from_crop(
                    crop,
                    recognizer,
                    scratch_dir=run_dir / "crops" / benchmark,
                    scratch_name=f"{query_id}_{slug[:40]}",
                )
                candidate_evidence[slug] = {
                    "year": candidate_year,
                    "year_box": list(bbox),
                    "provenance": provenance,
                    "alignment": {"good": alignment.good_matches, "inliers": alignment.inliers, "ratio": round(alignment.inlier_ratio, 3)},
                    "crop_box": list(crop_box),
                    "ocr_year": ocr_year,
                    "corrections": corrections,
                    "raw_lines": raw_lines[:6],
                    "match": bool(ocr_year and candidate_year and ocr_year == candidate_year),
                }
            rg_rows.append(
                {
                    "benchmark": benchmark,
                    "query_id": query_id,
                    "target_slug": challenge_row["target_slug"],
                    "target_year": challenge_row["target_year"],
                    "candidate_evidence": json.dumps(candidate_evidence, ensure_ascii=False),
                }
            )
            if len(rg_rows) % 20 == 0:
                print(f"  reference-guided processed {len(rg_rows)} queries", flush=True)

    write_csv(
        run_dir / "reference_guided_per_query.csv",
        ["benchmark", "query_id", "target_slug", "target_year", "candidate_evidence"],
        rg_rows,
    )
    alignment_summary = {
        benchmark: {
            "total_alignments": stats["total"],
            "success": stats["success"],
            "success_rate": round(stats["success"] / stats["total"], 4) if stats["total"] else None,
            "median_good_matches": statistics.median(stats["good"]) if stats["good"] else None,
            "median_inliers": statistics.median(stats["inliers"]) if stats["inliers"] else None,
            "median_inlier_ratio": statistics.median(stats["ratio"]) if stats["ratio"] else None,
        }
        for benchmark, stats in alignment_stats.items()
    }
    (run_dir / "alignment_summary.json").write_text(
        json.dumps(alignment_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("alignment summary:", json.dumps(alignment_summary), flush=True)

    # ------------------------------------------------------------------
    # Phase 7: reference-guided rerank (method G)
    # ------------------------------------------------------------------
    rg_by_query = {row["query_id"]: row for row in rg_rows}
    for row in evaluation_rows:
        rg_row = rg_by_query.get(row["query_id"])
        row["reference_guided_available"] = False
        row["rg_final_top1"] = ""
        row["rg_final_correct"] = ""
        if not rg_row:
            continue
        evidence = json.loads(rg_row["candidate_evidence"])
        # Candidate-specific year evidence per Top-5 member
        base = _baseline_row_cache(row["benchmark"], BASELINE_RUN)[row["query_id"]]
        top5 = json.loads(base["top5_slugs"])
        image_scores = json.loads(base["top5_scores"])
        candidate_years_top5 = {slug: candidate_years.get(slug) for slug in top5}
        # Candidate-specific projected-crop evidence per same-family member.
        targeted_row = next(
            targeted_row for targeted_row in targeted_rows if targeted_row["query_id"] == row["query_id"]
        )
        swap_slug = None
        top1_slug = top5[0]
        top1_family = family_by_slug.get(top1_slug)
        top1_year = candidate_years.get(top1_slug)
        for slug in top5[1:]:
            if family_by_slug.get(slug) != top1_family:
                continue
            candidate_year = candidate_years.get(slug)
            evidence_slug = evidence.get(slug, {})
            if (
                candidate_year
                and top1_year
                and candidate_year != top1_year
                and evidence_slug.get("ocr_year")
                and evidence_slug["ocr_year"] == candidate_year
                and evidence_slug.get("match") is not None
            ):
                # Candidate's own projected crop supports its year. The top1
                # member's projected crop may agree (its year read back) or
                # read a different year; if it reads the SAME year as the
                # swap candidate, that is positive evidence the bottle in
                # the query carries that year and the top1 product's year
                # does not match it.
                top1_evidence = evidence.get(top1_slug, {})
                if top1_evidence.get("ocr_year") == candidate_year:
                    swap_slug = slug
                    break
                if top1_evidence.get("ocr_year") and top1_evidence["ocr_year"] == top1_year:
                    continue  # top1 crop confirms its own year: no swap
                # top1 crop read a third/unknown year: weak conflict, stay
                # conservative.
                swap_slug = slug
                break
        row["reference_guided_available"] = swap_slug is not None
        if swap_slug:
            new_order = [top5.index(swap_slug), 0] + [
                index for index in range(1, len(top5)) if index != top5.index(swap_slug)
            ]
            final_top1 = top5[new_order[0]]
            row["rg_final_top1"] = final_top1
            row["rg_final_correct"] = str(bool(final_top1 == row["target_slug"]))
            row["rg_action"] = "family_swap"
        else:
            row["rg_action"] = "no_action"
    write_csv(
        run_dir / "reference_guided_rerank_per_query.csv",
        list(evaluation_rows[0]),
        evaluation_rows,
    )
    rg_summary: dict[str, dict] = {}
    for benchmark in BENCHMARKS:
        scoped = [row for row in evaluation_rows if row["benchmark"] == benchmark]
        rg_summary[benchmark] = {
            "challenge_queries": len(scoped),
            "current_pipeline_correct": sum(1 for row in scoped if row["current_rerank_correct"]),
            "rg_correct": sum(1 for row in scoped if row["rg_final_correct"] == "True"),
            "rg_swaps": sum(1 for row in scoped if row.get("rg_action") == "family_swap"),
            "rg_rescued_vs_current": sum(
                1 for row in scoped if not row["current_rerank_correct"] and row["rg_final_correct"] == "True"
            ),
            "rg_broken_vs_current": sum(
                1 for row in scoped if row["current_rerank_correct"] and row["rg_final_correct"] == "False"
            ),
        }
        print(f"reference_guided[{benchmark}]: {rg_summary[benchmark]}", flush=True)
    (run_dir / "reference_guided_summary.json").write_text(
        json.dumps(rg_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary: dict[str, dict] = {}
    for benchmark in BENCHMARKS:
        scoped = [row for row in evaluation_rows if row["benchmark"] == benchmark]
        n = len(scoped)
        baseline_correct = sum(1 for row in scoped if row["baseline_top1_correct"])
        current_correct = sum(1 for row in scoped if row["current_rerank_correct"])
        final_correct = sum(1 for row in scoped if row["final_correct"])
        year_decided = sum(1 for row in scoped if row["query_year"])
        swaps = sum(1 for row in scoped if row["vintage_action"] == "family_swap")
        summary[benchmark] = {
            "challenge_queries": n,
            "image_only_correct": baseline_correct,
            "current_pipeline_correct": current_correct,
            "vintage_stage_correct": final_correct,
            "year_decided": year_decided,
            "family_swaps": swaps,
            "rescued_vs_current": sum(
                1
                for row in scoped
                if not row["current_rerank_correct"] and row["final_correct"]
            ),
            "broken_vs_current": sum(
                1
                for row in scoped
                if row["current_rerank_correct"] and not row["final_correct"]
            ),
        }
        print(f"vintage_stage[{benchmark}]: {summary[benchmark]}", flush=True)
    (run_dir / "vintage_stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


_BASELINE_ROW_CACHE: dict[str, dict[str, dict]] = {}
_SIGNAL_CACHE: dict[tuple[str, str], dict[str, list[dict[str, float | str]]]] = {}
_QUERY_OCR_CACHE: dict[tuple[str, str], dict[str, dict]] = {}


def _baseline_row_cache(benchmark: str, baseline_run: Path) -> dict[str, dict]:
    if benchmark not in _BASELINE_ROW_CACHE:
        _BASELINE_ROW_CACHE[benchmark] = {
            row["query_id"]: row
            for row in _read_csv(baseline_run / "benchmarks" / benchmark / "baseline_predictions.csv")
        }
    return _BASELINE_ROW_CACHE[benchmark]


def query_ocr_record(cache_key: str, benchmark: str, query_id: str) -> dict:
    key = (cache_key, benchmark)
    if key not in _QUERY_OCR_CACHE:
        _QUERY_OCR_CACHE[key] = {
            record["query_id"]: record
            for record in load_jsonl(
                PROJECT_ROOT / "artifacts" / "ocr_cache" / cache_key / benchmark / "query_ocr.jsonl"
            )
        }
    return _QUERY_OCR_CACHE[key][query_id]


def compute_text_signals_cached(query_evidence, candidate_index, reference_lines):
    from recognition.text_signals import compute_text_signals as _compute

    reference_evidence = (
        build_reference_ocr_evidence([(line["text"], line["confidence"]) for line in reference_lines])
        if reference_lines
        else None
    )
    return _compute(query_evidence, candidate_index, reference_evidence)


def targeted_diagnostics(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    methods = ("full_image", "1x", "2x", "4x")
    for benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
        scoped = [row for row in rows if row["benchmark"] == benchmark]
        per_method: dict[str, dict] = {}
        for method in methods:
            correct = wrong = no_year = ambiguous = 0
            for row in scoped:
                decision = (
                    json.loads(row["full_image_decision"])
                    if method == "full_image"
                    else json.loads(row["variants"])[method]
                )
                decided = decision.get("decided_year")
                if not decided:
                    if decision.get("decision") == "ambiguous":
                        ambiguous += 1
                    else:
                        no_year += 1
                elif decided == row["target_year"]:
                    correct += 1
                else:
                    wrong += 1
            total = len(scoped)
            per_method[method] = {
                "queries": total,
                "correct_year": correct,
                "correct_year_rate": round(correct / total, 4) if total else None,
                "wrong_year": wrong,
                "ambiguous": ambiguous,
                "no_year": no_year,
            }
        result[benchmark] = per_method
    return result


if __name__ == "__main__":
    main()
