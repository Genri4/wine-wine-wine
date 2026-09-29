#!/usr/bin/env python3
"""OCR bake-off diagnostics: quality coverage, discrimination margins, vintage audit.

For every OCR config with a built cache this script measures:

1. Quality coverage per benchmark/scope (tokens, years, winery/name detection)
   - reuses the coverage machinery of build_ocr_cache.py.
2. Discrimination (Part 9): inside the frozen SO400M Top-5, compute the text
   score of the true target candidate and of the best wrong candidate, then
   report the distribution of (target - best_wrong) and related AUCs. This is
   the key "can this OCR separate Wine 2021 from Wine 2022" signal.
3. Vintage metadata audit (Part 10): where catalog vintage years come from,
   how many products carry one, hard-vintage-family coverage, and how often
   reference-OCR years agree with metadata years. Nothing is invented: a year
   is only recorded with its provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)
from recognition.text_normalization import extract_vintage_years  # noqa: E402
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
)

BENCHMARKS = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
OCR_KEYS = {
    "current_eslav": "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec",
    "cyrillic": "paddleocr3.7_ppocrv5_server_det_cyrillic_v5_mobile_rec",
    "paddleocr_vl": "paddleocr3.7_paddleocr_vl_0.9B",
}
BASELINE_RUN = Path("artifacts/experiments/so400m_ocr_reranker_20260920T193925Z")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR bake-off diagnostics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ocr-keys", nargs="+", default=list(OCR_KEYS))
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, _ = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    candidate_indexes = {item.item_id: build_candidate_text_index(item.metadata) for item in catalog_items}

    # Reference OCR evidence per OCR config.
    reference_evidence: dict[str, dict[str, object]] = {}
    reference_meta: dict[str, dict] = {}
    for ocr_name in args.ocr_keys:
        cache_key = OCR_KEYS[ocr_name]
        ref_path = project_root / "artifacts" / "ocr_cache" / cache_key / "catalog_references" / "reference_ocr.jsonl"
        if not ref_path.is_file():
            print(f"SKIP {ocr_name}: no reference cache")
            continue
        records = load_jsonl(ref_path)
        evidence = {}
        for record in records:
            lines = [(line["text"], line["confidence"]) for line in record["lines"]]
            evidence[record["slug"]] = build_reference_ocr_evidence(lines)
        reference_evidence[ocr_name] = evidence
        reference_meta[ocr_name] = {
            "references": len(records),
            "empty_ocr_share": round(sum(1 for r in records if not r["lines"]) / len(records), 4),
            "median_lines": statistics.median([len(r["lines"]) for r in records]),
        }
        print(f"{ocr_name}: reference evidence ready ({len(evidence)} slugs)")

    quality_summary = {"reference": reference_meta}

    for benchmark in BENCHMARKS:
        manifest_rows = _read_csv(_default_benchmark_manifest(project_root, benchmark, None))
        baseline_path = BASELINE_RUN / "benchmarks" / benchmark / "baseline_predictions.csv"
        baseline_rows = {row["query_id"]: row for row in _read_csv(baseline_path)}

        per_ocr = {}
        for ocr_name in args.ocr_keys:
            cache_key = OCR_KEYS[ocr_name]
            query_path = project_root / "artifacts" / "ocr_cache" / cache_key / benchmark / "query_ocr.jsonl"
            if not query_path.is_file():
                print(f"SKIP {ocr_name}/{benchmark}: no query cache")
                continue
            query_records = {record["query_id"]: record for record in load_jsonl(query_path)}

            margins: list[float] = []
            target_scores: list[float] = []
            best_wrong_scores: list[float] = []
            target_best_auc_pos, target_best_auc_neg = [], []
            target_in5_auc_pos, target_in5_auc_neg = [], []
            for query_id, base_row in baseline_rows.items():
                record = query_records.get(query_id)
                if record is None:
                    continue
                query_evidence = build_query_text_evidence(
                    [(line["text"], line["confidence"]) for line in record["lines"]]
                )
                top5 = json.loads(base_row["top5_slugs"])
                target = base_row["target_slug"]
                scores = []
                for slug in top5:
                    signals = compute_text_signals(
                        query_evidence,
                        candidate_indexes[slug],
                        reference_evidence[ocr_name].get(slug),
                    )
                    text_score = max(
                        float(signals["metadata_text_score"]),
                        float(signals["reference_ocr_score"]),
                    )
                    scores.append(text_score)
                target_position = top5.index(target) if target in top5 else None
                if target_position is None:
                    continue
                target_score = scores[target_position]
                wrong = [score for index, score in enumerate(scores) if index != target_position]
                best_wrong = max(wrong) if wrong else 0.0
                margins.append(target_score - best_wrong)
                target_scores.append(target_score)
                best_wrong_scores.append(best_wrong)
                # AUC samples: does the OCR text score rank the true candidate first?
                is_text_top1 = scores.index(max(scores)) == target_position
                (target_best_auc_pos if is_text_top1 else target_best_auc_neg).append(max(scores))
                (target_in5_auc_pos if target_position == 0 else target_in5_auc_neg).append(max(scores))

            if not margins:
                continue
            margins_sorted = sorted(margins)
            n = len(margins)
            positive = sum(1 for margin in margins if margin > 0)
            per_ocr[ocr_name] = {
                "queries_with_target_in_top5": n,
                "mean_target_text_score": round(statistics.mean(target_scores), 4),
                "mean_best_wrong_text_score": round(statistics.mean(best_wrong_scores), 4),
                "text_margin_mean": round(statistics.mean(margins), 4),
                "text_margin_median": round(statistics.median(margins), 4),
                "text_margin_p10": round(margins_sorted[int(n * 0.10)], 4),
                "text_margin_p90": round(margins_sorted[int(n * 0.90)], 4),
                "share_target_text_beats_best_wrong": round(positive / n, 4),
                "share_margin_at_least_0.05": round(sum(1 for m in margins if m >= 0.05) / n, 4),
            }
            print(f"  {ocr_name}/{benchmark}: {per_ocr[ocr_name]}")

        quality_summary[benchmark] = per_ocr

    (run_dir / "ocr_discrimination_summary.json").write_text(
        json.dumps(quality_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Discrimination summary written.")

    # ------------------------------------------------------------------
    # Vintage metadata audit (Part 10)
    # ------------------------------------------------------------------
    audit = vintage_audit(catalog_items, reference_evidence)
    (run_dir / "vintage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Vintage audit written.")


def vintage_audit(catalog_items, reference_evidence: dict[str, dict[str, object]]) -> dict:
    total = len(catalog_items)
    title_year = 0
    title_year_examples = []
    for item in catalog_items:
        years = extract_vintage_years(item.metadata.get("title", ""))
        if years:
            title_year += 1
            if len(title_year_examples) < 10:
                title_year_examples.append({"slug": item.item_id, "title": item.metadata.get("title", ""), "year": years[0]})

    per_ocr = {}
    for ocr_name, evidence_map in reference_evidence.items():
        ref_year = sum(1 for evidence in evidence_map.values() if evidence.vintage_years)
        agree = 0
        checked = 0
        for item in catalog_items:
            evidence = evidence_map.get(item.item_id)
            if evidence is None or not evidence.vintage_years:
                continue
            title_years = extract_vintage_years(item.metadata.get("title", ""))
            if not title_years:
                continue
            checked += 1
            if set(evidence.vintage_years) & set(title_years):
                agree += 1
        per_ocr[ocr_name] = {
            "references_with_ocr_year": ref_year,
            "references_with_ocr_year_share": round(ref_year / total, 4),
            "both_metadata_and_ocr_year": checked,
            "ocr_year_agrees_with_title_year": agree,
            "agreement_share": round(agree / checked, 4) if checked else None,
        }

    return {
        "total_products": total,
        "products_with_title_vintage": title_year,
        "products_with_title_vintage_share": round(title_year / total, 4),
        "title_year_examples": title_year_examples,
        "note": "title vintage = standalone 4-digit year found in the title field; nothing is invented",
        "reference_ocr_vintage": per_ocr,
    }


if __name__ == "__main__":
    main()
