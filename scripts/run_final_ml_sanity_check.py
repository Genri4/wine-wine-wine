#!/usr/bin/env python3
"""Final error audit and fixed-grid SIFT reranking check over frozen Top-5s."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.geometric_reranker import (  # noqa: E402
    DEFAULT_CONFIG,
    GeometryScore,
    SIFTFeatures,
    cache_filename,
    conservative_geometry_order,
    extract_sift,
    file_sha256,
    fuse_scores,
    load_feature_cache,
    match_sift_pair,
    normalize_geometry_scores,
    rank_candidates,
    read_rgb,
    save_feature_cache,
    sift_cache_fingerprint,
    transition_counts,
    valid_geometry_scores,
    validate_top5_candidates,
)
from recognition.ocr_reranker import FusionConfig, rerank_one_query  # noqa: E402


SOURCE_RUN = Path("artifacts/experiments/so400m_ocr_reranker_20260920T193925Z")
BENCHMARKS = ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
GRID_WEIGHTS = (0.10, 0.20, 0.30, 0.40)
CURRENT_POLICY = {
    "policy": "reference_ocr_blend",
    "alpha": 0.30,
    "vintage_bonus": 0.05,
    "vintage_penalty": 0.05,
    "min_text_margin": 0.05,
}
ERROR_LABELS = {
    "A": "Correct family, wrong vintage",
    "B": "Correct family, wrong subtype / grape / subline",
    "C": "Visually near-identical packaging",
    "D": "Target is in Top-5, wrong Top-1",
    "E": "Target outside Top-5 (retrieval failure)",
    "F": "Capture/domain-shift evidence",
    "G": "OCR/text evidence misleading",
    "H": "Unclear / mixed",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_list(value: str) -> list:
    return json.loads(value) if value else []


def number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def query_cache(path: Path, key_field: str) -> dict[str, dict]:
    cache = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cache[str(row[key_field])] = row
    return cache


def verify_frozen_current_pipeline(root: Path) -> dict:
    """Reproduce the frozen OCR reranker order from its saved Top-5 signals."""
    source = root / SOURCE_RUN
    selected = read_json(source / "selected_reranker.json")["config"]
    config = FusionConfig(**selected)
    all_results = {}
    for benchmark in BENCHMARKS:
        current = {row["query_id"]: row for row in read_csv(source / "benchmarks" / benchmark / "reranked_predictions.csv")}
        grouped = defaultdict(list)
        for row in read_csv(source / "rerank_candidates.csv"):
            if row["benchmark"] == benchmark:
                grouped[row["query_id"]].append(row)
        exact = 0
        for query_id, prediction in current.items():
            signals = sorted(grouped[query_id], key=lambda row: integer(row["candidate_position"]))
            if len(signals) != 5:
                raise AssertionError(f"Expected 5 saved reranker signals for {query_id}, found {len(signals)}")
            scores = [number(row["image_score"]) for row in signals]
            evidence = [{
                "metadata_text_score": number(row["metadata_text_score"]),
                "reference_ocr_score": number(row["reference_ocr_score"]),
                "vintage_match": row["vintage_match"],
            } for row in signals]
            outcome = rerank_one_query(scores, evidence, config)
            reproduced = [signals[index]["candidate_slug"] for index in outcome.final_order]
            frozen = as_list(prediction["top5_slugs"])
            exact += reproduced == frozen
        all_results[benchmark] = {"queries": len(current), "exact_top5_order": exact,
                                  "agreement": exact / max(len(current), 1), "reproduced": exact == len(current)}
        if exact != len(current):
            raise AssertionError(f"Current final pipeline did not reproduce exactly for {benchmark}: {exact}/{len(current)}")
    return {"source_run": str(SOURCE_RUN), "selected_reranker_config": selected, "benchmarks": all_results,
            "candidate_set_unchanged_between_image_and_current": True}


def years(text: str) -> list[str]:
    return sorted(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text or "")))


def image_path(root: Path, relative: str) -> Path:
    return root / relative


def metadata_and_families(root: Path) -> tuple[dict[str, dict], dict[str, str], dict[tuple[str, str], dict], dict[str, str]]:
    products = {row["slug"]: row for row in read_csv(root / "data/processed/catalog_manifest.csv")}
    family_by_slug: dict[str, str] = {}
    pair_evidence: dict[tuple[str, str], dict] = {}
    for row in read_csv(root / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv"):
        slug = row["slug"]
        family_by_slug[slug] = row["family_id"]
        try:
            evidence = json.loads(row.get("selection_evidence_json") or "[]")
        except json.JSONDecodeError:
            evidence = []
        for pair in evidence:
            left, right = pair.get("left_slug"), pair.get("right_slug")
            if left and right:
                pair_evidence[tuple(sorted((left, right)))] = pair
    pilot_family_type: dict[str, str] = {}
    for row in read_csv(root / "data/benchmarks/generated_stress_dev_pilot32/hard_families.csv"):
        try:
            members = json.loads(row["member_slugs"])
        except (json.JSONDecodeError, KeyError):
            continue
        for slug in members:
            family_by_slug[slug] = row["pilot_family_id"]
            pilot_family_type[slug] = row["family_type"]
        try:
            evidence = json.loads(row.get("source_selection_evidence_json") or "[]")
        except json.JSONDecodeError:
            evidence = []
        for pair in evidence:
            left, right = pair.get("left_slug"), pair.get("right_slug")
            if left and right:
                pair_evidence[tuple(sorted((left, right)))] = pair
    family_type_by_slug = {}
    hard_families = defaultdict(list)
    for row in read_csv(root / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv"):
        hard_families[row["family_id"]].append(row)
    for family_id, members in hard_families.items():
        distinct_years = {year for row in members for year in years(" ".join((row.get("product_name", ""), row.get("year_if_known", ""))))}
        grapes = {row.get("grape", "").strip().lower() for row in members if row.get("grape", "").strip()}
        if len(distinct_years) >= 2:
            family_type = "vintage"
        elif len(grapes) >= 2:
            family_type = "subtype"
        else:
            family_type = "other"
        for row in members:
            family_type_by_slug[row["slug"]] = family_type
    for row in read_csv(root / "data/benchmarks/generated_stress_dev_pilot32/hard_families.csv"):
        for slug in as_list(row.get("member_slugs", "")):
            family_type_by_slug[slug] = row["family_type"]
    return products, family_by_slug, pair_evidence, family_type_by_slug


def get_family_tag(target: str, predicted: str, products: dict, family_by_slug: dict,
                   family_type_by_slug: dict | None = None) -> tuple[str | None, bool]:
    if not family_by_slug.get(target) or family_by_slug.get(target) != family_by_slug.get(predicted):
        return None, False
    target_meta, predicted_meta = products.get(target, {}), products.get(predicted, {})
    target_years = years(target_meta.get("title", ""))
    predicted_years = years(predicted_meta.get("title", ""))
    if target_years and predicted_years and set(target_years) != set(predicted_years):
        return "A", True
    if family_type_by_slug and family_type_by_slug.get(target) == "vintage":
        return "A", True
    if family_type_by_slug and family_type_by_slug.get(target) == "subtype":
        return "B", True
    if any(target_meta.get(key, "") != predicted_meta.get(key, "")
           for key in ("grape", "category", "color")):
        return "B", True
    return None, True


def make_error_audit(root: Path, out_dir: Path) -> tuple[list[dict], dict, dict]:
    products, family_by_slug, pair_evidence, family_type_by_slug = metadata_and_families(root)

    errors: list[dict] = []
    audit_inputs = {}
    for benchmark in BENCHMARKS:
        directory = root / SOURCE_RUN / "benchmarks" / benchmark
        baseline_rows = {row["query_id"]: row for row in read_csv(directory / "baseline_predictions.csv")}
        current_rows = {row["query_id"]: row for row in read_csv(directory / "reranked_predictions.csv")}
        manifests = {row["query_id"]: row for row in read_csv(root / "data/benchmarks" / benchmark / "manifest.csv")}
        signal_rows = defaultdict(dict)
        for row in read_csv(root / SOURCE_RUN / "rerank_candidates.csv"):
            if row["benchmark"] == benchmark:
                signal_rows[row["query_id"]][row["candidate_slug"]] = row
        query_ocr = query_cache(root / SOURCE_RUN / "ocr" / f"query_ocr_{benchmark}.jsonl", "query_id")
        audit_inputs[benchmark] = (baseline_rows, current_rows, manifests, signal_rows, query_ocr)
        for query_id, current in current_rows.items():
            if current["correct_top1"].lower() == "true":
                continue
            baseline = baseline_rows[query_id]
            manifest = manifests[query_id]
            target, predicted = current["target_slug"], current["predicted_slug"]
            target_rank = integer(current.get("target_rank"), 999999)
            top5 = as_list(current["top5_slugs"])
            target_in_top5 = target in top5
            tags = ["D" if target_in_top5 else "E"]
            family_tag, same_family = get_family_tag(target, predicted, products, family_by_slug, family_type_by_slug)
            if family_tag:
                tags.append(family_tag)
            pair = pair_evidence.get(tuple(sorted((target, predicted))), {})
            if same_family and bool(pair.get("image_similarity")):
                tags.append("C")

            # Scenario provenance comes from the accepted benchmark manifest;
            # the displayed query/reference pair is reviewed in the gallery.
            scenario = manifest.get("scenario_id", "")
            domain_evidence = False
            if benchmark == "generated_stress_dev_pilot32" and scenario in {
                "distance_crop", "glare_bad_light", "handheld", "slight_angle"
            }:
                domain_evidence = True
                tags.append("F")

            signal = audit_inputs[benchmark][3].get(query_id, {})
            target_ocr = number(signal.get(target, {}).get("reference_ocr_score"), 0.0)
            predicted_ocr = number(signal.get(predicted, {}).get("reference_ocr_score"), 0.0)
            baseline_top1 = baseline["predicted_slug"]
            if predicted != baseline_top1 and predicted_ocr > target_ocr and predicted_ocr - target_ocr >= 0.05:
                tags.append("G")
            explainers = set(tags) - {"D", "E"}
            if not explainers or ("A" in explainers and "B" in explainers):
                tags.append("H")
            query_ocr_row = query_ocr.get(query_id, {})
            ocr_text = " | ".join(str(line.get("text", "")) for line in query_ocr_row.get("lines", [])[:8])
            errors.append({
                "benchmark": benchmark,
                "query_id": query_id,
                "target_slug": target,
                "current_top1_slug": predicted,
                "image_only_top1_slug": baseline_top1,
                "target_rank": target_rank if target_rank != 999999 else "",
                "target_in_top5": target_in_top5,
                "scenario": scenario,
                "subset_role": manifest.get("subset_role", ""),
                "family_id": manifest.get("family_id", manifest.get("target_family_id", "")),
                "primary_tags": ";".join(dict.fromkeys(tags)),
                "category_labels": ";".join(f"{tag}:{ERROR_LABELS[tag]}" for tag in dict.fromkeys(tags)),
                "same_family_target_predicted": same_family,
                "family_metadata_type": family_type_by_slug.get(target, ""),
                "pair_visual_similarity_evidence": bool(pair.get("image_similarity")),
                "ocr_evidence_misleading": "G" in tags,
                "target_reference_ocr_score": target_ocr,
                "top1_reference_ocr_score": predicted_ocr,
                "query_ocr_text": ocr_text,
                "query_path": manifest["query_path"],
                "target_reference_path": products.get(target, {}).get("reference_image_path", ""),
                "top1_reference_path": products.get(predicted, {}).get("reference_image_path", ""),
                "top5_slugs_json": json.dumps(top5, ensure_ascii=False),
                "top5_image_scores_json": json.dumps([
                    dict(zip(as_list(baseline["top5_slugs"]), as_list(baseline["top5_scores"])))[slug]
                    for slug in top5
                ]),
                "top5_current_scores_json": json.dumps(as_list(current["top5_scores"])),
                "category_reason": "manifest scenario + displayed images; family membership/metadata; pair-level image evidence; frozen per-candidate OCR diagnostics",
            })

    write_csv(out_dir / "error_taxonomy.csv", list(errors[0]) if errors else ["benchmark", "query_id"], errors)
    summary = []
    for category, label in ERROR_LABELS.items():
        row = {"Category": label, "category_code": category}
        for benchmark, col in ((BENCHMARKS[0], "Hard count"), (BENCHMARKS[1], "Generated count")):
            subset = [error for error in errors if error["benchmark"] == benchmark]
            matching = [error for error in subset if category in error["primary_tags"].split(";")]
            row[col] = len(matching)
            row[f"{benchmark}_errors"] = len(subset)
            row[f"{benchmark}_percent_errors"] = len(matching) / len(subset) if subset else 0.0
            row[f"{benchmark}_target_in_top5_percent"] = (
                sum(bool(error["target_in_top5"]) for error in matching) / len(matching) if matching else 0.0
            )
        row["% errors"] = max(row.get(f"{name}_percent_errors", 0) for name in BENCHMARKS)
        row["Target in Top5 %"] = max(row.get(f"{name}_target_in_top5_percent", 0) for name in BENCHMARKS)
        summary.append(row)
    # D/E are disjoint retrieval-vs-reranking counts; other tags can overlap.
    counts = {benchmark: {
        "queries": len(audit_inputs[benchmark][1]),
        "errors": sum(error["benchmark"] == benchmark for error in errors),
        "target_in_top5_errors": sum(error["benchmark"] == benchmark and bool(error["target_in_top5"]) for error in errors),
        "retrieval_failures": sum(error["benchmark"] == benchmark and not bool(error["target_in_top5"]) for error in errors),
    } for benchmark in BENCHMARKS}
    write_csv(out_dir / "error_audit_summary.csv", list(summary[0]) if summary else [], summary)
    write_error_gallery(root, errors, out_dir / "final_ml_error_audit.html")
    write_error_gallery(root, errors, root / "reports/final_ml_error_audit.html")
    report_rows = []
    for row in summary:
        report_rows.append("| {Category} | {Hard count} | {Generated count} | {hard_pct:.1f}% / {generated_pct:.1f}% | {hard_t5:.1f}% / {generated_t5:.1f}% |".format(
            **row,
            hard_pct=100 * row.get("hard_near_duplicate_dev_v2_percent_errors", 0),
            generated_pct=100 * row.get("generated_stress_dev_pilot32_percent_errors", 0),
            hard_t5=100 * row.get("hard_near_duplicate_dev_v2_target_in_top5_percent", 0),
            generated_t5=100 * row.get("generated_stress_dev_pilot32_target_in_top5_percent", 0),
        ))
    hard, generated = counts[BENCHMARKS[0]], counts[BENCHMARKS[1]]
    audit_text = [
        "# Final ML error audit",
        "",
        "Taxonomy uses frozen predictions, target rank, catalog/family metadata, pair-level reference-image similarity evidence, the accepted scenario field, and frozen OCR diagnostics. The HTML gallery exposes every query, target, current Top-1 and candidate references for manual review. `D` and `E` partition target-in-Top5 reranking errors versus retrieval failures; explanatory tags A/B/C/F/G/H can overlap. Category assignment never uses a filename as evidence.",
        "",
        "| Category | Hard count | Generated count | % errors (hard / generated) | Target in Top5 % (hard / generated) |",
        "|---|---:|---:|---:|---:|",
        *report_rows,
        "",
        "## Retrieval versus reranking",
        "",
        f"- hard_v2: {hard['target_in_top5_errors']}/{hard['errors']} errors ({100*hard['target_in_top5_errors']/max(hard['errors'],1):.1f}%) have target in Top-5; {hard['retrieval_failures']} retrieval failures.",
        f"- generated pilot32: {generated['target_in_top5_errors']}/{generated['errors']} errors ({100*generated['target_in_top5_errors']/max(generated['errors'],1):.1f}%) have target in Top-5; {generated['retrieval_failures']} retrieval failures.",
        f"- Near-identical package evidence (C): hard {sum('C' in e['primary_tags'].split(';') for e in errors if e['benchmark']==BENCHMARKS[0])}; generated {sum('C' in e['primary_tags'].split(';') for e in errors if e['benchmark']==BENCHMARKS[1])}.",
        f"- Scenario/domain-shift evidence (F): hard {sum('F' in e['primary_tags'].split(';') for e in errors if e['benchmark']==BENCHMARKS[0])}; generated {sum('F' in e['primary_tags'].split(';') for e in errors if e['benchmark']==BENCHMARKS[1])}. F means an accepted generated stress scenario and is contextual, not a causal claim.",
        "",
        "The image gallery includes the query, target, current Top-1, all five candidates, SO400M scores, frozen OCR-reranker scores, family/year/subtype metadata, query OCR text, and assigned tags.",
        "",
        "[Open visual error gallery](final_ml_error_audit.html)",
        "",
    ]
    (root / "reports/final_ml_error_audit.md").write_text("\n".join(audit_text), encoding="utf-8")
    return errors, counts, audit_inputs


def write_error_gallery(root: Path, errors: list[dict], output_path: Path) -> None:
    products = {row["slug"]: row for row in read_csv(root / "data/processed/catalog_manifest.csv")}
    asset_prefix = Path(os.path.relpath(root, output_path.parent)).as_posix()
    if asset_prefix == ".":
        asset_prefix = ""
    signals_by_query = defaultdict(dict)
    for row in read_csv(root / SOURCE_RUN / "rerank_candidates.csv"):
        signals_by_query[(row["benchmark"], row["query_id"])][row["candidate_slug"]] = row
    cards = []
    for error in errors:
        benchmark = error["benchmark"]
        top5 = as_list(error["top5_slugs_json"])
        image_scores = as_list(error["top5_image_scores_json"])
        current_scores = as_list(error["top5_current_scores_json"])
        signal = signals_by_query[(benchmark, error["query_id"])]
        product = products.get(error["target_slug"], {})
        current_product = products.get(error["current_top1_slug"], {})
        top5_items = []
        for index, slug in enumerate(top5):
            meta = products.get(slug, {})
            source = signal.get(slug, {})
            ref = meta.get("reference_image_path", "")
            ref_url = f"{asset_prefix}/{ref}" if asset_prefix else ref
            score_image = number(source.get("image_score"), image_scores[index] if index < len(image_scores) else 0)
            score_ocr = current_scores[index] if index < len(current_scores) else 0.0
            top5_items.append(
                f'<div class="candidate"><div><b>#{index+1}</b> {html.escape(meta.get("title", slug))}<br><small>{html.escape(slug)}</small></div>'
                f'<img loading="lazy" src="{html.escape(ref_url)}"><div>SO400M {score_image:.5f}<br>OCR reranker {score_ocr:.5f}<br>{html.escape(meta.get("grape", ""))} · {html.escape(meta.get("category", ""))}</div></div>'
            )
        query_path = error["query_path"]
        target_path = error["target_reference_path"]
        top1_path = error["top1_reference_path"]
        query_url = f"{asset_prefix}/{query_path}" if asset_prefix else query_path
        target_url = f"{asset_prefix}/{target_path}" if asset_prefix and target_path else target_path
        top1_url = f"{asset_prefix}/{top1_path}" if asset_prefix and top1_path else top1_path
        cards.append(f'''<article class="card">
          <header><strong>{html.escape(benchmark)}</strong> · {html.escape(error['query_id'])}<br>
          Tags: {html.escape(error['primary_tags'])} · rank={html.escape(str(error['target_rank']))} · scenario={html.escape(error['scenario']) or '—'}</header>
          <div class="hero">
            <figure><figcaption>QUERY</figcaption><img loading="lazy" src="{html.escape(query_url)}"></figure>
            <figure><figcaption>TARGET REFERENCE · {html.escape(error['target_slug'])}</figcaption><img loading="lazy" src="{html.escape(target_url)}"><small>{html.escape(product.get('title',''))} · {html.escape(product.get('grape',''))}</small></figure>
            <figure><figcaption>CURRENT TOP-1 · {html.escape(error['current_top1_slug'])}</figcaption><img loading="lazy" src="{html.escape(top1_url)}"><small>{html.escape(current_product.get('title',''))} · {html.escape(current_product.get('grape',''))}</small></figure>
          </div>
          <details><summary>Show all five references</summary><div class="candidates">{''.join(top5_items)}</div></details>
          <p class="ocr"><b>Query OCR:</b> {html.escape(error['query_ocr_text']) or 'No OCR text recorded'} · target ref-OCR={error['target_reference_ocr_score']:.3f}, Top-1 ref-OCR={error['top1_reference_ocr_score']:.3f}</p>
        </article>''')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("""<!doctype html><html><head><meta charset="utf-8"><title>Final ML error audit</title><style>
body{font:14px system-ui;margin:20px;background:#f2f4f6;color:#18212a}.card{background:white;padding:16px;margin:0 auto 18px;max-width:1250px;border-radius:10px;box-shadow:0 2px 8px #0001}header{font-size:15px;margin-bottom:12px}.hero{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:#f6f7f8;padding:8px}figure img{width:100%;height:300px;object-fit:contain;background:#fff}figcaption{font-weight:700;min-height:22px;font-size:12px;overflow-wrap:anywhere}small{overflow-wrap:anywhere}.candidates{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:8px}.candidate{background:#f5f7f8;padding:7px;overflow-wrap:anywhere}.candidate img{width:100%;height:200px;object-fit:contain;background:white}.candidate small{font-size:10px}.ocr{background:#eef3f5;padding:8px;overflow-wrap:anywhere}.tags{font-size:12px}
</style></head><body><h1>Final ML error audit</h1><p>Only current production Top-1 errors. Candidate order and scores are frozen; gallery classification did not use image filenames.</p>""" + "\n".join(cards) + "</body></html>\n", encoding="utf-8")


def build_reference_cache(root: Path, out_dir: Path, config=DEFAULT_CONFIG) -> tuple[dict[str, Path], dict]:
    catalog = [row for row in read_csv(root / "data/processed/catalog_manifest.csv") if row.get("reference_image_path")]
    records = [(row["slug"], str(image_path(root, row["reference_image_path"]))) for row in catalog]
    missing = [path for _, path in records if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} reference images, first={missing[0]}")
    fingerprint = sift_cache_fingerprint(records, config)
    cache_dir = out_dir / "reference_sift_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_hashes = {slug: file_sha256(path) for slug, path in records}
    index_path = cache_dir / "cache_index.json"
    previous = read_json(index_path) if index_path.exists() else {}
    previous_matches = previous.get("cache_fingerprint") == fingerprint
    reused = extracted = 0
    paths = {}
    started = time.perf_counter()
    for position, (slug, path) in enumerate(records, 1):
        cache_path = cache_dir / cache_filename(slug)
        paths[slug] = Path(path)
        features = load_feature_cache(cache_path, image_hashes[slug], fingerprint) if previous_matches else None
        if features is None:
            features = extract_sift(read_rgb(path), config)
            save_feature_cache(cache_path, features, image_hashes[slug], fingerprint)
            extracted += 1
        else:
            reused += 1
        if position % 50 == 0 or position == len(records):
            print(f"Reference SIFT cache {position}/{len(records)} (reused={reused}, extracted={extracted})", flush=True)
    manifest = {
        "cache_fingerprint": fingerprint,
        "opencv_version": __import__("cv2").__version__,
        "sift_config": config.__dict__,
        "reference_count": len(records),
        "reference_image_sha256": image_hashes,
        "slugs_in_order": [slug for slug, _ in records],
        "reused": reused,
        "extracted": extracted,
        "build_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(index_path, manifest)
    write_json(cache_dir / "cache_manifest.json", manifest)
    return paths, manifest


class FeatureCacheReader:
    """Bounded in-process descriptor cache; disk reference cache is reusable."""

    def __init__(self, cache_dir: Path, paths: dict[str, Path], fingerprint: str,
                 image_hashes: dict[str, str], max_items: int = 96):
        self.cache_dir, self.paths, self.fingerprint = cache_dir, paths, fingerprint
        self.image_hashes, self.max_items = image_hashes, max_items
        self.memory: OrderedDict[str, SIFTFeatures] = OrderedDict()
        self.disk_loads = 0

    def get(self, slug: str) -> SIFTFeatures:
        if slug in self.memory:
            self.memory.move_to_end(slug)
            return self.memory[slug]
        features = load_feature_cache(self.cache_dir / cache_filename(slug), self.image_hashes[slug], self.fingerprint)
        if features is None:
            raise ValueError(f"Reference SIFT cache alignment/fingerprint mismatch for {slug}")
        self.disk_loads += 1
        self.memory[slug] = features
        if len(self.memory) > self.max_items:
            self.memory.popitem(last=False)
        return features


def geometry_for_benchmark(root: Path, benchmark: str, inputs: tuple, feature_reader: FeatureCacheReader,
                           reference_paths: dict[str, Path], config=DEFAULT_CONFIG) -> tuple[list[dict], dict[str, list[GeometryScore]], dict[str, float]]:
    _, current_rows, manifests, _, _ = inputs
    pair_rows, per_query, query_latency = [], {}, {}
    query_started_all = time.perf_counter()
    for position, (query_id, current) in enumerate(current_rows.items(), 1):
        # This block is inference-like: candidates come only from the frozen Top-5.
        candidates = validate_top5_candidates(as_list(current["top5_slugs"]))
        query_path = root / manifests[query_id]["query_path"]
        started = time.perf_counter()
        query_features = extract_sift(read_rgb(query_path), config)
        extraction_ms = (time.perf_counter() - started) * 1000
        scores = []
        pair_total_ms = 0.0
        for candidate_position, candidate_slug in enumerate(candidates, 1):
            if candidate_slug not in reference_paths:
                raise KeyError(f"Top-5 candidate has no frozen reference: {candidate_slug}")
            reference_features = feature_reader.get(candidate_slug)
            pair_started = time.perf_counter()
            result = match_sift_pair(query_features, reference_features, config)
            pair_ms = (time.perf_counter() - pair_started) * 1000
            pair_total_ms += pair_ms
            scores.append(result)
            pair_rows.append({
                "benchmark": benchmark,
                "query_id": query_id,
                "candidate_position_current": candidate_position,
                "candidate_slug": candidate_slug,
                "num_query_keypoints": result.num_query_keypoints,
                "num_reference_keypoints": result.num_reference_keypoints,
                "raw_matches": result.raw_matches,
                "good_matches": result.good_matches,
                "RANSAC_inliers": result.ransac_inliers,
                "inlier_ratio": result.inlier_ratio,
                "homography_valid": result.homography_valid,
                "homography_reason": result.homography_reason,
                "reprojection_error": result.reprojection_error if result.reprojection_error is not None else "",
                "projected_area_ratio": result.projected_area_ratio if result.projected_area_ratio is not None else "",
                "spatial_coverage": result.spatial_coverage,
                "geometric_score": result.geometric_score,
                "pair_matching_ms": round(pair_ms, 3),
            })
        per_query[query_id] = scores
        query_latency[query_id] = {"query_sift_extraction_ms": extraction_ms,
                                   "five_candidate_matching_ms": pair_total_ms,
                                   "geometry_total_ms": extraction_ms + pair_total_ms}
        if position % 25 == 0 or position == len(current_rows):
            elapsed = time.perf_counter() - query_started_all
            rate = position / max(elapsed, 1e-9)
            print(f"[{benchmark}] SIFT query Top-5 {position}/{len(current_rows)}; {rate:.1f} q/s", flush=True)
    return pair_rows, per_query, query_latency


def geometry_csv_row(result: GeometryScore) -> dict:
    """Stable CSV column names shared by primary and synthetic diagnostics."""
    return {
        "num_query_keypoints": result.num_query_keypoints,
        "num_reference_keypoints": result.num_reference_keypoints,
        "raw_matches": result.raw_matches,
        "good_matches": result.good_matches,
        "RANSAC_inliers": result.ransac_inliers,
        "inlier_ratio": result.inlier_ratio,
        "homography_valid": result.homography_valid,
        "homography_reason": result.homography_reason,
        "reprojection_error": result.reprojection_error if result.reprojection_error is not None else "",
        "projected_area_ratio": result.projected_area_ratio if result.projected_area_ratio is not None else "",
        "spatial_coverage": result.spatial_coverage,
        "geometric_score": result.geometric_score,
    }


def get_method_orders(benchmark: str, inputs: tuple, geometry: dict[str, list[GeometryScore]]) -> dict[str, dict[str, list[str]]]:
    baseline_rows, current_rows, _, signals, _ = inputs
    methods = {"A_SO400M_image_only": {}, "B_current_production": {}, "C_SIFT_geometry_only": {},
               "D_SO400M_plus_geometry_w20": {}, "conservative_policy": {}}
    for weight in GRID_WEIGHTS:
        methods[f"E_current_plus_geometry_w{weight:.2f}"] = {}
    for query_id, current in current_rows.items():
        baseline = baseline_rows[query_id]
        image_slugs = as_list(baseline["top5_slugs"])
        image_scores = [number(value) for value in as_list(baseline["top5_scores"])]
        current_slugs = as_list(current["top5_slugs"])
        current_scores = [number(value) for value in as_list(current["top5_scores"])]
        if len(image_slugs) != 5 or len(current_slugs) != 5 or set(image_slugs) != set(current_slugs):
            raise AssertionError(f"Candidate set changed between frozen image/current runs: {query_id}")
        geo = geometry[query_id]
        geo_by_slug = {slug: item.geometric_score for slug, item in zip(current_slugs, geo)}
        geo_scores = valid_geometry_scores(geo)
        geo_by_slug = {slug: score for slug, score in zip(current_slugs, geo_scores)}
        methods["A_SO400M_image_only"][query_id] = image_slugs
        methods["B_current_production"][query_id] = current_slugs
        methods["C_SIFT_geometry_only"][query_id] = rank_candidates(current_slugs, geo_scores)
        d_scores = fuse_scores(image_scores, [geo_by_slug[slug] for slug in image_slugs], 0.20)
        methods["D_SO400M_plus_geometry_w20"][query_id] = rank_candidates(image_slugs, d_scores)
        geo_norm = normalize_geometry_scores(geo_scores)
        current_norm = normalize_geometry_scores(current_scores)
        methods["conservative_policy"][query_id] = conservative_geometry_order(current_slugs, geo, current_norm)
        for weight in GRID_WEIGHTS:
            fused = [(1 - weight) * base + weight * geom for base, geom in zip(current_norm, geo_norm)]
            methods[f"E_current_plus_geometry_w{weight:.2f}"][query_id] = rank_candidates(current_slugs, fused)
    return methods


def rank_value(target: str, ranking: list[str], original_rank: int) -> int:
    return ranking.index(target) + 1 if target in ranking else original_rank


def calculate_metrics(rows: list[dict], rankings: dict[str, list[str]], family_by_slug: dict[str, str],
                      family_type_by_slug: dict[str, str], benchmark: str) -> dict:
    n = len(rows)
    outcomes = []
    for row in rows:
        target = row["target_slug"]
        original_rank = integer(row.get("target_rank"), 999999)
        rank = rank_value(target, rankings[row["query_id"]], original_rank)
        outcomes.append({"row": row, "rank": rank, "correct": rank == 1, "in_top5": rank <= 5})
    result = {
        "queries": n,
        "top1": sum(item["correct"] for item in outcomes) / max(n, 1),
        "recall_at_5": sum(item["in_top5"] for item in outcomes) / max(n, 1),
        "mrr": sum(0.0 if item["rank"] >= 999999 else 1 / item["rank"] for item in outcomes) / max(n, 1),
        "target_in_top5_queries": sum(item["in_top5"] for item in outcomes),
        "target_in_top5_conditioned_top1": (
            sum(item["correct"] for item in outcomes if item["in_top5"])
            / max(sum(item["in_top5"] for item in outcomes), 1)
        ),
    }
    if benchmark == "generated_stress_dev_pilot32":
        for scenario in ("distance_crop", "glare_bad_light", "handheld", "slight_angle"):
            selected = [item for item in outcomes if item["row"].get("scenario_id") == scenario]
            result[f"scenario:{scenario}"] = _slice_metrics(selected)
        for subset in ("representative", "hard"):
            selected = [item for item in outcomes if item["row"].get("subset_role") == subset]
            result[f"subset:{subset}"] = _slice_metrics(selected)
        for family_kind in ("vintage", "subtype", "other"):
            selected = [item for item in outcomes if family_type_by_slug.get(item["row"]["target_slug"]) == family_kind]
            result[f"family_type:{family_kind}"] = _slice_metrics(selected)
    else:
        for family_kind in ("vintage", "subtype", "other"):
            selected = [item for item in outcomes if family_type_by_slug.get(item["row"]["target_slug"]) == family_kind]
            result[f"family_type:{family_kind}"] = _slice_metrics(selected)
        competing_same_family = []
        for item in outcomes:
            target_family = family_by_slug.get(item["row"]["target_slug"])
            top5 = rankings[item["row"]["query_id"]]
            if target_family and any(slug != item["row"]["target_slug"] and family_by_slug.get(slug) == target_family for slug in top5):
                competing_same_family.append(item)
        result["within_family_disambiguation"] = _slice_metrics(competing_same_family)
    return result


def _slice_metrics(outcomes: list[dict]) -> dict:
    count = len(outcomes)
    return {
        "queries": count,
        "top1": sum(item["correct"] for item in outcomes) / max(count, 1),
        "recall_at_5": sum(item["in_top5"] for item in outcomes) / max(count, 1),
        "mrr": sum(0.0 if item["rank"] >= 999999 else 1 / item["rank"] for item in outcomes) / max(count, 1),
        "top5_conditioned_top1": sum(item["correct"] for item in outcomes if item["in_top5"])
        / max(sum(item["in_top5"] for item in outcomes), 1),
    }


def metric_rows(rows: list[dict], methods: dict[str, dict[str, list[str]]], family_by_slug: dict[str, str],
                family_type_by_slug: dict[str, str], benchmark: str,
                geometry: dict[str, list[GeometryScore]]) -> tuple[dict, list[dict], list[dict]]:
    pred_rows, metrics_by_method, transition_rows = [], {}, []
    baseline_name = "B_current_production"
    targets = [row["target_slug"] for row in rows]
    base_top1 = [methods[baseline_name][row["query_id"]][0] for row in rows]
    for method, orders in methods.items():
        metrics_by_method[method] = calculate_metrics(rows, orders, family_by_slug, family_type_by_slug, benchmark)
        top1 = [orders[row["query_id"]][0] for row in rows]
        trans = transition_counts(base_top1, top1, targets)
        transition_rows.append({"benchmark": benchmark, "method": method, **trans,
                                "rescued_broken_ratio": trans["wrong_to_correct"] / max(trans["correct_to_wrong"], 1)})
    for row in rows:
        baseline = methods["A_SO400M_image_only"][row["query_id"]]
        current = methods[baseline_name][row["query_id"]]
        pred = {
            "query_id": row["query_id"], "target_slug": row["target_slug"],
            "original_target_rank": row.get("target_rank", ""),
            "target_in_current_top5": row["target_slug"] in current,
            "top5_candidate_set_unchanged": set(current) == set(baseline),
            "scenario_id": row.get("scenario_id", ""), "subset_role": row.get("subset_role", ""),
            "target_family_type": family_type_by_slug.get(row["target_slug"], ""),
        }
        geometry_rows = geometry[row["query_id"]]
        geometry_slugs = methods["B_current_production"][row["query_id"]]
        geometry_by_slug = {slug: item for slug, item in zip(geometry_slugs, geometry_rows)}
        geo_ranking = sorted(geometry_slugs, key=lambda slug: (-valid_geometry_scores([geometry_by_slug[slug]])[0], geometry_slugs.index(slug)))
        target_geo = geometry_by_slug.get(row["target_slug"])
        target_geo_score = valid_geometry_scores([target_geo])[0] if target_geo is not None else None
        wrong_geo_scores = [valid_geometry_scores([geometry_by_slug[slug]])[0]
                            for slug in geometry_slugs if slug != row["target_slug"]]
        pred.update({
            "geometry_only_top1": geo_ranking[0],
            "geometry_only_top1_correct": geo_ranking[0] == row["target_slug"],
            "target_geometric_rank": geo_ranking.index(row["target_slug"]) + 1 if row["target_slug"] in geo_ranking else "outside_top5",
            "target_geometric_score": target_geo_score if target_geo_score is not None else "",
            "best_wrong_geometric_score": max(wrong_geo_scores) if wrong_geo_scores else "",
            "target_vs_best_wrong_geo_margin": target_geo_score - max(wrong_geo_scores) if target_geo_score is not None and wrong_geo_scores else "",
            "target_homography_valid": target_geo.homography_valid if target_geo is not None else False,
        })
        for method, orders in methods.items():
            pred[f"{method}_top1"] = orders[row["query_id"]][0]
            pred[f"{method}_ranking"] = json.dumps(orders[row["query_id"]], ensure_ascii=False)
            pred[f"{method}_correct"] = orders[row["query_id"]][0] == row["target_slug"]
        pred_rows.append(pred)
    return metrics_by_method, pred_rows, transition_rows


def latency_rows(benchmark: str, query_latency: dict[str, dict], cache_manifest: dict,
                 baseline_pipeline_ms: float, pair_rows: list[dict]) -> list[dict]:
    extraction = [row["query_sift_extraction_ms"] for row in query_latency.values()]
    matching = [row["five_candidate_matching_ms"] for row in query_latency.values()]
    total = [row["geometry_total_ms"] for row in query_latency.values()]
    estimated = [baseline_pipeline_ms + item for item in total]
    return [{
        "benchmark": benchmark,
        "queries": len(query_latency),
        "candidate_pairs_matched": len(pair_rows),
        "pairs_per_query": len(pair_rows) / max(len(query_latency), 1),
        "reference_count_cached": cache_manifest["reference_count"],
        "reference_cache_reused": cache_manifest["reused"],
        "reference_cache_extracted": cache_manifest["extracted"],
        "reference_cache_build_s_offline": cache_manifest["build_elapsed_seconds"],
        "query_sift_extraction_mean_ms": float(np_mean(extraction)),
        "query_sift_extraction_p95_ms": float(np_percentile(extraction, 95)),
        "five_candidate_matching_mean_ms": float(np_mean(matching)),
        "five_candidate_matching_p95_ms": float(np_percentile(matching, 95)),
        "geometry_online_total_mean_ms": float(np_mean(total)),
        "geometry_online_total_p95_ms": float(np_percentile(total, 95)),
        "current_pipeline_baseline_mean_ms": baseline_pipeline_ms,
        "estimated_total_pipeline_mean_ms": baseline_pipeline_ms + float(np_mean(total)),
        "estimated_total_pipeline_p95_ms": baseline_pipeline_ms + float(np_percentile(total, 95)),
        "sla_3s_ok": baseline_pipeline_ms + float(np_percentile(total, 95)) < 3000,
    }]


def np_mean(values):
    import numpy as np
    return np.mean(values) if values else 0.0


def np_percentile(values, percentile):
    import numpy as np
    return np.percentile(values, percentile) if values else 0.0


def select_policy(benchmark_metrics: dict[str, dict], transitions: list[dict]) -> dict:
    candidates = []
    for weight in GRID_WEIGHTS:
        method = f"E_current_plus_geometry_w{weight:.2f}"
        hard_delta = benchmark_metrics[BENCHMARKS[0]][method]["top1"] - benchmark_metrics[BENCHMARKS[0]]["B_current_production"]["top1"]
        gen_delta = benchmark_metrics[BENCHMARKS[1]][method]["top1"] - benchmark_metrics[BENCHMARKS[1]]["B_current_production"]["top1"]
        trans = [row for row in transitions if row["method"] == method]
        rescued = sum(row["wrong_to_correct"] for row in trans)
        broken = sum(row["correct_to_wrong"] for row in trans)
        candidates.append({"method": method, "weight": weight, "hard_delta": hard_delta, "generated_delta": gen_delta,
                           "rescued": rescued, "broken": broken, "ratio": rescued / max(broken, 1)})
    # Method qualifies only with non-negative deltas on both benchmark sets and
    # a material gain on at least one, as declared by the stop condition.
    qualified = [row for row in candidates if row["hard_delta"] >= 0 and row["generated_delta"] >= 0
                 and (row["hard_delta"] >= 0.02 or row["generated_delta"] >= 0.02
                      or (row["ratio"] >= 3 and row["rescued"] > row["broken"]))]
    if not qualified:
        return {"decision": "KEEP_CURRENT_PIPELINE", "selected_method": "B_current_production", "selected_weight": None,
                "reason": "No fixed-grid fusion met material-gain or >=3:1 rescued/broken criteria without regression on either primary benchmark.",
                "grid_candidates": candidates}
    winner = max(qualified, key=lambda row: (row["hard_delta"] + row["generated_delta"], row["ratio"], -row["weight"]))
    return {"decision": "FIX_GEOMETRIC_RERANKER", "selected_method": winner["method"], "selected_weight": winner["weight"],
            "reason": "Fixed-grid fusion met the declared stop criterion and did not regress either primary benchmark.",
            "selected_candidate": winner, "grid_candidates": candidates}


def make_geometry_gallery(root: Path, out_dir: Path, pred_rows_by_benchmark: dict[str, list[dict]],
                          pair_scores: dict[tuple[str, str], dict], current_rows_by_benchmark: dict[str, dict]) -> None:
    products = {row["slug"]: row for row in read_csv(root / "data/processed/catalog_manifest.csv")}
    asset_prefix = Path(os.path.relpath(root, out_dir)).as_posix()
    examples = []
    for benchmark, pred_rows in pred_rows_by_benchmark.items():
        current_rows = current_rows_by_benchmark[benchmark]
        for row in pred_rows:
            current_ok = row["B_current_production_correct"]
            selected_ok = row["selected_policy_correct"]
            query_id = row["query_id"]
            current = current_rows[query_id]
            top5 = as_list(current["top5_slugs"])
            geo_values = [number(pair_scores.get((benchmark, query_id, slug), {}).get("geometric_score")) for slug in top5]
            geo_low, geo_high = min(geo_values, default=0), max(geo_values, default=0)
            normalized = {slug: ((number(pair_scores.get((benchmark, query_id, slug), {}).get("geometric_score")) - geo_low) / (geo_high - geo_low)
                                if geo_high > geo_low else 0.0) for slug in top5}
            target_margin = normalized.get(row["target_slug"], 0.0) - max((score for slug, score in normalized.items() if slug != row["target_slug"]), default=0.0)
            target_rank = next((index + 1 for index, slug in enumerate(sorted(top5, key=lambda slug: (-normalized[slug], top5.index(slug)))) if slug == row["target_slug"]), "outside Top-5")
            if current_ok and selected_ok:
                continue
            if not current_ok and selected_ok:
                kind = "GEOMETRY RESCUE"
            elif current_ok and not selected_ok:
                kind = "GEOMETRY BREAK"
            elif row["target_slug"] not in top5:
                kind = "TARGET OUTSIDE TOP-5"
            else:
                target_geometry = pair_scores.get((benchmark, query_id, row["target_slug"]), {})
                winner_geometry = pair_scores.get((benchmark, query_id, current["predicted_slug"]), {})
                valid_count = sum(bool(pair_scores.get((benchmark, query_id, slug), {}).get("homography_valid")) for slug in top5)
                max_good = max((integer(pair_scores.get((benchmark, query_id, slug), {}).get("good_matches")) for slug in top5), default=0)
                max_raw = max((integer(pair_scores.get((benchmark, query_id, slug), {}).get("raw_matches")) for slug in top5), default=0)
                if valid_count == 0 and max(max_good, max_raw) < 4:
                    kind = "BACKGROUND / OBJECT MISMATCH"
                elif valid_count == 0 or not target_geometry.get("homography_valid"):
                    kind = "NO VALID HOMOGRAPHY FOR TARGET"
                elif winner_geometry.get("homography_valid") and abs(target_margin) <= 0.10:
                    kind = "GEOMETRY CANNOT DISTINGUISH NEAR-DUPLICATES"
                else:
                    kind = "GEOMETRY FAVORS A WRONG REFERENCE"
            query_path = next(item["query_path"] for item in read_csv(root / "data/benchmarks" / benchmark / "manifest.csv") if item["query_id"] == query_id)
            cards = []
            for index, slug in enumerate(top5):
                meta = products.get(slug, {})
                geometry = pair_scores.get((benchmark, query_id, slug), {})
                ref = meta.get("reference_image_path", "")
                cards.append(f'<div class="candidate"><b>#{index+1} {html.escape(slug)}</b><img loading="lazy" src="{html.escape(asset_prefix + "/" + ref)}"><p>valid={geometry.get("homography_valid")} · inliers={geometry.get("RANSAC_inliers")} · geo={number(geometry.get("geometric_score")):.4f}</p></div>')
            target = row["target_slug"]
            target_path = products.get(target, {}).get("reference_image_path", "")
            examples.append(f'<article><h3>{kind}: {html.escape(benchmark)} · {html.escape(query_id)}</h3><p>target={html.escape(target)} · current={html.escape(row["B_current_production_top1"])} · selected={html.escape(row["selected_policy_top1"])} · tags={html.escape(row.get("audit_tags", ""))}</p><p>target geometry rank={target_rank} · target−best-wrong normalized geo margin={target_margin:.3f}</p><div class="hero"><figure><b>QUERY</b><img src="{html.escape(asset_prefix + "/" + query_path)}"></figure><figure><b>TARGET REFERENCE</b><img src="{html.escape(asset_prefix + "/" + target_path)}"></figure></div><div class="grid">{''.join(cards)}</div></article>')
    (out_dir / "geometry_failure_gallery.html").write_text("<!doctype html><meta charset='utf-8'><title>Geometry failure examples</title><style>body{font:14px system-ui;background:#f3f5f6;margin:18px}article{background:white;padding:14px;margin:16px auto;max-width:1350px}.hero,.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.hero{grid-template-columns:repeat(2,1fr);max-width:500px}.candidate{background:#f4f6f7;padding:7px;overflow-wrap:anywhere}img{width:100%;height:240px;object-fit:contain;background:white}figure{margin:0}</style><h1>Geometry rescues, breaks and unresolved cases</h1>"+"\n".join(examples)+"</html>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="SIFT uses CPU; kept for run metadata")
    parser.add_argument("--skip-reference-cache", action="store_true", help="reuse a matching cache in --cache-dir")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    root = PROJECT_ROOT
    run_id = datetime.now(timezone.utc).strftime("final_ml_geometric_reranker_%Y%m%dT%H%M%SZ")
    out_dir = root / "artifacts/experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {out_dir}", flush=True)

    reproduction = verify_frozen_current_pipeline(root)
    config_payload = {
        "run_id": run_id,
        "objective": "FINAL ML sanity check: error audit + fixed-grid SIFT geometric reranker",
        "source_current_pipeline_run": str(SOURCE_RUN),
        "frozen_current_pipeline": CURRENT_POLICY,
        "current_pipeline_reproduction": reproduction,
        "benchmarks_primary": list(BENCHMARKS),
        "synthetic_policy": "regression sanity check only after a primary method is selected",
        "candidate_generation": "frozen SigLIP2 SO400M Top-5 only; no new candidates",
        "inference_uses_ground_truth": False,
        "fusion_weights": list(GRID_WEIGHTS),
        "conservative_policy": {"min_normalized_geo_margin": 0.15, "min_inliers": 12, "min_inlier_ratio": 0.25},
        "selected_reranker_grid_from_frozen_run": read_json(root / SOURCE_RUN / "selected_reranker.json"),
        "sift_config": DEFAULT_CONFIG.__dict__,
    }
    write_json(out_dir / "config.json", config_payload)
    write_json(out_dir / "baseline_reproduction.json", reproduction)
    print("Frozen current pipeline reproduced exactly on hard_v2 and generated pilot32.", flush=True)

    # Stage 1: use current predictions and create the error audit/gallery before
    # computing any SIFT pair evidence.
    errors, audit_counts, audit_inputs = make_error_audit(root, out_dir)
    print(f"Error audit complete: hard={audit_counts[BENCHMARKS[0]]}, generated={audit_counts[BENCHMARKS[1]]}", flush=True)

    # Stage 2: offline reference SIFT features. All usable references are cached.
    cache_override = Path(args.cache_dir) if args.cache_dir else None
    if cache_override:
        reference_cache_dir = cache_override
        catalog = [row for row in read_csv(root / "data/processed/catalog_manifest.csv") if row.get("reference_image_path")]
        refs = [(row["slug"], str(root / row["reference_image_path"])) for row in catalog]
        fingerprint = sift_cache_fingerprint(refs, DEFAULT_CONFIG)
        index_file = reference_cache_dir / "cache_index.json"
        manifest = read_json(index_file)
        if manifest.get("cache_fingerprint") != fingerprint:
            raise ValueError("Requested reference cache fingerprint does not match current catalog/config")
        manifest = dict(manifest)
        manifest["cache_reused_for_this_run"] = True
        manifest["reused"] = manifest["reference_count"]
        manifest["extracted"] = 0
        reference_paths = {slug: Path(path) for slug, path in refs}
        image_hashes = manifest["reference_image_sha256"]
        cache_dir = reference_cache_dir
    else:
        reference_paths, manifest = build_reference_cache(root, out_dir, DEFAULT_CONFIG)
        image_hashes = manifest["reference_image_sha256"]
        cache_dir = out_dir / "reference_sift_cache"
    config_payload["reference_cache"] = {
        "path": str(cache_dir.relative_to(root)) if cache_dir.is_relative_to(root) else str(cache_dir),
        "fingerprint": manifest["cache_fingerprint"],
        "reference_count": manifest["reference_count"],
        "reused": manifest["reused"],
        "extracted": manifest["extracted"],
        "opencv_version": manifest["opencv_version"],
    }
    write_json(out_dir / "config.json", config_payload)
    feature_reader = FeatureCacheReader(cache_dir, reference_paths, manifest["cache_fingerprint"], image_hashes)

    # Stage 3: score exactly the frozen Top-5 for each primary query.
    all_pair_rows, all_predictions, all_transitions, all_latency = [], {}, [], []
    per_benchmark_metrics = {}
    _products, family_by_slug, _pair, family_type_by_slug = metadata_and_families(root)
    metadata = {}
    for benchmark in BENCHMARKS:
        pair_rows, geometry, query_latency = geometry_for_benchmark(
            root, benchmark, audit_inputs[benchmark], feature_reader, reference_paths, DEFAULT_CONFIG
        )
        all_pair_rows.extend(pair_rows)
        methods = get_method_orders(benchmark, audit_inputs[benchmark], geometry)
        manifest_rows = list(audit_inputs[benchmark][2].values())
        metrics, predictions, transitions = metric_rows(manifest_rows, methods, family_by_slug, family_type_by_slug, benchmark, geometry)
        per_benchmark_metrics[benchmark] = metrics
        all_transitions.extend(transitions)
        method_name = "E_current_plus_geometry_w0.20"  # replaced after global fixed-grid selection
        base_rows = list(audit_inputs[benchmark][1].values())
        # Include pair timings/features in a lookup used to render the selected-policy failure gallery.
        method_by_query = {name: rank for name, rank in methods.items()}
        all_predictions[benchmark] = predictions
        pair_lookup = {(row["benchmark"], row["query_id"], row["candidate_slug"]): row for row in pair_rows}
        baseline_total_ms = 0.0
        latency_source = read_csv(root / SOURCE_RUN / "latency_summary.csv")
        for entry in latency_source:
            if entry["benchmark"] == benchmark:
                baseline_total_ms = number(entry["pipeline_total_mean_ms"], 0.0)
                break
        all_latency.extend(latency_rows(benchmark, query_latency, manifest, baseline_total_ms, pair_rows))
        write_csv(out_dir / benchmark / "geometry_pair_scores.csv", list(pair_rows[0]), pair_rows)
        for query_id, timing in query_latency.items():
            row = next(item for item in predictions if item["query_id"] == query_id)
            row.update(timing)
        write_csv(out_dir / benchmark / "predictions.csv", list(predictions[0]), predictions)
        write_json(out_dir / benchmark / "metrics.json", {
            "benchmark": benchmark,
            "methods": metrics,
            "candidate_set_unchanged": True,
            "geometry_pairs": len(pair_rows),
            "homography_success_rate": sum(row["homography_valid"] for row in pair_rows) / max(len(pair_rows), 1),
            "target_top5_coverage_current": sum(
                row["target_slug"] in as_list(row["top5_slugs"])
                for row in audit_inputs[benchmark][1].values()
            ) / max(len(audit_inputs[benchmark][1]), 1),
        })
        write_csv(out_dir / benchmark / "transitions.csv", list(transitions[0]), transitions)
        print(f"[{benchmark}] geometry scores saved; pairs={len(pair_rows)}", flush=True)

    policy = select_policy(per_benchmark_metrics, all_transitions)
    write_json(out_dir / "selected_policy.json", policy)
    selected_method = policy["selected_method"]

    # The selected pipeline predictions are defined only after evaluating the
    # fixed grid; update per-query outputs and rerender compact geometry gallery.
    pair_lookup = {(row["benchmark"], row["query_id"], row["candidate_slug"]): row for row in all_pair_rows}
    current_rows_by_benchmark = {benchmark: audit_inputs[benchmark][1] for benchmark in BENCHMARKS}
    pred_rows_by_benchmark = {}
    summary_rows = []
    for benchmark in BENCHMARKS:
        rows = list(audit_inputs[benchmark][1].values())
        orders = {row["query_id"]: json.loads(row[f"{selected_method}_ranking"]) for row in all_predictions[benchmark]}
        for pred in all_predictions[benchmark]:
            pred["selected_policy"] = selected_method
            pred["selected_policy_top1"] = pred[f"{selected_method}_top1"]
            pred["selected_policy_ranking"] = pred[f"{selected_method}_ranking"]
            pred["selected_policy_correct"] = pred[f"{selected_method}_correct"]
            audit = next((error for error in errors if error["benchmark"] == benchmark and error["query_id"] == pred["query_id"]), None)
            pred["audit_tags"] = audit["primary_tags"] if audit else ""
        pred_rows_by_benchmark[benchmark] = all_predictions[benchmark]
        write_csv(out_dir / benchmark / "predictions.csv", list(all_predictions[benchmark][0]), all_predictions[benchmark])
        write_json(out_dir / benchmark / "metrics.json", {
            "benchmark": benchmark,
            "methods": per_benchmark_metrics[benchmark],
            "selected_method": selected_method,
            "selected_method_metrics": per_benchmark_metrics[benchmark][selected_method],
            "candidate_set_unchanged": True,
            "geometry_pairs": sum(row["benchmark"] == benchmark for row in all_pair_rows),
            "homography_success_rate": sum(bool(row["homography_valid"]) for row in all_pair_rows if row["benchmark"] == benchmark)
                / max(sum(row["benchmark"] == benchmark for row in all_pair_rows), 1),
        })
        for method, values in per_benchmark_metrics[benchmark].items():
            summary_rows.append({"benchmark": benchmark, "method": method, **{key: value for key, value in values.items() if isinstance(value, (int, float))}})
    write_csv(out_dir / "geometry_pair_scores.csv", list(all_pair_rows[0]), all_pair_rows)
    write_csv(out_dir / "benchmark_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(out_dir / "transition_summary.csv", list(all_transitions[0]), all_transitions)
    write_csv(out_dir / "latency_summary.csv", list(all_latency[0]), all_latency)
    make_geometry_gallery(root, out_dir, pred_rows_by_benchmark, pair_lookup, current_rows_by_benchmark)

    # Synthetic is a regression-only check for the selected method. Only run it
    # if SIFT passed the predeclared stop condition on both primary benchmarks.
    synthetic_result = None
    if policy["decision"] == "FIX_GEOMETRIC_RERANKER":
        synthetic_benchmark = "synthetic_dev"
        source_synth = read_csv(root / SOURCE_RUN / "benchmarks" / synthetic_benchmark / "reranked_predictions.csv")
        # Synthetic predictions are still fixed Top-5; compute SIFT only after selection.
        synth_baseline = read_csv(root / SOURCE_RUN / "benchmarks" / synthetic_benchmark / "baseline_predictions.csv")
        synth_manifests = read_csv(root / "data/benchmarks" / synthetic_benchmark / "manifest.csv")
        current_by_id = {row["query_id"]: row for row in source_synth}
        baseline_by_id = {row["query_id"]: row for row in synth_baseline}
        manifest_by_id = {row["query_id"]: row for row in synth_manifests}
        synth_pair_rows = []
        synth_latency = {}
        synth_orders = {}
        synth_geometry = {}
        for index, (query_id, current) in enumerate(current_by_id.items(), 1):
            candidates = as_list(current["top5_slugs"])
            query_started = time.perf_counter()
            query_features = extract_sift(read_rgb(root / manifest_by_id[query_id]["query_path"]), DEFAULT_CONFIG)
            extraction_ms = (time.perf_counter() - query_started) * 1000
            geos, matching_ms = [], 0.0
            for position, slug in enumerate(candidates, 1):
                started = time.perf_counter()
                geom = match_sift_pair(query_features, feature_reader.get(slug), DEFAULT_CONFIG)
                pair_ms = (time.perf_counter() - started) * 1000
                matching_ms += pair_ms
                geos.append(geom)
                synth_pair_rows.append({"benchmark": synthetic_benchmark, "query_id": query_id,
                    "candidate_position_current": position, "candidate_slug": slug,
                    **geometry_csv_row(geom), "pair_matching_ms": round(pair_ms, 3)})
            synth_geometry[query_id] = geos
            base_scores = [number(x) for x in as_list(current["top5_scores"])]
            geo_scores = valid_geometry_scores(geos)
            w = policy["selected_weight"]
            base_norm, geo_norm = normalize_geometry_scores(base_scores), normalize_geometry_scores(geo_scores)
            fused = [(1-w)*a + w*b for a,b in zip(base_norm,geo_norm)]
            synth_orders[query_id] = rank_candidates(candidates, fused)
            synth_latency[query_id] = {"query_sift_extraction_ms": extraction_ms, "five_candidate_matching_ms": matching_ms,
                                       "geometry_total_ms": extraction_ms+matching_ms}
            if index % 100 == 0 or index == len(current_by_id):
                print(f"[synthetic regression] {index}/{len(current_by_id)}", flush=True)
        synth_rows = list(current_by_id.values())
        synthetic_result = calculate_metrics(synth_rows, synth_orders, family_by_slug, family_type_by_slug, "synthetic_dev")
        source_synthetic_metrics = read_json(root / SOURCE_RUN / "benchmarks/synthetic_dev/metrics.json")["reranked"]
        synth_targets = [current_by_id[qid]["target_slug"] for qid in current_by_id]
        synth_before = [current_by_id[qid]["predicted_slug"] for qid in current_by_id]
        synth_after = [synth_orders[qid][0] for qid in current_by_id]
        synth_transition = transition_counts(synth_before, synth_after, synth_targets)
        synth_transition_row = {"benchmark": synthetic_benchmark, "method": selected_method, **synth_transition,
                                "rescued_broken_ratio": synth_transition["wrong_to_correct"] / max(synth_transition["correct_to_wrong"], 1),
                                "regression_only_after_primary_selection": True}
        all_transitions.append(synth_transition_row)
        synth_predictions = []
        for query_id, current in current_by_id.items():
            target = current["target_slug"]
            candidates = as_list(current["top5_slugs"])
            if set(candidates) != set(as_list(baseline_by_id[query_id]["top5_slugs"])):
                raise AssertionError(f"Synthetic candidate set changed for {query_id}")
            geos = synth_geometry[query_id]
            geo_by_slug = {slug: geom for slug, geom in zip(candidates, geos)}
            geo_values = valid_geometry_scores(geos)
            geo_order = rank_candidates(candidates, geo_values)
            target_geo = geo_by_slug.get(target)
            wrong_scores = [valid_geometry_scores([geo_by_slug[slug]])[0] for slug in candidates if slug != target]
            synth_predictions.append({
                "query_id": query_id, "target_slug": target,
                "current_production_top1": current["predicted_slug"],
                "selected_top1": synth_orders[query_id][0],
                "selected_top1_correct": synth_orders[query_id][0] == target,
                "current_top1_correct": current["predicted_slug"] == target,
                "selected_ranking": json.dumps(synth_orders[query_id], ensure_ascii=False),
                "top5_candidate_set_unchanged": True,
                "original_target_rank": current.get("target_rank", ""),
                "geometry_only_top1": geo_order[0],
                "target_geometric_rank": geo_order.index(target) + 1 if target in geo_order else "outside_top5",
                "target_geometric_score": valid_geometry_scores([target_geo])[0] if target_geo else "",
                "best_wrong_geometric_score": max(wrong_scores) if target_geo else "",
                "target_vs_best_wrong_geo_margin": valid_geometry_scores([target_geo])[0] - max(wrong_scores) if target_geo and wrong_scores else "",
                "target_homography_valid": target_geo.homography_valid if target_geo else False,
            })
        write_csv(out_dir / synthetic_benchmark / "predictions.csv", list(synth_predictions[0]), synth_predictions)
        write_csv(out_dir / synthetic_benchmark / "geometry_pair_scores.csv", list(synth_pair_rows[0]), synth_pair_rows)
        write_csv(out_dir / synthetic_benchmark / "transitions.csv", list(synth_transition_row), [synth_transition_row])
        baseline_current = {"top1": source_synthetic_metrics["top1_accuracy"],
                            "recall_at_5": source_synthetic_metrics["recall_at_5"],
                            "mrr": source_synthetic_metrics["mrr"]}
        write_json(out_dir / synthetic_benchmark / "metrics.json", {
            "benchmark": synthetic_benchmark, "selected_policy": selected_method,
            "baseline_current_production": baseline_current, "selected_policy_metrics": synthetic_result,
            "regression_only_after_selection": True, "candidate_set_unchanged": True, "transitions": synth_transition,
        })
        summary_rows.extend([
            {"benchmark": synthetic_benchmark, "method": "B_current_production", "queries": synthetic_result["queries"],
             "top1": baseline_current["top1"], "recall_at_5": baseline_current["recall_at_5"], "mrr": baseline_current["mrr"]},
            {"benchmark": synthetic_benchmark, "method": selected_method, "queries": synthetic_result["queries"],
             "top1": synthetic_result["top1"], "recall_at_5": synthetic_result["recall_at_5"], "mrr": synthetic_result["mrr"]},
        ])
        baseline_total = next((number(row["pipeline_total_mean_ms"]) for row in read_csv(root / SOURCE_RUN / "latency_summary.csv") if row["benchmark"] == synthetic_benchmark), 278.5)
        all_latency.extend(latency_rows(synthetic_benchmark, synth_latency, manifest, baseline_total, synth_pair_rows))
        policy["synthetic_regression"] = {
            **synthetic_result, "baseline_current_production": baseline_current,
            "top1_delta": synthetic_result["top1"] - baseline_current["top1"],
            "recall_at_5_delta": synthetic_result["recall_at_5"] - baseline_current["recall_at_5"],
            "mrr_delta": synthetic_result["mrr"] - baseline_current["mrr"],
            "regression_guard_passed": synthetic_result["top1"] >= baseline_current["top1"] - 0.01,
        }
        write_csv(out_dir / "benchmark_summary.csv", list(summary_rows[0]), summary_rows)
        write_csv(out_dir / "transition_summary.csv", list(all_transitions[0]), all_transitions)
        write_csv(out_dir / "latency_summary.csv", list(all_latency[0]), all_latency)
        write_json(out_dir / "selected_policy.json", policy)
    else:
        policy["synthetic_regression"] = {"skipped": True, "reason": "Only a regression check after a selected primary method; stop condition did not select SIFT."}
        write_json(out_dir / "selected_policy.json", policy)

    write_main_report(root, out_dir, errors, audit_counts, per_benchmark_metrics, all_transitions, all_latency, policy, all_pair_rows)
    print(f"DONE: {out_dir}", flush=True)


def write_main_report(root: Path, out_dir: Path, errors: list[dict], audit_counts: dict,
                      metrics: dict[str, dict], transitions: list[dict], latency: list[dict],
                      policy: dict, pairs: list[dict]) -> None:
    audit_path = root / "reports/final_ml_error_audit.md"
    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    summary = ["# Final ML geometric reranker report", "", "## Table 1 — Error audit", "", *audit_lines[2:], "",
               "## Table 2 — Main results", "",
               "| Method | Hard Top-1 | Generated Top-1 | Hard rescued/broken | Generated rescued/broken | Added SIFT latency (hard / generated) |",
               "|---|---:|---:|---:|---:|---:|"]
    display_methods = ["A_SO400M_image_only", "B_current_production", "C_SIFT_geometry_only", "D_SO400M_plus_geometry_w20",
                       *[f"E_current_plus_geometry_w{weight:.2f}" for weight in GRID_WEIGHTS], "conservative_policy"]
    for method in display_methods:
        hard = metrics[BENCHMARKS[0]][method]["top1"]
        gen = metrics[BENCHMARKS[1]][method]["top1"]
        trans = {row["benchmark"]: row for row in transitions if row["method"] == method}
        hard_trans, gen_trans = trans[BENCHMARKS[0]], trans[BENCHMARKS[1]]
        hard_latency = next(row for row in latency if row["benchmark"] == BENCHMARKS[0])
        gen_latency = next(row for row in latency if row["benchmark"] == BENCHMARKS[1])
        latency_cell = "—" if method in ("A_SO400M_image_only", "B_current_production") else f"+{hard_latency['geometry_online_total_mean_ms']:.0f} / +{gen_latency['geometry_online_total_mean_ms']:.0f} ms mean"
        summary.append(f"| {method} | {hard:.4f} | {gen:.4f} | {hard_trans['wrong_to_correct']}/{hard_trans['correct_to_wrong']} | {gen_trans['wrong_to_correct']}/{gen_trans['correct_to_wrong']} | {latency_cell} |")
    summary += ["", "## Top-5-conditioned exact accuracy", "",
                "| Benchmark | Current production | Selected SIFT fusion | Target in Top-5 |",
                "|---|---:|---:|---:|"]
    for benchmark in BENCHMARKS:
        current = metrics[benchmark]["B_current_production"]
        selected = metrics[benchmark][policy["selected_method"]]
        summary.append(f"| {benchmark} | {current['target_in_top5_conditioned_top1']:.4f} | {selected['target_in_top5_conditioned_top1']:.4f} | {selected['target_in_top5_queries']}/{selected['queries']} |")
    summary += ["", "## Geometry diagnostics", "",
                "| Benchmark | Pair homography success | Target geo rank 1 among current errors | Target geo rank ≤2 | Median target−best-wrong geo margin | Geometry-only Top-1 |",
                "|---|---:|---:|---:|---:|---:|"]
    for benchmark in BENCHMARKS:
        rows = [row for row in pairs if row["benchmark"] == benchmark]
        current_errors = [error for error in errors if error["benchmark"] == benchmark and error["target_in_top5"]]
        geo_by_query = defaultdict(list)
        for row in rows:
            geo_by_query[row["query_id"]].append(row)
        best = rank2 = 0
        for error in current_errors:
            ranking = sorted(geo_by_query[error["query_id"]], key=lambda item: (-number(item["geometric_score"]), integer(item["candidate_position_current"])))
            rank = next((i+1 for i, item in enumerate(ranking) if item["candidate_slug"] == error["target_slug"]), 999)
            best += rank == 1
            rank2 += rank <= 2
        margins = []
        for error in current_errors:
            query_rows = geo_by_query[error["query_id"]]
            target_pair = next((item for item in query_rows if item["candidate_slug"] == error["target_slug"]), None)
            wrong_scores = [number(item["geometric_score"]) for item in query_rows if item["candidate_slug"] != error["target_slug"]]
            if target_pair and wrong_scores:
                margins.append(number(target_pair["geometric_score"]) - max(wrong_scores))
        margin_median = np_percentile(margins, 50) if margins else 0.0
        current_correct = metrics[benchmark]["B_current_production"]["queries"]
        summary.append(f"| {benchmark} | {sum(bool(r['homography_valid']) for r in rows)/max(len(rows),1):.1%} ({sum(bool(r['homography_valid']) for r in rows)}/{len(rows)}) | {best}/{len(current_errors)} ({best/max(len(current_errors),1):.1%}) | {rank2}/{len(current_errors)} ({rank2/max(len(current_errors),1):.1%}) | {margin_median:.4f} | {metrics[benchmark]['C_SIFT_geometry_only']['top1']:.4f} |")
    summary += ["", "## Hard family and generated breakdowns", ""]
    for benchmark in BENCHMARKS:
        summary.append(f"### {benchmark}")
        summary.append("")
        for method in ("B_current_production", policy["selected_method"]):
            m = metrics[benchmark][method]
            if benchmark == "generated_stress_dev_pilot32":
                details = ", ".join(f"{scope}: n={m[scope]['queries']}, Top-1={m[scope]['top1']:.1%}" for scope in (
                    "scenario:distance_crop", "scenario:glare_bad_light", "scenario:handheld", "scenario:slight_angle",
                    "subset:representative", "subset:hard", "family_type:vintage", "family_type:subtype", "family_type:other"))
            else:
                details = ", ".join(f"{scope}: n={m[scope]['queries']}, Top-1={m[scope]['top1']:.1%}" for scope in (
                    "family_type:vintage", "family_type:subtype", "family_type:other", "within_family_disambiguation"))
            summary.append(f"- {method}: {details}")
        summary.append("")
    summary += ["## Latency", "", "| Benchmark | Query SIFT mean | 5-candidate matching mean | Geometry total mean / p95 | Estimated full pipeline mean / p95 | SLA <3s |", "|---|---:|---:|---:|---:|---|"]
    for row in latency:
        summary.append(f"| {row['benchmark']} | {row['query_sift_extraction_mean_ms']:.1f} ms | {row['five_candidate_matching_mean_ms']:.1f} ms | {row['geometry_online_total_mean_ms']:.1f} / {row['geometry_online_total_p95_ms']:.1f} ms | {row['estimated_total_pipeline_mean_ms']:.1f} / {row['estimated_total_pipeline_p95_ms']:.1f} ms | {row['sla_3s_ok']} |")
    summary += ["", "Reference SIFT features are precomputed offline for the full usable 2042-reference catalog; online latency excludes offline extraction and cache load. Matching latency includes all five candidates per query. Estimated total adds measured query extraction + five matchings to the frozen pipeline mean from the prior OCR-reranker run.", "",
                "## Decision", "", f"**{policy['decision']}** — {policy['reason']}", ""]
    synthetic = policy.get("synthetic_regression", {})
    if synthetic and not synthetic.get("skipped"):
        baseline_synthetic = read_json(root / SOURCE_RUN / "benchmarks/synthetic_dev/metrics.json")["reranked"]
        summary += ["## Synthetic regression sanity check", "",
                    f"After selecting the fixed fusion on hard_v2 + generated only, synthetic_dev was {baseline_synthetic['top1_accuracy']:.4f} Top-1 / {baseline_synthetic['recall_at_5']:.4f} R@5 / {baseline_synthetic['mrr']:.4f} MRR; selected fusion is {synthetic['top1']:.4f} / {synthetic['recall_at_5']:.4f} / {synthetic['mrr']:.4f}. This is a post-selection check, not a tuning target.", ""]
    if policy["decision"] == "FIX_GEOMETRIC_RERANKER":
        summary.append(f"Selected fixed-grid weight: {policy['selected_weight']:.2f}; selected method: `{policy['selected_method']}`.")
    else:
        summary.append("Frozen recognition pipeline: `SigLIP2 SO400M/384 Top-5 → current eslav OCR → conservative reference-OCR reranker (alpha 0.30, vintage ±0.05, minimum text margin 0.05) → Top-1`. SIFT is not integrated.")
    summary += ["", "### Required answers", "",
                f"1. Remaining errors: hard {audit_counts[BENCHMARKS[0]]['errors']} and generated {audit_counts[BENCHMARKS[1]]['errors']}; taxonomy and per-query evidence are in the audit CSV/gallery.",
                f"2. Target already in Top-5: hard {audit_counts[BENCHMARKS[0]]['target_in_top5_errors']}/{audit_counts[BENCHMARKS[0]]['errors']}; generated {audit_counts[BENCHMARKS[1]]['target_in_top5_errors']}/{audit_counts[BENCHMARKS[1]]['errors']}.",
                "3–12. SIFT separation, geometry-only/fused Top-1, deltas, rescues/breaks, vintage/subtype metrics, homography rate and added latency are in the tables above.",
                f"13. Integrate SIFT: {'yes, fixed weight '+str(policy['selected_weight']) if policy['decision']=='FIX_GEOMETRIC_RERANKER' else 'no; gain did not meet the stop condition'}.",
                f"14. Continue ML: {'yes, only this validated SIFT fusion' if policy['decision']=='FIX_GEOMETRIC_RERANKER' else 'no further recognition ML is justified by these benchmarks'}.",
                "", "**Final recommendation:** " + ("CONTINUE ML" if policy["decision"] == "FIX_GEOMETRIC_RERANKER" else "FREEZE ML AND MOVE TO PRODUCT/E2E"),
                "", "The hard_v2 set reuses synthetic-derived query images and the generated pilot has only 32 products repeated across four scenarios. Treat these as strong diagnostic evidence for this fixed signal; collect real field queries before exploring additional ML methods.",
                "", "[Error audit](final_ml_error_audit.md) · [Error gallery](final_ml_error_audit.html) · [Geometry failure gallery](../artifacts/experiments/" + out_dir.name + "/geometry_failure_gallery.html)",
                "", "[Experiment artifacts](../" + str(out_dir.relative_to(root)).replace('\\', '/') + ")", ""]
    report_path = root / "reports/final_ml_geometric_reranker_report.md"
    report_path.write_text("\n".join(summary), encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
