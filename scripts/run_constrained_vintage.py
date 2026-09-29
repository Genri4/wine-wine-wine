#!/usr/bin/env python3
"""Candidate-constrained vintage recognition bake-off.

For every challenge query the allowed years = known years of same-family
Top-5 candidates (catalog side; no target identity). Every method must
choose one allowed year from the query year crop or abstain.

Methods:
  A. current eslav OCR on the reference-guided query crop (baseline)
  B. visual reference-crop matching: query crop embedding vs each
     candidate's reference year-crop embedding (cosine, L2-normalized):
     SO400M / PE-Core / DINOv2 (local patch matcher)
  C. digit-only recognizer (CRNN trained on synthetic year crops),
     constrained decoding over allowed years
  D. VLM ceiling (Qwen2-VL-2B-Instruct, multiple-choice over allowed years)

Evaluation: constrained year accuracy, coverage, vintage Top-1 impact
(rescued/broken) via the family-safe swap, pairwise-only accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from recognition.ocr_engine import OCR_CONFIGS, create_ocr_engine  # noqa: E402
from recognition.ocr_reranker import FusionConfig  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.vintage_disambiguation import (  # noqa: E402
    conservative_vintage_rerank,
    extract_years_with_corrections,
)
from recognition.constrained_vintage import (  # noqa: E402
    decide_from_scores,
    normalize_crop,
    pairwise_year_accuracy,
)
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
)

BENCHMARKS = ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
PREV_RUN = PROJECT_ROOT / "artifacts" / "experiments" / "vintage_disambiguation_20260922T1"
CROPS_DIR_NAME = "crops"
BASELINE_RUN = PROJECT_ROOT / "artifacts" / "experiments" / "so400m_ocr_reranker_20260920T193925Z"
FROZEN_BLEND = FusionConfig(
    policy="reference_ocr_blend", alpha=0.30, vintage_bonus=0.05, vintage_penalty=0.05, min_text_margin=0.05
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-constrained vintage recognition")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["ocr", "so400m", "pe_core", "dinov2", "digit", "vlm"])
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, _ = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    catalog_by_slug = {item.item_id: item for item in catalog_items}

    manifest_by_query: dict[str, dict[str, dict]] = {}
    baseline_by_query: dict[str, dict[str, dict]] = {}
    family_by_slug: dict[str, str] = {}
    candidate_years: dict[str, str | None] = {}
    for benchmark in BENCHMARKS:
        manifest_by_query[benchmark] = {
            row["query_id"]: row
            for row in _read_csv(_default_benchmark_manifest(project_root, benchmark, None))
        }
        baseline_by_query[benchmark] = {
            row["query_id"]: row
            for row in _read_csv(BASELINE_RUN / "benchmarks" / benchmark / "baseline_predictions.csv")
        }
    with (project_root / "data" / "benchmarks" / "hard_near_duplicate_dev_v2" / "families.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            family_by_slug[row["slug"]] = row["family_id"]
    prev_challenge_rows = list(
        csv.DictReader(open(PREV_RUN / "vintage_rerank_per_query.csv", encoding="utf-8"))
    )
    # candidate years from the previous milestone's metadata audit
    audit_rows = list(csv.DictReader(open(PREV_RUN / "vintage_metadata_audit.csv", encoding="utf-8")))
    for row in audit_rows:
        if row["year"]:
            candidate_years[row["slug"]] = row["year"]

    crop_manifest = list(csv.DictReader(open(crops_dir / "crop_manifest.csv", encoding="utf-8")))

    # ------------------------------------------------------------------
    # Build the unified per-query task list
    # ------------------------------------------------------------------
    tasks: list[dict] = []
    for row in crop_manifest:
        benchmark = row["benchmark"]
        query_id = row["query_id"]
        evidence = json.loads(row["candidates"])
        base_row = baseline_by_query[benchmark][query_id]
        top5 = json.loads(base_row["top5_slugs"])
        target = row["target_slug"]
        # Allowed years: same-family candidates with known years (catalog side).
        target_family = family_by_slug.get(target)
        allowed: dict[str, list[str]] = {}
        for slug in top5:
            if family_by_slug.get(slug) != target_family:
                continue
            year = candidate_years.get(slug)
            if year:
                allowed.setdefault(year, []).append(slug)
        if len(allowed) < 2:
            continue  # years do not disambiguate this query
        candidates_in_top5 = [slug for slug in top5 if slug in evidence]
        tasks.append(
            {
                "benchmark": benchmark,
                "query_id": query_id,
                "target_slug": target,
                "target_year": row["target_year"],
                "allowed_years": sorted(allowed),
                "allowed_year_to_slugs": allowed,
                "candidates_in_top5": candidates_in_top5,
                "baseline_correct_top1": base_row["correct_top1"] == "True",
                "baseline_top1": base_row["predicted_slug"],
                "top5": top5,
                "top5_scores": json.loads(base_row["top5_scores"]),
            }
        )
    print(f"tasks with >=2 distinct allowed years: {len(tasks)}", flush=True)
    for benchmark in BENCHMARKS:
        count = sum(1 for task in tasks if task["benchmark"] == benchmark)
        baseline_wrong = sum(1 for task in tasks if task["benchmark"] == benchmark and not task["baseline_correct_top1"])
        print(f"  {benchmark}: {count} tasks, {baseline_wrong} baseline-wrong", flush=True)

    results: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Method A: current eslav OCR on the reference-guided query crop
    # ------------------------------------------------------------------
    if "ocr" in args.methods:
        engine = create_ocr_engine(OCR_CONFIGS["current_eslav"], device="gpu:0")
        rows = []
        started_total = perf_counter()
        for task in tasks:
            crop_path = crops_dir / task["benchmark"] / "query" / f"{task['query_id']}__{task['target_slug']}.png"
            if not crop_path.is_file():
                rows.append(_row(task, "ocr", None, {}, abstained=True, reason="no_crop"))
                continue
            started = perf_counter()
            lines = engine.predict_lines(crop_path)
            elapsed = (perf_counter() - started) * 1000
            years, _ = extract_years_with_corrections(
                [{"text": line.text, "confidence": line.confidence} for line in lines]
            )
            # Constrained: keep only allowed years, score = max confidence.
            scores = {}
            for year, observations in years.items():
                if year in task["allowed_years"]:
                    scores[year] = max(obs["confidence"] for obs in observations)
            decision = decide_from_scores(scores, task["allowed_years"])
            rows.append(_row(task, "ocr", decision, scores, latency_ms=elapsed))
        RESULTS_TOTAL = perf_counter() - started_total
        results["ocr"] = rows
        print(f"method ocr done ({RESULTS_TOTAL:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Method B: visual reference-crop matching
    # ------------------------------------------------------------------
    visual_encoders: dict[str, Any] = {}
    if "so400m" in args.methods:
        visual_encoders["so400m"] = _so400m_encoder()
    if "pe_core" in args.methods:
        try:
            visual_encoders["pe_core"] = _pe_core_encoder(args.device)
        except Exception as exc:
            print(f"pe_core unavailable: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    if "dinov2" in args.methods:
        visual_encoders["dinov2"] = _dinov2_encoder(args.device)

    for encoder_name, encoder in visual_encoders.items():
        rows = []
        started_total = perf_counter()
        # Precompute reference year-crop embeddings per slug.
        reference_embeddings: dict[str, Any] = {}
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        slugs_needed = set()
        for task in tasks:
            for slug in task["candidates_in_top5"]:
                if candidate_years.get(slug):
                    slugs_needed.add(slug)
        ref_batch = []
        ref_slugs = []
        for slug in sorted(slugs_needed):
            ref_crop_path = crops_dir / "*" / "reference" / f"{slug}.png"
            matches = list(crops_dir.glob(f"*/reference/{slug}.png"))
            if not matches:
                continue
            with Image.open(matches[0]) as crop:
                ref_batch.append(normalize_crop(crop))
                ref_slugs.append(slug)
        if ref_batch:
            embeddings = encoder.encode_images(ref_batch)
            for slug, embedding in zip(ref_slugs, embeddings):
                reference_embeddings[slug] = embedding

        for task in tasks:
            crop_path = crops_dir / task["benchmark"] / "query" / f"{task['query_id']}__{task['target_slug']}.png"
            if not crop_path.is_file():
                rows.append(_row(task, encoder_name, None, {}, abstained=True, reason="no_crop"))
                continue
            started = perf_counter()
            with Image.open(crop_path) as crop:
                query_embedding = encoder.encode_images([normalize_crop(crop)])[0]
            # Cosine similarity against each allowed-year candidate reference.
            scores: dict[str, float] = {}
            for year, slugs in task["allowed_year_to_slugs"].items():
                year_scores = []
                for slug in slugs:
                    reference = reference_embeddings.get(slug)
                    if reference is None:
                        continue
                    similarity = sum(a * b for a, b in zip(query_embedding, reference))
                    year_scores.append(similarity)
                if year_scores:
                    scores[year] = max(year_scores)
            elapsed = (perf_counter() - started) * 1000
            decision = decide_from_scores(scores, task["allowed_years"])
            row = _row(task, encoder_name, decision, scores, latency_ms=elapsed)
            # Store margins for diagnostics
            if decision.margin is not None:
                row["margin"] = round(decision.margin, 4)
            rows.append(row)
        results[encoder_name] = rows
        encoder.release() if hasattr(encoder, "release") else None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"method {encoder_name} done ({perf_counter() - started_total:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Method C: digit-only CRNN recognizer (synthetic training + constrained)
    # ------------------------------------------------------------------
    if "digit" in args.methods:
        from recognition.year_recognizer import (
            train_year_recognizer,
            YearRecognizer,
            synthetic_year_dataset,
        )

        train_dir = run_dir / "digit" / "synthetic_train"
        recognizer = train_year_recognizer(
            train_dir,
            years=range(1990, 2027),
            samples_per_year=120,
            epochs=6,
            seed=20260922,
        )
        rows = []
        for task in tasks:
            crop_path = crops_dir / task["benchmark"] / "query" / f"{task['query_id']}__{task['target_slug']}.png"
            if not crop_path.is_file():
                rows.append(_row(task, "digit", None, {}, abstained=True, reason="no_crop"))
                continue
            with Image.open(crop_path) as crop:
                scores = recognizer.constrained_scores(normalize_crop(crop), task["allowed_years"])
            decision = decide_from_scores(scores, task["allowed_years"])
            rows.append(_row(task, "digit", decision, scores))
        results["digit"] = rows
        print("method digit done", flush=True)

    # ------------------------------------------------------------------
    # Method D: VLM ceiling (Qwen2-VL-2B-Instruct)
    # ------------------------------------------------------------------
    if "vlm" in args.methods:
        try:
            from recognition.vlm_year import VlmYearJudge

            vlm = VlmYearJudge(device=args.device)
        except Exception as exc:
            print(f"VLM unavailable: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            vlm = None
        if vlm is not None:
            rows = []
            # Baseline-wrong first, then the rest.
            ordered_tasks = sorted(tasks, key=lambda task: task["baseline_correct_top1"])
            for task in ordered_tasks:
                crop_path = crops_dir / task["benchmark"] / "query" / f"{task['query_id']}__{task['target_slug']}.png"
                if not crop_path.is_file():
                    rows.append(_row(task, "vlm", None, {}, abstained=True, reason="no_crop"))
                    continue
                started = perf_counter()
                chosen = vlm.choose_year(crop_path, task["allowed_years"])
                elapsed = (perf_counter() - started) * 1000
                scores = {year: (1.0 if year == chosen else 0.0) for year in task["allowed_years"]} if chosen else {}
                decision = decide_from_scores(scores, task["allowed_years"]) if chosen else _abstain(task["allowed_years"])
                rows.append(_row(task, "vlm", decision, scores, latency_ms=elapsed))
            results["vlm"] = rows
            vlm.release()
            print("method vlm done", flush=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    summary_rows = []
    per_method_rows = []
    transition_rows = []
    latency_rows = []
    for method, rows in results.items():
        write_csv(
            run_dir / f"year_recognition_{method}.csv",
            list(rows[0]),
            rows,
        )
        for benchmark in BENCHMARKS:
            scoped = [row for row in rows if row["benchmark"] == benchmark]
            if not scoped:
                continue
            decided = [row for row in scoped if not row["abstained"]]
            correct = sum(1 for row in decided if row["chosen_year"] == row["target_year"])
            wrong = len(decided) - correct
            # Vintage Top-1 impact: family-safe swap when the chosen year
            # maps to exactly one same-family candidate and differs from the
            # current pipeline's Top-1 year.
            rescued = broken = swaps = 0
            for row in scoped:
                if row["abstained"] or row["chosen_year"] is None:
                    continue
                swap = _family_swap(
                    row, candidate_years, family_by_slug
                )
                if swap is None:
                    continue
                swaps += 1
                if swap == row["target_slug"] and not row["baseline_correct_top1"]:
                    rescued += 1
                elif swap != row["target_slug"] and row["baseline_correct_top1"]:
                    broken += 1
            summary_rows.append(
                {
                    "method": method,
                    "benchmark": benchmark,
                    "queries": len(scoped),
                    "year_decided": len(decided),
                    "year_correct": correct,
                    "year_correct_rate": round(correct / len(decided), 4) if decided else None,
                    "year_wrong": wrong,
                    "vintage_swaps": swaps,
                    "vintage_rescued": rescued,
                    "vintage_broken": broken,
                }
            )
            pairwise = pairwise_year_accuracy(scoped)
            per_method_rows.append({"method": method, "benchmark": benchmark, **pairwise})
            transition_rows.append(
                {
                    "method": method,
                    "benchmark": benchmark,
                    "rescued": rescued,
                    "broken": broken,
                    "baseline_wrong_total": sum(1 for row in scoped if not row["baseline_correct_top1"]),
                }
            )
            latencies = [row["latency_ms"] for row in scoped if row.get("latency_ms")]
            if latencies:
                latency_rows.append(
                    {
                        "method": method,
                        "benchmark": benchmark,
                        "mean_ms": round(statistics.mean(latencies), 1),
                        "p50_ms": round(statistics.median(latencies), 1),
                        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
                    }
                )

    write_csv(
        run_dir / "benchmark_summary.csv",
        ["method", "benchmark", "queries", "year_decided", "year_correct", "year_correct_rate", "year_wrong", "vintage_swaps", "vintage_rescued", "vintage_broken"],
        summary_rows,
    )
    write_csv(
        run_dir / "pairwise_summary.csv",
        ["method", "benchmark", "pairwise_queries", "pairwise_correct", "pairwise_accuracy"],
        per_method_rows,
    )
    write_csv(
        run_dir / "transition_summary.csv",
        ["method", "benchmark", "rescued", "broken", "baseline_wrong_total"],
        transition_rows,
    )
    write_csv(
        run_dir / "latency_summary.csv",
        ["method", "benchmark", "mean_ms", "p50_ms", "p95_ms"],
        latency_rows,
    )

    config = {
        "run_id": run_dir.name,
        "purpose": "candidate-constrained vintage recognition over SIFT-aligned year crops",
        "allowed_years_source": "same-family Top-5 candidates' catalog years (no target identity)",
        "crop_normalization": "grayscale, padded square, 224x224 Lanczos",
        "methods_run": args.methods,
        "tasks_total": len(tasks),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=1, default=str))


def _row(task, method, decision, scores, abstained=False, reason="", latency_ms=None):
    return {
        "benchmark": task["benchmark"],
        "query_id": task["query_id"],
        "target_slug": task["target_slug"],
        "target_year": task["target_year"],
        "allowed_years": ",".join(task["allowed_years"]),
        "top5": ",".join(task["top5"]),
        "baseline_correct_top1": task["baseline_correct_top1"],
        "pairwise": 1,
        "chosen_year": decision.chosen_year if decision else None,
        "abstained": abstained or (decision is None or decision.abstained),
        "reason": reason or (decision.reason if decision else ""),
        "scores": json.dumps(scores, ensure_ascii=False),
        "margin": decision.margin if decision else None,
        "latency_ms": round(latency_ms, 1) if latency_ms else None,
        "method": method,
    }


def _abstain(allowed_years):
    from recognition.constrained_vintage import ConstrainedDecision

    return ConstrainedDecision(None, {}, None, True, "skipped")


def _family_swap(row, candidate_years, family_by_slug):
    """Family-safe swap for the chosen year; returns the new Top-1 slug."""

    top5 = row["top5"].split(",") if isinstance(row.get("top5"), str) else row.get("top5", [])
    chosen_year = row["chosen_year"]
    if not chosen_year:
        return None
    top1 = top5[0]
    top1_family = family_by_slug.get(top1)
    top1_year = candidate_years.get(top1)
    if top1_year == chosen_year:
        return None  # already correct year; nothing to do
    for slug in top5[1:]:
        if family_by_slug.get(slug) != top1_family:
            continue
        if candidate_years.get(slug) == chosen_year:
            return slug
    return None


def _so400m_encoder():
    from recognition.encoder_adapters import SigLIP2So400m384Adapter

    return SigLIP2So400m384Adapter(device="cuda", batch_size=16)


def _pe_core_encoder(device):
    from recognition.encoder_adapters import PECoreL14_336Adapter

    return PECoreL14_336Adapter(device=device, batch_size=16)


def _dinov2_encoder(device):
    from recognition.encoder_adapters import DinoV2RegistersLargeAdapter

    return DinoV2RegistersLargeAdapter(device=device, batch_size=16)


if __name__ == "__main__":
    main()
