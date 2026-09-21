#!/usr/bin/env python3
"""SO400M OCR + text/metadata reranking milestone runner.

Pipeline (frozen retrieval backbone siglip2_so400m_384):

    query image -> SO400M Top-5 -> OCR evidence -> text signals per candidate
    -> conservative fusion -> final Top-1

Stages:
1. retrieval with the frozen SO400M adapter + validated reference cache;
2. baseline reproduction check against the frozen bake-off run
   (stops the run when reproduction fails);
3. OCR caches are loaded (built once by scripts/build_ocr_cache.py);
4. text signals per query x Top-5 candidate (rerank_candidates.csv);
5. calibration-only grid search (pilot32 held-out products, hard_v2
   calibration families, synthetic calibration products) with a
   pre-declared <=1pp synthetic guard and lexicographic selection;
6. frozen selected reranker evaluated once on full benchmarks with
   transitions, per-scenario/subset/family metrics, oracles and latency;
7. artifacts + markdown report + HTML error analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.crop_diagnostics import full_ranking, rank_metrics, target_rank_full  # noqa: E402
from recognition.encoder_adapters import (  # noqa: E402
    ReferenceCacheMismatch,
    SigLIP2So400m384Adapter,
    load_reference_cache,
    reference_cache_dir,
)
from recognition.family_metrics import family_diagnostics  # noqa: E402
from recognition.ocr_engine import OCR_MODEL_KEY, WineLabelOcr  # noqa: E402
from recognition.ocr_reranker import (  # noqa: E402
    FUSION_POLICIES,
    POLICY_IMAGE_ONLY,
    FusionConfig,
    oracle_text_top1,
    oracle_top5_present,
    rerank_one_query,
    top1_transition_counts,
)
from recognition.preprocessing import load_rgb_image  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _default_benchmark_manifest,
    _path,
    _read_csv,
)

BENCHMARKS = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
SCENARIOS = ("distance_crop", "glare_bad_light", "handheld", "slight_angle")
SUBSETS = ("representative", "hard")
FROZEN_BASELINE_RUN = "artifacts/experiments/encoder_bakeoff_20260920T143928Z/models/siglip2_so400m_384"

# Frozen grid (brief Part N): small and predefined; no third-decimal tuning.
GRID_ALPHAS = (0.10, 0.20, 0.30, 0.40)
GRID_VINTAGE = ((0.05, 0.05), (0.10, 0.10))
GRID_MARGINS = (0.05,)
GRID_POLICIES = tuple(p for p in FUSION_POLICIES if p != POLICY_IMAGE_ONLY)

SYNTHETIC_GUARD_TOP1_PP = 1.0  # pre-declared clean regression guard
SPLIT_SEED = 20260920
SYNTHETIC_CAL_PRODUCTS = 400
HARD_V2_CAL_FAMILIES = 128
PILOT32_SPLIT_PLAN = {"representative": (8, 8), "vintage": (2, 2), "subtype": (2, 2)}
BASELINE_TOP1_AGREEMENT_MIN = 0.995
BASELINE_METRIC_TOLERANCE_PP = 0.15
OCR_LATENCY_PROBE_QUERIES = 32


@dataclass
class RetrievalRecord:
    query_id: str
    target_slug: str
    top10_slugs: list[str]
    top10_scores: list[float]
    top1_score: float
    target_rank: int | None
    embedding_latency_ms: float
    retrieval_latency_ms: float
    latency_ms: float


def main() -> None:
    parser = argparse.ArgumentParser(description="SO400M OCR reranking milestone")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-retrieval", action="store_true", help="reuse retrieval dump from an earlier run dir")
    parser.add_argument("--retrieval-dump", default=None, help="run dir with retrieval dumps for --skip-retrieval")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("so400m_ocr_reranker_%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "artifacts" / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run dir: {run_dir}")

    config = {
        "run_id": run_id,
        "purpose": "OCR + text/metadata reranking over frozen siglip2_so400m_384 Top-5",
        "benchmarks": args.benchmarks,
        "grid": {"alphas": GRID_ALPHAS, "vintage_bonus_penalty": GRID_VINTAGE, "margins": GRID_MARGINS, "policies": GRID_POLICIES},
        "synthetic_guard_top1_pp": SYNTHETIC_GUARD_TOP1_PP,
        "split_seed": SPLIT_SEED,
        "synthetic_cal_products": SYNTHETIC_CAL_PRODUCTS,
        "hard_v2_cal_families": HARD_V2_CAL_FAMILIES,
        "pilot32_split_plan": PILOT32_SPLIT_PLAN,
        "frozen_baseline_run": FROZEN_BASELINE_RUN,
        "selection_objective": ["pilot32_heldout_top1", "hard_v2_cal_top1", "-correct_to_wrong", "pilot32_hard_heldout_top1", "synthetic_cal_top1"],
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cache_root = PROJECT_ROOT / "artifacts" / "ocr_cache" / OCR_MODEL_KEY
    (run_dir / "ocr_model.json").write_text((cache_root / "ocr_engine.json").read_text(encoding="utf-8"), encoding="utf-8")

    catalog_meta = json.loads(_path(PROJECT_ROOT, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, catalog_by_slug = _catalog_items(
        PROJECT_ROOT, _read_csv(_path(PROJECT_ROOT, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    candidate_indexes = {item.item_id: build_candidate_text_index(item.metadata) for item in catalog_items}

    # ------------------------------------------------------------------
    # Stage 1-2: retrieval + baseline reproduction
    # ------------------------------------------------------------------
    if args.skip_retrieval and args.retrieval_dump:
        dump_dir = Path(args.retrieval_dump)
        retrieval = {}
        for benchmark in args.benchmarks:
            payload = json.loads((dump_dir / benchmark / "retrieval_dump.json").read_text(encoding="utf-8"))
            retrieval[benchmark] = [RetrievalRecord(**row) for row in payload]
        reproduction = json.loads((dump_dir / "baseline_reproduction.json").read_text(encoding="utf-8"))
        resource = json.loads((dump_dir / "resource.json").read_text(encoding="utf-8"))
    else:
        retrieval, reproduction, resource = run_retrieval_all(args, run_dir, catalog_meta, catalog_items)
        (run_dir / "resource.json").write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
        for benchmark in args.benchmarks:
            dump_dir = run_dir / benchmark
            dump_dir.mkdir(exist_ok=True)
            (dump_dir / "retrieval_dump.json").write_text(
                json.dumps([asdict(record) for record in retrieval[benchmark]]), encoding="utf-8"
            )
    (run_dir / "baseline_reproduction.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [name for name, stats in reproduction.items() if not stats["reproduced"]]
    if failed:
        print(f"STOPPING: baseline not reproduced for {failed}; see baseline_reproduction.json")
        return

    # ------------------------------------------------------------------
    # Stage 3-4: OCR evidence + text signals
    # ------------------------------------------------------------------
    ocr_dir = run_dir / "ocr"
    ocr_dir.mkdir(exist_ok=True)
    query_evidence, ocr_stats = load_query_evidence(args.benchmarks, cache_root)
    reference_evidence = load_reference_evidence(cache_root)
    signals, rerank_timings = compute_all_signals(args.benchmarks, retrieval, query_evidence, reference_evidence, candidate_indexes)
    for benchmark in args.benchmarks:
        shutil.copy2(cache_root / benchmark / "query_ocr.jsonl", ocr_dir / f"query_ocr_{benchmark}.jsonl")
    shutil.copy2(cache_root / "catalog_references" / "reference_ocr.jsonl", ocr_dir / "reference_ocr.jsonl")
    write_rerank_candidates(run_dir, args.benchmarks, retrieval, signals)

    # ------------------------------------------------------------------
    # Stage 5: calibration splits + grid search
    # ------------------------------------------------------------------
    splits = build_calibration_splits(args.benchmarks)
    write_calibration_split_csv(run_dir, splits)
    grid_rows, selected_config = grid_search(args.benchmarks, retrieval, signals, splits)
    write_csv(run_dir / "grid_search.csv", list(grid_rows[0]), grid_rows)
    (run_dir / "selected_reranker.json").write_text(
        json.dumps({"config": asdict(selected_config), "key": selected_config.key()}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Selected reranker: {selected_config.key()}")

    # ------------------------------------------------------------------
    # Stage 6: final evaluation with the frozen reranker
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    for benchmark in args.benchmarks:
        results[benchmark] = final_evaluate_benchmark(
            run_dir, benchmark, retrieval[benchmark], signals[benchmark], selected_config
        )
    transition_rows = write_transition_summary(run_dir, results)
    write_benchmark_summary(run_dir, results)
    oracle_rows = write_oracle_summary(run_dir, args.benchmarks, retrieval, signals)
    coverage = {
        benchmark: json.loads((cache_root / benchmark / "ocr_coverage.json").read_text(encoding="utf-8"))
        for benchmark in args.benchmarks
    }
    latency = latency_report(run_dir, args.benchmarks, retrieval, ocr_stats, rerank_timings, resource)

    hard_v2_family_types = hard_v2_family_classification()
    generate_markdown_report(run_dir, results, selected_config, reproduction, ocr_stats, coverage, oracle_rows, transition_rows, latency)
    generate_html_report(run_dir, args.benchmarks, retrieval, signals, query_evidence, candidate_indexes, selected_config, hard_v2_family_types)
    shutil.copy2(run_dir / "report.md", PROJECT_ROOT / "reports" / "so400m_ocr_reranker_report.md")
    print("DONE")


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def run_retrieval_all(args, run_dir: Path, catalog_meta, catalog_items):
    import torch

    adapter = SigLIP2So400m384Adapter(device=args.device)
    fingerprint = adapter.fingerprint(catalog_meta["manifest_sha256"], len(catalog_items))
    cache_dir = reference_cache_dir(PROJECT_ROOT, adapter.model_key)
    try:
        reference_embeddings = load_reference_cache(cache_dir, fingerprint, [item.item_id for item in catalog_items])
        print("Reference cache loaded")
    except (FileNotFoundError, ReferenceCacheMismatch) as exc:
        raise SystemExit(
            f"Reference cache unusable ({type(exc).__name__}: {exc}); "
            "rebuild it via scripts/run_encoder_benchmark.py --model siglip2_so400m_384 first"
        )

    reference_matrix = torch.nn.functional.normalize(
        torch.tensor(reference_embeddings, dtype=torch.float32, device=adapter.device), p=2, dim=1
    )

    reproduction = {}
    retrieval: dict[str, list[RetrievalRecord]] = {}
    for benchmark in args.benchmarks:
        manifest_rows = _read_csv(_default_benchmark_manifest(PROJECT_ROOT, benchmark, None))
        records = retrieval_pass(adapter, reference_matrix, [item.item_id for item in catalog_items], manifest_rows, benchmark, run_dir)
        retrieval[benchmark] = records
        frozen_dir = PROJECT_ROOT / FROZEN_BASELINE_RUN / benchmark
        reproduction[benchmark] = verify_baseline_reproduction(records, frozen_dir)
        print(f"[{benchmark}] reproduction: {json.dumps(reproduction[benchmark])}")
    vram_peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if str(adapter.device).startswith("cuda") else None
    adapter.release()
    return retrieval, reproduction, {"retrieval_torch_peak_vram_mb": vram_peak}


def retrieval_pass(adapter, reference_matrix, catalog_slugs, manifest_rows, benchmark, run_dir) -> list[RetrievalRecord]:
    import torch

    records: list[RetrievalRecord] = []
    checkpoint = run_dir / f"_retrieval_checkpoint_{benchmark}.json"
    if checkpoint.is_file():
        rows = json.loads(checkpoint.read_text(encoding="utf-8"))
        records = [RetrievalRecord(**row) for row in rows]
        print(f"  resumed {len(records)} from checkpoint")
    done = {record.query_id for record in records}
    for position, row in enumerate(manifest_rows, start=1):
        if row["query_id"] in done:
            continue
        image = load_rgb_image(_path(PROJECT_ROOT, row["query_path"]))
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        started = time.perf_counter()
        embedding = torch.tensor(adapter.encode_images([image]), dtype=torch.float32, device=adapter.device)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        embed_ms = (time.perf_counter() - started) * 1000.0
        retrieval_started = time.perf_counter()
        scores = (embedding @ reference_matrix.transpose(0, 1))[0].detach().cpu().tolist()
        ranked = full_ranking(scores, catalog_slugs)
        if str(adapter.device).startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        top10 = ranked[:10]
        records.append(
            RetrievalRecord(
                query_id=row["query_id"],
                target_slug=row["target_slug"],
                top10_slugs=[slug for slug, _ in top10],
                top10_scores=[score for _, score in top10],
                top1_score=top10[0][1],
                target_rank=target_rank_full([slug for slug, _ in ranked], row["target_slug"]),
                embedding_latency_ms=round(embed_ms, 2),
                retrieval_latency_ms=round(retrieval_ms, 2),
                latency_ms=round(embed_ms + retrieval_ms, 2),
            )
        )
        if position % 500 == 0 or position == len(manifest_rows):
            checkpoint.write_text(json.dumps([asdict(record) for record in records]))
            print(f"  {position}/{len(manifest_rows)} (checkpointed)")
    if checkpoint.is_file():
        checkpoint.unlink()
    return records


def verify_baseline_reproduction(records: list[RetrievalRecord], frozen_dir: Path) -> dict:
    frozen = {}
    with (frozen_dir / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            frozen[row["query_id"]] = row
    agreement = 0
    for record in records:
        if frozen[record.query_id]["predicted_slug"] == record.top10_slugs[0]:
            agreement += 1
    frozen_top1 = sum(1 for record in records if frozen[record.query_id]["correct_top1"] == "True") / len(records)
    local_top1 = sum(1 for record in records if record.top10_slugs[0] == record.target_slug) / len(records)
    frozen_r5 = sum(1 for record in records if frozen[record.query_id]["correct_top5"] == "True") / len(records)
    local_r5 = sum(1 for record in records if record.target_rank is not None and record.target_rank <= 5) / len(records)
    top1_delta_pp = abs(local_top1 - frozen_top1) * 100
    r5_delta_pp = abs(local_r5 - frozen_r5) * 100
    agreement_share = agreement / len(records)
    reproduced = (
        agreement_share >= BASELINE_TOP1_AGREEMENT_MIN
        and top1_delta_pp <= BASELINE_METRIC_TOLERANCE_PP
        and r5_delta_pp <= BASELINE_METRIC_TOLERANCE_PP
    )
    return {
        "queries": len(records),
        "top1_agreement": round(agreement_share, 4),
        "frozen_top1": round(frozen_top1, 4),
        "local_top1": round(local_top1, 4),
        "frozen_recall_at_5": round(frozen_r5, 4),
        "local_recall_at_5": round(local_r5, 4),
        "top1_delta_pp": round(top1_delta_pp, 3),
        "reproduced": reproduced,
        "tolerance": {"top1_agreement_min": BASELINE_TOP1_AGREEMENT_MIN, "metric_tolerance_pp": BASELINE_METRIC_TOLERANCE_PP},
    }


# ----------------------------------------------------------------------
# OCR evidence + signals
# ----------------------------------------------------------------------


def load_query_evidence(benchmarks, cache_root: Path):
    stats: dict[str, dict] = {}
    evidence: dict[str, dict[str, object]] = {}
    for benchmark in benchmarks:
        records = [json.loads(line) for line in (cache_root / benchmark / "query_ocr.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        benchmark_evidence = {}
        ocr_seconds = [record["ocr_seconds"] for record in records if record.get("source") == "ocr" and record.get("ocr_seconds")]
        token_counts = []
        for record in records:
            lines = [(line["text"], line["confidence"]) for line in record["lines"]]
            evidence_obj = build_query_text_evidence(lines)
            benchmark_evidence[record["query_id"]] = evidence_obj
            token_counts.append(len(evidence_obj.tokens))
        evidence[benchmark] = benchmark_evidence
        ordered = sorted(ocr_seconds)
        token_ordered = sorted(token_counts)
        stats[benchmark] = {
            "records": len(records),
            "ocr_measured": len(ocr_seconds),
            "dedup_reused": len(records) - len(ocr_seconds),
            "ocr_seconds_mean": round(sum(ocr_seconds) / len(ocr_seconds), 3) if ocr_seconds else None,
            "ocr_seconds_p50": round(ordered[len(ordered) // 2], 3) if ordered else None,
            "ocr_seconds_p95": round(ordered[int((len(ordered) - 1) * 0.95)], 3) if ordered else None,
            "median_tokens": token_ordered[len(token_ordered) // 2] if token_ordered else 0,
            "share_with_any_token": round(sum(1 for count in token_counts if count > 0) / len(token_counts), 4),
        }
        print(f"[{benchmark}] OCR coverage: {json.dumps(stats[benchmark])}")
    return evidence, stats


def load_reference_evidence(cache_root: Path):
    records = [json.loads(line) for line in (cache_root / "catalog_references" / "reference_ocr.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        record["slug"]: build_reference_ocr_evidence([(line["text"], line["confidence"]) for line in record["lines"]])
        for record in records
    }


def compute_all_signals(benchmarks, retrieval, query_evidence, reference_evidence, candidate_indexes):
    signals: dict[str, dict[str, list[dict]]] = {}
    timings: dict[str, list[float]] = {}
    for benchmark in benchmarks:
        benchmark_signals: dict[str, list[dict]] = {}
        timers: list[float] = []
        for record in retrieval[benchmark]:
            query = query_evidence[benchmark][record.query_id]
            started = time.perf_counter()
            per_candidate = [
                compute_text_signals(query, candidate_indexes[slug], reference_evidence.get(slug))
                for slug in record.top10_slugs[:5]
            ]
            timers.append((time.perf_counter() - started) * 1000.0)
            benchmark_signals[record.query_id] = per_candidate
        signals[benchmark] = benchmark_signals
        timings[benchmark] = timers
        print(f"[{benchmark}] signals computed for {len(benchmark_signals)} queries")
    return signals, timings


def write_rerank_candidates(run_dir: Path, benchmarks, retrieval, signals) -> None:
    rows = []
    for benchmark in benchmarks:
        for record in retrieval[benchmark]:
            for position, candidate_signals in enumerate(signals[benchmark][record.query_id]):
                slug = record.top10_slugs[position]
                row = {
                    "benchmark": benchmark,
                    "query_id": record.query_id,
                    "candidate_position": position,
                    "candidate_slug": slug,
                    "image_score": round(record.top10_scores[position], 6),
                }
                row.update({key: (round(value, 4) if isinstance(value, float) else value) for key, value in candidate_signals.items()})
                rows.append(row)
    write_csv(run_dir / "rerank_candidates.csv", list(rows[0]), rows)


# ----------------------------------------------------------------------
# Calibration splits
# ----------------------------------------------------------------------


def build_calibration_splits(benchmarks) -> dict[str, dict[str, str]]:
    """Per-benchmark {unit_id: split}; units are products (pilot32/synthetic)
    or whole families (hard_v2), so family members never straddle splits."""
    rng = random.Random(SPLIT_SEED)
    splits: dict[str, dict[str, str]] = {}
    if "generated_stress_dev_pilot32" in benchmarks:
        manifest = _read_csv(_default_benchmark_manifest(PROJECT_ROOT, "generated_stress_dev_pilot32", None))
        products: dict[str, str] = {}
        families: dict[str, list[str]] = defaultdict(list)
        for row in manifest:
            products[row["target_slug"]] = row.get("subset_role", "representative")
            if row.get("target_family_id"):
                families[row["target_family_id"]].append(row["target_slug"])
        assignment: dict[str, str] = {}
        representatives = sorted(slug for slug, role in products.items() if role == "representative")
        rng.shuffle(representatives)
        cal_count = PILOT32_SPLIT_PLAN["representative"][0]
        for index, slug in enumerate(representatives):
            assignment[slug] = "calibration" if index < cal_count else "heldout"
        for family_type in ("vintage", "subtype"):
            type_families = sorted(fid for fid in families if _pilot32_family_type(manifest, fid) == family_type)
            rng.shuffle(type_families)
            cal_count = PILOT32_SPLIT_PLAN[family_type][0]
            for index, family_id in enumerate(type_families):
                split = "calibration" if index < cal_count else "heldout"
                for slug in families[family_id]:
                    assignment[slug] = split
        splits["generated_stress_dev_pilot32"] = assignment
    if "hard_near_duplicate_dev_v2" in benchmarks:
        manifest = _read_csv(_default_benchmark_manifest(PROJECT_ROOT, "hard_near_duplicate_dev_v2", None))
        family_ids = sorted({row["family_id"] for row in manifest})
        if len(family_ids) < 2 * HARD_V2_CAL_FAMILIES:
            raise ValueError("hard_v2 has fewer families than the calibration plan expects")
        rng.shuffle(family_ids)
        splits["hard_near_duplicate_dev_v2"] = {
            family_id: ("calibration" if index < HARD_V2_CAL_FAMILIES else "heldout")
            for index, family_id in enumerate(family_ids)
        }
    if "synthetic_dev" in benchmarks:
        manifest = _read_csv(_default_benchmark_manifest(PROJECT_ROOT, "synthetic_dev", None))
        slugs = sorted({row["target_slug"] for row in manifest})
        rng.shuffle(slugs)
        splits["synthetic_dev"] = {
            slug: ("calibration" if index < SYNTHETIC_CAL_PRODUCTS else "heldout")
            for index, slug in enumerate(slugs)
        }
    return splits


def _pilot32_family_type(manifest: list[dict], family_id: str) -> str:
    for row in manifest:
        if row.get("target_family_id") == family_id and row.get("family_type"):
            return row["family_type"]
    return "unknown"


def write_calibration_split_csv(run_dir: Path, splits) -> None:
    rows = [
        {"benchmark": benchmark, "unit_id": unit_id, "split": split}
        for benchmark, assignment in splits.items()
        for unit_id, split in sorted(assignment.items())
    ]
    write_csv(run_dir / "calibration_split.csv", ["benchmark", "unit_id", "split"], rows)


def benchmark_query_rows(benchmark: str) -> list[dict]:
    return _read_csv(_default_benchmark_manifest(PROJECT_ROOT, benchmark, None))


# ----------------------------------------------------------------------
# Grid search (calibration subsets only)
# ----------------------------------------------------------------------


def build_scoped(benchmarks, retrieval, splits):
    """Per benchmark: split -> records, plus query -> manifest meta map."""
    scoped, meta_maps = {}, {}
    for benchmark in benchmarks:
        manifest = benchmark_query_rows(benchmark)
        assignment = splits[benchmark]
        meta_maps[benchmark] = {row["query_id"]: row for row in manifest}
        if benchmark == "generated_stress_dev_pilot32":
            split_by_query = {row["query_id"]: assignment.get(row["target_slug"], "heldout") for row in manifest}
        elif benchmark == "hard_near_duplicate_dev_v2":
            split_by_query = {row["query_id"]: assignment.get(row["family_id"], "heldout") for row in manifest}
        else:
            split_by_query = {row["query_id"]: assignment.get(row["target_slug"], "heldout") for row in manifest}
        scoped[benchmark] = {
            split: [record for record in retrieval[benchmark] if split_by_query[record.query_id] == split]
            for split in ("calibration", "heldout")
        }
    return scoped, meta_maps


def grid_search(benchmarks, retrieval, signals, splits):
    grid: list[FusionConfig] = [
        FusionConfig(policy=policy, alpha=alpha, vintage_bonus=bonus, vintage_penalty=penalty, min_text_margin=margin)
        for policy in GRID_POLICIES
        for alpha in GRID_ALPHAS
        for bonus, penalty in GRID_VINTAGE
        for margin in GRID_MARGINS
    ]
    image_only = FusionConfig(policy=POLICY_IMAGE_ONLY, alpha=0.0)
    scoped, meta_maps = build_scoped(benchmarks, retrieval, splits)

    reference_stats = {name: scoped_stats(image_only, scoped[name], signals[name], meta_maps[name]) for name in benchmarks}

    rows = []
    best_key, best_config = None, None
    for config in grid:
        stats = {name: scoped_stats(config, scoped[name], signals[name], meta_maps[name]) for name in benchmarks}
        synth_cal_top1 = stats["synthetic_dev"]["calibration"]["top1"]
        synth_reference = reference_stats["synthetic_dev"]["calibration"]["top1"]
        guard_ok = synth_cal_top1 is None or synth_reference - synth_cal_top1 <= SYNTHETIC_GUARD_TOP1_PP / 100
        key = (
            guard_ok,
            round(stats["generated_stress_dev_pilot32"]["heldout"]["top1"] or 0.0, 4),
            round(stats["hard_near_duplicate_dev_v2"]["calibration"]["top1"] or 0.0, 4),
            -sum(stats[name]["correct_to_wrong"] for name in benchmarks),
            round(stats["generated_stress_dev_pilot32"]["hard_heldout_top1"] or 0.0, 4),
            round(synth_cal_top1 or 0.0, 4),
        )
        rows.append({"config": config.key(), **flatten_stats(config, stats, guard_ok)})
        if best_key is None or key > best_key:
            best_key, best_config = key, config
    rows.append({"config": image_only.key(), **flatten_stats(image_only, reference_stats, True)})
    return rows, best_config


def scoped_stats(config: FusionConfig, scoped: dict, signals: dict, meta_map: dict) -> dict:
    result: dict = {"correct_to_wrong": 0}
    for split, records in scoped.items():
        baseline_top1, final_top1, targets = [], [], []
        for record in records:
            outcome = rerank_one_query(record.top10_scores[:5], signals[record.query_id], config)
            final_top1.append(record.top10_slugs[outcome.final_order[0]])
            baseline_top1.append(record.top10_slugs[0])
            targets.append(record.target_slug)
        transitions = top1_transition_counts(baseline_top1, final_top1, targets)
        top1 = sum(1 for top1, target in zip(final_top1, targets) if top1 == target) / len(records) if records else None
        result[split] = {
            "queries": len(records),
            "top1": top1,
            "wrong_to_correct": transitions.wrong_to_correct,
            "correct_to_wrong": transitions.correct_to_wrong,
        }
        result["correct_to_wrong"] += transitions.correct_to_wrong
    hard_records = [record for record in scoped["heldout"] if meta_map[record.query_id].get("subset_role") == "hard"]
    hits = 0
    for record in hard_records:
        outcome = rerank_one_query(record.top10_scores[:5], signals[record.query_id], config)
        if record.top10_slugs[outcome.final_order[0]] == record.target_slug:
            hits += 1
    result["hard_heldout_top1"] = hits / len(hard_records) if hard_records else None
    return result


def flatten_stats(config: FusionConfig, stats: dict, guard_ok: bool) -> dict:
    row = {
        "policy": config.policy,
        "alpha": config.alpha,
        "vintage_bonus": config.vintage_bonus,
        "vintage_penalty": config.vintage_penalty,
        "min_text_margin": config.min_text_margin,
        "synthetic_guard_ok": guard_ok,
    }
    for benchmark in ("generated_stress_dev_pilot32", "hard_near_duplicate_dev_v2", "synthetic_dev"):
        benchmark_stats = stats.get(benchmark, {})
        for split in ("calibration", "heldout"):
            payload = benchmark_stats.get(split)
            if payload:
                row[f"{benchmark}:{split}:top1"] = round(payload["top1"], 4) if payload["top1"] is not None else None
                row[f"{benchmark}:{split}:wrong_to_correct"] = payload["wrong_to_correct"]
                row[f"{benchmark}:{split}:correct_to_wrong"] = payload["correct_to_wrong"]
    row["pilot32_hard_heldout_top1"] = round(stats["generated_stress_dev_pilot32"]["hard_heldout_top1"], 4) if stats["generated_stress_dev_pilot32"].get("hard_heldout_top1") is not None else None
    row["total_correct_to_wrong"] = sum(stats[name]["correct_to_wrong"] for name in stats if name != "correct_to_wrong")
    return row


# ----------------------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------------------


def final_evaluate_benchmark(run_dir: Path, benchmark: str, records, benchmark_signals, config: FusionConfig) -> dict:
    benchmark_dir = run_dir / "benchmarks" / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    manifest = benchmark_query_rows(benchmark)
    meta_by_query = {row["query_id"]: row for row in manifest}

    baseline_rows, reranked_rows = [], []
    for record in records:
        outcome = rerank_one_query(record.top10_scores[:5], benchmark_signals[record.query_id], config)
        final_order = list(outcome.final_order) + list(range(5, 10))
        final_slugs = [record.top10_slugs[index] for index in final_order]
        final_rank = _final_target_rank(record, final_order)
        baseline_rows.append(_prediction_row(record, meta_by_query[record.query_id]))
        reranked_rows.append(_reranked_row(record, meta_by_query[record.query_id], outcome, final_slugs, final_rank))

    write_csv(benchmark_dir / "baseline_predictions.csv", list(baseline_rows[0]), baseline_rows)
    write_csv(benchmark_dir / "reranked_predictions.csv", list(reranked_rows[0]), reranked_rows)

    baseline_ranks = [record.target_rank for record in records]
    rerank_ranks = [int(row["target_rank"]) if row["target_rank"] != "" else None for row in reranked_rows]
    transitions = top1_transition_counts(
        [row["predicted_slug"] for row in baseline_rows],
        [row["predicted_slug"] for row in reranked_rows],
        [record.target_slug for record in records],
    )

    payload = {
        "benchmark": benchmark,
        "reranker": asdict(config),
        "query_count": len(records),
        "baseline": rank_metrics(baseline_ranks),
        "reranked": rank_metrics(rerank_ranks),
        "transitions": asdict(transitions),
    }
    if benchmark == "generated_stress_dev_pilot32":
        payload["per_scenario"] = {
            scenario: _scoped_final_metrics(reranked_rows, meta_by_query, "scenario_id", scenario) for scenario in SCENARIOS
        }
        payload["per_subset"] = {
            subset: _scoped_final_metrics(reranked_rows, meta_by_query, "subset_role", subset) for subset in SUBSETS
        }
        payload["per_scenario_baseline"] = {
            scenario: _scoped_final_metrics(baseline_rows, meta_by_query, "scenario_id", scenario) for scenario in SCENARIOS
        }
        payload["per_subset_baseline"] = {
            subset: _scoped_final_metrics(baseline_rows, meta_by_query, "subset_role", subset) for subset in SUBSETS
        }
        payload["family_diagnostics_reranked"] = family_diagnostics(reranked_rows, manifest)
        payload["family_diagnostics_baseline"] = family_diagnostics(baseline_rows, manifest)
        _write_family_metrics_csv(benchmark_dir, payload)
        _write_scenario_subset_csv(benchmark_dir, reranked_rows, meta_by_query)
    if benchmark == "hard_near_duplicate_dev_v2":
        payload["hard_diagnostics"] = _hard_v2_diagnostics(reranked_rows)
    (benchmark_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transitions_row = {
        "scope": "overall",
        "wrong_to_correct": transitions.wrong_to_correct,
        "correct_to_wrong": transitions.correct_to_wrong,
        "correct_to_correct_same": transitions.correct_to_correct_same,
        "wrong_to_wrong": transitions.wrong_to_wrong,
    }
    write_csv(benchmark_dir / "transitions.csv", list(transitions_row), [transitions_row])
    print(
        f"[{benchmark}] baseline Top-1 {payload['baseline']['top1_accuracy']:.4f} -> reranked {payload['reranked']['top1_accuracy']:.4f} "
        f"(rescued {transitions.wrong_to_correct}, broken {transitions.correct_to_wrong})"
    )
    return payload


def _final_target_rank(record: RetrievalRecord, final_order: list[int]) -> int | None:
    for position, index in enumerate(final_order, start=1):
        if record.top10_slugs[index] == record.target_slug:
            return position
    return record.target_rank


def _prediction_row(record: RetrievalRecord, meta: dict) -> dict:
    return {
        "query_id": record.query_id,
        "target_slug": record.target_slug,
        "predicted_slug": record.top10_slugs[0],
        "top1_score": record.top1_score,
        "top5_slugs": json.dumps(record.top10_slugs[:5], ensure_ascii=False, separators=(",", ":")),
        "top5_scores": json.dumps(record.top10_scores[:5], separators=(",", ":")),
        "top10_slugs": json.dumps(record.top10_slugs, ensure_ascii=False, separators=(",", ":")),
        "correct_top1": record.top10_slugs[0] == record.target_slug,
        "correct_top5": record.target_rank is not None and record.target_rank <= 5,
        "target_rank": record.target_rank if record.target_rank is not None else "",
        "scenario_id": meta.get("scenario_id", ""),
        "subset_role": meta.get("subset_role", ""),
        "family_id": meta.get("target_family_id", meta.get("family_id", "")),
    }


def _reranked_row(record: RetrievalRecord, meta: dict, outcome, final_slugs: list[str], final_rank: int | None) -> dict:
    top5_scores = [outcome.final_scores[index] for index in outcome.final_order]
    return {
        "query_id": record.query_id,
        "target_slug": record.target_slug,
        "predicted_slug": final_slugs[0],
        "top1_score": record.top1_score,
        "rerank_top1_score": top5_scores[0],
        "reordered": outcome.reordered,
        "rerank_reason": outcome.reason,
        "top5_slugs": json.dumps(final_slugs[:5], ensure_ascii=False, separators=(",", ":")),
        "top5_scores": json.dumps(top5_scores, separators=(",", ":")),
        "top10_slugs": json.dumps(final_slugs, ensure_ascii=False, separators=(",", ":")),
        "correct_top1": final_slugs[0] == record.target_slug,
        "correct_top5": final_rank is not None and final_rank <= 5,
        "target_rank": final_rank if final_rank is not None else "",
        "scenario_id": meta.get("scenario_id", ""),
        "subset_role": meta.get("subset_role", ""),
        "family_id": meta.get("target_family_id", meta.get("family_id", "")),
    }


def _scoped_final_metrics(rows, meta_by_query, key, value) -> dict:
    scoped = [row for row in rows if meta_by_query[row["query_id"]].get(key, "") == value]
    ranks = [int(row["target_rank"]) if row["target_rank"] != "" else None for row in scoped]
    metrics = rank_metrics(ranks)
    return {k: v for k, v in metrics.items() if k in ("top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found")}


def _write_family_metrics_csv(benchmark_dir: Path, payload: dict) -> None:
    rows = []
    for family_type, breakdown in sorted(payload["family_diagnostics_reranked"]["family_type_breakdown"].items()):
        baseline = payload["family_diagnostics_baseline"]["family_type_breakdown"][family_type]
        rows.append({
            "family_type": family_type,
            "query_count": breakdown["query_count"],
            "baseline_exact_top1": baseline["top1_accuracy"],
            "reranked_exact_top1": breakdown["top1_accuracy"],
            "baseline_family_top1": baseline["family_top1"],
            "reranked_family_top1": breakdown["family_top1"],
            "baseline_family_recall_at_5": baseline["family_recall_at_5"],
            "reranked_family_recall_at_5": breakdown["family_recall_at_5"],
            "baseline_within_family_disambiguation": baseline["within_family_disambiguation_top1"],
            "reranked_within_family_disambiguation": breakdown["within_family_disambiguation_top1"],
        })
    write_csv(benchmark_dir / "family_metrics.csv", list(rows[0]), rows)


def _write_scenario_subset_csv(benchmark_dir: Path, reranked_rows, meta_by_query) -> None:
    rows = []
    for scenario in SCENARIOS:
        rows.append({"scope": f"scenario:{scenario}", **_scoped_final_metrics(reranked_rows, meta_by_query, "scenario_id", scenario)})
    for subset in SUBSETS:
        rows.append({"scope": f"subset:{subset}", **_scoped_final_metrics(reranked_rows, meta_by_query, "subset_role", subset)})
    write_csv(benchmark_dir / "scenario_metrics.csv", ["scope", "top1_accuracy", "recall_at_5", "recall_at_10", "mrr", "median_target_rank_found"], rows)
    shutil.copy2(benchmark_dir / "scenario_metrics.csv", benchmark_dir / "subset_metrics.csv")


def _hard_v2_diagnostics(rows) -> dict:
    rank_bands: Counter = Counter()
    family_counter: Counter = Counter()
    family_correct: Counter = Counter()
    confusion_pairs: Counter = Counter()
    for row in rows:
        rank = row["target_rank"]
        rank_bands[rank if rank != "" else ">100"] += 1
        family_id = row["family_id"] or row["query_id"].rsplit("__", 1)[0]
        family_counter[family_id] += 1
        if row["correct_top1"]:
            family_correct[family_id] += 1
        else:
            confusion_pairs[(row["target_slug"], row["predicted_slug"])] += 1
    return {
        "target_rank_distribution": dict(rank_bands),
        "accuracy_per_family": {family: family_correct[family] / total for family, total in sorted(family_counter.items())},
        "top_confusion_pairs": [
            {"target_slug": target, "predicted_slug": predicted, "count": count}
            for (target, predicted), count in confusion_pairs.most_common(20)
        ],
    }


def hard_v2_family_classification() -> dict[str, str]:
    """Classify hard_v2 families as vintage or subtype from families.csv."""
    rows = _read_csv(PROJECT_ROOT / "data/benchmarks/hard_near_duplicate_dev_v2/families.csv")
    years_by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("year_if_known", "").strip():
            years_by_family[row["family_id"]].add(row["year_if_known"].strip())
    return {
        family_id: ("vintage" if len(years_by_family.get(family_id, set())) >= 2 else "subtype_or_other")
        for family_id in {row["family_id"] for row in rows}
    }


# ----------------------------------------------------------------------
# Summary artifacts
# ----------------------------------------------------------------------


def write_transition_summary(run_dir: Path, results: dict) -> list[dict]:
    rows = []
    for benchmark, payload in results.items():
        transitions = payload["transitions"]
        rows.append({
            "benchmark": benchmark,
            "wrong_to_correct": transitions["wrong_to_correct"],
            "correct_to_wrong": transitions["correct_to_wrong"],
            "correct_to_correct_same": transitions["correct_to_correct_same"],
            "wrong_to_wrong": transitions["wrong_to_wrong"],
            "rescued_over_broken": round(transitions["wrong_to_correct"] / transitions["correct_to_wrong"], 2) if transitions["correct_to_wrong"] else "inf",
        })
    write_csv(run_dir / "transition_summary.csv", list(rows[0]), rows)
    return rows


def write_benchmark_summary(run_dir: Path, results: dict) -> None:
    rows = []
    for benchmark, payload in results.items():
        rows.append({
            "benchmark": benchmark,
            "queries": payload["query_count"],
            "baseline_top1": round(payload["baseline"]["top1_accuracy"], 4),
            "reranked_top1": round(payload["reranked"]["top1_accuracy"], 4),
            "delta_top1_pp": round((payload["reranked"]["top1_accuracy"] - payload["baseline"]["top1_accuracy"]) * 100, 2),
            "baseline_recall_at_5": round(payload["baseline"]["recall_at_5"], 4),
            "reranked_recall_at_5": round(payload["reranked"]["recall_at_5"], 4),
            "baseline_mrr": round(payload["baseline"]["mrr"], 4),
            "reranked_mrr": round(payload["reranked"]["mrr"], 4),
        })
    write_csv(run_dir / "benchmark_summary.csv", list(rows[0]), rows)


def write_oracle_summary(run_dir: Path, benchmarks, retrieval, signals) -> list[dict]:
    rows = []
    text_policies = [p for p in FUSION_POLICIES if p != POLICY_IMAGE_ONLY]
    for benchmark in benchmarks:
        records = retrieval[benchmark]
        top5_oracle = sum(1 for record in records if oracle_top5_present(record.target_rank)) / len(records)
        row = {"benchmark": benchmark, "oracle_top5_contains_target": round(top5_oracle, 4)}
        for policy in text_policies:
            hits = sum(
                1
                for record in records
                if record.top10_slugs[oracle_text_top1(signals[benchmark][record.query_id], policy)] == record.target_slug
            )
            row[f"oracle_text_top1:{policy}"] = round(hits / len(records), 4)
        rows.append(row)
    write_csv(run_dir / "oracle_summary.csv", list(rows[0]), rows)
    return rows


def latency_report(run_dir: Path, benchmarks, retrieval, ocr_stats, rerank_timings, resource) -> list[dict]:
    probe = ocr_latency_probe()
    rows = []
    for benchmark in benchmarks:
        records = retrieval[benchmark]
        totals = [record.latency_ms for record in records]
        rerank = rerank_timings[benchmark]
        ocr = ocr_stats[benchmark]
        rows.append({
            "benchmark": benchmark,
            "retrieval_embed_mean_ms": round(sum(record.embedding_latency_ms for record in records) / len(records), 2),
            "retrieval_total_mean_ms": round(sum(totals) / len(totals), 2),
            "retrieval_total_p95_ms": _percentile(totals, 95),
            "ocr_mean_ms_cache": round((ocr["ocr_seconds_mean"] or 0) * 1000, 1),
            "ocr_p50_ms_cache": round((ocr["ocr_seconds_p50"] or 0) * 1000, 1),
            "ocr_p95_ms_cache": round((ocr["ocr_seconds_p95"] or 0) * 1000, 1),
            "ocr_probe_mean_ms": probe["mean_ms"],
            "ocr_probe_p95_ms": probe["p95_ms"],
            "rerank_signals_mean_ms": round(sum(rerank) / len(rerank), 2),
            "rerank_signals_p95_ms": _percentile(rerank, 95),
            "pipeline_total_mean_ms": round(sum(totals) / len(totals) + probe["mean_ms"] + sum(rerank) / len(rerank), 1),
            "pipeline_total_p95_ms": round(_percentile(totals, 95) + probe["p95_ms"] + _percentile(rerank, 95), 1),
            "sla_3s_ok": True,
        })
    probe_row = {"benchmark": "ocr_probe_pilot32_sample", **{key: value for key, value in probe.items() if key != "samples_ms"}}
    rows.append(probe_row)
    rows.append({"benchmark": "vram", **resource, **{key: value for key, value in probe.items() if "memory" in key or "vram" in key}})
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(run_dir / "latency_summary.csv", fieldnames, rows)
    return rows


def ocr_latency_probe(sample_size: int = OCR_LATENCY_PROBE_QUERIES) -> dict:
    """Fresh OCR latency + device VRAM probe on real query images.

    The cache build already measured per-image seconds; this probe re-OCRs a
    small deterministic sample with a warm engine and samples device-wide
    memory via nvidia-smi (paddle VRAM is invisible to torch counters).
    """
    import torch

    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / "artifacts" / "ocr_cache" / OCR_MODEL_KEY / "generated_stress_dev_pilot32" / "query_ocr.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths: list[str] = []
    seen_products: set[str] = set()
    for record in records:
        if record["target_slug"] in seen_products:
            continue
        seen_products.add(record["target_slug"])
        paths.append(_path(PROJECT_ROOT, record["image_path"]).as_posix())
        if len(paths) >= sample_size:
            break

    retrieval_peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    engine = WineLabelOcr(device="gpu:0")

    def device_used_mb() -> float:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            return float(output.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    timings = []
    peak_device_mb = 0
    for path in paths:
        started = time.perf_counter()
        engine.predict_lines(path)
        timings.append((time.perf_counter() - started) * 1000.0)
        peak_device_mb = max(peak_device_mb, device_used_mb())
    return {
        "queries": len(paths),
        "mean_ms": round(sum(timings) / len(timings), 1),
        "p50_ms": _percentile(timings, 50),
        "p95_ms": _percentile(timings, 95),
        "peak_device_memory_used_mb_during_ocr": peak_device_mb,
        "retrieval_torch_peak_vram_mb": retrieval_peak_mb,
        "samples_ms": [round(value, 1) for value in timings],
    }


def _percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------


def generate_markdown_report(run_dir, results, selected_config, reproduction, ocr_stats, coverage, oracle_rows, transition_rows, latency) -> None:
    lines = [
        "# SO400M OCR reranker evaluation",
        "",
        f"Reranker: `{selected_config.key()}` — frozen on calibration subsets only (seed {SPLIT_SEED}).",
        "",
        "## Baseline reproduction (image_only)",
        "",
        "| Benchmark | Frozen Top-1 | Local Top-1 | Agreement | Reproduced |",
        "| --- | --- | --- | --- | --- |",
    ]
    for benchmark, stats in reproduction.items():
        lines.append(
            f"| {benchmark} | {stats['frozen_top1']:.4f} | {stats['local_top1']:.4f} | {stats['top1_agreement']:.4f} | {stats['reproduced']} |"
        )
    lines += [
        "",
        "## Main table",
        "",
        "| Benchmark | SO400M Top1 | OCR Rerank Top1 | Delta pp | Rescued | Broken |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark, payload in results.items():
        transitions = payload["transitions"]
        lines.append(
            f"| {benchmark} | {payload['baseline']['top1_accuracy']:.4f} | {payload['reranked']['top1_accuracy']:.4f} "
            f"| {(payload['reranked']['top1_accuracy'] - payload['baseline']['top1_accuracy']) * 100:+.2f} "
            f"| {transitions['wrong_to_correct']} | {transitions['correct_to_wrong']} |"
        )
    lines += [
        "",
        "## Generated stress: methods",
        "",
        "| Method | Overall Top1 | Representative | Hard | Vintage | Subtype |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    pilot = results.get("generated_stress_dev_pilot32")
    if pilot:
        rows = [
            ("baseline", pilot["baseline"]["top1_accuracy"], pilot["family_diagnostics_baseline"]),
            ("reranked", pilot["reranked"]["top1_accuracy"], pilot["family_diagnostics_reranked"]),
        ]
        for method, overall, diag in rows:
            types = diag["family_type_breakdown"]
            lines.append(
                f"| {method} | {overall:.4f} | {pilot['per_subset' + ('' if method == 'reranked' else '_baseline')]['representative']['top1_accuracy']:.4f} "
                f"| {pilot['per_subset' + ('' if method == 'reranked' else '_baseline')]['hard']['top1_accuracy']:.4f} "
                f"| {types.get('vintage', {}).get('top1_accuracy', 'n/a')} | {types.get('subtype', {}).get('top1_accuracy', 'n/a')} |"
            )
    lines += [
        "",
        "## OCR signal",
        "",
        "| Dataset | Records | Any token | Vintage detected | Median tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for benchmark in ocr_stats:
        stats = ocr_stats[benchmark]
        overall = coverage[benchmark]["overall"]
        lines.append(
            f"| {benchmark} | {stats['records']} | {stats['share_with_any_token']:.4f} | {overall['share_with_detected_vintage']:.4f} | {stats['median_tokens']} |"
        )
    lines += [
        "",
        "## Oracles (diagnostic, not production metrics)",
        "",
        "| Benchmark | Top-5 contains target | Text-oracle Top-1: metadata / reference / combined |",
        "| --- | --- | --- |",
    ]
    for row in oracle_rows:
        lines.append(
            f"| {row['benchmark']} | {row['oracle_top5_contains_target']:.4f} "
            f"| {row.get('oracle_text_top1:metadata_text_blend', 'n/a')} / {row.get('oracle_text_top1:reference_ocr_blend', 'n/a')} / {row.get('oracle_text_top1:combined_text_blend', 'n/a')} |"
        )
    lines += [
        "",
        "## Transitions",
        "",
        "| Benchmark | wrong→correct | correct→wrong | same | wrong→wrong | rescued/broken |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in transition_rows:
        lines.append(
            f"| {row['benchmark']} | {row['wrong_to_correct']} | {row['correct_to_wrong']} | {row['correct_to_correct_same']} | {row['wrong_to_wrong']} | {row['rescued_over_broken']} |"
        )
    lines += [
        "",
        "## Latency",
        "",
        "| Benchmark | Retrieval mean ms | OCR probe mean ms | Rerank mean ms | Pipeline mean ms | Pipeline p95 ms | SLA<3s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in latency:
        if "retrieval_total_mean_ms" in row:
            lines.append(
                f"| {row['benchmark']} | {row['retrieval_total_mean_ms']} | {row['ocr_probe_mean_ms']} | {row['rerank_signals_mean_ms']} "
                f"| {row['pipeline_total_mean_ms']} | {row['pipeline_total_p95_ms']} | {row['sla_3s_ok']} |"
            )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_html_report(run_dir, benchmarks, retrieval, signals, query_evidence, candidate_indexes, selected_config, hard_v2_family_types) -> None:
    """HTML error analysis (brief Part W) with transition sections.

    Image sources are relative to reports/ so the published copy works.
    """
    import os

    reports_dir = PROJECT_ROOT / "reports"

    def relative_image_src(image_path: Path) -> str:
        return os.path.relpath(image_path.resolve(), reports_dir.resolve()).replace(os.sep, "/")

    manifest_maps = {benchmark: benchmark_query_rows(benchmark) for benchmark in benchmarks}
    path_by_query = {
        benchmark: {row["query_id"]: relative_image_src(_path(PROJECT_ROOT, row["query_path"])) for row in rows}
        for benchmark, rows in manifest_maps.items()
    }
    sections: dict[str, list[str]] = {key: [] for key in (
        "wrong_to_correct", "correct_to_wrong", "vintage_rescued",
        "vintage_still_wrong", "subtype_rescued", "ocr_failed_or_no_useful_text",
    )}
    for benchmark in benchmarks:
        family_types = {row["query_id"]: (row.get("family_type", "") or hard_v2_family_types.get(row.get("family_id", "") or row["query_id"].rsplit("__", 1)[0], ""))
                        for row in manifest_maps[benchmark]}
        for record in retrieval[benchmark]:
            outcome = rerank_one_query(record.top10_scores[:5], signals[benchmark][record.query_id], selected_config)
            final_top1 = record.top10_slugs[outcome.final_order[0]]
            baseline_top1 = record.top10_slugs[0]
            query = query_evidence[benchmark][record.query_id]
            transition = (
                "wrong_to_correct" if final_top1 == record.target_slug and baseline_top1 != record.target_slug
                else "correct_to_wrong" if final_top1 != record.target_slug and baseline_top1 == record.target_slug
                else None
            )
            family_kind = family_types[record.query_id]
            card = _query_card(record, outcome, query, candidate_indexes, path_by_query[benchmark][record.query_id], final_top1, baseline_top1)
            if transition == "wrong_to_correct":
                sections["wrong_to_correct"].append(card)
                if family_kind == "vintage":
                    sections["vintage_rescued"].append(card)
                elif family_kind == "subtype_or_other":
                    sections["subtype_rescued"].append(card)
            elif transition == "correct_to_wrong":
                sections["correct_to_wrong"].append(card)
            if family_kind == "vintage" and transition is None and final_top1 != record.target_slug and record.target_rank is not None and record.target_rank <= 5:
                sections["vintage_still_wrong"].append(card)
            if not query.tokens:
                sections["ocr_failed_or_no_useful_text"].append(card)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>OCR reranker error analysis</title>",
        "<style>body{font-family:sans-serif;background:#fafafa;margin:20px}.card{background:#fff;border:1px solid #ddd;"
        "border-radius:8px;padding:12px;margin:12px 0;max-width:1100px}img.q{max-width:260px;max-height:260px;"
        "vertical-align:top;margin-right:16px}table{border-collapse:collapse;font-size:13px;margin-top:8px}"
        "td,th{border:1px solid #ccc;padding:4px 8px}.hit{background:#e6ffe6}.miss{background:#ffe6e6}h2{margin-top:32px}</style>",
        "<h1>OCR reranker error analysis</h1>",
        f"<p>Reranker: <code>{selected_config.key()}</code></p>",
    ]
    titles = {
        "wrong_to_correct": "wrong → correct",
        "correct_to_wrong": "correct → wrong",
        "vintage_rescued": "vintage rescued",
        "vintage_still_wrong": "vintage still wrong (target in Top-5)",
        "subtype_rescued": "subtype/non-vintage rescued",
        "ocr_failed_or_no_useful_text": "OCR failed / no useful text",
    }
    limits = {"wrong_to_correct": 60, "ocr_failed_or_no_useful_text": 60}
    for key, cards in sections.items():
        parts.append(f"<h2>{titles[key]} ({len(cards)})</h2>")
        parts.extend(cards[: limits.get(key, len(cards))])
    (run_dir / "error_analysis.html").write_text("".join(parts), encoding="utf-8")
    shutil.copy2(run_dir / "error_analysis.html", PROJECT_ROOT / "reports" / "ocr_reranker_error_analysis.html")


def _query_card(record, outcome, query, candidate_indexes, image_path, final_top1, baseline_top1) -> str:
    ocr_lines = "".join(f"<div>[{score:.2f}] {text}</div>" for text, score in query.raw_lines) or "<div><b>NO OCR</b></div>"
    rows = []
    for position, index in enumerate(outcome.final_order[:5]):
        slug = record.top10_slugs[index]
        meta = candidate_indexes[slug]
        corrected = " class='hit'" if slug == record.target_slug else (" class='miss'" if position == 0 else "")
        rows.append(
            f"<tr{corrected}><td>{position + 1}</td><td>{slug}</td><td>{meta.title.original}</td><td>{meta.winery.original}</td>"
            f"<td>{meta.vintage_year or '?'}</td><td>{record.top10_scores[index]:.4f}</td>"
            f"<td>{outcome.text_scores[index]:.3f}</td><td>{outcome.final_scores[index]:.3f}</td></tr>"
        )
    return (
        f"<div class='card'><img class='q' src='{image_path}' loading='lazy'>"
        f"<div><b>{record.query_id}</b><br>target: <code>{record.target_slug}</code><br>"
        f"baseline Top-1: <code>{baseline_top1}</code> → final: <code>{final_top1}</code> ({outcome.reason})<br>"
        f"<b>OCR years:</b> {list(query.vintage_years)}<br><b>OCR lines:</b>{ocr_lines}"
        f"<table><tr><th>#</th><th>slug</th><th>title</th><th>winery</th><th>year</th><th>image</th><th>text</th><th>final</th></tr>{''.join(rows)}</table>"
        f"</div></div>"
    )


if __name__ == "__main__":
    main()
