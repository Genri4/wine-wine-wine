#!/usr/bin/env python3
"""OCR + reranker bake-off over the frozen SO400M Top-5.

Candidate generation is fixed: siglip2_so400m_384 Top-5 from the previous
milestone's retrieval dump. OCR never changes the candidate set; all
rerankers only permute the Top-5.

Matrix (brief Part 18): {current_eslav, cyrillic, paddleocr_vl} x
{current blend, structured LR, BGE fusion} with the documented prunes.
Training uses calibration units only (product-level generated, family-level
hard_v2, product-level synthetic); held-out halves + generated held-out are
the clean evaluation; full-set numbers are reported with disclosure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.crop_diagnostics import rank_metrics  # noqa: E402
from recognition.gated_strategy import seeded_product_sample  # noqa: E402
from recognition.ocr_reranker import (  # noqa: E402
    FusionConfig,
    normalize_image_scores,
    rerank_one_query,
    top1_transition_counts,
)
from recognition.reporting import write_csv  # noqa: E402
from recognition.structured_reranker import (  # noqa: E402
    BgeReranker,
    LogisticReranker,
    build_feature_row,
    build_idf,
    candidate_document_text,
    fit_logistic_reranker,
)
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
OCR_KEYS = {
    "current_eslav": "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec",
    "cyrillic": "paddleocr3.7_ppocrv5_server_det_cyrillic_v5_mobile_rec",
    "paddleocr_vl": "paddleocr3.7_paddleocr_vl_0.9B",
}
BASELINE_RUN = PROJECT_ROOT / "artifacts" / "experiments" / "so400m_ocr_reranker_20260920T193925Z"
SPLIT_SEED = 20260920
SYNTHETIC_CAL_PRODUCTS = 400
HARD_V2_CAL_FAMILIES = 128
PILOT32_CAL_PRODUCTS = 16
BGE_WEIGHTS = (0.1, 0.2, 0.3)
FROZEN_BLEND = FusionConfig(
    policy="reference_ocr_blend", alpha=0.30, vintage_bonus=0.05, vintage_penalty=0.05, min_text_margin=0.05
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR + reranker bake-off")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-bge", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog_meta = json.loads(_path(project_root, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, catalog_by_slug = _catalog_items(
        project_root, _read_csv(_path(project_root, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    candidate_indexes = {item.item_id: build_candidate_text_index(item.metadata) for item in catalog_items}
    catalog_by_id = {item.item_id: item for item in catalog_items}

    manifests = {benchmark: _read_csv(_default_benchmark_manifest(project_root, benchmark, None)) for benchmark in BENCHMARKS}

    # ------------------------------------------------------------------
    # Retrieval Top-5 (frozen from the previous milestone's dump)
    # ------------------------------------------------------------------
    retrieval: dict[str, dict[str, dict]] = {}
    for benchmark in BENCHMARKS:
        rows = _read_csv(BASELINE_RUN / "benchmarks" / benchmark / "baseline_predictions.csv")
        retrieval[benchmark] = {row["query_id"]: row for row in rows}
        assert len(rows) == len(manifests[benchmark]), benchmark

    # ------------------------------------------------------------------
    # Group-aware splits (read-only reuse of the frozen calibration builder)
    # ------------------------------------------------------------------
    splits = build_splits(manifests)
    (run_dir / "calibration_split.csv").write_text(
        "benchmark,unit_id,split\n"
        + "".join(f"{benchmark},{unit},{side}\n" for benchmark, units in sorted(splits.items()) for unit, side in sorted(units.items())),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Per-OCR evidence + signal rows (cached incremental jsonl)
    # ------------------------------------------------------------------
    reference_evidence: dict[str, dict[str, object]] = {}
    idf_by_ocr: dict[str, dict[str, float]] = {}
    for ocr_name, cache_key in OCR_KEYS.items():
        ref_path = project_root / "artifacts" / "ocr_cache" / cache_key / "catalog_references" / "reference_ocr.jsonl"
        records = load_jsonl(ref_path)
        evidence_map = {
            record["slug"]: build_reference_ocr_evidence([(line["text"], line["confidence"]) for line in record["lines"]])
            for record in records
        }
        reference_evidence[ocr_name] = evidence_map
        idf_by_ocr[ocr_name] = build_idf({slug: evidence.text for slug, evidence in evidence_map.items()})

    signals: dict[tuple[str, str], dict[str, dict[str, dict[str, float | str]]]] = {}
    available_benchmarks_by_ocr: dict[str, set[str]] = {}
    for ocr_name, cache_key in OCR_KEYS.items():
        available_benchmarks = {
            benchmark
            for benchmark in BENCHMARKS
            if (project_root / "artifacts" / "ocr_cache" / cache_key / benchmark / "query_ocr.jsonl").is_file()
        }
        available_benchmarks_by_ocr[ocr_name] = available_benchmarks
        for benchmark in sorted(available_benchmarks):
            cache_path = run_dir / "signal_cache" / f"{ocr_name}__{benchmark}.jsonl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.is_file():
                build_signal_cache(
                    cache_path,
                    benchmark,
                    manifests[benchmark],
                    retrieval[benchmark],
                    OCR_KEYS[ocr_name],
                    project_root,
                    candidate_indexes,
                    reference_evidence[ocr_name],
                )
            signals[(ocr_name, benchmark)] = {
                record["query_id"]: record["candidates"] for record in load_jsonl(cache_path)
            }
        print(f"signals[{ocr_name}] ready for {sorted(available_benchmarks)}", flush=True)

    # ------------------------------------------------------------------
    # Structured reranker per OCR config (train on calibration units only)
    # ------------------------------------------------------------------
    structured_models: dict[str, LogisticReranker] = {}
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(exist_ok=True)
    for ocr_name in OCR_KEYS:
        train_features: list[dict[str, float]] = []
        train_labels: list[int] = []
        for benchmark in ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
            if (ocr_name, benchmark) not in signals:
                continue  # OCR cache absent for this benchmark (e.g. VL x synthetic_dev)
            units = splits[benchmark]
            for query_id, candidates in signals[(ocr_name, benchmark)].items():
                manifest_row = next(row for row in manifests[benchmark] if row["query_id"] == query_id)
                unit_id = benchmark_unit_id(benchmark, manifest_row)
                if units.get(unit_id) != "calibration":
                    continue
                base_row = retrieval[benchmark][query_id]
                image_scores = json.loads(base_row["top5_scores"])
                target = base_row["target_slug"]
                for position, candidate_slug in enumerate(json.loads(base_row["top5_slugs"])):
                    train_features.append(
                        build_feature_row(
                            image_scores,
                            position,
                            _query_evidence(ocr_name, benchmark, query_id, project_root),
                            candidate_indexes[candidate_slug],
                            reference_evidence[ocr_name].get(candidate_slug),
                            candidates[str(position)],
                            idf_by_ocr[ocr_name],
                        )
                    )
                    train_labels.append(1 if candidate_slug == target else 0)
        model = fit_logistic_reranker(train_features, train_labels)
        structured_models[ocr_name] = model
        with (structured_dir / f"model_{ocr_name}.json").open("w", encoding="utf-8") as stream:
            json.dump({"trained_rows": len(train_labels), "positives": sum(train_labels), **model.to_json()}, stream, indent=2, sort_keys=True)
        print(f"structured[{ocr_name}] trained on {len(train_labels)} rows", flush=True)

    # ------------------------------------------------------------------
    # BGE scoring cache (Top-5 pairs, incremental jsonl, per OCR config)
    # ------------------------------------------------------------------
    bge_scores: dict[tuple[str, str, str], dict[str, float]] = {}
    if not args.skip_bge:
        bge = BgeReranker(device=args.device)
        for ocr_name in ("current_eslav", "cyrillic"):
            for benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
                cache_path = run_dir / "bge" / f"{ocr_name}__{benchmark}.jsonl"
                cache_path.parent.mkdir(exist_ok=True)
                done: set[str] = set()
                if cache_path.is_file():
                    for record in load_jsonl(cache_path):
                        done.add(record["query_id"])
                        bge_scores[(ocr_name, benchmark, record["query_id"])] = record["bge_scores"]
                pending = [row for row in manifests[benchmark] if row["query_id"] not in done]
                with cache_path.open("a", encoding="utf-8") as stream:
                    for position, manifest_row in enumerate(pending, start=1):
                        query_id = manifest_row["query_id"]
                        base_row = retrieval[benchmark][query_id]
                        record = query_ocr_record(project_root, OCR_KEYS[ocr_name], benchmark, query_id)
                        query_evidence = build_query_text_evidence(
                            [(line["text"], line["confidence"]) for line in record["lines"]]
                        )
                        pairs = []
                        for slug in json.loads(base_row["top5_slugs"]):
                            doc = candidate_document_text(
                                candidate_indexes[slug], reference_evidence[ocr_name].get(slug)
                            )
                            pairs.append((query_evidence.text or query_evidence.transliterated, doc))
                        scores = bge.score_pairs(pairs)
                        payload = {"query_id": query_id, "bge_scores": [round(score, 6) for score in scores]}
                        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        stream.flush()
                        bge_scores[(ocr_name, benchmark, query_id)] = payload["bge_scores"]
                        if position % 500 == 0:
                            print(f"  bge {ocr_name}/{benchmark} {position}/{len(pending)}", flush=True)
        bge.release()
        print("bge scoring done", flush=True)

    # ------------------------------------------------------------------
    # Matrix evaluation
    # ------------------------------------------------------------------
    # Family membership map (slug -> family id) for hard/generated metrics.
    family_by_slug: dict[str, str] = {}
    for benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
        family_column = "family_id" if benchmark == "hard_near_duplicate_dev_v2" else "target_family_id"
        for row in manifests[benchmark]:
            if row.get(family_column):
                family_by_slug[row["target_slug"]] = row[family_column]

    # hard_v2 family_type (vintage vs subtype_or_other) from the frozen families.csv
    from scripts.run_so400m_ocr_reranker import hard_v2_family_classification

    hard_family_type = hard_v2_family_classification()
    family_type_by_slug = {
        slug: hard_family_type.get(family_id, "")
        for slug, family_id in family_by_slug.items()
        if benchmark_slug_is_hard(slug)
    }
    family_type_by_query: dict[str, str] = {}
    for benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
        family_column = "family_id" if benchmark == "hard_near_duplicate_dev_v2" else "target_family_id"
        for row in manifests[benchmark]:
            if benchmark == "hard_near_duplicate_dev_v2":
                family_type_by_query[row["query_id"]] = hard_family_type.get(row.get(family_column, ""), "")
            else:
                family_type_by_query[row["query_id"]] = row.get("family_type", "")

    combos = build_combo_list(include_bge=not args.skip_bge)
    benchmark_summary: list[dict] = []
    transition_rows: list[dict] = []
    per_combo_dir = run_dir / "combos"
    per_combo_dir.mkdir(exist_ok=True)
    family_rows: list[dict] = []
    slice_rows: list[dict] = []

    for combo in combos:
        for benchmark in BENCHMARKS:
            if not combo_available(combo, benchmark, available_benchmarks_by_ocr):
                continue
            evaluation = evaluate_combo(
                combo, benchmark, manifests[benchmark], retrieval[benchmark], signals, bge_scores,
                structured_models, candidate_indexes, reference_evidence, idf_by_ocr, project_root,
                family_by_slug, family_type_by_query,
            )
            combo_dir = per_combo_dir / f"{combo.key}__{benchmark}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            write_csv(combo_dir / "predictions.csv", list(evaluation["predictions"][0]), evaluation["predictions"])
            (combo_dir / "metrics.json").write_text(
                json.dumps(evaluation["metrics"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            write_csv(combo_dir / "transitions.csv", ["metric", "value"], evaluation["transitions_csv"])
            benchmark_summary.append({"combo": combo.key, "ocr": combo.ocr, "reranker": combo.reranker, "benchmark": benchmark, **evaluation["metrics_cell"]})
            transition_rows.append({"combo": combo.key, "ocr": combo.ocr, "reranker": combo.reranker, "benchmark": benchmark, **evaluation["transitions"]})
            for family_type, metrics in evaluation["family"].items():
                family_rows.append({"combo": combo.key, "benchmark": benchmark, "family_type": family_type, **metrics})
            for slice_name, metrics in evaluation["slices"].items():
                slice_rows.append({"combo": combo.key, "benchmark": benchmark, "slice": slice_name, **metrics})
        print(f"combo done: {combo.key}", flush=True)

    write_csv(
        run_dir / "benchmark_summary.csv",
        ["combo", "ocr", "reranker", "benchmark", "top1_accuracy", "recall_at_5", "mrr", "rescued", "broken"],
        benchmark_summary,
    )
    write_csv(
        run_dir / "transition_summary.csv",
        ["combo", "ocr", "reranker", "benchmark", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong", "correct_to_correct"],
        transition_rows,
    )
    write_csv(
        run_dir / "family_summary.csv",
        ["combo", "benchmark", "family_type", "query_count", "exact_top1", "family_top1", "family_recall_at_5", "within_family_disambiguation_top1"],
        family_rows,
    )
    write_csv(
        run_dir / "slices_summary.csv",
        ["combo", "benchmark", "slice", "queries", "exact_top1", "exact_top1_count"],
        slice_rows,
    )

    config = {
        "run_id": run_dir.name,
        "purpose": "OCR + reranker bake-off over the frozen SO400M Top-5",
        "retrieval_backbone": "siglip2_so400m_384 (frozen, candidate generation unchanged)",
        "candidate_policy": "SO400M Top-5 only; OCR/rerankers permute the Top-5 and never change the set",
        "ocr_configs": {name: {"cache_key": key} for name, key in OCR_KEYS.items()},
        "frozen_blend": {
            "policy": FROZEN_BLEND.policy,
            "alpha": FROZEN_BLEND.alpha,
            "vintage_bonus": FROZEN_BLEND.vintage_bonus,
            "vintage_penalty": FROZEN_BLEND.vintage_penalty,
            "min_text_margin": FROZEN_BLEND.min_text_margin,
        },
        "structured": {
            "model": "LogisticRegression(lbfgs, standardized, C=1.0)",
            "features": 21,
            "train_units": "calibration halves only (product-level generated, family-level hard_v2, product-level synthetic)",
        },
        "bge": {"model_id": "BAAI/bge-reranker-v2-m3", "fusion_weights": list(BGE_WEIGHTS)},
        "split_seed": SPLIT_SEED,
        "vl_note": "paddleocr_vl cache covers references + hard_v2 + generated; synthetic_dev skipped (throughput), disclosed in the report",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Bake-off artifacts: {run_dir}")


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------

def build_splits(manifests: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    splits: dict[str, dict[str, str]] = {}
    # synthetic: product-level calibration sample
    synthetic_products = sorted({row["target_slug"] for row in manifests["synthetic_dev"]})
    cal_products = set(seeded_product_sample(synthetic_products, SYNTHETIC_CAL_PRODUCTS, SPLIT_SEED))
    splits["synthetic_dev"] = {
        slug: ("calibration" if slug in cal_products else "heldout") for slug in synthetic_products
    }
    # hard_v2: family-level split
    families = sorted({row["family_id"] for row in manifests["hard_near_duplicate_dev_v2"] if row.get("family_id")})
    cal_families = set(seeded_product_sample(families, HARD_V2_CAL_FAMILIES, SPLIT_SEED))
    units = {}
    for row in manifests["hard_near_duplicate_dev_v2"]:
        family_id = row.get("family_id") or row["target_slug"]
        units[family_id] = "calibration" if family_id in cal_families else "heldout"
    splits["hard_near_duplicate_dev_v2"] = units
    # generated: product-level split, all 4 scenarios together
    generated_products = sorted({row["target_slug"] for row in manifests["generated_stress_dev_pilot32"]})
    pilot_cal = set(seeded_product_sample(generated_products, PILOT32_CAL_PRODUCTS, SPLIT_SEED))
    splits["generated_stress_dev_pilot32"] = {
        slug: ("calibration" if slug in pilot_cal else "heldout") for slug in generated_products
    }
    return splits


def benchmark_unit_id(benchmark: str, manifest_row: dict[str, str]) -> str:
    if benchmark == "hard_near_duplicate_dev_v2":
        return manifest_row.get("family_id") or manifest_row["target_slug"]
    return manifest_row["target_slug"]


# ----------------------------------------------------------------------
# Signal cache building
# ----------------------------------------------------------------------

def _query_evidence(ocr_name: str, benchmark: str, query_id: str, project_root: Path):
    record = query_ocr_record(project_root, OCR_KEYS[ocr_name], benchmark, query_id)
    return build_query_text_evidence([(line["text"], line["confidence"]) for line in record["lines"]])


def query_ocr_record(project_root: Path, cache_key: str, benchmark: str, query_id: str) -> dict:
    path = project_root / "artifacts" / "ocr_cache" / cache_key / benchmark / "query_ocr.jsonl"
    if not hasattr(query_ocr_record, "_cache"):
        query_ocr_record._cache = {}
    key = (cache_key, benchmark)
    if key not in query_ocr_record._cache:
        query_ocr_record._cache[key] = {
            record["query_id"]: record for record in load_jsonl(path)
        }
    return query_ocr_record._cache[key][query_id]


def build_signal_cache(
    cache_path: Path,
    benchmark: str,
    manifest_rows: list[dict[str, str]],
    retrieval_rows: dict[str, dict],
    cache_key: str,
    project_root: Path,
    candidate_indexes: dict[str, Any],
    reference_evidence: dict[str, object],
) -> None:
    done: set[str] = set()
    if cache_path.is_file():
        for record in load_jsonl(cache_path):
            done.add(record["query_id"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        for position, manifest_row in enumerate(manifest_rows, start=1):
            query_id = manifest_row["query_id"]
            if query_id in done:
                continue
            record = query_ocr_record(project_root, cache_key, benchmark, query_id)
            query_evidence = build_query_text_evidence(
                [(line["text"], line["confidence"]) for line in record["lines"]]
            )
            base_row = retrieval_rows[query_id]
            candidates: dict[str, dict[str, float | str]] = {}
            for candidate_position, slug in enumerate(json.loads(base_row["top5_slugs"])):
                signals = compute_text_signals(
                    query_evidence,
                    candidate_indexes[slug],
                    reference_evidence.get(slug),
                )
                candidates[str(candidate_position)] = signals
            stream.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "target_slug": manifest_row["target_slug"],
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            if position % 500 == 0:
                print(f"  signals[{cache_key[:24]}...] {benchmark} {position}/{len(manifest_rows)}", flush=True)


# ----------------------------------------------------------------------
# Combos
# ----------------------------------------------------------------------

def build_combo_list(include_bge: bool = True) -> list["Combo"]:
    combos = []
    for ocr in ("current_eslav", "cyrillic", "paddleocr_vl"):
        combos.append(Combo(ocr=ocr, reranker="image_only", key=f"{ocr}__image_only"))
        combos.append(Combo(ocr=ocr, reranker="current_blend", key=f"{ocr}__current_blend"))
    for ocr in ("current_eslav", "cyrillic", "paddleocr_vl"):
        combos.append(Combo(ocr=ocr, reranker="structured", key=f"{ocr}__structured"))
    if include_bge:
        for ocr in ("current_eslav", "cyrillic"):
            for weight in BGE_WEIGHTS:
                combos.append(Combo(ocr=ocr, reranker=f"bge_fusion_{weight}", key=f"{ocr}__bge_fusion_{weight}"))
    return combos


def combo_available(combo: "Combo", benchmark: str, available_benchmarks_by_ocr: dict[str, set[str]]) -> bool:
    if benchmark not in available_benchmarks_by_ocr.get(combo.ocr, set()):
        return False  # OCR cache absent for this benchmark (e.g. VL x synthetic_dev, disclosed)
    if combo.reranker.startswith("bge_fusion"):
        return benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
    return True


def evaluate_combo(
    combo: "Combo",
    benchmark: str,
    manifest_rows: list[dict[str, str]],
    retrieval_rows: dict[str, dict],
    signals: dict[tuple[str, str], dict[str, dict[str, dict[str, float | str]]]],
    bge_scores: dict[tuple[str, str, str], dict[str, float]],
    structured_models: dict[str, LogisticReranker],
    candidate_indexes: dict[str, Any],
    reference_evidence: dict[str, dict[str, object]],
    idf_by_ocr: dict[str, dict[str, float]],
    project_root: Path,
    family_by_slug: dict[str, str],
    family_type_by_query: dict[str, str],
) -> dict:
    predictions = []
    ranks: list[int | None] = []
    transitions = {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
    model = structured_models.get(combo.ocr)
    for manifest_row in manifest_rows:
        query_id = manifest_row["query_id"]
        base_row = retrieval_rows[query_id]
        top5 = json.loads(base_row["top5_slugs"])
        image_scores = json.loads(base_row["top5_scores"])
        target = base_row["target_slug"]
        candidate_signals = signals[(combo.ocr, benchmark)][query_id]
        baseline_rank = int(base_row["target_rank"]) if base_row["target_rank"] else None

        if combo.reranker == "image_only":
            final_order = list(range(len(top5)))
            reason = "image_only"
        elif combo.reranker == "current_blend":
            outcome = rerank_one_query(image_scores, [candidate_signals[str(i)] for i in range(len(top5))], FROZEN_BLEND)
            final_order = list(outcome.final_order)
            reason = outcome.reason
        elif combo.reranker == "structured":
            query_evidence = _query_evidence_cached(combo.ocr, benchmark, query_id, project_root)
            features = [
                build_feature_row(
                    image_scores,
                    position,
                    query_evidence,
                    candidate_indexes[slug],
                    reference_evidence[combo.ocr].get(slug),
                    candidate_signals[str(position)],
                    idf_by_ocr[combo.ocr],
                )
                for position, slug in enumerate(top5)
            ]
            probabilities = [model.score_row(row) for row in features]
            final_order = sorted(range(len(top5)), key=lambda index: (-probabilities[index], index))
            reason = "structured_lr"
        elif combo.reranker.startswith("bge_fusion_"):
            weight = float(combo.reranker.rsplit("_", 1)[1])
            bge = bge_scores[(combo.ocr, benchmark, query_id)]
            image_norm = normalize_image_scores(image_scores)
            bge_norm = normalize_image_scores(bge)
            fused = [(1.0 - weight) * image_norm[i] + weight * bge_norm[i] for i in range(len(top5))]
            final_order = sorted(range(len(top5)), key=lambda index: (-fused[index], index))
            reason = f"bge_fusion_w{weight}"
        else:
            raise ValueError(combo.reranker)

        final_rank = None
        for final_position, candidate_index in enumerate(final_order, start=1):
            if top5[candidate_index] == target:
                final_rank = final_position
                break
        ranks.append(final_rank)
        final_top1 = top5[final_order[0]]
        baseline_correct = base_row["correct_top1"] == "True"
        final_correct = final_top1 == target
        if not baseline_correct and final_correct:
            transitions["wrong_to_correct"] += 1
        elif baseline_correct and not final_correct:
            transitions["correct_to_wrong"] += 1
        elif baseline_correct and final_correct:
            transitions["correct_to_correct"] += 1
        else:
            transitions["wrong_to_wrong"] += 1
        predictions.append(
            {
                "query_id": query_id,
                "target_slug": target,
                "baseline_predicted_slug": base_row["predicted_slug"],
                "final_predicted_slug": final_top1,
                "final_top5_slugs": json.dumps([top5[index] for index in final_order], ensure_ascii=False, separators=(",", ":")),
                "correct_top1": final_correct,
                "baseline_correct_top1": baseline_correct,
                "target_rank": final_rank if final_rank is not None else "",
                "reason": reason,
            }
        )

    metrics = rank_metrics(ranks)
    metrics_cell = {
        "queries": metrics["query_count"],
        "top1_accuracy": metrics["top1_accuracy"],
        "recall_at_5": metrics["recall_at_5"],
        "mrr": metrics["mrr"],
        "rescued": transitions["wrong_to_correct"],
        "broken": transitions["correct_to_wrong"],
    }
    family = _family_slice_metrics(manifest_rows, predictions, family_by_slug, family_type_by_query)
    slices = _diagnostic_slices(benchmark, manifest_rows, predictions, family_type_by_query)
    return {
        "predictions": predictions,
        "ranks": ranks,
        "metrics": metrics,
        "metrics_cell": metrics_cell,
        "transitions": transitions,
        "transitions_csv": [{"metric": key, "value": value} for key, value in transitions.items()],
        "family": family,
        "slices": slices,
    }


def _query_evidence_cached(ocr_name: str, benchmark: str, query_id: str, project_root: Path):
    return _query_evidence(ocr_name, benchmark, query_id, project_root)


def _family_slice_metrics(
    manifest_rows: list[dict[str, str]],
    predictions: list[dict],
    family_by_slug: dict[str, str],
    family_type_by_query: dict[str, str],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for family_type in ("vintage", "subtype_or_other"):
        scoped = [row for row in predictions if family_type_by_query.get(row["query_id"]) == family_type]
        if not scoped:
            continue
        exact = sum(1 for row in scoped if row["correct_top1"])
        family_top1 = 0
        baseline_wrong_but_family_top1 = 0
        for row in scoped:
            target_family = family_by_slug.get(row["target_slug"], "")
            if target_family and family_by_slug.get(row["final_predicted_slug"], "") == target_family:
                family_top1 += 1
                if not row["baseline_correct_top1"]:
                    baseline_wrong_but_family_top1 += 1
        result[family_type] = {
            "query_count": len(scoped),
            "exact_top1": round(exact / len(scoped), 4),
            "exact_top1_count": exact,
            "family_top1": round(family_top1 / len(scoped), 4),
            "baseline_wrong_but_family_top1": baseline_wrong_but_family_top1,
        }
    return result


def _diagnostic_slices(
    benchmark: str,
    manifest_rows: list[dict[str, str]],
    predictions: list[dict],
    family_type_by_query: dict[str, str],
) -> dict[str, dict]:
    slices: dict[str, dict] = {}
    prediction_by_query = {row["query_id"]: row for row in predictions}
    if benchmark in ("hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32"):
        for family_type in ("vintage", "subtype_or_other"):
            scoped = [
                prediction_by_query[row["query_id"]]
                for row in manifest_rows
                if family_type_by_query.get(row["query_id"]) == family_type
            ]
            if scoped:
                slices[f"{family_type}_slice"] = {
                    "queries": len(scoped),
                    "exact_top1": round(sum(1 for row in scoped if row["correct_top1"]) / len(scoped), 4),
                    "exact_top1_count": sum(1 for row in scoped if row["correct_top1"]),
                }
    if benchmark == "generated_stress_dev_pilot32":
        hard_rows = [row for row in manifest_rows if row.get("subset_role", "") == "hard"]
        scoped = [prediction_by_query[row["query_id"]] for row in hard_rows]
        slices["generated_hard"] = {
            "queries": len(scoped),
            "exact_top1": round(sum(1 for row in scoped if row["correct_top1"]) / len(scoped), 4),
            "exact_top1_count": sum(1 for row in scoped if row["correct_top1"]),
        }
    return slices


def benchmark_slug_is_hard(slug: str) -> bool:
    del slug
    return True


def _family_metrics(manifest_rows: list[dict[str, str]], predictions: list[dict]) -> dict:
    del manifest_rows, predictions
    return {}


class Combo:
    def __init__(self, ocr: str, reranker: str, key: str) -> None:
        self.ocr = ocr
        self.reranker = reranker
        self.key = key


if __name__ == "__main__":
    main()
