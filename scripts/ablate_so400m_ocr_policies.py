#!/usr/bin/env python3
"""Policy ablation on full benchmark sets (diagnostic, Part H/R of the brief).

The frozen milestone selection used calibration subsets only. This script
replays every text policy at its grid-best alpha on the FULL benchmarks and
writes ablation_summary.csv next to the milestone run artifacts, plus the
hard_v2 vintage/subtype case split. Nothing here changes the frozen
selection; this is the A/B/C/D evidence table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.crop_diagnostics import rank_metrics  # noqa: E402
from recognition.ocr_reranker import (  # noqa: E402
    POLICY_IMAGE_ONLY,
    FusionConfig,
    rerank_one_query,
    top1_transition_counts,
)
from scripts.run_siglip2_baseline import _default_benchmark_manifest, _read_csv  # noqa: E402
from scripts.run_so400m_ocr_reranker import (  # noqa: E402
    GRID_ALPHAS,
    hard_v2_family_classification,
    benchmark_query_rows,
)

POLICY_ALPHA_GRID = {
    "metadata_text_blend": (0.30, 0.40),
    "reference_ocr_blend": (0.20, 0.30),
    "combined_text_blend": (0.30, 0.40),
    "combined_vintage_blend": (0.30, 0.40),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-set policy ablation for an OCR reranker run")
    parser.add_argument("--run-dir", required=True, help="run dir with per-benchmark retrieval_dump.json")
    parser.add_argument("--selection-dir", default=None, help="run dir with selected_reranker.json (defaults to --run-dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    selection_dir = Path(args.selection_dir) if args.selection_dir else run_dir
    config = json.loads((selection_dir / "selected_reranker.json").read_text(encoding="utf-8"))["config"]
    selected = FusionConfig(**config)

    import shutil

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from scripts.run_so400m_ocr_reranker import (  # noqa: E402
        compute_all_signals,
        load_query_evidence,
        load_reference_evidence,
    )

    cache_root = PROJECT_ROOT / "artifacts" / "ocr_cache" / "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec"
    benchmarks = ("synthetic_dev", "hard_near_duplicate_dev_v2", "generated_stress_dev_pilot32")
    query_evidence, _ = load_query_evidence(benchmarks, cache_root)
    reference_evidence = load_reference_evidence(cache_root)

    from recognition.text_signals import build_candidate_text_index  # noqa: E402
    from scripts.run_siglip2_baseline import _catalog_items, _path  # noqa: E402

    catalog_meta = json.loads(_path(PROJECT_ROOT, "data/processed/catalog_v1.json").read_text(encoding="utf-8"))
    catalog_items, _ = _catalog_items(
        PROJECT_ROOT, _read_csv(_path(PROJECT_ROOT, "data/processed/catalog_manifest.csv")), catalog_meta
    )
    candidate_indexes = {item.item_id: build_candidate_text_index(item.metadata) for item in catalog_items}

    retrieval = {}
    for benchmark in benchmarks:
        payload = json.loads((run_dir / benchmark / "retrieval_dump.json").read_text(encoding="utf-8"))
        retrieval[benchmark] = [type("R", (), row) for row in payload]
    signals, _ = compute_all_signals(benchmarks, retrieval, query_evidence, reference_evidence, candidate_indexes)

    rows: list[dict] = []
    for benchmark in benchmarks:
        manifest = benchmark_query_rows(benchmark)
        meta = {row["query_id"]: row for row in manifest}
        records = retrieval[benchmark]
        policies = [(POLICY_IMAGE_ONLY, None)] + [
            (policy, alpha) for policy, alphas in POLICY_ALPHA_GRID.items() for alpha in alphas
        ]
        for policy, alpha in policies:
            config_i = FusionConfig(policy=policy, alpha=alpha or 0.0)
            final_top1, targets = [], []
            for record in records:
                outcome = rerank_one_query(record.top10_scores[:5], signals[benchmark][record.query_id], config_i)
                final_top1.append(record.top10_slugs[outcome.final_order[0]])
                targets.append(record.target_slug)
            transitions = top1_transition_counts(
                [record.top10_slugs[0] for record in records], final_top1, targets
            )
            top1 = sum(1 for top1, target in zip(final_top1, targets) if top1 == target) / len(records)
            rows.append({
                "benchmark": benchmark,
                "policy": policy,
                "alpha": alpha or "",
                "top1": round(top1, 4),
                "rescued": transitions.wrong_to_correct,
                "broken": transitions.correct_to_wrong,
            })

    # hard_v2 vintage/subtype split for the SELECTED reranker.
    family_types = hard_v2_family_classification()
    manifest = _read_csv(_default_benchmark_manifest(PROJECT_ROOT, "hard_near_duplicate_dev_v2", None))
    hard_v2_meta = {row["query_id"]: row for row in manifest}
    for policy, alpha in [(POLICY_IMAGE_ONLY, None), (selected.policy, selected.alpha)]:
        config_i = FusionConfig(policy=policy, alpha=alpha or 0.0)
        by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for record in retrieval["hard_near_duplicate_dev_v2"]:
            family_id = hard_v2_meta[record.query_id].get("family_id", "")
            kind = family_types.get(family_id, "subtype_or_other")
            by_kind[kind]["total"] += 1
            if policy != POLICY_IMAGE_ONLY:
                outcome = rerank_one_query(record.top10_scores[:5], signals["hard_near_duplicate_dev_v2"][record.query_id], config_i)
                predicted = record.top10_slugs[outcome.final_order[0]]
            else:
                predicted = record.top10_slugs[0]
            by_kind[kind]["correct"] += int(predicted == record.target_slug)
        for kind, counts in sorted(by_kind.items()):
            rows.append({
                "benchmark": f"hard_v2_by_family_kind:{kind}",
                "policy": policy,
                "alpha": alpha or "",
                "top1": round(counts["correct"] / counts["total"], 4),
                "rescued": "",
                "broken": "",
            })

    out_path = run_dir / "ablation_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "policy", "alpha", "top1", "rescued", "broken"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    for row in rows:
        print(row)
    if selection_dir != run_dir:
        shutil.copy2(out_path, selection_dir / "ablation_summary.csv")
        print(f"Copied to {selection_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
