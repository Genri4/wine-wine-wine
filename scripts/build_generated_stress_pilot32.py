#!/usr/bin/env python3
"""Build the isolated, no-generation generated_stress_dev pilot32 plan.

The script selects 16 representative products and 16 products from eight
evidence-carrying hard families.  It only reads existing catalog/benchmark
artifacts and writes a new isolated benchmark directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import csv
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.cache import sha256_file  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from scripts.build_generated_stress_plan import (  # noqa: E402
    SCENARIOS,
    SCENARIO_IDS,
    _usable_rows,
    select_products,
    make_generation_id,
)


DEFAULT_OUTPUT_DIR = "data/benchmarks/generated_stress_dev_pilot32"
DEFAULT_CATALOG_MANIFEST = "data/processed/catalog_manifest.csv"
DEFAULT_EXISTING_SELECTION = "data/benchmarks/generated_stress_dev/selected_products.csv"
DEFAULT_HARD_FAMILIES = "data/benchmarks/hard_near_duplicate_dev_v2/families.csv"
DEFAULT_HARD_SUMMARY = "data/benchmarks/hard_near_duplicate_dev_v2/family_summary.csv"
DEFAULT_SEED = 20260916
REPRESENTATIVE_COUNT = 16
HARD_FAMILY_COUNT = 8
PRODUCTS_PER_FAMILY = 2
VINTAGE_FAMILY_COUNT = 4
SUBTYPE_FAMILY_COUNT = 4

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
GENERIC_NAME_TOKENS = {
    "wine", "vino", "вино", "красное", "белое", "розовое", "оранжевое",
    "red", "white", "rose", "orange", "сухое", "полусухое", "полусладкое",
    "брют", "экстра", "brut", "extra", "bryut", "suhoe", "beloe", "krasnoe",
    "rezerv", "reserve", "limited", "edishn", "edition",
}

SELECTED_FIELDS = [
    "slug", "reference_image", "product_name", "winery", "category", "color",
    "region", "grape", "subset_role", "selection_stratum", "selection_seed",
    "representative_selection_reason", "representative_stratum", "pilot_family_id",
    "target_family_id", "family_type", "family_size", "family_member_order",
    "hard_selection_reason", "source_hard_family_id",
]

FAMILY_FIELDS = [
    "pilot_family_id", "source_family_id", "family_type", "family_reason",
    "evidence_summary", "family_size", "member_slugs", "selection_score",
    "source_selection_reason", "source_edge_count", "source_confusion_count",
    "source_confusion_rank_2_5_count", "source_small_margin_count",
    "source_target_ranks", "source_min_margin", "source_max_margin",
    "source_visual_pair_count", "source_name_pair_count", "source_selection_evidence_json",
]

GENERATION_FIELDS = [
    "generation_id", "target_slug", "reference_image", "scenario_id", "prompt",
    "output_filename", "subset_role", "target_family_id", "family_type",
    "family_size", "family_member_order", "generation_status",
]


@dataclass(frozen=True)
class FamilyCandidate:
    source_family_id: str
    family_type: str
    members: tuple[dict[str, str], ...]
    source_summary: dict[str, str]
    selection_score: float
    classifier_reason: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated generated_stress_dev_pilot32 plan")
    parser.add_argument("--catalog-manifest", default=DEFAULT_CATALOG_MANIFEST)
    parser.add_argument("--existing-selection", default=DEFAULT_EXISTING_SELECTION)
    parser.add_argument("--hard-families", default=DEFAULT_HARD_FAMILIES)
    parser.add_argument("--hard-summary", default=DEFAULT_HARD_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    catalog_path = _path(args.catalog_manifest)
    existing_selection_path = _path(args.existing_selection)
    hard_families_path = _path(args.hard_families)
    hard_summary_path = _path(args.hard_summary)
    output_dir = _path(args.output_dir)

    catalog_rows = _read_csv(catalog_path)
    usable_rows = _usable_rows(catalog_rows)
    catalog_by_slug = {row["slug"]: row for row in usable_rows}
    existing_rows = _read_csv(existing_selection_path)
    hard_rows = _read_csv(hard_families_path)
    hard_summary_rows = _read_csv(hard_summary_path)

    hard_families = select_hard_families(
        hard_rows,
        hard_summary_rows,
        catalog_by_slug,
        vintage_count=VINTAGE_FAMILY_COUNT,
        subtype_count=SUBTYPE_FAMILY_COUNT,
    )
    hard_slugs = {member["slug"] for family in hard_families for member in family.members}
    all_hard_source_slugs = {row["slug"] for row in hard_rows}
    representatives, representative_source, quotas = select_representatives(
        usable_rows,
        existing_rows,
        all_hard_source_slugs,
        REPRESENTATIVE_COUNT,
        args.seed,
    )
    representative_slugs = {row["slug"] for row in representatives}
    if hard_slugs & representative_slugs:
        raise AssertionError("hard and representative selections overlap")

    selected_rows = representatives + _hard_product_rows(hard_families, catalog_by_slug, args.seed)
    selected_rows.sort(key=lambda row: (str(row["subset_role"]), str(row["slug"])))
    if len(selected_rows) != 32 or len({row["slug"] for row in selected_rows}) != 32:
        raise AssertionError("pilot32 must contain 32 unique products")

    generation_rows = build_generation_rows(selected_rows)
    if len(generation_rows) != 128:
        raise AssertionError("pilot32 must contain 128 planned generations")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_runs").mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_products.csv", SELECTED_FIELDS, selected_rows)
    write_csv(
        output_dir / "hard_families.csv",
        FAMILY_FIELDS,
        [_family_row(family) for family in hard_families],
    )
    write_csv(output_dir / "generation_manifest.csv", GENERATION_FIELDS, generation_rows)
    (output_dir / "scenarios.json").write_text(
        json.dumps(
            {"scenario_version": "generated-stress-scenarios-v1", "scenarios": [SCENARIOS[key] for key in SCENARIO_IDS]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = _metadata(
        catalog_path=catalog_path,
        hard_families_path=hard_families_path,
        existing_selection_path=existing_selection_path,
        seed=args.seed,
        representative_source=representative_source,
        quotas=quotas,
        hard_families=hard_families,
        selected_rows=selected_rows,
        generation_rows=generation_rows,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = selection_report(
        usable_rows=usable_rows,
        existing_rows=existing_rows,
        representatives=representatives,
        selected_rows=selected_rows,
        hard_families=hard_families,
        quotas=quotas,
        seed=args.seed,
        catalog_path=catalog_path,
        hard_families_path=hard_families_path,
        representative_source=representative_source,
    )
    (output_dir / "selection_report.md").write_text(report, encoding="utf-8")
    report_path = PROJECT_ROOT / "reports/generated_stress_dev_pilot32_selection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(metadata), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def select_representatives(
    usable_rows: Sequence[Mapping[str, str]],
    existing_rows: Sequence[Mapping[str, str]],
    excluded_hard_slugs: set[str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, object]], str, dict[str, int]]:
    """Select representatives from the existing 150-product plan when possible."""

    usable_by_slug = {row["slug"]: row for row in usable_rows}
    existing_slugs = [row.get("slug", "").strip() for row in existing_rows]
    candidates = [
        usable_by_slug[slug]
        for slug in existing_slugs
        if slug in usable_by_slug and slug not in excluded_hard_slugs
    ]
    source = "existing generated_stress_dev 150-product selection, excluding all v2 hard-family products"
    if len(candidates) < count:
        candidates = [row for row in usable_rows if row["slug"] not in excluded_hard_slugs]
        source = "full usable catalog fallback, excluding all v2 hard-family products"
    selected, quotas = select_products([dict(row) for row in candidates], count, seed)
    result: list[dict[str, object]] = []
    for row in selected:
        result.append({
            "slug": row["slug"],
            "reference_image": row["reference_image"],
            "product_name": row["product_name"],
            "winery": row["winery"],
            "category": row["category"],
            "color": row["color"],
            "region": row["region"],
            "grape": row["grape"],
            "subset_role": "representative",
            "selection_stratum": row["selection_stratum"],
            "selection_seed": seed,
            "representative_selection_reason": (
                f"deterministic proportional category×region quota with winery round-robin; {source}"
            ),
            "representative_stratum": row["selection_stratum"],
            "pilot_family_id": "",
            "target_family_id": "",
            "family_type": "",
            "family_size": "",
            "family_member_order": "",
            "hard_selection_reason": "",
            "source_hard_family_id": "",
        })
    return result, source, quotas


def select_hard_families(
    hard_rows: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
    catalog_by_slug: Mapping[str, Mapping[str, str]],
    *,
    vintage_count: int,
    subtype_count: int,
) -> list[FamilyCandidate]:
    """Choose diverse, two-member families using only v2 evidence plus explicit type rules."""

    members_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_rows:
        slug = row.get("slug", "").strip()
        if slug in catalog_by_slug:
            members_by_family[row["family_id"]].append(dict(row))
    summaries = {row["family_id"]: dict(row) for row in summary_rows}
    vintage_candidates: list[FamilyCandidate] = []
    subtype_candidates: list[FamilyCandidate] = []
    for family_id, members in sorted(members_by_family.items()):
        if len(members) != PRODUCTS_PER_FAMILY:
            continue
        summary = summaries.get(family_id)
        if not summary:
            continue
        signals = _source_signals(summary)
        if len(signals) < 2:
            continue
        family_type, reason, score = _classify_family(members, summary)
        if family_type == "vintage":
            vintage_candidates.append(FamilyCandidate(family_id, family_type, tuple(sorted(members, key=lambda row: row["slug"])), summary, score, reason))
        elif family_type == "subtype":
            subtype_candidates.append(FamilyCandidate(family_id, family_type, tuple(sorted(members, key=lambda row: row["slug"])), summary, score, reason))

    vintage = _choose_diverse(vintage_candidates, vintage_count)
    if len(vintage) != vintage_count:
        raise ValueError(
            f"Could not find {vintage_count} reliable vintage families; found {len(vintage)}. "
            "Review hard_near_duplicate_dev_v2 family evidence before using a fallback."
        )
    used_wineries = {member["winery"] for family in vintage for member in family.members}
    subtype = _choose_diverse(subtype_candidates, subtype_count, excluded_wineries=set(), excluded_family_ids={family.source_family_id for family in vintage})
    if len(subtype) != subtype_count:
        raise ValueError(
            f"Could not find {subtype_count} reliable subtype families; found {len(subtype)}. "
            "Review hard_near_duplicate_dev_v2 family evidence before using a fallback."
        )
    # Prefer winery diversity across the two hard types when alternatives have the same evidence score.
    subtype = _maximize_cross_type_diversity(subtype, subtype_candidates, used_wineries, subtype_count, {family.source_family_id for family in vintage})
    selected: list[FamilyCandidate] = []
    for index, family in enumerate(vintage, start=1):
        selected.append(_rename_family(family, f"pilot-family-vintage-{index:02d}"))
    for index, family in enumerate(subtype, start=1):
        selected.append(_rename_family(family, f"pilot-family-subtype-{index:02d}"))
    return selected


def _classify_family(members: Sequence[Mapping[str, str]], summary: Mapping[str, str]) -> tuple[str | None, str, float]:
    years = [_years(row) for row in members]
    base_names = [_name_without_year(row.get("product_name", "")) for row in members]
    signals = _source_signals(summary)
    evidence_score = (
        len(signals) * 10
        + _int(summary.get("source_confusion_count", summary.get("confusion_count", "0"))) * 2
        + _int(summary.get("source_confusion_rank_2_5_count", summary.get("confusion_rank_2_5_count", "0")))
        + _int(summary.get("source_visual_pair_count", summary.get("visual_pair_count", "0")))
        + _int(summary.get("source_name_pair_count", summary.get("name_pair_count", "0")))
    )
    if all(values and len(values) == 1 for values in years):
        distinct_years = {next(iter(values)) for values in years}
        if len(distinct_years) >= 2 and len(set(base_names)) == 1:
            return (
                "vintage",
                "same normalized product line after removing explicit year; distinct explicit years="
                + ",".join(sorted(distinct_years))
                + "; source v2 evidence="
                + "+".join(signals),
                float(evidence_score + 20),
            )

    left, right = members
    left_tokens = _line_tokens(left)
    right_tokens = _line_tokens(right)
    common = left_tokens & right_tokens
    line_similarity = SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio()
    metadata_diff = any(
        left.get(field, "").strip() != right.get(field, "").strip()
        for field in ("category", "color", "grape")
    )
    variant_diff = _variant_tokens(left) != _variant_tokens(right)
    same_winery = left.get("winery", "").strip() == right.get("winery", "").strip()
    if same_winery and common and line_similarity >= 0.55 and (metadata_diff or variant_diff):
        reason_parts = [
            "same winery/product-line tokens",
            f"line_similarity={line_similarity:.3f}",
            "metadata difference" if metadata_diff else "name subtype difference",
            "source v2 evidence=" + "+".join(signals),
        ]
        return "subtype", "; ".join(reason_parts), float(evidence_score + line_similarity * 10)
    return None, "not a reliable explicit vintage or subtype pair", 0.0


def _choose_diverse(
    candidates: Sequence[FamilyCandidate],
    count: int,
    *,
    excluded_wineries: set[str] | None = None,
    excluded_family_ids: set[str] | None = None,
) -> list[FamilyCandidate]:
    excluded_wineries = excluded_wineries or set()
    excluded_family_ids = excluded_family_ids or set()
    usable = [
        candidate for candidate in candidates
        if candidate.source_family_id not in excluded_family_ids
        and candidate.members[0].get("winery", "") not in excluded_wineries
    ]
    fast = _choose_single_winery_groups(usable, count)
    if fast is not None:
        return fast
    best: tuple[tuple[object, ...], tuple[FamilyCandidate, ...]] | None = None
    for combo in itertools.combinations(usable, count):
        wineries = {member["winery"] for family in combo for member in family.members}
        score = sum(family.selection_score for family in combo)
        key = (
            len(wineries),
            score,
            min(family.selection_score for family in combo),
            tuple(sorted(family.source_family_id for family in combo)),
        )
        if best is None or key[:3] > best[0][:3] or (key[:3] == best[0][:3] and key[3] < best[0][3]):
            best = (key, tuple(combo))
    return sorted(best[1], key=lambda family: family.source_family_id) if best else []


def _choose_single_winery_groups(
    candidates: Sequence[FamilyCandidate], count: int
) -> list[FamilyCandidate] | None:
    """Exact fast path for the current pair-family shape.

    All eligible v2 pairs used here have one winery per family.  When enough
    distinct wineries exist, the original lexicographic objective is solved by
    taking the best-scoring candidate from each winery and then the best
    ``count`` groups.  The general combination search remains as a fallback
    for future multi-winery family shapes.
    """
    groups: dict[str, list[FamilyCandidate]] = defaultdict(list)
    for candidate in candidates:
        wineries = {member.get("winery", "") for member in candidate.members}
        if len(wineries) != 1:
            return None
        groups[next(iter(wineries))].append(candidate)
    if len(groups) < count:
        return None
    best_per_group = []
    for winery, group in groups.items():
        best_per_group.append(
            min(group, key=lambda family: (-family.selection_score, family.source_family_id))
        )
    selected = sorted(
        best_per_group,
        key=lambda family: (-family.selection_score, family.source_family_id),
    )[:count]
    return sorted(selected, key=lambda family: family.source_family_id)


def _maximize_cross_type_diversity(
    selected: Sequence[FamilyCandidate],
    candidates: Sequence[FamilyCandidate],
    vintage_wineries: set[str],
    count: int,
    excluded_family_ids: set[str],
) -> list[FamilyCandidate]:
    pool = [candidate for candidate in candidates if candidate.source_family_id not in excluded_family_ids]
    non_vintage = [
        candidate for candidate in pool
        if candidate.members[0].get("winery", "") not in vintage_wineries
    ]
    fast = _choose_single_winery_groups(non_vintage, count)
    if fast is not None:
        return fast
    best: tuple[tuple[object, ...], tuple[FamilyCandidate, ...]] | None = None
    for combo in itertools.combinations(pool, count):
        wineries = {family.members[0].get("winery", "") for family in combo}
        cross = len(wineries - vintage_wineries)
        total_score = sum(family.selection_score for family in combo)
        key = (cross, len(wineries), total_score, tuple(sorted(family.source_family_id for family in combo)))
        if best is None or key[:3] > best[0][:3] or (key[:3] == best[0][:3] and key[3] < best[0][3]):
            best = (key, tuple(combo))
    return sorted(best[1], key=lambda family: family.source_family_id) if best else list(selected)


def _rename_family(family: FamilyCandidate, pilot_family_id: str) -> FamilyCandidate:
    return FamilyCandidate(
        source_family_id=family.source_family_id,
        family_type=family.family_type,
        members=family.members,
        source_summary={**family.source_summary, "pilot_family_id": pilot_family_id},
        selection_score=family.selection_score,
        classifier_reason=family.classifier_reason,
    )


def _hard_product_rows(
    hard_families: Sequence[FamilyCandidate],
    catalog_by_slug: Mapping[str, Mapping[str, str]],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in hard_families:
        pilot_family_id = family.source_summary["pilot_family_id"]
        evidence = _family_evidence_summary(family.source_summary)
        for order, member in enumerate(family.members, start=1):
            catalog = catalog_by_slug[member["slug"]]
            reason = (
                f"selected from hard_near_duplicate_dev_v2 {family.source_family_id}; "
                f"family_type={family.family_type}; {family.classifier_reason}; {evidence}"
            )
            rows.append({
                "slug": member["slug"],
                "reference_image": catalog["reference_image_path"],
                "product_name": catalog.get("title", member.get("product_name", "")),
                "winery": catalog.get("winery", member.get("winery", "")),
                "category": catalog.get("category", member.get("category", "")),
                "color": catalog.get("color", member.get("color", "")),
                "region": catalog.get("region", member.get("region", "")),
                "grape": catalog.get("grape", member.get("grape", "")),
                "subset_role": "hard",
                "selection_stratum": f"hard_family_type={family.family_type}",
                "selection_seed": seed,
                "representative_selection_reason": "",
                "representative_stratum": "",
                "pilot_family_id": pilot_family_id,
                "target_family_id": pilot_family_id,
                "family_type": family.family_type,
                "family_size": len(family.members),
                "family_member_order": order,
                "hard_selection_reason": reason,
                "source_hard_family_id": family.source_family_id,
            })
    return rows


def build_generation_rows(selected_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for product in sorted(selected_rows, key=lambda row: str(row["slug"])):
        for scenario_id in SCENARIO_IDS:
            rows.append({
                "generation_id": make_generation_id(str(product["slug"]), scenario_id),
                "target_slug": product["slug"],
                "reference_image": product["reference_image"],
                "scenario_id": scenario_id,
                "prompt": SCENARIOS[scenario_id]["prompt_template"],
                "output_filename": f"{make_generation_id(str(product['slug']), scenario_id)}.png",
                "subset_role": product["subset_role"],
                "target_family_id": product.get("target_family_id", ""),
                "family_type": product.get("family_type", ""),
                "family_size": product.get("family_size", ""),
                "family_member_order": product.get("family_member_order", ""),
                "generation_status": "pending",
            })
    return rows


def _family_row(family: FamilyCandidate) -> dict[str, object]:
    summary = family.source_summary
    return {
        "pilot_family_id": summary["pilot_family_id"],
        "source_family_id": family.source_family_id,
        "family_type": family.family_type,
        "family_reason": family.classifier_reason,
        "evidence_summary": _family_evidence_summary(summary),
        "family_size": len(family.members),
        "member_slugs": json.dumps([member["slug"] for member in family.members], ensure_ascii=False),
        "selection_score": f"{family.selection_score:.3f}",
        "source_selection_reason": summary.get("selection_reason", ""),
        "source_edge_count": summary.get("edge_count", ""),
        "source_confusion_count": summary.get("confusion_count", ""),
        "source_confusion_rank_2_5_count": summary.get("confusion_rank_2_5_count", ""),
        "source_small_margin_count": summary.get("small_margin_count", ""),
        "source_target_ranks": summary.get("target_ranks", ""),
        "source_min_margin": summary.get("min_margin", ""),
        "source_max_margin": summary.get("max_margin", ""),
        "source_visual_pair_count": summary.get("visual_pair_count", ""),
        "source_name_pair_count": summary.get("name_pair_count", ""),
        "source_selection_evidence_json": summary.get("selection_evidence_json", ""),
    }


def _metadata(*, catalog_path, hard_families_path, existing_selection_path, seed, representative_source, quotas, hard_families, selected_rows, generation_rows) -> dict[str, object]:
    hard_rows = [row for row in selected_rows if row["subset_role"] == "hard"]
    return {
        "benchmark_version": "generated-stress-pilot32-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "benchmark_name": "generated_stress_dev_pilot32",
        "selection_seed": seed,
        "num_selected_products": len(selected_rows),
        "representative_products": sum(row["subset_role"] == "representative" for row in selected_rows),
        "hard_products": len(hard_rows),
        "hard_families": len(hard_families),
        "hard_vintage_families": sum(family.family_type == "vintage" for family in hard_families),
        "hard_subtype_families": sum(family.family_type == "subtype" for family in hard_families),
        "scenario_ids": list(SCENARIO_IDS),
        "scenarios_per_product": len(SCENARIO_IDS),
        "planned_queries": len(generation_rows),
        "accepted_queries": 0,
        "rejected_queries": 0,
        "pending_queries": len(generation_rows),
        "reference_catalog_fingerprint": sha256_file(catalog_path),
        "reference_catalog_manifest": _relative(catalog_path),
        "existing_generated_stress_selection": _relative(existing_selection_path),
        "hard_source_benchmark": "hard_near_duplicate_dev_v2",
        "hard_source_fingerprint": sha256_file(hard_families_path),
        "representative_selection_source": representative_source,
        "representative_stratification_fields": ["category", "region"],
        "representative_stratification_quotas": {key: value for key, value in sorted(quotas.items())},
        "family_type_rule": {
            "vintage": "two products with distinct explicit years and identical normalized product line after removing year",
            "subtype": "same winery/product-line tokens with v2 evidence and a meaningful category/color/grape or name-variant difference",
        },
        "product_selection_model_informed": True,
        "representative_selection_model_informed": False,
        "hard_selection_uses_frozen_v2_evidence": True,
        "image_generation_api_called": False,
        "generated_images_created": False,
        "full_generated_stress_dev_modified": False,
        "synthetic_dev_modified": False,
        "hard_near_duplicate_dev_v2_modified": False,
    }


def selection_report(*, usable_rows, existing_rows, representatives, selected_rows, hard_families, quotas, seed, catalog_path, hard_families_path, representative_source) -> str:
    full = {field: Counter(str(row.get(field, "")).strip() for row in usable_rows) for field in ("category", "region", "winery", "color", "grape")}
    current = {field: Counter(str(row.get(field, "")).strip() for row in existing_rows) for field in ("category", "region", "winery", "color", "grape")}
    representative = {field: Counter(str(row.get(field, "")).strip() for row in representatives) for field in ("category", "region", "winery", "color", "grape")}
    combined = {field: Counter(str(row.get(field, "")).strip() for row in selected_rows) for field in ("category", "region", "winery", "color", "grape")}
    lines = [
        "# generated_stress_dev_pilot32 selection report",
        "",
        "Изолированный budgeted pilot plan. Скрипт только читает существующие артефакты; paid generation не запускалась, generated images не создавались, full `generated_stress_dev` не изменялся.",
        "",
        "## Design",
        "",
        f"- Fixed seed: **{seed}**.",
        "- Products: **16 representative + 16 hard = 32 unique products**.",
        "- Hard structure: **8 families × 2 products = 16 hard products**.",
        "- Hard type split: **4 vintage families / 8 products** and **4 subtype families / 8 products**.",
        f"- Planned generations: **{len(selected_rows) * len(SCENARIO_IDS)} = 32 × 4**.",
        f"- Representative source: `{representative_source}`.",
        f"- Hard source: `{_relative(hard_families_path)}`; selection uses frozen v2 evidence, not a new model run.",
        "",
        "Representative products cover the capture/domain-shift axis without deliberately concentrating the sample in v2 hard families. Hard products add a controlled family-aware diagnostic axis; keeping the halves separate allows both effects to be reported independently.",
        "",
        "## Selection rules",
        "",
        "- Representative: proportional `category × region` quotas, deterministic winery round-robin, fixed SHA-256 ordering inherited from the existing generated-stress selector; all v2 hard-family products are excluded.",
        "- Vintage family: exactly two products, two distinct explicit years, identical normalized product line after removing the year, and at least two v2 strong signals.",
        "- Subtype family: exactly two products, same winery/product-line tokens, at least two v2 strong signals, and a meaningful category/color/grape or name-variant difference.",
        "- Winery diversity is maximized before evidence score; ties are resolved by source family id.",
        "- Metadata-only equality is not sufficient to create a hard family.",
        "",
        "## Catalog and subset distributions",
        "",
        f"Usable catalog: **{len(usable_rows)}** products. Existing full generated-stress selection: **{len(existing_rows)}**. Representative subset: **{len(representatives)}**. Combined pilot32: **{len(selected_rows)}**.",
        "",
        "### Category",
        "",
        "| value | catalog | current 150 | representative | pilot32 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(_distribution_table(full["category"], current["category"], representative["category"], combined["category"], len(usable_rows), len(existing_rows), len(representatives), len(selected_rows)))
    lines.extend([
        "",
        "### Region",
        "",
        "| value | catalog | current 150 | representative | pilot32 |",
        "|---|---:|---:|---:|---:|",
    ])
    lines.extend(_distribution_table(full["region"], current["region"], representative["region"], combined["region"], len(usable_rows), len(existing_rows), len(representatives), len(selected_rows)))
    lines.extend([
        "",
        "### Winery concentration",
        "",
        "| metric | catalog | current 150 | representative | pilot32 |",
        "|---|---:|---:|---:|---:|",
        f"| unique wineries | {len(full['winery'])} | {len(current['winery'])} | {len(representative['winery'])} | {len(combined['winery'])} |",
        f"| largest winery count | {max(full['winery'].values())} | {max(current['winery'].values())} | {max(representative['winery'].values())} | {max(combined['winery'].values())} |",
        f"| top-5 winery share | {_share(sum(value for _, value in full['winery'].most_common(5)), len(usable_rows))} | {_share(sum(value for _, value in current['winery'].most_common(5)), len(existing_rows))} | {_share(sum(value for _, value in representative['winery'].most_common(5)), len(representatives))} | {_share(sum(value for _, value in combined['winery'].most_common(5)), len(selected_rows))} |",
        "",
        "### Representative quotas",
        "",
        "| category × region stratum | quota |",
        "|---|---:|",
    ])
    lines.extend(f"| `{key}` | {quotas[key]} |" for key in sorted(quotas))
    lines.extend(["", "## Hard families", "", "| pilot family | source family | type | products | source signals | confusion | rank 2–5 | margin count | visual pairs | name pairs |", "|---|---|---|---|---|---:|---:|---:|---:|---:|"])
    for family in hard_families:
        summary = family.source_summary
        lines.append(
            f"| `{summary['pilot_family_id']}` | `{family.source_family_id}` | `{family.family_type}` | {', '.join(member['product_name'] for member in family.members)} | `{summary.get('selection_reason', '')}` | {summary.get('confusion_count', '')} | {summary.get('confusion_rank_2_5_count', '')} | {summary.get('small_margin_count', '')} | {summary.get('visual_pair_count', '')} | {summary.get('name_pair_count', '')} |"
        )
        lines.append(f"|  |  |  |  | {family.classifier_reason}; margin=[{summary.get('min_margin', '')}, {summary.get('max_margin', '')}]; target_ranks={summary.get('target_ranks', '')} |  |  |  |  |  |")
    lines.extend(["", "## All 32 products", "", "| role | slug | product | winery | category | region | family | type | selection reason |", "|---|---|---|---|---|---|---|---|---|"])
    for row in sorted(selected_rows, key=lambda value: (str(value["subset_role"]), str(value["slug"]))):
        reason = row.get("representative_selection_reason") or row.get("hard_selection_reason") or ""
        lines.append(
            f"| {row['subset_role']} | `{row['slug']}` | {row['product_name']} | {row['winery']} | {row['category']} | {row['region']} | {row.get('target_family_id', '')} | {row.get('family_type', '')} | {reason} |"
        )
    lines.extend([
        "",
        "## Planned generations",
        "",
        "| group | products | scenarios per product | planned generations |",
        "|---|---:|---:|---:|",
        f"| representative | 16 | 4 | 64 |",
        f"| hard vintage | 8 | 4 | 32 |",
        f"| hard subtype | 8 | 4 | 32 |",
        f"| total | 32 | 4 | **128** |",
        "",
        "Every row starts with `generation_status=pending`; generation IDs are derived deterministically from slug and scenario. The plan is intentionally separate from the existing 150-product full plan, and no generated image is accepted or scored by this milestone.",
        "",
        "## Why pilot32 fits the budget",
        "",
        "It reduces the paid scope from 600 planned requests to 128 while retaining all four capture scenarios, a non-hard representative control half, and family-aware hard diagnostics. The 16+16 split makes it possible to compare capture robustness on ordinary products against capture robustness on already difficult near-duplicate families without collapsing the two axes into one score.",
        "",
        "## Reproducibility",
        "",
        f"- Catalog source: `{_relative(catalog_path)}`.",
        f"- Seed: `{seed}`.",
        "- Selection and generation IDs use stable lexical/SHA-256 ordering; Python hash/random state is not used.",
        "- Re-running the builder with the same inputs and seed reproduces selected slugs, family IDs, family membership, and generation IDs.",
    ])
    return "\n".join(lines) + "\n"


def _distribution_table(full, current, representative, combined, full_total, current_total, representative_total, combined_total):
    values = sorted(set(full) | set(current) | set(representative) | set(combined))
    return [
        f"| {value or '(empty)'} | {full.get(value, 0)} ({_share(full.get(value, 0), full_total)}) | {current.get(value, 0)} ({_share(current.get(value, 0), current_total)}) | {representative.get(value, 0)} ({_share(representative.get(value, 0), representative_total)}) | {combined.get(value, 0)} ({_share(combined.get(value, 0), combined_total)}) |"
        for value in values
    ]


def _family_evidence_summary(summary: Mapping[str, str]) -> str:
    return (
        f"signals={summary.get('selection_reason', '')}; confusion={summary.get('confusion_count', '')}; "
        f"rank_2_5={summary.get('confusion_rank_2_5_count', '')}; "
        f"small_margin={summary.get('small_margin_count', '')}; "
        f"target_ranks={summary.get('target_ranks', '')}; "
        f"margin=[{summary.get('min_margin', '')},{summary.get('max_margin', '')}]; "
        f"visual_pairs={summary.get('visual_pair_count', '')}; name_pairs={summary.get('name_pair_count', '')}"
    )


def _source_signals(summary: Mapping[str, str]) -> list[str]:
    return [value for value in summary.get("selection_reason", "").split("+") if value]


def _years(row: Mapping[str, str]) -> set[str]:
    values = " ".join([row.get("year_if_known", ""), row.get("product_name", ""), row.get("slug", "")])
    return set(YEAR_RE.findall(values))


def _name_without_year(value: str) -> str:
    value = YEAR_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold())
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _tokens(value: str) -> set[str]:
    normalized = _name_without_year(value)
    return {token for token in normalized.split() if len(token) > 1}


def _line_tokens(row: Mapping[str, str]) -> set[str]:
    tokens = _tokens(row.get("product_name", ""))
    winery_tokens = _tokens(row.get("winery", ""))
    return tokens - winery_tokens - GENERIC_NAME_TOKENS


def _variant_tokens(row: Mapping[str, str]) -> set[str]:
    return _tokens(row.get("product_name", "")) - _tokens(row.get("winery", ""))


def _int(value: object) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _share(value: int, total: int) -> str:
    return f"{value / total * 100:.1f}%" if total else "—"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _readme(metadata: Mapping[str, object]) -> str:
    return f"""# generated_stress_dev_pilot32

Budgeted generated stress pilot with **16 representative** and **16 hard**
products, **8 hard families × 2 products**, and **{metadata['planned_queries']}**
planned generations. This directory is isolated from the existing
`generated_stress_dev` 150-product plan.

No paid generation is run by the builder. All rows in `generation_manifest.csv`
start as `generation_status=pending`.

## Inspect and later generate

First inspect the exact scope without an API call:

```bash
.venv/bin/python scripts/generate_stress_aitunnel.py \\
  --manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \\
  --all \\
  --generated-dir data/benchmarks/generated_stress_dev_pilot32/generated_raw \\
  --runs-dir data/benchmarks/generated_stress_dev_pilot32/generation_runs \\
  --dry-run
```

If generation is explicitly authorized later, remove `--dry-run` from the same
command. The generator reads `AITUNNEL_API_KEY` only from the environment.

## Review and scoring workflow

```bash
.venv/bin/python scripts/import_generated_stress.py \\
  --generation-manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \\
  --selected-products data/benchmarks/generated_stress_dev_pilot32/selected_products.csv \\
  --generated-dir data/benchmarks/generated_stress_dev_pilot32/generated_raw \\
  --review-csv data/benchmarks/generated_stress_dev_pilot32/review.csv \\
  --review-html data/benchmarks/generated_stress_dev_pilot32/review.html

# manually set only identity-preserving rows to accepted
.venv/bin/python scripts/build_generated_stress_benchmark.py \\
  --generation-manifest data/benchmarks/generated_stress_dev_pilot32/generation_manifest.csv \\
  --review-csv data/benchmarks/generated_stress_dev_pilot32/review.csv \\
  --catalog-manifest data/processed/catalog_manifest.csv \\
  --output-dir data/benchmarks/generated_stress_dev_pilot32

.venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev_pilot32
.venv/bin/python scripts/evaluate_benchmark.py \\
  --benchmark generated_stress_dev_pilot32 \\
  --predictions artifacts/experiments/<run_id>/predictions.csv
```

The scored manifest can contain only `accepted` and valid, non-reference-copy
images. Family fields in the manifest enable family-aware diagnostics for the
hard half.
"""


if __name__ == "__main__":
    main()
