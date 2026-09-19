#!/usr/bin/env python3
"""Create a deterministic generated-stress benchmark plan.

This script only selects canonical products and writes prompts/manifests for a
later external image-generation step. It never calls an image API and never
creates generated images.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.cache import sha256_file  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402


DEFAULT_SEED = 20260916
DEFAULT_NUM_PRODUCTS = 150
SCENARIO_IDS = ("slight_angle", "glare_bad_light", "distance_crop", "handheld")
STRATIFICATION_FIELDS = ("category", "region")

IDENTITY_PRESERVATION = (
    "Preserve the exact original wine bottle and product identity. "
    "Preserve the exact label design, winery logo, wine name, vintage/year, "
    "grape information and all printed text. Do not redesign, rewrite, "
    "replace or invent any part of the label. Only change the photography "
    "conditions and surrounding environment."
)

SCENARIOS = {
    "slight_angle": {
        "scenario_id": "slight_angle",
        "title": "Slight angle",
        "prompt_template": (
            f"{IDENTITY_PRESERVATION} Use the attached catalog reference as the source. "
            "Create a realistic smartphone photograph of the same bottle at a slight "
            "10–30 degree yaw with mild perspective distortion. Keep the bottle and "
            "front label clearly visible and recognizable. Use natural handheld framing "
            "and moderate indoor ambient light. Do not introduce any other bottle or "
            "change the product."
        ),
        "constraints": ["yaw approximately 10–30 degrees", "mild perspective distortion", "label remains readable"],
        "negative_constraints": ["no 60–90 degree angle", "no label redesign", "no product identity change"],
    },
    "glare_bad_light": {
        "scenario_id": "glare_bad_light",
        "title": "Glare and bad light",
        "prompt_template": (
            f"{IDENTITY_PRESERVATION} Use the attached catalog reference as the source. "
            "Create a realistic smartphone photo taken in a shop or home under warm, "
            "uneven indoor lighting. Add a plausible mild glass reflection or glare "
            "on part of the bottle, but keep the front label and printed identity "
            "readable to a person. Keep the scene moderately difficult, not extreme, "
            "and do not introduce any other bottle or change the product."
        ),
        "constraints": ["warm uneven indoor lighting", "mild plausible glass glare", "label remains readable"],
        "negative_constraints": ["no fully obscured label", "no extreme bloom", "no label redesign"],
    },
    "distance_crop": {
        "scenario_id": "distance_crop",
        "title": "Distance and crop",
        "prompt_template": (
            f"{IDENTITY_PRESERVATION} Use the attached catalog reference as the source. "
            "Create a realistic smartphone photograph from farther away, with the "
            "bottle occupying a smaller part of the frame and believable shelf or "
            "store surroundings. Include a slightly imperfect crop and natural depth "
            "or background context while keeping the complete bottle and front label "
            "recognizable. Do not introduce another bottle or change the product."
        ),
        "constraints": ["bottle smaller in frame", "realistic shelf/background", "slightly imperfect crop"],
        "negative_constraints": ["no unreadably tiny bottle", "no missing front label", "no product identity change"],
    },
    "handheld": {
        "scenario_id": "handheld",
        "title": "Handheld smartphone",
        "prompt_template": (
            f"{IDENTITY_PRESERVATION} Use the attached catalog reference as the source. "
            "Create an ordinary imperfect handheld smartphone photo of the same bottle "
            "with a mild tilt, subtle motion blur, slightly imperfect framing and a "
            "plausible everyday background. Keep the bottle, front label and printed "
            "identity sufficiently clear for a person to recognize. Do not introduce "
            "another bottle or change the product."
        ),
        "constraints": ["mild motion blur", "small tilt", "imperfect handheld framing", "realistic background"],
        "negative_constraints": ["no severe blur", "no fully blocked label", "no product identity change"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generated_stress_dev selection and generation plan")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--output-dir", default="data/benchmarks/generated_stress_dev")
    parser.add_argument("--num-products", type=int, default=DEFAULT_NUM_PRODUCTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.num_products < 1:
        parser.error("--num-products must be positive")

    catalog_path = _path(args.catalog_manifest)
    output_dir = _path(args.output_dir)
    rows = _read_csv(catalog_path)
    usable = _usable_rows(rows)
    if args.num_products > len(usable):
        parser.error(f"--num-products cannot exceed usable catalog size ({len(usable)})")
    selected, quotas = select_products(usable, args.num_products, args.seed)
    plan_rows = build_generation_rows(selected, SCENARIOS)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_raw").mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "selected_products.csv",
        ["slug", "reference_image", "product_name", "winery", "category", "color", "region", "grape", "selection_stratum", "selection_seed"],
        selected,
    )
    (output_dir / "scenarios.json").write_text(
        json.dumps({"scenario_version": "generated-stress-scenarios-v1", "scenarios": [SCENARIOS[key] for key in SCENARIO_IDS]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "generation_manifest.csv",
        ["generation_id", "target_slug", "reference_image", "scenario_id", "prompt", "output_filename", "generation_status"],
        plan_rows,
    )
    metadata = {
        "benchmark_version": "generated-stress-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_seed": args.seed,
        "requested_products": args.num_products,
        "num_selected_products": len(selected),
        "scenario_ids": list(SCENARIO_IDS),
        "scenarios_per_product": len(SCENARIO_IDS),
        "planned_queries": len(plan_rows),
        "accepted_queries": 0,
        "rejected_queries": 0,
        "pending_queries": len(plan_rows),
        "reference_catalog_fingerprint": sha256_file(catalog_path),
        "reference_catalog_manifest": _relative(catalog_path),
        "stratification_fields": list(STRATIFICATION_FIELDS),
        "stratification_quotas": {key: value for key, value in sorted(quotas.items())},
        "product_selection_not_model_informed": True,
        "image_generation_api_called": False,
        "generated_images_created": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_readme(metadata), encoding="utf-8")
    report_path = PROJECT_ROOT / "reports/generated_stress_dev_selection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_selection_report(usable, selected, quotas, args.seed, catalog_path), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def select_products(rows: list[dict[str, str]], num_products: int, seed: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select a proportional category×region sample with winery round-robin diversity."""

    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = _stratum_key(row)
        strata[key].append(row)
    quotas = _allocate_quotas(strata, num_products)
    selected: list[dict[str, object]] = []
    for key in sorted(strata):
        quota = quotas.get(key, 0)
        if not quota:
            continue
        candidates = _round_robin_wineries(strata[key], seed)
        for row in candidates[:quota]:
            selected.append(_selected_row(row, key, seed))
    selected.sort(key=lambda row: row["slug"])
    if len(selected) != num_products:
        raise AssertionError(f"selection produced {len(selected)} products, expected {num_products}")
    return selected, quotas


def build_generation_rows(selected: Iterable[Mapping[str, object]], scenarios: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for product in sorted(selected, key=lambda row: str(row["slug"])):
        for scenario_id in SCENARIO_IDS:
            generation_id = make_generation_id(str(product["slug"]), scenario_id)
            rows.append({
                "generation_id": generation_id,
                "target_slug": product["slug"],
                "reference_image": product["reference_image"],
                "scenario_id": scenario_id,
                "prompt": scenarios[scenario_id]["prompt_template"],
                "output_filename": f"{generation_id}.png",
                "generation_status": "pending",
            })
    return rows


def make_generation_id(slug: str, scenario_id: str) -> str:
    safe_slug = re.sub(r"[^a-z0-9]+", "-", slug.casefold()).strip("-") or "product"
    slug_digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]
    return f"gstress-{safe_slug[:96]}-{slug_digest}__{scenario_id}"


def _usable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    required = {"slug", "title", "reference_image_path", "mapping_status", "category", "region", "winery"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Catalog manifest missing required columns: {sorted(required - set(rows[0]) if rows else required)}")
    usable = []
    seen = set()
    for row in rows:
        slug = row.get("slug", "").strip()
        reference = row.get("reference_image_path", "").strip()
        if row.get("mapping_status") != "matched" or not slug or slug in seen or not reference:
            continue
        if not _path(reference).is_file():
            continue
        if not row.get("category", "").strip() or not row.get("region", "").strip() or not row.get("winery", "").strip():
            continue
        seen.add(slug)
        usable.append(row)
    return sorted(usable, key=lambda row: row["slug"])


def _selected_row(row: Mapping[str, str], stratum: str, seed: int) -> dict[str, object]:
    return {
        "slug": row["slug"],
        "reference_image": row["reference_image_path"],
        "product_name": row.get("title", ""),
        "winery": row.get("winery", ""),
        "category": row.get("category", ""),
        "color": row.get("color", ""),
        "region": row.get("region", ""),
        "grape": row.get("grape", ""),
        "selection_stratum": stratum,
        "selection_seed": seed,
    }


def _stratum_key(row: Mapping[str, str]) -> str:
    return f"category={row.get('category', '').strip()}|region={row.get('region', '').strip()}"


def _allocate_quotas(strata: Mapping[str, list[dict[str, str]]], target: int) -> dict[str, int]:
    total = sum(len(rows) for rows in strata.values())
    quotas = {key: min(len(rows), int(target * len(rows) // total)) for key, rows in strata.items()}
    remaining = target - sum(quotas.values())
    fractions = {
        key: (target * len(rows) / total) - quotas[key]
        for key, rows in strata.items()
    }
    while remaining:
        available = [key for key in strata if quotas[key] < len(strata[key])]
        if not available:
            raise AssertionError("quota allocation exhausted before reaching target")
        for key in sorted(available, key=lambda value: (-fractions[value], value)):
            if not remaining:
                break
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1
    return quotas


def _round_robin_wineries(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("winery", "").strip()].append(row)
    for winery in groups:
        groups[winery].sort(key=lambda row: (_stable_key(seed, row["slug"]), row["slug"]))
    winery_order = sorted(groups, key=lambda winery: (_stable_key(seed, winery), winery))
    ordered = []
    for offset in range(max((len(values) for values in groups.values()), default=0)):
        for winery in winery_order:
            values = groups[winery]
            if offset < len(values):
                ordered.append(values[offset])
    return ordered


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _selection_report(usable, selected, quotas, seed, catalog_path) -> str:
    full = {field: Counter(row.get(field, "").strip() for row in usable) for field in ("category", "region", "winery", "color", "grape")}
    sample = {field: Counter(row.get(field, "").strip() for row in selected) for field in ("category", "region", "winery", "color", "grape")}
    lines = [
        "# generated_stress_dev selection audit",
        "",
        "Этот файл описывает только deterministic product selection и generation plan. Image API не вызывался, изображения не создавались, model predictions/errors не использовались.",
        "",
        "## Input and method",
        "",
        f"- Source: `{_relative(catalog_path)}`; usable products: **{len(usable)}**.",
        f"- Selected products: **{len(selected)}**; fixed seed: **{seed}**.",
        "- Primary strata: `category × region` — оба поля заполнены, компактны и семантически устойчивы.",
        "- `winery` используется для deterministic round-robin внутри каждой страты, чтобы крупные producers не заняли весь sample.",
        "- `color` не использован как stratum: 822 unique values на 2042 products, поле слишком granular для устойчивых квот.",
        "- `grape` не использован как stratum: 328 unique values и sparse multi-grape values; он полезен как описательная metadata, но noisy для квот.",
        "- Selection order uses SHA-256(seed, value), not Python hash or model output; output rows are slug-sorted.",
        "",
        "## Metadata profile",
        "",
        "| field | unique non-empty values | empty |",
        "|---|---:|---:|",
        f"| category | {len(full['category'])} | {full['category'].get('', 0)} |",
        f"| region | {len(full['region'])} | {full['region'].get('', 0)} |",
        f"| winery | {len(full['winery'])} | {full['winery'].get('', 0)} |",
        f"| color | {len(full['color'])} | {full['color'].get('', 0)} |",
        f"| grape | {len(full['grape'])} | {full['grape'].get('', 0)} |",
        "",
        "## Category distribution",
        "",
        "| value | catalog | catalog share | selected | selected share |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(_distribution_rows(full["category"], sample["category"], len(usable), len(selected)))
    lines.extend(["", "## Region distribution", "", "| value | catalog | catalog share | selected | selected share |", "|---|---:|---:|---:|---:|"])
    lines.extend(_distribution_rows(full["region"], sample["region"], len(usable), len(selected)))
    lines.extend(["", "## Winery concentration", "", "| metric | catalog | selected |", "|---|---:|---:|"])
    lines.extend([
        f"| unique wineries | {len(full['winery'])} | {len(sample['winery'])} |",
        f"| largest winery count | {max(full['winery'].values())} | {max(sample['winery'].values())} |",
        f"| top-10 winery share | {_share(sum(count for _, count in full['winery'].most_common(10)), len(usable))} | {_share(sum(count for _, count in sample['winery'].most_common(10)), len(selected))} |",
    ])
    lines.extend(["", "## Stratum quotas", "", "| stratum | quota |", "|---|---:|"])
    lines.extend(f"| `{key}` | {quotas[key]} |" for key in sorted(quotas))
    lines.extend([
        "",
        "## Planned generation contract",
        "",
        f"- Four scenarios per product: `{', '.join(SCENARIO_IDS)}`.",
        f"- Planned generations: **{len(selected) * len(SCENARIO_IDS)}**.",
        "- All rows start with `generation_status=pending`.",
        "- External generator must write exactly `generated_raw/<output_filename>`; see `generation_manifest.csv`.",
        "",
    ])
    return "\n".join(lines)


def _distribution_rows(full, sample, full_total, sample_total):
    rows = []
    for value in sorted(full):
        rows.append(f"| {value or '(empty)'} | {full[value]} | {_share(full[value], full_total)} | {sample.get(value, 0)} | {_share(sample.get(value, 0), sample_total)} |")
    return rows


def _readme(metadata: Mapping[str, object]) -> str:
    return f"""# generated_stress_dev

Infrastructure-only generative stress benchmark. The plan contains
**{metadata['num_selected_products']}** products × **{len(SCENARIO_IDS)}** scenarios =
**{metadata['planned_queries']}** pending generations. No image API is connected
and no images are generated by this repository.

## External generation contract

For each row in `generation_manifest.csv`, use the attached
`reference_image` and prompt, then write the result to:

`data/benchmarks/generated_stress_dev/generated_raw/<output_filename>`

Do not rename files or change `generation_id`. The generation plan is not yet a
scored benchmark. Every output must pass importer validation and manual review.

## AITUNNEL pilot

Use `scripts/generate_stress_aitunnel.py` only after inspecting its dry-run.
The default pilot is the first 5 products from this existing plan, all four
scenarios (20 requests). It uses the documented
the documented [AITUNNEL image-edit endpoint](https://docs.aitunnel.ru/features/image-editing)
`https://api.aitunnel.ru/v1/images/generations` image-edit endpoint with
`model=gpt-image-2.5-sunburst`, `quality=low`, and `size=1024x1024`.
`AITUNNEL_API_KEY` is read only from the environment. The script does not
rewrite this plan; status, retries, usage and reported cost are runtime data in
`generation_runs/<run_id>/`.

```bash
.venv/bin/python scripts/generate_stress_aitunnel.py --pilot-products 5 --dry-run
# after checking the dry-run:
.venv/bin/python scripts/generate_stress_aitunnel.py --pilot-products 5
```

The explicit `--all` option is reserved for a future, deliberate full-plan
run. A valid existing output is skipped unless `--overwrite` is supplied.

## Workflow

```text
generation_manifest.csv
  → external generation into generated_raw/
  → .venv/bin/python scripts/import_generated_stress.py
  → edit review.csv: review_status=accepted/rejected/pending
  → .venv/bin/python scripts/build_generated_stress_benchmark.py
  → .venv/bin/python scripts/run_siglip2_baseline.py --benchmark generated_stress_dev
  → .venv/bin/python scripts/evaluate_benchmark.py --benchmark generated_stress_dev --predictions <run>/predictions.csv
```

`pending` and `rejected` rows can never enter `manifest.csv`. Generated images
must not be used for training/fine-tuning while this benchmark is frozen.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _share(value: int, total: int) -> str:
    return f"{value / total * 100:.2f}%" if total else "—"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
