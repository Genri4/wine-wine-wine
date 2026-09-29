#!/usr/bin/env python3
"""Rebuild candidate-constrained vintage crops deterministically.

From the frozen reference_guided_per_query.csv (previous milestone) this
re-extracts, without re-running SIFT:

- query year crops (projected, via saved crop_box)
- reference year crops (via saved reference year_box)

Then every method (OCR / visual matching / digit recognizer / VLM) scores
the same crop set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from scripts.run_siglip2_baseline import _path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild constrained vintage crops")
    parser.add_argument("--rg-csv", default="artifacts/experiments/vintage_disambiguation_20260922T1/reference_guided_per_query.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(project_root / args.rg_csv, encoding="utf-8")))
    manifest_rows = []
    for row in rows:
        benchmark = row["benchmark"]
        query_id = row["query_id"]
        evidence = json.loads(row["candidate_evidence"])
        # Query image path from the benchmark manifest.
        manifest_path = (
            project_root / "data" / "benchmarks" / benchmark / "manifest.csv"
        )
        manifest_by_query = {
            m["query_id"]: m for m in csv.DictReader(open(manifest_path, encoding="utf-8-sig"))
        }
        manifest_row = manifest_by_query[query_id]
        query_image_path = _path(project_root, manifest_row["query_path"])

        with Image.open(query_image_path) as query_pil:
            query_pil.load()
            for slug, entry in evidence.items():
                crop_box = entry.get("crop_box")
                if not crop_box:
                    continue
                crop = query_pil.crop(tuple(crop_box))
                crop_dir = out_dir / benchmark / "query"
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"{query_id}__{slug}.png"
                crop.save(crop_dir / crop_name)

                year_box = entry.get("year_box")
                if not year_box:
                    continue
                reference_image_path = _reference_image_path(project_root, slug)
                with Image.open(reference_image_path) as reference_pil:
                    reference_pil.load()
                    reference_crop = reference_pil.crop(
                        (int(year_box[0]), int(year_box[1]), int(year_box[2]), int(year_box[3]))
                    )
                    ref_dir = out_dir / benchmark / "reference"
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    reference_crop.save(ref_dir / f"{slug}.png")
        manifest_rows.append(
            {
                "benchmark": benchmark,
                "query_id": query_id,
                "target_slug": row["target_slug"],
                "target_year": row["target_year"],
                "candidates": json.dumps(
                    {
                        slug: {
                            "year": entry.get("year"),
                            "crop_box": entry.get("crop_box"),
                            "year_box": entry.get("year_box"),
                            "has_reference_crop": bool(entry.get("year_box")),
                        }
                        for slug, entry in evidence.items()
                    },
                    ensure_ascii=False,
                ),
            }
        )

    with (out_dir / "crop_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"crops rebuilt: {len(manifest_rows)} queries -> {out_dir}")


def _reference_image_path(project_root: Path, slug: str) -> Path:
    import csv as csv_module

    with (project_root / "data" / "processed" / "catalog_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv_module.DictReader(stream):
            if row["slug"] == slug:
                return project_root / row["reference_image_path"]
    raise FileNotFoundError(slug)


if __name__ == "__main__":
    main()
