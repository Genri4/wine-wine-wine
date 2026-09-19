#!/usr/bin/env python3
"""Recheck already downloaded web-extra images after a stricter dedup change."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collect_web_extra import (  # noqa: E402
    ASPECT_RATIO_THRESHOLD,
    CENTER_PHASH_HAMMING_THRESHOLD,
    CENTER_THUMBNAIL_DIFF_THRESHOLD,
    PHASH_HAMMING_THRESHOLD,
    THUMBNAIL_DIFF_THRESHOLD,
    _duplicate_check,
    _image_signature,
    _path,
    _read_csv,
    _readme,
    _reference_signatures,
    _write_manifest,
    _write_rejections,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recheck existing web-extra downloads")
    parser.add_argument("--output-dir", default="data/benchmarks/web_extra_dev")
    args = parser.parse_args()
    output_dir = _path(args.output_dir)
    catalog_rows = _read_csv(_path("data/processed/catalog_manifest.csv"))
    catalog = {
        row["slug"]: row
        for row in catalog_rows
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
        and _path(row["reference_image_path"]).is_file()
    }
    references = _reference_signatures(catalog)
    accepted_rows = _read_csv(output_dir / "manifest.csv")
    rejected_rows = _read_csv(output_dir / "rejected.csv")
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for rejected in rejected_rows:
        rejected["manual_review_status"] = "rejected_by_conservative_rules"
    kept: list[dict[str, object]] = []
    newly_rejected: list[dict[str, object]] = []
    for row in accepted_rows:
        signature = _image_signature(_path(row["query_path"]).read_bytes())
        check = _duplicate_check(signature, references, target_slug=row["target_slug"])
        if check["is_reference_duplicate"]:
            rejected = dict(row)
            rejected["retrieval_timestamp"] = rejected.get("retrieval_timestamp") or metadata.get(
                "retrieval_timestamp", ""
            )
            rejected["catalog_slug"] = row["target_slug"]
            rejected["reference_duplicate_check"] = json.dumps(check, ensure_ascii=False, sort_keys=True)
            rejected["rejection_reason"] = check["reason"]
            rejected["manual_review_status"] = "rejected_by_conservative_rules"
            rejected_rows.append(rejected)
            newly_rejected.append(rejected)
        else:
            row["reference_duplicate_check"] = json.dumps(check, ensure_ascii=False, sort_keys=True)
            kept.append(row)

    _write_manifest(output_dir, kept)
    _write_rejections(output_dir / "rejected.csv", rejected_rows)
    metadata["accepted"] = len(kept)
    metadata["rejected_reference_duplicates"] = len(rejected_rows)
    metadata["rechecked_existing_downloads"] = True
    metadata["duplicate_policy"]["phash_hamming_threshold"] = PHASH_HAMMING_THRESHOLD
    metadata["duplicate_policy"]["aspect_ratio_threshold"] = ASPECT_RATIO_THRESHOLD
    metadata["duplicate_policy"]["thumbnail_mean_abs_diff_threshold"] = THUMBNAIL_DIFF_THRESHOLD
    metadata["duplicate_policy"]["center_phash_hamming_threshold"] = CENTER_PHASH_HAMMING_THRESHOLD
    metadata["duplicate_policy"]["center_thumbnail_mean_abs_diff_threshold"] = CENTER_THUMBNAIL_DIFF_THRESHOLD
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(metadata), encoding="utf-8")
    print(json.dumps({"accepted": len(kept), "newly_rejected": len(newly_rejected), "rejected_total": len(rejected_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
