#!/usr/bin/env python3
"""Manual oracle-crop diagnostic for pilot32 (Part J tooling).

Purpose: measure the theoretical upper bound if the bottle were perfectly
localized by a human. A human fills bboxes in

    data/benchmarks/generated_stress_dev_pilot32/oracle_crop_bboxes.csv

with columns ``query_id,x1,y1,x2,y2`` (pixel coordinates in the original
query image). Running this script crops each bbox deterministically, encodes
the crop with the same frozen SigLIP2, and writes a full-catalog ranking to

    artifacts/experiments/manual_oracle_crop_<timestamp>/

Rules:
- If the bbox CSV is missing or contains no rows, nothing is computed.
- Bboxes are provided manually; none are invented automatically.
- Results are named ``manual_oracle_crop`` and must never be pooled with
  normal benchmark metrics; they are an oracle diagnostic only.
- The frozen encoder, catalog, and benchmark are not modified.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.cache import (  # noqa: E402
    EmbeddingCacheMismatch,
    load_embedding_cache,
    save_embedding_cache,
    sha256_file,
)
from recognition.crop_diagnostics import (  # noqa: E402
    full_ranking,
    rank_metrics,
    target_rank_full,
)
from recognition.encoder import VisualEncoder  # noqa: E402
from recognition.preprocessing import load_rgb_image  # noqa: E402
from recognition.reporting import write_csv  # noqa: E402
from scripts.run_siglip2_baseline import (  # noqa: E402
    _catalog_items,
    _path,
    _read_csv,
)

DEFAULT_BBOX_CSV = "data/benchmarks/generated_stress_dev_pilot32/oracle_crop_bboxes.csv"
BBOX_COLUMNS = ["query_id", "x1", "y1", "x2", "y2", "note"]


def create_bbox_template(path: Path) -> None:
    """Create a header-only bbox template without inventing any boxes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, BBOX_COLUMNS, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual oracle crop diagnostic (no auto boxes)")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--catalog-version", default="data/processed/catalog_v1.json")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument(
        "--benchmark-manifest",
        default="data/benchmarks/generated_stress_dev_pilot32/manifest.csv",
    )
    parser.add_argument("--bbox-csv", default=DEFAULT_BBOX_CSV)
    parser.add_argument("--cache", default="artifacts/catalog_embeddings/siglip2_catalog.pt")
    parser.add_argument("--artifacts-root", default="artifacts/experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    bbox_path = _path(project_root, args.bbox_csv)
    if not bbox_path.is_file():
        create_bbox_template(_path(project_root, DEFAULT_BBOX_CSV))
        print(
            "No oracle bboxes provided. A header-only template was created at "
            f"{_path(project_root, DEFAULT_BBOX_CSV).relative_to(project_root)}. "
            "Fill query_id,x1,y1,x2,y2 manually and rerun. Nothing was computed."
        )
        return
    bbox_rows = _read_csv(bbox_path)
    if not bbox_rows:
        print("Oracle bbox CSV exists but has no rows; nothing was computed.")
        return

    manifest_path = _path(project_root, args.benchmark_manifest)
    manifest_rows = {row["query_id"]: row for row in _read_csv(manifest_path)}
    catalog_meta = json.loads(
        _path(project_root, args.catalog_version).read_text(encoding="utf-8")
    )
    catalog_items, _ = _catalog_items(
        project_root, _read_csv(_path(project_root, args.catalog_manifest)), catalog_meta
    )
    catalog_slugs = [item.item_id for item in catalog_items]

    encoder = VisualEncoder(model_name="siglip", device=args.device, batch_size=args.batch_size)
    expected_cache_metadata = {
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "catalog_version": catalog_meta["catalog_version"],
        "catalog_manifest_sha256": catalog_meta["manifest_sha256"],
        "preprocessing_config": encoder.preprocessing_config,
        "catalog_count": len(catalog_items),
        "embedding_dim": encoder.embedding_dim,
        "catalog_item_ids": [item.item_id for item in catalog_items],
    }
    try:
        catalog_embeddings = load_embedding_cache(_path(project_root, args.cache), expected_cache_metadata)
    except (FileNotFoundError, EmbeddingCacheMismatch) as exc:
        print(f"Catalog embedding cache not reused: {exc}; computing it now.")
        catalog_embeddings = encoder.encode([item.image_path for item in catalog_items])
        save_embedding_cache(_path(project_root, args.cache), catalog_embeddings, expected_cache_metadata)

    import torch

    device = encoder.device
    catalog_matrix = torch.nn.functional.normalize(
        torch.tensor(catalog_embeddings, dtype=torch.float32, device=device), p=2, dim=1
    )

    cropped_images = []
    prepared_rows = []
    for row in bbox_rows:
        query_id = row["query_id"]
        if query_id not in manifest_rows:
            raise ValueError(f"bbox query_id is not in the pilot32 manifest: {query_id}")
        manifest_row = manifest_rows[query_id]
        image = load_rgb_image(_path(project_root, manifest_row["query_path"]))
        try:
            x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"bbox for {query_id} must contain integer x1,y1,x2,y2") from exc
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError(f"bbox for {query_id} is outside the image bounds: {x1},{y1},{x2},{y2}")
        cropped_images.append(image.crop((x1, y1, x2, y2)))
        prepared_rows.append((row, manifest_row))

    embeddings = torch.tensor(
        encoder.encode_pil(cropped_images), dtype=torch.float32, device=device
    )
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    scores = embeddings @ catalog_matrix.transpose(0, 1)
    scores_matrix = scores.detach().cpu().tolist()

    run_id = datetime.now(timezone.utc).strftime("manual_oracle_crop_%Y%m%dT%H%M%SZ")
    run_dir = _path(project_root, args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prediction_rows = []
    ranks = []
    for (bbox_row, manifest_row), score_vector in zip(prepared_rows, scores_matrix):
        ranked = full_ranking(score_vector, catalog_slugs)
        ranked_slugs = [slug for slug, _ in ranked]
        target_slug = manifest_row["target_slug"]
        rank = target_rank_full(ranked_slugs, target_slug)
        ranks.append(rank)
        prediction_rows.append(
            {
                "query_id": bbox_row["query_id"],
                "target_slug": target_slug,
                "predicted_slug": ranked_slugs[0],
                "top1_score": ranked[0][1],
                "top5_slugs": json.dumps(ranked_slugs[:5], separators=(",", ":")),
                "correct_top1": ranked_slugs[0] == target_slug,
                "target_rank": rank if rank is not None else "",
                "bbox": f"{bbox_row['x1']},{bbox_row['y1']},{bbox_row['x2']},{bbox_row['y2']}",
                "note": bbox_row.get("note", ""),
            }
        )

    write_csv(
        run_dir / "predictions.csv",
        ["query_id", "target_slug", "predicted_slug", "top1_score", "top5_slugs", "correct_top1", "target_rank", "bbox", "note"],
        prediction_rows,
    )
    metrics = rank_metrics(ranks)
    config = {
        "run_id": run_id,
        "diagnostic_name": "manual_oracle_crop",
        "warning": "manual oracle diagnostic only; never pool with benchmark metrics",
        "model_id": encoder.model_id,
        "model_version": encoder.model_version,
        "device": device,
        "query_count": len(prepared_rows),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "metrics": metrics,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
