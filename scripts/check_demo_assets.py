#!/usr/bin/env python3
"""Check the repository-local files required by the frozen demo runtime.

This lightweight preflight uses only the Python standard library. The server
still performs deeper startup checks, including every catalog image checksum,
cache shape and OpenCV/SIFT configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    "artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/"
    "checkpoints/epoch_005.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "7ff3bba0cc416629ee159f1692a221295da322480d1c7b41f06d9eea1c41bb28"
ADAPTED_CACHE = Path(
    "artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/"
    "lora_evaluation/adapted_reference_cache"
)
SLUGS_PATH = Path("artifacts/reference_embeddings/siglip2_so400m_384/slugs.json")
OCR_PATH = Path(
    "artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/"
    "ocr/reference_ocr.jsonl"
)
SIFT_CONFIG = Path("artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/config.json")
EXPECTED_MODEL_ID = "google/siglip2-so400m-patch14-384"
EXPECTED_MODEL_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
EXPECTED_LORA_STATE_SHA256 = "b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04"


class Preflight:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.passes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        (self.passes if condition else self.errors).append(message)

    def read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads((ROOT / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.require(False, f"{path}: cannot read valid JSON ({exc})")
            return None
        if not isinstance(value, dict):
            self.require(False, f"{path}: expected a JSON object")
            return None
        return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    check = Preflight()
    manifest_path = ROOT / "data/processed/catalog_manifest.csv"
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        check.require(False, f"data/processed/catalog_manifest.csv: unavailable ({exc})")
        rows = []

    catalog = {
        str(row.get("slug", "")): str(row.get("reference_image_path", ""))
        for row in rows if row.get("slug") and row.get("reference_image_path")
    }
    check.require(bool(catalog), "tracked catalog manifest is readable and has usable references")
    slugs = set(catalog)

    missing_images: list[str] = []
    unsafe_paths: list[str] = []
    for relpath in catalog.values():
        path = (ROOT / relpath).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            unsafe_paths.append(relpath)
            continue
        if not path.is_file():
            missing_images.append(relpath)
    if unsafe_paths:
        check.require(False, f"catalog contains paths outside the repository: {unsafe_paths[:3]}")
    if missing_images:
        suffix = " …" if len(missing_images) > 5 else ""
        check.require(False, f"catalog reference images missing: {len(missing_images)}; first: {missing_images[:5]}{suffix}")
    else:
        check.require(bool(catalog), f"all {len(catalog)} catalog reference images are present")

    checkpoint = ROOT / CHECKPOINT
    if not checkpoint.is_file():
        check.require(False, f"missing frozen LoRA checkpoint: {CHECKPOINT}")
    else:
        check.require(sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256,
                      f"frozen LoRA checkpoint SHA-256 matches ({CHECKPOINT})")

    metadata_path = ADAPTED_CACHE / "metadata.json"
    embeddings_path = ADAPTED_CACHE / "embeddings.npy"
    for path in (metadata_path, embeddings_path):
        check.require((ROOT / path).is_file(), f"required adapted-cache file exists: {path}")
    metadata = check.read_json(metadata_path) if (ROOT / metadata_path).is_file() else None
    if metadata is not None:
        check.require(metadata.get("base_model_id") == EXPECTED_MODEL_ID
                      and metadata.get("base_revision") == EXPECTED_MODEL_REVISION,
                      "adapted cache uses the pinned SigLIP2 model revision")
        check.require(metadata.get("lora_checkpoint_sha256") == EXPECTED_LORA_STATE_SHA256,
                      "adapted cache fingerprint matches the selected R8 LoRA state")
        check.require(metadata.get("reference_count") == len(catalog),
                      f"adapted cache reference count matches the catalog ({len(catalog)})")
        image_hashes = metadata.get("catalog_reference_image_hashes")
        check.require(isinstance(image_hashes, dict) and set(image_hashes) == slugs,
                      "adapted cache slug manifest matches the catalog")

    slug_path = ROOT / SLUGS_PATH
    if not slug_path.is_file():
        check.require(False, f"missing frozen catalog slug list: {SLUGS_PATH}")
    else:
        try:
            cache_slugs = json.loads(slug_path.read_text(encoding="utf-8"))
            check.require(isinstance(cache_slugs, list) and set(cache_slugs) == slugs,
                          "frozen embedding slug coverage matches the catalog")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            check.require(False, f"invalid frozen embedding slug list: {SLUGS_PATH}")

    ocr_path = ROOT / OCR_PATH
    if not ocr_path.is_file():
        check.require(False, f"missing reference OCR cache: {OCR_PATH}")
    else:
        try:
            ocr_slugs = {
                str(item.get("slug", ""))
                for line in ocr_path.read_text(encoding="utf-8").splitlines() if line.strip()
                for item in [json.loads(line)]
            }
            check.require(ocr_slugs == slugs, "reference OCR cache covers the catalog")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            check.require(False, f"invalid reference OCR cache: {OCR_PATH}")

    config = check.read_json(SIFT_CONFIG)
    if config is not None:
        reference_cache = config.get("reference_cache")
        cache_rel = reference_cache.get("path") if isinstance(reference_cache, dict) else None
        cache_path = (ROOT / str(cache_rel)).resolve() if cache_rel else None
        if cache_path is None:
            check.require(False, f"SIFT config has no reference-cache path: {SIFT_CONFIG}")
        else:
            try:
                cache_path.relative_to(ROOT)
            except ValueError:
                check.require(False, "SIFT cache path resolves outside the repository")
                cache_path = None
        if cache_path is not None:
            index_path = cache_path / "cache_index.json"
            index = None
            if not index_path.is_file():
                check.require(False, f"missing frozen SIFT index: {index_path.relative_to(ROOT)}")
            else:
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    check.require(False, f"invalid frozen SIFT index: {index_path.relative_to(ROOT)}")
            sift_count = len(list(cache_path.glob("*.npz")))
            check.require(sift_count == len(catalog),
                          f"frozen SIFT descriptor count matches the catalog ({sift_count}/{len(catalog)})")
            if isinstance(index, dict):
                check.require(index.get("reference_count") == len(catalog)
                              and index.get("slugs_in_order") == list(catalog),
                              "frozen SIFT index order matches the catalog")

    print("Demo asset preflight")
    for message in check.passes:
        print(f"PASS  {message}")
    for message in check.errors:
        print(f"FAIL  {message}")
    print("NOTE  Hugging Face and PaddleOCR model caches are checked by the actual runtime, "
          "not by this standard-library preflight. The pinned SigLIP2 snapshot must be "
          "downloaded before startup; PaddleOCR may download its models on first startup.")
    if check.errors:
        print(f"\nPreflight failed: {len(check.errors)} issue(s). See web/README.md for the asset list and setup.")
        return 1
    print("\nPreflight passed. Start the server; it will perform full checksum/cache validation and warm-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
