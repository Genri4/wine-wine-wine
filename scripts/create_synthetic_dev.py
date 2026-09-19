#!/usr/bin/env python3
"""Create a deterministic, internal synthetic DEV benchmark from references."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.cache import sha256_file  # noqa: E402
from recognition.data import IMAGE_SUFFIXES  # noqa: E402


DEFAULT_SEED = 20260916
TRANSFORM_VERSION = "synthetic-dev-transforms-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic DEV queries")
    parser.add_argument(
        "--catalog-manifest",
        default="data/processed/catalog_manifest.csv",
        help="Canonical catalog manifest, relative to project root",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/synthetic_dev",
        help="Benchmark directory, relative to project root",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variants", type=int, default=2, choices=(2, 3, 4))
    args = parser.parse_args()

    manifest_path = _project_path(args.catalog_manifest)
    output_dir = _project_path(args.output_dir)
    rows = _usable_catalog_rows(manifest_path)
    if not rows:
        raise ValueError("No usable matched catalog rows found")

    queries_dir = output_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "manifest.csv"
    output_rows: list[dict[str, object]] = []
    for product_index, row in enumerate(rows):
        slug = row["slug"]
        source_relative = row["reference_image_path"]
        source_path = _project_path(source_relative)
        if source_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported reference image suffix: {source_path}")
        for variant in range(1, args.variants + 1):
            seed = args.seed + product_index * args.variants + variant - 1
            rng = random.Random(seed)
            query_id = f"{slug}__v{variant:02d}"
            with Image.open(source_path) as opened:
                source_image = opened.convert("RGB")
            # Keep generated queries practical to create and inspect. This is
            # applied to the in-memory copy only; the reference is untouched.
            source_image.thumbnail((1200, 1200), resample=Image.Resampling.LANCZOS)
            transformed, transform_type, transform_params, output_format = _transform(
                source_image, rng, variant
            )
            source_image.close()
            query_name = f"{query_id}.{output_format.lower()}"
            query_path = queries_dir / query_name
            save_kwargs = {"format": output_format}
            if output_format == "WEBP":
                # method=0 keeps the benchmark creation time practical while
                # preserving the requested lossy WebP variant.
                save_kwargs.update(quality=92, method=0)
            else:
                save_kwargs.update(quality=84, optimize=False, progressive=False)
            transformed.save(query_path, **save_kwargs)
            transformed.close()
            output_rows.append(
                {
                    "query_id": query_id,
                    "query_path": _relative(query_path),
                    "target_slug": slug,
                    "source_reference_path": source_relative,
                    "transform_type": transform_type,
                    "transform_params": json.dumps(
                        transform_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                    "seed": seed,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "query_id",
                "query_path",
                "target_slug",
                "source_reference_path",
                "transform_type",
                "transform_params",
                "seed",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "benchmark_version": "synthetic-dev-v1",
        "transform_version": TRANSFORM_VERSION,
        "seed": args.seed,
        "variants_per_product": args.variants,
        "products": len(rows),
        "queries": len(output_rows),
        "source_catalog_manifest": _relative(manifest_path),
        "source_catalog_manifest_sha256": sha256_file(manifest_path),
        "manifest_sha256": sha256_file(output_manifest),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "internal_comparative_benchmark_only": True,
        "reference_images_unchanged": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _usable_catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"slug", "reference_image_path", "mapping_status"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Catalog manifest missing required columns: {sorted(required)}")
    usable = []
    for row in rows:
        reference = row.get("reference_image_path", "").strip()
        if row.get("mapping_status") == "matched" and reference and _project_path(reference).is_file():
            usable.append(row)
    return sorted(usable, key=lambda row: row["slug"])


def _transform(
    image: Image.Image, rng: random.Random, variant: int
) -> tuple[Image.Image, str, dict[str, object], str]:
    width, height = image.size
    if variant == 1:
        dx = max(1, int(width * rng.uniform(0.01, 0.035)))
        dy = max(1, int(height * rng.uniform(0.01, 0.035)))
        quad = (
            dx,
            dy,
            dx,
            height - dy,
            width - dx,
            height - dy,
            width - dx,
            dy,
        )
        image = image.transform(
            (width, height), Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC
        )
        angle = rng.uniform(-3.0, 3.0)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        brightness = rng.uniform(0.94, 1.06)
        contrast = rng.uniform(0.94, 1.06)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return image, "perspective_rotation_photometric_webp", {
            "perspective_margin_x_px": dx,
            "perspective_margin_y_px": dy,
            "rotation_deg": round(angle, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "output": "webp",
            "quality": 92,
        }, "WEBP"

    crop_fraction = rng.uniform(0.015, 0.05)
    left = int(width * crop_fraction)
    top = int(height * crop_fraction)
    right = max(left + 1, width - left)
    bottom = max(top + 1, height - top)
    image = image.crop((left, top, right, bottom))
    scale = rng.uniform(0.68, 0.82)
    small_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    image = image.resize(small_size, resample=Image.Resampling.LANCZOS)
    blur_radius = rng.uniform(0.15, 0.45)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    angle = rng.uniform(-2.0, 2.0)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    return image, "crop_downscale_blur_jpeg", {
        "crop_fraction": round(crop_fraction, 4),
        "crop_box": [left, top, right, bottom],
        "scale": round(scale, 4),
        "blur_radius": round(blur_radius, 4),
        "rotation_deg": round(angle, 4),
        "output": "jpeg",
        "quality": 84,
    }, "JPEG"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    main()
