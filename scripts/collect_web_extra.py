#!/usr/bin/env python3
"""Collect conservatively labelled extra images from the official wine site.

Only exact product URLs from the official ``vino-svoe.ru`` wine sitemap are
considered. Images that look like any canonical reference are rejected before
they can enter the scored manifest.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "MyWine-benchmark-collector/1.0 (research; contact project owner)"
SITEMAP_URL = "https://vino-svoe.ru/wines-sitemap.xml"
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
# Average-hash is computed over 32x32 pixels (1024 bits). A small amount of
# resampling/compositing noise can change more than a handful of bits, so the
# threshold is deliberately paired with the aspect-ratio guard below.
PHASH_HAMMING_THRESHOLD = 180
ASPECT_RATIO_THRESHOLD = 0.02
# The official site often serves the same transparent bottle render at a
# different resolution/background. A 6% normalized thumbnail difference is
# intentionally conservative for treating those as the same asset.
THUMBNAIL_DIFF_THRESHOLD = 0.06
# A center crop catches the common case where a website derivative adds a
# decorative background around the same bottle render. It is used together
# with image-distance guards, never as a standalone identity signal.
CENTER_PHASH_HAMMING_THRESHOLD = 240
CENTER_THUMBNAIL_DIFF_THRESHOLD = 0.12
TOKEN_RE = re.compile(r"/wines/([^/?#]+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect official-site web-extra candidates")
    parser.add_argument("--catalog-manifest", default="data/processed/catalog_manifest.csv")
    parser.add_argument("--output-dir", default="data/benchmarks/web_extra_dev")
    parser.add_argument("--sitemap-url", default=SITEMAP_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Deterministic smoke-test limit")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    output_dir = _path(args.output_dir)
    catalog_rows = _read_csv(_path(args.catalog_manifest))
    catalog = {
        row["slug"]: row
        for row in catalog_rows
        if row.get("mapping_status") == "matched" and row.get("reference_image_path")
        and _path(row["reference_image_path"]).is_file()
    }
    references = _reference_signatures(catalog)
    sitemap_bytes = _download(args.sitemap_url)
    sitemap_entries = _parse_sitemap(sitemap_bytes)
    candidates = [entry for entry in sitemap_entries if entry["slug"] in catalog]
    candidates.sort(key=lambda entry: entry["slug"])
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        candidates = candidates[: args.limit]

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    results: dict[int, tuple[str, bytes | None, dict[str, object] | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download, entry["source_url"]): index
            for index, entry in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = ("ok", future.result(), None, None)
            except Exception as exc:
                results[index] = ("failed", None, None, f"{type(exc).__name__}: {exc}")

    for index, entry in enumerate(candidates):
        status, content, _, download_error = results[index]
        base = {
            "query_id": f"web-{entry['slug']}",
            "target_slug": entry["slug"],
            "source_url": entry["source_url"],
            "page_url": entry["page_url"],
            "source_domain": urlparse(entry["source_url"]).netloc,
            "retrieval_timestamp": timestamp,
            "catalog_slug": entry["slug"],
            "identity_method": "exact official sitemap /wines/{slug} match",
            "provenance": "official vino-svoe.ru product sitemap image; exact page slug equals canonical slug",
            "manual_review_status": "not_required",
        }
        if status != "ok" or content is None:
            base.update({
                "candidate_path": "",
                "reference_duplicate_check": "not_run",
                "review_reason": download_error or "download_failed",
                "manual_review_status": "pending",
            })
            review.append(base)
            continue
        try:
            candidate_signature = _image_signature(content)
        except Exception as exc:
            base.update({
                "candidate_path": "",
                "reference_duplicate_check": "not_an_image",
                "review_reason": f"image_decode_failed: {type(exc).__name__}: {exc}",
                "manual_review_status": "pending",
            })
            review.append(base)
            continue
        check = _duplicate_check(candidate_signature, references, target_slug=entry["slug"])
        base["reference_duplicate_check"] = json.dumps(check, ensure_ascii=False, sort_keys=True)
        if check["is_reference_duplicate"]:
            base["rejection_reason"] = check["reason"]
            rejected.append(base)
            continue

        suffix = _suffix_from_url(entry["source_url"])
        output_path = output_dir / "images" / f"{entry['slug']}{suffix}"
        relative_path = _relative(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        base.update({
            "query_id": f"web-{entry['slug']}",
            "query_path": relative_path,
            "manual_review_status": "accepted_by_conservative_rules",
        })
        accepted.append(base)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_dir, accepted)
    _write_rejections(output_dir / "rejected.csv", rejected)
    _write_review(output_dir / "review_candidates.csv", review)
    (output_dir / "review.html").write_text(_review_html(review), encoding="utf-8")
    metadata = {
        "benchmark_version": "web-extra-v1",
        "catalog_manifest": _relative(_path(args.catalog_manifest)),
        "sitemap_url": args.sitemap_url,
        "sitemap_sha256": hashlib.sha256(sitemap_bytes).hexdigest(),
        "exact_catalog_slug_matches": len(candidates),
        "catalog_products_without_exact_sitemap_slug": len(catalog) - len(candidates),
        "downloaded_candidates": len(candidates) - len(review),
        "accepted": len(accepted),
        "rejected_reference_duplicates": len(rejected),
        "manual_review": len(review),
        "workers": args.workers,
        "retrieval_timestamp": timestamp,
        "duplicate_policy": {
            "exact_content_hash": True,
            "perceptual_hash": "32x32 grayscale average hash",
            "phash_hamming_threshold": PHASH_HAMMING_THRESHOLD,
            "aspect_ratio_threshold": ASPECT_RATIO_THRESHOLD,
            "thumbnail_mean_abs_diff_threshold": THUMBNAIL_DIFF_THRESHOLD,
            "center_phash_hamming_threshold": CENTER_PHASH_HAMMING_THRESHOLD,
            "center_thumbnail_mean_abs_diff_threshold": CENTER_THUMBNAIL_DIFF_THRESHOLD,
            "conservative": True,
        },
        "generated_images_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(metadata), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _reference_signatures(catalog: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    signatures = []
    for slug in sorted(catalog):
        path = _path(catalog[slug]["reference_image_path"])
        signatures.append({"slug": slug, **_image_signature(path.read_bytes())})
    return signatures


def _image_signature(content: bytes) -> dict[str, object]:
    with Image.open(io.BytesIO(content)) as opened:
        oriented = ImageOps.exif_transpose(opened)
        original_width, original_height = oriented.size
        if "A" in oriented.getbands():
            alpha = oriented.getchannel("A").point(lambda value: 255 if value > 10 else 0)
            bbox = alpha.getbbox()
            image = oriented.crop(bbox).convert("RGB") if bbox else oriented.convert("RGB")
        else:
            image = oriented.convert("RGB")
    width, height = image.size
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    values = _pixel_values(gray)
    average = sum(values) / len(values)
    phash = sum(1 << index for index, value in enumerate(values) if value >= average)
    thumbnail = image.resize((64, 64), Image.Resampling.LANCZOS)
    center_left = int(width * 0.25)
    center_right = max(center_left + 1, int(width * 0.75))
    center = image.crop((center_left, 0, center_right, height))
    center_gray = center.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    center_values = _pixel_values(center_gray)
    center_average = sum(center_values) / len(center_values)
    center_phash = sum(
        1 << index for index, value in enumerate(center_values) if value >= center_average
    )
    center_thumbnail = center.resize((64, 64), Image.Resampling.LANCZOS)
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": original_width,
        "height": original_height,
        "content_width": width,
        "content_height": height,
        "content_aspect_ratio": width / height if height else 0.0,
        "aspect_ratio": original_width / original_height if original_height else 0.0,
        "phash": phash,
        "thumbnail": thumbnail.tobytes(),
        "center_phash": center_phash,
        "center_thumbnail": center_thumbnail.tobytes(),
    }


def _duplicate_check(
    candidate: dict[str, object],
    references: list[dict[str, object]],
    *,
    target_slug: str | None = None,
) -> dict[str, object]:
    exact_matches = [row["slug"] for row in references if row["sha256"] == candidate["sha256"]]
    best = None
    for reference in references:
        match = _match_details(candidate, reference)
        if best is None or (match["phash_hamming"], match["aspect_ratio_delta"]) < (
            best["phash_hamming"], best["aspect_ratio_delta"]
        ):
            best = match
    assert best is not None
    target_reference = next((row for row in references if row["slug"] == target_slug), None)
    target_match = _match_details(candidate, target_reference) if target_reference else None
    is_near_duplicate = _is_near_duplicate(best)
    # The page's exact product reference must always be checked separately.
    # A visually similar reference from another product can otherwise win the
    # nearest-neighbour diagnostic and hide a target/reference copy.
    is_target_duplicate = bool(target_match and _is_near_duplicate(target_match))
    is_duplicate = bool(exact_matches or is_near_duplicate or is_target_duplicate)
    reason = "exact_content_hash" if exact_matches else (
        "target_reference_duplicate" if is_target_duplicate else (
            "perceptual_duplicate" if is_near_duplicate else "no_conservative_duplicate_match"
        )
    )
    return {
        "is_reference_duplicate": is_duplicate,
        "reason": reason,
        "exact_reference_slugs": exact_matches,
        "target_slug": target_slug or "",
        "target_match": target_match,
        **best,
    }


def _match_details(candidate: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    return {
        "reference_slug": reference["slug"],
        "phash_hamming": (int(candidate["phash"]) ^ int(reference["phash"])).bit_count(),
        "aspect_ratio_delta": abs(float(candidate["aspect_ratio"]) - float(reference["aspect_ratio"])),
        "thumbnail_mean_abs_diff": _thumbnail_diff(candidate["thumbnail"], reference["thumbnail"]),
        "content_aspect_ratio_delta": abs(
            float(candidate["content_aspect_ratio"]) - float(reference["content_aspect_ratio"])
        ),
        "center_phash_hamming": (
            int(candidate["center_phash"]) ^ int(reference["center_phash"])
        ).bit_count(),
        "center_thumbnail_mean_abs_diff": _thumbnail_diff(
            candidate["center_thumbnail"], reference["center_thumbnail"]
        ),
    }


def _is_near_duplicate(match: dict[str, object]) -> bool:
    return (
        match["aspect_ratio_delta"] <= ASPECT_RATIO_THRESHOLD
        and (
            match["phash_hamming"] <= PHASH_HAMMING_THRESHOLD
            or (
                match["thumbnail_mean_abs_diff"] <= THUMBNAIL_DIFF_THRESHOLD
                and match["content_aspect_ratio_delta"] <= 0.03
            )
            or (
                match["center_phash_hamming"] <= CENTER_PHASH_HAMMING_THRESHOLD
                and match["center_thumbnail_mean_abs_diff"] <= CENTER_THUMBNAIL_DIFF_THRESHOLD
                and match["content_aspect_ratio_delta"] <= 0.03
            )
        )
    )


def _thumbnail_diff(left: bytes, right: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / (len(left) * 255.0)


def _pixel_values(image: Image.Image) -> list[int]:
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened()) if flattened else list(image.getdata())


def _parse_sitemap(content: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(content)
    ns = {
        "s": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "i": "http://www.google.com/schemas/sitemap-image/1.1",
    }
    entries = []
    for url in root.findall("s:url", ns):
        page_url = url.findtext("s:loc", default="", namespaces=ns)
        source_url = url.findtext("i:image/i:loc", default="", namespaces=ns)
        match = TOKEN_RE.search(urlparse(page_url).path)
        if match and source_url:
            entries.append({"slug": match.group(1), "page_url": page_url, "source_url": source_url})
    return entries


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*,application/xml,text/xml,*/*"})
    with urlopen(request, timeout=45) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
    return content


def _write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "query_id", "query_path", "target_slug", "source_url", "page_url", "source_domain",
        "provenance", "identity_method", "reference_duplicate_check", "manual_review_status",
    ]
    _write_csv(output_dir / "manifest.csv", fields, rows)


def _write_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "query_id", "target_slug", "source_url", "page_url", "source_domain",
        "retrieval_timestamp", "catalog_slug", "provenance", "identity_method",
        "reference_duplicate_check", "rejection_reason", "manual_review_status",
    ]
    _write_csv(path, fields, rows)


def _write_review(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "query_id", "target_slug", "candidate_path", "source_url", "page_url", "source_domain",
        "retrieval_timestamp", "catalog_slug", "provenance", "identity_method",
        "reference_duplicate_check", "review_reason", "manual_review_status",
    ]
    _write_csv(path, fields, rows)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _review_html(rows: list[dict[str, object]]) -> str:
    if not rows:
        body = "<p>No candidates require manual review.</p>"
    else:
        body = "<ul>" + "".join(
            f"<li><b>{row['target_slug']}</b>: {row.get('review_reason', '')} — "
            f"<a href='{row['page_url']}'>product page</a></li>" for row in rows
        ) + "</ul>"
    return "<!doctype html><meta charset='utf-8'><title>web_extra review</title>" \
        "<h1>web_extra_dev manual review</h1>" + body


def _readme(metadata: dict[str, object]) -> str:
    return f"""# web_extra_dev

Этот benchmark собирается с официального сайта `https://vino-svoe.ru/`.
В scored manifest попадают только image URLs из exact `/wines/{{slug}}` sitemap
entries, где slug совпал с canonical slug. Generated images не используются.

## Current collection

- Exact sitemap slug matches: {metadata['exact_catalog_slug_matches']}.
- Accepted: {metadata['accepted']}.
- Rejected as reference duplicates: {metadata['rejected_reference_duplicates']}.
- Manual review: {metadata['manual_review']}.

`manifest.csv` — единственный scored set. `rejected.csv` и
`review_candidates.csv` не могут попасть в него автоматически.
Каталог `images/` может содержать download-cache от предыдущих
консервативных recheck; файлы, не перечисленные в `manifest.csv`, не являются
частью scored dataset.

## Leakage protection

Each candidate is checked against every canonical reference using exact SHA-256
and conservative 32x32 grayscale perceptual hashes on the full image and its
center crop, plus aspect-ratio and 64x64 thumbnail difference thresholds. A
candidate is rejected if it is an exact or conservative perceptual duplicate,
including resized/recompressed reference assets and derivatives with a
decorative background. Rejected and review rows remain provenance-only.

Collection is rate-limited by a small worker pool and uses public sitemap/image
URLs only. Robots and anti-bot restrictions must be respected on future runs.
"""


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


def _suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".bin"


if __name__ == "__main__":
    main()
