"""Small CSV/HTML writers for benchmark experiment artifacts."""

from __future__ import annotations

import csv
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_error_report(
    path: str | Path,
    errors: Iterable[Mapping[str, Any]],
    *,
    project_root: str | Path,
    catalog_by_slug: Mapping[str, Mapping[str, Any]],
    benchmark_rows_by_query: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write a self-contained-in-layout HTML report using local image files."""

    output_path = Path(path)
    root = Path(project_root).resolve()
    cards: list[str] = []
    for error in errors:
        query_id = str(error["query_id"])
        target = str(error["target_slug"])
        predicted = str(error.get("predicted_slug") or "")
        query_row = benchmark_rows_by_query.get(query_id, {})
        target_row = catalog_by_slug.get(target, {})
        predicted_row = catalog_by_slug.get(predicted, {})
        query_src = _relative_image_src(root, query_row.get("query_path"), output_path)
        target_src = _relative_image_src(root, target_row.get("reference_image_path"), output_path)
        predicted_src = _relative_image_src(root, predicted_row.get("reference_image_path"), output_path)
        target_label = _product_label(target, target_row)
        predicted_label = _product_label(predicted or "(none)", predicted_row)
        target_score = error.get("target_score")
        cards.append(
            "<article class='card'>"
            f"<h2>{escape(query_id)}</h2>"
            f"<p><b>transform:</b> {escape(str(error.get('transform_type', '')))} · "
            f"<b>target rank:</b> {escape(str(error.get('target_rank', 'not found')))}</p>"
            "<div class='images'>"
            f"<figure><figcaption>Query</figcaption>{_img(query_src, query_id)}</figure>"
            f"<figure><figcaption>Correct: {escape(target_label)}<br>score: {escape(str(target_score))}</figcaption>"
            f"{_img(target_src, target)}</figure>"
            f"<figure><figcaption>Predicted: {escape(predicted_label)}<br>score: {escape(str(error.get('top1_score')))}</figcaption>"
            f"{_img(predicted_src, predicted or 'no prediction')}</figure>"
            "</div></article>"
        )
    body = "\n".join(cards) or "<p>No Top-1 errors.</p>"
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Recognition error report</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}
.card{background:white;margin:0 0 20px;padding:16px;border-radius:8px;box-shadow:0 1px 4px #bbb}
.card h2{margin:0 0 6px;font-size:18px}.images{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0;width:250px}figcaption{font-size:13px;min-height:42px}img{display:block;max-width:250px;max-height:360px;object-fit:contain;background:#eee}
</style></head><body><h1>Recognition error report</h1>
<p>Only Top-1 errors are listed. Images are local project files; no new image data is embedded.</p>
""" + body + "</body></html>\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _relative_image_src(root: Path, value: Any, report_path: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    image_path = (root / value).resolve()
    if not image_path.is_file():
        return None
    return os.path.relpath(image_path, report_path.parent.resolve()).replace(os.sep, "/")


def _img(src: str | None, alt: str) -> str:
    return f"<img src='{escape(src)}' alt='{escape(alt)}'>" if src else "<div>image unavailable</div>"


def _product_label(slug: str, row: Mapping[str, Any]) -> str:
    if not row:
        return slug
    title = str(row.get("title") or slug)
    winery = str(row.get("winery") or "")
    category = str(row.get("category") or "")
    details = " · ".join(value for value in (winery, category) if value)
    return f"{title} ({details})" if details else title
