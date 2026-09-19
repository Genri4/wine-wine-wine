#!/usr/bin/env python3
"""Validate externally generated stress images and prepare manual review."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
from typing import Mapping

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.reporting import write_csv  # noqa: E402


MIN_DIMENSION = 128
MAX_DIMENSION = 12000
MAX_PIXELS = 50_000_000
REVIEW_STATUSES = {"accepted", "rejected", "pending"}

REVIEW_FIELDS = [
    "generation_id", "target_slug", "reference_image", "product_name", "winery", "category",
    "scenario_id", "output_filename", "generated_path", "validation_status", "validation_note",
    "image_format", "image_width", "image_height", "generated_sha256", "reference_sha256",
    "exact_reference_duplicate", "review_status", "reject_reason", "review_note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and validate generated stress images")
    parser.add_argument("--generation-manifest", default="data/benchmarks/generated_stress_dev/generation_manifest.csv")
    parser.add_argument("--selected-products", default="data/benchmarks/generated_stress_dev/selected_products.csv")
    parser.add_argument("--generated-dir", default="data/benchmarks/generated_stress_dev/generated_raw")
    parser.add_argument("--review-csv", default="data/benchmarks/generated_stress_dev/review.csv")
    parser.add_argument("--review-html", default="data/benchmarks/generated_stress_dev/review.html")
    args = parser.parse_args()
    result = import_generated_samples(
        _path(args.generation_manifest),
        _path(args.selected_products),
        _path(args.generated_dir),
        _path(args.review_csv),
        _path(args.review_html),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def import_generated_samples(
    generation_manifest_path: Path,
    selected_products_path: Path,
    generated_dir: Path,
    review_csv_path: Path,
    review_html_path: Path,
) -> dict[str, object]:
    manifest = _read_csv(generation_manifest_path)
    products = {row["slug"]: row for row in _read_csv(selected_products_path)}
    _validate_plan(manifest, products)
    generated_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_existing_review(review_csv_path)
    reference_hashes: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for plan in manifest:
        slug = plan["target_slug"]
        product = products[slug]
        generated_path = generated_dir / plan["output_filename"]
        validation = _validate_file(generated_path, _path(plan["reference_image"]), reference_hashes)
        old = previous.get(plan["generation_id"], {})
        old_status = old.get("review_status", "pending")
        if old_status not in REVIEW_STATUSES:
            old_status = "pending"
        auto_missing_from_previous_import = (
            old.get("validation_status") == "missing_file"
            and not old.get("reject_reason")
            and not old.get("review_note")
        )
        if validation["validation_status"] == "missing_file":
            review_status = "pending" if old_status != "accepted" or auto_missing_from_previous_import else "rejected"
        else:
            review_status = "rejected" if validation["validation_status"] != "valid" else old_status
        reject_reason = old.get("reject_reason", "") if review_status == "rejected" else ""
        if validation["validation_status"] == "reference_exact_duplicate":
            reject_reason = "too_easy_reference_copy"
        elif validation["validation_status"] == "corrupted_image":
            reject_reason = "corrupted_image"
        rows.append({
            "generation_id": plan["generation_id"],
            "target_slug": slug,
            "reference_image": plan["reference_image"],
            "product_name": product.get("product_name", ""),
            "winery": product.get("winery", ""),
            "category": product.get("category", ""),
            "scenario_id": plan["scenario_id"],
            "output_filename": plan["output_filename"],
            "generated_path": _display_path(generated_path),
            **validation,
            "review_status": review_status,
            "reject_reason": reject_reason,
            "review_note": old.get("review_note", ""),
        })
    write_csv(review_csv_path, REVIEW_FIELDS, rows)
    review_html_path.parent.mkdir(parents=True, exist_ok=True)
    review_html_path.write_text(_review_html(rows, review_html_path), encoding="utf-8")
    counts = Counter(row["review_status"] for row in rows)
    validation_counts = Counter(row["validation_status"] for row in rows)
    report = {
        "planned": len(rows),
        "review_status": dict(sorted(counts.items())),
        "validation_status": dict(sorted(validation_counts.items())),
        "untracked_files": sorted(_untracked_files(generated_dir, {row["output_filename"] for row in manifest})),
    }
    (review_csv_path.parent / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _validate_plan(manifest: list[dict[str, str]], products: Mapping[str, Mapping[str, str]]) -> None:
    if not manifest:
        raise ValueError("generation_manifest.csv is empty")
    required = {"generation_id", "target_slug", "reference_image", "scenario_id", "output_filename"}
    missing = required - set(manifest[0])
    if missing:
        raise ValueError(f"generation manifest missing columns: {sorted(missing)}")
    generation_ids = [row["generation_id"] for row in manifest]
    output_names = [row["output_filename"] for row in manifest]
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("generation_manifest.csv contains duplicate generation_id")
    normalized_names = [name.casefold() for name in output_names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("generation_manifest.csv contains duplicate output_filename")
    for row in manifest:
        if row["target_slug"] not in products:
            raise ValueError(f"generation target is not in selected products: {row['target_slug']}")
        reference = _path(row["reference_image"])
        if not reference.is_file():
            raise FileNotFoundError(f"reference image missing: {reference}")
        output = Path(row["output_filename"])
        if output.is_absolute() or output.name != row["output_filename"]:
            raise ValueError(f"unsafe output filename: {row['output_filename']}")


def _validate_file(path: Path, reference_path: Path, reference_hashes: dict[str, str]) -> dict[str, object]:
    base = {
        "validation_status": "missing_file",
        "validation_note": "",
        "image_format": "",
        "image_width": "",
        "image_height": "",
        "generated_sha256": "",
        "reference_sha256": "",
        "exact_reference_duplicate": False,
    }
    if not path.is_file():
        base["validation_note"] = "expected generated file is not present"
        return base
    try:
        with Image.open(path) as image:
            width, height, image_format = image.width, image.height, image.format or ""
            image.load()
        if min(width, height) < MIN_DIMENSION:
            base.update({"validation_status": "invalid_dimensions", "validation_note": f"minimum dimension is {MIN_DIMENSION}px"})
        elif max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
            base.update({"validation_status": "invalid_dimensions", "validation_note": "dimensions exceed importer safety limits"})
        else:
            base.update({"validation_status": "valid"})
        base.update({"image_format": image_format, "image_width": width, "image_height": height})
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        base.update({"validation_status": "corrupted_image", "validation_note": f"{type(exc).__name__}: {exc}"})
        return base
    generated_hash = _sha256(path)
    reference_key = str(reference_path.resolve())
    if reference_key not in reference_hashes:
        reference_hashes[reference_key] = _sha256(reference_path)
    reference_hash = reference_hashes[reference_key]
    base["generated_sha256"] = generated_hash
    base["reference_sha256"] = reference_hash
    if generated_hash == reference_hash:
        base.update({
            "validation_status": "reference_exact_duplicate",
            "validation_note": "generated bytes are an exact copy of the canonical reference",
            "exact_reference_duplicate": True,
        })
    return base


def _review_html(rows: list[dict[str, object]], output_path: Path) -> str:
    cards = []
    for row in rows:
        reference_src = _image_src(str(row["reference_image"]), output_path)
        generated_src = _image_src(str(row["generated_path"]), output_path)
        status_class = str(row["review_status"])
        accept_disabled = " disabled" if row["validation_status"] != "valid" else ""
        reason_options = _reason_options(str(row["reject_reason"]))
        cards.append(
            f"<article class='card {escape(status_class)}' data-review-id='{escape(str(row['generation_id']), quote=True)}'>"
            f"<h2>{escape(str(row['generation_id']))}</h2>"
            f"<p><b>slug:</b> <code>{escape(str(row['target_slug']))}</code> · "
            f"<b>product:</b> {escape(str(row['product_name']))} · "
            f"<b>winery:</b> {escape(str(row['winery']))} · "
            f"<b>category:</b> {escape(str(row['category']))}</p>"
            f"<p><b>scenario:</b> {escape(str(row['scenario_id']))} · "
            f"<b>review_status:</b> <strong class='status-label'>{escape(status_class)}</strong> · "
            f"<b>validation:</b> {escape(str(row['validation_status']))} "
            f"{escape(str(row['validation_note']))}</p>"
            "<div class='images'>"
            f"{_figure(reference_src, 'Catalog reference', str(row['reference_image']))}"
            f"{_figure(generated_src, 'Generated query', str(row['generated_path']))}"
            "</div>"
            "<div class='review-controls'>"
            f"<button class='accept' data-action='accept'{accept_disabled}>ACCEPT</button>"
            "<button class='reject' data-action='reject'>REJECT</button>"
            "<button class='pending' data-action='pending'>PENDING</button>"
            f"<label>Причина отказа <select class='reason'>{reason_options}</select></label>"
            f"<label>Заметка <input class='note' type='text' value='{escape(str(row['review_note']), quote=True)}' placeholder='необязательно'></label>"
            "</div>"
            "</article>"
        )
    serialized_rows = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    serialized_fields = json.dumps(REVIEW_FIELDS, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<html lang='ru'><head><meta charset='utf-8'><title>generated_stress_dev review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f1ec;color:#25211e}.toolbar{position:sticky;top:0;z-index:2;background:#25211e;color:#fff;margin:-24px -24px 22px;padding:14px 24px;box-shadow:0 2px 8px #999}.toolbar button,.toolbar select{font:inherit;margin-right:8px;padding:7px 10px;border-radius:6px;border:1px solid #bbb}.toolbar button{cursor:pointer}.toolbar .save{background:#2e8b57;color:#fff;border-color:#2e8b57}.counts{display:inline-flex;gap:12px;margin-left:8px;font-size:14px}.card{background:#fff;margin:0 0 22px;padding:16px;border-radius:12px;box-shadow:0 1px 5px #c9c0b7}.card.rejected{border-left:6px solid #c0392b}.card.accepted{border-left:6px solid #2e8b57}.card.pending{border-left:6px solid #b8860b}.images{display:flex;gap:18px;flex-wrap:wrap}figure{margin:0;width:310px}figcaption{font-size:13px;min-height:38px;overflow-wrap:anywhere}img{display:block;width:310px;height:420px;object-fit:contain;background:#eee;border:1px solid #ddd}code{font-size:12px;overflow-wrap:anywhere}h1{margin-bottom:6px}.review-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:14px;padding-top:12px;border-top:1px solid #eee}.review-controls button{padding:8px 14px;border:0;border-radius:6px;color:#fff;font-weight:700;cursor:pointer}.review-controls button:disabled{opacity:.4;cursor:not-allowed}.review-controls .accept{background:#2e8b57}.review-controls .reject{background:#c0392b}.review-controls .pending{background:#b8860b}.review-controls label{font-size:13px;display:flex;align-items:center;gap:5px}.review-controls select,.review-controls input{padding:7px;border:1px solid #bbb;border-radius:5px;background:#fff}.review-controls .note{min-width:230px}.missing{width:310px;height:420px;display:flex;align-items:center;justify-content:center;background:#eee;color:#777;border:1px solid #ddd}
</style></head><body><div class='toolbar'><button class='save' id='save-review'>Сохранить review.csv</button><button id='download-review'>Скачать review.csv</button><label>Показать <select id='filter'><option value='all'>все</option><option value='pending'>pending</option><option value='accepted'>accepted</option><option value='rejected'>rejected</option></select></label><span class='counts'><span>pending: <b id='count-pending'>0</b></span><span>accepted: <b id='count-accepted'>0</b></span><span>rejected: <b id='count-rejected'>0</b></span></span><span id='save-note'></span></div><h1>generated_stress_dev manual review</h1>
<p>Слева — catalog reference, справа — generated query. Выбери решение в каждой карточке. ACCEPT доступен только для валидного изображения без exact reference copy. Кнопка «Сохранить review.csv» пытается записать файл напрямую; если браузер не разрешит, используй «Скачать review.csv» и замени им файл рядом с этим HTML.</p>
""" + "\n".join(cards) + f"""<script id='review-data' type='application/json'>{serialized_rows}</script><script id='review-fields' type='application/json'>{serialized_fields}</script><script>
const reviewRows = JSON.parse(document.getElementById('review-data').textContent);
const reviewFields = JSON.parse(document.getElementById('review-fields').textContent);
const cards = Array.from(document.querySelectorAll('.card'));
const byId = new Map(reviewRows.map(row => [row.generation_id, row]));

function applyRow(card, row) {{
  card.className = 'card ' + (row.review_status || 'pending');
  card.querySelector('.status-label').textContent = row.review_status || 'pending';
  card.querySelector('.reason').value = row.reject_reason || '';
  card.querySelector('.note').value = row.review_note || '';
}}
function updateCounts() {{
  for (const status of ['pending', 'accepted', 'rejected']) {{
    document.getElementById('count-' + status).textContent = reviewRows.filter(row => row.review_status === status).length;
  }}
}}
function setStatus(card, status) {{
  const row = byId.get(card.dataset.reviewId);
  row.review_status = status;
  if (status === 'accepted') row.reject_reason = '';
  if (status === 'rejected' && !row.reject_reason) row.reject_reason = 'other';
  applyRow(card, row);
  updateCounts();
}}
function csvCell(value) {{
  if (value === null || value === undefined) return '';
  if (typeof value === 'boolean') value = value ? 'True' : 'False';
  const text = String(value);
  return /[",\\n\\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}}
function reviewCsv() {{
  return [reviewFields.join(','), ...reviewRows.map(row => reviewFields.map(field => csvCell(row[field])).join(','))].join('\\n') + '\\n';
}}
function downloadReview() {{
  const blob = new Blob([reviewCsv()], {{type: 'text/csv;charset=utf-8'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'review.csv';
  link.click();
  URL.revokeObjectURL(link.href);
  document.getElementById('save-note').textContent = 'review.csv скачан';
}}
async function saveReview() {{
  if (!window.showSaveFilePicker) {{ downloadReview(); return; }}
  try {{
    const handle = await window.showSaveFilePicker({{suggestedName: 'review.csv', types: [{{description: 'CSV', accept: {{'text/csv': ['.csv']}}}}]}});
    const writable = await handle.createWritable();
    await writable.write(reviewCsv());
    await writable.close();
    document.getElementById('save-note').textContent = 'review.csv сохранён';
  }} catch (error) {{
    if (error.name !== 'AbortError') downloadReview();
  }}
}}
cards.forEach(card => {{
  const row = byId.get(card.dataset.reviewId);
  card.querySelector('[data-action="accept"]').addEventListener('click', () => setStatus(card, 'accepted'));
  card.querySelector('[data-action="reject"]').addEventListener('click', () => setStatus(card, 'rejected'));
  card.querySelector('[data-action="pending"]').addEventListener('click', () => setStatus(card, 'pending'));
  card.querySelector('.reason').addEventListener('change', event => {{ row.reject_reason = event.target.value; }});
  card.querySelector('.note').addEventListener('input', event => {{ row.review_note = event.target.value; }});
  applyRow(card, row);
}});
document.getElementById('filter').addEventListener('change', event => {{
  const value = event.target.value;
  cards.forEach(card => {{ card.hidden = value !== 'all' && !card.classList.contains(value); }});
}});
document.getElementById('download-review').addEventListener('click', downloadReview);
document.getElementById('save-review').addEventListener('click', saveReview);
updateCounts();
</script></body></html>\n"""


def _reason_options(selected: str) -> str:
    reasons = [
        ("", "— выбрать причину —"),
        ("label_changed", "label_changed"),
        ("year_changed", "year_changed"),
        ("text_corrupted", "text_corrupted"),
        ("logo_changed", "logo_changed"),
        ("wrong_product", "wrong_product"),
        ("unrealistic_generation", "unrealistic_generation"),
        ("too_easy_reference_copy", "too_easy_reference_copy"),
        ("too_degraded", "too_degraded"),
        ("other", "other"),
    ]
    return "".join(
        f"<option value='{escape(value, quote=True)}'{' selected' if value == selected else ''}>{escape(label)}</option>"
        for value, label in reasons
    )


def _figure(src: str | None, caption: str, path: str) -> str:
    image = f"<img src='{escape(src)}' alt='{escape(caption)}'>" if src else "<div class='missing'>image unavailable</div>"
    return f"<figure><figcaption><b>{escape(caption)}</b><br>{escape(path)}</figcaption>{image}</figure>"


def _read_existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row.get("generation_id", ""): row for row in csv.DictReader(stream) if row.get("generation_id")}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _untracked_files(directory: Path, expected: set[str]) -> list[str]:
    if not directory.is_dir():
        return []
    return [path.name for path in sorted(directory.iterdir()) if path.is_file() and path.name not in expected]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_src(value: str, output_path: Path) -> str | None:
    if not value:
        return None
    path = _path(value).resolve()
    if not path.is_file():
        return None
    return os.path.relpath(path, output_path.parent.resolve()).replace(os.sep, "/")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
