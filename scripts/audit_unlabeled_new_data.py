#!/usr/bin/env python3
"""Create an open-set review for unlabeled images in data/new_data.

This read-only audit does not edit input images or assign ground-truth labels.
It compares each image against the full catalog and exports frozen SO400M and
R16 top-5 candidates for review. The review UI lets a human mark a catalog
match, no catalog match, or unresolved. Query inference is intentionally
single-image to preserve the project's canonical encoder behavior.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_so400m_hard_negative_lora as runner  # noqa: E402
from recognition.so400m_lora import (  # noqa: E402
    lora_sha256,
    load_lora_state_dict,
    pooled_image_features,
    set_lora_enabled,
)

INPUT_DIR = ROOT / "data/new_data"
RUN_ROOT = ROOT / "artifacts/experiments/new_data_slug_audit_20260924"
R16_RUN = ROOT / "artifacts/experiments/encoder_followup_r16_capturev2_last2_20260924T083837Z"
R16_CHECKPOINT = R16_RUN / "r16/checkpoints/best.pt"
R16_SELECTED = R16_RUN / "selected_model.json"
R16_REFERENCE_CACHE = R16_RUN / "external_eval/r16/adapted_reference_cache"
NAME_RE = re.compile(r"(?P<prefix>\d+(?:\.\d+)?)_(?P<date>\d{2}-\d{2}-\d{4})_(?P<time>\d{2}-\d{2}-\d{2})\.webp$", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    temp.replace(path)


def rank_candidates(scores: np.ndarray, slugs: list[str], catalog_by_slug: dict[str, dict], limit: int = 5) -> list[dict]:
    order = np.argsort(-scores, kind="stable")[:limit]
    return [
        {
            "rank": rank,
            "slug": slugs[int(index)],
            "title": catalog_by_slug[slugs[int(index)]].get("title", ""),
            "winery": catalog_by_slug[slugs[int(index)]].get("winery", ""),
            "category": catalog_by_slug[slugs[int(index)]].get("category", ""),
            "similarity": float(scores[int(index)]),
            "reference_image_path": catalog_by_slug[slugs[int(index)]].get("reference_image_path", ""),
        }
        for rank, index in enumerate(order, 1)
    ]


def main() -> None:
    files = sorted(path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".webp")
    if not files:
        raise RuntimeError(f"No top-level WebP images found under {INPUT_DIR}")
    if any(not NAME_RE.fullmatch(path.name) for path in files):
        raise RuntimeError("At least one WebP filename does not match the observed score_date_time pattern")

    catalog, slugs, base_refs, base_meta, image_hashes = runner.load_catalog(ROOT)
    catalog_by_slug = {str(row["slug"]): row for row in catalog}
    r16_state_payload = torch.load(R16_CHECKPOINT, map_location="cpu", weights_only=False)
    r16_state = r16_state_payload.get("lora_state")
    if not isinstance(r16_state, dict):
        raise RuntimeError("Frozen R16 checkpoint is missing LoRA state")
    selected = read_json(R16_SELECTED)
    state_hash = lora_sha256(r16_state)
    if selected.get("checkpoint_sha256") != state_hash:
        raise RuntimeError("R16 checkpoint hash differs from the internally selected model")

    cache_metadata = read_json(R16_REFERENCE_CACHE / "metadata.json")
    cached_slugs = read_json(R16_REFERENCE_CACHE / "slugs.json")
    expected_reference_fingerprint = runner.adapted_reference_fingerprint(
        str(base_meta["fingerprint"]), state_hash, slugs, image_hashes, query_batch_size=1,
    )
    if (
        cache_metadata.get("fingerprint") != expected_reference_fingerprint
        or cache_metadata.get("lora_checkpoint_sha256") != state_hash
        or int(cache_metadata.get("query_batch_size", -1)) != 1
        or cached_slugs != slugs
    ):
        raise RuntimeError("R16 adapted reference cache fingerprint/alignment check failed")
    cached_r16_refs = np.load(R16_REFERENCE_CACHE / "embeddings.npy")
    reference_embedding_storage_dtype = str(cached_r16_refs.dtype)
    r16_refs = cached_r16_refs.astype(np.float32)
    if r16_refs.shape != (len(slugs), runner.HIDDEN_SIZE):
        raise RuntimeError("R16 adapted reference embeddings do not align with the catalog")

    runner.CONFIG["rank"] = 16
    runner.CONFIG["alpha"] = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, _modules, _ = runner.load_model(device)
    load_lora_state_dict(model, r16_state)
    if lora_sha256(model) != state_hash:
        raise RuntimeError("Loaded R16 model does not match the selected checkpoint")
    model.requires_grad_(False)
    model.vision_model.eval()
    base_refs = base_refs.astype(np.float32)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    source_manifest = [{"image": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in files]
    progress_path = RUN_ROOT / "progress.json"
    progress_identity = {
        "source_manifest": source_manifest,
        "r16_checkpoint_sha256": state_hash,
        "reference_cache_fingerprint": expected_reference_fingerprint,
        "query_batch_size": 1,
    }
    rows: list[dict] = []
    if progress_path.is_file():
        saved_progress = read_json(progress_path)
        if all(saved_progress.get(key) == value for key, value in progress_identity.items()):
            rows = list(saved_progress.get("rows", []))
            print(f"[new data slug audit] resuming with {len(rows)}/{len(files)} completed images", flush=True)
        else:
            print("[new data slug audit] prior progress fingerprint changed; starting a clean candidate pass", flush=True)
    completed_by_image = {row["image"]: row for row in rows}
    prefix_values: list[float] = []
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    dimensions = Counter()
    filename_datetimes: list[datetime] = []
    for position, path in enumerate(files, 1):
        match = NAME_RE.fullmatch(path.name)
        assert match is not None
        dt = datetime.strptime(f"{match['date']} {match['time']}", "%d-%m-%Y %H-%M-%S")
        filename_datetimes.append(dt)
        prefix_value = float(match["prefix"])
        prefix_values.append(prefix_value)
        raw_hash = sha256(path)
        exact_hashes[raw_hash].append(path.name)
        with Image.open(path) as opened:
            dimensions[str(opened.size)] += 1
            image = opened.convert("RGB")
        relative_path = path.relative_to(ROOT).as_posix()
        if relative_path in completed_by_image:
            continue
        pixels = runner._processor_pixels(processor, [image], model, device)
        set_lora_enabled(model, False)
        with torch.inference_mode():
            frozen_query = torch.nn.functional.normalize(
                pooled_image_features(model, pixels).float(), p=2, dim=-1, eps=1e-12,
            ).cpu().numpy()[0]
        set_lora_enabled(model, True)
        with torch.inference_mode():
            adapted_query = runner._features(model, pixels, device).cpu().numpy()[0].astype(np.float32)

        frozen_scores = frozen_query @ base_refs.T
        r16_scores = adapted_query @ r16_refs.T
        frozen_top = rank_candidates(frozen_scores, slugs, catalog_by_slug)
        r16_top = rank_candidates(r16_scores, slugs, catalog_by_slug)
        agreement = frozen_top[0]["slug"] == r16_top[0]["slug"]
        result_row = {
            "image": relative_path,
            "sha256": raw_hash,
            "width": image.width,
            "height": image.height,
            "filename_numeric_prefix_uninterpreted": match["prefix"],
            "filename_datetime_unverified": dt.isoformat(sep=" "),
            "frozen_top1_slug_candidate": frozen_top[0]["slug"],
            "frozen_top1_title_candidate": frozen_top[0]["title"],
            "frozen_top1_similarity": f"{frozen_top[0]['similarity']:.8f}",
            "frozen_top1_top2_margin": f"{frozen_top[0]['similarity'] - frozen_top[1]['similarity']:.8f}",
            "frozen_top5_json": json.dumps(frozen_top, ensure_ascii=False),
            "r16_top1_slug_candidate": r16_top[0]["slug"],
            "r16_top1_title_candidate": r16_top[0]["title"],
            "r16_top1_similarity": f"{r16_top[0]['similarity']:.8f}",
            "r16_top1_top2_margin": f"{r16_top[0]['similarity'] - r16_top[1]['similarity']:.8f}",
            "r16_top5_json": json.dumps(r16_top, ensure_ascii=False),
            "frozen_r16_top1_agree": agreement,
            "consensus_slug_candidate": frozen_top[0]["slug"] if agreement else "",
            "review_status": "unreviewed",
            "verified_slug": "",
            "review_notes": "",
        }
        rows.append(result_row)
        completed_by_image[relative_path] = result_row
        write_json_atomic(progress_path, {**progress_identity, "completed_images": len(rows), "rows": rows})
        if position % 20 == 0 or position == len(files):
            print(f"[new data slug audit] encoded {position}/{len(files)} images, query_batch_size=1", flush=True)

    rows.sort(key=lambda row: row["image"])
    for row in rows:
        row.setdefault("review_status", "unreviewed")
        row.setdefault("verified_slug", "")
        row.setdefault("review_notes", "")

    write_csv(RUN_ROOT / "slug_candidates.csv", rows)
    metadata = {
        "stage": "open_set_manual_review",
        "source_dir": "data/new_data",
        "image_count": len(files),
        "catalog_slug_count": len(slugs),
        "candidate_methods": ["frozen_so400m_image_only_top5", "selected_r16_image_only_top5"],
        "query_batch_size": 1,
        "frozen_model": runner.BASE_MODEL_ID,
        "frozen_revision": runner.BASE_REVISION,
        "r16_checkpoint_sha256": state_hash,
        "reference_cache_fingerprint": expected_reference_fingerprint,
        "reference_cache_dtype": reference_embedding_storage_dtype,
        "dimensions": dimensions,
        "appledouble_sidecar_count": len(list((INPUT_DIR / "__MACOSX").glob("._*"))) if (INPUT_DIR / "__MACOSX").is_dir() else 0,
        "filename_datetime_range_unverified": {
            "earliest": min(filename_datetimes).isoformat(sep=" "),
            "latest": max(filename_datetimes).isoformat(sep=" "),
        },
        "exact_duplicate_groups": [names for names in exact_hashes.values() if len(names) > 1],
        "filename_numeric_prefix": {
            "min": min(prefix_values),
            "median": float(np.median(prefix_values)),
            "max": max(prefix_values),
            "interpretation": "unverified; preserved without treating it as confidence or label",
        },
        "top1_agreement_frozen_vs_r16": sum(bool(row["frozen_r16_top1_agree"]) for row in rows),
        "ground_truth_labels_present": False,
        "automatic_slug_assignment": False,
        "open_set_threshold_calibrated": False,
        "review_outcomes": ["catalog_match", "no_catalog_match", "uncertain"],
        "raw_images_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(RUN_ROOT / "audit_metadata.json", metadata)
    write_html(RUN_ROOT / "slug_candidate_review.html", rows, catalog)
    write_report(RUN_ROOT / "README.md", metadata)
    print(json.dumps({"output": str(RUN_ROOT), "images": len(rows), "top1_agreement": metadata["top1_agreement_frozen_vs_r16"]}, ensure_ascii=False), flush=True)


def write_html(path: Path, rows: list[dict], catalog: list[dict]) -> None:
    cards = []
    for row in rows:
        query_src = "../../../" + row["image"]
        candidates = {"frozen": json.loads(row["frozen_top5_json"]), "R16": json.loads(row["r16_top5_json"])}
        shared = sorted({item["slug"] for item in candidates["frozen"]} & {item["slug"] for item in candidates["R16"]})
        card = [
            f'<article class="sample" data-image="{html.escape(row["image"], quote=True)}">',
            f'<div class="query"><img loading="lazy" src="{html.escape(query_src)}"><div><b>{html.escape(Path(row["image"]).name)}</b><br><code>prefix={html.escape(row["filename_numeric_prefix_uninterpreted"])}</code><br>frozen/R16 Top-1 agree: {html.escape(str(row["frozen_r16_top1_agree"]))}<br>shared Top-5 slugs: {html.escape(", ".join(shared) or "none")}</div></div>',
            '<div class="decision">',
            '<label>Решение по каталогу <select class="review-status">'
            '<option value="unreviewed">Не проверено</option>'
            '<option value="catalog_match">Slug найден в каталоге</option>'
            '<option value="no_catalog_match">Slug в каталоге нет</option>'
            '<option value="uncertain">Пока неясно</option></select></label>',
            f'<label>Подтверждённый slug <input class="verified-slug" list="catalog-slugs" value="{html.escape(row.get("verified_slug", ""), quote=True)}" placeholder="Введите или выберите slug" autocomplete="off"></label>',
            f'<label>Заметка <input class="review-notes" value="{html.escape(row.get("review_notes", ""), quote=True)}" placeholder="Необязательно"></label>',
            '<small>Если slug существует, но отсутствует в Top-5 моделей, его можно ввести из общего списка каталога.</small>',
            '</div>',
        ]
        for model_name, options in candidates.items():
            card.append(f'<section><h3>{model_name}</h3><ol>')
            for item in options:
                reference = "../../../" + str(item["reference_image_path"])
                card.append(
                    f'<li><img loading="lazy" src="{html.escape(reference)}"><div><code>{html.escape(item["slug"])}</code><br>{html.escape(item["title"])} — {html.escape(item["winery"])}<br>similarity {item["similarity"]:.5f}<br><button type="button" class="choose-slug" data-slug="{html.escape(item["slug"], quote=True)}">Подставить slug</button></div></li>'
                )
            card.append('</ol></section>')
        card.append('</article>')
        cards.append("\n".join(card))
    catalog_options = "\n".join(
        f'<option value="{html.escape(str(item.get("slug", "")), quote=True)}" label="{html.escape(str(item.get("title", "")) + " — " + str(item.get("winery", "")), quote=True)}"></option>'
        for item in catalog
    )
    export_rows = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    document = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Open-set review — new data</title>
<style>body{font:14px system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#222}.toolbar{position:sticky;top:0;background:#fff;border-bottom:1px solid #bbb;padding:1rem 0;z-index:2}.sample{border:1px solid #bbb;border-radius:8px;padding:1rem;margin:1rem 0}.query{display:flex;align-items:flex-start;gap:1rem}.query img{max-height:360px;max-width:260px;object-fit:contain}.decision{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.75rem;margin:1rem 0;padding:1rem;background:#f4f6f8;border-radius:6px}.decision label{display:flex;flex-direction:column;gap:.3rem;font-weight:600}.decision input,.decision select{padding:.5rem;font:inherit}.decision small{grid-column:1/-1;color:#555}section ol{display:flex;flex-wrap:wrap;gap:1rem;padding:0;list-style:none}li{width:280px;padding:.5rem;border:1px solid #ddd;display:flex;gap:.5rem;align-items:flex-start;overflow-wrap:anywhere}li img{width:90px;height:120px;object-fit:contain}code{overflow-wrap:anywhere}button{padding:.4rem .6rem;cursor:pointer}</style>
<header class="toolbar"><h1>Open-set review: сопоставление со всем каталогом</h1><p>У каждого фото выберите одно: slug найден, slug отсутствует в каталоге или пока неясно. Top-5 — подсказки поиска, не готовые метки; низкое сходство не доказывает отсутствие товара.</p><button id="export-csv" type="button">Скачать CSV с решениями</button> <span id="review-count"></span> <span id="save-state">Решения сохраняются в этом браузере.</span></header>
<datalist id="catalog-slugs">__CATALOG_OPTIONS__</datalist>
""".replace("__CATALOG_OPTIONS__", catalog_options) + "\n".join(cards) + """
<script>
const sourceRows = __EXPORT_ROWS__;
const storageKey = "new-data-open-set-review-v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) {}
function getFields(card) { return { review_status: card.querySelector(".review-status").value, verified_slug: card.querySelector(".verified-slug").value.trim(), review_notes: card.querySelector(".review-notes").value }; }
function updateCount() { const cards = [...document.querySelectorAll(".sample")]; const done = cards.filter(card => card.querySelector(".review-status").value !== "unreviewed").length; document.getElementById("review-count").textContent = `Решено: ${done}/${cards.length}`; }
function persist(card) { saved[card.dataset.image] = getFields(card); try { localStorage.setItem(storageKey, JSON.stringify(saved)); document.getElementById("save-state").textContent = "Сохранено в этом браузере."; } catch (_) { document.getElementById("save-state").textContent = "Автосохранение браузера недоступно; скачивайте CSV после проверки."; } updateCount(); }
for (const card of document.querySelectorAll(".sample")) {
  const status = card.querySelector(".review-status"), slug = card.querySelector(".verified-slug"), notes = card.querySelector(".review-notes");
  const initial = sourceRows.find(row => row.image === card.dataset.image) || {};
  const value = saved[card.dataset.image] || {review_status: initial.review_status || "unreviewed", verified_slug: initial.verified_slug || "", review_notes: initial.review_notes || ""};
  status.value = value.review_status; slug.value = value.verified_slug; notes.value = value.review_notes;
  for (const input of [status, slug, notes]) input.addEventListener("input", () => persist(card));
  status.addEventListener("change", () => { if (status.value !== "catalog_match") slug.value = ""; persist(card); });
  for (const button of card.querySelectorAll(".choose-slug")) button.addEventListener("click", () => { slug.value = button.dataset.slug; status.value = "catalog_match"; persist(card); slug.focus(); });
}
updateCount();
document.getElementById("export-csv").addEventListener("click", () => {
  const answers = Object.fromEntries([...document.querySelectorAll(".sample")].map(card => [card.dataset.image, getFields(card)]));
  const rows = sourceRows.map(row => ({...row, ...(answers[row.image] || {})}));
  const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const quote = value => '"' + String(value ?? "").replaceAll('"', '""') + '"';
  const csv = "\\ufeff" + [headers.map(quote).join(","), ...rows.map(row => headers.map(key => quote(typeof row[key] === "object" ? JSON.stringify(row[key]) : row[key])).join(","))].join("\\r\\n");
  const url = URL.createObjectURL(new Blob([csv], {type:"text/csv;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = "new_data_open_set_review.csv"; link.click(); URL.revokeObjectURL(url);
});
</script></html>
""".replace("__EXPORT_ROWS__", export_rows)
    path.write_text(document, encoding="utf-8")


def write_report(path: Path, metadata: dict) -> None:
    duplicate_groups = metadata["exact_duplicate_groups"]
    content = [
        "# Open-set review of unlabeled `data/new_data`",
        "",
        f"- Images: {metadata['image_count']} WebP; catalog has {metadata['catalog_slug_count']} usable slugs.",
        f"- Frozen SO400M and selected R16 Top-1 agree on {metadata['top1_agreement_frozen_vs_r16']}/{metadata['image_count']} images.",
        f"- Exact duplicate image groups: {len(duplicate_groups)}.",
        f"- Dimension distribution: `{json.dumps(metadata['dimensions'], ensure_ascii=False)}`.",
        f"- Filename numeric-prefix range: {metadata['filename_numeric_prefix']['min']}–{metadata['filename_numeric_prefix']['max']}; meaning unverified.",
        "- The 100 photos have no supplied slugs. This is an open-set task: a human review can mark a catalog slug, no catalog match, or unresolved; this pass computes no accuracy.",
        "- Candidate rankings are image-only embeddings (frozen SO400M and R16); OCR/SIFT have not been run on this dataset.",
        "- Each image was searched against the full catalog. A Top-5 list is only a review aid; there is no calibrated automatic no-match threshold, so scores do not assign an outcome.",
        "- Raw WebP files were read only. The `__MACOSX/._*` AppleDouble sidecars are not treated as labels.",
        "- Query inference uses batch size 1. Frozen and R16 Top-5 lists are exported separately; consensus is only a review hint. The HTML review page supports decisions and CSV export.",
        "",
        "Files: `slug_candidates.csv`, `slug_candidate_review.html`, `audit_metadata.json`.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
