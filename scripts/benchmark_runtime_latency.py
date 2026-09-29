#!/usr/bin/env python3
"""Measure serial end-to-end latency for existing catalog references.

Only a complete batch of successful HTTP 200 responses receives percentile and
SLA statistics. Busy responses and client timeouts abort the batch so they can
never masquerade as fast inference results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_ready(url: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = _get_json(f"{url}/api/health", timeout=2.0)
            if last.get("status") == "ready":
                return last
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.025)
    raise TimeoutError(f"Recognition service did not become ready: {last}")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _read_catalog() -> list[dict[str, str]]:
    with (ROOT / "data/processed/catalog_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("slug") and row.get("reference_image_path")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--timeout", type=float, default=60.0, help="per recognition request, seconds")
    parser.add_argument("--sla-ms", type=float, default=3000.0)
    args = parser.parse_args()

    catalog = _read_catalog()
    if args.count < 1 or args.count > len(catalog):
        raise SystemExit(f"count must be between 1 and {len(catalog)}")
    selected = random.Random(args.seed).sample(catalog, args.count)
    url = args.url.rstrip("/")
    _wait_ready(url)
    rows: list[dict[str, Any]] = []
    failure: str | None = None

    for index, item in enumerate(selected, start=1):
        _wait_ready(url)
        image_path = ROOT / item["reference_image_path"]
        payload = image_path.read_bytes()
        request = urllib.request.Request(
            f"{url}/api/recognize", data=payload,
            headers={"Content-Type": "image/webp"}, method="POST",
        )
        started = time.perf_counter()
        status_code = 0
        output: dict[str, Any] = {}
        error = ""
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                status_code = response.status
                output = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            error = f"HTTP {exc.code}: {exc.reason}"
        except Exception as exc:  # preserve the diagnostic and stop a censored batch
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append({
            "index": index,
            "slug": item["slug"],
            "predicted_slug": output.get("slug", ""),
            "exact_slug_match": bool(output.get("slug") == item["slug"]),
            "reference_image": image_path.name,
            "http_status": status_code,
            "status": output.get("status", ""),
            "reason": output.get("reason", ""),
            "latency_ms": round(latency_ms, 3),
            "error": error,
        })
        print(f"{index:03d}/{args.count} HTTP {status_code} {latency_ms:.1f} ms "
              f"{output.get('status', '')}:{output.get('reason', '')} {image_path.name}", flush=True)
        if status_code != 200 or error:
            failure = error or f"unexpected HTTP status {status_code}"
            break

    health_after: dict[str, Any] = {}
    try:
        health_after = _wait_ready(url)
    except TimeoutError as exc:
        failure = failure or str(exc)

    complete = len(rows) == args.count and all(row["http_status"] == 200 for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows if row["http_status"] == 200]
    metrics: dict[str, Any] = {
        "status": "complete" if complete else "aborted_incomplete_batch",
        "measurement_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": "POST /api/recognize",
        "catalog_reference_queries_requested": args.count,
        "catalog_reference_queries_completed": len(rows),
        "selection": f"distinct existing catalog references; deterministic random seed {args.seed}",
        "concurrency": 1,
        "latency_boundary": "client request start through complete JSON response read; excludes readiness polling",
        "timeout_per_request_seconds": args.timeout,
        "sla_threshold_ms": args.sla_ms,
        "failure": failure,
        "health_after": health_after,
        "valid_percentile_sample": complete,
    }
    if complete:
        metrics.update({
            "http_200_count": len(rows),
            "found_count": sum(row["status"] == "found" for row in rows),
            "retry_count": sum(row["status"] == "retry" for row in rows),
            "found_exact_slug_count": sum(row["status"] == "found" and row["exact_slug_match"] for row in rows),
            "found_wrong_slug_count": sum(row["status"] == "found" and not row["exact_slug_match"] for row in rows),
            "retry_suggested_slug_matches": sum(row["status"] == "retry" and row["exact_slug_match"] for row in rows),
            "mean_ms": round(fmean(latencies), 3),
            "p50_ms": round(_percentile(latencies, 50), 3),
            "p90_ms": round(_percentile(latencies, 90), 3),
            "p95_ms": round(_percentile(latencies, 95), 3),
            "p99_ms": round(_percentile(latencies, 99), 3),
            "max_ms": round(max(latencies), 3),
            "under_sla_count": sum(value < args.sla_ms for value in latencies),
            "under_sla_percent": round(100 * sum(value < args.sla_ms for value in latencies) / len(latencies), 2),
            "p95_sla_passed": _percentile(latencies, 95) < args.sla_ms,
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = ROOT / "reports" / f"runtime_latency_{'sla' if complete else 'aborted'}_{args.count}_{stamp}"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["index", "slug", "http_status"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    title = "Runtime latency SLA benchmark" if complete else "Aborted runtime latency batch"
    if complete:
        body = (f"- Endpoint: `{metrics['endpoint']}`; sample: {len(rows)}; concurrency: 1.\n"
                f"- p50/p90/p95/p99: {metrics['p50_ms']}/{metrics['p90_ms']}/"
                f"{metrics['p95_ms']}/{metrics['p99_ms']} ms; max: {metrics['max_ms']} ms.\n"
                f"- `< {args.sla_ms:g} ms`: {metrics['under_sla_count']}/{len(rows)} "
                f"({metrics['under_sla_percent']}%). p95 gate: {'PASS' if metrics['p95_sla_passed'] else 'FAIL'}.\n"
                f"- Recognition on exact catalog references: {metrics['found_exact_slug_count']} exact `found`, "
                f"{metrics['retry_count']} `retry` ({metrics['retry_suggested_slug_matches']} suggested slugs match), "
                f"{metrics['found_wrong_slug_count']} wrong `found`.\n")
    else:
        body = (f"- Batch aborted after {len(rows)} of {args.count} requests.\n"
                f"- Percentiles and SLA pass rate: **not computed**.\n"
                f"- Reason: {failure or 'incomplete batch'}.\n")
    md_path.write_text(f"# {title} — {datetime.now().date().isoformat()}\n\n{body}\n"
                       f"Per-query data: [{csv_path.name}]({csv_path.name}); "
                       f"summary: [{json_path.name}]({json_path.name}).\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"REPORT={md_path}")
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
