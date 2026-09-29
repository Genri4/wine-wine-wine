#!/usr/bin/env python3
"""Serve the desktop web app and its frozen local recognition endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from functools import lru_cache
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recognition.smart_retry import MAX_IMAGE_BYTES, SmartRetryRuntime  # noqa: E402
from recognition.official_wine_details import get_official_wine_details  # noqa: E402

MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _extract_multipart_image(content_type: str, body: bytes) -> bytes:
    """Extract the single evaluator image from multipart/form-data."""
    try:
        mime_message = BytesParser(policy=email_policy).parsebytes(
            b"MIME-Version: 1.0\r\nContent-Type: "
            + content_type.encode("ascii")
            + b"\r\n\r\n"
            + body
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("invalid_multipart") from exc

    if not mime_message.is_multipart():
        raise ValueError("invalid_multipart")

    image_parts = [
        part
        for part in mime_message.iter_parts()
        if part.get_content_disposition() == "form-data"
        and part.get_param("name", header="content-disposition") == "image"
    ]
    if len(image_parts) != 1:
        raise ValueError("expected_one_image_field")

    image_bytes = image_parts[0].get_payload(decode=True)
    if not image_bytes:
        raise ValueError("empty_image")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    return image_bytes


@lru_cache(maxsize=1)
def _catalog_slugs() -> frozenset[str]:
    manifest = ROOT / "data/processed/catalog_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        return frozenset(row["slug"] for row in csv.DictReader(stream) if row.get("slug"))


class InferenceStatus:
    """Expose one-request-at-a-time inference state to health checks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._started_at: float | None = None
        self._last_duration: float | None = None
        self._last_result: str | None = None

    def begin(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._started_at = time.monotonic()
            return True

    def finish(self, duration: float, result: str) -> None:
        with self._lock:
            self._busy = False
            self._started_at = None
            self._last_duration = duration
            self._last_result = result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "busy" if self._busy else "ready",
                "inference_busy": self._busy,
                "busy_for_seconds": round(time.monotonic() - self._started_at, 3)
                if self._started_at is not None else None,
                "last_request_seconds": round(self._last_duration, 3) if self._last_duration is not None else None,
                "last_result": self._last_result,
            }


class SmartRetryHandler(SimpleHTTPRequestHandler):
    runtime: SmartRetryRuntime
    inference_status: InferenceStatus

    def __init__(self, *args: Any, directory: str, **kwargs: Any):
        super().__init__(*args, directory=directory, **kwargs)

    def _json(self, status: int, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            self._json(200, {"runtime": "validated-r8-ocr-sift", **self.inference_status.snapshot()})
            return
        if parsed.path == "/api/wine-details":
            try:
                query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=4)
            except ValueError:
                self._json(400, {"error": "invalid_query"})
                return
            slugs = query.get("slug", [])
            slug = slugs[0].strip() if len(slugs) == 1 else ""
            if not slug or slug not in _catalog_slugs():
                self._json(404, {"error": "unknown_wine_slug"})
                return
            self._json(200, {"slug": slug, **get_official_wine_details(slug)})
            return
        if not self._is_public_static_path():
            self.send_error(404)
            return
        super().do_GET()

    def _is_public_static_path(self) -> bool:
        path = unquote(urlsplit(self.path).path)
        if path == "/web" or path.startswith("/web/"):
            allowed_root = (ROOT / "web").resolve()
            allowed_file = None
        elif path == "/data/processed/catalog_manifest.csv":
            allowed_root = None
            allowed_file = (ROOT / "data/processed/catalog_manifest.csv").resolve()
        elif path.startswith("/data/processed/reference_images/"):
            allowed_root = (ROOT / "data/processed/reference_images").resolve()
            allowed_file = None
        else:
            return False

        translated = Path(self.translate_path(path)).resolve()
        if allowed_file is not None:
            return translated == allowed_file and translated.is_file()
        try:
            translated.relative_to(allowed_root)
        except ValueError:
            return False
        return translated.is_file() or (translated.is_dir() and (translated / "index.html").is_file())

    def do_POST(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path not in {"/api/recognize", "/api/predict", "/v1/eval/predict"}:
            self._json(404, {"error": "not_found"})
            return
        content_type_header = self.headers.get("Content-Type", "")
        content_type = content_type_header.split(";", 1)[0].strip().lower()
        is_evaluator_request = request_path == "/v1/eval/predict"
        if is_evaluator_request:
            if content_type != "multipart/form-data":
                self._json(415, {"error": "multipart_form_required"})
                return
        elif content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
            self._json(415, {"error": "unsupported_image_type"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if content_length <= 0:
            self._json(400, {"error": "empty_image"})
            return
        max_request_bytes = MAX_IMAGE_BYTES + (MAX_MULTIPART_OVERHEAD_BYTES if is_evaluator_request else 0)
        if content_length > max_request_bytes:
            self._json(413, {"error": "image_too_large"})
            return
        request_body = self.rfile.read(content_length)
        if len(request_body) != content_length:
            self._json(400, {"error": "incomplete_image_upload"})
            return
        if is_evaluator_request:
            try:
                image_bytes = _extract_multipart_image(content_type_header, request_body)
            except ValueError as exc:
                error = str(exc)
                status = 413 if error == "image_too_large" else 400
                self._json(status, {"error": error})
                return
        else:
            image_bytes = request_body
        if not self.inference_status.begin():
            self._json(503, {"error": "recognition_busy"}, {"Retry-After": "1"})
            return

        started = time.perf_counter()
        outcome = "failed"
        try:
            if request_path in {"/api/predict", "/v1/eval/predict"}:
                result = self.runtime.predict_slug(image_bytes)
            else:
                result = self.runtime.recognize(image_bytes)
        except ValueError as exc:
            if request_path not in {"/api/predict", "/v1/eval/predict"}:
                self.log_error("local recognition failed")
                self._json(500, {"error": "recognition_failed"})
            else:
                outcome = "invalid_image"
                self._json(400, {"error": "invalid_image", "reason": str(exc)})
        except Exception:
            self.log_error("local recognition failed")
            self._json(500, {"error": "recognition_failed"})
        else:
            outcome = str(result.get("status", "ok"))
            if result.get("reason"):
                outcome += f":{result['reason']}"
            self._json(200, result)
        finally:
            duration = time.perf_counter() - started
            self.inference_status.finish(duration, outcome)
            self.log_message("inference %s result=%s duration=%.3fs", request_path, outcome, duration)

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid logging uploaded file names or recognition payloads.
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local desktop Smart Retry web app")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ocr-device", default=None, choices=("cpu", "gpu:0"))
    args = parser.parse_args()

    print("Loading the frozen R8 + OCR + SIFT runtime…", flush=True)
    runtime = SmartRetryRuntime(ROOT, ocr_device=args.ocr_device)

    class BoundHandler(SmartRetryHandler):
        pass

    BoundHandler.runtime = runtime
    BoundHandler.inference_status = InferenceStatus()
    server = ThreadingHTTPServer((args.host, args.port), lambda *a, **kw: BoundHandler(*a, directory=str(ROOT), **kw))
    server.daemon_threads = True
    print(f"Smart Retry is ready at http://{args.host}:{args.port}/web/", flush=True)
    print(
        "API readiness: /api/health · UI: POST /api/recognize · evaluator contract: "
        "POST /v1/eval/predict (multipart field: image)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Smart Retry server…", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
