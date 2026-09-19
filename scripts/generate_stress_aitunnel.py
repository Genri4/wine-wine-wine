#!/usr/bin/env python3
"""Generate a small, restart-safe generated_stress_dev pilot via AITUNNEL.

The generation plan and prompt templates are immutable inputs here. Runtime
status is written only to data/benchmarks/generated_stress_dev/generation_runs.
This module intentionally has no image-generation fallback and never changes
raw catalog data.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.reporting import write_csv  # noqa: E402


DEFAULT_MANIFEST = "data/benchmarks/generated_stress_dev/generation_manifest.csv"
DEFAULT_OUTPUT_DIR = "data/benchmarks/generated_stress_dev/generated_raw"
DEFAULT_RUNS_DIR = "data/benchmarks/generated_stress_dev/generation_runs"
DEFAULT_BASE_URL = "https://api.aitunnel.ru"
DEFAULT_ENDPOINT_PATH = "/v1/images/generations"
DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_QUALITY = "low"
DEFAULT_SIZE = "1024x1024"
DEFAULT_PILOT_PRODUCTS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 180.0
DEFAULT_BACKOFF_SECONDS = 2.0
EXPECTED_SCENARIOS = ("slight_angle", "glare_bad_light", "distance_crop", "handheld")
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
MIN_OUTPUT_DIMENSION = 128
MAX_OUTPUT_DIMENSION = 12000
MAX_OUTPUT_PIXELS = 50_000_000

REQUEST_FIELDS = [
    "generation_id",
    "target_slug",
    "scenario_id",
    "model",
    "quality",
    "size",
    "started_at",
    "finished_at",
    "latency_ms",
    "status",
    "success",
    "retry_count",
    "reported_usage",
    "reported_cost_rub",
    "output_filename",
    "response_format",
    "http_status",
    "error",
]


class AITunnelError(RuntimeError):
    """A provider or transport error with safe retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.fatal = fatal


@dataclass(frozen=True)
class ImageEditResult:
    image_bytes: bytes
    response_format: str
    usage: Any = None
    cost_rub: Any = None


class AITunnelClient:
    """Minimal stdlib-only client for the documented AITUNNEL image edit API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        quality: str = DEFAULT_QUALITY,
        size: str = DEFAULT_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.quality = quality
        self.size = size
        self.timeout = timeout
        self.opener = opener

    def edit(self, reference_path: Path, prompt: str) -> ImageEditResult:
        if not reference_path.is_file():
            raise AITunnelError(f"reference image is missing: {reference_path}")
        if reference_path.stat().st_size > MAX_REFERENCE_BYTES:
            raise AITunnelError("reference image exceeds the documented 25MB input limit")
        image_bytes = reference_path.read_bytes()
        content_type = mimetypes.guess_type(reference_path.name)[0] or "application/octet-stream"
        image_data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "quality": self.quality,
            "size": self.size,
            "output_format": "png",
            "input_references": [{
                "type": "image_url",
                "image_url": {"url": image_data_url},
            }],
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.base_url}{DEFAULT_ENDPOINT_PATH}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response_body, status_code = self._post(request)
        payload = _parse_json_response(response_body, status_code)
        return _decode_image_response(payload, self)

    def _post(self, request: Request) -> tuple[bytes, int]:
        try:
            with self.opener(request, timeout=self.timeout) as response:
                return response.read(), int(getattr(response, "status", 200))
        except HTTPError as exc:
            body = exc.read(2_000_000)
            reason = str(exc.reason) if exc.reason else ""
            message = _provider_error_message(body) or f"HTTP {exc.code}{': ' + reason if reason else ''}"
            raise AITunnelError(
                message,
                status_code=exc.code,
                retryable=exc.code in {408, 429} or exc.code >= 500,
                fatal=_fatal_provider_error(exc.code, message),
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise AITunnelError(
                f"transport error: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a safe AITUNNEL pilot for generated_stress_dev; default is 5 products × 4 scenarios"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--generated-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--pilot-products", type=int, default=None, help="distinct products; default: 5")
    parser.add_argument("--all", action="store_true", help="explicitly select every product in the existing plan")
    parser.add_argument("--dry-run", action="store_true", help="show the exact scope; never calls the API")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output after a successful validated response")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--base-url", default=os.environ.get("AITUNNEL_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="retries after the first attempt")
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--run-id", default=None, help="optional unique runtime artifact directory name")
    args = parser.parse_args()

    if args.all and args.pilot_products is not None:
        parser.error("use either --all or --pilot-products, not both")
    if args.pilot_products is not None and args.pilot_products < 1:
        parser.error("--pilot-products must be positive")
    if args.timeout <= 0 or args.max_retries < 0 or args.backoff_seconds < 0:
        parser.error("timeout must be positive; retries and backoff must be non-negative")

    manifest_path = _path(args.manifest)
    rows = _read_manifest(manifest_path)
    selected_rows, selected_slugs = select_scope(
        rows,
        all_products=args.all,
        pilot_products=DEFAULT_PILOT_PRODUCTS if args.pilot_products is None and not args.all else args.pilot_products,
    )
    _validate_selected_rows(selected_rows)

    scope = {
        "mode": "all" if args.all else "pilot",
        "pilot_products": None if args.all else (args.pilot_products or DEFAULT_PILOT_PRODUCTS),
        "products": len(selected_slugs),
        "requests": len(selected_rows),
        "scenarios": list(EXPECTED_SCENARIOS),
    }
    if args.dry_run:
        print(json.dumps(_dry_run_payload(scope, selected_rows, args), ensure_ascii=False, indent=2))
        return

    api_key = require_api_key()

    run_id = args.run_id or _new_run_id(_path(args.runs_dir))
    run_dir = _path(args.runs_dir) / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}; choose another --run-id")
    client = AITunnelClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        quality=args.quality,
        size=args.size,
        timeout=args.timeout,
    )
    result = generate_scope(
        selected_rows,
        output_dir=_path(args.generated_dir),
        run_dir=run_dir,
        client=client,
        model=args.model,
        quality=args.quality,
        size=args.size,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        api_key=api_key,
        scope=scope,
        endpoint=f"{args.base_url.rstrip('/')}{DEFAULT_ENDPOINT_PATH}",
    )
    _update_benchmark_metadata(manifest_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def select_scope(
    rows: Sequence[Mapping[str, str]],
    *,
    all_products: bool = False,
    pilot_products: int | None = DEFAULT_PILOT_PRODUCTS,
) -> tuple[list[dict[str, str]], list[str]]:
    """Select rows in existing manifest order, never inventing a new scope."""

    if all_products:
        slugs = _distinct_slugs(rows)
    else:
        count = pilot_products if pilot_products is not None else DEFAULT_PILOT_PRODUCTS
        if count < 1:
            raise ValueError("pilot_products must be positive")
        slugs = _distinct_slugs(rows)[:count]
    selected = [dict(row) for row in rows if row.get("target_slug", "") in set(slugs)]
    if len(selected) != len(slugs) * len(EXPECTED_SCENARIOS):
        raise ValueError(
            f"selected products must have exactly four planned rows each: products={len(slugs)}, rows={len(selected)}"
        )
    return selected, slugs


def require_api_key(environment: Mapping[str, str] | None = None) -> str:
    """Read the only supported credential location without exposing its value."""

    source = os.environ if environment is None else environment
    api_key = source.get("AITUNNEL_API_KEY", "")
    if not api_key:
        raise SystemExit("AITUNNEL_API_KEY is not set; use --dry-run to inspect the scope without an API key")
    return api_key


def generate_scope(
    rows: Sequence[Mapping[str, str]],
    *,
    output_dir: Path,
    run_dir: Path,
    client: Any,
    model: str,
    quality: str,
    size: str,
    overwrite: bool,
    max_retries: int,
    backoff_seconds: float,
    api_key: str,
    scope: Mapping[str, Any],
    endpoint: str,
    sleep_fn: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Generate rows with atomic writes and a request-level runtime artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    request_rows: list[dict[str, Any]] = []
    counts = {"generated": 0, "skipped_existing": 0, "failed": 0, "not_attempted": 0}
    cost_values: list[float] = []
    usage_values: list[Any] = []
    stopped_error: str | None = None

    for position, row in enumerate(rows):
        output_filename = row["output_filename"]
        output_path = _safe_output_path(output_dir, output_filename)
        if output_path.exists() and not overwrite:
            if _valid_image_file(output_path):
                counts["skipped_existing"] += 1
                request_rows.append(_request_row(row, model, quality, size, status="skipped_existing", success=True))
                continue
            counts["failed"] += 1
            request_rows.append(_request_row(
                row,
                model,
                quality,
                size,
                status="failed",
                success=False,
                error="existing output is invalid; use --overwrite to replace it",
            ))
            continue

        started_at = _now()
        request_started = monotonic()
        last_error: AITunnelError | None = None
        result: ImageEditResult | None = None
        retries_used = 0
        for attempt in range(max_retries + 1):
            try:
                candidate_result = client.edit(_path(row["reference_image"]), row["prompt"])
                _atomic_save_validated(output_path, candidate_result.image_bytes)
                result = candidate_result
                break
            except AITunnelError as exc:
                last_error = exc
                retries_used = attempt
                if not exc.retryable or exc.fatal or attempt >= max_retries:
                    break
                delay = min(30.0, backoff_seconds * (2**attempt))
                sleep_fn(delay)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                last_error = AITunnelError(
                    f"output validation failed: {type(exc).__name__}: {exc}",
                    retryable=True,
                )
                retries_used = attempt
                if attempt >= max_retries:
                    break
                delay = min(30.0, backoff_seconds * (2**attempt))
                sleep_fn(delay)

        latency_ms = round((monotonic() - request_started) * 1000.0, 3)
        if result is not None:
            counts["generated"] += 1
            if result.usage is not None:
                usage_values.append(result.usage)
            numeric_cost = _numeric_cost(result.cost_rub)
            if numeric_cost is not None:
                cost_values.append(numeric_cost)
            request_rows.append(_request_row(
                row,
                model,
                quality,
                size,
                status="generated",
                success=True,
                started_at=started_at,
                latency_ms=latency_ms,
                retry_count=retries_used,
                usage=result.usage,
                cost_rub=result.cost_rub,
                response_format=result.response_format,
            ))
        else:
            counts["failed"] += 1
            error = _safe_error(str(last_error or "generation failed"), api_key)
            request_rows.append(_request_row(
                row,
                model,
                quality,
                size,
                status="failed",
                success=False,
                started_at=started_at,
                latency_ms=latency_ms,
                retry_count=retries_used,
                error=error,
                http_status=getattr(last_error, "status_code", None),
            ))
            if last_error and last_error.fatal:
                stopped_error = error
                counts["not_attempted"] = len(rows) - position - 1
                break

    requests_path = run_dir / "requests.csv"
    write_csv(requests_path, REQUEST_FIELDS, request_rows)
    summary = {
        "run_id": run_dir.name,
        "endpoint": endpoint,
        "model": model,
        "quality": quality,
        "size": size,
        "scope": dict(scope),
        "requested": len(rows),
        **counts,
        "total_cost_rub": round(sum(cost_values), 6) if cost_values else None,
        "cost_status": "reported_by_provider" if cost_values else "unavailable_in_provider_response",
        "reported_usage": usage_values or None,
        "stopped_on_error": stopped_error,
        "requests_csv": _relative(requests_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"generation manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"generation manifest is empty: {path}")
    required = {"generation_id", "target_slug", "reference_image", "scenario_id", "prompt", "output_filename"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"generation manifest missing columns: {sorted(missing)}")
    generation_ids = [row["generation_id"] for row in rows]
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("generation manifest contains duplicate generation_id")
    output_names = [row["output_filename"] for row in rows]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("generation manifest contains duplicate output_filename")
    _validate_selected_rows(rows)
    return rows


def _validate_selected_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError("no generation rows selected")
    by_slug: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        slug = row.get("target_slug", "").strip()
        if not slug:
            raise ValueError("generation manifest contains empty target_slug")
        reference = _path(row.get("reference_image", ""))
        if not reference.is_file():
            raise FileNotFoundError(f"reference image missing for {slug}: {reference}")
        filename = row.get("output_filename", "")
        output = Path(filename)
        if not filename or output.is_absolute() or output.name != filename or ".." in output.parts:
            raise ValueError(f"unsafe output filename: {filename}")
        by_slug.setdefault(slug, []).append(row)
    for slug, product_rows in by_slug.items():
        scenarios = [row.get("scenario_id", "") for row in product_rows]
        if len(product_rows) != len(EXPECTED_SCENARIOS) or set(scenarios) != set(EXPECTED_SCENARIOS):
            raise ValueError(
                f"product {slug} must have exactly the four planned scenarios; got {sorted(scenarios)}"
            )


def _distinct_slugs(rows: Sequence[Mapping[str, str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        slug = row.get("target_slug", "")
        if slug and slug not in seen:
            result.append(slug)
            seen.add(slug)
    return result


def _dry_run_payload(scope: Mapping[str, Any], rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dry_run": True,
        "endpoint": f"{args.base_url.rstrip('/')}{DEFAULT_ENDPOINT_PATH}",
        "model": args.model,
        "quality": args.quality,
        "size": args.size,
        "scope": dict(scope),
        "products": _distinct_slugs(rows),
        "requests": [
            {
                "generation_id": row["generation_id"],
                "target_slug": row["target_slug"],
                "reference_image": row["reference_image"],
                "scenario_id": row["scenario_id"],
                "output_filename": row["output_filename"],
            }
            for row in rows
        ],
        "api_called": False,
    }


def _update_benchmark_metadata(manifest_path: Path, result: Mapping[str, Any]) -> None:
    """Record runtime generation state beside the plan, when metadata exists."""
    metadata_path = manifest_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    metadata.update({
        "image_generation_api_called": True,
        "generated_images_created": bool(result.get("generated", 0) or result.get("skipped_existing", 0)),
        "last_generation_run_id": result.get("run_id"),
        "last_generation_model": result.get("model"),
        "last_generation_quality": result.get("quality"),
        "last_generation_requested": result.get("requested"),
        "last_generation_generated": result.get("generated"),
        "last_generation_failed": result.get("failed"),
        "last_generation_cost_rub": result.get("total_cost_rub"),
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_json_response(body: bytes, status_code: int) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AITunnelError(
            f"invalid JSON response (HTTP {status_code})",
            status_code=status_code,
            retryable=True,
        ) from exc
    if not isinstance(payload, dict):
        raise AITunnelError("invalid AITUNNEL response: expected a JSON object", status_code=status_code, retryable=True)
    if "error" in payload:
        message = _provider_error_message(json.dumps(payload, ensure_ascii=False).encode("utf-8")) or "provider returned an error"
        raise AITunnelError(
            message,
            status_code=status_code,
            retryable=status_code in {408, 429} or status_code >= 500,
            fatal=_fatal_provider_error(status_code, message),
        )
    if status_code < 200 or status_code >= 300:
        raise AITunnelError(
            f"AITUNNEL returned HTTP {status_code}",
            status_code=status_code,
            retryable=status_code in {408, 429} or status_code >= 500,
            fatal=_fatal_provider_error(status_code, ""),
        )
    return payload


def _decode_image_response(payload: Mapping[str, Any], client: AITunnelClient) -> ImageEditResult:
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        raise AITunnelError("invalid response: data[0] image payload is missing", retryable=True)
    item = data[0]
    if isinstance(item.get("b64_json"), str) and item["b64_json"]:
        raw = item["b64_json"]
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise AITunnelError("invalid base64 image response", retryable=True) from exc
        return ImageEditResult(
            image_bytes=image_bytes,
            response_format="b64_json",
            usage=payload.get("usage"),
            cost_rub=_reported_cost(payload),
        )
    if isinstance(item.get("url"), str) and item["url"]:
        parsed = urlparse(item["url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AITunnelError("invalid image URL in AITUNNEL response", retryable=False)
        try:
            with urlopen(Request(item["url"], headers={"Accept": "image/*"}), timeout=client.timeout) as response:
                image_bytes = response.read(MAX_OUTPUT_PIXELS * 4)
        except (TimeoutError, URLError, OSError) as exc:
            raise AITunnelError(f"image URL download failed: {type(exc).__name__}: {exc}", retryable=True) from exc
        return ImageEditResult(
            image_bytes=image_bytes,
            response_format="url",
            usage=payload.get("usage"),
            cost_rub=_reported_cost(payload),
        )
    raise AITunnelError("unsupported AITUNNEL response: expected b64_json", retryable=True)


def _atomic_save_validated(path: Path, image_bytes: bytes) -> None:
    if not image_bytes:
        raise ValueError("provider returned an empty image")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(image_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if not _valid_image_file(temp_path):
            raise ValueError("provider returned a malformed or unsafe image")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            width, height = image.width, image.height
            image.load()
        return (
            width > 0
            and height > 0
            and min(width, height) >= MIN_OUTPUT_DIMENSION
            and max(width, height) <= MAX_OUTPUT_DIMENSION
            and width * height <= MAX_OUTPUT_PIXELS
        )
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def _request_row(
    row: Mapping[str, str],
    model: str,
    quality: str,
    size: str,
    *,
    status: str,
    success: bool,
    started_at: str = "",
    latency_ms: Any = "",
    retry_count: int = 0,
    usage: Any = None,
    cost_rub: Any = None,
    output_filename: str | None = None,
    response_format: str = "",
    http_status: Any = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "generation_id": row.get("generation_id", ""),
        "target_slug": row.get("target_slug", ""),
        "scenario_id": row.get("scenario_id", ""),
        "model": model,
        "quality": quality,
        "size": size,
        "started_at": started_at,
        "finished_at": _now() if started_at else "",
        "latency_ms": latency_ms,
        "status": status,
        "success": str(bool(success)).lower(),
        "retry_count": retry_count,
        "reported_usage": _json_cell(usage),
        "reported_cost_rub": "" if cost_rub is None else cost_rub,
        "output_filename": output_filename or row.get("output_filename", ""),
        "response_format": response_format,
        "http_status": "" if http_status is None else http_status,
        "error": error,
    }


def _provider_error_message(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body[:500].decode("utf-8", errors="replace").strip()
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for field in ("message", "detail", "description"):
                if error.get(field):
                    return str(error[field])
        elif error:
            return str(error)
        for field in ("message", "detail", "description"):
            if payload.get(field):
                return str(payload[field])
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return compact[:500]
    return ""


def _fatal_provider_error(status_code: int | None, message: str) -> bool:
    if status_code in {401, 402, 403}:
        return True
    normalized = message.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "unsupported model",
            "unknown model",
            "invalid model",
            "unsupported parameter",
            "invalid parameter",
            "model is not available",
        )
    )


def _reported_cost(payload: Mapping[str, Any]) -> Any:
    if "cost_rub" in payload:
        return payload["cost_rub"]
    usage = payload.get("usage")
    if isinstance(usage, Mapping) and "cost_rub" in usage:
        return usage["cost_rub"]
    return None


def _numeric_cost(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_error(message: str, api_key: str) -> str:
    sanitized = message.replace(api_key, "<redacted>") if api_key else message
    sanitized = re.sub(r"Bearer\s+\S+", "Bearer <redacted>", sanitized, flags=re.IGNORECASE)
    return sanitized[:2000]


def _safe_output_path(output_dir: Path, output_filename: str) -> Path:
    candidate = Path(output_filename)
    if candidate.is_absolute() or candidate.name != output_filename or ".." in candidate.parts:
        raise ValueError(f"unsafe output filename: {output_filename}")
    return output_dir / candidate


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_run_id(runs_dir: Path) -> str:
    prefix = datetime.now(timezone.utc).strftime("aitunnel-%Y%m%dT%H%M%SZ")
    candidate = runs_dir / prefix
    suffix = 2
    while candidate.exists():
        candidate = runs_dir / f"{prefix}-{suffix}"
        suffix += 1
    return candidate.name


if __name__ == "__main__":
    main()
