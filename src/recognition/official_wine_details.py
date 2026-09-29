"""Fetch a small, on-demand set of public wine details from vino-svoe.ru."""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import subprocess
import threading
import time
from html.parser import HTMLParser
from typing import Any
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SITE_ROOT = "https://vino-svoe.ru"
MAX_PAGE_BYTES = 3 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 5
_CACHE_TTL_SECONDS = 6 * 60 * 60
_FAILURE_CACHE_TTL_SECONDS = 60
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class _OfficialRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"vino-svoe.ru", "www.vino-svoe.ru"}:
            raise HTTPError(newurl, code, "Redirect outside the official wine site", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _WinePageText(HTMLParser):
    """Collect visible text and a few standard metadata/JSON-LD values."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt", "h1", "h2",
        "h3", "h4", "li", "main", "p", "section", "table", "td", "th", "tr", "ul",
    }
    _IGNORED_TAGS = {"style", "noscript", "svg"}
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.metadata: dict[str, str] = {}
        self.json_scripts: list[str] = []
        self._ignored_stack: list[str] = []
        self._active_json_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {name.lower(): value or "" for name, value in attrs}
        if tag == "meta":
            key = values.get("property") or values.get("name")
            content = values.get("content", "").strip()
            if key and content and key.lower() in {"description", "og:description", "twitter:description"}:
                self.metadata[key.lower()] = content
        if tag == "script":
            script_type = values.get("type", "").lower()
            self._active_json_script = script_type == "application/ld+json"
            self._script_parts = []
            if not self._active_json_script:
                self._ignored_stack.append(tag)
            return
        if self._ignored_stack:
            if tag not in self._VOID_TAGS:
                self._ignored_stack.append(tag)
            return
        if tag in self._IGNORED_TAGS:
            self._ignored_stack.append(tag)
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script":
            if self._active_json_script:
                content = "".join(self._script_parts).strip()
                if content:
                    self.json_scripts.append(content)
            self._active_json_script = False
            self._script_parts = []
            if "script" in self._ignored_stack:
                index = len(self._ignored_stack) - 1 - self._ignored_stack[::-1].index("script")
                del self._ignored_stack[index:]
            return
        if self._ignored_stack:
            if tag in self._ignored_stack:
                index = len(self._ignored_stack) - 1 - self._ignored_stack[::-1].index(tag)
                del self._ignored_stack[index:]
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._active_json_script:
            self._script_parts.append(data)
        elif not self._ignored_stack:
            self.parts.append(data)


def _clean(value: Any, limit: int = 360) -> str | None:
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = ", ".join(str(item) for item in value if isinstance(item, (str, int, float)))
    else:
        return None
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n:;,.|·")
    if not text or len(text) > limit:
        return None
    return text


def _json_values(value: Any, fields: dict[str, str]) -> None:
    """Read standard schema.org product values when present."""
    if isinstance(value, list):
        for item in value:
            _json_values(item, fields)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        normalized = re.sub(r"[^a-zа-яё]", "", str(key).lower())
        mapping = {
            "description": "description",
            "region": "region",
            "grape": "grape",
            "grapevariety": "grape",
            "alcohol": "alcohol",
            "alcoholcontent": "alcohol",
            "servingtemperature": "serving_temperature",
            "foodpairing": "pairings",
            "aroma": "aroma",
            "taste": "taste",
        }
        target = mapping.get(normalized)
        if target and target not in fields:
            candidate = item
            if target == "alcohol" and isinstance(item, (int, float)):
                candidate = f"{item}%"
            cleaned = _clean(candidate)
            if cleaned:
                fields[target] = cleaned
        if normalized in {"aggregaterating", "rating"} and isinstance(item, dict):
            rating = _clean(item.get("ratingValue"))
            count = _clean(item.get("reviewCount") or item.get("ratingCount"), limit=32)
            if rating:
                fields["rating"] = f"{rating}/5" + (f" ({count})" if count else "")
        _json_values(item, fields)


_FIELD_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("region", ("регион производства", "регион")),
    ("grape", ("сорта винограда", "сорт винограда", "сорт")),
    ("category", ("категория и цвет", "категория", "тип вина", "тип")),
    ("color", ("цвет вина", "цвет")),
    ("alcohol", ("крепость", "содержание алкоголя", "алкоголь")),
    ("serving_temperature", ("температура подачи", "температура сервировки", "подача")),
    ("pairings", (
        "сочетание с блюдами", "гастрономические сочетания", "гастрономия", "сочетается с",
        "рекомендуется к", "подходит к", "к чему подходит", "сочетание с едой",
    )),
    ("rating", ("народный рейтинг", "рейтинг пользователей", "оценка пользователей", "оценка", "рейтинг")),
    ("aroma", ("аромат", "ароматика")),
    ("taste", ("вкус", "вкусовые характеристики", "вкусовые качества")),
)


def _extract_labeled_fields(lines: list[str], fields: dict[str, str]) -> None:
    normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    for index, line in enumerate(normalized_lines):
        low = line.casefold()
        aroma_and_taste = re.match(r"^аромат\s*[:—-]\s*(.*?)\s+вкус\s*[:—-]\s*(.*)$", line, re.IGNORECASE)
        if aroma_and_taste:
            fields.setdefault("aroma", aroma_and_taste.group(1).strip())
            fields.setdefault("taste", aroma_and_taste.group(2).strip())
        for key, labels in _FIELD_LABELS:
            if key in fields:
                continue
            for label in labels:
                if not low.startswith(label):
                    continue
                rest = line[len(label):].strip(" \t:;—–-|,%")
                if key == "category" and label == "категория и цвет":
                    values: list[str] = []
                    for following in normalized_lines[index + 1:index + 4]:
                        if not following:
                            continue
                        next_low = following.casefold()
                        if any(next_low.startswith(candidate) for _, candidates in _FIELD_LABELS for candidate in candidates):
                            break
                        values.append(following)
                    if values:
                        category = _clean(values[0], limit=120)
                        color = _clean(values[1], limit=240) if len(values) > 1 else None
                        if category:
                            fields["category"] = category
                        if color:
                            fields.setdefault("color", color)
                    break
                if key == "alcohol":
                    candidates = [rest, *normalized_lines[index + 1:index + 5]]
                    alcohol = next((
                        re.search(r"\d{1,2}(?:[.,]\d{1,2})?\s*%?", candidate)
                        for candidate in candidates
                        if re.search(r"\d{1,2}(?:[.,]\d{1,2})?\s*%?", candidate)
                    ), None)
                    if alcohol:
                        alcohol_value = alcohol.group(0).replace(',', '.').replace(' ', '')
                        fields[key] = alcohol_value if alcohol_value.endswith("%") else f"{alcohol_value}%"
                    break
                # A brief label is commonly followed by its value in the next block.
                if not rest or re.fullmatch(r"[, %]*", rest):
                    next_parts: list[str] = []
                    for following in normalized_lines[index + 1:index + 5]:
                        if not following:
                            continue
                        next_low = following.casefold()
                        if any(next_low.startswith(candidate) for _, candidates in _FIELD_LABELS for candidate in candidates):
                            break
                        next_parts.append(following)
                        if key not in {"pairings", "aroma", "taste"} or len(next_parts) >= 2:
                            break
                    rest = ", ".join(next_parts)
                cleaned = _clean(rest, limit=360 if key in {"pairings", "aroma", "taste"} else 120)
                if cleaned:
                    fields[key] = cleaned
                break


def parse_official_wine_page(page: bytes) -> dict[str, str]:
    parser = _WinePageText()
    parser.feed(page.decode("utf-8", errors="replace"))
    text = "\n".join(parser.parts)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    fields: dict[str, str] = {}
    for script in parser.json_scripts:
        try:
            _json_values(json.loads(script), fields)
        except (json.JSONDecodeError, RecursionError):
            continue
    _extract_labeled_fields(lines, fields)
    if "description" not in fields:
        for key in ("og:description", "description", "twitter:description"):
            description = _clean(parser.metadata.get(key), limit=600)
            if description and not {"aroma", "taste"}.issubset(fields):
                fields["description"] = description
                break
    return fields


def _fetch_page_via_windows_powershell(slug: str) -> bytes:
    """Use the Windows host's network from WSL, where direct egress may be disabled."""
    executable = shutil.which("powershell.exe")
    if not executable:
        raise FileNotFoundError("powershell.exe is unavailable")
    command = (
        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
        "$slug=[Console]::In.ReadToEnd().Trim(); "
        "$uri='https://vino-svoe.ru/wines/'+[Uri]::EscapeDataString($slug); "
        "$response=Invoke-WebRequest -UseBasicParsing -Uri $uri -TimeoutSec 5 "
        "-MaximumRedirection 3 -Headers @{Accept='text/html,application/xhtml+xml'}; "
        "$hostName=$response.BaseResponse.ResponseUri.Host; "
        "if($hostName -notin @('vino-svoe.ru','www.vino-svoe.ru')){exit 3}; "
        "[Console]::Out.Write([Convert]::ToBase64String("
        "[Text.Encoding]::UTF8.GetBytes([string]$response.Content)))"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        input=slug,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=REQUEST_TIMEOUT_SECONDS + 7,
        check=False,
    )
    if completed.returncode != 0:
        raise URLError("official site request through Windows host failed")
    encoded = completed.stdout.strip()
    if not encoded or len(encoded) > ((MAX_PAGE_BYTES + 2) // 3) * 4:
        raise ValueError("invalid official page response")
    try:
        page = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid official page encoding") from exc
    if len(page) > MAX_PAGE_BYTES:
        raise ValueError("page_too_large")
    return page


def get_official_wine_details(slug: str) -> dict[str, Any]:
    """Return cached official wine characteristics; network failures are non-fatal."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(slug)
        if cached and cached[0] > now:
            return dict(cached[1])

    url = f"{SITE_ROOT}/wines/{quote(slug, safe='-_')}"
    payload: dict[str, Any]
    try:
        if shutil.which("powershell.exe"):
            page = _fetch_page_via_windows_powershell(slug)
        else:
            request = Request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": "MyWine local wine-card enrichment/1.0",
                },
            )
            opener = build_opener(_OfficialRedirectHandler())
            with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                final_url = urlsplit(response.geturl())
                content_type = response.headers.get_content_type()
                if final_url.scheme != "https" or final_url.hostname not in {"vino-svoe.ru", "www.vino-svoe.ru"}:
                    raise ValueError("unexpected_redirect")
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError("unexpected_content_type")
                page = response.read(MAX_PAGE_BYTES + 1)
                if len(page) > MAX_PAGE_BYTES:
                    raise ValueError("page_too_large")
        payload = {
            "available": True,
            "source_url": url,
            "fields": parse_official_wine_page(page),
        }
        ttl = _CACHE_TTL_SECONDS
    except (HTTPError, URLError, TimeoutError, OSError, HTTPException, ValueError, subprocess.SubprocessError):
        payload = {"available": False, "source_url": url, "fields": {}}
        ttl = _FAILURE_CACHE_TTL_SECONDS

    with _cache_lock:
        _cache[slug] = (time.monotonic() + ttl, payload)
    return dict(payload)
