"""Local frozen recognition runtime and conservative Smart Retry decisions.

The runtime composes only already frozen artifacts: the validated R8 LoRA
checkpoint, the selected eslav reference-OCR blend, and SIFT Top-5 fusion.
It does not train, tune, or read benchmark labels at inference time.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from recognition.geometric_reranker import (  # noqa: E402
    DEFAULT_CONFIG as SIFT_CONFIG,
    cache_filename,
    extract_sift,
    file_sha256,
    fuse_scores,
    load_feature_cache,
    match_sift_pair,
    normalize_geometry_scores,
)
from recognition.ocr_engine import OCR_CONFIGS, create_ocr_engine  # noqa: E402
from recognition.ocr_reranker import FusionConfig, rerank_one_query  # noqa: E402
from recognition.so400m_lora import (  # noqa: E402
    BASE_MODEL_ID,
    BASE_REVISION,
    HIDDEN_SIZE,
    LoRALinear,
    inject_last_vision_lora,
    lora_sha256,
    load_lora_state_dict,
    pooled_image_features,
    set_lora_enabled,
)
from recognition.text_signals import (  # noqa: E402
    build_candidate_text_index,
    build_query_text_evidence,
    build_reference_ocr_evidence,
    compute_text_signals,
)

LORA_CHECKPOINT = ROOT / "artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/checkpoints/epoch_005.pt"
LORA_CHECKPOINT_SHA256 = "7ff3bba0cc416629ee159f1692a221295da322480d1c7b41f06d9eea1c41bb28"
LORA_STATE_SHA256 = "b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04"
ADAPTED_CACHE = ROOT / "artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/lora_evaluation/adapted_reference_cache"
ADAPTED_CACHE_METADATA = ADAPTED_CACHE / "metadata.json"
ADAPTED_CACHE_EMBEDDINGS = ADAPTED_CACHE / "embeddings.npy"
CATALOG_SLUGS = ROOT / "artifacts/reference_embeddings/siglip2_so400m_384/slugs.json"
OCR_REFERENCE_CACHE = ROOT / "artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/ocr/reference_ocr.jsonl"
SIFT_CONFIG_PATH = ROOT / "artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/config.json"
OCR_FUSION = FusionConfig(
    policy="reference_ocr_blend",
    alpha=0.30,
    vintage_bonus=0.05,
    vintage_penalty=0.05,
    min_text_margin=0.05,
)
SIFT_WEIGHT = 0.40
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 4096
MAX_OCR_IMAGE_SIDE = 1600
MIN_LONGEST_SIDE = 640
MIN_SHARPNESS = 12.0
MIN_TEXT_MARGIN = 0.05  # selected OCR text-margin guard
MIN_GEOMETRY_MARGIN = 0.15  # frozen conservative SIFT corroboration guard


def read_catalog(path: Path | None = None) -> list[dict[str, str]]:
    path = path or ROOT / "data/processed/catalog_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("empty_image")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    # Inspect dimensions before decoding pixels; the endpoint has its own hard
    # bound so it does not rely on Pillow's process-global warning threshold.
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(__import__("io").BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("image_too_large")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except ValueError as exc:
        if str(exc) == "image_too_large":
            raise
        raise ValueError("unreadable") from exc
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("unreadable") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image_too_large")
    if max(image.size) < MIN_LONGEST_SIDE:
        raise ValueError("resolution")
    if max(image.size) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image


def image_quality(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, 512.0 / max(image.size))
    resized = cv2.resize(
        rgb,
        (max(3, round(image.width * scale)), max(3, round(image.height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    highlights = float(np.mean((gray >= 250) & (hsv[:, :, 1] <= 28)))
    return {"sharpness": sharpness, "highlight_ratio": highlights}


def _box_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    try:
        values = np.asarray(box, dtype=np.float32)
        if values.shape == (4,):
            x1, y1, x2, y2 = values.tolist()
        elif values.ndim == 2 and values.shape[1] >= 2 and len(values) >= 2:
            points = values[:, :2]
            x1, y1 = points.min(axis=0).tolist()
            x2, y2 = points.max(axis=0).tolist()
        else:
            return None
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    except (TypeError, ValueError):
        return None


def _text_coverage(lines: Sequence[Any], width: int, height: int) -> tuple[float, float]:
    boxes = [coords for line in lines if (coords := _box_xyxy(getattr(line, "box", None))) is not None]
    if not boxes:
        return 0.0, 0.0
    area = sum(max(0.0, x2 - x1) * max(0.0, y2 - y1) for x1, y1, x2, y2 in boxes)
    max_line_height = max(y2 - y1 for _, y1, _, y2 in boxes)
    return area / max(width * height, 1), max_line_height / max(min(width, height), 1)


def _barcode_detected(image: Image.Image) -> bool:
    namespace = getattr(cv2, "barcode", None)
    detector_type = getattr(namespace, "BarcodeDetector", None) if namespace is not None else None
    if detector_type is None:
        detector_type = getattr(cv2, "barcode_BarcodeDetector", None)
    if detector_type is None:
        return False
    try:
        detector = detector_type()
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        found, decoded, _types, points = detector.detectAndDecodeWithType(bgr)
        has_points = points is not None and np.asarray(points).size > 0
        decoded_values = list(decoded) if decoded is not None else []
        return bool(found and (has_points or any(str(value).strip() for value in decoded_values)))
    except (cv2.error, TypeError, ValueError):
        return False


def _title_without_year(title: str) -> str:
    without_year = re.sub(r"\b(?:19|20)\d{2}\b", " ", title)
    without_year = re.sub(r"\s+", " ", without_year).strip().casefold()
    return without_year


class _SiftReader:
    def __init__(self, cache_dir: Path, paths: Mapping[str, Path], image_hashes: Mapping[str, str], fingerprint: str):
        self.cache_dir = cache_dir
        self.paths = dict(paths)
        self.image_hashes = dict(image_hashes)
        self.fingerprint = fingerprint
        self.memory: OrderedDict[str, Any] = OrderedDict()
        self.max_items = 96

    def get(self, slug: str):
        if slug in self.memory:
            self.memory.move_to_end(slug)
            return self.memory[slug]
        features = load_feature_cache(
            self.cache_dir / cache_filename(slug),
            self.image_hashes[slug],
            self.fingerprint,
        )
        if features is None:
            raise RuntimeError(f"Frozen SIFT feature cache mismatch for {slug}")
        self.memory[slug] = features
        if len(self.memory) > self.max_items:
            self.memory.popitem(last=False)
        return features


class SmartRetryRuntime:
    """Load and serve one frozen recognition stack for the local desktop app."""

    def __init__(self, root: Path = ROOT, ocr_device: str | None = None):
        self.root = root.resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ocr_device = ocr_device or ("gpu:0" if self.device.type == "cuda" else "cpu")
        self.lock = threading.Lock()
        self.catalog = [row for row in read_catalog(self.root / "data/processed/catalog_manifest.csv")
                        if row.get("slug") and row.get("reference_image_path")]
        self.by_slug = {row["slug"]: row for row in self.catalog}
        print("Validating frozen adapted reference images…", flush=True)
        self.slugs, self.reference_embeddings, image_hashes = self._load_adapted_cache()
        self.candidate_text = {slug: build_candidate_text_index(row) for slug, row in self.by_slug.items()}
        print("Loading frozen OCR and SIFT evidence…", flush=True)
        self.reference_ocr = self._load_reference_ocr()
        self.sift_reader = self._load_sift_reader(image_hashes)
        print("Loading validated R8 encoder…", flush=True)
        self.model, self.processor = self._load_encoder()
        print(f"Loading query OCR on {ocr_device}…", flush=True)
        self.ocr = create_ocr_engine(OCR_CONFIGS["current_eslav"], device=ocr_device)
        if self.device.type == "cuda" and ocr_device == "gpu:0":
            self._warmup()

    def _warmup(self) -> None:
        """Initialize kernels across representative catalog images before serving."""
        if not self.catalog:
            raise RuntimeError("No catalog reference is available for runtime warm-up")
        indexes = sorted({0, len(self.catalog) // 2, len(self.catalog) - 1})
        print(f"Warming the local inference path on {len(indexes)} catalog references…", flush=True)
        for index in indexes:
            image_path = self.root / self.catalog[index]["reference_image_path"]
            self.recognize(image_path.read_bytes())

    def _load_adapted_cache(self) -> tuple[list[str], np.ndarray, dict[str, str]]:
        metadata = json.loads(ADAPTED_CACHE_METADATA.read_text(encoding="utf-8"))
        slugs = json.loads(CATALOG_SLUGS.read_text(encoding="utf-8"))
        expected = metadata.get("catalog_reference_image_hashes", {})
        if metadata.get("base_model_id") != BASE_MODEL_ID or metadata.get("base_revision") != BASE_REVISION:
            raise RuntimeError("Adapted reference cache uses a different frozen base model")
        if metadata.get("lora_checkpoint_sha256") != LORA_STATE_SHA256:
            raise RuntimeError("Adapted reference cache is not aligned with the validated R8 checkpoint")
        if int(metadata.get("reference_count", -1)) != len(slugs) or len(slugs) != len(self.by_slug):
            raise RuntimeError("Adapted reference cache and catalog row counts differ")
        if set(slugs) != set(self.by_slug) or set(expected) != set(self.by_slug):
            raise RuntimeError("Adapted reference cache slugs do not align with the usable catalog")

        def hash_reference(slug: str) -> tuple[str, str]:
            path = self.root / self.by_slug[slug]["reference_image_path"]
            return slug, file_sha256(path)

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="catalog-hash") as pool:
            image_hashes = dict(pool.map(hash_reference, slugs))
        for slug, digest in image_hashes.items():
            if digest != expected.get(slug):
                raise RuntimeError(f"Catalog image changed since the frozen adapted cache: {slug}")

        embeddings = np.load(ADAPTED_CACHE_EMBEDDINGS, mmap_mode="r")
        if embeddings.shape != (len(slugs), HIDDEN_SIZE) or not np.isfinite(embeddings).all():
            raise RuntimeError("Frozen adapted reference embeddings are malformed")
        norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise RuntimeError("Frozen adapted reference embeddings are not normalized")
        return slugs, embeddings, image_hashes

    def _load_reference_ocr(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        with OCR_REFERENCE_CACHE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                slug = str(row.get("slug", ""))
                if slug not in self.by_slug:
                    continue
                lines = [(str(item.get("text", "")), float(item.get("confidence", 0.0)))
                         for item in row.get("lines", [])]
                output[slug] = build_reference_ocr_evidence(lines)
        if len(output) != len(self.by_slug):
            raise RuntimeError("Frozen reference OCR cache does not cover the current catalog")
        return output

    def _load_sift_reader(self, image_hashes: Mapping[str, str]) -> _SiftReader:
        config = json.loads(SIFT_CONFIG_PATH.read_text(encoding="utf-8"))
        cache_dir = self.root / config["reference_cache"]["path"]
        index = json.loads((cache_dir / "cache_index.json").read_text(encoding="utf-8"))
        expected_slugs = list(self.by_slug)
        if index.get("opencv_version") != cv2.__version__:
            raise RuntimeError("Frozen SIFT cache was built with a different OpenCV version")
        if index.get("sift_config") != asdict(SIFT_CONFIG):
            raise RuntimeError("Frozen SIFT cache configuration does not match the selected pipeline")
        if index.get("slugs_in_order") != expected_slugs or index.get("reference_image_sha256") != dict(image_hashes):
            raise RuntimeError("Frozen SIFT cache and catalog images are misaligned")
        if int(index.get("reference_count", -1)) != len(expected_slugs):
            raise RuntimeError("Frozen SIFT cache reference count is invalid")
        return _SiftReader(cache_dir, {slug: self.root / self.by_slug[slug]["reference_image_path"]
                                       for slug in expected_slugs}, image_hashes,
                           str(index["cache_fingerprint"]))

    def _load_encoder(self):
        if not LORA_CHECKPOINT.is_file() or file_sha256(LORA_CHECKPOINT) != LORA_CHECKPOINT_SHA256:
            raise RuntimeError("Validated R8 epoch-5 checkpoint is missing or has changed")
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION,
                                                       local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION, dtype=dtype,
                                          local_files_only=True)
        model.requires_grad_(False)
        model.vision_model.to(self.device)
        inject_last_vision_lora(model, rank=8, alpha=16, dropout=0.05, last_blocks=4)
        checkpoint = torch.load(LORA_CHECKPOINT, map_location="cpu", weights_only=False)
        state = checkpoint.get("lora_state")
        if not isinstance(state, Mapping) or lora_sha256(state) != LORA_STATE_SHA256:
            raise RuntimeError("Validated R8 LoRA state checksum does not match")
        load_lora_state_dict(model, state)
        set_lora_enabled(model, True)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.eval()
                module.requires_grad_(False)
        model.eval()
        return model, processor

    def _encode_query(self, image: Image.Image) -> np.ndarray:
        pixels = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        dtype = next(self.model.vision_model.parameters()).dtype
        pixels = pixels.to(device=self.device, dtype=dtype, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                   enabled=self.device.type == "cuda"):
            vector = pooled_image_features(self.model, pixels)
        return vector[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            image = decode_image(image_bytes)
        except ValueError as exc:
            reason = str(exc)
            return {"status": "retry", "reason": reason if reason in {"resolution", "image_too_large"}
                    else "unreadable"}

        quality = image_quality(image)
        if quality["sharpness"] < MIN_SHARPNESS:
            return {"status": "retry", "reason": "blur", "diagnostics": quality}

        with self.lock:
            return self._recognize_decoded(image, quality)

    def predict_slug(self, image_bytes: bytes) -> dict[str, str]:
        """Return the hackathon evaluator's flat slug contract for a decodable image."""
        image = decode_image(image_bytes)
        with self.lock:
            result = self._recognize_decoded(image, image_quality(image))
        slug = str(result.get("slug") or "")
        if not slug:
            raise ValueError("unreadable")
        return {"slug": slug}

    def _recognize_decoded(self, image: Image.Image, quality: Mapping[str, float]) -> dict[str, Any]:
        # PaddleOCR's detector is very slow near its 4000 px input limit. Keep
        # full resolution for the frozen encoder/SIFT path, but make a bounded
        # copy for OCR. Its boxes are then measured against that copy below.
        ocr_image = image
        if max(image.size) > MAX_OCR_IMAGE_SIDE:
            scale = MAX_OCR_IMAGE_SIDE / max(image.size)
            ocr_image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        phase_started = time.perf_counter()
        print("[inference] phase=ocr start", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temporary:
            ocr_image.save(temporary, format="JPEG", quality=95)
            temporary.flush()
            try:
                ocr_lines = self.ocr.predict_lines(temporary.name)
            finally:
                print(f"[inference] phase=ocr seconds={time.perf_counter() - phase_started:.3f}", flush=True)

        query_evidence = build_query_text_evidence(
            [(line.text, line.confidence) for line in ocr_lines]
        )
        text_area_ratio, max_text_height_ratio = _text_coverage(
            ocr_lines, ocr_image.width, ocr_image.height
        )
        phase_started = time.perf_counter()
        print("[inference] phase=encoder start", flush=True)
        try:
            query_vector = self._encode_query(image)
        finally:
            print(f"[inference] phase=encoder seconds={time.perf_counter() - phase_started:.3f}", flush=True)
        similarities = np.asarray(self.reference_embeddings, dtype=np.float32) @ query_vector
        image_order = sorted(range(len(self.slugs)), key=lambda index: (-float(similarities[index]), self.slugs[index]))[:5]
        image_top5 = [self.slugs[index] for index in image_order]
        image_scores = [float(similarities[index]) for index in image_order]

        candidate_signals = [
            compute_text_signals(query_evidence, self.candidate_text[slug], self.reference_ocr.get(slug))
            for slug in image_top5
        ]
        ocr = rerank_one_query(image_scores, candidate_signals, OCR_FUSION)
        ocr_order = list(ocr.final_order)
        ocr_slugs = [image_top5[index] for index in ocr_order]
        ocr_scores = [float(ocr.final_scores[index]) for index in ocr_order]

        query_rgb = np.asarray(image, dtype=np.uint8)
        phase_started = time.perf_counter()
        print("[inference] phase=sift start", flush=True)
        try:
            query_sift = extract_sift(query_rgb, SIFT_CONFIG)
            geometries = [match_sift_pair(query_sift, self.sift_reader.get(slug), SIFT_CONFIG) for slug in ocr_slugs]
        finally:
            print(f"[inference] phase=sift seconds={time.perf_counter() - phase_started:.3f}", flush=True)
        geometry_scores = [item.geometric_score if item.homography_valid else 0.0 for item in geometries]
        fused_scores = fuse_scores(ocr_scores, geometry_scores, SIFT_WEIGHT)
        final_order = sorted(range(5), key=lambda index: (-float(fused_scores[index]), index))
        final_slugs = [ocr_slugs[index] for index in final_order]
        winner = final_slugs[0]

        text_scores = [float(signals["reference_ocr_score"]) for signals in candidate_signals]
        text_ranking = sorted(range(5), key=lambda index: (-text_scores[index], index))
        text_margin = text_scores[text_ranking[0]] - text_scores[text_ranking[1]]
        text_winner = image_top5[text_ranking[0]]
        text_votes = bool(query_evidence.text and text_margin >= MIN_TEXT_MARGIN and text_winner == winner)

        normalized_geometry = normalize_geometry_scores(geometry_scores)
        geometry_ranking = sorted(range(5), key=lambda index: (-normalized_geometry[index], index))
        geometry_margin = normalized_geometry[geometry_ranking[0]] - normalized_geometry[geometry_ranking[1]]
        geometry_winner = ocr_slugs[geometry_ranking[0]]
        geometry_votes = bool(
            geometries[geometry_ranking[0]].homography_valid
            and geometry_margin >= MIN_GEOMETRY_MARGIN
            and geometry_winner == winner
        )
        image_winner = image_top5[0]
        votes = int(image_winner == winner) + int(text_votes) + int(geometry_votes)

        if votes < 2:
            reason = self._retry_reason(
                image, ocr_lines, query_evidence, image_top5, candidate_signals,
                geometries, quality, text_area_ratio, max_text_height_ratio,
            )
            return {
                "status": "retry",
                "reason": reason,
                "slug": winner,
                "diagnostics": {
                    "signal_agreement": votes,
                    "ocr_lines": len(ocr_lines),
                    "query_has_vintage": bool(query_evidence.vintage_years),
                    "text_margin": round(float(text_margin), 4),
                    "geometry_margin": round(float(geometry_margin), 4),
                },
            }

        product = dict(self.by_slug[winner])
        return {
            "status": "found",
            "slug": winner,
            "product": {key: product.get(key, "") for key in (
                "slug", "title", "category", "color", "region", "grape", "winery", "reference_image_path"
            )},
            "evidence": {
                "signal_agreement": votes,
                "ocr_lines": len(ocr_lines),
                "image_winner_matches": image_winner == winner,
                "ocr_corroborates": text_votes,
                "geometry_corroborates": geometry_votes,
            },
        }

    def _retry_reason(
        self,
        image: Image.Image,
        ocr_lines: Sequence[Any],
        query_evidence: Any,
        top5: Sequence[str],
        candidate_signals: Sequence[Mapping[str, float | str]],
        geometries: Sequence[Any],
        quality: Mapping[str, float],
        text_area_ratio: float,
        max_text_height_ratio: float,
    ) -> str:
        barcode = _barcode_detected(image)
        if barcode and len(query_evidence.tokens) < 5:
            return "barcode"
        if quality["highlight_ratio"] >= 0.16 and (len(ocr_lines) < 3 or text_area_ratio < 0.004):
            return "glare"
        if ocr_lines and (text_area_ratio < 0.003 or max_text_height_ratio < 0.012):
            return "distance"
        if not query_evidence.vintage_years:
            for left_index, left_slug in enumerate(top5):
                left = self.by_slug[left_slug]
                left_year = self.candidate_text[left_slug].vintage_year
                if not left_year:
                    continue
                for right_slug in top5[left_index + 1:]:
                    right_year = self.candidate_text[right_slug].vintage_year
                    if right_year and right_year != left_year and _title_without_year(left.get("title", "")) == _title_without_year(self.by_slug[right_slug].get("title", "")):
                        return "year"
        if not ocr_lines or not query_evidence.text:
            if quality["highlight_ratio"] >= 0.10:
                return "glare"
            return "front_label"
        if len(geometries) and sum(item.homography_valid for item in geometries) == 0:
            return "uncertain"
        return "uncertain"
