"""Small research-only residual metric adapter for frozen image embeddings.

All training inputs are catalog reference images and deterministic views made
from those images. Benchmark data belongs to the separate evaluation runner.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class ResidualMetricAdapter(nn.Module):
    """Identity-initialized residual MLP that returns L2-normalized vectors."""

    def __init__(self, input_dim: int = 1152, hidden_dim: int = 256, residual_scale: float = 0.1):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or residual_scale < 0:
            raise ValueError("invalid adapter dimensions or residual scale")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(self.input_dim, self.hidden_dim)
        self.up = nn.Linear(self.hidden_dim, self.input_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"expected embedding dimension {self.input_dim}, got {embeddings.shape[-1]}")
        base = embeddings.float()
        residual = self.up(F.gelu(self.down(base))).float()
        return F.normalize(base + self.residual_scale * residual, p=2, dim=-1, eps=1e-12)


def derive_augmentation_seed(global_seed: int, slug: str, split: str, view_index: int) -> int:
    """Stable split-specific seed; independent of Python's randomized hash()."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    payload = f"{int(global_seed)}\0{slug}\0{split}\0{int(view_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def build_view_manifest(
    catalog_rows: Sequence[Mapping[str, str]],
    image_sha256: Mapping[str, str],
    global_seed: int,
    train_views_per_sku: int = 8,
    validation_views_per_sku: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Describe catalog-only augmented views without admitting query data."""
    if train_views_per_sku < 1 or validation_views_per_sku < 1:
        raise ValueError("train and validation view counts must both be positive")
    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for row in catalog_rows:
        slug, reference_path = row.get("slug", ""), row.get("reference_image_path", "")
        if not slug or not reference_path or not reference_path.startswith("data/processed/reference_images/"):
            raise ValueError("view manifests may contain catalog reference images only")
        if slug not in image_sha256:
            raise KeyError(f"missing catalog reference image hash for {slug}")
        for split, count, output in (
            ("train", train_views_per_sku, train_rows),
            ("validation", validation_views_per_sku, validation_rows),
        ):
            for view_index in range(count):
                output.append({
                    "split": split,
                    "slug": slug,
                    "reference_image_path": reference_path,
                    "reference_sha256": image_sha256[slug],
                    "view_index": view_index,
                    "augmentation_seed": derive_augmentation_seed(global_seed, slug, split, view_index),
                    "augmentation_version": "catalog_capture_v1",
                })
    assert_training_manifest_catalog_only(train_rows)
    assert_training_manifest_catalog_only(validation_rows)
    return train_rows, validation_rows


def assert_training_manifest_catalog_only(rows: Sequence[Mapping[str, object]]) -> None:
    """Fail closed if a training manifest exposes benchmark/query/GT fields."""
    forbidden_field_fragments = ("benchmark", "query", "target", "ground_truth", "prediction", "rank")
    for row in rows:
        if any(fragment in str(key).casefold() for key in row for fragment in forbidden_field_fragments):
            raise ValueError("training manifest contains a benchmark/query/ground-truth field")
        path = str(row.get("reference_image_path", ""))
        if not path.startswith("data/processed/reference_images/"):
            raise ValueError("training manifest contains a non-catalog image path")
        if "data/benchmarks/" in path or "/queries/" in path:
            raise ValueError("benchmark images are forbidden in the training manifest")


def rank_fixed_top5(candidate_slugs: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Stable ranking over exactly the original five candidates."""
    if len(candidate_slugs) != 5 or len(scores) != 5 or len(set(candidate_slugs)) != 5:
        raise ValueError("metric adapter must rerank exactly five unique frozen candidates")
    return [slug for _, slug in sorted(enumerate(candidate_slugs), key=lambda item: (-float(scores[item[0]]), item[0]))]


def transition_name(baseline_correct: bool, candidate_correct: bool) -> str:
    if baseline_correct:
        return "correct_to_correct" if candidate_correct else "correct_to_wrong"
    return "wrong_to_correct" if candidate_correct else "wrong_to_wrong"


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", (value or "").casefold().replace("ё", "е"))


def catalog_family_keys(catalog_rows: Sequence[Mapping[str, str]]) -> dict[str, str | None]:
    """Derive conservative product-line keys only from catalog metadata.

    The manifest has no explicit family column. A family proxy requires the
    same winery and at least two shared distinctive title-core tokens after
    removing catalog grape terms, years, winery tokens, and generic wine words.
    """
    generic = set(_tokens(
        "wine vino вино белое красное розовое оранжевое сухое полусухое "
        "полусладкое сладкое брют экстра игристое тихое резерв резервное "
        "купаж сорт виноград выдержанное молодое бутылка"
    ))
    grape_lexicon = {token for row in catalog_rows for token in _tokens(row.get("grape", ""))}
    result: dict[str, str | None] = {}
    for row in catalog_rows:
        slug = row.get("slug", "")
        winery = " ".join(_tokens(row.get("winery", "")))
        winery_tokens = set(winery.split())
        title_tokens = _tokens(row.get("title", ""))
        core = sorted({
            token for token in title_tokens
            if len(token) >= 3 and not token.isdigit() and not re.fullmatch(r"\d+(?:\d+)?", token)
            and token not in generic and token not in grape_lexicon and token not in winery_tokens
            and not re.fullmatch(r"(?:19|20)\d{2}", token)
        })
        result[slug] = f"{winery}|{' '.join(core)}" if winery and len(core) >= 2 else None
    return result


def build_hard_negative_map(
    slugs: Sequence[str],
    reference_embeddings: np.ndarray,
    family_keys: Mapping[str, str | None],
    seed: int,
    *,
    family_count: int = 4,
    hard_count: int = 8,
    random_count: int = 2,
    neighbor_pool: int = 20,
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[dict[str, object]]]:
    """Create per-anchor negatives using only catalog metadata and ref vectors."""
    vectors = np.asarray(reference_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(slugs):
        raise ValueError("reference embeddings must align with catalog slugs")
    if len(set(slugs)) != len(slugs):
        raise ValueError("catalog slugs must be unique")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    sims = vectors @ vectors.T
    index_by_slug = {slug: i for i, slug in enumerate(slugs)}
    family_members: dict[str, list[int]] = defaultdict(list)
    for index, slug in enumerate(slugs):
        key = family_keys.get(slug)
        if key:
            family_members[key].append(index)

    negative_indices: dict[str, list[int]] = {}
    family_indices: dict[str, list[int]] = {}
    rows: list[dict[str, object]] = []
    for anchor_index, slug in enumerate(slugs):
        key = family_keys.get(slug)
        family_candidates = [i for i in family_members.get(key, []) if i != anchor_index] if key else []
        family_candidates.sort(key=lambda i: (-float(sims[anchor_index, i]), slugs[i]))
        family_indices[slug] = family_candidates
        selected: list[tuple[int, str]] = []
        for i in family_candidates[:family_count]:
            selected.append((i, "catalog_metadata_same_family"))

        neighbor_candidates = [i for i in range(len(slugs)) if i != anchor_index]
        neighbor_candidates.sort(key=lambda i: (-float(sims[anchor_index, i]), slugs[i]))
        neighbor_candidates = neighbor_candidates[:neighbor_pool]
        for i in neighbor_candidates:
            if len(selected) >= hard_count:
                break
            if i not in {item[0] for item in selected}:
                selected.append((i, "frozen_so400m_reference_top20"))

        excluded = {anchor_index, *(i for i, _ in selected)}
        random_pool = [i for i in range(len(slugs)) if i not in excluded and
                       (not key or family_keys.get(slugs[i]) != key)]
        anchor_seed = derive_augmentation_seed(seed, slug, "train", 999_983)
        rng = random.Random(anchor_seed)
        if len(random_pool) < random_count:
            random_pool = [i for i in range(len(slugs)) if i not in excluded]
        random_choices = rng.sample(random_pool, k=min(random_count, len(random_pool)))
        selected.extend((i, "uniform_catalog_random") for i in random_choices)

        # If tiny catalogs or overlapping pools leave a short set, fill from
        # the frozen nearest-reference ordering without ever selecting self.
        for i in neighbor_candidates:
            if len(selected) >= hard_count + random_count:
                break
            if i not in {item[0] for item in selected}:
                selected.append((i, "frozen_so400m_reference_top20_fallback"))
        if len(selected) != hard_count + random_count:
            raise ValueError(f"could not build fixed negative set for {slug}")
        if any(i == anchor_index for i, _ in selected) or len({i for i, _ in selected}) != len(selected):
            raise AssertionError(f"invalid negative set for {slug}")
        negative_indices[slug] = [i for i, _ in selected]
        for priority, (negative_index, source) in enumerate(selected, start=1):
            rows.append({
                "anchor_slug": slug,
                "negative_slug": slugs[negative_index],
                "negative_rank": priority,
                "negative_source": source,
                "catalog_family_key": key or "",
                "frozen_reference_cosine": float(sims[anchor_index, negative_index]),
            })
    return negative_indices, family_indices, rows


def candidate_index_matrix(slugs: Sequence[str], negative_indices: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Return positive-first fixed candidate IDs for one row per catalog SKU."""
    by_index = {slug: i for i, slug in enumerate(slugs)}
    matrix = []
    for slug in slugs:
        negatives = list(negative_indices[slug])
        if by_index[slug] in negatives or len(set(negatives)) != len(negatives):
            raise ValueError(f"positive/negative candidate collision for {slug}")
        matrix.append([by_index[slug], *negatives])
    return np.asarray(matrix, dtype=np.int64)


def reference_cache_fingerprint(
    base_fingerprint: str,
    adapter_fingerprint: str,
    slugs: Sequence[str],
    image_sha256: Mapping[str, str],
) -> str:
    payload = {
        "base_reference_fingerprint": base_fingerprint,
        "adapter_fingerprint": adapter_fingerprint,
        "references": sorted((slug, image_sha256[slug]) for slug in slugs),
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def adapter_state_fingerprint(adapter: ResidualMetricAdapter) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_catalog_view(image: Image.Image, seed: int) -> Image.Image:
    """Create a deterministic, non-generative user-capture augmentation."""
    rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))
    source = image.convert("RGB")
    source.thumbnail((640, 640), Image.Resampling.LANCZOS)
    rgb = np.asarray(source, dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    if min(h, w) < 16:
        raise ValueError("catalog image is too small to augment")

    # Mild/moderate affine capture changes.
    angle = rng.uniform(-9.0, 9.0)
    scale = rng.uniform(0.88, 1.12)
    tx = rng.uniform(-0.055, 0.055) * w
    ty = rng.uniform(-0.055, 0.055) * h
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    border = tuple(int(np.clip(v, 0, 255)) for v in np.median(edge, axis=0))
    warped = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    if rng.random() < 0.8:
        jitter = rng.uniform(0.008, 0.035)
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst_pts = src_pts + np.float32([
            [rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h]
            for _ in range(4)
        ])
        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(warped, homography, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    # Imperfect crop and smaller object/framing variation on a neutral catalog-derived canvas.
    if rng.random() < 0.55:
        crop_fraction = rng.uniform(0.86, 0.98)
        crop_w, crop_h = max(8, int(w * crop_fraction)), max(8, int(h * crop_fraction))
        left = rng.randint(0, max(0, w - crop_w))
        top = rng.randint(0, max(0, h - crop_h))
        warped = cv2.resize(warped[top:top + crop_h, left:left + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.42:
        frame_scale = rng.uniform(0.72, 0.91)
        inner_w, inner_h = max(8, int(w * frame_scale)), max(8, int(h * frame_scale))
        canvas = np.empty_like(warped)
        canvas[:] = border
        resized = cv2.resize(warped, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        x = rng.randint(0, max(0, w - inner_w))
        y = rng.randint(0, max(0, h - inner_h))
        canvas[y:y + inner_h, x:x + inner_w] = resized
        warped = canvas

    result = warped.astype(np.float32)
    # Mild exposure, contrast, and warm/cool white-balance changes.
    contrast = rng.uniform(0.86, 1.14)
    brightness = rng.uniform(-14.0, 14.0)
    channel_gain = np.asarray([rng.uniform(0.94, 1.06) for _ in range(3)], dtype=np.float32)
    result = ((result - 127.5) * contrast + 127.5 + brightness) * channel_gain[None, None, :]
    if rng.random() < 0.58:
        x_axis = np.linspace(rng.uniform(0.92, 1.0), rng.uniform(1.0, 1.08), w, dtype=np.float32)
        y_axis = np.linspace(rng.uniform(0.95, 1.0), rng.uniform(1.0, 1.05), h, dtype=np.float32)
        result *= (y_axis[:, None] * x_axis[None, :])[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    if rng.random() < 0.22:
        mask = np.zeros((h, w), dtype=np.float32)
        center = (rng.randrange(w), rng.randrange(h))
        axes = (max(5, int(w * rng.uniform(0.025, 0.08))), max(5, int(h * rng.uniform(0.02, 0.07))))
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(2, min(w, h) * 0.025))
        alpha = rng.uniform(0.025, 0.11) * mask[:, :, None]
        result = np.clip(result.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)
    if rng.random() < 0.28:
        k = rng.choice([3, 5])
        if rng.random() < 0.5:
            result = cv2.GaussianBlur(result, (k, k), rng.uniform(0.2, 0.65))
        else:
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            result = cv2.filter2D(result, -1, kernel)
    if rng.random() < 0.30:
        factor = rng.uniform(0.62, 0.90)
        small = cv2.resize(result, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.35:
        noise = np_rng.normal(0.0, rng.uniform(1.0, 4.0), size=result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.48:
        quality = rng.randint(58, 94)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(result, mode="RGB")
