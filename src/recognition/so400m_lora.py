"""Catalog-only LoRA adaptation helpers for the local SigLIP2 SO400M model.

This module deliberately knows nothing about benchmark queries or predictions.
It contains the LoRA injection, deterministic capture-v2 view generation,
catalog-only split manifests, and checkpoint fingerprint utilities used by the
controlled milestone runner.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


BASE_MODEL_ID = "google/siglip2-so400m-patch14-384"
BASE_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
HIDDEN_SIZE = 1152
VISION_BLOCKS = 27
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
CAPTURE_BUCKETS = ("clean-ish", "perspective", "background_scale", "blur", "glare")
CAPTURE_VERSION = "catalog_capture_v2"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(global_seed: int, slug: str, split: str, view_index: int) -> int:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    payload = f"{int(global_seed)}\0{slug}\0{split}\0{int(view_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def assert_catalog_only_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    forbidden = ("benchmark", "query", "target", "ground_truth", "prediction", "rank")
    for row in rows:
        if any(part in str(key).casefold() for key in row for part in forbidden):
            raise ValueError("training manifest contains benchmark/query/ground-truth fields")
        path = str(row.get("reference_image_path", ""))
        if not path.startswith("data/processed/reference_images/"):
            raise ValueError("training manifest must contain catalog reference images only")
        if "data/benchmarks/" in path or "/queries/" in path:
            raise ValueError("benchmark images are forbidden in training manifests")


def build_capture_manifest(
    catalog_rows: Sequence[Mapping[str, str]],
    image_hashes: Mapping[str, str],
    seed: int,
    train_views_per_sku: int = 8,
    validation_views_per_sku: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_views_per_sku < 1 or validation_views_per_sku < 1:
        raise ValueError("train and validation view counts must be positive")
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for product in catalog_rows:
        slug = product.get("slug", "")
        path = product.get("reference_image_path", "")
        if not slug or not path or not path.startswith("data/processed/reference_images/"):
            raise ValueError("catalog manifest has a missing slug or non-reference image path")
        if slug not in image_hashes:
            raise KeyError(f"missing reference image hash for {slug}")
        for split, count in (("train", train_views_per_sku), ("validation", validation_views_per_sku)):
            for view_index in range(count):
                view_seed = derive_seed(seed, slug, split, view_index)
                bucket = CAPTURE_BUCKETS[view_seed % len(CAPTURE_BUCKETS)]
                output[split].append({
                    "split": split,
                    "slug": slug,
                    "reference_image_path": path,
                    "reference_sha256": image_hashes[slug],
                    "view_index": view_index,
                    "augmentation_seed": view_seed,
                    "capture_bucket": bucket,
                    "augmentation_version": CAPTURE_VERSION,
                })
    assert_catalog_only_manifest(output["train"])
    assert_catalog_only_manifest(output["validation"])
    if {row["augmentation_seed"] for row in output["train"]} & {
        row["augmentation_seed"] for row in output["validation"]
    }:
        raise AssertionError("train and validation view seeds overlap")
    return output["train"], output["validation"]


def _procedural_background(size: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """Create a subdued indoor-like background without external assets."""
    hue = rng.uniform(0.0, 1.0)
    if hue < 0.25:
        base = np.asarray([rng.randint(185, 232), rng.randint(178, 222), rng.randint(160, 208)], dtype=np.float32)
    elif hue < 0.5:
        base = np.asarray([rng.randint(92, 166), rng.randint(96, 162), rng.randint(88, 148)], dtype=np.float32)
    elif hue < 0.75:
        base = np.asarray([rng.randint(146, 205), rng.randint(142, 196), rng.randint(128, 182)], dtype=np.float32)
    else:
        base = np.asarray([rng.randint(62, 126), rng.randint(66, 128), rng.randint(70, 134)], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :, None]
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None, None]
    gx, gy = rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10)
    canvas = np.broadcast_to(base, (size, size, 3)).copy()
    canvas *= np.clip(1.0 + gx * x + gy * y, 0.74, 1.22)
    low = np_rng.normal(0.0, 1.0, (max(4, size // 36), max(4, size // 36))).astype(np.float32)
    low = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    canvas += low[:, :, None] * rng.uniform(2.0, 7.0)
    # A few wide, faint vertical/wall-like bands add texture without creating
    # high-frequency label-shaped patterns.
    if rng.random() < 0.65:
        band_x = rng.randrange(size)
        band_width = rng.randint(size // 16, size // 5)
        band = np.zeros((size, size), dtype=np.float32)
        left, right = max(0, band_x - band_width), min(size, band_x + band_width)
        if right > left:
            band[:, left:right] = 1.0
            band = cv2.GaussianBlur(band, (0, 0), max(2, size / 24))
            canvas += (band[:, :, None] - 0.25) * rng.uniform(-7.0, 7.0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _view_ranges(bucket: str, split: str) -> tuple[tuple[float, float], float, float]:
    """Return object/card scale and perspective jitter; validation is shifted."""
    if split == "train":
        ranges = {
            "clean-ish": ((0.82, 1.00), 0.012, 0.08),
            "perspective": ((0.66, 0.92), 0.055, 0.18),
            "background_scale": ((0.50, 0.76), 0.025, 0.18),
            "blur": ((0.63, 0.88), 0.035, 0.16),
            "glare": ((0.69, 0.94), 0.035, 0.16),
        }
    else:
        ranges = {
            "clean-ish": ((0.78, 0.98), 0.020, 0.10),
            "perspective": ((0.61, 0.88), 0.070, 0.22),
            "background_scale": ((0.46, 0.71), 0.035, 0.22),
            "blur": ((0.59, 0.84), 0.045, 0.20),
            "glare": ((0.65, 0.91), 0.045, 0.20),
        }
    if bucket not in ranges:
        raise ValueError(f"unknown capture bucket: {bucket}")
    return ranges[bucket]


def make_capture_view_v2(image: Image.Image, seed: int, split: str, bucket: str, size: int = 512) -> Image.Image:
    """Make a deterministic user-like capture from a catalog reference.

    No object mask is assumed. The complete reference photo is placed as a
    softly shadowed, perspective-distorted photo card on a procedural scene.
    This models framing and capture variation without pretending to remove a
    bottle background that cannot be segmented reliably.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if min(source.shape[:2]) < 16:
        raise ValueError("catalog reference is too small to augment")
    h, w = source.shape[:2]
    (scale_low, scale_high), perspective, _ = _view_ranges(bucket, split)
    scale = rng.uniform(scale_low, scale_high)
    max_dim = max(32, round(size * scale))
    fit = min(max_dim / max(w, 1), max_dim / max(h, 1))
    card_w, card_h = max(24, round(w * fit)), max(24, round(h * fit))
    source = cv2.resize(source, (card_w, card_h), interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_CUBIC)

    canvas = _procedural_background(size, rng, np_rng).astype(np.float32)
    center_x = size * rng.uniform(0.40, 0.60)
    center_y = size * rng.uniform(0.40, 0.60)
    if bucket == "background_scale":
        center_x = size * rng.uniform(0.28, 0.72)
        center_y = size * rng.uniform(0.28, 0.72)
    elif bucket == "perspective":
        center_x = size * rng.uniform(0.32, 0.68)
        center_y = size * rng.uniform(0.30, 0.70)
    x0, y0 = center_x - card_w / 2, center_y - card_h / 2
    corners = np.float32([[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]])
    dst = np.float32([[x0, y0], [x0 + card_w, y0], [x0 + card_w, y0 + card_h], [x0, y0 + card_h]])
    jitter = perspective * size
    # A minority of views clip one edge; the capture never deliberately hides
    # most of the label.
    if rng.random() < (0.12 if bucket != "clean-ish" else 0.04):
        shift_axis = rng.randrange(4)
        dst[shift_axis, 0] += rng.choice((-1, 1)) * size * rng.uniform(0.01, 0.07)
        dst[shift_axis, 1] += rng.choice((-1, 1)) * size * rng.uniform(0.01, 0.07)
    if bucket in {"perspective", "background_scale", "glare", "blur"}:
        dst += np.asarray([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(source, transform, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((card_h, card_w), 255, dtype=np.uint8), transform, (size, size),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255.0
    shadow = cv2.GaussianBlur(np.roll(mask, shift=(max(1, size // 100), max(1, size // 120)), axis=(0, 1)), (0, 0), size / 55)
    canvas *= (1.0 - 0.30 * shadow[:, :, None])
    alpha = np.clip(mask[:, :, None], 0.0, 1.0)
    canvas = canvas * (1.0 - alpha) + warped.astype(np.float32) * alpha
    result = canvas

    # Capture lighting: white balance, exposure and broad uneven illumination.
    contrast = rng.uniform(0.88, 1.12)
    brightness = rng.uniform(-12, 12)
    temp = rng.uniform(-0.07, 0.07)
    gains = np.asarray([1.0 + temp, 1.0, 1.0 - temp], dtype=np.float32)
    result = ((result - 127.5) * contrast + 127.5 + brightness) * gains[None, None, :]
    if bucket != "clean-ish" or rng.random() < 0.40:
        x_light = np.linspace(rng.uniform(0.90, 1.0), rng.uniform(1.0, 1.11), size, dtype=np.float32)
        y_light = np.linspace(rng.uniform(0.92, 1.0), rng.uniform(1.0, 1.08), size, dtype=np.float32)
        result *= (y_light[:, None] * x_light[None, :])[:, :, None]

    if bucket == "glare" or rng.random() < 0.14:
        glare = np.zeros((size, size), dtype=np.float32)
        gx, gy = rng.randrange(size), rng.randrange(size)
        axes = (max(8, round(size * rng.uniform(0.035, 0.09))), max(10, round(size * rng.uniform(0.06, 0.17))))
        cv2.ellipse(glare, (gx, gy), axes, rng.uniform(-45, 45), 0, 360, rng.uniform(0.10, 0.26), -1)
        glare = cv2.GaussianBlur(glare, (0, 0), size / 36)
        result = result * (1.0 - glare[:, :, None]) + 255.0 * glare[:, :, None]

    if bucket in {"blur", "perspective"} or rng.random() < 0.22:
        factor = rng.uniform(0.56, 0.90) if bucket == "blur" else rng.uniform(0.78, 0.96)
        small = cv2.resize(np.clip(result, 0, 255).astype(np.uint8),
                           (max(24, round(size * factor)), max(24, round(size * factor))), interpolation=cv2.INTER_AREA)
        result = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if bucket == "blur" or rng.random() < 0.16:
        if rng.random() < 0.68:
            k = rng.choice((3, 5))
            result = cv2.GaussianBlur(result, (k, k), rng.uniform(0.25, 0.9))
        else:
            kernel = np.zeros((3, 3), dtype=np.float32)
            kernel[1, :] = (0.18, 0.64, 0.18)
            result = cv2.filter2D(result, -1, kernel)
    if bucket != "clean-ish" and rng.random() < 0.42:
        noise = np_rng.normal(0.0, rng.uniform(1.0, 3.2), size=(size, size, 3)).astype(np.float32)
        result += noise
    result = np.clip(result, 0, 255).astype(np.uint8)

    if bucket != "clean-ish" and rng.random() < 0.68:
        quality_low, quality_high = ((54, 88) if bucket == "blur" else (62, 94))
        buffer = io.BytesIO()
        Image.fromarray(result, "RGB").save(buffer, format="JPEG", quality=rng.randint(quality_low, quality_high))
        buffer.seek(0)
        with Image.open(buffer) as jpeg:
            result = np.asarray(jpeg.convert("RGB"), dtype=np.uint8).copy()
    return Image.fromarray(result, mode="RGB")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA configuration")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.enabled = True
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        if not self.enabled:
            return base_output
        low_rank = F.linear(F.linear(self.dropout(inputs).float(), self.lora_A), self.lora_B)
        return base_output + (self.scale * low_rank).to(dtype=base_output.dtype)


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def inject_last_vision_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    last_blocks: int = 4,
    target_suffixes: Sequence[str] = LORA_TARGETS,
) -> list[str]:
    """Attach LoRA only to the named attention linears in the last blocks."""
    vision = getattr(model, "vision_model", None)
    encoder = getattr(vision, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if vision is None or layers is None or len(layers) < last_blocks:
        raise ValueError("expected a vision_model.encoder.layers module layout")
    if tuple(target_suffixes) != LORA_TARGETS:
        raise ValueError("only Q/K/V/attention-output projection targets are supported")
    model.requires_grad_(False)
    first_block = len(layers) - last_blocks
    intended: list[tuple[nn.Module, str, str]] = []
    for block_index in range(first_block, len(layers)):
        block = layers[block_index]
        for suffix in LORA_TARGETS:
            matches = [(name, module) for name, module in block.named_modules()
                       if name.split(".")[-1] == suffix and isinstance(module, nn.Linear)]
            if len(matches) != 1:
                raise ValueError(f"block {block_index} must expose exactly one Linear for {suffix}; found {len(matches)}")
            relative, module = matches[0]
            parent_path, child_name = relative.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            intended.append((parent, child_name, f"vision_model.encoder.layers.{block_index}.{relative}"))
    names = []
    for parent, child_name, full_name in intended:
        original = getattr(parent, child_name)
        _set_child(parent, child_name, LoRALinear(original, rank, alpha, dropout))
        names.append(full_name)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".lora_A" in name or ".lora_B" in name)
    if not names or any(name.startswith("text_model.") for name in names):
        raise AssertionError("LoRA targets must be vision-only")
    return names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if ".lora_A" in name or ".lora_B" in name}


def load_lora_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = lora_state_dict(model)
    if set(current) != set(state):
        raise ValueError("LoRA checkpoint keys do not match the injected model")
    with torch.no_grad():
        model_state = model.state_dict()
        for name, tensor in state.items():
            model_state[name].copy_(tensor.to(device=model_state[name].device, dtype=model_state[name].dtype))


def lora_sha256(model_or_state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    state = lora_state_dict(model_or_state) if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = bool(enabled)


def build_hard_negative_map(
    slugs: Sequence[str],
    frozen_reference_embeddings: np.ndarray,
    family_keys: Mapping[str, str | None],
    seed: int,
    family_count: int = 4,
    visual_count: int = 6,
    random_count: int = 2,
    neighbor_pool: int = 20,
) -> tuple[dict[str, dict[str, list[int]]], list[dict[str, Any]]]:
    """Build negatives exclusively from reference vectors and catalog family keys."""
    vectors = np.asarray(frozen_reference_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(slugs) or len(set(slugs)) != len(slugs):
        raise ValueError("frozen reference embeddings must align with unique catalog slugs")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarity = vectors @ vectors.T
    idx_by_slug = {slug: index for index, slug in enumerate(slugs)}
    family_members: dict[str, list[int]] = defaultdict(list)
    for index, slug in enumerate(slugs):
        key = family_keys.get(slug)
        if key:
            family_members[key].append(index)
    result: dict[str, dict[str, list[int]]] = {}
    rows: list[dict[str, Any]] = []
    for anchor_index, slug in enumerate(slugs):
        key = family_keys.get(slug)
        family = [i for i in family_members.get(key, []) if i != anchor_index]
        family.sort(key=lambda i: (-float(similarity[anchor_index, i]), slugs[i]))
        chosen_family = family[:family_count]
        top_neighbors = sorted((i for i in range(len(slugs)) if i != anchor_index),
                               key=lambda i: (-float(similarity[anchor_index, i]), slugs[i]))[:neighbor_pool]
        visual = [i for i in top_neighbors if i not in chosen_family and i != anchor_index][:visual_count]
        if len(visual) < visual_count:
            visual += [i for i in sorted((j for j in range(len(slugs)) if j != anchor_index),
                                         key=lambda j: (-float(similarity[anchor_index, j]), slugs[j]))
                       if i not in chosen_family and i not in visual][:visual_count - len(visual)]
        excluded = {anchor_index, *chosen_family, *visual}
        random_pool = [i for i in range(len(slugs)) if i not in excluded
                       and (not key or family_keys.get(slugs[i]) != key)]
        local_rng = random.Random(derive_seed(seed, slug, "train", 1_000_003))
        random_negatives = local_rng.sample(random_pool, k=min(random_count, len(random_pool)))
        selected = list(dict.fromkeys([*chosen_family, *visual, *random_negatives]))
        for i in top_neighbors:
            if len(selected) >= len(chosen_family) + visual_count + random_count:
                break
            if i not in selected:
                selected.append(i)
        if len(selected) != len(chosen_family) + visual_count + random_count:
            raise ValueError(f"catalog too small to build negatives for {slug}")
        if anchor_index in selected or len(set(selected)) != len(selected):
            raise AssertionError(f"self/duplicate negative selected for {slug}")
        result[slug] = {"same_family": chosen_family, "visual": visual, "random": random_negatives,
                        "all": selected}
        source_by_idx = {i: "catalog_metadata_same_family" for i in chosen_family}
        source_by_idx.update({i: "frozen_so400m_reference_top20" for i in visual})
        source_by_idx.update({i: "uniform_catalog_random" for i in random_negatives})
        for rank, index in enumerate(selected, 1):
            rows.append({"anchor_slug": slug, "negative_slug": slugs[index], "negative_rank": rank,
                         "negative_source": source_by_idx.get(index, "frozen_so400m_reference_top20_fallback"),
                         "catalog_family_key": key or "",
                         "frozen_reference_cosine": float(similarity[anchor_index, index])})
    return result, rows


def adapted_reference_fingerprint(
    base_fingerprint: str,
    checkpoint_sha256: str,
    slugs: Sequence[str],
    image_hashes: Mapping[str, str],
    query_batch_size: int = 1,
) -> str:
    if isinstance(query_batch_size, bool) or int(query_batch_size) != 1:
        raise ValueError("canonical LoRA cache identity requires query_batch_size=1")
    payload = {
        "base_reference_fingerprint": base_fingerprint,
        "lora_checkpoint_sha256": checkpoint_sha256,
        "query_batch_size": 1,
        "references": sorted((slug, image_hashes[slug]) for slug in slugs),
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def assert_same_lora_checkpoint(query_checkpoint_sha256: str, reference_checkpoint_sha256: str) -> None:
    if not query_checkpoint_sha256 or query_checkpoint_sha256 != reference_checkpoint_sha256:
        raise ValueError("query and reference embeddings must use the same selected LoRA checkpoint")


def pooled_image_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return the same 1152-d pooled SigLIP image representation as the app."""
    if hasattr(model, "get_image_features"):
        features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        projection = getattr(model, "visual_projection", None)
        if projection is not None and features.shape[-1] == projection.in_features:
            features = projection(features)
    else:
        output = model.vision_model(pixel_values=pixel_values, return_dict=True)
        features = getattr(output, "pooler_output", None)
        if features is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise TypeError("SigLIP vision output has neither pooler_output nor last_hidden_state")
            features = hidden[:, 0]
    if features.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE}-dimensional image features, got {features.shape[-1]}")
    return F.normalize(features.float(), p=2, dim=-1, eps=1e-12)


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def catalog_family_keys(catalog_rows: Sequence[Mapping[str, str]]) -> dict[str, str | None]:
    """Catalog-only conservative product-line keys, shared with Phase A."""
    token = lambda value: re.findall(r"[a-zа-яё0-9]+", (value or "").casefold().replace("ё", "е"))
    generic = set(token("wine vino вино белое красное розовое оранжевое сухое полусухое полусладкое сладкое брют экстра игристое тихое резерв резервное купаж сорт виноград выдержанное молодое бутылка"))
    grapes = {part for row in catalog_rows for part in token(row.get("grape", ""))}
    result: dict[str, str | None] = {}
    for row in catalog_rows:
        slug = row.get("slug", "")
        winery_tokens = set(token(row.get("winery", "")))
        core = sorted({part for part in token(row.get("title", ""))
                       if len(part) >= 3 and not part.isdigit() and part not in generic and part not in grapes
                       and part not in winery_tokens and not re.fullmatch(r"(?:19|20)\d{2}", part)})
        winery = " ".join(token(row.get("winery", "")))
        result[slug] = f"{winery}|{' '.join(core)}" if winery and len(core) >= 2 else None
    return result


def infer_catalog_family_types(
    catalog_rows: Sequence[Mapping[str, str]],
    family_keys: Mapping[str, str | None],
) -> dict[str, str]:
    """Classify catalog-only proxy families as vintage, subtype, or other."""
    members: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in catalog_rows:
        key = family_keys.get(row.get("slug", ""))
        if key:
            members[key].append(row)
    family_types: dict[str, str] = {}
    for key, rows in members.items():
        year_values = set()
        for row in rows:
            year_values.update(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", row.get("title", "")))
        if len(year_values) >= 2:
            tag = "vintage"
        elif len({(row.get("grape", "").casefold(), row.get("category", "").casefold()) for row in rows}) >= 2:
            tag = "subtype"
        else:
            tag = "other"
        for row in rows:
            family_types[row["slug"]] = tag
    return family_types
