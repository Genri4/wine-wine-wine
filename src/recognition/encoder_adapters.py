"""Image encoder adapters for the model bake-off.

Every adapter wraps one specific pretrained checkpoint behind a single
interface: deterministic preprocessing of its own, one documented global
image representation, L2-normalized outputs, and a validated per-model
reference embedding cache. Models run sequentially; each adapter fully
releases GPU memory via ``release()``.

Retrieval protocol is identical for all models: cosine similarity over
L2-normalized global embeddings, single full image per query, no crops,
no gates, no OCR, no reranking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .preprocessing import load_rgb_image


class ImageEncoderAdapter:
    """Common interface for one frozen pretrained image encoder."""

    model_key: str = ""
    hf_model_id: str = ""
    checkpoint_revision: str = ""
    embedding_dim: int = 0
    native_resolution: str = ""
    global_representation: str = ""
    dtype: str = "float32"
    device: str = "cpu"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size  # None -> adapter default
        self.preprocessing_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        """Encode PIL images in batches; returns L2-normalized vectors.

        Default routing goes through :meth:`_batched` and the adapter's
        ``_encode_batch``; adapters with a different native batching path
        (e.g. the frozen SigLIP2 baseline wrapper) override this method.
        """

        return self._batched(images)

    def encode_paths(self, paths: Sequence[str | Path]) -> list[list[float]]:
        return self.encode_images([load_rgb_image(path) for path in paths])

    def release(self) -> None:
        """Drop model weights and free GPU memory."""

        for attribute in ("_model", "_processor", "_transform", "_preprocess"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
        gc_collect_and_empty_cache(self.device)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "hf_model_id": self.hf_model_id,
            "checkpoint_revision": self.checkpoint_revision,
            "embedding_dim": self.embedding_dim,
            "native_resolution": self.native_resolution,
            "global_representation": self.global_representation,
            "dtype": self.dtype,
            "device": self.device,
            "preprocessing_config": self.preprocessing_config,
        }

    def fingerprint(self, catalog_manifest_sha256: str, reference_count: int) -> str:
        payload = json.dumps(
            {
                "model_key": self.model_key,
                "hf_model_id": self.hf_model_id,
                "checkpoint_revision": self.checkpoint_revision,
                "embedding_dim": self.embedding_dim,
                "dtype": self.dtype,
                "preprocessing_config": self.preprocessing_config,
                "catalog_manifest_sha256": catalog_manifest_sha256,
                "reference_count": reference_count,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _batched(self, images: Sequence[Any]) -> list[list[float]]:
        batch_size = self.batch_size or 16
        embeddings: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                embeddings.extend(self._encode_batch(batch))
        return embeddings

    def _encode_batch(self, images: Sequence[Any]) -> list[list[float]]:
        raise NotImplementedError

    def _normalized(self, tensor: torch.Tensor) -> list[list[float]]:
        tensor = torch.nn.functional.normalize(tensor.float(), p=2, dim=1)
        return tensor.detach().cpu().tolist()


def gc_collect_and_empty_cache(device: str) -> None:
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


class CurrentSigLIP2Adapter(ImageEncoderAdapter):
    """The frozen baseline: google/siglip2-base-patch16-224 via the existing
    VisualEncoder path, fp32, to guarantee exact baseline reproduction."""

    model_key = "current_siglip2"
    hf_model_id = "google/siglip2-base-patch16-224"
    embedding_dim = 768
    native_resolution = "224x224 (anisotropic stretch, no center crop)"
    global_representation = "get_image_features pooled output + visual projection"
    dtype = "float32"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(device, batch_size)
        import sys

        project_root = Path(__file__).resolve().parents[2]
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))
        from recognition.encoder import VisualEncoder

        self._encoder = VisualEncoder(model_name="siglip", device=self.device, batch_size=batch_size or 32)
        self.device = self._encoder.device
        self.checkpoint_revision = self._encoder.model_version
        self.preprocessing_config = self._encoder.preprocessing_config

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        return self._encoder.encode_pil(list(images))

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "parameter_count": 375_187_970,
            "vision_parameter_count": 92_884_224,
            "note": "fp32 and identical preprocessing to the frozen baseline for exact reproduction",
        }


class SigLIP2So400m384Adapter(ImageEncoderAdapter):
    """google/siglip2-so400m-patch14-384 via transformers, fp16 on CUDA."""

    model_key = "siglip2_so400m_384"
    hf_model_id = "google/siglip2-so400m-patch14-384"
    embedding_dim = 1152
    native_resolution = "384x384 (anisotropic stretch, no center crop)"
    global_representation = "get_image_features pooled output + visual projection"
    dtype = "float16"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(device, batch_size)
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        self._model = AutoModel.from_pretrained(
            self.hf_model_id, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.checkpoint_revision = self._model.config._commit_hash
        ip = getattr(self._processor, "image_processor", self._processor)
        self.preprocessing_config = {
            "class": type(ip).__name__,
            "size": str(getattr(ip, "size", None)),
            "resample": str(getattr(ip, "resample", None)),
            "image_mean": str(getattr(ip, "image_mean", None)),
            "image_std": str(getattr(ip, "image_std", None)),
            "do_center_crop": str(getattr(ip, "do_center_crop", None)),
        }

    def _encode_batch(self, images: Sequence[Any]) -> list[list[float]]:
        dtype = next(self._model.parameters()).dtype
        pixel_values = self._processor(images=list(images), return_tensors="pt")["pixel_values"].to(
            self.device, dtype
        )
        with torch.inference_mode():
            output = self._model.get_image_features(pixel_values=pixel_values)
            if hasattr(output, "pooler_output"):
                output = output.pooler_output
            projection = getattr(self._model, "visual_projection", None)
            if projection is not None and output.shape[-1] == projection.in_features:
                output = projection(output)
        return self._normalized(output)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "parameter_count": sum(p.numel() for p in self._model.parameters()),
        }


class DinoV2RegistersLargeAdapter(ImageEncoderAdapter):
    """facebook/dinov2-with-registers-large via transformers; global
    representation is the official CLS token (last_hidden_state[:, 0]).
    The HF pooler is not used because it is not part of the DINOv2
    backbone weights."""

    model_key = "dinov2_vitl14_reg"
    hf_model_id = "facebook/dinov2-with-registers-large"
    embedding_dim = 1024
    native_resolution = "224x224 (shortest edge 256 resize + center crop 224)"
    global_representation = "CLS token last_hidden_state[:,0]"
    dtype = "float16"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(device, batch_size)
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        self._model = AutoModel.from_pretrained(
            self.hf_model_id, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.checkpoint_revision = self._model.config._commit_hash
        ip = getattr(self._processor, "image_processor", self._processor)
        self.preprocessing_config = {
            "class": type(ip).__name__,
            "size": str(getattr(ip, "size", None)),
            "crop_size": str(getattr(ip, "crop_size", None)),
            "do_resize": str(getattr(ip, "do_resize", None)),
            "do_center_crop": str(getattr(ip, "do_center_crop", None)),
            "image_mean": str(getattr(ip, "image_mean", None)),
            "image_std": str(getattr(ip, "image_std", None)),
            "resample": str(getattr(ip, "resample", None)),
        }

    def _encode_batch(self, images: Sequence[Any]) -> list[list[float]]:
        dtype = next(self._model.parameters()).dtype
        pixel_values = self._processor(images=list(images), return_tensors="pt")["pixel_values"].to(
            self.device, dtype
        )
        with torch.inference_mode():
            output = self._model(pixel_values=pixel_values)
            cls = output.last_hidden_state[:, 0]
        return self._normalized(cls)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "parameter_count": sum(p.numel() for p in self._model.parameters()),
            "pooler_used": False,
            "pooler_note": "HF pooler is not part of DINOv2 backbone weights; CLS token is the documented global representation",
        }


class OpenClipAdapter(ImageEncoderAdapter):
    """open_clip based encoder (PE-Core, DFN5B) loaded from a local fp16
    checkpoint converted once from the official fp32 release to keep RAM
    peaks low on the 16 GB host."""

    def __init__(
        self,
        model_key: str,
        hf_model_id: str,
        architecture: str,
        embedding_dim: int,
        native_resolution: str,
        global_representation: str,
        local_weights_path: str,
        preprocess_cfg: dict[str, Any],
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(device, batch_size)
        import open_clip

        self.model_key = model_key
        self.hf_model_id = hf_model_id
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.native_resolution = native_resolution
        self.global_representation = global_representation
        self.dtype = "float16"
        self._open_clip = open_clip
        self._model, _, _ = open_clip.create_model_and_transforms(architecture)
        open_clip.load_checkpoint(self._model, local_weights_path)
        self._model = self._model.to(self.device).eval()
        if self.device == "cuda":
            self._model = self._model.half()
        self._preprocess = open_clip.transform.image_transform_v2(
            open_clip.transform.PreprocessCfg(**preprocess_cfg), is_train=False
        )
        self.preprocessing_config = {"architecture": architecture, **{k: str(v) for k, v in preprocess_cfg.items()}}
        self.parameter_count = sum(p.numel() for p in self._model.parameters())

    def _encode_batch(self, images: Sequence[Any]) -> list[list[float]]:
        dtype = next(self._model.parameters()).dtype
        tensor = torch.stack([self._preprocess(image) for image in images]).to(self.device, dtype)
        with torch.inference_mode():
            output = self._model.encode_image(tensor)
        return self._normalized(output)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "parameter_count": self.parameter_count,
            "local_weights": "official fp32 release converted once to fp16 (conversion logged)",
        }


class PECoreL14_336Adapter(ImageEncoderAdapter):
    """PE-Core-L/14-336 (Meta Perception Encoder) loaded through the official
    open_clip pretrained tag ``PE-Core-L-14-336:meta``, which applies the
    checkpoint key conversion from the original PE release format."""

    model_key = "pe_core_l14_336"
    hf_model_id = "facebook/PE-Core-L14-336"
    embedding_dim = 1024
    native_resolution = "336x336 (squash resize, bilinear)"
    global_representation = "open_clip encode_image (attention-pooled vision tower output)"
    dtype = "float16"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(device, batch_size)
        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "PE-Core-L-14-336", pretrained="meta"
        )
        self._model = self._model.to(self.device).eval()
        if self.device == "cuda":
            self._model = self._model.half()
        self._open_clip = open_clip
        self.preprocessing_config = {
            "architecture": "PE-Core-L-14-336",
            "pretrained_tag": "meta",
            "size": "336",
            "interpolation": "bilinear",
            "resize_mode": "squash",
            "mean": "(0.5, 0.5, 0.5)",
            "std": "(0.5, 0.5, 0.5)",
        }
        self.parameter_count = sum(p.numel() for p in self._model.parameters())

    def _encode_batch(self, images: Sequence[Any]) -> list[list[float]]:
        dtype = next(self._model.parameters()).dtype
        tensor = torch.stack([self._preprocess(image) for image in images]).to(self.device, dtype)
        with torch.inference_mode():
            output = self._model.encode_image(tensor)
        return self._normalized(output)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "parameter_count": self.parameter_count,
            "local_weights": "official 'meta' pretrained tag (key conversion handled by open_clip)",
        }


class DFN5BAdapter(OpenClipAdapter):
    """DFN5B CLIP ViT-H/14 at 378px (Apple), official OpenCLIP-supported
    checkpoint apple/dfn5b-clip-vit-h-14-378 converted to fp16."""

    model_key = "dfn5b_h14_378"
    hf_model_id = "apple/dfn5b-clip-vit-h-14-378"
    embedding_dim = 1024
    native_resolution = "378x378 (squash resize, bicubic)"
    global_representation = "open_clip encode_image (pooled vision tower output)"

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        super().__init__(
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            architecture="ViT-H-14-378",
            embedding_dim=self.embedding_dim,
            native_resolution=self.native_resolution,
            global_representation=self.global_representation,
            local_weights_path="artifacts/reference_embeddings/_converted/dfn5b_h14_378_fp16.pt",
            preprocess_cfg={
                "mean": (0.48145466, 0.4578275, 0.40821073),
                "std": (0.26862954, 0.26130258, 0.27577711),
                "size": 378,
                "interpolation": "bicubic",
                "resize_mode": "squash",
            },
            device=device,
            batch_size=batch_size,
        )


ADAPTER_REGISTRY: dict[str, type[ImageEncoderAdapter]] = {
    CurrentSigLIP2Adapter.model_key: CurrentSigLIP2Adapter,
    SigLIP2So400m384Adapter.model_key: SigLIP2So400m384Adapter,
    DinoV2RegistersLargeAdapter.model_key: DinoV2RegistersLargeAdapter,
    PECoreL14_336Adapter.model_key: PECoreL14_336Adapter,
    DFN5BAdapter.model_key: DFN5BAdapter,
}


def create_adapter(model_key: str, device: str | None = None, batch_size: int | None = None) -> ImageEncoderAdapter:
    if model_key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {sorted(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[model_key](device=device, batch_size=batch_size)


# ----------------------------------------------------------------------
# Per-model reference embedding cache with fingerprint validation
# ----------------------------------------------------------------------

REFERENCE_CACHE_VERSION = 1


def reference_cache_dir(root: str | Path, model_key: str) -> Path:
    return Path(root) / "artifacts" / "reference_embeddings" / model_key


def save_reference_cache(
    cache_dir: str | Path,
    embeddings: Sequence[Sequence[float]],
    slugs: Sequence[str],
    adapter_metadata: dict[str, Any],
    fingerprint: str,
) -> None:
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": REFERENCE_CACHE_VERSION, "fingerprint": fingerprint, "embeddings": torch.tensor(list(embeddings), dtype=torch.float32)}, output_dir / "embeddings.pt")
    (output_dir / "slugs.json").write_text(json.dumps(list(slugs), ensure_ascii=False, indent=0), encoding="utf-8")
    metadata = {**adapter_metadata, "fingerprint": fingerprint, "reference_count": len(slugs)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


class ReferenceCacheMismatch(ValueError):
    """Raised when a reference cache belongs to another model/config."""


def load_reference_cache(
    cache_dir: str | Path,
    expected_fingerprint: str,
    expected_slugs: Sequence[str],
) -> list[list[float]]:
    cache_dir = Path(cache_dir)
    embeddings_path = cache_dir / "embeddings.pt"
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != REFERENCE_CACHE_VERSION:
        raise ReferenceCacheMismatch("reference cache format version differs")
    if payload.get("fingerprint") != expected_fingerprint:
        raise ReferenceCacheMismatch(
            "reference cache fingerprint differs (another model/config/catalog)"
        )
    stored_slugs = json.loads((cache_dir / "slugs.json").read_text(encoding="utf-8"))
    if stored_slugs != list(expected_slugs):
        raise ReferenceCacheMismatch("reference cache slug order differs from the canonical catalog")
    embeddings = payload.get("embeddings")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_slugs):
        raise ReferenceCacheMismatch("reference cache embeddings have an invalid shape")
    return embeddings.tolist()
