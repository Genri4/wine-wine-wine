"""Pretrained visual encoders used by the baseline comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .preprocessing import load_rgb_image, preprocess_image


SUPPORTED_MODEL_NAMES = ("resnet18", "resnet50", "dinov2", "siglip")

HUGGINGFACE_MODEL_IDS = {
    "dinov2": "facebook/dinov2-base",
    # Keep the short local model name for CLI compatibility, while selecting
    # the requested SigLIP 2 checkpoint.
    "siglip": "google/siglip2-base-patch16-224",
}


class VisualEncoder:
    """Extract L2-normalized image embeddings from one selected model.

    The model is selected with one ``model_name`` value. ResNet models use
    torchvision weights; DINOv2 and SigLIP use their Hugging Face pretrained
    checkpoints and model-specific image processors.
    """

    _SUPPORTED_MODELS = frozenset(SUPPORTED_MODEL_NAMES)

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if model_name not in self._SUPPORTED_MODELS:
            supported = ", ".join(SUPPORTED_MODEL_NAMES)
            raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for the pretrained encoder; install project dependencies first"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_name
        self.model_version = "unknown"
        self.preprocessing_config: dict[str, object] = {}

        if model_name in {"resnet18", "resnet50"}:
            self._init_torchvision(model_name)
        else:
            self._init_huggingface(model_name)

    def _init_torchvision(self, model_name: str) -> None:
        try:
            from torchvision import models
        except Exception as exc:
            raise RuntimeError(
                "torchvision is required for ResNet encoders; install project dependencies first"
            ) from exc

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)

        embedding_dim = model.fc.in_features
        # The classifier is not used; the penultimate representation is the embedding.
        model.fc = self._torch.nn.Identity()
        model.eval().to(self.device)

        self.encoder_name = f"torchvision/{model_name}@DEFAULT"
        self.model_id = self.encoder_name
        self.model_version = "DEFAULT"
        self._model = model
        self._transform = weights.transforms()
        self.preprocessing_config = {
            "kind": "torchvision_weights_transform",
            "weights": self.encoder_name,
        }
        self.embedding_dim = embedding_dim

    def _init_huggingface(self, model_name: str) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for DINOv2 and SigLIP encoders; "
                "install project dependencies first"
            ) from exc

        model_id = HUGGINGFACE_MODEL_IDS[model_name]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval().to(self.device)

        self.encoder_name = model_id
        self.model_id = model_id
        self.model_version = _model_version(model)
        self._model = model
        self._processor = processor
        self._transform = self._transform_huggingface
        self.preprocessing_config = _processor_config(processor)
        self.embedding_dim = self._huggingface_embedding_dim(model, model_name)

    def _transform_huggingface(self, image: Any) -> Any:
        processed = self._processor(images=image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)

    def _huggingface_embedding_dim(self, model: Any, model_name: str) -> int:
        if model_name == "dinov2":
            return int(model.config.hidden_size)

        projection = getattr(model, "visual_projection", None)
        if projection is not None and hasattr(projection, "out_features"):
            return int(projection.out_features)
        projection_dim = getattr(model.config, "projection_dim", None)
        if projection_dim is not None:
            return int(projection_dim)
        vision_config = getattr(model.config, "vision_config", None)
        if vision_config is not None and hasattr(vision_config, "hidden_size"):
            return int(vision_config.hidden_size)
        return int(model.config.hidden_size)

    def _forward(self, batch: Any) -> Any:
        if self.model_name == "dinov2":
            outputs = self._model(pixel_values=batch)
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is not None:
                return pooler_output
            return outputs.last_hidden_state[:, 0]

        if self.model_name == "siglip":
            # get_image_features has returned either a tensor or a vision output
            # across Transformers versions. Handle both without using text inputs.
            if hasattr(self._model, "get_image_features"):
                output = self._model.get_image_features(pixel_values=batch)
            else:
                output = self._model.vision_model(pixel_values=batch)
            if hasattr(output, "pooler_output"):
                output = output.pooler_output

            # Older SigLIP versions expose a projection separately. Apply it only
            # when the returned vector is still in the projection input space.
            projection = getattr(self._model, "visual_projection", None)
            if projection is not None and output.shape[-1] == projection.in_features:
                output = projection(output)
            return output

        return self._model(batch)

    def encode(self, image_paths: Sequence[str | Path]) -> list[list[float]]:
        """Encode image paths in batches and return L2-normalized vectors."""

        paths = [Path(path) for path in image_paths]
        if not paths:
            return []
        images = [load_rgb_image(path) for path in paths]
        return self.encode_pil(images)

    def encode_pil(self, images: Sequence[Any]) -> list[list[float]]:
        """Encode already-loaded images with the same pipeline as ``encode``."""

        if not images:
            return []

        torch = self._torch
        embeddings: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            batch = torch.stack(
                [self._transform(image) for image in batch_images]
            ).to(self.device)
            with torch.inference_mode():
                output = self._forward(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.flatten(start_dim=1)
                output = torch.nn.functional.normalize(output, p=2, dim=1)
            embeddings.extend(output.detach().cpu().tolist())
        return embeddings


def _model_version(model: Any) -> str:
    """Return a stable checkpoint/config marker for cache validation."""

    commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    config = getattr(model, "config", None)
    if config is None:
        return "pretrained-default"
    try:
        serialized = json.dumps(config.to_dict(), sort_keys=True, default=str)
    except Exception:
        serialized = repr(config)
    return "config-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _processor_config(processor: Any) -> dict[str, object]:
    """Keep only the relevant, JSON-serializable preprocessing contract."""

    image_processor = getattr(processor, "image_processor", processor)
    config: dict[str, object] = {"class": type(image_processor).__name__}
    for name in (
        "do_resize",
        "size",
        "crop_size",
        "do_center_crop",
        "do_normalize",
        "image_mean",
        "image_std",
        "resample",
    ):
        value = getattr(image_processor, name, None)
        if value is not None:
            config[name] = value
    try:
        json.dumps(config)
    except TypeError:
        config = {key: str(value) for key, value in config.items()}
    return config


class TorchvisionEncoder(VisualEncoder):
    """Backward-compatible name for the original ResNet-only encoder."""

    _SUPPORTED_MODELS = frozenset({"resnet18", "resnet50"})

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name=model_name, device=device, batch_size=batch_size)
