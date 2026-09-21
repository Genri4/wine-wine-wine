"""Local OCR engine boundary for the OCR reranking milestone.

Engine decision (documented per the milestone brief, Part A):

- Engine: PaddleOCR 3.7.0 (paddlex 3.7.2), fully local, no cloud APIs.
- Runtime: paddlepaddle-gpu 3.3.1 (official cu126 wheel). CPU mode also works
  but was ~6x slower; the CPU run needed ``enable_mkldnn=False`` because
  paddle 3.3.1 crashed with a PIR/oneDNN conversion error otherwise.
- Detection: PP-OCRv5_server_det (best available quality).
- Recognition: eslav_PP-OCRv5_mobile_rec — the East-Slavic Cyrillic model
  PaddleOCR itself selects for ``lang="ru"``. The server eslav variant is not
  registered in this version, and the default multilingual rec model reads
  Cyrillic poorly, so the model names are pinned explicitly.
- Languages: Russian Cyrillic + Latin/digits. PaddleOCR was preferred over
  EasyOCR because it installed cleanly and reads the required Cyrillic
  labels; no second engine was needed, so no engine bake-off was run.
- Settings: doc orientation/unwarping/textline orientation disabled (labels
  are already upright in all frozen benchmarks), images passed by path.

Only ``(text, confidence)`` lines plus boxes leave this module; the reranker
never sees engine internals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "eslav_PP-OCRv5_mobile_rec"

# Deterministic cache-directory name for artifacts/ocr_cache/<ocr_model>/.
OCR_MODEL_KEY = "paddleocr3.7_ppocrv5_server_det_eslav_v5_mobile_rec"


@dataclass(frozen=True)
class OcrLine:
    """One recognized text box."""

    text: str
    confidence: float
    box: tuple[tuple[float, float], ...] | None = None


def engine_metadata(device: str) -> dict[str, Any]:
    """Report the exact OCR engine configuration for artifacts/reports."""
    import paddle
    import paddleocr
    import paddlex

    return {
        "engine": "paddleocr",
        "paddleocr_version": paddleocr.__version__,
        "paddlex_version": paddlex.__version__,
        "paddle_version": paddle.__version__,
        "paddle_build": "paddlepaddle-gpu (official cu126 wheel)",
        "detection_model": DETECTION_MODEL,
        "recognition_model": RECOGNITION_MODEL,
        "languages": "ru (East-Slavic Cyrillic) + Latin/digits",
        "device": device,
        "inference_settings": {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        },
        "model_source": "PaddleOCR official model hoster, auto-download at first init",
        "cloud_apis_used": False,
    }


class WineLabelOcr:
    """Thin deterministic wrapper around one PaddleOCR pipeline instance."""

    def __init__(self, device: str = "gpu:0") -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self.device = device
        self._pipeline = PaddleOCR(
            text_detection_model_name=DETECTION_MODEL,
            text_recognition_model_name=RECOGNITION_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            enable_mkldnn=False,
        )

    def predict_lines(self, image_path: str | Path) -> list[OcrLine]:
        """Run detection+recognition on one image file."""
        result = self._pipeline.predict(str(image_path))
        lines: list[OcrLine] = []
        for page in result:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", page.get("dt_polys", []))
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                lines.append(
                    OcrLine(text=str(text), confidence=score, box=_normalize_box(boxes[index] if index < len(boxes) else None))
                )
        return lines


def _normalize_box(raw: Any) -> tuple[tuple[float, float], ...] | None:
    """Convert a detector box (flat [x1, y1, x2, y2] or polygon) to tuples."""
    if raw is None:
        return None
    try:
        shape = getattr(raw, "shape", None)
        if shape is not None and len(shape) == 1:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return tuple(tuple(float(value) for value in point) for point in raw)
    except (TypeError, ValueError, IndexError):
        return None
