"""VLM year-judge ceiling (diagnostic only): Qwen2-VL-2B-Instruct.

Given a year crop and the allowed candidate years, the model must answer
with exactly one of them. No target identity or ground truth enters the
prompt. Runs locally on the available GPU; latency is recorded but this
path is never a production integration in this milestone.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


class VlmYearJudge:
    def __init__(self, device: str = "cuda", max_new_tokens: int = 8) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, dtype=dtype
        ).to(device).eval()
        self.max_new_tokens = max_new_tokens

    def choose_year(self, image_path: str, allowed_years: Sequence[str]) -> str | None:
        from PIL import Image

        years = list(allowed_years)
        options = "\n".join(f"- {year}" for year in years)
        prompt = (
            "Which year is printed in this image? "
            f"Choose exactly one of:\n{options}\nAnswer with the year only."
        )
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        dtype = next(self.model.parameters()).dtype
        inputs = {key: value.to(dtype) if value.dtype.is_floating_point else value for key, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_text = self.processor.batch_decode(
            generated[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]
        return self._extract_year(output_text, years)

    @staticmethod
    def _extract_year(text: str, allowed_years: Sequence[str]) -> str | None:
        for year in allowed_years:
            if year in text:
                return year
        match = re.search(r"(19|20)\d{2}", text)
        return match.group(0) if match else None

    def release(self) -> None:
        import gc

        self.model = None
        self.processor = None
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
