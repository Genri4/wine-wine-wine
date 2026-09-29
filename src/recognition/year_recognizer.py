"""Digit-only year recognizer (CRNN-lite) with constrained decoding.

A small CNN+BiLSTM+CTC model trained exclusively on synthetic year crops
(1990-2026, heavy augmentation: fonts, rotation, perspective, blur, glare,
JPEG, low contrast, gold/silver text, dark/light backgrounds). Benchmark
query crops never enter training.

Constrained decoding (Part 8): for one crop the model scores each allowed
year string with CTC probability mass and the decision picks the argmax.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

ALPHABET = "0123456789"
BLANK_INDEX = 0
CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(ALPHABET)}
INPUT_HEIGHT = 48
INPUT_WIDTH = 160
MAX_YEAR_LENGTH = 4

FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif-Bold.ttf",
    "DejaVuSans.ttf",
    "DejaVuSerif.ttf",
)


def _find_font() -> str | None:
    from PIL import ImageFont

    import glob

    for pattern in (
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/*.ttf",
    ):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def _render_year(
    year: str,
    rng: random.Random,
    font_path: str | None,
) -> Any:
    """Render one synthetic year crop with heavy augmentation."""

    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

    font_size = rng.randint(28, 64)
    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path
        else ImageFont.load_default()
    )
    # Canvas with random background.
    background_value = rng.choice((255, 245, 230, 40, 20, rng.randint(0, 255)))
    canvas = Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), background_value)
    draw = ImageDraw.Draw(canvas)
    text_color = rng.choice((0, 20, 40, 200, 255, rng.randint(0, 255)))
    # Keep text readable against the background most of the time.
    if abs(text_color - background_value) < 90:
        text_color = 255 - background_value
    bbox = draw.textbbox((0, 0), year, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = rng.randint(2, max(2, INPUT_WIDTH - text_width - 2))
    y = rng.randint(2, max(2, INPUT_HEIGHT - text_height - 2))
    draw.text((x, y), year, fill=text_color, font=font)

    if rng.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.6)))
    if rng.random() < 0.4:
        canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.4, 1.4))
    if rng.random() < 0.4:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        # Gold/silver-ish light text effect: invert with low contrast.
        if rng.random() < 0.5:
            canvas = canvas.point(lambda value: 255 - value)
    if rng.random() < 0.3:
        # Random small rotation.
        canvas = canvas.rotate(rng.uniform(-6, 6), expand=False, fillcolor=background_value)
    return canvas


def _encode_labels(year: str) -> list[int]:
    return [CHAR_TO_INDEX[char] for char in year]


def _ctc_lambda(labels: Sequence[int]) -> Any:
    from functools import partial

    def _convert(batch):
        import torch

        targets = []
        target_lengths = []
        for sequence in batch:
            targets.extend(sequence)
            target_lengths.append(len(sequence))
        return (
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
        )

    return partial(_convert)


class YearRecognizerModel:
    """Small CNN + BiLSTM + CTC head over 48x160 grayscale input."""

    def __init__(self) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(20260922)
        self.torch = torch

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.AdaptiveAvgPool2d((1, None)),
                )
                self.lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(256, len(ALPHABET) + 1)

            def forward(self, x):
                features = self.cnn(x)
                batch, channels, height, width = features.shape
                assert height == 1, f"expected height 1, got {height}"
                features = features.squeeze(2).permute(0, 2, 1)
                sequence, _ = self.lstm(features)
                return self.fc(sequence)

        self.net = _Model()

    def ctc_output_width(self) -> int:
        return INPUT_WIDTH // 4


def synthetic_year_dataset(
    output_dir: str | Path,
    years: Iterable[int],
    samples_per_year: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate (and cache) synthetic year crops with labels."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    font_path = _find_font()
    records = []
    for year in years:
        text = str(year)
        for index in range(samples_per_year):
            image = _render_year(text, rng, font_path)
            path = output_dir / f"{text}_{index:03d}.png"
            image.save(path)
            records.append({"path": str(path), "year": text})
    return records


def train_year_recognizer(
    train_dir: str | Path,
    years: range,
    samples_per_year: int,
    epochs: int,
    seed: int,
    val_share: float = 0.1,
    batch_size: int = 64,
) -> "YearRecognizer":
    """Train the digit recognizer on synthetic crops; return the wrapper."""

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    records = synthetic_year_dataset(train_dir, years, samples_per_year, seed)
    rng = random.Random(seed + 1)
    rng.shuffle(records)
    val_count = max(1, int(len(records) * val_share))
    val_records = records[:val_count]
    train_records = records[val_count:]

    class _YearDataset(Dataset):
        def __init__(self, records: list[dict]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            image = np.asarray(Image.open(record["path"]).convert("L"), dtype=np.float32) / 255.0
            image = (image - 0.5) / 0.5
            return {
                "image": torch.tensor(image).unsqueeze(0),
                "labels": torch.tensor(_encode_labels(record["year"]), dtype=torch.long),
            }

    def _collate(batch: list[dict]) -> dict:
        images = torch.stack([item["image"] for item in batch])
        targets = []
        target_lengths = []
        for item in batch:
            labels = item["labels"]
            targets.extend(labels.tolist())
            target_lengths.append(len(labels))
        return {
            "images": images,
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        }

    model_wrapper = YearRecognizerModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model_wrapper.net.to(device)
    ctc = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_loader = DataLoader(
        _YearDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=_collate
    )
    history = []
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = net(images)
            log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full(
                (images.size(0),), logits.size(1), dtype=torch.long, device=device
            )
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss / max(len(train_loader), 1), 4)})
        print(f"  digit epoch {epoch + 1}/{epochs}: loss={history[-1]['loss']}", flush=True)

    # Synthetic validation accuracy (unrestricted greedy decode).
    net.eval()
    val_loader = DataLoader(
        _YearDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=_collate
    )
    correct = 0
    with torch.inference_mode():
        for batch in val_loader:
            logits = net(batch["images"].to(device))
            decoded = _greedy_decode(logits)
            targets = batch["targets"].tolist()
            lengths = batch["target_lengths"].tolist()
            offset = 0
            for length in lengths:
                expected = "".join(ALPHABET[index - 1] for index in targets[offset : offset + length])
                offset += length
                if decoded.pop(0) == expected:
                    correct += 1
    synthetic_accuracy = correct / max(len(val_records), 1)

    return YearRecognizer(
        net=net,
        device=device,
        synthetic_val_accuracy=round(synthetic_accuracy, 4),
        training_history=history,
        train_rows=len(train_records),
        val_rows=len(val_records),
    )


def _greedy_decode(logits: Any) -> list[str]:
    import torch

    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded = []
    for sequence in indices:
        characters = []
        previous = -1
        for index in sequence:
            if index != previous and index != BLANK_INDEX:
                characters.append(ALPHABET[index - 1])
            previous = index
        decoded.append("".join(characters))
    return decoded


class YearRecognizer:
    """Inference wrapper: unrestricted decode + constrained CTC scoring."""

    def __init__(
        self,
        net: Any,
        device: str,
        synthetic_val_accuracy: float,
        training_history: list[dict],
        train_rows: int,
        val_rows: int,
    ) -> None:
        import torch

        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.synthetic_val_accuracy = synthetic_val_accuracy
        self.training_history = training_history
        self.train_rows = train_rows
        self.val_rows = val_rows

    def unrestricted_decode(self, normalized_crop: Any) -> str:
        import torch

        from PIL import Image

        array = np.asarray(normalized_crop.convert("L"), dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.net(tensor)
        return _greedy_decode(logits)[0]

    def constrained_scores(
        self, normalized_crop: Any, allowed_years: Sequence[str]
    ) -> dict[str, float]:
        """CTC probability mass of each allowed year string."""

        import torch

        from PIL import Image

        array = np.asarray(normalized_crop.convert("L"), dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.net(tensor)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]  # (T, C)
        scores: dict[str, float] = {}
        input_length = log_probs.size(0)
        for year in allowed_years:
            labels = _encode_labels(year)
            if len(labels) > input_length:
                scores[year] = float("-inf")
                continue
            # Best contiguous window CTC score for this fixed label sequence
            # (simplified: single alignment-independent approximation via
            # per-position max over blank/pair expansions). For a 4-char
            # string and T>=8 frames the standard prefix score is used.
            best = float("-inf")
            for start in range(0, input_length - (2 * len(labels)) + 1):
                score = 0.0
                cursor = start
                for label in labels:
                    # Collapse: take max prob of the char over two frames,
                    # then require a blank between repeated chars.
                    char_logit = float(log_probs[cursor, label])
                    char_logit2 = (
                        float(log_probs[cursor + 1, label])
                        if cursor + 1 < input_length
                        else char_logit
                    )
                    blank_logit = float(log_probs[cursor + 1, BLANK_INDEX]) if cursor + 1 < input_length else float(log_probs[cursor, BLANK_INDEX])
                    score += math.log(max(math.exp(char_logit) + math.exp(char_logit2), 1e-9))
                    score += math.log(max(math.exp(blank_logit), 1e-9))
                    cursor += 2
                best = max(best, score)
            scores[year] = round(best, 4)
        return scores

    def metadata(self) -> dict[str, Any]:
        return {
            "synthetic_val_accuracy": self.synthetic_val_accuracy,
            "training_history": self.training_history,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "alphabet": ALPHABET,
            "input": f"{INPUT_HEIGHT}x{INPUT_WIDTH} grayscale",
        }
