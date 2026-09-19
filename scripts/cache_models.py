#!/usr/bin/env python3
"""Download and initialize every pretrained baseline encoder once."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition.encoder import SUPPORTED_MODEL_NAMES, VisualEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache pretrained visual encoders")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODEL_NAMES),
        choices=SUPPORTED_MODEL_NAMES,
        help="Models to cache; defaults to all baseline encoders",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    for model_name in args.models:
        encoder = VisualEncoder(model_name, args.device, args.batch_size)
        print(
            f"{model_name}: encoder={encoder.encoder_name} "
            f"embedding_dim={encoder.embedding_dim} device={encoder.device}"
        )
        used_cuda = encoder.device.startswith("cuda")
        del encoder
        gc.collect()

        # Sequential loading keeps the cache command usable on an 8 GB GPU.
        if used_cuda:
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
