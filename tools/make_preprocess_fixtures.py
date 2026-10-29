#!/usr/bin/env python3
"""Generate deterministic PyTorch preprocessing fixtures for the Rust and WASM tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "preprocess"


def sparse_frames() -> np.ndarray:
    """Build four distinct deterministic frames using only the observed levels."""
    levels = np.array([0, 4, 7, 11, 15, 18], dtype=np.uint8)
    frame, row, col = np.indices((4, 73, 61))
    indices = (frame * 3 + row * 2 + col) % len(levels)
    return levels[indices]


def random_frames(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).numpy()


def reference_preprocess(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x = torch.from_numpy(frames).to(device="cpu").unsqueeze(0)
    x_float = x.float() / 255.0
    y = F.interpolate(
        x_float,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).half().float()
    return y.squeeze(0).numpy()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("sparse_levels", sparse_frames(), (44, 37)),
        ("uniform_random_small", random_frames((4, 73, 61), seed=42), (44, 37)),
        ("all_zero_small", np.zeros((4, 73, 61), dtype=np.uint8), (44, 37)),
        ("uniform_random_full", random_frames((1, 732, 732), seed=43), (448, 448)),
        ("all_zero_full", np.zeros((1, 732, 732), dtype=np.uint8), (448, 448)),
    ]

    metadata = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cases": [],
    }
    with torch.inference_mode():
        for name, frames, output_size in cases:
            input_path = FIXTURE_DIR / f"input_{name}.npy"
            reference_path = FIXTURE_DIR / f"reference_{name}.npy"
            np.save(input_path, np.asarray(frames, dtype=np.uint8), allow_pickle=False)
            reference = reference_preprocess(frames, output_size)
            np.save(reference_path, np.asarray(reference, dtype=np.float32), allow_pickle=False)
            metadata["cases"].append(
                {
                    "name": name,
                    "input": input_path.name,
                    "reference": reference_path.name,
                    "input_shape": list(frames.shape),
                    "input_dtype": "uint8",
                    "output_size": list(output_size),
                    "reference_dtype": "float32",
                }
            )

    (FIXTURE_DIR / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
