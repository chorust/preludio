from pathlib import Path
import json
import numpy as np
import PIL
import torch
import torchvision
from torch.nn import functional as F
from torchvision.transforms import Resize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
H = W = 732
OUT = 448


def sparse_radar_frames():
    yy, xx = np.mgrid[:H, :W]
    frames = np.zeros((4, H, W), dtype=np.uint8)
    levels = np.array([0, 4, 7, 11, 15, 18], dtype=np.uint8)
    # Deterministic translating / expanding echoes. Every pixel is assigned one
    # of the sparse observed levels; independent geometry per frame avoids a
    # degenerate interpolation-only test.
    echoes = [
        (173.0, 224.0, 73.0, 53.0),
        (184.0, 235.0, 78.0, 60.0),
        (198.0, 245.0, 81.0, 64.0),
        (212.0, 252.0, 84.0, 69.0),
        (493.0, 517.0, 92.0, 77.0),
        (507.0, 505.0, 98.0, 79.0),
        (524.0, 492.0, 103.0, 82.0),
        (541.0, 480.0, 108.0, 86.0),
    ]
    for t in range(4):
        a = echoes[t]
        b = echoes[4 + t]
        norm_a = ((xx - a[0]) / a[2]) ** 2 + ((yy - a[1]) / a[3]) ** 2
        norm_b = ((xx - b[0]) / b[2]) ** 2 + ((yy - b[1]) / b[3]) ** 2
        # A seeded, fixed mottling field creates irregular radar-like boundaries.
        rng = np.random.default_rng(481516 + t)
        texture = rng.random((H, W))
        score = np.maximum(1.0 - np.minimum(norm_a, norm_b), 0.0)
        score = np.clip(score + 0.14 * (texture - 0.5) * (score > 0), 0.0, 1.0)
        # Quantize into the six observed raw levels.
        bins = np.minimum((score * 6).astype(np.int32), 5)
        frames[t] = levels[bins]
    return frames


def random_frames():
    return np.random.default_rng(20261024).integers(0, 256, size=(4, H, W), dtype=np.uint8)


CASES = {
    "sparse_blobs": sparse_radar_frames(),
    "uniform_u8": random_frames(),
    "all_zero": np.zeros((4, H, W), dtype=np.uint8),
}

resize_tv = Resize([OUT, OUT], antialias=True)
metadata = {
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "pillow": PIL.__version__,
    "device": "cpu",
    "torch_intraop_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "input_shape": [4, H, W],
    "output_shape": [4, OUT, OUT],
    "cases": {},
}

with torch.inference_mode():
    for name, frames_np in CASES.items():
        frames = torch.from_numpy(frames_np)
        normalized = frames.to(torch.float32) / 255.0
        prehalf = F.interpolate(
            normalized.unsqueeze(0),
            size=(OUT, OUT),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0).contiguous()
        tv_prehalf = resize_tv(normalized).contiguous()
        tv_equal = torch.equal(prehalf, tv_prehalf)
        if not tv_equal:
            diff = (prehalf - tv_prehalf).abs()
            raise RuntimeError(
                f"torchvision mismatch for {name}: differing={(diff != 0).sum().item()} "
                f"max_abs={diff.max().item()}"
            )
        rounded = prehalf.half().float().contiguous()
        frames_np.tofile(DATA / f"{name}_input.raw")
        np.save(DATA / f"{name}_input.npy", frames_np)
        np.save(DATA / f"{name}_prehalf.npy", prehalf.numpy())
        np.save(DATA / f"{name}_output.npy", rounded.numpy())
        metadata["cases"][name] = {
            "unique_input_levels": np.unique(frames_np).tolist(),
            "torchvision_prehalf_bit_exact": tv_equal,
            "prehalf_min": float(prehalf.min()),
            "prehalf_max": float(prehalf.max()),
            "posthalf_nonzero": int((rounded != 0).sum()),
        }
        print(
            f"case={name} shape={tuple(frames_np.shape)} "
            f"unique={metadata['cases'][name]['unique_input_levels']} "
            f"torchvision_exact={tv_equal} posthalf_nonzero={metadata['cases'][name]['posthalf_nonzero']}"
        )

(DATA / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
