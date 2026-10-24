# GhostUNet3D deployment experiment

This experiment exports a training checkpoint, runs ONNX Runtime on CPU, and
compares its output with the original PyTorch network. It establishes a model
artifact and numerical reference for a future Preludio runtime.

For project scope, milestones, and the runtime design, see the
[spec](spec.md), [roadmap](roadmap.md), and [architecture](architecture.md).

## Model contract

The initial checkpoint is `step_22000.ckpt` from the Onix run
`ghost_unet3d_wmse/2026-10-23_23-54-23`. The filename is `step_22000.ckpt`, not
`22000.ckpt`; `last.ckpt` is a moving symlink and is not a version identifier.

| Property | Value |
| --- | --- |
| Architecture | GhostUNet3D with CDU upsampling and GroupNorm |
| Parameters | 7,515,108 |
| Input | float32 `[1, 4, 448, 448]`, oldest frame first |
| Output | float32 `[1, 20, 448, 448]`, earliest lead first |
| Configured cadence | 6 minutes |
| Configured forecast leads | +6 through +120 minutes from the latest input |
| Tensor values | Normalized training grayscale PNG intensities in `[0, 1]` |
| Output activation | `relu6(x) / 6` |
| Export | ONNX opset 18; fixed batch, time, and spatial dimensions |

The input frames cover 18 minutes between the first and last observations.
Forecast leads describe the configuration; parity checks do not establish
forecast accuracy at those leads.

The training pipeline converts grayscale PNG pixels to float32, divides by
255, and resizes to 448 x 448 with bilinear interpolation and antialiasing.
This run uses no rotation. It transports the resized tensor as float16 and
casts it back to float32 before the network. A fixture reproducing this run
should include that rounding step.

The PNG-to-physical-dBZ mapping and geographic grid have not been established
from the supplied configuration. The manifest leaves both unset. Do not feed
Radiust dBZ fields directly into this model or label its normalized output as
dBZ, rain rate, or precipitation probability without verifying the data
encoding and grid first. Missing observations and no-echo pixels also need
separate semantics before live integration.

## Export and verify

Use Python 3.11 and an isolated development environment:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python -m unittest discover -s tools -p 'test_*.py' -v
```

To recreate the real input used for the recorded validation, use these four
six-minute PNG observations from station AZ9002, in chronological order:

```sh
uv pip install --python .venv/bin/python pillow==12.2.0
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

root = Path('/home/blizhan/data/radar/china/png/AZ9002/2023/11/30')
frames = []
for time in ('092400', '093000', '093600', '094200'):
    with Image.open(root / f'AZ9002_20231130_{time}Z.png') as image:
        frames.append(np.array(image.convert('L')))
values = torch.from_numpy(np.stack(frames)).float().div_(255).unsqueeze(0)
values = F.interpolate(values, size=(448, 448), mode='bilinear',
                       align_corners=False, antialias=True).half().float()
output = Path('artifacts/ghost-unet3d-input')
output.mkdir(parents=True, exist_ok=True)
np.save(output / 'input.npy', values.numpy(), allow_pickle=False)
PY
```

Export using the original model source and its Hydra configuration:

```sh
ONIX_ROOT=/home/blizhan/repo/skv/onix
MODEL_RUN="$ONIX_ROOT/outputs/radar_video_prediction/ghost_unet3d_wmse/2026-10-23_23-54-23"
.venv/bin/python tools/export_ghost_unet3d.py \
  --onix-root "$ONIX_ROOT" \
  --checkpoint "$MODEL_RUN/checkpoints/step_22000.ckpt" \
  --config "$MODEL_RUN/.hydra/config.yaml" \
  --input artifacts/ghost-unet3d-input/input.npy \
  --output artifacts/ghost-unet3d-step-22000
```

The output directory must be new or empty. `--input` accepts a preprocessed
float32 NumPy array with the exact input shape and finite values in `[0, 1]`.
Omitting it creates a deterministic synthetic input; the validation report
records which input was used. Synthetic parity is not a forecast evaluation.

The script reads the Onix model source directly. It does not construct a
Lightning trainer, enable `torch.compile`, or modify the training repository.
It strips the Lightning `net.` and compiled `_orig_mod.` prefixes and loads
model weights strictly. Only use trusted source code and checkpoints: Lightning
checkpoints also contain pickled Hydra and NumPy objects.

Successful exports produce these local files under the ignored `artifacts/`
directory:

| File | Purpose |
| --- | --- |
| `model.onnx` | Float32 inference graph with embedded weights |
| `manifest.json` | Shapes, value encoding, configured leads, and SHA-256 hashes |
| `input.npy` | Exact normalized input used for verification |
| `reference.npy` | Original PyTorch prediction |
| `forecast.npy` | ONNX Runtime prediction |
| `validation.json` | Numerical error, first-run timings, versions, and operators |

The script checks ONNX validity, output shape, finiteness, and numerical
agreement with `rtol=1e-4, atol=1e-5`. It writes the manifest only after those
checks pass. Timings are one CPU invocation with a configurable thread count,
not a sustained benchmark or a measurement of macOS hardware.

## Recorded CPU validation

On 2026-10-24, the original 732 x 732 PNG observations from 09:24 through
09:42 UTC on 2023-11-30 were preprocessed as described above. The resulting
input range was `[0, 0.058837890625]`; the reference forecast range was
`[0, 0.07919428497552872]`. This is one real observation window, not a test set.

| Measurement | Result |
| --- | --- |
| Host | Linux ARM64, 4 CPU inference threads |
| Versions | PyTorch 2.12.1+cu130, ONNX Runtime 1.31.0, ONNX 1.23.2 |
| ONNX size | 31,072,725 bytes (29.63 MiB) |
| Maximum absolute error | 9.61125e-7 |
| Mean absolute error | 3.85843e-9 |
| PyTorch first inference | 4.582 s |
| ONNX session initialization | 0.029 s |
| ONNX first inference | 4.097 s |

The checkpoint SHA-256 is
`c5d8b457ad603ff53502616e97cc048aee0b22ded033b91acc5742b88fbc4a12`.
The local `manifest.json` and `validation.json` retain the artifact hashes,
full environment, error measurements, and operator inventory.

## Normalization export

The default PyTorch lowering expresses GroupNorm using InstanceNormalization.
On this ARM64 CPU, the first convolution matched exactly, while the first
lowered normalization differed by up to 0.06358. Disabling ONNX Runtime graph
optimizations did not change that error. The full prediction differed by up
to 0.001436, failing the parity tolerance.

The exporter uses PyTorch's custom translation table to express GroupNorm with
centered variance and a series of small reductions over individual axes.
This avoids the long floating-point accumulations while keeping the graph in
float32 and using standard ONNX operators. A regression test exercises
normalization over large spatial groups. The original network and its trained
parameters are unchanged.

## Deployment boundary

The checked execution provider is CPU. macOS Core ML, browser WASM, and actual
device latency and memory remain to be tested.

This network contains 3D convolutions. As checked on 2026-10-24, the ONNX Runtime WebGPU
[operator support table](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)
marks Conv3D unsupported. Export success therefore does not establish
WebGPU compatibility. Browser GPU deployment needs a compatible model or a
separately validated operator conversion.

Radiust integration also requires the physical value encoding, geographic
grid, missing-data policy, and timestamp checks described above. The Rust crate
still exposes its initial placeholder API; this experiment adds deployment
tooling without selecting a public runtime API.
