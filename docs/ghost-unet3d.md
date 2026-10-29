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
| Configured cadence | 6 minutes in the training configuration; runtime inputs need only be strictly increasing in time |
| Configured forecast leads | +6 through +120 minutes in the fixed training/reference configuration; runtime leads use output tensor count and input average interval |
| Tensor values | Normalized grayscale values in `[0, 1]` |
| Output activation | `relu6(x) / 6` |
| Export | ONNX opset 18; fixed batch, time, and spatial dimensions |

The recorded input frames cover 18 minutes between the first and last observations. These cadence and lead values describe that training/reference configuration; numerical parity does not establish forecast accuracy or constrain runtime input intervals.

For the recorded experiment, grayscale PNG bytes were converted to float32, divided by
255, and resized to 448 x 448 with bilinear interpolation and antialiasing.
This run uses no rotation. It transports the resized tensor as float16 and
casts it back to float32 before the network. A fixture reproducing this run
should include that rounding step. This describes the fixed reference path; it
does not establish that the grayscale extraction matches duckomo gray or that
the repository implements a Duckomo conversion.

The supplied configuration does not establish a physical dBZ mapping or
geographic grid; neither is required by the current raw-u8 runtime contract.
Inputs must be single-channel u8 gray values whose meaning matches duckomo
gray. This repository has no evidence of a Duckomo conversion implementation
or a concrete Radiust source supplying those bytes. Do not feed dBZ fields
directly into this model or label normalized outputs as dBZ, rain rate, or
precipitation probability. Missing and OutsideCoverage quality values are
rejected; a valid no-echo value is u8 zero. The optional zero threshold is
strictly less than its value and runs before `/255` and resize.

## Export and verify

Use Python 3.11 and an isolated development environment:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python -m unittest discover -s tools -p 'test_*.py' -v
```

The documented reproduction recipe for the fixed validation uses these four
six-minute PNG observations from station AZ9002, in chronological order. Its
grayscale extraction is not evidence of a Duckomo conversion or compatibility
with a concrete Radiust source:

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
  --run ghost_unet3d_wmse/2026-10-23_23-54-23 \
  --config "$MODEL_RUN/.hydra/config.yaml" \
  --input artifacts/ghost-unet3d-input/input.npy \
  --output artifacts/ghost-unet3d-step-22000
```

The output directory must be new or empty. `--input` accepts a preprocessed
float32 NumPy array with the exact input shape and finite values in `[0, 1]`.
Omitting it creates a deterministic synthetic input; the validation report
records which input was used. Synthetic parity is not a forecast evaluation.

`--run` is required and must identify the training run that produced the supplied
checkpoint. It is recorded verbatim in `origin.checkpoint.run`; the exporter does
not infer it from the checkpoint's filename or location.

The training configuration must set `datamodule.rot: 0` and
`datamodule.transport_dtype: float16`. The exporter rejects other values before
loading model source or checkpoint weights, because the package preprocessing
contract specifies no rotation and includes float16 rounding.

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

On 2026-10-24, the fixed reference input was associated with original 732 x 732
PNG observations from 09:24 through 09:42 UTC on 2023-11-30. The recorded input
range was `[0, 0.058837890625]`. This is one real observation window, not a test
set. The artifacts validate only the fixed input and default no-threshold path;
they do not establish Duckomo-compatible gray extraction, arbitrary positive
input dimensions, or non-default threshold behavior.

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

The checked execution providers are native CPU (Linux ARM64 above; macOS
arm64 in the [runtime research](../specs/001-local-nowcast-runtime/research.md))
and ONNX Runtime Web WASM (below, plus Chrome 150 and Safari 26.6 in the same
research). On macOS, native ORT must be pinned to the CPU provider: automatic
device selection placed part of the graph on CoreML and failed parity.

### WASM CPU check

On 2026-10-24 the same `model.onnx` and `input.npy` ran on the
`onnxruntime-web` 1.30.0 WASM execution provider under Bun 1.4.3 on an Apple
M4 Pro (14 cores, macOS). This is a JavaScript runtime, not a browser. Conv3D,
3D MaxPool, and Resize all ran without unsupported-operator errors.

| WASM threads | Session creation | Inference (run 1 / run 2) | Max abs error vs PyTorch | Tolerance failures |
| --- | --- | --- | --- | --- |
| 1 | 0.119 s | 7.744 s / 7.600 s | 9.40636e-7 | 0 |
| 4 | 0.127 s | 2.242 s / 2.165 s | 9.40636e-7 | 0 |

Mean absolute error was 4.94896e-9 and the maximum difference from the Linux
ONNX Runtime CPU forecast was 4.91738e-7. Process RSS was about 1.7 GiB,
including the JavaScript runtime; it is not an isolated model peak. Browsers
only allow WASM threads on cross-origin isolated pages (COOP/COEP); otherwise
expect the single-thread time. Chrome’s earlier fixed-reference and memory
records are historical evidence, not current support: a later headless
four-thread run took 21.15–28.60 s warm and had a 1064.54 ms maximum rAF gap,
so the Chrome target is no longer registered. Safari is not currently
supported. The standard four-thread browser and ten-run offline acceptance
remain incomplete; see the [runtime validation records](../specs/001-local-nowcast-runtime/quickstart.md).

This network contains 3D convolutions. As checked on 2026-10-24, the ONNX Runtime WebGPU
[operator support table](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)
marks Conv3D unsupported. Export success therefore does not establish
WebGPU compatibility. Browser GPU deployment needs a compatible model or a
separately validated operator conversion.

The current runtime contract does not require physical value conversion or
georeferencing. Radiust integration still needs a concrete source that can
provide duckomo-compatible raw u8 gray bytes and an end-to-end run. The adapter
guide describes that boundary but does not establish the conversion or source
compatibility; US5 remains incomplete.
