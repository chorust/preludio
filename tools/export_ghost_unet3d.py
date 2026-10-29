#!/usr/bin/env python3
"""Export the trusted Onix GhostUNet3D checkpoint and check ONNX CPU parity."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model_package import build_manifest, validate_manifest


def network_weights(state_dict):
    return {
        name.removeprefix("net.").removeprefix("_orig_mod."): tensor
        for name, tensor in state_dict.items()
        if name.startswith("net.")
    }


def load_input(path, shape):
    values = (
        np.load(path, allow_pickle=False)
        if path is not None
        else np.random.default_rng(42).random(shape, dtype=np.float32)
    )
    if values.dtype != np.float32 or values.shape != tuple(shape):
        raise ValueError(f"Expected float32 input with shape {shape}, got {values.dtype} {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Input must contain finite, normalized PNG values in [0, 1]")
    return np.ascontiguousarray(values)


def write_export_artifacts(output, values, reference, forecast, report, manifest):
    reference_dir = output / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    np.save(reference_dir / "input.npy", values, allow_pickle=False)
    np.save(reference_dir / "reference.npy", reference, allow_pickle=False)
    np.save(reference_dir / "forecast.npy", forecast, allow_pickle=False)
    (reference_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_manifest(manifest_path)


class _ExportGroupNorm(torch.nn.Module):
    def __init__(self, group_norm):
        super().__init__()
        self.num_groups = group_norm.num_groups
        self.eps = group_norm.eps
        self.weight = group_norm.weight
        self.bias = group_norm.bias

    def forward(self, input):
        grouped = input.reshape(input.shape[0], self.num_groups, -1, *input.shape[2:])
        mean = grouped
        for axis in range(input.dim(), 1, -1):
            mean = mean.mean(dim=axis, keepdim=True)
        centered = grouped - mean
        variance = centered * centered
        for axis in range(input.dim(), 1, -1):
            variance = variance.mean(dim=axis, keepdim=True)
        normalized = (centered / torch.sqrt(variance + self.eps)).reshape(input.shape)
        if self.weight is not None:
            affine_shape = [1, -1] + [1] * (input.dim() - 2)
            normalized = normalized * self.weight.reshape(affine_shape)
            normalized = normalized + self.bias.reshape(affine_shape)
        return normalized


def _replace_group_norms(module):
    replacements = []
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GroupNorm):
            replacements.append((module, name, child))
            setattr(module, name, _ExportGroupNorm(child))
        else:
            replacements.extend(_replace_group_norms(child))
    return replacements


def export_model(model, tensor, path):
    # Decompose GroupNorm into reductions to avoid inaccurate InstanceNormalization lowering.
    if isinstance(model, torch.nn.GroupNorm):
        exportable_model = _ExportGroupNorm(model)
        replacements = []
    else:
        exportable_model = model
        replacements = _replace_group_norms(model)
    try:
        torch.onnx.export(
            exportable_model, (tensor,), path,
            input_names=["radar"], output_names=["forecast"],
            opset_version=18, dynamo=False, external_data=False,
        )
    finally:
        for parent, name, group_norm in reversed(replacements):
            setattr(parent, name, group_norm)


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onix-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run", required=True, help="Training run identifier recorded in checkpoint provenance")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Already preprocessed float32 [1, 4, 448, 448] .npy")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ghost-unet3d"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.run.strip():
        parser.error("--run must not be blank")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be an empty or new directory")

    config_bytes = args.config.read_bytes()
    config = OmegaConf.create(config_bytes.decode("utf-8"))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    if model_config.pop("_target_") != "onix.models.ghost_unet3d.GhostUNet3D":
        raise ValueError("This exporter only supports Onix GhostUNet3D")
    if config.datamodule._target_ != "onix.datamodules.radar_video.RadarPngVideoDataModule":
        raise ValueError("This exporter only describes normalized PNG training inputs")
    if config.datamodule.rot != 0:
        raise ValueError("This exporter requires datamodule.rot = 0 (no rotation)")
    if config.datamodule.transport_dtype != "float16":
        raise ValueError("This exporter requires datamodule.transport_dtype = float16")
    height, width = config.datamodule.resize_size
    input_shape = [1, model_config["in_frames"], height, width]
    output_shape = [1, model_config["out_frames"], height, width]
    values = load_input(args.input, input_shape)

    source = args.onix_root / "src/onix/models/ghost_unet3d.py"
    spec = importlib.util.spec_from_file_location("preludio_ghost_unet3d", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(args.threads)
    model = module.GhostUNet3D(**model_config).eval()
    checkpoint_bytes = args.checkpoint.read_bytes()
    # Local Lightning checkpoints also pickle the Hydra config and NumPy scalars.
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    model.load_state_dict(network_weights(checkpoint["state_dict"]), strict=True)
    step = checkpoint["global_step"]
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    del checkpoint_bytes, checkpoint
    tensor = torch.from_numpy(values)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Running PyTorch reference: {input_shape} -> {output_shape}", flush=True)
    with torch.inference_mode():
        started = time.perf_counter()
        reference = model(tensor).numpy()
        reference_seconds = time.perf_counter() - started
        print("Exporting ONNX (fixed batch and spatial dimensions)", flush=True)
        export_model(model, tensor, args.output / "model.onnx")

    graph = onnx.load(args.output / "model.onnx")
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.output / "model.onnx"), sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = session.run(["forecast"], {"radar": values})[0]
    onnx_seconds = time.perf_counter() - started
    if reference.shape != tuple(output_shape) or actual.shape != tuple(output_shape):
        raise ValueError("Unexpected forecast shape")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Forecast contains non-finite values")
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    frame_minutes = int(config.datamodule.frame_minutes)
    model_sha256 = hashlib.sha256((args.output / "model.onnx").read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    opset_version = next(
        imported_opset.version
        for imported_opset in graph.opset_import
        if imported_opset.domain in ("", "ai.onnx")
    )
    forecast_grid_id = (
        f"onix-station-png-732/{height}"
        if width == height
        else f"onix-station-png-732/{height}x{width}"
    )
    manifest = build_manifest(
        model_id=f"ghost-unet3d-step-{step}",
        model={"file": "model.onnx", "sha256": model_sha256, "format": "onnx",
               "opset": opset_version},
        origin={
            "checkpoint": {"file": args.checkpoint.name, "step": int(step),
                           "sha256": checkpoint_hash,
                           "run": args.run},
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "exporter": "tools/export_ghost_unet3d.py",
        },
        observation={
            "frames": int(model_config["in_frames"]),
            "cadence_seconds": frame_minutes * 60,
            "cadence_tolerance_seconds": 0,
            "encoding": "training-grayscale-u8",
            "grid": {"id": "onix-station-png-732", "shape": [732, 732],
                     "georeferenced": False},
            "missing_policy": "reject",
        },
        tensor_input={"name": "radar", "dtype": "float32", "shape": input_shape,
                      "axes": ["batch", "time", "y", "x"], "range": [0.0, 1.0]},
        tensor_output={"name": "forecast", "dtype": "float32", "shape": output_shape,
                       "axes": ["batch", "lead_time", "y", "x"], "range": [0.0, 1.0]},
        preprocessing=[
            {"op": "u8_div_255"},
            {"op": "resize_bilinear_aa", "height": height, "width": width,
             "align_corners": False},
            {"op": "round_f16"},
        ],
        forecast={
            "lead_minutes": list(
                range(frame_minutes, frame_minutes * output_shape[1] + 1, frame_minutes)
            ),
            "encoding": "training-grayscale-normalized",
            "grid": {"id": forecast_grid_id, "shape": [height, width],
                     "georeferenced": False},
        },
    )
    error = np.abs(actual - reference)
    report = {
        "input_source": str(args.input.resolve()) if args.input else "synthetic_uniform_seed_42",
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_bytes": (args.output / "model.onnx").stat().st_size,
        "parity": {"rtol": 1e-4, "atol": 1e-5, "max_absolute_error": float(error.max()),
                   "mean_absolute_error": float(error.mean())},
        "cpu_first_run_seconds": {"pytorch": reference_seconds, "onnx_session": session_seconds,
                                  "onnx_inference": onnx_seconds},
        "environment": {"platform": platform.platform(), "threads": args.threads,
                        "torch": torch.__version__, "onnx": onnx.__version__,
                        "onnxruntime": ort.__version__, "numpy": np.__version__},
        "onnx_opsets": {op.domain or "ai.onnx": op.version for op in graph.opset_import},
        "onnx_operators": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
    }
    write_export_artifacts(args.output, values, reference, actual, report, manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
