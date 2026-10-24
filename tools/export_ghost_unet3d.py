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
from onnxscript import opset18 as op
from omegaconf import OmegaConf
import torch


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


def onnx_group_norm(input, num_groups: int, weight, bias, eps: float = 1e-5, cudnn_enabled: bool = True):
    grouped = op.Reshape(input, [0, num_groups, -1, *input.shape[2:]])
    # Small reductions avoid accumulation error across millions of spatial values.
    mean = grouped
    for axis in range(len(input.shape), 1, -1):
        mean = op.ReduceMean(mean, [axis], keepdims=1)
    centered = grouped - mean
    variance = centered * centered
    for axis in range(len(input.shape), 1, -1):
        variance = op.ReduceMean(variance, [axis], keepdims=1)
    normalized = op.Reshape(centered / op.Sqrt(variance + eps), op.Shape(input))
    affine_shape = [1, -1] + [1] * (len(input.shape) - 2)
    return normalized * op.Reshape(weight, affine_shape) + op.Reshape(bias, affine_shape)


def export_model(model, tensor, path):
    # InstanceNormalization lowering loses CPU precision over these large groups.
    torch.onnx.export(
        model, (tensor,), path,
        input_names=["radar"], output_names=["forecast"],
        opset_version=18, dynamo=True, external_data=False,
        custom_translation_table={torch.ops.aten.group_norm.default: onnx_group_norm},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onix-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Already preprocessed float32 [1, 4, 448, 448] .npy")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ghost-unet3d"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
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
    np.save(args.output / "input.npy", values, allow_pickle=False)
    np.save(args.output / "reference.npy", reference, allow_pickle=False)
    np.save(args.output / "forecast.npy", actual, allow_pickle=False)

    frame_minutes = int(config.datamodule.frame_minutes)
    manifest = {
        "id": f"ghost-unet3d-step-{step}",
        "model": "model.onnx",
        "sha256": hashlib.sha256((args.output / "model.onnx").read_bytes()).hexdigest(),
        "checkpoint": {"file": args.checkpoint.name, "step": step, "sha256": checkpoint_hash},
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_config": model_config,
        "input": {"name": "radar", "dtype": "float32", "shape": input_shape,
                  "axes": ["batch", "time", "y", "x"], "range": [0, 1],
                  "time_order": "oldest_to_newest", "frame_minutes": frame_minutes},
        "output": {"name": "forecast", "dtype": "float32", "shape": output_shape,
                   "axes": ["batch", "lead_time", "y", "x"], "range": [0, 1],
                   "lead_minutes": list(range(frame_minutes, frame_minutes * output_shape[1] + 1, frame_minutes))},
        "preprocessing": {"encoding": "training_grayscale_png_uint8_div_255",
                          "resize": "bilinear_float32_antialias_align_corners_false",
                          "rotation_quarter_turns": int(config.datamodule.rot),
                          "training_transport_dtype": str(config.datamodule.transport_dtype),
                          "physical_dbz_mapping": None, "georeferencing": None},
        "validated_providers": ["CPUExecutionProvider"],
    }
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
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
