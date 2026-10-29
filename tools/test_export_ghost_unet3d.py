import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from export_ghost_unet3d import (
    export_model,
    load_input,
    main,
    network_weights,
    write_export_artifacts,
)
from model_package import build_manifest, validate_manifest


class ExportInputTests(unittest.TestCase):
    def test_rejects_unsupported_training_preprocessing_before_loading_model(self):
        for rotation, dtype, message in (
            (1, "float16", "rot"),
            (-1, "float16", "rot"),
            (0.5, "float16", "rot"),
            (0, "float32", "transport_dtype"),
            (0, "bfloat16", "transport_dtype"),
        ):
            with self.subTest(rotation=rotation, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.json"
                config.write_text(json.dumps({
                    "model": {"_target_": "onix.models.ghost_unet3d.GhostUNet3D",
                              "in_frames": 4, "out_frames": 20},
                    "datamodule": {
                        "_target_": "onix.datamodules.radar_video.RadarPngVideoDataModule",
                        "rot": rotation, "transport_dtype": dtype, "resize_size": [2, 3],
                    },
                }), encoding="utf-8")
                with patch("sys.argv", [
                    "export_ghost_unet3d.py", "--onix-root", str(root),
                    "--checkpoint", str(root / "missing.ckpt"), "--run", "test-run",
                    "--config", str(config), "--output", str(root / "output"),
                ]):
                    with self.assertRaisesRegex(ValueError, message):
                        main()
                self.assertFalse((root / "output").exists())

    def test_group_norm_parity_with_large_spatial_reduction(self):
        torch.set_num_threads(2)
        model = torch.nn.GroupNorm(4, 16).eval()
        generator = torch.Generator().manual_seed(42)
        values = torch.rand((1, 16, 4, 128, 128), generator=generator) * 0.01 + 0.1
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = Path(directory) / "norm.onnx"
            reference = model(values).numpy()
            export_model(model, values, path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            actual = session.run(["forecast"], {"radar": values.numpy()})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)

    def test_lightning_prefixes_load_strictly(self):
        model = torch.nn.Linear(2, 1)
        for prefix in ("net.", "net._orig_mod."):
            with self.subTest(prefix=prefix):
                state = {prefix + key: value for key, value in model.state_dict().items()}
                state["criterion.unused"] = torch.tensor(0.0)
                restored = torch.nn.Linear(2, 1)
                restored.load_state_dict(network_weights(state), strict=True)
                torch.testing.assert_close(restored.weight, model.weight)

    def test_synthetic_input_is_reproducible(self):
        shape = (1, 4, 16, 16)
        first = load_input(None, shape)
        np.testing.assert_array_equal(first, load_input(None, shape))
        self.assertEqual(first.dtype, np.float32)

    def test_normalized_input_and_invalid_inputs(self):
        shape = (1, 4, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npy"
            valid = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
            np.save(path, valid)
            np.testing.assert_array_equal(load_input(path, shape), valid)
            for values in (valid.astype(np.float64), valid[:, :3], valid + 1, valid - 1,
                           np.full(shape, np.nan, dtype=np.float32),
                           np.full(shape, np.inf, dtype=np.float32)):
                with self.subTest(dtype=values.dtype, shape=values.shape):
                    np.save(path, values)
                    with self.assertRaises(ValueError):
                        load_input(path, shape)


    def test_export_artifacts_use_reference_layout_and_valid_manifest(self):
        values = np.linspace(0, 1, 24, dtype=np.float32).reshape(1, 4, 2, 3)
        forecast = np.zeros((1, 20, 2, 3), dtype=np.float32)
        manifest = build_manifest(
            model={"file": "model.onnx", "sha256": "a" * 64, "format": "onnx", "opset": 18},
            tensor_input={"name": "radar", "dtype": "float32", "shape": list(values.shape),
                          "axes": ["batch", "time", "y", "x"], "range": [0.0, 1.0]},
            tensor_output={"name": "forecast", "dtype": "float32", "shape": list(forecast.shape),
                           "axes": ["batch", "lead_time", "y", "x"], "range": [0.0, 1.0]},
        )
        report = {"parity": {"max_absolute_error": 0.0}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_export_artifacts(output, values, forecast, forecast, report, manifest)

            reference_dir = output / "reference"
            for name in ("input.npy", "reference.npy", "forecast.npy", "validation.json"):
                self.assertTrue((reference_dir / name).is_file())
                self.assertFalse((output / name).exists())
            np.testing.assert_array_equal(np.load(reference_dir / "input.npy"), values)
            np.testing.assert_array_equal(np.load(reference_dir / "reference.npy"), forecast)
            np.testing.assert_array_equal(np.load(reference_dir / "forecast.npy"), forecast)
            with (reference_dir / "validation.json").open(encoding="utf-8") as report_file:
                self.assertEqual(json.load(report_file), report)

            manifest_path = output / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            validate_manifest(manifest_path)
            with manifest_path.open(encoding="utf-8") as manifest_file:
                self.assertEqual(json.load(manifest_file)["schema"], "preludio.model-package/1")


if __name__ == "__main__":
    unittest.main()
