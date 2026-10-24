from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
import torch

from export_ghost_unet3d import export_model, load_input, network_weights


class ExportInputTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
