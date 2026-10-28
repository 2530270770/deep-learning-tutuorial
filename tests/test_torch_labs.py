from __future__ import annotations

import importlib
import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TorchLabTests(unittest.TestCase):
    def test_activation_identities(self) -> None:
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        torch.testing.assert_close(
            x * torch.sigmoid(x),
            F.silu(x),
            rtol=1e-6,
            atol=1e-6,
        )
        tau = torch.erf(x / 2**0.5)
        torch.testing.assert_close(
            0.5 * x * (1.0 + tau),
            F.gelu(x),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_merged_qwen3_mlp(self) -> None:
        module = importlib.import_module("labs.04_qwen3_mlp")
        torch.manual_seed(3)
        model = module.MergedSwiGLU(32, 64).eval()
        x = torch.randn(3, 32)
        with torch.inference_mode():
            merged = model(x)
            separate = model.separate_forward(x)
        torch.testing.assert_close(merged, separate, rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(model.gate_up_output(x).shape), (3, 128))


if __name__ == "__main__":
    unittest.main()

