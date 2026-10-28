from __future__ import annotations

import math
import unittest


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


class MathReferenceTests(unittest.TestCase):
    def test_activation_values(self) -> None:
        expected_silu = [-0.238406, -0.268941, 0.0, 0.731059, 1.761594]
        expected_gelu = [-0.045500, -0.158655, 0.0, 0.841345, 1.954500]
        points = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for x, s_expected, g_expected in zip(points, expected_silu, expected_gelu):
            self.assertAlmostEqual(silu(x), s_expected, places=6)
            self.assertAlmostEqual(gelu(x), g_expected, places=6)

    def test_tiny_swiglu(self) -> None:
        x = [1.0, 2.0]
        w_gate = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        w_up = [[1.0, 1.0], [2.0, 0.0], [1.0, 1.0]]
        w_down = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        gate = [sum(a * b for a, b in zip(x, row)) for row in w_gate]
        up = [sum(a * b for a, b in zip(x, row)) for row in w_up]
        hidden = [silu(a) * b for a, b in zip(gate, up)]
        output = [sum(a * b for a, b in zip(hidden, row)) for row in w_down]
        self.assertAlmostEqual(output[0], 10.766343, places=5)
        self.assertAlmostEqual(output[1], 3.523188, places=5)

    def test_qwen_parameter_and_traffic_accounts(self) -> None:
        hidden, intermediate, layers = 1024, 3072, 28
        per_layer = 3 * hidden * intermediate
        self.assertEqual(per_layer, 9_437_184)
        self.assertEqual(per_layer * layers, 264_241_152)
        self.assertEqual(per_layer * layers * 2 / 2**20, 504.0)

        tokens = 32
        n = tokens * intermediate
        unfused_bytes = 5 * n * 2
        fused_bytes = 3 * n * 2
        self.assertEqual(unfused_bytes, 983_040)
        self.assertEqual(fused_bytes, 589_824)
        self.assertEqual(unfused_bytes - fused_bytes, 393_216)

    def test_qwen_attention_shapes(self) -> None:
        q_width = 16 * 128
        kv_width = 8 * 128
        self.assertEqual(q_width, 2048)
        self.assertEqual(kv_width, 1024)
        self.assertEqual(q_width + 2 * kv_width, 4096)
        per_token_kv = 2 * 28 * 8 * 128 * 2
        self.assertEqual(per_token_kv, 114_688)

    def test_linear_layer_can_collapse_without_nonlinearity(self) -> None:
        # Symbolic shape account: two affine maps are still one affine map.
        x = [1.0, 2.0]
        w1 = [[1.0, 0.0], [0.0, 1.0]]
        w2 = [[2.0, 1.0]]
        h = [sum(a * b for a, b in zip(x, row)) for row in w1]
        y = [sum(a * b for a, b in zip(h, row)) for row in w2]
        # (x @ W1^T) @ W2^T == x @ (W2 W1)^T
        w_effective = [sum(w2[0][mid] * w1[mid][col] for mid in range(2)) for col in range(2)]
        x_w1_w2 = sum(x_i * w_i for x_i, w_i in zip(x, w_effective))
        self.assertAlmostEqual(y[0], x_w1_w2)


if __name__ == "__main__":
    unittest.main()

