from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import print_machine_info, print_rule, resolve_device, test_close


class MergedSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def gate_up_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.gate_up.weight)

    def separate_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_up.weight[: self.intermediate_size])
        up = F.linear(x, self.gate_up.weight[self.intermediate_size :])
        return F.linear(F.silu(gate) * up, self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_output(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.down.weight)


def set_tiny_weights(module: MergedSwiGLU) -> None:
    with torch.no_grad():
        module.gate_up.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=module.gate_up.weight.dtype,
                device=module.gate_up.weight.device,
            )
        )
        module.down.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=module.down.weight.dtype,
                device=module.down.weight.device,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    print_machine_info(device)

    print_rule("hand-calculated tiny SwiGLU")
    tiny = MergedSwiGLU(2, 3).to(device=device, dtype=dtype)
    set_tiny_weights(tiny)
    x = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)
    gate_up = tiny.gate_up_output(x)
    gate, up = gate_up.chunk(2, dim=-1)
    expected = torch.tensor([[10.766343, 3.523188]], device=device, dtype=dtype)
    actual = tiny(x)
    print("gate:", gate.tolist())
    print("up:", up.tolist())
    print("output:", actual.tolist())
    test_close(actual, expected, dtype=dtype)

    print_rule("merged and separate projections are equivalent")
    torch.manual_seed(0)
    qwen = MergedSwiGLU(1024, 3072).to(device=device, dtype=dtype)
    tokens = 2
    x = torch.randn(tokens, 1024, device=device, dtype=dtype)
    merged = qwen(x)
    separate = qwen.separate_forward(x)
    max_error = test_close(merged, separate, dtype=dtype)
    print("merged:", tuple(merged.shape), "separate:", tuple(separate.shape))
    print(f"max_abs_error: {max_error:.6g}")

    print_rule("Qwen3 shapes")
    gate_up = qwen.gate_up_output(x)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    print("x:       ", tuple(x.shape))
    print("gate_up: ", tuple(gate_up.shape))
    print("gate:    ", tuple(gate.shape))
    print("up:      ", tuple(up.shape))
    print("hidden:  ", tuple(hidden.shape))
    print("output:  ", tuple(merged.shape))
    assert gate_up.shape == (tokens, 6144)
    assert hidden.shape == (tokens, 3072)
    assert merged.shape == (tokens, 1024)

    print_rule("parameter and activation accounting")
    params = sum(p.numel() for p in qwen.parameters())
    print("MLP params per layer:", params)
    print("MLP params 28 layers:", params * 28)
    print("BF16 weight MiB:", params * 28 * 2 / 2**20)
    assert params == 9_437_184

    for t in (1, 8, 32, 128):
        gate_up_elems = t * 6144
        activation_elems = t * 3072
        print(
            f"T={t:<3d} gate_up={gate_up_elems:>9,d} elems, "
            f"hidden={activation_elems:>9,d} elems, "
            f"BF16 gate_up={gate_up_elems * 2 / 1024:.1f} KiB"
        )

    print("\nPASS: Qwen3 SwiGLU shapes, merged weights, parameters and activation accounting")


if __name__ == "__main__":
    main()

