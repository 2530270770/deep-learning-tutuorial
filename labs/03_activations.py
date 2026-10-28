from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import measure_ms, print_machine_info, print_rule, resolve_device


def silu_formula(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def gelu_formula(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh_formula(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print_machine_info(device)

    print_rule("fixed numerical examples")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    outputs = {
        "ReLU": F.relu(x),
        "GELU": F.gelu(x),
        "GELU_tanh": gelu_tanh_formula(x),
        "SiLU": F.silu(x),
    }
    for name, values in outputs.items():
        print(f"{name:10s}", [round(v, 6) for v in values.tolist()])

    torch.testing.assert_close(gelu_formula(x), F.gelu(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(silu_formula(x), F.silu(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gelu_tanh_formula(x), F.gelu(x), rtol=3e-4, atol=3e-4)

    print_rule("SiLU gradient")
    gx = x.clone().detach().requires_grad_(True)
    F.silu(gx).sum().backward()
    expected_grad = torch.tensor(
        [-0.090784, 0.072329, 0.5, 0.927671, 1.090784],
        device=device,
    )
    print("autograd:", [round(v, 6) for v in gx.grad.tolist()])
    torch.testing.assert_close(gx.grad, expected_grad, rtol=1e-5, atol=1e-5)

    print_rule("SwiGLU local gradient")
    gate = torch.tensor([1.0, -1.0], device=device, requires_grad=True)
    up = torch.tensor([2.0, 3.0], device=device, requires_grad=True)
    (F.silu(gate) * up).sum().backward()
    print("gate.grad:", gate.grad.tolist())
    print("up.grad:", up.grad.tolist())
    # d/da SiLU(a)b and d/db SiLU(a)
    torch.testing.assert_close(
        gate.grad,
        torch.tensor([0.927671 * 2.0, 0.072329 * 3.0], device=device),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        up.grad,
        torch.tensor([0.731059, -0.268941], device=device),
        rtol=1e-5,
        atol=1e-5,
    )

    print_rule("shape and dtype")
    for dtype in (torch.float32, torch.bfloat16):
        xb = torch.randn(4, 8, device=device, dtype=dtype)
        for name, fn in (("ReLU", F.relu), ("GELU", F.gelu), ("SiLU", F.silu)):
            y = fn(xb)
            print(f"{str(dtype):16s} {name:4s} {tuple(xb.shape)} -> {tuple(y.shape)} {y.dtype}")
            assert y.shape == xb.shape
            assert y.dtype == xb.dtype

    if args.benchmark:
        print_rule("elementwise benchmark")
        if device.type != "cuda":
            print("CUDA unavailable; GPU benchmark skipped.")
            return
        for tokens in args.tokens:
            xb = torch.randn(tokens, 3072, device=device, dtype=torch.bfloat16)
            read_write_bytes = 2 * xb.numel() * xb.element_size()
            for name, fn in (("ReLU", F.relu), ("GELU", F.gelu), ("SiLU", F.silu)):
                stats = measure_ms(lambda: fn(xb), device=device, warmup=20, repeats=100)
                bandwidth = read_write_bytes / (stats["median_ms"] * 1e-3) / 1e9
                print(
                    f"T={tokens:<4d} {name:4s} "
                    f"median={stats['median_ms']:.4f} ms p90={stats['p90_ms']:.4f} ms "
                    f"effective={bandwidth:.1f} GB/s"
                )

    print("\nPASS: activation values, gradients, shapes and dtype")


if __name__ == "__main__":
    main()

