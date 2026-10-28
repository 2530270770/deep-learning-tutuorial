from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import print_rule, resolve_device, tensor_logical_bytes


def main() -> None:
    device = resolve_device()
    print("device:", device)

    print_rule("shape and stride")
    x = torch.arange(12, dtype=torch.float32, device=device).view(3, 4)
    print(x)
    print("shape:", tuple(x.shape), "stride:", x.stride())
    assert x.shape == (3, 4)
    assert x.stride() == (4, 1)

    print_rule("view requires compatible layout")
    transposed = x.transpose(0, 1)
    print("transposed shape:", tuple(transposed.shape))
    print("transposed stride:", transposed.stride())
    print("contiguous:", transposed.is_contiguous())
    assert not transposed.is_contiguous()
    try:
        transposed.view(12)
    except RuntimeError:
        print("view failed as expected")
    else:
        raise AssertionError("non-contiguous transpose unexpectedly accepted view")
    reshaped = transposed.reshape(12)
    print("reshape shape:", tuple(reshaped.shape))

    print_rule("chunk returns views")
    gate_up = torch.arange(12, dtype=torch.float32, device=device).view(2, 6)
    gate, up = gate_up.chunk(2, dim=-1)
    print("gate_up:", tuple(gate_up.shape), "gate:", tuple(gate.shape), "up:", tuple(up.shape))
    assert gate.data_ptr() == gate_up.data_ptr()
    assert up.storage_offset() == 3

    print_rule("broadcast")
    scale = torch.tensor([1.0, 10.0, 100.0, 1000.0], device=device)
    scaled = x * scale
    print("broadcast shape:", tuple(scaled.shape))
    assert scaled.shape == (3, 4)

    print_rule("Qwen3 shapes and BF16 bytes")
    tokens = 128
    hidden = 1024
    intermediate = 3072
    heads = 16
    kv_heads = 8
    head_dim = 128
    shapes = {
        "embedding": (tokens, hidden),
        "q": (tokens, heads, head_dim),
        "k": (tokens, kv_heads, head_dim),
        "v": (tokens, kv_heads, head_dim),
        "gate_up": (tokens, 2 * intermediate),
        "hidden": (tokens, intermediate),
    }
    for name, shape in shapes.items():
        dummy = torch.empty(shape, dtype=torch.bfloat16, device=device)
        print(f"{name:10s} {str(shape):20s} {tensor_logical_bytes(dummy) / 1024:.1f} KiB")
    assert shapes["gate_up"] == (128, 6144)

    print("\nPASS: tensor shape, stride, view, broadcast and Qwen3 byte checks")


if __name__ == "__main__":
    main()

