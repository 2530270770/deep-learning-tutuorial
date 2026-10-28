from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    tokens,
    intermediate,
    gate_stride_t,
    up_stride_t,
    out_stride_t,
    BLOCK_I: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (token < tokens) & (cols < intermediate)

    gate = tl.load(
        gate_ptr + token * gate_stride_t + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + token * up_stride_t + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + token * out_stride_t + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def triton_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    block_i: int = 256,
) -> torch.Tensor:
    if gate.ndim != 2 or up.ndim != 2:
        raise ValueError("gate and up must be 2-D [tokens, intermediate]")
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch: {gate.shape} vs {up.shape}")
    if gate.device.type != "cuda" or up.device.type != "cuda":
        raise ValueError("the Triton lab requires CUDA tensors")
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("the last dimension must be contiguous")
    if gate.dtype != up.dtype:
        raise ValueError(f"dtype mismatch: {gate.dtype} vs {up.dtype}")

    tokens, intermediate = gate.shape
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)
    if gate.numel() == 0:
        return out

    grid = (tokens, triton.cdiv(intermediate, block_i))
    silu_mul_kernel[grid](
        gate,
        up,
        out,
        tokens,
        intermediate,
        gate.stride(0),
        up.stride(0),
        out.stride(0),
        BLOCK_I=block_i,
        num_warps=4 if block_i <= 512 else 8,
    )
    return out

