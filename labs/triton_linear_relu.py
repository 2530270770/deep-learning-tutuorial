from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def linear_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # W is [N, K], as in nn.Linear. The matmul needs [K, N].
        w = tl.load(
            w_ptr + cols[:, None] * stride_wn + ks[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    activated = tl.maximum(acc + bias[None, :], 0.0)
    tl.store(
        y_ptr + rows[:, None] * stride_ym + cols[None, :],
        activated.to(y_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def linear_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x [M,K], weight [N,K], bias [N]")
    if x.shape[1] != weight.shape[1] or weight.shape[0] != bias.shape[0]:
        raise ValueError("incompatible linear shapes")
    if x.device.type != "cuda":
        raise ValueError("this Triton lab requires CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("use float16 or bfloat16 to exercise Tensor Core tl.dot")
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("x and weight last dimensions must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    linear_relu_kernel[grid](
        x,
        weight,
        bias,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return out

