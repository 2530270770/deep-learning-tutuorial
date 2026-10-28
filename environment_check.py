from __future__ import annotations

import importlib.util
import shutil
import sys

import torch

from labs.common import print_machine_info, resolve_device


def dependency(name: str) -> str:
    return "available" if importlib.util.find_spec(name) else "missing"


def main() -> None:
    print("Learning environment")
    print(f"executable: {sys.executable}")
    print(f"python:     {sys.version.split()[0]}")
    print(f"pytorch:    {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda ready: {torch.cuda.is_available()}")
    print(f"nvcc:       {shutil.which('nvcc') or 'not on PATH'}")
    print(f"nsys:       {shutil.which('nsys') or 'not on PATH'}")
    print(f"ncu:        {shutil.which('ncu') or 'not on PATH'}")
    print(f"triton:     {dependency('triton')}")
    print(f"transformers:{dependency('transformers')}")
    print(f"flash_attn: {dependency('flash_attn')}")
    print(f"xxhash:     {dependency('xxhash')}")

    device = resolve_device()
    print_machine_info(device)

    if device.type != "cuda":
        print("\nWARNING: CUDA is unavailable. Numerical labs can run, but GPU benchmarks cannot.")
        return

    props = torch.cuda.get_device_properties(device)
    supports_bf16 = props.major >= 8
    print(f"BF16 support (compute capability >= 8.0): {supports_bf16}")

    pixels = torch.ones(8, 8, device=device, dtype=torch.bfloat16)
    print(f"BF16 tensor smoke test: {pixels.sum().item()}")

    # Qwen3-0.6B, BF16, TP=1, 1024-token context.
    per_token_kv = 2 * 28 * 8 * 128 * 2
    print(f"Qwen3 KV per token: {per_token_kv / 1024:.2f} KiB")
    print(f"Qwen3 KV at 1024 tokens: {per_token_kv * 1024 / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()

