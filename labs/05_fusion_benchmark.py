from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import measure_ms, print_machine_info, print_rule, resolve_device, test_close


def eager_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-triton", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    print_machine_info(device)
    print(f"dtype={dtype}, intermediate={args.intermediate}, triton_block={args.block}")

    use_compile = not args.no_compile
    use_triton = not args.no_triton and device.type == "cuda"
    compiled_fn = torch.compile(eager_silu_mul) if use_compile else None
    triton_fn = None
    if use_triton:
        try:
            from labs.triton_silu_mul import triton_silu_mul

            triton_fn = triton_silu_mul
        except Exception as exc:
            print(f"Triton unavailable, skipped: {type(exc).__name__}: {exc}")

    compile_start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for tokens in args.tokens:
        print_rule(f"tokens={tokens}")
        torch.manual_seed(1)
        gate_up = torch.randn(tokens, 2 * args.intermediate, device=device, dtype=dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        expected = eager_silu_mul(gate_up)

        compiled = None
        if compiled_fn is not None:
            compiled = compiled_fn(gate_up)
            max_error_compile = test_close(compiled, expected, dtype=dtype)
            print(f"compiled max_abs_error={max_error_compile:.6g}")

        fused = None
        max_error_triton = None
        if triton_fn is not None:
            fused = triton_fn(gate, up, block_i=args.block)
            max_error_triton = test_close(fused, expected, dtype=dtype)
            print(f"triton   max_abs_error={max_error_triton:.6g}")

        # Compile every shape before timing it.
        if compiled_fn is not None:
            compiled_fn(gate_up)
        if triton_fn is not None:
            triton_fn(gate, up, block_i=args.block)

        stats_eager = measure_ms(
            lambda: eager_silu_mul(gate_up),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        stats_compile = (
            measure_ms(
                lambda: compiled_fn(gate_up),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if compiled_fn is not None
            else None
        )
        stats_triton = (
            measure_ms(
                lambda: triton_fn(gate, up, block_i=args.block),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if triton_fn is not None
            else None
        )

        n = tokens * args.intermediate
        bytes_per_element = torch.empty((), dtype=dtype).element_size()
        bytes_eager = 5 * n * bytes_per_element
        bytes_fused = 3 * n * bytes_per_element
        bw_eager = bytes_eager / (stats_eager["median_ms"] * 1e-3) / 1e9
        bw_compile = (
            bytes_fused / (stats_compile["median_ms"] * 1e-3) / 1e9
            if stats_compile is not None
            else None
        )
        bw_triton = (
            bytes_fused / (stats_triton["median_ms"] * 1e-3) / 1e9
            if stats_triton is not None
            else None
        )

        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}"

        print(
            f"eager    median={stats_eager['median_ms']:.4f} ms "
            f"p90={stats_eager['p90_ms']:.4f} ms effective={bw_eager:.1f} GB/s"
        )
        if stats_compile is not None:
            print(
                f"compiled median={stats_compile['median_ms']:.4f} ms "
                f"p90={stats_compile['p90_ms']:.4f} ms effective={bw_compile:.1f} GB/s"
            )
        if stats_triton is not None:
            print(
                f"triton   median={stats_triton['median_ms']:.4f} ms "
                f"p90={stats_triton['p90_ms']:.4f} ms effective={bw_triton:.1f} GB/s"
            )

        rows.append(
            {
                "tokens": tokens,
                "intermediate": args.intermediate,
                "dtype": args.dtype,
                "eager_median_ms": stats_eager["median_ms"],
                "eager_p90_ms": stats_eager["p90_ms"],
                "compiled_median_ms": stats_compile["median_ms"] if stats_compile else "",
                "compiled_p90_ms": stats_compile["p90_ms"] if stats_compile else "",
                "triton_median_ms": stats_triton["median_ms"] if stats_triton else "",
                "triton_p90_ms": stats_triton["p90_ms"] if stats_triton else "",
                "eager_effective_gbps": bw_eager,
                "compiled_effective_gbps": bw_compile if bw_compile is not None else "",
                "triton_effective_gbps": bw_triton if bw_triton is not None else "",
                "compile_error": max_error_compile if compiled is not None else "",
                "triton_error": max_error_triton if max_error_triton is not None else "",
            }
        )

    compile_elapsed = time.perf_counter() - compile_start
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    csv_path = output / "lab05_fusion_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path.resolve()}")
    print(f"Correctness + all compile/autotune work took {compile_elapsed:.2f} s")
    print("Interpret absolute times only together with dtype, shape and machine information.")


if __name__ == "__main__":
    main()

