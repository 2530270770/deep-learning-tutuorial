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


def eager_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.relu(F.linear(x, weight, bias))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("This comparison requires CUDA and Triton.")
    dtype = getattr(torch, args.dtype)
    print_machine_info(device)

    try:
        from labs.triton_linear_relu import linear_relu
    except Exception as exc:
        raise SystemExit(f"Triton import failed: {type(exc).__name__}: {exc}") from exc

    torch.manual_seed(11)
    weight = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) * 0.02
    bias = torch.randn(args.out_features, device=device, dtype=dtype) * 0.02
    compiled = (
        None
        if args.no_compile
        else torch.compile(lambda x: eager_linear_relu(x, weight, bias))
    )

    rows: list[dict[str, object]] = []
    started_all = time.perf_counter()
    for tokens in args.tokens:
        print_rule(f"tokens={tokens}")
        x = torch.randn(tokens, args.in_features, device=device, dtype=dtype)
        with torch.inference_mode():
            expected = eager_linear_relu(x, weight, bias)
            fused = linear_relu(x, weight, bias)
            compiled_out = compiled(x) if compiled is not None else None
        max_error_triton = test_close(fused, expected, dtype=dtype)
        max_error_compile = (
            test_close(compiled_out, expected, dtype=dtype)
            if compiled_out is not None
            else None
        )
        print(f"triton max_abs_error={max_error_triton:.6g}")
        if max_error_compile is not None:
            print(f"compile max_abs_error={max_error_compile:.6g}")

        # Compile/autotune every implementation before timing its shape.
        linear_relu(x, weight, bias)
        if compiled is not None:
            compiled(x)

        with torch.inference_mode():
            stats_eager = measure_ms(
                lambda: eager_linear_relu(x, weight, bias),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            stats_triton = measure_ms(
                lambda: linear_relu(x, weight, bias),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            stats_compile = (
                measure_ms(
                    lambda: compiled(x),
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                if compiled is not None
                else None
            )

        print(
            f"eager    median={stats_eager['median_ms']:.4f} ms "
            f"p90={stats_eager['p90_ms']:.4f} ms"
        )
        print(
            f"triton   median={stats_triton['median_ms']:.4f} ms "
            f"p90={stats_triton['p90_ms']:.4f} ms"
        )
        if stats_compile is not None:
            print(
                f"compiled median={stats_compile['median_ms']:.4f} ms "
                f"p90={stats_compile['p90_ms']:.4f} ms"
            )

        rows.append(
            {
                "tokens": tokens,
                "dtype": args.dtype,
                "in_features": args.in_features,
                "out_features": args.out_features,
                "eager_median_ms": stats_eager["median_ms"],
                "eager_p90_ms": stats_eager["p90_ms"],
                "triton_median_ms": stats_triton["median_ms"],
                "triton_p90_ms": stats_triton["p90_ms"],
                "compiled_median_ms": stats_compile["median_ms"] if stats_compile else "",
                "compiled_p90_ms": stats_compile["p90_ms"] if stats_compile else "",
                "triton_error": max_error_triton,
            }
        )

    Path("outputs").mkdir(exist_ok=True)
    csv_path = Path("outputs/lab09_linear_relu_metrics.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path.resolve()}")
    print(f"Correctness + compilation took {time.perf_counter() - started_all:.2f} s")
    print("A teaching Triton GEMM may be slower than cuBLAS; compare the whole Kernel, not only its epilogue.")


if __name__ == "__main__":
    main()

