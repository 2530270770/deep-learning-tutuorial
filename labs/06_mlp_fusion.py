from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import measure_ms, print_machine_info, print_rule, resolve_device, test_close


class Qwen3MLPReference(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.down.weight)


def profile_once(fn, x: torch.Tensor) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
    sort_by = "self_cuda_time_total" if x.device.type == "cuda" else "self_cpu_time_total"
    print(
        prof.key_averages().table(
            sort_by=sort_by,
            row_limit=20,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    print_machine_info(device)
    torch.manual_seed(7)
    model = Qwen3MLPReference(args.hidden, args.intermediate).to(device=device, dtype=dtype).eval()
    compiled = None if args.no_compile else torch.compile(model)

    print_rule("parameter accounting")
    params = sum(p.numel() for p in model.parameters())
    print("MLP parameters:", f"{params:,}")
    print("weight storage MiB:", params * torch.empty((), dtype=dtype).element_size() / 2**20)

    rows: list[dict[str, object]] = []
    start_all = time.perf_counter()
    for tokens in args.tokens:
        print_rule(f"MLP tokens={tokens}")
        x = torch.randn(tokens, args.hidden, device=device, dtype=dtype)
        with torch.inference_mode():
            expected = model(x)
            actual_compiled = compiled(x) if compiled is not None else None
        if actual_compiled is not None:
            error = test_close(actual_compiled, expected, dtype=dtype)
            print(f"compiled max_abs_error={error:.6g}")
            compiled(x)

        with torch.inference_mode():
            stats_eager = measure_ms(
                lambda: model(x),
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
        if stats_compile is not None:
            speedup = stats_eager["median_ms"] / stats_compile["median_ms"]
            print(
                f"compiled median={stats_compile['median_ms']:.4f} ms "
                f"p90={stats_compile['p90_ms']:.4f} ms speedup={speedup:.3f}x"
            )

        if args.profile:
            print_rule(f"profile tokens={tokens}")
            profile_once(compiled or model, x)

        rows.append(
            {
                "tokens": tokens,
                "dtype": args.dtype,
                "eager_median_ms": stats_eager["median_ms"],
                "eager_p90_ms": stats_eager["p90_ms"],
                "compiled_median_ms": stats_compile["median_ms"] if stats_compile else "",
                "compiled_p90_ms": stats_compile["p90_ms"] if stats_compile else "",
                "speedup_median": (
                    stats_eager["median_ms"] / stats_compile["median_ms"]
                    if stats_compile
                    else ""
                ),
            }
        )

    Path("outputs").mkdir(exist_ok=True)
    csv_path = Path("outputs/lab06_mlp_metrics.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path.resolve()}")
    print(f"Correctness + all compile/autotune work took {time.perf_counter() - start_all:.2f} s")
    print("A full-MLP speedup does not have to equal the activation-chain speedup.")


if __name__ == "__main__":
    main()

