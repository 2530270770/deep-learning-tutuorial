from __future__ import annotations

import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MachineInfo:
    python: str
    pytorch: str
    torch_cuda: str | None
    gpu: str
    capability: str | None
    total_memory_mib: float | None


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 100,
) -> dict[str, float]:
    """Measure individual calls; compilation must happen before this function."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")

    for _ in range(warmup):
        fn()
    synchronize(device)

    if device.type == "cuda":
        starts: list[torch.cuda.Event] = []
        ends: list[torch.cuda.Event] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            starts.append(start)
            ends.append(end)
        synchronize(device)
        times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    else:
        times_ms = []
        for _ in range(repeats):
            start_ns = time.perf_counter_ns()
            fn()
            times_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
        times = times_ms

    ordered = sorted(times)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
    }


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    """Actual storage backing a tensor, including a possible larger base."""
    return tensor.untyped_storage().nbytes()


def tensor_logical_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def machine_info(device: torch.device | None = None) -> MachineInfo:
    device = device or resolve_device()
    gpu = "cpu only"
    capability = None
    total_memory_mib = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = props.name
        capability = f"{props.major}.{props.minor}"
        total_memory_mib = props.total_memory / 2**20
    return MachineInfo(
        python=platform.python_version(),
        pytorch=torch.__version__,
        torch_cuda=torch.version.cuda,
        gpu=gpu,
        capability=capability,
        total_memory_mib=total_memory_mib,
    )


def print_machine_info(device: torch.device) -> None:
    info = machine_info(device)
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2))


def print_rule(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def test_close(actual: torch.Tensor, expected: torch.Tensor, *, dtype: torch.dtype) -> float:
    reference = expected.float()
    error = (actual.float() - reference).abs()
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 2e-2, 2e-2
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(actual.float(), reference, rtol=rtol, atol=atol)
    return error.max().item()

