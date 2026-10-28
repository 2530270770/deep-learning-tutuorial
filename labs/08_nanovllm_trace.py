from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_PATH", os.path.expanduser("~/huggingface/Qwen3-0.6B")),
    )
    parser.add_argument(
        "--nanovllm-root",
        default=os.environ.get("NANO_VLLM_ROOT", ""),
    )
    parser.add_argument("--prompt", default="用一句话解释 PagedAttention。")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=256)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")
    if args.nanovllm_root:
        root = Path(args.nanovllm_root).expanduser().resolve()
        if not (root / "nanovllm").is_dir():
            raise SystemExit(f"Nano-vLLM package not found under: {root}")
        sys.path.insert(0, str(root))

    if not torch.cuda.is_available():
        raise SystemExit("nano-vLLM's Qwen3 runner requires CUDA.")

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        str(model_path),
        enforce_eager=not args.cuda_graph,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )

    model = llm.model_runner.model
    first_mlp = model.model.layers[0].mlp
    captured: dict[str, object] = {}

    def mlp_hook(module, inputs, output) -> None:
        if captured:
            return
        captured["input_shape"] = tuple(inputs[0].shape)
        captured["output_shape"] = tuple(output.shape)
        captured["dtype"] = str(output.dtype)

    handle = first_mlp.register_forward_hook(mlp_hook)
    original_run_model = llm.model_runner.run_model

    def run_model_with_nvtx(input_ids, positions, is_prefill):
        label = "prefill" if is_prefill else "decode"
        torch.cuda.nvtx.range_push(f"nanovllm:{label}")
        try:
            return original_run_model(input_ids, positions, is_prefill)
        finally:
            torch.cuda.nvtx.range_pop()

    llm.model_runner.run_model = run_model_with_nvtx
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        outputs = llm.generate(
            [args.prompt],
            SamplingParams(temperature=0.6, max_tokens=args.max_tokens, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        handle.remove()
        llm.model_runner.run_model = original_run_model

    elapsed = time.perf_counter() - start
    print("first MLP:", captured)
    print("output:", outputs[0]["text"])
    print("output token ids:", outputs[0]["token_ids"])
    print(f"elapsed: {elapsed:.3f} s")
    print(f"throughput: {args.max_tokens / elapsed:.2f} output tok/s")
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"peak reserved:  {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()

