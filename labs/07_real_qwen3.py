from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_PATH", os.path.expanduser("~/huggingface/Qwen3-0.6B")),
    )
    parser.add_argument("--prompt", default="用一句话解释 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_path}\n"
            "Set QWEN3_PATH or pass --model /path/to/Qwen3-0.6B."
        )
    if not torch.cuda.is_available():
        raise SystemExit("This lab expects a CUDA device.")

    device = torch.device("cuda")
    print("model:", model_path)
    print("GPU:", torch.cuda.get_device_name(device))

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    config = model.config
    print(
        "config:",
        {
            "layers": config.num_hidden_layers,
            "hidden": config.hidden_size,
            "intermediate": config.intermediate_size,
            "heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", None),
            "vocab": config.vocab_size,
            "dtype": str(model.dtype),
        },
    )

    captured: dict[str, object] = {}

    def mlp_hook(module, inputs, output) -> None:
        if captured:
            return
        captured["input_shape"] = tuple(inputs[0].shape)
        captured["output_shape"] = tuple(output.shape)
        captured["dtype"] = str(output.dtype)

    first_mlp = model.model.layers[0].mlp
    handle = first_mlp.register_forward_hook(mlp_hook)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof:
                    output = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=25,
                    )
                )
            else:
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
    finally:
        handle.remove()

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("first MLP:", captured)
    print("output:\n" + text)
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"peak reserved:  {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()

