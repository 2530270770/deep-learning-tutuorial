# 从 PyTorch 到推理算子融合

## 面向 Qwen3-0.6B 与 nano-vLLM 的动手教程

这份教程不是“深度学习概论”，而是一条面向 AI Infra 推理引擎研发的专项路线：

```text
PyTorch 张量
    -> 自动求导与反向传播
    -> 线性层和激活函数
    -> Transformer FFN / SwiGLU
    -> Qwen3-0.6B 的 MLP 计算
    -> nano-vLLM 的 Prefill / Decode 执行链
    -> 算子融合、正确性验证与 Benchmark
    -> Nsight / Triton / CUDA 性能进阶
```

最终目标不是手写一个完整推理框架，而是能够做到：

1. 看到 `[T, H] -> [T, I] -> [T, H]` 时，立即说清每个维度代表什么、占多少显存。
2. 读懂 nano-vLLM 中 Qwen3 的 Attention、MLP、KV Cache 和 logits 计算。
3. 独立验证 ReLU、GELU、SiLU、SwiGLU 的前向与梯度。
4. 解释“合并两个线性层”“融合逐元素算子”“GEMM epilogue 融合”的区别。
5. 用 CUDA Event、`torch.profiler` 和 Nsight Systems 获得可复现的性能证据。
6. 从 `torch.compile` 自动融合过渡到 Triton，再过渡到 CUDA Kernel。

## 源码与配置基线

本教程的源码引用固定在 nano-vLLM 上游提交：

```text
commit: bb823b3e06983d71485a8e1f23715ebd87d98ef8
date:   2026-04-26
```

参考源码位于 [reference/nano-vllm](reference/nano-vllm)。它只用于阅读和对照，不覆盖你已有的 nano-vLLM 项目。

Qwen3-0.6B 的关键配置来自官方 `config.json`：

| 字段 | 数值 | 在本教程中的用途 |
|---|---:|---|
| `hidden_size` | 1024 | 每层主残差流宽度 |
| `intermediate_size` | 3072 | FFN 的 gate/up 宽度 |
| `num_hidden_layers` | 28 | 参数和 KV Cache 统计 |
| `num_attention_heads` | 16 | Q 头数 |
| `num_key_value_heads` | 8 | K/V 头数，GQA |
| `head_dim` | 128 | 单头宽度 |
| `vocab_size` | 151936 | logits 和词嵌入大小 |
| `hidden_act` | `silu` | 使用 SwiGLU 的门控组合 |
| `torch_dtype` | `bfloat16` | 推荐推理精度 |
| `tie_word_embeddings` | `true` | 输入嵌入与 LM Head 共享权重 |

## 适合谁

适合已经能写普通 Python、知道矩阵乘法，但还不熟悉 PyTorch 计算图和 GPU 性能测量的学习者。需要同步补齐的数学只有：

- 矩阵形状和矩阵乘法；
- 指数、对数、Sigmoid、链式法则；
- 平均值、平方根和误差；
- 能把公式逐项翻译成循环或张量运算。

不要求先学完整个机器学习课程，也不要求训练 Qwen3。全部核心实验都是小张量推理或 toy 反传，6 GB 显存足够。

## 课程目录

建议按顺序阅读，每章最后都有验收标准。

| 阶段 | 正文 | 核心实验 | 建议时间 |
|---|---|---|---:|
| 0 | [环境、精度与性能测量](tutorial/00_环境精度与性能测量.md) | `environment_check.py` | 2 小时 |
| 1 | [PyTorch 张量与形状](tutorial/01_PyTorch张量与形状.md) | `lab01_tensors.py` | 1 天 |
| 2 | [自动求导与反向传播](tutorial/02_自动求导与反向传播.md) | `lab02_autograd.py` | 1 天 |
| 3 | [线性层与激活函数](tutorial/03_线性层与激活函数.md) | `lab03_activations.py` | 2～3 天 |
| 4 | [Transformer FFN 与 Qwen3 SwiGLU](tutorial/04_Transformer_FFN与Qwen3_SwiGLU.md) | `lab04_qwen3_mlp.py` | 2 天 |
| 5 | [nano-vLLM 真实计算链](tutorial/05_nano-vLLM真实计算链.md) | `lab08_nanovllm_trace.py` | 3 天 |
| 6 | [算子融合原理与实践](tutorial/06_算子融合原理与实践.md) | `lab05_fusion_benchmark.py` | 3～4 天 |
| 7 | [Profile、Triton 与 CUDA 进阶](tutorial/07_Profile_Triton与CUDA进阶.md) | `lab06_mlp_fusion.py`、`lab07_real_qwen3.py` | 1 周 |
| 附录 | [公式与形状速查](tutorial/附录A_公式与形状速查.md) | — | 随时查阅 |

配套文件：

```text
labs/
  common.py                   # CUDA Event 计时和机器信息
  01_tensors.py
  02_autograd.py
  03_activations.py
  04_qwen3_mlp.py
  05_fusion_benchmark.py
  06_mlp_fusion.py
  07_real_qwen3.py
  08_nanovllm_trace.py
  09_linear_relu_fusion.py
  triton_silu_mul.py
tests/
  test_math_reference.py      # 无需 PyTorch 的数学和形状账本
  test_torch_labs.py          # 有 PyTorch 时自动运行
```

## WSL2 快速开始

下面的命令假设你已经在 WSL Ubuntu 中激活了装有 PyTorch 2.6.0+cu124 的 Python 3.11 环境。

```bash
cd /mnt/d/desktop/project/deeplearning

python --version
python environment_check.py

python labs/01_tensors.py
python labs/02_autograd.py
python labs/03_activations.py
python labs/04_qwen3_mlp.py
python labs/05_fusion_benchmark.py --tokens 1 8 32 128 512
python labs/09_linear_relu_fusion.py --tokens 1 8 32
```

建议明确设置模型路径，不把路径写死在代码里：

```bash
export QWEN3_PATH="$HOME/huggingface/Qwen3-0.6B"
export NANO_VLLM_ROOT="$HOME/nano-vllm"

python labs/07_real_qwen3.py --model "$QWEN3_PATH"
python labs/08_nanovllm_trace.py \
  --model "$QWEN3_PATH" \
  --nanovllm-root "$NANO_VLLM_ROOT"
```

运行标准库自检和 Python 语法检查：

```bash
python -m unittest tests/test_math_reference.py
python verify_course.py
```

## 6 GB 显存实验规则

RTX 3060 Laptop 6 GB 可以做全部核心学习任务，但要遵守以下约束：

1. 不在本地训练或微调 Qwen3-0.6B。
2. 模型使用 `bfloat16` 推理；Ampere 架构支持 BF16 Tensor Core。
3. nano-vLLM 的 baseline 实验先用 `enforce_eager=True`，再做 CUDA Graph 对比。
4. 首轮使用 `max_model_len=512` 或 `1024`、`max_num_batched_tokens=256`、`gpu_memory_utilization=0.70~0.80`。
5. 每次只改变一个变量；记录模型、源码 commit、dtype、序列长度、batch、预热和重复次数。
6. Profiling 使用短序列、小请求数，避免 trace 文件过大。

粗略显存账：

- 0.6B 参数以 BF16 存放大约 1.2 GB。
- KV Cache 与层数、上下文长度、KV 头数、头维度同时成正比。
- 中间激活、CUDA context、cuBLAS workspace、kernel 编译缓存、CUDA Graph pool 还要额外占空间。

## 真正需要掌握的三个形状

设本步实际参与计算的 token 数为 `T`。

### Qwen3 MLP

```text
x          [T, 1024]
gate_up    [T, 6144]      = x @ W_gate_up.T
gate       [T, 3072]
up         [T, 3072]
hidden     [T, 3072]      = SiLU(gate) * up
output     [T, 1024]      = hidden @ W_down.T
```

### Prefill

```text
输入 token：一个或多个请求拼接后的 total_tokens
x：         [total_tokens, 1024]
q：         [total_tokens, 16, 128]
k / v：     [total_tokens, 8, 128]
logits：    只取每条请求最后一个 token 后再做 LM Head
```

### Decode

```text
batch 中有 B 条请求，本步每条只生成 1 个 token
x：         [B, 1024]
q：         [B, 16, 128]
k / v：     [B, 8, 128]
logits：    [B, 151936]
```

## 学习闭环

每个知识点都按下面的顺序完成：

1. **先预测**：看代码前先写出形状、dtype 和大致字节数。
2. **再运行**：执行对应 lab，不只阅读代码。
3. **做反证**：故意改错维度、dtype 或 mask，观察报错。
4. **做测量**：性能结论必须来自同步后的 CUDA Event、Profiler 或 Nsight。
5. **写结论**：格式为“在什么配置下，哪个瓶颈，改了什么，正确性如何，性能变化多少”。

## 验收总分

| 项目 | 分值 | 及格证据 |
|---|---:|---|
| 基础与形状 | 20 | 能解释 view/copy、广播、`[T,H]` 和梯度形状 |
| 激活函数 | 20 | 手算 ReLU/GELU/SiLU/SwiGLU 的值和梯度 |
| Qwen3 / nano-vLLM | 25 | 画出真实 Prefill/Decode 调用链并标出形状 |
| 融合实验 | 25 | 正确性通过，基准可复现，能解释快或慢的原因 |
| 口头与报告 | 10 | 能用 5 分钟讲清一次优化而不只报“快了 2 倍” |

低于 80 分不建议直接进入 CUDA GEMM 优化，先补 Qwen3 MLP 和 Benchmark 两项。

## 最重要的边界

- `torch.compile` 是编译器，不保证某个版本一定生成你想象中的单 Kernel。
- 合并权重只减少一次 GEMM 启动和输入读取，不等于完全消除中间结果。
- 逐元素激活融合通常减少 HBM 读写和 launch；线性层融合还可能受 Tensor Core tile、布局和 epilogue 设计约束。
- Decode 的矩阵乘常有很小的 `M`，可能受权重读取和 launch 限制；Prefill 的较大 `M` 更可能进入计算密集区。
- benchmark 没有 warmup、没有同步或只测一次，通常不能支持任何优化结论。

