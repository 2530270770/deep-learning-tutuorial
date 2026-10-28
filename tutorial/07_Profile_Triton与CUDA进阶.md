# 第 7 章 Profile、Triton 与 CUDA 进阶

## 7.1 本章目标与前置知识

本章把“我知道它应该更快”变成“我能证明它在哪里慢、改动减少了什么、是否值得”。完成后你应该能够：

- 用 `torch.profiler` 找到最大的 CUDA Kernel；
- 用 Nsight Systems 看 CPU launch、GPU 执行和空档；
- 用算术强度和 Roofline 判断先优化计算还是访存；
- 为 Qwen3 MLP 设计一个完整实验矩阵；
- 从 `torch.compile` 生成的 Triton 逐步过渡到手写 Triton；
- 说明何时应从 Triton 下沉到 CUDA/CUTLASS；
- 完成一个能写进项目记录的 MLP 融合小项目。

前置知识：第 6 章、CUDA Event 计时、基础矩阵乘。此时不需要已经写过高性能 CUDA Kernel。

## 7.2 性能分析的三层工具

| 工具 | 视角 | 回答的问题 | 本教程用途 |
|---|---|---|---|
| CUDA Event | 操作/代码区间 | 这段代码在 GPU 上耗时多久 | microbenchmark |
| `torch.profiler` | PyTorch 算子与 Kernel | 哪个算子、哪个 Kernel 最贵 | 快速定位 |
| Nsight Systems | 系统时间线 | CPU/GPU 谁在等，launch 是否密集 | 端到端引擎 |
| Nsight Compute | 单个 Kernel 内部 | 带宽、occupancy、stall、Tensor Core 利用率 | 深挖 Kernel |

先用低开销工具确认方向，再加入高开销工具。不要一开始就用 Nsight Compute 采集整个模型。

## 7.3 用 `torch.profiler`

```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    for _ in range(10):
        out = mlp(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=25,
))
```

重点字段：

- `Self CUDA Time Total`：Kernel 本身的时间；
- `Self CUDA Time Avg`：单次平均；
- `Calls`：调用次数；
- `Input Shapes`：是否意外出现其他 shape；
- `CUDA Mem`：粗略的分配活动。

分析时先找“最大的几项”，不要被几十个很小的 Kernel 分散注意力。真正的问题可能是：

- 一个 GEMM 占比 60%；
- Attention 占比 20%；
- 很多小点式 Kernel 合计占比 15%；
- 其余是采样、拷贝或空白。

不同回答对应不同优化方向。

## 7.4 Nsight Systems：看端到端时间线

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  -o outputs/nanovllm_profile \
  python labs/08_nanovllm_trace.py \
    --model "$QWEN3_PATH" \
    --nanovllm-root "$NANO_VLLM_ROOT"
```

打开 `.nsys-rep` 后按顺序检查：

1. NVTX 的 Prefill 和 Decode 区间；
2. CPU 是否长时间没有提交 Kernel；
3. GPU 是否有空档等待 CPU；
4. 一组小 Kernel 是否集中在很短的时间；
5. CUDA Graph replay 是否把大量 launch 合并；
6. GEMM、FlashAttention、RMSNorm、SiLU/mul 的比例。

对短模型可以加 `--sample=none` 减小 trace 体积；需要 CPU 调用栈时再用 sampling。

## 7.5 Nsight Compute：看单 Kernel 内部

先用 `torch.profiler` 或 nsys 找到 Kernel 名，再做定向采集：

```bash
ncu \
  --kernel-name regex:gemm \
  --launch-skip 10 \
  --launch-count 3 \
  --set basic \
  python labs/06_mlp_fusion.py --tokens 32 --dtype bfloat16
```

关键指标：

- DRAM Throughput；
- L2 Cache Hit Rate；
- SM Active / SM Busy；
- Warp Stall；
- Registers Per Thread；
- Shared Memory；
- Tensor Core Utilization / issue。

不要因为 occupancy 低就立即改 block size。如果 Kernel 是 memory-bound 且已经接近带宽，多 occupancy 不一定加速。先把“瓶颈资源”与“当前达成率”对上。

## 7.6 Roofline 的最小用法

设 Kernel 执行 `FLOPs`，最低移动 `Bytes`：

```text
arithmetic intensity = FLOPs / Bytes

计算时间下限 = FLOPs / peak_flops
访存时间下限 = Bytes / peak_bandwidth

理论上限 throughput = min(peak_flops,
                          peak_bandwidth × arithmetic_intensity)
```

真正的上限还要乘实际可达比例。RTX 3060 Laptop 不同功耗版本的频率和算力差异较大，不建议凭一张规格表断言精确 peak。可以做：

1. 用 PyTorch 大矩阵乘估计有效算力；
2. 用大 Tensor copy 估计有效带宽；
3. 再把目标 Kernel 与实测 baseline 比较。

示例：一个激活链移动约 6 MiB、计算只有少量乘加，算术强度低，应优先看带宽和 launch；一个较大 Prefill GEMM 的算术强度高，应看 Tensor Core 和 tile 调度。

## 7.7 推荐实验矩阵

固定：

```text
GPU: RTX 3060 Laptop 6GB
dtype: bfloat16
H=1024
I=3072
engine: eager / torch.compile / triton
warmup: 20
repeats: 100~200
```

变量：

```text
T ∈ {1, 2, 8, 16, 32, 64, 128, 512}
BLOCK ∈ {128, 256, 512, 1024}
mode ∈ {eager, compiled, triton}
```

对每一项保存：

```text
correctness max/mean error
median_ms
p90_ms
theoretical_bytes
estimated_effective_bandwidth
peak_allocated_MiB
compile_time_s
```

至少要能回答下面三个问题：

1. 小 batch 下 launch 是否占主导？
2. 大 batch 下融合流量收益是否更明显？
3. 完整 MLP 的收益为什么可能小于激活 microbenchmark？

## 7.8 实验一：激活链

```bash
python labs/05_fusion_benchmark.py \
  --tokens 1 8 32 128 512 \
  --intermediate 3072 \
  --dtype bfloat16
```

预期：

- 三条实现数值一致到容差；
- `compiled` 首次调用慢于后续；
- `T=1` 的微秒级测量噪声较大；
- 大 `T` 下融合的有效带宽和绝对节省更容易比较；
- 如果 Triton 比 Inductor 慢，先检查 BLOCK、mask 和 dtype 转换。

## 7.9 实验二：完整 MLP

```bash
python labs/06_mlp_fusion.py \
  --tokens 1 16 64 \
  --dtype bfloat16
```

完整 MLP 包含两次 GEMM。即使激活完全融合，它也只占总时间的一部分。Profiling 时应看到：

- GEMM；
- 激活融合 Kernel；
- 可能的 transpose/contiguous；
- 无自动梯度的少量 bookkeeping。

如果 compile 没有让 GEMM 和激活进入同一 Kernel，看到两/多个 Kernel 是正常的，不应脑补成“融合失败”。

## 7.9.1 实验三：Linear + bias + ReLU epilogue

```bash
python labs/09_linear_relu_fusion.py \
  --tokens 1 8 32 128 \
  --dtype bfloat16
```

`triton_linear_relu.py` 将 `[M,K] @ W.T` 分块累加，在同一个 Kernel 中加 bias、做 ReLU 并写入 `[M,N]`。它直接省略了 pre-activation `[M,N]` 的写回和再读取。

这个实验最重要的预期不是“自定义 Kernel 必然赢”，而是：

- 融合后的 epilogue 确实少了中间 Tensor 和 Kernel；
- 但 Triton 教学 Kernel 的 tile、流水和寄存器策略可能不如 cuBLAS；
- 只有完整 Kernel 时间、正确性和 Profiler 一起改善，才算真实优化；
- 这正是从“会融合”走向“会做高性能 GEMM”的分界线。

## 7.10 从 `torch.compile` 到 Triton

学习顺序：

1. 用 eager 写清楚的 reference；
2. 用 `torch.compile` 看自动优化结果；
3. 打开 `TORCH_LOGS=output_code` 阅读生成的 Triton；
4. 手写一个等价的逐元素 Triton Kernel；
5. 比较正确性和性能；
6. 再把注意力放到更难的 GEMM tile/epilogue。

Triton 的优势是 block-level 抽象，不必手管每个线程、shared memory swizzle 和 warp shuffle。对逐元素、Softmax、RMSNorm、短序列 Attention，它是很好的教学和原型工具。

## 7.11 什么情况需要进一步下沉到 CUDA/CUTLASS

Triton 不适合所有场景。需要 CUDA/CUTLASS 的常见情况：

- 需要极细致的 Tensor Core tile、warp specialization 或数据布局；
- 需要使用目标架构特有的指令或异步拷贝路径；
- 编译器和 autotune 后仍达不到目标；
- 需要把 GEMM epilogue 与非常复杂的 fused computation 组合；
- Kernel 有多个 stage，依赖 shared memory pipeline、cluster 或特殊同步；
- 需要确定性的寄存器/共享内存布局。

不是“Triton 慢就一定要手写 CUDA”。先看 SASS/Kernel 指标和瓶颈；有时只是 shape、BLOCK 或数据布局问题。

## 7.12 CUDA 的下一步学习顺序

你已经有 CUDA 基础方向，建议从推理算子需求倒序补：

1. CUDA event 和 stream 同步；
2. `grid-stride` 与 coalesced load/store；
3. 一维逐元素 Kernel；
4. 按行 reduce：mean、sum、max；
5. Softmax 与 RMSNorm；
6. shared memory GEMM；
7. packed 低精度输入和 Tensor Core；
8. epilogue 融合；
9. CUDA Graph 和多 stream；
10. 回到 Qwen3 MLP 做端到端验证。

每步都用“正确性、实测带宽/算力、Profiler 证据”三件套验收。

## 7.13 综合项目：优化 Qwen3 MLP

### 目标

在 RTX 3060 Laptop 6GB 上，针对 Qwen3-0.6B 的 MLP，写一个可复现实验报告，回答：

> 合并 gate/up、融合 SiLU×up 或尝试 GEMM epilogue，分别减少了哪些工作？在 Prefill 与 Decode 的哪个形状获益更多？

### 阶段 A：建立 reference

- 实现 separate gate/up + eager SiLU×up；
- 固定 seed；
- 用 `torch.testing.assert_close` 与合并权重版本比较；
- 保存 FP32 小 shape 和 BF16 真实 shape 的误差。

### 阶段 B：microbenchmark

- 测激活链的 eager、compile、Triton；
- `T=1,8,32,128,512`；
- 记录 median/p90、理论字节和有效带宽；
- 每个 shape 单独预热，避免编译污染。

### 阶段 C：完整 MLP

- 测 separate、merged eager、merged compiled；
- Profile 两次 GEMM 与激活链；
- 比较 T 小时的 launch 与权重读取；
- 比较 T 大时的 GEMM 与激活占比。

### 阶段 D：真实引擎

- 使用 nano-vLLM Qwen3-0.6B；
- 先 `enforce_eager=True` 验证正确性；
- 再测短 prompt；不追求长上下文大规模压测；
- 记录 TTFT、TPOT、吞吐和峰值显存；
- 用 NVTX 看 Prefill/Decode 比例。

### 阶段 E：报告

报告至少包含：

1. 实验问题与假设；
2. 源码调用链；
3. 形状和字节账；
4. 正确性证据；
5. 性能表格和图表；
6. 哪些收益来自减少 launch，哪些来自减少 HBM；
7. 哪些情况下没有变快；
8. 下一项优化建议。

## 7.14 一个合格的结论怎么写

弱结论：

```text
我用了算子融合，速度提升了 2 倍。
```

合格结论：

```text
在 RTX 3060 Laptop、BF16、Qwen3 MLP 形状 T=32,I=3072 下，
把 SiLU 和逐元素乘法从两条 Eager 点式 Kernel 融合为一条 Triton Kernel，
最大绝对误差为 x，median 从 a ms 降到 b ms，p90 从 c ms 降到 d ms。
理论点式流量从 5TI 个元素降到 3TI，P95 没有明显变差。
完整 MLP 的端到端收益为 e%，说明 GEMM 仍是主要瓶颈。
```

必须说明适用范围，不能把一次 small-shape 结果外推成所有 GPU、dtype 和 batch。

## 7.15 常见故障定位

| 症状 | 优先检查 |
|---|---|
| 数值不一致 | `chunk` 顺序、dtype、mask、FP32 提升、容差 |
| 第一次很慢 | JIT/编译、首次 cuBLAS 算法、allocator、warmup |
| GPU 占用低 | CPU 提交、同步点、batch 太小、Graph 未用 |
| Kernel 很多但每个很短 | Python 操作、编译融合不足、`chunk`/index 导致图断裂 |
| 显存持续上涨 | 中间结果被引用、profiler/trace buffer、未清缓存 |
| Triton 比 PyTorch 慢 | block size、grid、布局、FP32 转换、缺少满带宽访问 |
| 编译后仍无收益 | 目标 Kernel 不是时间大头，或 GEMM 未 epilogue 融合 |
| Nsight 里看到大量 memcpy | CPU→GPU 准备、固定缓冲区、checkpoint/采样同步 |

## 7.16 与后续 CUDA 学习的衔接

完成本章后，可以把下一项工作定义为：

```text
用 CUDA 写一个 RMSNorm + residual 融合 Kernel，
与 PyTorch reference 比较正确性，
再用 CUDA Event 和 Nsight Compute 分析带宽与 occupancy。
```

这比直接挑战完整 GEMM + SwiGLU 更容易形成可验证的闭环，也与 nano-vLLM 的 `RMSNorm.add_rms_forward` 直接对应。

## 7.17 练习

1. 用 `torch.profiler` 记录 eager 和 compiled MLP 的 Kernel 数，列出前三项。
2. 在 nsys 中找一次 Prefill 和一次 Decode，比较 Kernel launch 密度。
3. 用理论字节计算 `T=1,32,512` 的激活链有效带宽。
4. 把 Triton BLOCK 改为 128/256/512，保存结果并解释不同。
5. 让 kernel 处理非连续输入，记录错误和修复方法。
6. 用 Nsight Compute 比较两个 Kernel 的 DRAM throughput。
7. 写出本项目的一页实验报告，至少包含一个“没有变快”的负结果。

## 7.18 最终验收

能够独立做到以下四点，才算真正完成本教程：

1. 从输入形状推出 Qwen3 Attention/MLP/KV Cache 的主要形状和字节；
2. 从公式推出激活函数的前向与梯度，并能用 PyTorch 验证；
3. 从源码追到 nano-vLLM 的 Prefill/Decode 与 Sampler；
4. 用可复现实验证明一次融合的收益来源，同时说明它的边界和代价。

如果还能把结果写成一页包含 baseline、误差、median/p90、Profiler 和下一步的工程报告，就已经具备进入 CUDA/Triton 算子研发实践的坚实基础。

