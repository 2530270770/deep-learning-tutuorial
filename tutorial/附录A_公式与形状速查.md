# 附录 A 公式与形状速查

## A.1 PyTorch 基本形状

```text
bytes = numel × element_size
FP32   = 4 B
FP16   = 2 B
BF16   = 2 B
MiB    = bytes / 1024²
```

```text
线性层：
x [T,H] @ W.T [H,I] -> y [T,I]

view/reshape：
view 要求布局允许；reshape 可能复制

transpose：
只交换 shape/stride，通常变成非连续
```

## A.2 线性层反向

```text
Y = XWᵀ + b
X: [T,H]   W: [I,H]   Y: [T,I]

dW = dYᵀ @ X       [I,H]
db = sum_T dY      [I]
dX = dY @ W        [T,H]
```

## A.3 激活函数

```text
ReLU(x) = max(0,x)
ReLU'(x) = 1 if x>0, 0 if x<0

GELU(x) = 0.5x(1 + erf(x/√2))
GELU'(x) = Φ(x) + xφ(x)
Φ(x) = 0.5(1 + erf(x/√2))
φ(x) = exp(-x²/2)/√(2π)

SiLU(x) = xσ(x)
σ(x) = 1/(1+e^{-x})
SiLU'(x) = σ(x)(1 + x(1-σ(x)))

SwiGLU(a,b) = SiLU(a) × b
∂SwiGLU/∂a = b × SiLU'(a)
∂SwiGLU/∂b = SiLU(a)
```

固定数值：

| x | -2 | -1 | 0 | 1 | 2 |
|---:|---:|---:|---:|---:|---:|
| ReLU | 0 | 0 | 0 | 1 | 2 |
| GELU | -0.045500 | -0.158655 | 0 | 0.841345 | 1.954500 |
| SiLU | -0.238406 | -0.268941 | 0 | 0.731059 | 1.761594 |
| SiLU' | -0.090784 | 0.072329 | 0.5 | 0.927671 | 1.090784 |

## A.4 Qwen3-0.6B MLP

```text
H=1024, I=3072, layers=28

x        [T,H]
gate_up  [T,2I] = [T,6144]
gate     [T,I]  = [T,3072]
up       [T,I]  = [T,3072]
hidden   [T,I]  = SiLU(gate)*up
output   [T,H]  = hidden @ W_down.T
```

参数与权重：

```text
每层参数 = 3HI = 9,437,184
28 层参数 = 264,241,152
BF16 权重 = 528,482,304 B = 504 MiB

gate_up 每层权重 = 2HI，BF16 读取 = 12 MiB
down 每层权重 = HI，BF16 读取 = 6 MiB
```

## A.5 Attention / KV Cache

```text
Q heads=16, KV heads=8, head_dim=128
q width = 16×128 = 2048
k width = 8×128 = 1024
v width = 8×128 = 1024
qkv width = 4096

Prefill:
q [T,16,128]
k,v [T,8,128]

Decode:
q [B,16,128]
k_new,v_new [B,8,128]
context_lens [B]
block_tables [B,max_blocks]
```

每 token 全部 28 层的 KV 字节：

```text
2 × 28 × 8 × 128 × 2 = 114,688 B ≈ 112 KiB
```

这里没有计 block padding 和其他运行时开销。

## A.6 融合流量

设 `T` 个 token、中间宽度 `I`、dtype 字节数为 `b`：

```text
未融合 SiLU、Mul：
5 × T × I × b

融合 SiLU×up：
3 × T × I × b

节省：
2 × T × I × b
```

Qwen3 BF16 `I=3072`：

```text
每层每 token 的激活段理论节省 = 2×3072×2 = 12,288 B
T=32 每层节省 = 384 KiB
T=32 28 层节省 = 10.5 MiB
T=512 每层节省 = 6 MiB
T=512 28 层节省 = 168 MiB
```

这些只是点式 SiLU×up 的流量账，不含 GEMM、Attention 和 KV Cache。

## A.7 线性层算术强度

```text
FLOPs ≈ 2THI
Bytes ≈ (TH + HI + TI) × b
AI = FLOPs / Bytes
```

`T` 小时，`HI` 权重读取占主导；`T` 增大时，计算量增长更快。Prefill 和 Decode 不应共用同一个优化结论。

## A.8 Benchmark 模板

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(warmup):
    fn()

start.record()
for _ in range(reps):
    fn()
end.record()
torch.cuda.synchronize()
ms_per_call = start.elapsed_time(end) / reps
```

记录：GPU、PyTorch、CUDA、commit、dtype、shape、模式、warmup、repeats、median、p90、正确性误差。

## A.9 nano-vLLM 阅读顺序

```text
llm_engine.py
  -> scheduler.py
  -> model_runner.py
     -> qwen3.py
        -> attention.py
        -> activation.py
        -> linear.py
        -> embed_head.py
  -> sampler.py
```

核心字段：

| 字段 | 含义 |
|---|---|
| `cu_seqlens_q/k` | packed 序列边界 |
| `slot_mapping` | 当前 K/V 写入的物理槽位 |
| `context_lens` | 每条请求当前总上下文长度 |
| `block_tables` | 每条请求的逻辑 block 到物理 block 映射 |
| `num_scheduled_tokens` | 本步调度 token 数 |
| `is_prefill` | 本步是 Prefill 还是 Decode |

## A.10 常用命令

```bash
python environment_check.py
python -m unittest tests/test_math_reference.py
python verify_course.py

python labs/01_tensors.py
python labs/02_autograd.py
python labs/03_activations.py
python labs/04_qwen3_mlp.py
python labs/05_fusion_benchmark.py --tokens 1 8 32 128 512
python labs/06_mlp_fusion.py --tokens 1 16 64
python labs/09_linear_relu_fusion.py --tokens 1 8 32

TORCH_LOGS=output_code python labs/05_fusion_benchmark.py --tokens 32
nsys profile --trace=cuda,nvtx,osrt,cublas \
  -o outputs/nanovllm_trace python labs/08_nanovllm_trace.py
```

## A.11 常见错误快查

```text
OOM                       -> 看模型+KV+activate+workspace+graph
第一次慢                  -> 编译/算法选择/allocator，先用 warmup
数值不一致                -> gate/up 顺序、dtype、mask、容差、TP 分片
GPU 空档                  -> CPU 提交、同步点、batch 太小
Kernel 数很多             -> compile 图断裂、逐元素链、数据拷贝
融合不加速                -> 计算/寄存器/布局/非目标瓶颈
nsys 看到 CPU wait        -> 不是 GPU Kernel 慢，是提交或同步问题
```

## A.12 学习验收勾选

- [ ] 能解释 `[T,H]` 中 T 在 Prefill/Decode 的含义
- [ ] 能手算 ReLU、GELU、SiLU、SwiGLU 的值
- [ ] 能手算线性层反向形状
- [ ] 能手算 Qwen3 MLP 参数与激活字节
- [ ] 能画出 nano-vLLM 请求链路
- [ ] 能解释 KV Cache、block table、slot mapping
- [ ] 能用 CUDA Event 做同步基准
- [ ] 能运行 PyTorch、compile、Triton 融合对比
- [ ] 能用 Profiler 解释融合是否真的生效
- [ ] 能报告一次端到端 MLP 实验结果及负面结果

