# 第 4 章 Transformer FFN 与 Qwen3 SwiGLU

## 4.1 本章目标与前置知识

本章把前面的线性层和激活函数组合成真实 FFN，并逐行对应 Qwen3-0.6B：

- 理解 Transformer Decoder Block 中 Attention 与 MLP 的分工；
- 掌握普通 FFN、GeGLU、SwiGLU 的形状和参数量；
- 推导 Qwen3 gate/up 合并后的权重映射；
- 手算一个微型 SwiGLU；
- 算清 Qwen3-0.6B 每层和全模型的 MLP 参数、激活字节；
- 理解 Prefill 与 Decode 为什么执行同一个 MLP，但形状完全不同。

前置知识：线性层、SiLU、矩阵乘、第 1 章的形状记账。

运行：

```bash
python labs/04_qwen3_mlp.py
```

## 4.2 Decoder Block 的位置

简化后的 Qwen3 Decoder Layer：

```text
x ──────────────────────────────┐
│                               │
RMSNorm -> Attention -> o_proj ├─ + 残差
│                               │
RMSNorm -> MLP -> down_proj ──── + 残差
```

Attention 做两件事：

1. 用 Q/K 计算 token 之间的关系；
2. 用权重聚合 V，把上下文信息写回 hidden 表示。

MLP 对每个 token 独立做非线性变换。它不在 token 之间直接交换信息，但因为 Attention 已经把上下文混合到每个 token 的 hidden 中，MLP 可以对这个上下文表示继续加工。

在 Qwen3-0.6B 中，MLP 参数量约为 2.64 亿，占总 6 亿参数的大头。这也是推理 Kernel 优化的重点之一。

## 4.3 普通 FFN

经典 Transformer FFN：

```text
h = GELU(xW₁ᵀ + b₁)
y = hW₂ᵀ + b₂
```

形状：

```text
x: [T, H]
h: [T, I]
y: [T, H]
```

常设 `I=4H`。无 bias 时参数量：

```text
H×I + I×H = 2HI = 8H²
```

若 `H=1024`、`I=4096`，每层约 `8 × 1024² = 8,388,608` 个参数。

## 4.4 GeGLU 与 SwiGLU

GeGLU：

```text
h = GELU(xW_gateᵀ) × (xW_upᵀ)
y = hW_downᵀ
```

SwiGLU：

```text
gate = xW_gateᵀ
up   = xW_upᵀ
h    = SiLU(gate) × up
y    = hW_downᵀ
```

Qwen3 的 `hidden_act` 是 `silu`，走 SwiGLU 路径。

SwiGLU 中的“门”不是二值开关，而是用 SiLU 产生连续门控值：

- `gate` 的每个位置决定 `up` 的对应位置通过多少；
- 两者逐元素相乘，所以形状必须完全相同；
- 相比单一路径 FFN，多了一条 gate 投影，但通常使用比 `4H` 更小的中间宽度。

Qwen3-0.6B 取 `I=3072=3H`。无 bias 参数量：

```text
gate: H×I
up:   H×I
down: I×H
合计: 3HI = 3 × 1024 × 3072 = 9,437,184
```

因此虽然“中间宽度看起来从 4H 降到 3H”，由于有三条矩阵，参数量仍大于经典 `2×H×4H=8H²`。

## 4.5 一个可手算的微型 SwiGLU

设：

```text
x = [1, 2]

W_gate = [[1, 0],
          [0, 1],
          [1, 1]]

W_up = [[1, 1],
        [2, 0],
        [1, 1]]

W_down = [[1, 0, 1],
          [0, 1, 0]]
```

### 第一步：gate 和 up

```text
gate = xW_gateᵀ = [1, 2, 3]
up   = xW_upᵀ   = [3, 2, 3]
```

### 第二步：门控

```text
SiLU(gate) = [
  1×σ(1),
  2×σ(2),
  3×σ(3)
] ≈ [0.731059, 1.761594, 2.857722]

h = SiLU(gate) × up
  ≈ [0.731059×3, 1.761594×2, 2.857722×3]
  ≈ [2.193177, 3.523188, 8.573166]
```

### 第三步：down 投影

```text
y[0] = 2.193177×1 + 3.523188×0 + 8.573166×1 = 10.766343
y[1] = 2.193177×0 + 3.523188×1 + 8.573166×0 = 3.523188
```

所以：

```text
output ≈ [10.766343, 3.523188]
```

这个手工例子完整复现了 `x -> gate/up -> SiLU×up -> down` 的四个阶段，下一章会把这个例子映射到 nano-vLLM。

## 4.6 Qwen3 实际形状

单个 token（`T=1`）：

```text
x           [1, 1024]
gate_up     [1, 6144]
gate        [1, 3072]
up          [1, 3072]
hidden      [1, 3072]
output      [1, 1024]
```

一个 batch 或 packed 序列中的 `T` 个 token：

```text
x           [T, 1024]
gate_up     [T, 6144]
hidden      [T, 3072]
output      [T, 1024]
```

### BF16 激活字节

对 `T=2`：

```text
x       = 2×1024×2 = 4,096 B
gate_up = 2×6144×2 = 24,576 B
gate    = 2×3072×2 = 12,288 B
up      = 2×3072×2 = 12,288 B
hidden  = 2×3072×2 = 12,288 B
output  = 2×1024×2 = 4,096 B
```

注意 `gate` 和 `up` 是 `gate_up` 的两个 chunk，通常共享底层存储，不应重复计算一份新的持久内存。但一旦执行 `SiLU(gate)` 和乘法，就会产生新的中间输出。

## 4.7 nano-vLLM 如何构造 MLP

参考源码：

- [models/qwen3.py](../reference/nano-vllm/nanovllm/models/qwen3.py)
- [layers/activation.py](../reference/nano-vllm/nanovllm/layers/activation.py)
- [layers/linear.py](../reference/nano-vllm/nanovllm/layers/linear.py)

核心初始化：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
)
self.act_fn = SiluAndMul()
```

核心前向：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
```

`SiluAndMul` 的源码：

```python
@torch.compile
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

这里有三个重要事实：

1. `gate_up_proj` 是一个宽度为 `2I=6144` 的线性层，不是两个完全独立启动的 Python 线性层；
2. `chunk` 是 view，不是把权重或激活复制一份；
3. `torch.compile` 有机会把 `F.silu(x) * y` 合成一个 Kernel，但必须以导出代码或 Profiler 为证据。

## 4.8 权重合并的数学等价性

独立写法：

```python
gate = F.linear(x, W_gate)
up = F.linear(x, W_up)
hidden = F.silu(gate) * up
```

合并写法：

```python
W_gate_up = torch.cat([W_gate, W_up], dim=0)  # [2I, H]
gate_up = F.linear(x, W_gate_up)              # [T, 2I]
gate, up = gate_up.chunk(2, dim=-1)
hidden = F.silu(gate) * up
```

从数学上：

```text
xW_gateᵀ 和 xW_upᵀ 的结果按列拼接
```

与：

```text
x [W_gate; W_up]ᵀ
```

严格等价。

从工程上：

- 只启动一次 GEMM；
- 输入 `x` 只需为这次 GEMM 读取一遍；
- 权重总字节不变；
- 但输出了完整的 `[T, 6144]`，随后激活仍需读取它；
- 两个独立 GEMM 可能有不同的库调用和并行度，合并后的单次 GEMM 未必在所有 shape 上更快，必须实测。

### 为什么权重能这样加载

`packed_modules_mapping` 把 Hugging Face 模型中的名字映射到 nano-vLLM 的合并参数：

```python
"gate_proj": ("gate_up_proj", 0),
"up_proj":   ("gate_up_proj", 1),
```

加载时把原模型 `gate_proj.weight` 写到合并权重的第 0 段，把 `up_proj.weight` 写到第 1 段。分段使用 `MergedColumnParallelLinear.weight_loader()`。

### 常见错误

- 把 `[W_gate; W_up]` 与 `[W_gate, W_up]` 混淆：前者沿输出维拼接，后者沿输入维拼接；
- `chunk` 后把 gate/up 顺序颠倒；
- 合并权重后忘掉 bias 也必须按段处理；
- Tensor Parallel 下按全局 `I` 分片，而不是每个 rank 的局部 `I`；
- 只看权重拼接数学，忽略 GEMM 后仍是 `[T,2I]` 的中间激活。

## 4.9 残差与 RMSNorm：为什么 MLP 输入不是 Plain Linear

Qwen3 每层在 Attention 和 MLP 前都有 RMSNorm。RMSNorm 不减去均值：

```text
rms(x) = sqrt(mean(x²) + eps)
RMSNorm(x) = x / rms(x) × weight
```

每个 token 的 hidden 向量独立归一化，所以输入 `[T,H]` 输出仍为 `[T,H]`。

nano-vLLM 还用了一种重要的残差写法：

```text
(normalized_input, new_residual)
    = RMSNorm(old_input, old_residual)
```

内部先把 `old_input + old_residual` 作为新残差，再归一化。MLP 的 `down_proj` 输出这个残差分支，在下一层再合并。

与逐层写：

```python
residual = residual + hidden
normed = rms_norm(residual)
```

数学相同，但融合后的实现可以减少一次显式残差写入和带回 HBM 的中间结果。nano-vLLM 的 `add_rms_forward()` 就是这个思路。

## 4.10 Prefill 和 Decode 的 MLP 差异

### Prefill

若本步调度了 512 个新 token：

```text
x:      [512, 1024]
gate_up:[512, 6144]
```

大 `T` 给 GEMM 更多行，Tensor Core 更容易充分利用，计算量/参数读取的比值更高。

### Decode

若一个 batch 有 32 条请求，每条生成本步 1 个 token：

```text
x:      [32, 1024]
gate_up:[32, 6144]
```

`T=32` 比 Prefill 小，但每个 decode step 都要重新读取整层权重。权重读取和 Kernel launch 的占比上升。

这不是“Decode 的 FLOPs 更便宜所以一定快”：Decode 的串行性质和权重带宽限制了单步速度。

## 4.11 每层和整模型参数账

Qwen3-0.6B MLP：

```text
每层参数 = 3 × 1024 × 3072
         = 9,437,184

28 层参数 = 264,241,152
BF16 字节 = 528,482,304 B
           = 504 MiB
```

这部分只是 MLP 权重，不含 Attention、Embedding、Norm 和可能的共享 LM Head。整个 0.6B 模型 BF16 参数约 1.2 GB。

## 4.12 本章实验会验证什么

`labs/04_qwen3_mlp.py`：

1. 用 tiny 维度复现手算 SwiGLU；
2. 比较“独立 gate/up”与“合并 gate_up”的输出；
3. 用 Qwen3-0.6B 实际维度检查 `[T,1024]` 到 `[T,6144]` 的形状；
4. 统计 MLP 参数和 BF16 权重字节；
5. 打印 `T=1,8,32` 时的主要激活元素数。

预期关键断言：

```text
merged output close to separate output: PASS
gate_up.shape == [T, 6144]: PASS
mlp params == 9,437,184: PASS
```

## 4.13 常见错误

| 错误 | 后果 | 检查 |
|---|---|---|
| `gate` 和 `up` 顺序弄反 | 模型输出错误但不一定报错 | 用固定 seed 比较 HF/nano 输出 |
| `down_proj` 输入宽度写成 6144 | 形状错误或错误权重 | down 输入应为 3072 |
| 把 SiLU 应用于整个 6144 宽度 | 语义不同 | 必须 chunk 后只作用于 gate |
| 把 `chunk` 当成复制 | 低估额外内存/高估收益 | 看 storage_offset 和 data_ptr |
| 使用 FP16 导致溢出 | 大 logits 得到 inf/nan | Qwen3 优先 BF16 |
| 直接比较大批次与单请求 | 结论不可比 | 固定 T、dtype、engine 模式和 warmup |

## 4.14 练习

1. 用给定的微型权重复算 `[10.766343, 3.523188]`。
2. 证明 `cat([xW1.T, xW2.T], dim=-1) = x @ cat([W1,W2],dim=0).T`。
3. 计算 Qwen3 MLP 在 `T=128` 时 gate/up/merged hidden 的 BF16 元素数与字节。
4. 如果 `I=4H` 且使用 SwiGLU，每层 MLP 参数量是多少个 `H²`？
5. 为什么 Decode 中 MLP 的 `T` 是 batch 数（在本步每请求一个 token 的情况下）？
6. 用 `torch.profiler` 观察 tiny MLP，找出哪些 Kernel 可能在融合后减少。

## 4.15 验收标准

- 不查资料画出 Qwen3 MLP 数据流；
- 写出实际 shape、参数和 BF16 字节；
- 能解释合并 gate/up 与融合 SiLU/mul 的不同；
- 能说明残差和 RMSNorm 在这一层中的作用；
- 能通过 tiny SwiGLU 的等价性实验。

