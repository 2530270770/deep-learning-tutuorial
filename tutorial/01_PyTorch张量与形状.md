# 第 1 章 PyTorch 张量与形状

## 1.1 本章目标与前置知识

本章只学推理引擎源码中最常用的 PyTorch 能力：

- 创建、索引、切片、广播；
- 理解 `shape`、`stride`、连续内存；
- 区分 `view`、`reshape`、`transpose`、`contiguous`；
- 读懂 `split`、`chunk`、`view`、`flatten`；
- 为 `[T, H]`、`[T, N, D]` 和批量矩阵乘建立直觉。

前置知识是矩阵的行列、Python 列表和基本算术。

运行：

```bash
python labs/01_tensors.py
```

## 1.2 Tensor 是“数据 + 形状 + 步长 + dtype + device”

创建一个二维张量：

```python
import torch

x = torch.arange(12, dtype=torch.float32)
x = x.view(3, 4)
print(x)
print("shape:", x.shape)
print("stride:", x.stride())
```

预期输出：

```text
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
shape: torch.Size([3, 4])
stride: (4, 1)
```

`stride=(4, 1)` 表示：

- 行号加 1，内存地址跳过 4 个元素；
- 列号加 1，内存地址跳过 1 个元素。

形状描述“逻辑上看到什么”，stride 描述“底层怎样找到元素”。两个形状相同的 Tensor 不一定是同样的内存布局。

### 常见错误

- 把 `shape` 当成总字节数：还要乘 dtype 的 `element_size()`；
- 看到 `[T, H]` 就以为是 `[batch, hidden]`：在 nano-vLLM 里 `T` 常是拼接后的 token 总数；
- 只检查数值，不检查 stride 和 device，导致后续 Kernel 要求连续内存时报错。

## 1.3 dtype、device 与字节账

```python
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.empty(2, 3, dtype=dtype)
    print(dtype, x.element_size(), x.numel(), x.nbytes)
```

预期：

```text
torch.float32 4 6 24
torch.float16 2 6 12
torch.bfloat16 2 6 12
```

推理实验默认让输入、权重和输出 dtype 一致。混用 FP32 与 BF16 可能发生隐式提升或直接报错。

实际字节：

```text
bytes = numel × element_size
MiB   = bytes / 1024²
```

练一个关键数字：

```text
Qwen3 gate_up 对 2 个 token：
[2, 6144] × 2 B = 24,576 B = 24 KiB
```

它只是激活的一部分，不含权重和 GEMM workspace。

## 1.4 广播：不会复制，但会改变输出形状

形状为 `[T, H]` 的输入乘 `[H]` 的 RMSNorm 权重：

```python
x = torch.arange(12, dtype=torch.float32).view(3, 4)
scale = torch.tensor([1.0, 10.0, 100.0, 1000.0])
y = x * scale
print(y)
```

广播规则从最右维开始对齐：

```text
x:      [3, 4]
scale:     [4]
result: [3, 4]
```

被“补成”的维度只参与索引映射，通常不会真的创建 `[3,4]` 的第二份 scale。但运算结果还是会产生新张量。

显式写法：

```python
y = x * scale.unsqueeze(0)
```

### 常见错误

把 `[T] * [T, H]` 当成“每一行乘一个标量”。形状从右侧对齐后，`[T]` 会沿最后一维 `H` 广播，可能因 `T != H` 报错，或者在恰好相等时得到完全错误的语义。

正确的“每 token 一个标量”是：

```python
weights = torch.randn(T, 1)
out = x * weights        # [T, 1] 与 [T, H] 对齐
```

## 1.5 view、reshape、transpose 与 contiguous

### view

`view` 只改变解释形状，要求底层布局允许新的步长：

```python
x = torch.arange(12)
a = x.view(3, 4)
b = a.view(2, 6)
```

它们共享存储。

### reshape

`reshape` 优先返回 view；如果布局不允许，则复制成连续张量。因此“reshape 永远零拷贝”是错误认识。

### transpose

```python
a = torch.arange(12).view(3, 4)
b = a.transpose(0, 1)
print(a.shape, b.shape)
print(a.stride(), b.stride())
print(a.is_contiguous(), b.is_contiguous())
```

预期：

```text
torch.Size([3, 4]) torch.Size([4, 3])
(4, 1) (1, 4)
True False
```

`transpose` 只交换 shape 和 stride，不搬运元素。因此 `b` 共享存储但非连续。此时直接 `b.view(12)` 很可能报错，`b.reshape(12)` 可能复制。

### 为什么推理代码在意这个

- `view` 比复制便宜；
- 非连续 Tensor 可能迫使 Kernel 使用低效访存或先做 contiguous；
- GEMM 和自定义 Kernel 常要求特定的 last-dim contiguous；
- 一次不起眼的 `.contiguous()` 可能多读一遍、写一遍整个激活。

### 小实验

```python
b = a.transpose(0, 1)
try:
    print(b.view(12))
except RuntimeError as exc:
    print("view failed:", type(exc).__name__)
print("reshape works:", b.reshape(12).shape)
```

预期：`view failed: RuntimeError`，`reshape works: torch.Size([12])`。但后者可能已经复制。

## 1.6 split 和 chunk：SwiGLU 的第一步

假设 `gate_up` 为 `[T, 6144]`：

```python
gate, up = gate_up.split(3072, dim=-1)
```

结果各为 `[T, 3072]`，都是原 Tensor 的 view，不复制数据。

也可以用：

```python
gate, up = gate_up.chunk(2, dim=-1)
```

两者在“宽度可以被 2 整除”时结果相同。`split` 显式指定每段大小；`chunk` 指定段数，最后一段可能更短。

nano-vLLM 的真实实现是：

```python
x, y = x.chunk(2, -1)
return F.silu(x) * y
```

这行代码有两个不同阶段：

1. `chunk` 通常只是 view；
2. `F.silu(x) * y` 是两个逐元素算子，经过 `torch.compile` 有机会融合成一个 Kernel。

## 1.7 从 token 到 Qwen3 Attention 的形状

Prefill 中把多个请求的新 token 拼成一条“packed”序列：

```text
input_ids  [T]
embedding  [T, 1024]
qkv        [T, 4096]
q          [T, 2048] -> [T, 16, 128]
k          [T, 1024] -> [T, 8, 128]
v          [T, 1024] -> [T, 8, 128]
attention output flattened [T, 2048]
o_proj output [T, 1024]
```

`4096 = (16 + 2 × 8) × 128`。

这里 `T` 不是一个 batch 里的统一序列长度，而是所有被调度 token 数之和。真实引擎用 `cu_seqlens_q`、`cu_seqlens_k` 标记每条请求的边界。

### 手算

若两个请求本步分别有 40 和 88 个 token：

```text
T = 128
q 元素数 = 128 × 16 × 128 = 262,144
q BF16 字节 = 524,288 B = 512 KiB
k 元素数 = 128 × 8 × 128 = 131,072
k BF16 字节 = 262,144 B = 256 KiB
```

## 1.8 unsqueeze、squeeze 和 flatten

```python
x = torch.randn(2, 3)
print(x.unsqueeze(0).shape)       # [1, 2, 3]
print(x.unsqueeze(-1).shape)      # [2, 3, 1]
print(x.unsqueeze(-1).squeeze(-1).shape)  # [2, 3]
print(x.flatten(0, 1).shape)      # [6]
```

`unsqueeze` 与 `squeeze` 只改形状和 stride。`flatten(start,end)` 在可能时返回 view，否则内部会做连续化处理。

在 attention 中经常见到：

```python
q.unsqueeze(1)   # [B, 1, num_heads, head_dim]
```

索引 1 通常是人为增加的 query-length 维，使单 token Decode 符合 FlashAttention 的输入约定。

## 1.9 批量矩阵乘的形状

```python
x = torch.randn(8, 1024)
w = torch.randn(3072, 1024)
y = x @ w.T
print(y.shape)
```

预期：`torch.Size([8, 3072])`。

从数学看：

```text
W: [I, H]
W.T: [H, I]
[T, H] @ [H, I] -> [T, I]
```

PyTorch `nn.Linear` 的权重存储就是 `[out_features, in_features]`，前向等价于 `F.linear(x, W)`，其中内部执行 `x @ W.T`。

批量矩阵乘：

```text
[B, M, K] @ [B, K, N] -> [B, M, N]
```

### 常见错误

- 把 `w` 写成 `[H, I]` 再直接 `x @ w`，要与 `F.linear` 的 reference 对齐；
- 只补全维形状，却不检查 stride；
- 把 `[T, heads, head_dim]` 直接 flatten 后假设 head 顺序没有改变。

## 1.10 索引只取需要的位置

nano-vLLM 的 LM Head 在 Prefill 时只对每条请求最后一个 token 算 logits：

```python
hidden = torch.randn(128, 1024)
last_indices = torch.tensor([39, 127])
x = hidden[last_indices]
print(x.shape)
```

预期：`torch.Size([2, 1024])`。

若对整个 `[128, 1024]` 都算 `[128, 151936]` logits，会产生约 148 MiB 的 BF16 中间结果；只取最后 token 可把这一步缩小到 2 行。

## 1.11 本章实验的预期输出

`python labs/01_tensors.py` 会输出：

- `[3,4]` 的 shape/stride；
- 一个 transpose 后非连续 Tensor；
- `view` 失败而 `reshape` 成功的对照；
- `chunk` 两段共享 base storage；
- Qwen3 关键形状和 BF16 字节数；
- 广播后的形状检查。

脚本内包含断言。任何断言失败都说明本章基础没有通过。

## 1.12 练习题

1. 写出 `[5, 16, 128]` flatten 成 `[5, 2048]` 的方式，并说明是否可能复制。
2. 一个 tensor 的 shape 为 `[2, 3, 4]`，stride 为 `(24, 8, 2)`，它连续吗？
3. 证明 `Qwen3 qkv` 输出宽度是 4096。
4. 设有 `T=128`，分别计算 Q、K、V 的 BF16 字节数。
5. 解释为什么 `chunk` 不一定是性能瓶颈，而 `F.silu` 和乘法可能是。
6. 给定 `hidden=[T,1024]` 和 `hidden[:, None]`，说出与 `[T,1024]` 运算后的形状。

## 1.13 验收标准

- 能不运行代码就推出第五章中的主要形状；
- 能解释 `view`、`reshape`、`transpose`、`contiguous` 的复制语义；
- 能独立算出主要激活的 BF16 字节；
- 看到 `[T, H]` 时能说清 `T` 在 Prefill 与 Decode 中分别是什么。

## 1.14 与 nano-vLLM 的关联

nano-vLLM 的 `prepare_prefill()` 把多个请求的新 token 拼成 `input_ids [T]`，再经 Embedding 到 `[T,1024]`；`Qwen3Attention.forward()` 使用 `split + view` 把 QKV 变成 `[T,16,128]`/`[T,8,128]`；`ParallelLMHead.forward()` 用索引只保留每请求最后一个位置。本章的 shape、stride、view 和索引能力，就是读懂这些代码的必要条件。

