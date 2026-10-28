# 第 5 章 nano-vLLM 真实计算链

## 5.1 本章目标与前置知识

本章从一次 `LLM.generate()` 出发，追到 Qwen3 的 Attention、MLP 和 logits，目标是能够：

- 说出 Scheduler、ModelRunner、Qwen3Model、Attention、Sampler 的职责；
- 区分 packed Prefill 与 batched Decode；
- 标出每一步真实张量形状；
- 解释 KV Cache 的写入与读取；
- 定位 Qwen3 MLP 和激活函数在源码中的位置；
- 用 NVTX 和 Profiler 把算法阶段映射到时间线。

前置知识：第 1～4 章。运行实际实验前先完成 [lab08_nanovllm_trace.py](../labs/08_nanovllm_trace.py) 的小模型或短请求验证。

源码基线：

```text
bb823b3e06983d71485a8e1f23715ebd87d98ef8
```

与你的本地源码有差异时，名称可能变化，但语义链路通常仍是：

```text
Scheduler -> ModelRunner -> Qwen3Model -> DecoderLayer -> Attention / MLP
```

## 5.2 文件地图

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| [engine/llm_engine.py](../reference/nano-vllm/nanovllm/engine/llm_engine.py) | 用户 API、请求循环、step | `generate`、`step` |
| [engine/scheduler.py](../reference/nano-vllm/nanovllm/engine/scheduler.py) | 选择本条前向处理哪些请求 | Prefill、Decode、抢占 |
| [engine/model_runner.py](../reference/nano-vllm/nanovllm/engine/model_runner.py) | 输入打包、KV Cache、运行模型、CUDA Graph | `prepare_prefill`、`prepare_decode`、`run_model` |
| [engine/block_manager.py](../reference/nano-vllm/nanovllm/engine/block_manager.py) | 物理 block 的分配与 prefix cache | `can_allocate`、`can_append` |
| [models/qwen3.py](../reference/nano-vllm/nanovllm/models/qwen3.py) | Qwen3 网络结构 | Attention、MLP、DecoderLayer |
| [layers/attention.py](../reference/nano-vllm/nanovllm/layers/attention.py) | KV 写入、FlashAttention 调用 | `store_kvcache`、Prefill/Decode 分支 |
| [layers/activation.py](../reference/nano-vllm/nanovllm/layers/activation.py) | `SiLU(gate)*up` | 融合后的激活 |
| [layers/linear.py](../reference/nano-vllm/nanovllm/layers/linear.py) | QKV/gate-up 合并、TP 线性层 | weight loader |
| [layers/embed_head.py](../reference/nano-vllm/nanovllm/layers/embed_head.py) | 词嵌入和 LM Head | 只取最后 token |
| [layers/sampler.py](../reference/nano-vllm/nanovllm/layers/sampler.py) | 温度缩放和采样 | `torch.compile` |

## 5.3 从 generate 到一次 step

调用：

```python
llm = LLM(model_path, enforce_eager=True)
outputs = llm.generate(
    ["解释一下 KV Cache"],
    SamplingParams(temperature=0.6, max_tokens=16),
)
```

`LLMEngine.generate()` 做：

1. 用 tokenizer 把字符串变 token id；
2. 为每条请求创建 `Sequence`；
3. 把 Sequence 放入 Scheduler 的 `waiting`；
4. 反复调用 `step()`，直到所有请求完成；
5. decode 输出 token。

一次 `step()`：

```text
scheduler.schedule()
    -> 返回 seqs 和 is_prefill
model_runner.run(seqs, is_prefill)
    -> 准备 packed/batched 输入
    -> 执行 Qwen3 前向
    -> 计算 logits
    -> sample token
scheduler.postprocess()
    -> 追加 token、更新缓存、完成/回收
```

注意：nano-vLLM 一个 `step()` 只处于 Prefill 或 Decode 之一。它的 Scheduler 优先处理 waiting 中的 Prefill，只有在没有可调度 Prefill 时才执行 Decode。

## 5.4 Scheduler 的简化逻辑

调度器维护两个 deque：

```text
waiting：还没完成 Prefill 的请求
running：已经完成 Prefill、每步继续 Decode 的请求
```

Prefill 阶段的关键约束：

- 最多 `max_num_seqs` 条请求；
- 总新 token 数不超过 `max_num_batched_tokens`；
- KV Cache block 必须可分配；
- 若第一条请求太大，可对它做 chunked Prefill；
- 若 block 不够，Decode 请求可能被抢占并回到 waiting，后续需要重新 Prefill。

Decode 阶段每条被调度请求生成一个 token：

```text
num_scheduled_tokens = 1
```

所以批量 Decode 的 `T=B`，其中 `B` 是这个 step 调度到的请求数。

### 例子

设置：

```text
max_num_batched_tokens = 256
max_num_seqs = 4
```

若三个请求各自的未缓存 prompt 长度为：

```text
A=200, B=100, C=40
```

一种可能调度：

```text
第 1 步：A 的 200 个 token 全部 Prefill
第 2 步：B 的 100 + C 的 40 = 140 个 token Prefill
```

但实际还需看 KV block 是否足够、waiting/running 顺序和 prefix cache。不要只看最大 token 数就断言调度结果。

## 5.5 Prefill 的真实输入打包

`prepare_prefill()` 把多个请求的新 token 拼成 1D：

```text
input_ids: [total_tokens]
positions: [total_tokens]
cu_seqlens_q: [num_requests + 1]
cu_seqlens_k: [num_requests + 1]
```

假设两个请求本步长度分别是 3 和 2：

```text
input_ids = [a0,a1,a2, b0,b1]
cu_seqlens_q = [0,3,5]
cu_seqlens_k = [0,3,5]
```

FlashAttention varlen 通过 `cu_seqlens` 知道：

- 第一条占 offset `[0,3)`；
- 第二条占 offset `[3,5)`；
- 它们不能被错误地当成一条长度 5 的对话互相注意。

### 为什么叫 packed

没有必要用 padding 把所有请求凑成相同长度。把有效 token 首尾拼接，再用索引表描述边界，可以减少无用 padding 的算力和显存。

## 5.6 Prefill 的完整形状

设 packed 后总 token 数为 `T`，TP=1：

```text
input_ids          [T]
embedding          [T,1024]
qkv_proj           [T,4096]
q                  [T,16,128]
k                  [T,8,128]
v                  [T,8,128]
attention output   [T,16,128]
flatten            [T,2048]
o_proj             [T,1024]
MLP gate_up        [T,6144]
MLP hidden         [T,3072]
MLP output         [T,1024]
```

多层重复 Attention/MLP 后，最终 hidden 仍为 `[T,1024]`。LM Head 在 Prefill 只选择每条请求最后一个 token，得到：

```text
last hidden    [num_requests,1024]
logits         [num_requests,151936]
sample tokens  [num_requests]
```

## 5.7 Decode 的完整形状

设本步 batch 为 `B`：

```text
input_ids          [B]               每请求上一个生成 token
embedding          [B,1024]
q                  [B,16,128]
k_new / v_new      [B,8,128]
attention output   [B,16,128]
MLP output         [B,1024]
logits             [B,151936]
sample tokens      [B]
```

每条请求的 K/V 历史从 KV Cache 读取。当前新 K/V 先写入缓存，再由 FlashAttention 根据 `context_lens` 和 `block_tables` 读取。

## 5.8 QKV 投影与 GQA

Qwen3-0.6B：

```text
hidden H = 1024
Q heads = 16
KV heads = 8
head_dim = 128
```

Q 宽度：

```text
16 × 128 = 2048
```

K 宽度：

```text
8 × 128 = 1024
```

V 宽度也是 1024。

合并 QKV 投影的输出宽度：

```text
2048 + 1024 + 1024 = 4096
```

源码：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

GQA 共 16 个 Q 头、8 个 KV 头，通常多个 Q 头共享一组 K/V，从而降低 KV Cache 的容量和读取量。

## 5.9 KV Cache 写入与 Attention

`store_kvcache_kernel` 根据 `slot_mapping` 把当前 token 的 K/V 写入分页缓存。`slot_mapping` 中的一个整数代表一个 token 在分页缓存中的物理位置。

Attention 的两个分支：

```text
Prefill:
  flash_attn_varlen_func(q, k, v, cu_seqlens...)

Decode:
  flash_attn_with_kvcache(
      q.unsqueeze(1),
      k_cache,
      v_cache,
      cache_seqlens=context_lens,
      block_table=block_tables,
      causal=True,
  )
```

### 为什么 Prefill 还可能有 block_table

如果 prefix cache 命中，部分 K/V 已经在缓存里。`prepare_prefill()` 会让 K 维度包含已有上下文，因此通过 block table 从缓存读取旧部分，而 Q 只计算当前新 token。

### 常见错误

- 把 `context_lens` 当成“新 token 数”：它是每条请求当前总上下文长度；
- 把 `slot_mapping` 当成 token id：它指向物理 KV 槽位；
- 看到 `q.unsqueeze(1)` 就以为 head_dim 变了：只是增加 query-length=1 的维；
- prefix cache 命中后仍假设 K/V 输入只有本次 token。

## 5.10 Qwen3 的残差流

源码中的 `Qwen3Model.forward()`：

```python
hidden_states = self.embed_tokens(input_ids)
residual = None
for layer in self.layers:
    hidden_states, residual = layer(positions, hidden_states, residual)
hidden_states, _ = self.norm(hidden_states, residual)
```

每层返回 `hidden_states, residual`。第一层 residual 为 `None`，后续由 `RMSNorm.add_rms_forward()` 将当前分支输出和前一份残差合并。

阅读时不要把它误认为是“两个无关输出”。它是对残差加法与 RMSNorm 合并后的状态编码。

## 5.11 MLP 在源码中的位置

`Qwen3MLP.forward()`：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
```

形状：

```text
[T,1024] -> [T,6144] -> [T,3072] -> [T,1024]
```

`SiluAndMul`：

```python
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

下一章会对这个点进行正确性对比和性能实验。

## 5.12 logits 与采样

Qwen3 LM Head 是 `[151936,1024]` 的线性映射。若每次对 `[B,1024]` 计算：

```text
logits: [B,151936]
```

当 `B=32` 时，BF16 logits 有：

```text
32 × 151936 × 2 = 9,723,904 B ≈ 9.27 MiB
```

但这只是 logits 本身，矩阵乘还需要读取约 151936×1024 BF16 的 LM Head 权重。Qwen3-0.6B 使用 tied embeddings，因此 LM Head 与 embedding 共享同一份权重语义，但这不意味着 logits 计算免费；它仍要执行一次大型投影。

`Sampler.forward()` 大致是：

```text
logits /= temperature
probs = softmax(logits)
从 probs 采样
```

采样通常只在 rank 0 或逻辑主进程执行，不参与模型每层计算。

## 5.13 Eager 与 CUDA Graph

nano-vLLM 的 Decode 路径在条件允许时会：

1. 为若干固定 batch size 预捕获 CUDA Graph；
2. 每次把新的 input ids、positions、slot mapping、context lens 和 block tables 复制到 graph 固定缓冲区；
3. 调用 `graph.replay()`；
4. 取输出并计算 logits、采样。

CUDA Graph 主要减少大量小 Kernel 的 CPU launch 序列和调度开销，不减少数学计算或 HBM 字节。它需要固定地址、固定形状和可控的上下文；因此源码只对 Decode 和有限的 batch size 使用，非法形状回退到 Eager。

学习顺序：

1. `enforce_eager=True` 先保证数值；
2. 再比较 `enforce_eager=False` 的稳态延迟；
3. 再分析 Graph replay 后还有哪些 GPU 计算主导时间。

## 5.14 如何把源码映射到 Profile

运行：

```bash
export QWEN3_PATH="$HOME/huggingface/Qwen3-0.6B"
export NANO_VLLM_ROOT="$HOME/nano-vllm"

python labs/08_nanovllm_trace.py \
  --model "$QWEN3_PATH" \
  --nanovllm-root "$NANO_VLLM_ROOT"
```

脚本会：

- 以保守显存配置启动 nano-vLLM；
- 给一个短 prompt；
- 给 Prefill/Decode 包 NVTX 区间；
- 捕获第一层 MLP 的输入输出 shape；
- 打印生成结果和 GPU 峰值显存。

要保留系统时间线：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  -o outputs/nanovllm_trace \
  python labs/08_nanovllm_trace.py \
    --model "$QWEN3_PATH" \
    --nanovllm-root "$NANO_VLLM_ROOT"

nsys stats outputs/nanovllm_trace.nsys-rep
```

在 Nsight Systems 中依次看：

1. NVTX 中的 Prefill/Decode 比例；
2. CUDA Kernel 的 launch 密度；
3. GEMM、FlashAttention、SiLU/mul、RMSNorm 各自时间；
4. CPU 是否存在长时间空白等待 GPU 或 GPU 等待 CPU；
5. Decode 的 batch 是否太小导致 GPU 利用率低。

## 5.15 一次请求从文字到下一个 token

把全链路压缩成：

```text
prompt text
  -> tokenizer
  -> input_ids [P]
  -> Embedding [P,1024]
  -> 28 × (RMSNorm + Attention + RMSNorm + SwiGLU MLP)
  -> final RMSNorm [P,1024]
  -> 只取最后位置 [1,1024]
  -> LM Head [1,151936]
  -> sampler [1]
  -> 追加到 Sequence.token_ids
  -> 下一次 Decode 从新 token 开始
```

对多个请求，Prefill 的 token 被 packed，Decode 的“每条一个新 token”被 batched。理解这一点，就能把书上单独的 Transformer 公式和引擎真实运行时对应起来。

## 5.16 常见错误

| 错误 | 原因 | 检查 |
|---|---|---|
| 以为一次 step 同时有 Prefill 和 Decode | nano-vLLM Scheduler 每次返回一个模式 | `is_prefill` 分支 |
| 把 padding 当成必须 | 使用 varlen packed 输入 | 看 `cu_seqlens` |
| 把序列长度当 batch | FlashAttention 张量布局不同 | 看 `[T,heads,dim]` |
| 忘记 Decode 也要写 KV | 下一个 token 无法读取历史 | `store_kvcache` |
| Eager 正常、Graph 异常 | 固定缓冲区/shape/context 假设 | 先回退 eager |
| 把采样耗时算进每层 MLP | 执行阶段不同 | Nsight 时间线 |

## 5.17 练习

1. Prefill 中两个请求长度分别为 37 和 91，写出 `cu_seqlens_q`。
2. Prefill 总 `T=128`，计算 QKV 各段、Attention 输出、MLP 激活形状。
3. Decode batch 为 16，计算 logits 的 BF16 字节。
4. 解释 scheduler 为什么优先 Prefill；这样做有什么延迟/吞吐取舍？
5. 在 Nsight Systems 中找到一次 CUDA Graph replay，数出它包含哪些 Kernel。
6. 修改 nano-vLLM 的微分支，观察 profile 变化。

## 5.18 验收标准

- 能不看源码画出 `generate -> step -> Scheduler -> ModelRunner -> Qwen3`；
- 能分别为 Prefill 和 Decode 填出至少 10 个张量形状；
- 能解释 `cu_seqlens`、`slot_mapping`、`context_lens`、`block_tables`；
- 能指出 Qwen3 MLP 的位置并解释三行前向；
- 能为一次短请求生成 Nsight trace，并区分 Prefill 与 Decode。

