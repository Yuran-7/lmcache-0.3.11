# LMCache Blend（缓存混合）机制设计文档

## 概述

Blend（缓存混合）是 LMCache 的一项关键功能，解决了"**缓存的 KV 与当前请求的实际 KV 存在偏差**"的问题。

### 背景：为什么需要 Blend？

当多个请求共享同一段前缀（system prompt 等）时，LMCache 会复用缓存的 KV Cache。但由于不同请求的上下文不同，从缓存中取出的 KV 可能与该请求"应有的" KV 存在误差。Blend 的目标是：

1. **识别**偏差最大的 token（重要 token）
2. 对这些 token **重新计算** KV
3. 其余 token 直接**复用缓存**

这样在保证大部分计算省略的同时，对误差较大的 token 做局部修正。

---

## 一、如何选取重要 Token

### 1.1 使用 K 的 L2 偏差，而不是 V

**核心问题：是计算 K 的偏差还是 V 的偏差？**

答案是：**计算的是新旧 K（Key）之间的 L2 偏差**，而不是 V（Value）。

代码位于 `lmcache/v1/compute/blend/blender.py`，`LMCBlender.process_qkv()` 方法：

```python
# 从 GPU 取出缓存的旧 KV
old_k, old_v = self.gpu_connector.get_kv(layer_id)

# 对当前 Q、K 做 RoPE 位置编码
q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

# 如果当前层是检查层，计算 K 的 L2 偏差
if layer_id in self.common_metadata.check_layers:
    diff_k = torch.sum(
        (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
    )
```

**为什么用 K 而非 V？**

- K（Key）决定了 attention 的查询匹配方式，是"哪些位置会被注意到"的核心
- K 的错误会直接影响 attention 分布，进而影响 V 的聚合结果
- K 的偏差更能反映位置信息（RoPE）的变化，对语义理解影响更显著

### 1.2 选取 Top-K 偏差最大的 Token

```python
# 确定需要重算的 token 数量（按比例）
topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
topk_num = max(topk_num, 1)  # 至少 1 个

# 选出 diff_k 最大的 topk_num 个 token 索引
top_indices = torch.topk(diff_k, k=topk_num).indices
top_indices, _ = torch.sort(top_indices)  # 排序保持顺序

# 只对这些 token 继续进行完整计算
k, v = k[top_indices], v[top_indices]
q = q[top_indices]
residual = residual[top_indices]
```

所选比例由配置项 `blend_recompute_ratios` 控制，默认约为 0.15（即只重算 15% 的 token）。

### 1.3 更新 K、V 并混合

对重要 token 重算完成后，**用新的 K、V 覆盖缓存中对应位置的旧值**：

```python
old_k[self.metadata.imp_indices] = k   # 用新 K 替换旧 K（重要位置）
old_v[self.metadata.imp_indices] = v   # 用新 V 替换旧 V（重要位置）
# 返回：混合后的完整 K、V（其余位置保持缓存）
return q, old_k, old_v, residual, attn_output, attn_metadata
```

这便是"Blend（混合）"名称的由来：最终的 KV = 缓存的旧 KV + 重新计算的新 KV（覆盖偏差大的位置）。

---

## 二、从哪一层开始计算偏差？

**核心问题：是从第 0 层还是第 1 层开始？**

### 2.1 `blend_check_layers` 配置项

偏差计算只在**指定的检查层（check layers）**进行，由配置项 `blend_check_layers` 控制：

```yaml
# 配置文件示例
blend_check_layers: [1]   # 在第 1 层检测重要 token
```

文档中的默认值描述：
> `blend_check_layers`: Layers to determine the recomputed tokens. **Default: 1**

### 2.2 为什么默认从第 1 层（而非第 0 层）？

| 层 | 情况 | 原因 |
|----|------|------|
| **第 0 层** | 不检查 | 第 0 层 K 完全由输入 embedding 决定，几乎没有上下文积累，偏差信号较弱 |
| **第 1 层** | 默认检查层 | 经过一层 attention 后，K 已积累了跨 token 的上下文信息，偏差信号更可靠 |

### 2.3 代码逻辑

在 `process_qkv` 中，每一层都会被调用，但只有 `check_layers` 中的层才会触发重要 token 选取：

```python
if layer_id in self.common_metadata.check_layers:
    # 仅在此层计算偏差并确定 imp_indices
    diff_k = torch.sum(...)
    top_indices = torch.topk(diff_k, k=topk_num).indices
    self.metadata.imp_indices = top_indices   # 保存到 metadata
```

一旦 `imp_indices` 确定（在第 1 层），**后续所有层**都会使用同一份重要 token 集合，不再重新计算：

```python
# 若 imp_indices 已存在（在非 check_layers 的层），直接使用
if self.metadata.imp_indices is not None:
    old_k[self.metadata.imp_indices] = k
    old_v[self.metadata.imp_indices] = v
    return q, old_k, old_v, ...
```

**结论：偏差从第 1 层（Layer 1，非最底层 Layer 0）开始计算，并且只计算一次，后续层复用同一组重要 token 索引。**

---

## 三、Blend 的 KV 加载是否使用按层流水线？

**答：是的，Blend 的 KV 加载使用按层流水线（Layer-wise Pipeline）。**

### 3.1 `blend_layer` 方法

`LMCBlender.blend_layer()` 明确调用了 `cache_engine.retrieve_layer()`，这正是层级流水线加载接口：

```python
def blend_layer(self, tokens, mask, **kwargs):
    """
    Perform layerwise retrieve + blending.
    """
    layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
    layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

    next(layerwise_retriever)  # 初始化流水线
    yield

    for i in range(self.num_layers):
        next(layerwise_retriever)       # 加载第 i 层 KV（异步提交）
        next(layerwise_model_executor)  # 同时推进模型重算第 i 层
        yield

    next(layerwise_retriever)  # 完成最后的清理
    self.metadata.clean()
    yield
```

### 3.2 三路流水线并行

Blend 实际上构建了一个**三路流水线**：

```
时间轴 ─────────────────────────────────────────────────────────►

               Layer 0          Layer 1          Layer 2
               
KV加载         [异步提交L0读取] [异步提交L1读取] [异步提交L2读取]
(retrieve_layer)    ↓                  ↓                 ↓
               [等L0完成]       [等L1完成]       [等L2完成]
               [GPU写入L0 KV]   [GPU写入L1 KV]   [GPU写入L2 KV]

模型重算                       [L0重要token      [L1重要token
(compute_layer)               QKV/Attn/FFN]    QKV/Attn/FFN]

KV混合         ←─ process_qkv() 在每层内：新K覆盖缓存K，比较偏差 ─→
```

| 流 | 来源 | 职责 |
|----|------|------|
| `retrieve_layer` | `CacheEngine` | 按层从 CPU/磁盘异步读取缓存 KV，写入 GPU |
| `compute_layer` | `LMCBaseModel` | 对**重要 token** 执行完整 forward pass（LN → QKV → Attn → FFN） |
| `process_qkv` | `LMCBlender` | 每层内：对比新旧 K，选重要 token（仅 check_layer）；混合新旧 KV |

### 3.3 KV 格式差异

当 `enable_blending=True` 时，KV 的内存格式被设置为 `KV_2TD`（而非普通的 `KV_T2D`）：

```python
# cache_engine.py
elif config.enable_blending:
    self.fmt = MemoryFormat.KV_2TD
```

`KV_2TD` 格式将 K 和 V 分离存储，方便 `get_kv(layer_id)` 分别取出 old_k 和 old_v 进行比较和混合。

---

## 四、完整流程总结

```
用户请求到来
     │
     ▼
SegmentTokenDatabase.process_tokens()
     │  将 tokens 分段，处理特殊分隔符（blend_special_str）
     │
     ▼
LMCBlender.blend_layer()
     │
     ├─── retrieve_layer() ────────────────────────────────────────┐
     │    （CacheEngine 层级流水线，从 CPU/Disk 异步加载缓存 KV）  │
     │                                                             │
     ├─── compute_layer() ─────────────────────────────────────────┤
     │    （LMCBaseModel 对所有 token 做 embedding + layernorm）   │
     │                                                             │
     │    for each layer i:                                        │
     │      ├── retrieve_layer: 载入第 i 层缓存 KV 到 GPU         │
     │      ├── compute_layer: 计算第 i 层 QKV                    │
     │      └── process_qkv:                                       │
     │            ├── 施加 RoPE 位置编码                          │
     │            ├── [check_layer 才执行]                        │
     │            │     diff_k = ||new_K - old_K||²（逐 token）  │
     │            │     imp_indices = topk(diff_k)               │
     │            └── old_K[imp_indices] = new_K（混合）         │
     │                old_V[imp_indices] = new_V（混合）         │
     │                Attention 计算用混合后的完整 KV             │
     │                                                             │
     ▼                                                             │
混合完成的 KV Cache 已在 GPU paged memory 中                      │
（可直接被推理引擎的后续 decode 使用）                            │
```

---

## 五、关键配置参数

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| `enable_blending` | `LMCACHE_ENABLE_BLENDING` | `false` | 是否启用 Blend |
| `blend_check_layers` | `LMCACHE_BLEND_CHECK_LAYERS` | `[1]` | 在哪几层计算 K 偏差以确定重要 token（**默认第 1 层**，非第 0 层） |
| `blend_recompute_ratios` | `LMCACHE_BLEND_RECOMPUTE_RATIOS` | `[0.15]` | 重算 token 的比例（如 0.15 = 15%） |
| `blend_thresholds` | `LMCACHE_BLEND_THRESHOLDS` | `None` | 偏差阈值（目前代码中预留，暂未使用） |
| `blend_min_tokens` | `LMCACHE_BLEND_MIN_TOKENS` | `256` | 触发 Blend 的最小 token 数 |
| `blend_special_str` | `LMCACHE_BLEND_SPECIAL_STR` | `" # # "` | 分隔不同请求段落的特殊字符串 |

---

## 六、结论

| 问题 | 答案 |
|------|------|
| Blend 用什么衡量 token 的重要性？ | **K（Key）的 L2 偏差**（新旧 K 之差的平方和），而非 V 的偏差 |
| 从第几层开始计算偏差？ | **默认从第 1 层**（Layer 1，非最底层 Layer 0），由 `blend_check_layers` 配置 |
| 偏差检查只在一层还是每层都做？ | **只在 check_layers 指定的层做一次**，后续层复用同一份 `imp_indices` |
| Blend 的 KV 加载是否使用流水线？ | **是**，调用 `retrieve_layer()` 实现按层异步流水线加载 |
| 流水线有几路并发？ | **三路**：KV 异步加载 + 模型重算 + KV 混合，三者交替 `yield` 完成流水线 |
