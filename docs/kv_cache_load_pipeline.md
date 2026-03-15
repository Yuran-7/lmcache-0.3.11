# LMCache KV 缓存加载流水线设计

## 概述

LMCache 提供了从不同存储层（CPU 内存、磁盘）将 KV 缓存加载回 GPU 显存的功能。加载过程经过精心设计，**默认采用按模型层级（Layer-wise）的流水线方式**，通过重叠 "从存储读取" 和 "写入 GPU" 两个操作来最小化端到端延迟。

本文档基于 `lmcache/v1/` 目录下的源代码分析。

---

## 存储层级结构

LMCache 采用多级存储架构，按访问速度从快到慢依次为：

```
┌────────────────────────────────────────────────────────────────┐
│                       GPU KV Cache (显存)                       │  ← 推理引擎直接使用
└───────────────────────────────┬────────────────────────────────┘
                                │ 卸载 (store) / 加载 (retrieve)
┌───────────────────────────────▼────────────────────────────────┐
│              LocalCPUBackend (CPU 内存热缓存)                    │  ← 最快的外部存储
│              (~通常几十 GB 的 Pin Memory)                       │
└───────────────────────────────┬────────────────────────────────┘
                                │ 异步预取 / 写回 (Write-back)
┌───────────────────────────────▼────────────────────────────────┐
│              LocalDiskBackend (本地磁盘 SSD/NVMe)               │  ← 次级存储
│              (~支持 O_DIRECT, 优先级队列任务调度)               │
└────────────────────────────────────────────────────────────────┘
```

**核心类对应关系：**

| 组件 | 文件 | 职责 |
|------|------|------|
| `LMCacheEngine` | `v1/cache_engine.py` | 对外暴露 `retrieve()` / `retrieve_layer()` 等接口 |
| `StorageManager` | `v1/storage_backend/storage_manager.py` | 统一管理多个存储后端，执行查找、读取、写回 |
| `LocalCPUBackend` | `v1/storage_backend/local_cpu_backend.py` | CPU 内存管理，固定（Pinned）内存分配与 LRU 淘汰 |
| `LocalDiskBackend` | `v1/storage_backend/local_disk_backend.py` | 磁盘读写，异步优先级任务队列 |
| `GPUConnector` | `v1/gpu_connector.py` | 负责在 CPU MemoryObj ↔ GPU KV Cache 之间搬运数据 |

---

## 两种加载模式

### 模式一：批量加载（`retrieve`）

适用于所有层的 KV Cache 已完整存储在同一存储后端的简单场景。

**调用入口：** `LMCacheEngine.retrieve(tokens, mask, **kwargs)`

**流程：**

```
1. process_tokens()        → 将 token 序列切分为 chunks，生成 CacheEngineKey
2. StorageManager.get()    → 从 LocalCPU / LocalDisk 后端阻塞读取所有层的 MemoryObj
3. GPUConnector.batched_to_gpu() → 将 MemoryObj 一次性写入 GPU paged memory
```

**特点：** 实现简单，但所有层都准备好后才能开始写 GPU，存在串行瓶颈。

---

### 模式二：按层流水线加载（`retrieve_layer`）⭐ 默认推荐

这是 LMCache v1 版本的核心设计，**流水线默认按模型层级从第 0 层到第 N-1 层依次加载**。

**调用入口：** `LMCacheEngine.retrieve_layer(tokens, mask, **kwargs)`

#### 2.1 核心机制：双生成器协同

`retrieve_layer` 的核心是两个 Python 生成器（Generator）通过 `yield` 配合，实现流水线并发：

```
Generator 1: StorageManager.layerwise_batched_get(keys_layer_major)
             → 按层提交异步 Future，每次 yield 一个 Future

Generator 2: GPUConnector.batched_to_gpu(starts, ends, **kwargs)
             → 接收每层的 MemoryObjs，写入 GPU，每次 yield 一次
```

#### 2.2 流水线时序图（以 N 层模型为例）

```
时间轴 ─────────────────────────────────────────────────────────►

               t0          t1          t2          t3
Storage        [发起L0读取] [发起L1读取] [发起L2读取] ...
               ─────────────────────►
                            ↓
CPU→GPU                    [等L0完成]  [写GPU L0]  [写GPU L1] ...
                            [L0.result()]─────────────────────►
```

**规则：**
- 第 `layer_id` 轮迭代：
  1. `next(get_generator)` → 提交第 `layer_id` 层的异步读取 Future（非阻塞）
  2. `yield` → 将控制权交还给模型推理，推理引擎处理当前层的 forward pass
  3. `task.result()` → 阻塞等待第 `layer_id` 层读取完成
  4. `mem_obj_consumer.send(mem_objs_layer)` → 触发 GPUConnector 将该层数据写入 GPU

#### 2.3 代码核心逻辑（来自 `cache_engine.py`）

```python
# 1. 生成按「层优先」排列的 keys（从 chunk-major 转置为 layer-major）
keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

# 2. 按层批量获取的生成器（NonBlocking，每层返回一个 Future）
get_generator = self.storage_manager.layerwise_batched_get(
    keys_layer_major, location=location
)

# 3. GPU 写入的生成器（send 驱动）
mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
next(mem_obj_consumer)  # 初始化，准备接收数据

# 4. 流水线主循环
for layer_id in range(self.num_layers):
    task = next(get_generator)   # 提交该层的异步读取任务
    yield torch.sum(ret_mask) if layer_id == 0 else None  # 让出控制权
    
    mem_objs_layer = task.result()   # 等待该层读取完成
    mem_obj_consumer.send(mem_objs_layer)  # 触发写 GPU
```

---

## GPU Connector 中的流水线细节

GPU Connector 分为两类，均支持层级流水线但实现略有不同：

### VLLMPagedMemLayerwiseGPUConnector

适用于 vLLM paged memory 格式（不含 RoPE 缓存）：

```
每层 MemoryObj (CPU) → [load_stream] → GPU Buffer (可选) → paged KV Cache
```

- `use_gpu=False`：CPU Tensor 直接通过 `lmc_ops.single_layer_kv_transfer()` 写入 paged memory
- `use_gpu=True`：CPU → GPU Buffer → paged memory（通过 GPU 中间缓冲区优化小批量的内存写入效率）
- 内部使用独立 CUDA Stream（`load_stream`）与主计算流并发执行

### VLLMBufferLayerwiseGPUConnector（双缓冲模式）

适用于需要恢复 RoPE 位置编码的场景，采用 **Ping-Pong 双缓冲**：

```
时间轴：
                  layer 0      layer 1      layer 2
load_stream:   [CPU→buf_A] [CPU→buf_B] [CPU→buf_A] ...
compute_stream:            [RoPE(buf_A)] [RoPE(buf_B)] ...
main_stream:                            [buf→GPU KV] ...
```

- `compute_gpu_buffer_obj` 与 `load_gpu_buffer_obj` 交替使用（`buf_A`/`buf_B` 交换）
- 在写入 GPU paged memory 之前，会调用 `fused_rotary_emb()` 恢复 K cache 的位置编码

---

## 从磁盘加载的特殊处理

当 KV Cache 存储在磁盘时（`LocalDiskBackend`），加载路径为：

```
磁盘文件 → CPU Pinned Memory → GPU KV Cache
```

**关键设计：**

1. **优先级任务队列（`AsyncPQThreadPoolExecutor`）**
   - 预取（Prefetch）优先级最高（`priority=0`）
   - 删除（Delete）次之（`priority=1`）
   - 写入（Put）优先级最低（`priority=2`）

2. **异步预取（`batched_get_non_blocking`）**
   ```python
   # 先在 CPU 预分配内存
   memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
   # 再异步提交磁盘读取任务，返回 Future
   task = asyncio.run_coroutine_threadsafe(
       backend.batched_get_non_blocking("lookup_id", keys_multi_chunk),
       self.loop
   )
   yield task  # 立即 yield，不阻塞
   ```

3. **Write-back 缓存策略**
   - 从磁盘读取的数据会自动写回到 `LocalCPUBackend`（热缓存），避免下次再从磁盘读

4. **O_DIRECT 支持**
   - 可通过配置 `use_odirect=True` 绕过 OS 页缓存，直接 DMA 读取，减少内存拷贝

---

## `layerwise_batched_get` 的实现

`StorageManager.layerwise_batched_get()` 是流水线的关键调度器：

```python
def layerwise_batched_get(
    self,
    keys: List[List[CacheEngineKey]],  # shape: [num_layers][num_chunks]
    location: Optional[str] = None,
) -> Generator[Future, None, None]:
    if location is None:
        location = "LocalCPUBackend"  # 默认从 CPU 内存读取
    
    for keys_multi_chunk in keys:  # 按层迭代
        backend = self.storage_backends[location]
        # 提交该层所有 chunks 的非阻塞读取（返回 coroutine 包装的 Future）
        coro = backend.batched_get_non_blocking("fake_lookup_id", keys_multi_chunk)
        task = asyncio.run_coroutine_threadsafe(coro, self.loop)
        yield task  # 每次 yield 一个 Future，实现逐层流水
```

> **注意：** 当前实现要求所有 chunks 必须来自同一个存储后端（`location` 相同），这一限制在代码中有注释说明未来将支持跨后端的多位置检索。

---

## 存储的对称流水线（`store_layer`）

KV Cache 的存储（GPU → CPU/Disk）也采用了类似的流水线设计：

```
for layer_id in 0..N:
    yield                    # 等待推理引擎完成当前层的 attention 计算
    next(mem_obj_generator)  # 将当前层 KV 从 GPU 卸载到 CPU MemoryObj
    StorageManager.batched_put(keys[layer_id], memory_objs[layer_id])
    # ↑ 异步提交到 CPU 存储（或进一步写磁盘），不阻塞下一层的卸载
```

**存储流水线时序：**

```
推理引擎     [Layer 0 Attn] → yield → [Layer 1 Attn] → yield → [Layer 2 Attn]
                  ↓                         ↓
GPUConnector   [copy L0: GPU→CPU]       [copy L1: GPU→CPU]
                                                ↓
StorageManager                            [put L0 to CPU/Disk] (异步)
```

---

## 配置与触发条件

流水线加载模式（`retrieve_layer`）在以下条件下被激活：

1. `use_layerwise=True`（配置项）
2. `gpu_connector` 必须是以下三者之一：
   - `VLLMPagedMemLayerwiseGPUConnector`
   - `VLLMBufferLayerwiseGPUConnector`
   - `SGLangLayerwiseGPUConnector`

非 layerwise 模式下会退回到 `retrieve()`（批量加载）。

---

## 总结

| 特性 | 批量加载（`retrieve`） | 层级流水线（`retrieve_layer`） |
|------|----------------------|-------------------------------|
| 存储后端读取 | 顺序阻塞 | 按层异步 Future |
| GPU 写入 | 等所有层就绪后一次性写入 | 每层就绪后立即写入，与下层读取重叠 |
| 位置编码恢复 | 不支持（数据直接使用） | 支持（`VLLMBufferLayerwiseGPUConnector`） |
| 适用场景 | 简单场景 / 兼容性 | **生产推荐**，最低延迟 |
| 磁盘加载 | 串行 | 通过 AsyncPQThreadPoolExecutor 并发 |
| CUDA Stream | 单流 | 独立 `load_stream`/`store_stream`，与计算流并发 |

**核心设计哲学：** 通过 Python Generator + CUDA Stream + asyncio 三级并发，将"从 CPU/磁盘读"和"写 GPU"两个操作在时间上重叠，使整体加载时间接近于最慢单层的加载时间，而非所有层时间之和。
