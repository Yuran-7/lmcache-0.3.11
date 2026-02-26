# LMCache 能否独立执行模型推理？

## 结论

**LMCache 本身不是推理引擎，无法独立执行完整的模型推理任务。**  
在生产使用中，它必须配合 vLLM 或 SGLang 等推理引擎才能完成端到端的推理工作。

但 LMCache 提供了一个**独立运行的 KV Cache 服务模式（Standalone 模式）**，可以在没有推理引擎的情况下作为 KV 缓存存储服务启动——这只是缓存的存取服务，不涉及任何模型推理。

---

## 架构定位

LMCache 的职责是 **KV Cache 的存储、检索与优化**，而不是模型推理本身。其核心能力包括：

| 能力 | 是否具备 |
|------|---------|
| 加载模型权重 | ❌ |
| Tokenize 输入 / 生成输出 | ❌ |
| 完整 Attention / FFN 前向计算 | ❌ |
| KV Cache 的存储与检索（CPU / Disk / 远端） | ✅ |
| KV Cache 的跨请求复用 | ✅ |
| CacheBlend 部分重计算（Partial Prefill） | ✅（依赖推理引擎提供模型） |

---

## 三种运行模式

### 1. 嵌入推理引擎模式（主要使用方式）

LMCache 作为 **KVConnector 插件**嵌入到 vLLM 或 SGLang 中运行。推理引擎负责完整的模型前向计算，LMCache 负责 KV Cache 的存取加速。

```
vLLM / SGLang
    ├── Scheduler
    ├── Model Executor（完整推理计算）
    └── KVConnector
           └── LMCache（KV 存储 + 检索 + CacheBlend）
```

相关代码：
- `lmcache/integration/vllm/vllm_v1_adapter.py` — vLLM v1 适配器
- `lmcache/integration/vllm/lmcache_connector_v1.py` — vLLM connector 接口
- `lmcache/integration/sglang/` — SGLang 集成

### 2. Standalone 模式（仅 KV Cache 服务）

通过以下命令可以独立启动一个 LMCache KV Cache 服务节点（**不执行推理**）：

```bash
python -m lmcache.v1.standalone \
    --config config.yaml \
    --kv-shape "32,2,256,32,128" \
    --device cpu
```

代码入口：`lmcache/v1/standalone/__main__.py`

此模式的用途是：
- 作为远端 KV Cache 存储服务供多个推理节点共享
- 在 Disaggregated Prefill（prefill/decode 分离部署）场景中承担 KV 传输中间节点的角色
- 无需 CUDA / GPU 即可运行（使用 `MockGPUConnector`）

此模式下 **不会执行任何模型前向计算**，仅提供 KV 数据的 store / retrieve 接口。

### 3. CacheBlend 部分重计算（依赖推理引擎）

CacheBlend 是 LMCache 的核心优化功能，它在 KV Cache 命中后对少量"重要 token"执行**部分重计算**（Partial Prefill），以修正因上下文变化导致的 KV 偏差。

关键点：CacheBlend 需要访问**推理引擎中的真实模型层**（`vllm_model.model.layers`），因此它无法脱离推理引擎独立运行。

```python
# lmcache/v1/compute/blend/blender.py
class LMCBlender:
    def __init__(self, cache_engine, gpu_connector, vllm_model, config):
        # 必须传入 vllm_model，blender 直接操作其 attention layers
        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)
```

CacheBlend 相关代码：
- `lmcache/v1/compute/blend/blender.py` — 核心 Blender，驱动逐层 fetch + partial prefill
- `lmcache/v1/compute/models/base.py` — `compute_layer()` 逐层计算驱动
- `lmcache/v1/compute/models/llama.py` / `qwen3.py` — 模型结构适配
- `lmcache/v1/cache_engine.py` — `retrieve_layer()` 逐层 KV 检索

---

## 依赖关系总结

```
                ┌─────────────────────────────────────┐
                │        推理引擎（vLLM / SGLang）      │
                │  - 加载模型权重                       │
                │  - 完整 Attention + FFN 计算           │
                │  - Token 生成                         │
                └──────────────┬──────────────────────┘
                               │ 调用 KVConnector 接口
                ┌──────────────▼──────────────────────┐
                │            LMCache                   │
                │  - KV Cache 存储（CPU / Disk / 远端） │
                │  - KV Cache 检索与复用                │
                │  - CacheBlend 部分重计算（借用模型层）  │
                │  - Standalone KV 服务节点             │
                └─────────────────────────────────────┘
```

---

## 参考代码路径

| 功能 | 代码路径 |
|------|---------|
| vLLM 集成适配器 | `lmcache/integration/vllm/vllm_v1_adapter.py` |
| SGLang 集成 | `lmcache/integration/sglang/` |
| Standalone 服务入口 | `lmcache/v1/standalone/__main__.py` |
| KV Cache 核心引擎 | `lmcache/v1/cache_engine.py` |
| CacheBlend 主逻辑 | `lmcache/v1/compute/blend/blender.py` |
| 逐层模型计算 | `lmcache/v1/compute/models/base.py` |
| CacheBlend 示例 | `examples/blend_kv_v1/blend.py` |
