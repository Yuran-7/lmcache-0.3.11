# LMCache 目录结构说明

## 概述

LMCache 是一个 LLM 推理引擎扩展，通过在多种存储位置（GPU、CPU DRAM、本地磁盘、远程存储）缓存和复用 KV Cache 来降低 TTFT（Time To First Token）并提升吞吐量。本文档详细介绍 `lmcache/` 源码目录下各子目录和文件的作用，重点说明核心的 `v1/` 目录。

---

## 顶层目录结构

```
lmcache/
├── __init__.py                  # 包初始化
├── config.py                    # 旧版引擎配置（LMCacheEngineConfig、LMCacheEngineMetadata）
├── connections.py               # HTTP 连接工具类（HTTPConnection）
├── logging.py                   # 日志系统（自定义 Formatter、彩色日志输出）
├── observability.py             # 可观测性模块（Prometheus 指标、统计监控）
├── protocol.py                  # 客户端-服务端通信协议（ClientMetaMessage、ServerMetaMessage）
├── usage_context.py             # 使用统计上下文（环境、引擎、持续性统计上报）
├── utils.py                     # 通用工具函数（CacheEngineKey、NVTX 标注等）
├── non_cuda_equivalents.py      # 非 CUDA 环境下的内存分配等价实现
├── c_ops.*.so                   # 编译好的 C/CUDA 扩展动态链接库
├── integration/                 # 与推理引擎的集成层
├── server/                      # 独立 LMCache 服务端
├── storage_backend/             # 旧版存储后端实现
├── tools/                       # 工具与基准测试
└── v1/                          # ⭐ 核心 v1 版本实现（重点介绍）
```

---

## 一、顶层文件说明

### `config.py`
定义了 LMCache 的旧版配置体系，包括：
- **`LMCacheEngineMetadata`**: 描述 LLM 模型元数据（模型名称、分布式设置、KV 张量格式/形状/数据类型等）。
- **`LMCacheEngineConfig`**: 缓存引擎的核心配置（chunk_size、本地/远程存储设置、blending 配置等），支持从 YAML 文件、环境变量或默认值加载。
- **`GlobalConfig`**: 全局调试开关。

### `connections.py`
封装了同步和异步的 HTTP 请求工具类 `HTTPConnection`，用于下载文件、获取 JSON/文本/二进制资源等，从 vLLM 项目移植而来。

### `logging.py`
LMCache 的日志框架，特点包括：
- 自定义彩色日志格式（不同级别不同颜色）。
- 通过 `LMCACHE_LOG_LEVEL` 环境变量控制日志级别。
- 使用 `init_logger(name)` 创建带独立 handler 的 logger。

### `observability.py`
完善的可观测性系统，核心组件包括：
- **`LMCStatsMonitor`**: 单例统计监控，跟踪 store/retrieve/lookup 请求数、命中率、吞吐等指标。
- **`PrometheusLogger`**: 将缓存指标导出为 Prometheus 格式（Counter、Gauge、Histogram）。
- 支持 P2P 传输指标、远程后端读写指标、缓存驱逐指标等。

### `protocol.py`
定义客户端-服务端之间的二进制通信协议：
- `ClientMetaMessage`（PUT/GET/EXIST/LIST 命令）。
- `ServerMetaMessage`（SUCCESS/FAIL 返回码）。

### `usage_context.py`
使用统计与上报模块：
- 收集运行环境信息（云提供商、CPU/GPU 信息等）。
- 记录引擎配置和运行元数据。
- 定期向统计服务器上报缓存使用情况（命中/存储 token 数、缓存生命周期等）。

### `utils.py`
通用工具函数集合，包括 `CacheEngineKey`（缓存键）、`CacheStoreEvent`（存储事件）、NVTX 标注装饰器、slot_mapping 压缩等。

### `non_cuda_equivalents.py`
为非 CUDA 环境提供的 pinned/NUMA 内存分配的纯 Python 等价实现，使得代码可在 CPU-only 环境中运行。

---

## 二、`integration/` — 推理引擎集成层

```
integration/
├── vllm/                        # vLLM 集成
│   ├── vllm_adapter.py          # vLLM v0 适配器
│   ├── vllm_v1_adapter.py       # vLLM v1 适配器（最新）
│   ├── lmcache_connector_v1.py  # LMCache 连接器 v1
│   ├── lmcache_connector_v1_085.py  # 兼容 v0.8.5 版本的连接器
│   └── utils.py                 # vLLM 集成工具
└── sglang/                      # SGLang 集成
    ├── sglang_adapter.py        # SGLang 适配器
    └── utils.py                 # SGLang 集成工具
```

该层负责将 LMCache 嵌入到不同的 LLM 推理引擎中。每种引擎都有对应的 **Adapter**，负责：
- 拦截 KV Cache 的 store 和 retrieve 调用。
- 将引擎内部的 KV Cache 格式转换为 LMCache 通用格式。
- 管理前缀匹配、lookup 请求等。

---

## 三、`server/` — 独立 LMCache 服务

```
server/
├── __main__.py                  # 服务启动入口
└── server_storage_backend/      # 服务端专用存储后端
```

独立的 LMCache 服务器进程，可作为远程 KV Cache 存储服务运行，接受来自多个推理引擎实例的缓存请求。

---

## 四、`storage_backend/` — 旧版存储后端

```
storage_backend/
├── evictor/                     # 缓存驱逐策略
│   ├── base_evictor.py          # 驱逐器基类
│   └── lru_evictor.py           # LRU 驱逐策略
├── mem_pool/                    # 内存池管理
│   ├── base_pool.py             # 内存池基类
│   └── local_pool.py            # 本地内存池实现
└── serde/                       # 序列化/反序列化
    ├── serde.py                 # 序列化接口
    ├── torch_serde.py           # PyTorch 原生序列化
    ├── fast_serde.py            # 高性能序列化
    ├── safe_serde.py            # 安全序列化
    ├── cachegen_basics.py       # CacheGen 编解码基础
    ├── cachegen_encoder.py      # CacheGen 编码器
    └── cachegen_decoder.py      # CacheGen 解码器
```

旧版存储后端实现（v1 中有更完善的版本）。包含基础的驱逐策略、内存池和 KV Cache 序列化/反序列化功能。

---

## 五、`tools/` — 工具集

```
tools/
└── controller_benchmark/        # 控制器性能基准测试工具
```

---

## 六、⭐ `v1/` — 核心 v1 版本实现（重点）

`v1/` 是 LMCache 的核心实现目录，包含了完整的 KV Cache 管理系统。相比旧版，v1 引入了更高效的内存管理、分层存储架构、控制器系统、异步加载等高级特性。

```
v1/
├── __init__.py
├── config.py                    # v1 配置系统
├── config_base.py               # 配置基类（自动化配置管理框架）
├── cache_engine.py              # ⭐ 核心缓存引擎
├── cache_interface.py           # 缓存接口定义
├── gpu_connector.py             # GPU 连接器（GPU ↔ CPU 数据传输）
├── memory_management.py         # 内存管理系统
├── lazy_memory_allocator.py     # 懒加载内存分配器
├── token_database.py            # 令牌数据库
├── kv_layer_groups.py           # KV 层分组管理
├── event_manager.py             # 事件管理器
├── protocol.py                  # v1 协议定义
├── rpc_utils.py                 # RPC 工具
├── system_detection.py          # 系统检测（NUMA 映射、内存检测）
├── basic_check.py               # 基础检查工具
├── mock_gpu_connector.py        # GPU 连接器 Mock 实现
├── xpu_connector.py             # XPU（Intel GPU）连接器
│
├── cache_controller/            # 缓存控制器系统
├── compute/                     # 计算模块（attention、blending）
├── storage_backend/             # v1 存储后端系统
├── lookup_client/               # 查找客户端
├── internal_api_server/         # 内部 API 服务
├── multiprocess/                # 多进程支持
├── transfer_channel/            # 数据传输通道
├── server/                      # v1 服务端
├── api_server/                  # API 服务端
├── standalone/                  # 独立运行模式
├── offload_server/              # Offload 服务
├── plugin/                      # 运行时插件系统
├── check/                       # 检查与验证工具
└── utils/                       # v1 工具集（Bloom Filter 等）
```

---

### 6.1 `config.py` / `config_base.py` — v1 配置系统

v1 的配置系统相比旧版进行了大幅升级：
- **`config_base.py`**: 提供自动化的配置管理框架，支持声明式配置定义、环境变量覆盖、YAML 文件加载、命令行参数覆盖、配置校验和日志输出。
- **`config.py`**: 在此基础上定义了 v1 专用的所有配置项，涵盖：
  - 基础配置（chunk_size、local_cpu、max_local_cache_size）
  - 远程存储配置（remote_url、remote_serde）
  - Blending 配置（混合计算比例、阈值、分隔符等）
  - 控制器配置（controller_pull_url、lmcache_worker_ports 等）
  - PD（Prefill-Decode 分离）配置
  - 懒加载内存分配器配置
  - KV Events 配置
  - NUMA 配置等

### 6.2 `cache_engine.py` — ⭐ 核心缓存引擎

**`LMCacheEngine`** 是整个系统的核心类，负责协调所有组件完成 KV Cache 的存储和检索。核心功能包括：

- **`store()`**: 将 GPU 上的 KV Cache 存储到 CPU/远程存储。流程为：Token 处理 → 内存分配 → GPU→CPU 数据拷贝 → 存储后端写入。
- **`retrieve()`**: 从存储后端检索 KV Cache 并加载到 GPU。流程为：Token 匹配 → 存储后端读取 → CPU→GPU 数据拷贝。
- **`store_layer()` / `retrieve_layer()`**: 逐层（layerwise）存储/检索 KV Cache，使用 Python generator 实现流水线化，在存储/加载当前层的同时处理前一层。
- **`prefetch()`**: 预取 KV Cache，与 retrieve 共享存储后端的请求管理。
- **`freeze()`**: 冻结缓存引擎，跳过 store 操作以保护热数据。

支持的高级特性：
- `save_only_first_rank`: MLA 模式下仅第一个 rank 保存，其他 rank 通过广播获取。
- 异步加载（`async_loading`）。
- 缓存控制器集成（`LMCacheWorker`）。
- KV Events 事件追踪。

### 6.3 `gpu_connector.py` — GPU 连接器

负责 GPU 和 CPU 之间 KV Cache 数据的高效传输。提供多种连接器实现：

| 连接器 | 说明 |
|--------|------|
| **`GPUConnectorInterface`** | 抽象基类，定义 `to_gpu` / `from_gpu` / `batched_to_gpu` / `batched_from_gpu` 接口 |
| **`VLLMPagedMemGPUConnectorV2`** | 面向 vLLM 分页式 KV Cache 的连接器，使用 CUDA kernel 进行高效多层 KV 传输 |
| **`VLLMPagedMemGPUConnectorV3`** | V2 的增强版，支持不同层组（group）具有不同 shape/dtype 的场景（如 DeepSeek V3.2 DSA） |
| **`VLLMBufferLayerwiseGPUConnector`** | 逐层缓冲的 GPU 连接器，配合 Blending 使用 GPU 中间 buffer |
| **`VLLMPagedMemLayerwiseGPUConnector`** | 逐层分页式连接器 |
| **`SGLangLayerwiseGPUConnector`** | SGLang 引擎的逐层连接器 |

核心机制：
- 使用 `lmc_ops.multi_layer_kv_transfer` CUDA kernel 实现批量多层 KV 传输。
- 使用独立的 store_stream 和 load_stream 实现存储和加载的异步执行。
- 支持 MLA（Multi-head Latent Attention）格式。

### 6.4 `memory_management.py` — 内存管理系统

这是一个复杂而精密的内存管理子系统，包含 2100+ 行代码，核心组件包括：

- **`MemoryFormat`**: 定义 KV Cache 的内存格式枚举（KV_2LTD、KV_T2D、KV_2TD、KV_MLA_FMT、BINARY 等）。
- **`MemoryObj`**: 内存对象的统一抽象，支持引用计数、pin/unpin、有效性检查、tensor/byte_array 视图等。
- **`MemoryObjMetadata`**: 内存对象的元数据（形状、数据类型、地址、物理大小、引用计数、格式等）。
- **`MemoryAllocatorInterface`**: 内存分配器接口。
- **`TensorMemoryAllocator`**: 基于 pinned memory 的张量内存分配器，支持对齐分配和空闲块合并。
- **`PagedTensorMemoryAllocator`**: 分页式内存分配器。
- **`MixedMemoryAllocator`**: 混合分配器（pinned + paged + buffer 分配器）。
- **`GPUMemoryAllocator`**: GPU 显存分配器。
- **`CuFileMemoryAllocator`**: cuFile/GDS 专用内存分配器。

### 6.5 `lazy_memory_allocator.py` — 懒加载内存分配器

对 `MixedMemoryAllocator` 的扩展，实现渐进式内存扩展：
- **`CompositeBuffer`**: 管理多段物理内存，提供统一视图。
- **`CompositeTensorMemoryAllocator`**: 支持跨段分配的张量分配器，保证不跨段边界分配。
- **`AsyncMemoryExpander`**: 后台守护线程，渐进式扩展内存直到目标大小。
- **`LazyMixedMemoryAllocator`**: 启动时仅分配一小部分内存（如 10%），使用率达到阈值后触发后台异步扩展。

好处：降低启动延迟、减少初始内存占用。

### 6.6 `token_database.py` — 令牌数据库

负责将输入 token 序列转换为缓存引擎键（`CacheEngineKey`），是缓存查找/匹配的基础。提供两种实现：

| 实现 | 说明 |
|------|------|
| **`ChunkedTokenDatabase`** | 将 token 序列按 chunk_size 分块，对每个 chunk 计算 prefix-hash，生成缓存键 |
| **`SegmentTokenDatabase`** | 按特殊分隔符分割 token 序列，用于 Blending 场景 |

hash 函数兼容 vLLM 多个版本的 hash 实现（sha256_cbor 等）。

### 6.7 `cache_controller/` — 缓存控制器系统

控制器系统是 LMCache 分布式架构的核心，负责跨实例的 KV Cache 协调管理。

```
cache_controller/
├── __init__.py
├── config.py                    # 控制器配置
├── controller_manager.py        # 控制器管理器（协调多个 worker）
├── executor.py                  # 命令执行器
├── locks.py                     # 分布式锁
├── message.py                   # 控制器消息类型定义
├── observability.py             # 控制器可观测性
├── utils.py                     # 控制器工具函数
├── worker.py                    # LMCache Worker（每个推理实例运行一个）
├── commands/                    # 控制器命令定义
├── controllers/                 # 控制器策略实现
└── frontend/                    # 控制器前端接口
```

- **`LMCacheControllerManager`**: 中心控制器，接收各 worker 的注册和消息，协调跨实例的缓存决策。使用 ZMQ 进行通信。
- **`LMCacheWorker`**: 运行在每个推理引擎实例上的代理，与控制器进行通信，执行缓存 admit/evict 命令。

### 6.8 `compute/` — 计算模块

```
compute/
├── attention/                   # 注意力计算
│   ├── abstract.py              # 注意力计算抽象接口
│   ├── flash_attn.py            # Flash Attention 实现
│   ├── flash_infer_sparse.py    # FlashInfer 稀疏注意力实现
│   ├── metadata.py              # 注意力元数据
│   └── utils.py                 # 注意力工具
├── blend/                       # KV Cache 混合（CacheBlend）
│   ├── blender.py               # 混合器核心逻辑
│   ├── metadata.py              # 混合元数据
│   └── utils.py                 # 混合工具
├── models/                      # 模型特定实现
│   ├── base.py                  # 模型基类
│   ├── llama.py                 # LLaMA 模型支持
│   ├── qwen3.py                 # Qwen3 模型支持
│   └── utils.py                 # 模型工具
└── positional_encoding.py       # 位置编码处理（RoPE 反转/融合）
```

该模块实现了 CacheBlend 技术所需的计算组件：
- **注意力计算**: 支持 Flash Attention 和 FlashInfer 的稀疏注意力。
- **混合器**: 实现 KV Cache 的非前缀混合，对缓存命中和未命中的部分进行融合计算。
- **位置编码**: 提供 RoPE 反转和融合编码，用于 KV Cache 位置调整。

### 6.9 `storage_backend/` — v1 存储后端系统

v1 的存储后端系统是最丰富的模块之一，使用了丰富的存储后端和连接器：

```
storage_backend/
├── abstract_backend.py          # 存储后端抽象基类
├── storage_manager.py           # ⭐ 存储管理器（协调多层存储）
├── local_cpu_backend.py         # 本地 CPU 内存后端
├── local_disk_backend.py        # 本地磁盘后端
├── gds_backend.py               # GPUDirect Storage 后端
├── weka_gds_backend.py          # WekaFS + GDS 后端
├── remote_backend.py            # 远程存储后端
├── p2p_backend.py               # 点对点传输后端
├── pd_backend.py                # Prefill-Decode 分离后端
├── nixl_storage_backend.py      # NIXL 传输后端
├── audit_backend.py             # 审计后端（记录操作日志）
├── remote_monitor.py            # 远程监控
├── storage_backend_listener.py  # 存储后端事件监听
├── full_sync_sender.py          # 全量同步发送器
├── batched_message_sender.py    # 批量消息发送器
│
├── connector/                   # 远程连接器（详见下文）
├── cache_policy/                # 缓存策略（LRU/LFU/FIFO/MRU）
├── naive_serde/                 # 序列化/反序列化（CacheGen、KiVi 等）
└── job_executor/                # 异步任务执行器
```

#### `storage_manager.py` — 存储管理器

**`StorageManager`** 是整个存储子系统的中枢，负责：
- 管理多级存储后端（CPU→Disk→Remote）。
- 协调 put/get/prefetch 请求在不同后端之间的路由。
- 处理缓存驱逐与内存分配。
- 使用 `AsyncMultiSerializer` / `AsyncSingleSerializer` 防止并发死锁。
- 支持冻结模式（freeze），保护热缓存。

#### `connector/` — 远程连接器

提供与各种远程存储系统的连接：

| 连接器 | 对接的存储系统 |
|--------|---------------|
| `redis_connector.py` | Redis |
| `valkey_connector.py` | Valkey |
| `s3_connector.py` | Amazon S3 |
| `infinistore_connector.py` | InfiniStore |
| `mooncakestore_connector.py` | Mooncake Store |
| `fs_connector.py` | 文件系统 |
| `lm_connector.py` | LMCache 远程服务 |
| `eic_connector.py` | EIC (Elastic InfiniBand Cache) |
| `sagemaker_hyperpod_connector.py` | AWS SageMaker HyperPod |
| `audit_connector.py` | 审计连接器 |
| `mock_connector.py` | Mock 连接器（测试用） |
| `blackhole_connector.py` | 黑洞连接器（丢弃写入） |

每个连接器配有对应的 **Adapter**（如 `redis_adapter.py`），负责连接参数的解析和初始化。

#### `cache_policy/` — 缓存策略

| 策略 | 说明 |
|------|------|
| `lru.py` | Least Recently Used（最近最少使用） |
| `lfu.py` | Least Frequently Used（最不经常使用） |
| `fifo.py` | First In First Out（先进先出） |
| `mru.py` | Most Recently Used（最近最多使用） |

#### `naive_serde/` — 序列化/反序列化

| 模块 | 说明 |
|------|------|
| `naive_serde.py` | 简单序列化 |
| `cachegen_basics.py` | CacheGen 编解码基础类 |
| `cachegen_encoder.py` | CacheGen 压缩编码器 |
| `cachegen_decoder.py` | CacheGen 解压解码器 |
| `kivi_serde.py` | KiVi 量化序列化 |

### 6.10 `lookup_client/` — 查找客户端

```
lookup_client/
├── abstract_client.py                    # 查找客户端抽象基类
├── factory.py                            # 查找客户端工厂
├── lmcache_lookup_client.py              # 同步查找客户端
├── lmcache_async_lookup_client.py        # 异步查找客户端
├── lmcache_lookup_client_bypass.py       # 旁路查找客户端
├── chunk_statistics_lookup_client.py     # 基于统计的查找客户端
├── hit_limit_lookup_client.py            # 命中限制查找客户端
├── mooncake_lookup_client.py             # Mooncake 查找客户端
└── record_strategies/                    # 记录策略（Bloom Filter、文件 Hash 等）
```

查找客户端负责在存储操作之前进行缓存查找（lookup），判断 KV Cache 是否已存在。支持多种策略：
- 同步/异步查找。
- 基于 Bloom Filter 的统计去重（避免重复存储）。
- 命中限制（控制 lookup 返回的最大命中数）。

### 6.11 `internal_api_server/` — 内部 API 服务

```
internal_api_server/
├── api_server.py                # API 服务核心
├── api_registry.py              # API 注册表
├── utils.py                     # 工具
├── common/                      # 通用 API 实现
├── controller/                  # 控制器 API
└── vllm/                        # vLLM 特定 API 实现
```

提供 HTTP API 接口，用于运行时查询和控制 LMCache 行为，包括：
- 查询缓存统计信息。
- 动态调整配置。
- KV Cache 检查日志的开关。
- 冻结/解冻缓存。

### 6.12 `multiprocess/` — 多进程支持

```
multiprocess/
├── custom_types.py              # 自定义类型定义
├── futures.py                   # Future 异步结果封装
├── mp_storage_manager.py        # 多进程存储管理器
├── mq.py                        # 消息队列
├── protocol.py                  # 多进程通信协议
└── server.py                    # 多进程服务端
```

支持 LMCache 在多进程模式下运行，使用消息队列实现进程间通信，保证存储管理器的线程安全。

### 6.13 `transfer_channel/` — 数据传输通道

```
transfer_channel/
├── abstract.py                  # 传输通道抽象基类
├── nixl_channel.py              # NIXL 高性能传输通道
├── py_socket_channel.py         # Python Socket 传输通道
├── mock_memory_channel.py       # Mock 内存通道（测试用）
└── transfer_utils.py            # 传输工具
```

抽象化了节点间数据传输的底层通道，支持：
- **NIXL**: 高性能 GPU Direct RDMA 传输。
- **Python Socket**: 标准网络传输（较低性能，兼容性好）。
- **Mock**: 内存模拟通道（测试/开发用）。

### 6.14 其他子目录

| 目录 | 说明 |
|------|------|
| **`server/`** | v1 服务端实现，包含 `__main__.py` 启动入口和存储后端配置 |
| **`api_server/`** | API 服务端入口 |
| **`standalone/`** | 独立运行模式入口 |
| **`offload_server/`** | KV Cache Offload 专用服务，包含 ZMQ 通信 |
| **`plugin/`** | 运行时插件启动器（`runtime_plugin_launcher.py`），支持动态加载扩展 |
| **`check/`** | 检查与验证工具集（存储管理器测试、远程后端测试、代码生成检查等） |
| **`utils/`** | 工具集，包含 Bloom Filter 实现，用于高效的集合成员检测 |

### 6.15 v1 顶层独立文件

| 文件 | 说明 |
|------|------|
| **`cache_interface.py`** | 缓存引擎的对外接口定义 |
| **`event_manager.py`** | 线程安全的异步事件管理器，跟踪 LOADING 等事件的状态生命周期 |
| **`kv_layer_groups.py`** | KV 层分组管理，将相同 shape/dtype 的层编为一组，支持混合架构模型 |
| **`system_detection.py`** | 系统环境检测，包含 NUMA 拓扑自动发现和可用内存检测 |
| **`protocol.py`** | v1 版本的通信协议定义 |
| **`rpc_utils.py`** | RPC 通信工具函数 |
| **`basic_check.py`** | 基础运行环境检查 |
| **`mock_gpu_connector.py`** | GPU 连接器的 Mock 实现（用于测试） |
| **`xpu_connector.py`** | Intel XPU 连接器实现 |

---

## 架构总结

LMCache v1 的整体架构可以用以下数据流来理解：

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM 推理引擎 (vLLM / SGLang)              │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │            integration/ (Adapter层)                   │  │
│   └─────────────────────┬────────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   LMCacheEngine       │  ← cache_engine.py
              │   (核心缓存引擎)       │
              └──┬────────┬───────┬───┘
                 │        │       │
        ┌────────┘   ┌────┘       └────────┐
        ▼            ▼                      ▼
 ┌─────────────┐ ┌──────────────┐  ┌──────────────────┐
 │GPUConnector │ │TokenDatabase │  │CacheController   │
 │(GPU↔CPU)    │ │(Token→Key)   │  │(分布式协调)       │
 └──────┬──────┘ └──────────────┘  └──────────────────┘
        │
        ▼
 ┌──────────────────┐
 │ StorageManager   │  ← storage_manager.py
 │ (存储管理器)      │
 └──┬──────┬──────┬─┘
    │      │      │
    ▼      ▼      ▼
 ┌──────┐┌──────┐┌───────────────┐
 │ CPU  ││ Disk ││ Remote        │
 │Backend││Backend││(Redis/S3/...) │
 └──────┘└──────┘└───────────────┘
```

这一架构设计使得 LMCache 能够：
1. **透明集成**: 通过 Adapter 层无缝嵌入各种推理引擎。
2. **高效传输**: GPU 连接器使用 CUDA kernel 和 stream 实现高吞吐数据搬运。
3. **灵活存储**: 多级存储后端支持多种部署场景。
4. **可扩展**: 插件系统和连接器架构使得扩展新存储后端非常方便。
5. **可观测**: 完善的 Prometheus 指标和日志系统。
