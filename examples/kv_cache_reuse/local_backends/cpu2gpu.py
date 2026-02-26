import torch

def benchmark_transfer(tensor, use_non_blocking=False, num_runs=100):
    """测试张量传输到 GPU 的平均耗时"""
    
    # 1. 预热 (Warm-up)
    # GPU 刚开始工作时需要初始化上下文，前几次操作会特别慢，需要先预热
    for _ in range(10):
        _ = tensor.to("cuda", non_blocking=use_non_blocking)
    torch.cuda.synchronize() # 确保预热完成

    # 创建 CUDA 事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    
    # 2. 正式测试
    for _ in range(num_runs):
        start_event.record()
        
        # 将数据移动到显存
        _ = tensor.to("cuda", non_blocking=use_non_blocking)
        
        end_event.record()
        # 必须等待当前传输真正完成再记录时间
        torch.cuda.synchronize() 
        
        # 计算耗时 (毫秒)
        times.append(start_event.elapsed_time(end_event))

    # 计算平均耗时
    avg_time = sum(times) / num_runs
    return avg_time

def main():
    if not torch.cuda.is_available():
        print("未检测到可用的 GPU！")
        return

    print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")

    # 构造 0.5 GB 的数据
    # float32 每个元素占 4 字节
    # 0.5 GB = 512 MB = 512 * 1024 * 1024 bytes
    # 需要的元素数量 = (512 * 1024 * 1024) / 4 = 134,217,728
    num_elements = 134_217_728
    
    print("\n正在生成测试数据...")
    cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
    
    size_mb = cpu_tensor.element_size() * cpu_tensor.nelement() / (1024 ** 2)
    print(f"测试张量大小: {size_mb:.2f} MB\n")

    # --- 测试 1：普通分页内存 (Pageable Memory) ---
    print("测试 1: 普通内存 (未锁页) -> GPU")
    time_normal = benchmark_transfer(cpu_tensor, use_non_blocking=False)
    bandwidth_normal = size_mb / (time_normal / 1000) / 1024 # GB/s
    print(f"平均耗时: {time_normal:.2f} ms")
    print(f"实际带宽: {bandwidth_normal:.2f} GB/s\n")

    # --- 测试 2：锁页内存 (Pinned Memory) ---
    print("测试 2: 锁页内存 (Pinned Memory) -> GPU")
    # 将普通的 CPU 张量转换为锁页张量
    pinned_tensor = cpu_tensor.pin_memory()
    time_pinned = benchmark_transfer(pinned_tensor, use_non_blocking=True)
    bandwidth_pinned = size_mb / (time_pinned / 1000) / 1024 # GB/s
    print(f"平均耗时: {time_pinned:.2f} ms")
    print(f"实际带宽: {bandwidth_pinned:.2f} GB/s\n")

    # --- 结论 ---
    speedup = time_normal / time_pinned
    print(f"结论: 使用锁页内存比普通内存快了约 {speedup:.2f} 倍！")

if __name__ == "__main__":
    main()