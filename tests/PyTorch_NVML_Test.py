"""
PyTorch + nvidia-ml-py NVML 诊断脚本
前提：已安装 nvidia-ml-py（pip install nvidia-ml-py）
"""
import torch
import sys

def check_nvml():
    # ---------- 1. 检查 nvidia-ml-py ----------
    try:
        import pynvml
        print("✅ pynvml 模块已加载（来自 nvidia-ml-py）")
    except ImportError:
        print("❌ 未安装 nvidia-ml-py，请先执行：pip install nvidia-ml-py")
        sys.exit(1)

    # ---------- 2. NVML 初始化与查询 ----------
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("✅ NVML 初始化成功")
    except Exception as e:
        print(f"❌ NVML 初始化失败: {e}")
        sys.exit(1)

    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        print(f"✅ GPU 利用率: {util}%")
    except Exception as e:
        print(f"❌ 查询利用率失败: {e}")

    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"✅ 显存: 已用 {mem.used / 1024**3:.2f} GiB / 总计 {mem.total / 1024**3:.2f} GiB")
    except Exception as e:
        print(f"❌ 查询显存失败: {e}")

    pynvml.nvmlShutdown()

    # ---------- 3. torch.cuda.utilization 测试 ----------
    print("\n--- torch.cuda.utilization ---")
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return

    has_util = hasattr(torch.cuda, 'utilization')
    print(f"函数存在: {has_util}")
    if has_util:
        try:
            util = torch.cuda.utilization(0)
            print(f"✅ 调用成功: {util}%")
        except Exception as e:
            print(f"❌ 调用失败: {e}")

    print(f"PyTorch 版本: {torch.__version__}")

if __name__ == "__main__":
    check_nvml()