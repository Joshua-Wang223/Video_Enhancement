#!/usr/bin/env python3
"""
系统资源自动探测模块
参考模式：tests/analyze_video_pipeline_v3.py、benchmark_ifrnet_versions_v3.py、verify_segment_bitstream_v5.py
功能：CPU/RAM 容器感知探测、GPU 型号识别、自动 workers 计算、GPU 并发上限、信号量管理
"""
import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: str
    driver_version: str = ""
    is_consumer: bool = False

    @property
    def max_nvdec_sessions(self) -> int:
        """NVDEC 解码会话[硬件预算]。全仓库的单一来源。

        [P1-3] 本属性是 GPU 解码并发口径的唯一来源：生产侧
        video_utils._get_gpu_hwaccel_config() 与验收侧
        tests/verify_segment_bitstream_v5.py::compute_gpu_workers() 都由此派生
        （后者留 25% 余量得出建议并发数：8→6、4→3、2→2）。

        分档（名称子串匹配，大小写不敏感）:
          · High-end 数据中心/专业卡    → 8  (A100/A6000/L40/H100/H800/RTX 6000 Ada)
          · 工作站 + 桌面中端           → 4  (A10/L4/T4/V100/P100 等；
                                              RTX 40/30 系桌面卡驱动上限 5 会话)
          · 未知/无法识别               → 2  (保守)

        注意：本项是「会话预算」（能同时开几路），不是「吞吐甜点」。
        实测 Tesla T4 解 4K H.264 时 2~3 路并发即达吞吐平台（4 路起墙钟回升），
        故调用方通常还应再留余量。
        """
        name_l = self.name.lower()
        if any(x in name_l for x in ("a100", "a6000", "l40", "h100", "h800",
                                     "rtx 6000 ada", "rtx 5000 ada")):
            return 8
        if any(x in name_l for x in ("a10", "l4", "t4", "v100", "p100",
                                     "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060",
                                     "rtx 3090", "rtx 3080", "rtx 3070", "rtx 3060",
                                     "rtx 3050", "a5000", "a4000", "a30", "a16", "a2")):
            return 4
        return 2


@dataclass
class SystemResources:
    cpu_logical: int = 1
    cpu_physical: int = 1
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_limit_gb: Optional[float] = None
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    in_container: bool = False
    container_type: str = ""


class SystemResourceDetector:
    _instance: Optional["SystemResourceDetector"] = None
    _cached: Optional[SystemResources] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemResourceDetector, cls).__new__(cls)
        return cls._instance

    def detect(self, force_refresh: bool = False) -> SystemResources:
        if self._cached and not force_refresh:
            return self._cached
        res = SystemResources(
            cpu_logical=self._get_cpu_logical(),
            cpu_physical=self._get_cpu_physical(),
        )
        res.ram_total_gb, res.ram_available_gb, res.ram_limit_gb = self._get_ram_info()
        res.in_container, res.container_type = self._detect_container()
        if HAS_TORCH and torch.cuda.is_available():
            res.gpu_count = torch.cuda.device_count()
            res.gpus = self._get_gpu_info()
            res.vram_total_gb = sum(g.vram_total_gb for g in res.gpus)
            res.vram_free_gb = sum(g.vram_free_gb for g in res.gpus)
        self._cached = res
        return res

    def _get_cpu_logical(self) -> int:
        for path in ["/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                     "/sys/fs/cgroup/cpu.max"]:
            try:
                with open(path) as f:
                    quota_str = f.read().strip()
                if quota_str != "max" and quota_str != "":
                    quota = int(quota_str.split()[0] if " " in quota_str else quota_str)
                    if quota > 0:
                        period_path = path.replace("quota", "period").replace("max", "max")
                        try:
                            with open(period_path) as f2:
                                period_str = f2.read().strip()
                            period = int(period_str.split()[0] if " " in period_str else period_str)
                            return max(1, quota // period)
                        except Exception:
                            pass
            except Exception:
                pass
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                parts = f.read().strip().split()
                if parts[0] != "max" and len(parts) >= 2:
                    quota = int(parts[0])
                    period = int(parts[1])
                    if quota > 0:
                        return max(1, quota // period)
        except Exception:
            pass
        if HAS_PSUTIL:
            return psutil.cpu_count(logical=True) or 1
        return os.cpu_count() or 1

    def _get_cpu_physical(self) -> int:
        if HAS_PSUTIL:
            return psutil.cpu_count(logical=False) or 1
        return max(1, self._get_cpu_logical() // 2)

    def _get_ram_info(self) -> Tuple[float, float, Optional[float]]:
        ram_limit = None
        for path in ["/sys/fs/cgroup/memory/memory.limit_in_bytes",
                     "/sys/fs/cgroup/memory.max"]:
            try:
                with open(path) as f:
                    val = f.read().strip()
                if val not in ("max", ""):
                    try:
                        ram_limit = int(val) / (1024 ** 3)
                        break
                    except ValueError:
                        pass
            except Exception:
                pass
        if HAS_PSUTIL:
            vm = psutil.virtual_memory()
            total = vm.total / (1024 ** 3)
            avail = vm.available / (1024 ** 3)
            if ram_limit and ram_limit < total:
                total = ram_limit
                avail = min(avail, ram_limit * 0.9)
            return round(total, 1), round(avail, 1), ram_limit
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        info[k.strip()] = int(v.strip().split()[0]) * 1024
            total = info.get("MemTotal", 0) / (1024 ** 3)
            avail = info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 3)
            if ram_limit and ram_limit < total:
                total = ram_limit
                avail = min(avail, ram_limit * 0.9)
            return round(total, 1), round(avail, 1), ram_limit
        except Exception:
            pass
        return 8.0, 4.0, None

    def _detect_container(self) -> Tuple[bool, str]:
        if Path("/.dockerenv").exists():
            return True, "docker"
        try:
            with open("/proc/1/cgroup") as f:
                content = f.read()
            if "docker" in content:
                return True, "docker"
            if "kubepods" in content or "kubernetes" in content:
                return True, "k8s"
            if "podman" in content:
                return True, "podman"
        except Exception:
            pass
        return False, ""

    def _get_gpu_info(self) -> List[GPUInfo]:
        gpus: List[GPUInfo] = []
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                for i in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name_bytes = pynvml.nvmlDeviceGetName(handle)
                    name = name_bytes.decode() if isinstance(name_bytes, bytes) else str(name_bytes)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_total = mem.total / (1024 ** 3)
                    vram_free = mem.free / (1024 ** 3)
                    props = torch.cuda.get_device_properties(i)
                    cc = f"{props.major}.{props.minor}"
                    gpus.append(GPUInfo(
                        index=i, name=name,
                        vram_total_gb=round(vram_total, 1),
                        vram_free_gb=round(vram_free, 1),
                        compute_capability=cc,
                        driver_version=pynvml.nvmlSystemGetDriverVersion().decode()
                        if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes) else "",
                        is_consumer=any(x in name.lower() for x in ("geforce", "rtx", "gtx", "titan")),
                    ))
                return gpus
            except Exception:
                pass
        if HAS_TORCH:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    name_bytes = props.name
                    name = name_bytes.decode() if isinstance(name_bytes, bytes) else str(name_bytes)
                    vram_total = props.total_memory / (1024 ** 3)
                    vram_free = vram_total * 0.8
                    try:
                        out = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.free",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=3,
                        )
                        if out.returncode == 0:
                            free_vals = [int(x) for x in out.stdout.strip().split("\n")]
                            if i < len(free_vals):
                                vram_free = free_vals[i] / 1024
                    except Exception:
                        pass
                    gpus.append(GPUInfo(
                        index=i, name=name,
                        vram_total_gb=round(vram_total, 1),
                        vram_free_gb=round(vram_free, 1),
                        compute_capability=f"{props.major}.{props.minor}",
                        is_consumer=any(x in name.lower()
                                        for x in ("geforce", "rtx", "gtx", "titan")),
                    ))
            except Exception:
                pass
        return gpus


def compute_auto_workers(
    task_ram_mb: int = 1024,
    reserve_ratio: float = 0.10,
    resources: Optional[SystemResources] = None,
    gpu_task: bool = False,
) -> int:
    if resources is None:
        resources = SystemResourceDetector().detect()
    cpu_limit = resources.cpu_physical
    ram_limit = int(
        resources.ram_available_gb * 1024 * (1 - reserve_ratio) / max(1, task_ram_mb)
    )
    ram_limit = max(1, ram_limit)
    gpu_limit = cpu_limit
    if gpu_task and resources.gpu_count > 0:
        gpu_limit = sum(g.max_nvdec_sessions for g in resources.gpus)
        gpu_limit = max(1, gpu_limit)
    return min(cpu_limit, ram_limit, gpu_limit)


def get_gpu_semaphore(
    resources: Optional[SystemResources] = None,
    gpu_workers: Optional[int] = None,
) -> Optional["threading.BoundedSemaphore"]:
    import threading
    if resources is None:
        resources = SystemResourceDetector().detect()
    if resources.gpu_count == 0:
        return None
    if gpu_workers == 0:
        return None
    if gpu_workers is None:
        gpu_workers = max(1, sum(
            g.max_nvdec_sessions for g in resources.gpus))
    gpu_workers = max(1, min(gpu_workers, 8))
    return threading.BoundedSemaphore(gpu_workers)
