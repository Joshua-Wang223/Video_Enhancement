#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验收脚本：校验 [FIX-PIPE4-LA8] 修复后插帧输出段的 H.264 码流完整性。

修复前失败特征（段 1 花屏根因码流解剖结论）:
  1. ffprobe: frames(845) << packets(4960) —— 解码器静默丢帧
  2. 段首 17 连 IDR（per-slot IDR warmup 在 LA 预热期永不 warm）
  3. 每第 9 帧 frame_num 回退一次（LA 输出 buffer 重路由错位 → 帧序乱）
  4. pts_anomaly（ffmpeg -v verbose 检出）

修复后验收标准:
  1. nb_read_frames == nb_read_packets（帧数守恒，无静默丢帧）
  2. 段首仅 1 个 IDR（首个 IDR 之后长时间无后续 IDR）
  3. frame_num 单调不递减（无回退）
  4. 无 pts_anomaly / 无解码错误输出

跨平台: pathlib / subprocess(shell=False) / shutil.which，Windows 与 Linux 均可运行。

v2 新增 (并行批处理):
  - 自动识别输入为文件或文件夹，文件夹递归扫描 *.mp4 / *.h264 / *.mkv / *.avi
  - 多线程并行引擎 (ThreadPoolExecutor)，按 CPU/RAM 自动计算 workers
  - GPU 信号量闸门 (--gpu-workers)，防 NVDEC 会话耗尽
  - 逐文件进度输出 + 最终汇总表

v3 新增 (GPU 自适应):
  - --gpu-workers 默认值不再硬编码为 4，改为按 GPU 型号自动最大化计算
    (T4→2, A10→4, A100/L40→6, 未知/消费级→2，最高 8)
  - 支持 pynvml + nvidia-smi 双路径 GPU 型号探测
  - --help 含详细并发双闸门说明 (workers / gpu-workers 区别与联系)
  - 支持 --gpu-workers 0 禁用 GPU 闸门（所有 NVDEC 任务无限制并发）

用法:
  单文件: python tests/verify_segment_bitstream.py <video.mp4> [--dump-nal temp/nal.txt]
  多文件: python tests/verify_segment_bitstream.py a.mp4 b.mp4 c.mp4 --workers 4
  文件夹: python tests/verify_segment_bitstream.py ./segments/ --hwaccel cuda --gpu-workers 2
  混合:    python tests/verify_segment_bitstream.py a.mp4 ./dir1/ b.mp4
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path


def _fmt_elapsed(seconds):
    """秒 → 人类可读耗时字符串。"""
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1000)
    return '%.2f s' % seconds

# Windows 控制台默认 GBK 无法编码 ✅/❌ 等符号，统一 stdout/stderr 为 UTF-8
# （Python 3.7+ reconfigure；UTF-8 终端正常显示，GBK 终端不崩溃）
# Python 3.7+ TextIO 均带 reconfigure；getattr 方式避免旧 typeshed 误报
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, 'reconfigure', None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')


# 硬件加速自动模式优先级: CUDA(NVDEC) → Vulkan → VAAPI
# ffmpeg -hwaccel 初始化失败时内置 software fallback，不会中断解码。
# 注: 不加 -hwaccel_output_format cuda，因 -f null 无法直接消费 GPU 帧
#      会触发 "Error initializing a simple filtergraph" 误报。
_HWACCEL_AUTO_FLAGS = ['-hwaccel', 'cuda']

# GPU 并发信号量 (thread 模式下生效，限制同时跑 GPU 的任务数)
_GPU_SEMAPHORE = None


# ═══════════════════════════════════════════════
# 1. 核心检查函数（保持原有逻辑不变）
# ═══════════════════════════════════════════════

def check_frames_packets(path, hwaccel='off'):
    """断言 nb_read_frames == nb_read_packets（帧数守恒，无静默丢帧）。

    Packets: 始终用 ffprobe -count_packets（读包头 O(1)，无需解码）。
    Frames: hwaccel='cuda' 或 'auto' 时使用 ffmpeg hwaccel null 管线（NVDEC 解码），
      解析 stderr 末行 frame= N 得到精确帧数；ffprobe 本身不支持 -hwaccel。
      GPU 失败时自动回退 CPU ffprobe -count_frames。
    """
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None, None, 'ffprobe 不可用（PATH 缺失）'

    # packets: always ffprobe -count_packets (O(1), read packet headers only)
    r_pkts = _run([ffprobe, '-v', 'error', '-count_packets',
                   '-select_streams', 'v:0',
                   '-show_entries', 'stream=nb_read_packets',
                   '-of', 'csv=p=0', str(path)])
    m_pkts = re.match(r'(\d+)', r_pkts.stdout.strip())
    if not m_pkts:
        return None, None, 'ffprobe packets 解析失败: %r' % r_pkts.stdout[:200]
    packets = int(m_pkts.group(1))

    # frames: hwaccel-aware
    use_gpu = hwaccel in ('cuda', 'auto')
    frames = None

    if use_gpu:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            # ffprobe 不支持 -hwaccel → 用 ffmpeg NVDEC/null 管线等效替代
            hwargs = (['-hwaccel', 'cuda'] if hwaccel == 'cuda' else _HWACCEL_AUTO_FLAGS)
            cmd = [ffmpeg, '-hide_banner'] + hwargs + \
                  ['-i', str(path), '-an', '-f', 'null', '-']
            try:
                r = _run(cmd)
                # 从 stderr 末行解析 frame= N（等价于 ffprobe nb_read_frames）
                for line in reversed((r.stderr or '').splitlines()):
                    m = re.search(r'frame=\s*(\d+)', line)
                    if m:
                        frames = int(m.group(1))
                        break
            except Exception:
                pass

    if frames is None:
        # fallback: ffprobe -count_frames (CPU 软解)，-threads auto 启用多线程解码
        r_fr = _run([ffprobe, '-v', 'error', '-threads', 'auto', '-count_frames',
                     '-select_streams', 'v:0',
                     '-show_entries', 'stream=nb_read_frames',
                     '-of', 'csv=p=0', str(path)])
        m_fr = re.match(r'(\d+)', r_fr.stdout.strip())
        if not m_fr:
            return None, None, 'ffprobe frames 解析失败: %r' % r_fr.stdout[:200]
        frames = int(m_fr.group(1))

    return frames, packets, None


def extract_annexb_es(path):
    """ffmpeg -c:v copy -bsf:v h264_mp4toannexb 提取 H.264 Annex B ES 字节。"""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg 不可用（PATH 缺失）')
    r = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path),
                        '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb',
                        '-f', 'h264', 'pipe:1'],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg 提取 ES 失败: %s' % r.stderr.decode('utf-8', 'replace')[:500])
    return r.stdout


class _BitReader:
    """RBSP 位读取器（MSB-first），构造时去除 emulation prevention bytes（0x000003）。"""
    def __init__(self, data):
        # 去除 start_code_emulation_prevention: 0x00 00 03 xx -> 0x00 00 xx
        rbsp = bytearray()
        zeros = 0
        for b in data:
            if zeros >= 2 and b == 0x03:
                zeros = 0
                continue
            if b == 0:
                zeros += 1
            else:
                zeros = 0
            rbsp.append(b)
        self.data = bytes(rbsp)
        self.pos = 0  # bit offset

    def read_bit(self):
        if self.pos >= len(self.data) * 8:
            raise ValueError('RBSP 位越界')
        bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def read_bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_ue(self):
        # exp-golomb ue(v)
        leading = 0
        while self.read_bit() == 0:
            leading += 1
        if leading > 31:
            raise ValueError('ue(v) 溢出')
        return (1 << leading) - 1 + (self.read_bits(leading) if leading else 0)


def _parse_sps(payload):
    """解析 SPS RBSP → (frame_num_bits, separate_colour_plane_flag)。"""
    br = _BitReader(payload)
    profile_idc = br.read_bits(8)
    br.read_bits(8)  # constraint_set0..5 + reserved_zero_2bits
    br.read_bits(8)  # level_idc
    br.read_ue()     # seq_parameter_set_id
    separate_colour_plane = False
    if profile_idc in (44, 83, 86, 100, 110, 118, 122, 128, 134, 135, 138, 139, 244):
        separate_colour_plane = bool(br.read_bit())
    log2_max_frame_num_minus4 = br.read_ue()
    return log2_max_frame_num_minus4 + 4, separate_colour_plane


def _parse_slice_frame_num(payload, frame_num_bits, separate_colour_plane):
    """解析 slice header 的 frame_num（按 SPS 位宽精确读取）。"""
    br = _BitReader(payload)
    br.read_ue()  # first_mb_in_slice
    br.read_ue()  # slice_type
    br.read_ue()  # pic_parameter_set_id
    if separate_colour_plane:
        br.read_bits(2)  # colour_plane_id
    return br.read_bits(frame_num_bits)


def parse_h264_es(es):
    """解析 Annex B ES → [(nal_type, frame_num), ...]（仅 VCL/SPS/PPS/IDR）。"""
    nals = []
    start = 0
    n = len(es)
    frame_num_bits = 8  # 默认值；遇到 SPS 后按 log2_max_frame_num_minus4 更新
    separate_colour_plane = False
    while start < n:
        if es[start:start + 4] == b'\x00\x00\x00\x01':
            sc_len = 4
        elif es[start:start + 3] == b'\x00\x00\x01':
            sc_len = 3
        else:
            start += 1
            continue
        end = start + sc_len
        while end + 4 <= n and not (es[end:end + 4] == b'\x00\x00\x00\x01' or es[end:end + 3] == b'\x00\x00\x01'):
            end += 1
        payload = es[start + sc_len:end]
        if len(payload) >= 2:
            header = payload[0]
            nal_type = header & 0x1F
            if nal_type == 7:  # SPS: 更新 frame_num 位宽
                try:
                    frame_num_bits, separate_colour_plane = _parse_sps(payload[1:])
                except ValueError:
                    pass
            if nal_type in (1, 5):  # VCL slice: 按 SPS 位宽精确解析 frame_num
                try:
                    frame_num = _parse_slice_frame_num(payload[1:], frame_num_bits, separate_colour_plane)
                except ValueError:
                    frame_num = payload[1] & 0xFF  # 兜底近似（解析异常时）
            else:
                frame_num = payload[1] & 0xFF  # 非 VCL 占位（统计时被忽略）
            nals.append((nal_type, frame_num))
        start = end
    return nals, frame_num_bits


# NOTE: frame_num 严格按 SPS log2_max_frame_num_minus4 位宽解 slice header。
# 早期版本曾用 RBSP 第 2 字节近似——但该字节实际是 first_mb_in_slice/slice_type/
# pps_id 的位域，frame_num 并不在其中，对部分码流会产生大量误报回退；
# 现改为完整 slice header 解析（含 emulation prevention 去除与
# separate_colour_plane_flag 处理）。
def check_nal_stats(nals, frame_num_bits, dump_path=None):
    idr_count = sum(1 for t, _ in nals if t == 5)  # 5: IDR slice
    first_idr_at = next((i for i, (t, _) in enumerate(nals) if t == 5), None)
    # 段首连 IDR 检查：首个 IDR 后 32 NAL 内是否又出现 IDR
    consecutive_idr_after_first = 0
    if first_idr_at is not None:
        for t, _ in nals[first_idr_at + 1:first_idr_at + 33]:
            if t == 5:
                consecutive_idr_after_first += 1
    # frame_num 回退检查：帧号应在 GOP 内单调递增，允许 +1 跳跃但禁止小幅递减。
    # 注意两点：(1) frame_num 位宽有限（log2_max_frame_num_minus4 可很小，如 4 位
    # 每 16 帧回绕一次），满位宽回绕是合法编码；(2) 每个 IDR 开启新 GOP，frame_num
    # 重置为 0 也是合法行为。故仅对非 IDR slice 之间检查"小幅减少"（回退量 < 半程）
    half = 1 << (frame_num_bits - 1)
    regress = 0
    prev_fn = None
    for t, fn in nals:
        if t == 5:
            prev_fn = fn  # IDR: 新 GOP 起点，frame_num 重置为 0 合法，重新锚定
        elif t == 1 and prev_fn is not None:
            diff = prev_fn - fn
            if 0 < diff < half:
                regress += 1
            prev_fn = fn
    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as f:
            for i, (t, fn) in enumerate(nals):
                f.write('%d,%d,%d\n' % (i, t, fn))
    return {'idr_count': idr_count,
            'first_idr_at': first_idr_at,
            'idr_within_32_after_first': consecutive_idr_after_first,
            'frame_num_regress': regress}


def check_pts_anomaly(path, hwaccel='off'):
    """ffmpeg -v verbose: 检出 pts_anomaly / 解码错误行。

    hwaccel='cuda' / 'auto' 时优先尝试硬件解码；若 hwaccel 初始化失败
    （stderr 无 frame=N，即无实际解码输出），自动回退 CPU -threads auto。
    hwaccel='off' 时使用纯软件解码，默认启用多线程。
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return None

    hw_flags = None
    if hwaccel == 'cuda':
        hw_flags = ['-hwaccel', 'cuda']
    elif hwaccel == 'auto':
        hw_flags = _HWACCEL_AUTO_FLAGS

    if hw_flags:
        # GPU 路径: 先尝试硬件解码
        r = _run([ffmpeg, '-v', 'verbose'] + hw_flags +
                 ['-i', str(path), '-f', 'null', '-'])
        # 若 hwaccel 初始化失败导致无实际解码（stderr 中无 frame=N），
        # 回退 CPU 软解。判断依据: ffmpeg 正常解码必定输出 frame=N 进度行。
        if 'frame=' not in (r.stderr or ''):
            r = _run([ffmpeg, '-v', 'verbose', '-threads', 'auto',
                      '-i', str(path), '-f', 'null', '-'])
    else:
        # CPU 路径
        r = _run([ffmpeg, '-v', 'verbose', '-threads', 'auto',
                  '-i', str(path), '-f', 'null', '-'])

    issues = []
    for line in r.stderr.splitlines():
        low = line.lower()
        # 排除 ffmpeg 正常统计汇总行（"N packets read ...; N frames decoded; N decode errors;"），
        # 其中 "decode errors" 字面量会误命中 'error' 关键字导致误报
        if 'packets read' in low and 'frames decoded' in low:
            continue
        # 排除硬件加速设备初始化/探测失败（非视频码流错误）。
        if any(kw in low for kw in ('avhwdevicecontext', 'instance creation failure')):
            continue
        # 排除输出/filtergraph 初始化失败（muxer 兼容性问题，非解码错误）。
        if 'error' in low and any(kw in low for kw in ('opening output', 'filtergraph')):
            continue
        if 'pts_anomaly' in low or 'non-existing' in low or 'error' in low:
            issues.append(line.strip()[:200])
    return issues


# ═══════════════════════════════════════════════
# 2. 系统资源探测（跨平台 + 容器感知）
# ═══════════════════════════════════════════════

def _read_text_strip(path: str) -> str:
    """读取文件并 strip，失败返回 None。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int(path: str) -> int:
    text = _read_text_strip(path)
    if text is None:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _parse_cpuset_count(cpuset_text: str) -> int:
    """解析 cpuset.cpus 如 '0-3,5,7-9'，返回核心数。"""
    if not cpuset_text:
        return 0
    count = 0
    for part in cpuset_text.split(','):
        part = part.strip()
        if '-' in part:
            try:
                a, b = part.split('-', 1)
                count += int(b) - int(a) + 1
            except Exception:
                continue
        elif part:
            try:
                count += 1
            except Exception:
                continue
    return count


def detect_cpu_count() -> int:
    """
    容器感知的 CPU 核数探测。
      - cgroup v2: /sys/fs/cgroup/cpu.max
      - cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us / cpu.cfs_period_us
      - 无 quota 时回退 cpuset（v1/v2 路径），最后再回退 os.cpu_count()
    """
    # cgroup v2
    cpu_max = _read_text_strip('/sys/fs/cgroup/cpu.max')
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0].lower() != 'max':
            try:
                quota = int(parts[0])
                period = int(parts[1])
                if period > 0 and quota > 0:
                    return max(1, math.ceil(quota / period))
            except Exception:
                pass
    # cgroup v1
    quota = _read_int('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = _read_int('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota is not None and period and period > 0 and quota > 0:
        return max(1, math.ceil(quota / period))
    # cpuset 回退
    for cpuset_path in (
        '/sys/fs/cgroup/cpuset/cpuset.cpus',
        '/sys/fs/cgroup/cpuset.cpus.effective',
        '/sys/fs/cgroup/cpuset.cpus',
    ):
        cpuset = _read_text_strip(cpuset_path)
        if cpuset:
            cnt = _parse_cpuset_count(cpuset)
            if cnt > 0:
                return cnt
    return os.cpu_count() or 1


def detect_ram_gb():
    """
    跨平台 + 容器感知探测内存，返回 (total_gb, avail_gb)；失败返回 (0.0, 0.0)。
    优先 psutil；Windows 回退 GlobalMemoryStatusEx；Linux 回退 /proc/meminfo → sysconf。
    再用 cgroup v1/v2 memory.limit/usage 修正容器限制。
    """

    def _host_ram():
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.total / 1e9, vm.available / 1e9
        except Exception:
            pass
        if sys.platform == 'win32':
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / 1e9, stat.ullAvailPhys / 1e9
            except Exception:
                return 0.0, 0.0
        else:
            try:
                with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                    meminfo = f.read()
                total = int(re.search(r'MemTotal:\s+(\d+)\s*kB', meminfo).group(1)) / 1e6
                avail = int(re.search(r'MemAvailable:\s+(\d+)\s*kB', meminfo).group(1)) / 1e6
                return total, avail
            except Exception:
                try:
                    page_size = os.sysconf('SC_PAGE_SIZE')
                    return (os.sysconf('SC_PHYS_PAGES') * page_size / 1e9,
                            os.sysconf('SC_AVPHYS_PAGES') * page_size / 1e9)
                except Exception:
                    return 0.0, 0.0

    def _cgroup_ram():
        limit_gb = None
        usage_gb = None
        # cgroup v2
        limit_text = _read_text_strip('/sys/fs/cgroup/memory.max')
        if limit_text is not None:
            if limit_text.lower() != 'max':
                try:
                    limit_gb = int(limit_text) / 1e9
                except Exception:
                    pass
            usage_text = _read_text_strip('/sys/fs/cgroup/memory.current')
            if usage_text:
                try:
                    usage_gb = int(usage_text) / 1e9
                except Exception:
                    pass
            return limit_gb, usage_gb
        # cgroup v1
        limit_text = _read_text_strip('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if limit_text is not None:
            try:
                limit_gb = int(limit_text) / 1e9
            except Exception:
                pass
            usage_text = _read_text_strip('/sys/fs/cgroup/memory/memory.usage_in_bytes')
            if usage_text:
                try:
                    usage_gb = int(usage_text) / 1e9
                except Exception:
                    pass
        return limit_gb, usage_gb

    total, avail = _host_ram()
    cg_limit, cg_usage = _cgroup_ram()
    if cg_limit is not None:
        total = min(total, cg_limit) if total > 0 else cg_limit
        if cg_usage is not None:
            avail = max(0.0, total - cg_usage)
        else:
            avail = min(avail, total) if avail > 0 else total
    return total, avail


def compute_auto_workers(task_ram_mb: int = 300, reserve_ratio: float = 0.10) -> int:
    """
    按容器感知后的系统资源自动计算并行 workers：
      workers = min(容器可用 CPU, RAM总计*(1-预留比例) / 单任务内存估计)
    默认预留 10% 给系统/其他进程。
    """
    cpu = detect_cpu_count()
    total_gb, avail_gb = detect_ram_gb()
    by_ram = cpu
    if total_gb > 0:
        usable_gb = total_gb * (1 - reserve_ratio)
    elif avail_gb > 0:
        usable_gb = avail_gb * 0.9
    else:
        usable_gb = 0.0
    usable_mb = max(0.0, usable_gb * 1024)
    if usable_mb > 0:
        by_ram = max(1, int(usable_mb // max(1, task_ram_mb)))
    return max(1, min(cpu, by_ram))


# ═══════════════════════════════════════════════
# 3. GPU 探测与 --gpu-workers 自动计算
# ═══════════════════════════════════════════════

def detect_gpu_name():
    """检测首个 NVIDIA GPU 型号名称，失败返回 None。

    优先 pynvml；其次 nvidia-smi --query-gpu=name。
    """
    # 方式 1: pynvml（可选依赖 nvidia-ml-py；无 GPU 开发机通常未安装）
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8', 'replace')
            return name
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # 方式 2: nvidia-smi CLI
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace',
        )
        if r.returncode == 0:
            first_line = (r.stdout or '').strip().split('\n')[0].strip()
            if first_line:
                return first_line
    except Exception:
        pass

    return None


def compute_gpu_workers(gpu_name=None, hwaccel_mode='auto'):
    """根据 GPU 型号自动计算 --gpu-workers 建议值。

    原则: 按 GPU 架构代际自动估计最大 NVDEC 解码会话数，留 ~25% 余量
          给其他 GPU 任务（CUDA compute / NVENC 编码 / 显存争用）。

    Tier 分级 (名称子串匹配，大小写不敏感):

      High-end (Ada/Hopper/高端 Ampere, ~7-8 NVDEC 会话) → 6 workers
        A100, H100, H200, L40, L40S, A40, A6000, RTX 6000 Ada

      Mid-range (Ampere/Lovelace, ~5-6 NVDEC 会话) → 4 workers
        A10, A5000, A4000, L4, RTX 4090/4080/4070/4060,
        RTX 3090/3080/3070/3060, A30, A16

      Standard (Turing/Volta/Pascal, ~3-4 NVDEC 会话) → 2 workers
        T4, V100, P100, P4, P40, T4G, RTX 2080/2070/2060, GTX 1080/1070

      未知/检测失败 → 2 workers (保守)

    仅在 hwaccel=cuda/auto 时有效；hwaccel=off 时返回 0（GPU 不参与）。
    用户可通过 --gpu-workers N 显式覆盖。
    """
    if hwaccel_mode not in ('cuda', 'auto'):
        return 0

    if gpu_name is None:
        gpu_name = detect_gpu_name() or ''

    name_upper = gpu_name.upper()

    # High-end: Ada Lovelace / Hopper / high-end Ampere  (7-8 NVDEC sessions)
    _HIGH_END = {
        'A100', 'H100', 'H200', 'L40S', 'L40', 'A40', 'A6000',
        'RTX 6000 ADA', 'RTX 5000 ADA',
    }
    # Mid-range: Ampere / Lovelace desktop / L4  (5-6 NVDEC sessions)
    _MID_RANGE = {
        'A10', 'A5000', 'A4000', 'L4', 'A30', 'A16', 'A2',
        'RTX 4090', 'RTX 4080', 'RTX 4070', 'RTX 4060',
        'RTX 3090', 'RTX 3080', 'RTX 3070', 'RTX 3060', 'RTX 3050',
    }
    # Standard: Turing / Volta / Pascal  (3-4 NVDEC sessions)
    _STANDARD = {
        'T4', 'V100', 'P100', 'P40', 'P4', 'T4G',
        'RTX 2080', 'RTX 2070', 'RTX 2060',
        'GTX 1080', 'GTX 1070', 'GTX 1060', 'GTX 1660', 'GTX 1650',
        'QUADRO RTX', 'QUADRO P',
    }

    for key in _HIGH_END:
        if key in name_upper:
            return 6
    for key in _MID_RANGE:
        if key in name_upper:
            return 4
    for key in _STANDARD:
        if key in name_upper:
            return 2

    # 未知 GPU: 保守值 2
    return 2


# ═══════════════════════════════════════════════
# 4. 文件扫描
# ═══════════════════════════════════════════════

_VIDEO_EXTENSIONS = {'.mp4', '.h264', '.mkv', '.avi', '.mov', '.m4v', '.webm'}


def _glob_expand(pattern):
    """展开单条路径中的通配符 (* ? [])，返回匹配到的已存在 Path 列表。
    无通配符或零匹配 → 空列表。使用 glob.glob 兼容 Windows/Linux 分隔符。
    """
    if '*' in pattern or '?' in pattern or '[' in pattern:
        import glob as _g
        matches = _g.glob(pattern, recursive=('**' in pattern))
        if matches:
            return [Path(m).resolve() for m in sorted(matches) if Path(m).exists()]
    return []


def _scan_dir_for_video(dir_path, seen):
    """扫描目录下的视频文件，返回 [(Path, label), ...] 并更新 seen 集合。"""
    found = []
    for ext in sorted(_VIDEO_EXTENSIONS):
        for match in sorted(dir_path.rglob('*' + ext)):
            s = str(match.resolve())
            if s not in seen:
                try:
                    label = str(match.relative_to(dir_path))
                except ValueError:
                    label = match.name
                found.append((match.resolve(), label))
                seen.add(s)
        for match in sorted(dir_path.rglob('*' + ext.upper())):
            s = str(match.resolve())
            if s not in seen:
                try:
                    label = str(match.relative_to(dir_path))
                except ValueError:
                    label = match.name
                found.append((match.resolve(), label))
                seen.add(s)
    return found


def collect_video_files(paths):
    """自动识别文件/文件夹/通配符，递归收集所有视频文件。

    - 单个文件 → 直接加入
    - 文件夹 → 递归扫描匹配扩展名的文件
    - 通配符 (如 segment_00*.mp4, ./test/*/video.*) → glob 展开后按文件/文件夹处理
    - 多路径混合 → 合并后按路径排序，去重
    返回 [(Path, 显示标签), ...]
    """
    files = []
    seen = set()

    for p in paths:
        pp = Path(p).resolve()

        if pp.is_file():
            s = str(pp)
            if s not in seen:
                files.append((pp, pp.name))
                seen.add(s)

        elif pp.is_dir():
            files.extend(_scan_dir_for_video(pp, seen))

        else:
            # 路径不存在 → 尝试通配符展开 (支持 Windows cmd/PowerShell 不展开 *)
            expanded = _glob_expand(p)
            if expanded:
                for ep in expanded:
                    if ep.is_file():
                        s = str(ep)
                        if s not in seen:
                            files.append((ep, ep.name))
                            seen.add(s)
                    elif ep.is_dir():
                        files.extend(_scan_dir_for_video(ep, seen))
            else:
                print(f"[WARN] 路径不存在或非文件/文件夹，已跳过: {p}")

    return files


# ═══════════════════════════════════════════════
# 5. 单文件验收（模块级函数，ThreadPoolExecutor 可 pickle）
# ═══════════════════════════════════════════════

def _verify_one_video(video_path, hwaccel='auto', dump_nal=None):
    """对单个视频执行全部 3 项检查，返回结构化结果。

    GPU 任务受 _GPU_SEMAPHORE 闸门限制 (thread 模式)；
    process 模式信号量不可跨进程共享，由 --gpu-workers 上限约束。

    返回 dict:
        path:           输入路径字符串
        label:          显示标签（传参传入，无则用文件名）
        frames:         nb_read_frames
        packets:        nb_read_packets
        fp_match:       frames == packets
        fp_err:         检查 1 错误描述 (None=通过)
        nal_stats:      check_nal_stats 返回的 dict (None=解析失败)
        nal_err:        检查 2 错误描述 (None=通过)
        pts_issues:     check_pts_anomaly 返回的 list (None=跳过/失败)
        pts_err:        检查 3 错误描述 (None=通过)
        elapsed:        总耗时 (秒)
        pass_all:       全部 3 项检查通过
        fails:          失败项列表 ['frames/packets', 'nal', 'pts']
    """
    gpu_sem = _GPU_SEMAPHORE
    gpu_task = hwaccel in ('cuda', 'auto')
    if gpu_task and gpu_sem is not None:
        gpu_sem.acquire()

    t0 = time.monotonic()
    result = {
        'path': video_path,
        'label': Path(video_path).name,
        'frames': None,
        'packets': None,
        'fp_match': None,
        'fp_err': None,
        'nal_stats': None,
        'nal_err': None,
        'pts_issues': None,
        'pts_err': None,
        'fails': [],
        'pass_all': False,
    }

    try:
        # ── 1) 帧数守恒 ──
        frames, packets, err = check_frames_packets(video_path, hwaccel)
        result['frames'] = frames
        result['packets'] = packets
        if err:
            result['fp_err'] = err
            result['fails'].append('frames/packets')
        elif frames is not None and packets is not None:
            result['fp_match'] = (frames == packets)
            if not result['fp_match']:
                result['fp_err'] = 'frames(%d) != packets(%d)' % (frames, packets)
                result['fails'].append('frames/packets')

        # ── 2) NAL 统计 ──
        try:
            es = extract_annexb_es(video_path)
            nals, frame_num_bits = parse_h264_es(es)
            stats = check_nal_stats(nals, frame_num_bits, dump_nal)
            result['nal_stats'] = stats
            if stats['idr_within_32_after_first'] > 0:
                result['nal_err'] = '段首出现连续 IDR (%d 个，修复前 per-slot IDR 特征)' % stats['idr_within_32_after_first']
                result['fails'].append('nal')
            if stats['frame_num_regress'] > 0:
                msg = 'frame_num 回退 %d 次（LA 重路由错位特征）' % stats['frame_num_regress']
                if result['nal_err']:
                    result['nal_err'] += '; ' + msg
                else:
                    result['nal_err'] = msg
                if 'nal' not in result['fails']:
                    result['fails'].append('nal')
        except Exception as e:
            result['nal_err'] = 'NAL 解析失败: %s' % e
            result['fails'].append('nal')

        # ── 3) pts_anomaly / 解码错误 ──
        issues = check_pts_anomaly(video_path, hwaccel)
        result['pts_issues'] = issues
        if issues is None:
            result['pts_err'] = 'ffmpeg 不可用（无法执行 pts 检查）'
            result['fails'].append('pts')
        elif len(issues) > 0:
            result['pts_err'] = 'ffmpeg verbose 检出 %d 条异常' % len(issues)
            result['fails'].append('pts')

        result['pass_all'] = len(result['fails']) == 0

    finally:
        if gpu_task and gpu_sem is not None:
            gpu_sem.release()

    result['elapsed'] = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════
# 6. 并行执行引擎
# ═══════════════════════════════════════════════

def run_verify_parallel(video_files, hwaccel='auto', dump_nal=None,
                         workers=1, parallel_mode='thread',
                         gpu_workers=0) -> list:
    """
    并行执行全部视频文件的 3 项检查，返回按输入顺序排列的结果列表。
    video_files: [(Path, label), ...]
    """
    total = len(video_files)
    if total == 0:
        return []

    # 构建任务列表
    tasks = []
    for fpath, label in video_files:
        tasks.append({
            'path': str(fpath),
            'label': label,
            'hwaccel': hwaccel,
            'dump_nal': dump_nal,
        })

    results_map = {}  # path -> result
    workers = max(1, min(workers, total))
    t_batch = time.time()

    if workers == 1:
        # 串行模式（保持原有进度打印风格）
        for i, t in enumerate(tasks, 1):
            print(f"\n── [{i}/{total}] {t['label']} ──")
            res = _verify_one_video(t['path'], hwaccel=t['hwaccel'],
                                     dump_nal=t['dump_nal'])
            res['label'] = t['label']
            results_map[t['path']] = res
            _print_one_result(res, i, total)
    else:
        executor_cls = ProcessPoolExecutor if parallel_mode == 'process' else ThreadPoolExecutor
        done = 0
        with executor_cls(max_workers=workers) as ex:
            fut2task = {}
            for t in tasks:
                fut2task[ex.submit(_verify_one_video, t['path'],
                                   t['hwaccel'], t['dump_nal'])] = t
            for fut in as_completed(fut2task):
                t = fut2task[fut]
                done += 1
                try:
                    res = fut.result()
                    res['label'] = t['label']
                    results_map[t['path']] = res
                except Exception as e:
                    results_map[t['path']] = {
                        'path': t['path'], 'label': t['label'],
                        'frames': None, 'packets': None, 'fp_match': None,
                        'fp_err': '并行执行异常: %s' % e,
                        'nal_stats': None, 'nal_err': None,
                        'pts_issues': None, 'pts_err': None,
                        'fails': ['frames/packets', 'nal', 'pts'],
                        'pass_all': False, 'elapsed': 0,
                    }
                _print_one_result(results_map[t['path']], done, total)

    total_elapsed = time.time() - t_batch
    print(f"\n  批次完成: {total} 文件, 用时 {_fmt_elapsed(total_elapsed)}")

    # 按输入顺序返回
    ordered = [results_map[str(fpath)] for fpath, _ in video_files]
    return ordered


def _print_one_result(res: dict, idx: int, total: int):
    """打印单文件验收结果的紧凑摘要。"""
    label = res.get('label', Path(res['path']).name)
    elapsed = _fmt_elapsed(res.get('elapsed', 0))
    if res['fp_match']:
        ch1 = ' [1] frames=packets=%d OK' % res['frames']
    elif res['fp_err']:
        ch1 = ' [1] %s' % res['fp_err'][:80]
    else:
        ch1 = ' [1] frames=%s packets=%s' % (res['frames'], res['packets'])

    if res['nal_stats']:
        s = res['nal_stats']
        ch2 = ' [2] IDR=%d 连IDR=%d FN回退=%d' % (
            s['idr_count'], s['idr_within_32_after_first'], s['frame_num_regress'])
    elif res['nal_err']:
        ch2 = ' [2] %s' % res['nal_err'][:60]
    else:
        ch2 = ' [2] N/A'

    if res['pts_issues'] is not None and len(res['pts_issues']) == 0:
        ch3 = ' [3] 无异常'
    elif res['pts_err']:
        ch3 = ' [3] %s' % res['pts_err'][:60]
    else:
        ch3 = ' [3] N/A'

    status = 'PASS' if res['pass_all'] else 'FAIL'
    print(f"[{idx}/{total}] {status} {label} ({elapsed})")
    print(f"      {ch1}{ch2}{ch3}")


def _print_summary_table(results: list):
    """打印终末汇总表。"""
    n_total = len(results)
    n_pass = sum(1 for r in results if r['pass_all'])
    n_fail = n_total - n_pass

    # 按失败类型统计
    fp_fail = sum(1 for r in results if 'frames/packets' in r['fails'])
    nal_fail = sum(1 for r in results if 'nal' in r['fails'])
    pts_fail = sum(1 for r in results if 'pts' in r['fails'])

    print("\n" + "=" * 72)
    print("  验收汇总")
    print("=" * 72)
    print("  文件总数: %d  |  通过: %d  |  失败: %d" % (n_total, n_pass, n_fail))
    if fp_fail or nal_fail or pts_fail:
        print("  失败明细:")
        if fp_fail:
            print("    检查1 [frames/packets 不匹配]: %d 文件" % fp_fail)
        if nal_fail:
            print("    检查2 [NAL 异常/连IDR/帧号回退]: %d 文件" % nal_fail)
        if pts_fail:
            print("    检查3 [pts_anomaly/解码错误]: %d 文件" % pts_fail)

    # 列出全部失败文件 (分行显示，避免单行被截断)
    if n_fail > 0:
        print("\n  失败文件列表:")
        for r in results:
            if not r['pass_all']:
                label = r.get('label', Path(r['path']).name)
                print("    %s" % label)
                if r['fp_err']:
                    print("        FP : %s" % r['fp_err'])
                if r['nal_err']:
                    print("        NAL: %s" % r['nal_err'])
                if r['pts_err']:
                    print("        PTS: %s" % r['pts_err'])

    print("=" * 72)


# ═══════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='插帧段码流完整性验收 v2（支持文件/文件夹自动识别 + 并行批处理）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
参数说明：
  paths     待验收的视频文件或文件夹（支持混合传入，结尾带/不带/均合法）。
            文件夹自动递归扫描 *.mp4 *.h264 *.mkv *.avi *.mov *.m4v *.webm。

并发双闸门：
  --workers      "总并发流水线数"：控制 ThreadPoolExecutor 的总线程数，
                 决定最多同时处理多少文件（含 CPU + GPU 任务）。
                 默认按 min(容器CPU核数, RAM可用量/300MB) 自动计算。
  --gpu-workers  "GPU 并发闸门"：控制可同时占用 NVDEC 硬件解码的任务数，
                 通过 BoundedSemaphore 限制（仅 --hwaccel cuda/auto + thread 模式生效）。
                 默认根据 GPU 型号自动最大化计算（T4→2, A10→4, A100→6）。
                 关系: --gpu-workers <= --workers，防止 GPU NVDEC 会话耗尽。

示例：
  python tests/verify_segment_bitstream.py a.mp4
  python tests/verify_segment_bitstream.py ./segments --hwaccel cuda --gpu-workers 4
  python tests/verify_segment_bitstream.py a.mp4 ./dir1/ b.mp4 --workers 8 --hwaccel auto
''',
    )
    ap.add_argument('paths', nargs='+',
                    help='待验收的视频文件或文件夹（支持混合传入）')
    ap.add_argument('--dump-nal', default=None,
                    help='导出 NAL (nal_type,frame_num) 到文件（仅单文件模式有效；'
                         '多文件时会被忽略以免覆盖）')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--hwaccel', choices=['auto', 'cuda', 'off'], default='auto',
                    help='硬件解码加速: auto(自动, 默认) | cuda(NVDEC) | off(纯软件)')
    ap.add_argument('--workers', type=int, default=None,
                    help='总并发流水线数 (默认: 按 CPU 核数/可用RAM 自动计算)')
    ap.add_argument('--parallel', choices=['auto', 'thread', 'process'], default='auto',
                    help='并行模式: auto=thread | thread=多线程 | process=多进程')
    ap.add_argument('--gpu-workers', type=int, default=None,
                    help='GPU 任务并发闸门 (默认: 按 GPU 型号自动最大化；设 0 禁用 GPU 闸门)')
    ap.add_argument('--task-ram-mb', type=int, default=300,
                    help='自动 workers 时每任务内存估计 MB (默认 300)')
    args = ap.parse_args()

    # ── 环境检查 ──
    if shutil.which('ffprobe') is None:
        print("[WARN] PATH 中未找到 ffprobe，ffprobe 统计任务将失败")
    if shutil.which('ffmpeg') is None:
        print("[WARN] PATH 中未找到 ffmpeg，pts_anomaly 与帧数 GPU 解码将失败")

    # ── 文件扫描 ──
    video_files = collect_video_files(args.paths)
    if not video_files:
        sys.exit('未发现任何视频文件，退出')
    print("发现 %d 个视频文件" % len(video_files))

    # ── 单文件兼容：保持原有进度输出风格 ──
    single_file = len(video_files) == 1

    # ── GPU 硬件加速自检 ──
    hwaccel_mode = args.hwaccel
    gpu_enabled = hwaccel_mode in ('cuda', 'auto')
    if hwaccel_mode == 'cuda':
        # 快速自检: ffmpeg 是否有 cuda hwaccel
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'],
                               capture_output=True, text=True, timeout=10)
            if 'cuda' not in (r.stdout or '').lower():
                print("[WARN] ffmpeg 未编译 CUDA hwaccel，回退 CPU 软解")
                hwaccel_mode = 'off'
                gpu_enabled = False
        except Exception:
            print("[WARN] ffmpeg 不可用，回退纯 CPU")
            hwaccel_mode = 'off'
            gpu_enabled = False
    elif hwaccel_mode == 'auto':
        # auto 模式: 探测 NVIDIA GPU 是否存在
        gpu_name_pre = detect_gpu_name()
        if gpu_name_pre is None:
            print("[WARN] 未检测到 NVIDIA GPU，--hwaccel auto 回退 CPU 软解 (可用 --hwaccel off 跳过探测)")
            hwaccel_mode = 'off'
            gpu_enabled = False

    # ── 系统资源探测与并行参数 ──
    parallel_mode = 'thread' if args.parallel == 'auto' else args.parallel
    if gpu_enabled and parallel_mode == 'process':
        print("[WARN] GPU 启用时 --parallel process 不兼容 GPU 并发闸门，自动切换为 thread")
        parallel_mode = 'thread'

    cpu_count = detect_cpu_count()
    cpu_host = os.cpu_count() or 1
    ram_total, ram_avail = detect_ram_gb()
    ram_desc = "总计 %.1fGB / 可用 %.1fGB" % (ram_total, ram_avail) if ram_total > 0 else "未知"

    workers = args.workers if (args.workers and args.workers > 0) else compute_auto_workers(args.task_ram_mb)
    cpu_note = " (宿主机 %d)" % cpu_host if cpu_count != cpu_host else ""
    print("系统资源: CPUx%d%s | RAM %s" % (cpu_count, cpu_note, ram_desc))

    # ── GPU 型号探测与 --gpu-workers 自动计算 ──
    gpu_name = None
    gpu_workers = 0
    if gpu_enabled:
        gpu_name = detect_gpu_name()
        if args.gpu_workers is not None:
            # 用户显式指定
            gpu_workers = max(0, args.gpu_workers)
            gpu_source = 'user'
        else:
            # 自动按 GPU 型号计算
            gpu_workers = compute_gpu_workers(gpu_name, hwaccel_mode)
            gpu_source = 'auto'
        gpu_name_str = ' (%s)' % gpu_name if gpu_name else ''
        gpu_note = " | GPU%s hwaccel=%s gpu_workers=%d(%s)" % (
            gpu_name_str, hwaccel_mode, gpu_workers, gpu_source)
    else:
        gpu_note = ""
    print("并行配置: workers=%d | 模式=%s | 单任务RAM估计=%dMB%s" %
          (workers, parallel_mode, args.task_ram_mb, gpu_note))

    # ── 初始化 GPU 并发信号量 ──
    # 上限兜底: 不论自动计算还是用户指定，不超过 8（单 GPU NVDEC 硬件上限）
    gpu_workers = min(gpu_workers, 8)
    global _GPU_SEMAPHORE
    if gpu_enabled and parallel_mode == 'thread' and gpu_workers > 0:
        _GPU_SEMAPHORE = threading.BoundedSemaphore(gpu_workers)

    # ── 多文件模式下禁用 --dump-nal（避免相互覆盖）──
    dump_nal = args.dump_nal if single_file else None
    if not single_file and args.dump_nal:
        print("[WARN] 多文件模式下 --dump-nal 已忽略（避免覆盖），单文件模式可用")

    # ── 执行验收 ──
    t_start = time.monotonic()
    if single_file:
        # 单文件模式：保持原有三步骤详细信息输出
        fpath, label = video_files[0]
        print("\n" + "=" * 60)
        print(" 验收: %s" % label)
        print(" 路径: %s" % fpath)
        print("=" * 60)

        res = _verify_one_video(str(fpath), hwaccel=hwaccel_mode, dump_nal=dump_nal)
        res['label'] = label

        # 详细输出（兼容原来风格）
        if res['fp_err']:
            print('[1] frames/packets 检查失败: %s' % res['fp_err'])
        else:
            match_str = 'OK' if res['fp_match'] else 'MISMATCH'
            print('[1] frames=%d packets=%d %s' % (res['frames'] or 0, res['packets'] or 0, match_str))

        if res['nal_stats']:
            s = res['nal_stats']
            print('[2] IDR=%d 首个IDR@%d 其后32NAL内IDR=%d frame_num回退=%d' % (
                s['idr_count'], s['first_idr_at'] if s['first_idr_at'] is not None else -1,
                s['idr_within_32_after_first'], s['frame_num_regress']))
        if res['nal_err']:
            print('[2] NAL 异常: %s' % res['nal_err'])

        if res['pts_issues'] is not None and len(res['pts_issues']) == 0:
            print('[3] 无 pts_anomaly / 解码错误 OK')
        elif res['pts_err']:
            print('[3] %s' % res['pts_err'])
            if res['pts_issues']:
                for line in res['pts_issues'][:5]:
                    print('     ' + line)

        total = time.monotonic() - t_start
        if res['pass_all']:
            print('\n✅ 验收通过 (总用时 %s): 帧数守恒 / 段首单 IDR / frame_num 单调 / 无 pts_anomaly' % _fmt_elapsed(total))
        else:
            print('\n❌ 验收未通过 (总用时 %s):' % _fmt_elapsed(total))
            if res['fp_err']:
                print('  - %s' % res['fp_err'])
            if res['nal_err']:
                print('  - %s' % res['nal_err'])
            if res['pts_err']:
                print('  - %s' % res['pts_err'])
            sys.exit(1)
    else:
        # 多文件并行模式
        print("\n" + "=" * 60)
        print(" 并行批处理验收: %d 文件 | workers=%d" % (len(video_files), workers))
        print("=" * 60)

        results = run_verify_parallel(video_files, hwaccel=hwaccel_mode,
                                       dump_nal=dump_nal,
                                       workers=workers,
                                       parallel_mode=parallel_mode,
                                       gpu_workers=gpu_workers)
        _print_summary_table(results)

        total = time.monotonic() - t_start
        n_pass = sum(1 for r in results if r['pass_all'])
        n_fail = len(results) - n_pass
        if n_fail > 0:
            print('\n❌ %d/%d 文件未通过验收 (总用时 %s)' % (n_fail, len(results), _fmt_elapsed(total)))
            sys.exit(1)
        else:
            print('\n✅ 全部 %d 文件通过验收 (总用时 %s)' % (len(results), _fmt_elapsed(total)))


if __name__ == '__main__':
    main()
