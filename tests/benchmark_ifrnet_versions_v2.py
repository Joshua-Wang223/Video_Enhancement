#!/usr/bin/env python3
"""
IFRNet 版本对比基准测试 v2（精准帧数检测 + PDF 报告）
================================================================
基于 benchmark_ifrnet_versions.py 升级：

  v2 新增：
    1. 精准帧数检测：参考 analyze_video_pipeline_v3.py，所有版本处理完成后
       统一进行 ffprobe -count_frames 全量解码 + ffmpeg -f image2pipe 管线
       捕获 dup/drop/解码错误。
    2. GPU 初筛：--verify-hwaccel auto|cuda|off（默认 auto），NVDEC 加速解码；
       默认 GPU 闸门 --verify-workers=4，防止 NVDEC 会话耗尽。
    3. 可选 CPU 复检：--cpu-recheck 开启，对异常输出（errors/dup>100/|帧差|>2）
       用纯 CPU 软件解码复检对比。
    4. 原有控制台表格新增列：期望帧/精准解码帧/帧差/dup/drop/解码错误数/解码器；
       FPS 优先基于精准解码帧数重算；JSON 报告增加 verify 字段。
    5. PDF 报告：reportlab 生成，含概要、主对比表、帧完整性验证表、
       Pipeline/NVENC/编码级别诊断表、FPS/耗时/文件大小/GPU% 条形图、结论。

  v1 原有功能（保持不变）：
    对同一视频分别调用 v6.3.5 / v6.4.1–v6.4.5.2 进行插帧测试，
    收集耗时、FPS、输出帧数、文件大小、GPU 利用率等指标，输出对比表格。

用法:
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/ --scale 4 --model IFRNet_L_Vimeo90K
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/ --versions v6.4.5.1 --no-warmup
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib import colors, pagesizes
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.textlabels import Label
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("[WARNING] reportlab 未安装，将不会生成 PDF 报告。安装: pip install reportlab")


# ═══════════════════════════════════════════════════════════════════════════
# 多语言输出辅助（仅影响用户可见的表头/报告文本，内部错误信息保持英文）
# ═══════════════════════════════════════════════════════════════════════════

_I18N = {
    'zh': {
        'title': 'IFRNet 版本基准测试结果 v2',
        'input': '输入',
        'scale': '插帧',
        'batch_size': 'batch-size',
        'model': '模型',
        'crf': 'crf',
        'preset': 'preset',
        'version': '版本',
        'time': '耗时',
        'input_frames': '输入帧',
        'output_frames': '输出帧',
        'decoded_frames': '精准帧',
        'expected_frames': '期望帧',
        'frame_delta': '帧差',
        'dup': 'dup',
        'drop': 'drop',
        'errors': '错误',
        'container_frames': '容器帧',
        'fps': 'FPS',
        'file_size': '文件大小',
        'rc_mode': '码率模式',
        'bitrate': '平均码率',
        'gpu_util': 'GPU%',
        'gpu_mem': '显存',
        'encoder': '编码器',
        'fastest_version': '最快版本',
        'performance_spread': '性能差异',
        'time_range': '耗时范围',
        'frame_integrity': '帧完整性验证',
        'frame_integrity_hint': '期望帧 = (输入帧 - 1) × scale + 1；精准帧为 ffprobe -count_frames 全量解码结果。',
        'status': '状态',
        'decoder': '解码器',
        'no_verify_data': '未启用精准检测或没有成功结果。',
        'pipeline_diagnosis': 'Pipeline 帧计数诊断',
        'pipeline_hint': '管内期望 = infer对 × scale；总期望 = 外部首帧 + 管内期望；文件Δ = ffprobe − 总期望。',
        'reader_pairs': 'reader对',
        'infer_batches': 'infer批',
        'infer_pairs': 'infer对',
        'writer_frames': 'writer帧',
        'pipe_expected': '管内期望',
        'pipe_delta': '管内偏移',
        'total_expected': '总期望',
        'file_delta': '文件Δ',
        'nvenc_diagnosis': 'NVENC 编码精度诊断',
        'gpu_batches': 'GPU批次',
        'encoded_frames': '编码输出帧',
        'empty_h264': '空H264',
        'lock_fallback': 'Lock降级',
        'encoder_level': '编码级别诊断',
        'level': '级别',
        'description': '说明',
        'level_map': {1: "NVENC GPU直通", 2: "RingBuf+NVENC", 3: "RingBuf+软编码", 4: "标准路径"},
        'unknown': '未知',
        'performance_charts': '性能图表',
        'fps_chart': 'FPS 对比',
        'time_chart': '耗时对比 (s)',
        'size_chart': '输出文件大小对比 (MB)',
        'gpu_chart': 'GPU 利用率对比 (%)',
        'conclusion': '结论',
        'frame_integrity_summary': '帧完整性',
        'normal': '正常',
        'warn': '警告',
        'error': '异常',
        'no_success': '没有成功的测试版本。',
        'pdf_title': 'IFRNet 版本基准测试报告 v2',
        'input_video': '输入视频',
        'interp_scale': '插帧倍数',
        'generated_at': '生成时间',
        'total_elapsed': '总耗时（含检测）',
        'main_comparison': '1. 主对比表',
        'time_s': '耗时(s)',
        'size_mb': '大小(MB)',
        'bitrate_kbps': '码率(kbps)',
        'vram_mb': '显存(MB)',
        'not_verified': '未检测',
        'fastest_vs_slowest': '（最快 vs 最慢）',
        'sec_frame_integrity': '2. 帧完整性验证',
        'sec_pipeline_diagnosis': '3. Pipeline 帧计数诊断',
        'sec_nvenc_diagnosis': '4. NVENC 编码精度诊断',
        'sec_encoder_level': '5. 编码级别诊断',
        'sec_performance_charts': '6. 性能图表',
        'sec_conclusion': '7. 结论',
    },
    'en': {
        'title': 'IFRNet Version Benchmark Results v2',
        'input': 'Input',
        'scale': 'Scale',
        'batch_size': 'Batch Size',
        'model': 'Model',
        'crf': 'CRF',
        'preset': 'Preset',
        'version': 'Version',
        'time': 'Time',
        'input_frames': 'Input',
        'output_frames': 'Output',
        'decoded_frames': 'Decoded',
        'expected_frames': 'Expected',
        'frame_delta': 'Delta',
        'dup': 'dup',
        'drop': 'drop',
        'errors': 'Errors',
        'container_frames': 'Container',
        'fps': 'FPS',
        'file_size': 'Size',
        'rc_mode': 'RC Mode',
        'bitrate': 'Bitrate',
        'gpu_util': 'GPU%',
        'gpu_mem': 'VRAM',
        'encoder': 'Encoder',
        'fastest_version': 'Fastest Version',
        'performance_spread': 'Performance Spread',
        'time_range': 'Time Range',
        'frame_integrity': 'Frame Integrity Verification',
        'frame_integrity_hint': 'Expected = (input_frames - 1) × scale + 1; Decoded = ffprobe -count_frames full decode.',
        'status': 'Status',
        'decoder': 'Decoder',
        'no_verify_data': 'Precise verification disabled or no successful results.',
        'pipeline_diagnosis': 'Pipeline Frame Count Diagnosis',
        'pipeline_hint': 'Pipe expected = infer_pairs × scale; Total expected = external_first + pipe_expected; File Δ = ffprobe − total_expected.',
        'reader_pairs': 'Reader Pairs',
        'infer_batches': 'Infer Batches',
        'infer_pairs': 'Infer Pairs',
        'writer_frames': 'Writer Frames',
        'pipe_expected': 'Pipe Exp',
        'pipe_delta': 'Pipe Δ',
        'total_expected': 'Total Exp',
        'file_delta': 'File Δ',
        'nvenc_diagnosis': 'NVENC Encoding Accuracy Diagnosis',
        'gpu_batches': 'GPU Batches',
        'encoded_frames': 'Encoded Frames',
        'empty_h264': 'Empty H264',
        'lock_fallback': 'Lock Fallback',
        'encoder_level': 'Encoder Level Diagnosis',
        'level': 'Level',
        'description': 'Description',
        'level_map': {1: "NVENC GPU passthrough", 2: "RingBuf+NVENC", 3: "RingBuf+software", 4: "Standard path"},
        'unknown': 'Unknown',
        'performance_charts': 'Performance Charts',
        'fps_chart': 'FPS Comparison',
        'time_chart': 'Elapsed Time Comparison (s)',
        'size_chart': 'Output File Size Comparison (MB)',
        'gpu_chart': 'GPU Utilization Comparison (%)',
        'conclusion': 'Conclusion',
        'frame_integrity_summary': 'Frame Integrity',
        'normal': 'OK',
        'warn': 'Warn',
        'error': 'Error',
        'no_success': 'No successful benchmark versions.',
        'pdf_title': 'IFRNet Version Benchmark Report v2',
        'input_video': 'Input Video',
        'interp_scale': 'Interpolation Scale',
        'generated_at': 'Generated At',
        'total_elapsed': 'Total Elapsed (incl. verification)',
        'main_comparison': '1. Main Comparison',
        'time_s': 'Time(s)',
        'size_mb': 'Size(MB)',
        'bitrate_kbps': 'Bitrate(kbps)',
        'vram_mb': 'VRAM(MB)',
        'not_verified': 'Not verified',
        'fastest_vs_slowest': '(fastest vs slowest)',
        'sec_frame_integrity': '2. Frame Integrity Verification',
        'sec_pipeline_diagnosis': '3. Pipeline Frame Count Diagnosis',
        'sec_nvenc_diagnosis': '4. NVENC Encoding Accuracy Diagnosis',
        'sec_encoder_level': '5. Encoder Level Diagnosis',
        'sec_performance_charts': '6. Performance Charts',
        'sec_conclusion': '7. Conclusion',
    }
}


def _t(key: str, lang: str = 'zh') -> str:
    """获取指定语言的翻译文本。"""
    return _I18N.get(lang, _I18N['zh']).get(key, key)


def _t_level(level: int, lang: str = 'zh') -> str:
    """获取编码级别描述。"""
    return _I18N.get(lang, _I18N['zh']).get('level_map', {}).get(level, f"{_t('unknown', lang)}({level})")


def _normalize_lang(lang: str) -> str:
    """统一语言参数为 'zh' 或 'en'。"""
    if lang in ('chinese', 'cn', 'zh'):
        return 'zh'
    return 'en'


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 脚本在 tests/ 子目录，需回退一层到项目根
IFRNET_DIR = PROJECT_ROOT / "external" / "IFRNet"
MODELS_DIR = PROJECT_ROOT / "models_IFRNet" / "checkpoints"

VERSIONS = [
    "process_video_v6_3_5_single",
    "process_video_v6_4_1_single",
    "process_video_v6_4_2_single",
    "process_video_v6_4_3_single",
    "process_video_v6_4_3_1_single",
    "process_video_v6_4_3_2_single",
    "process_video_v6_4_4_single",
    "process_video_v6_4_4_1_single",
    "process_video_v6_4_4_2_single",
    "process_video_v6_4_5_single",
    "process_video_v6_4_5_1_single",
    "process_video_v6_4_5_2_single",
]

VERSION_LABELS = {
    "process_video_v6_3_5_single": "v6.3.5",
    "process_video_v6_4_1_single": "v6.4.1",
    "process_video_v6_4_2_single": "v6.4.2",
    "process_video_v6_4_3_single": "v6.4.3",
    "process_video_v6_4_3_1_single": "v6.4.3.1",
    "process_video_v6_4_3_2_single": "v6.4.3.2",
    "process_video_v6_4_4_single": "v6.4.4",
    "process_video_v6_4_4_1_single": "v6.4.4.1",
    "process_video_v6_4_4_2_single": "v6.4.4.2",
    "process_video_v6_4_5_single": "v6.4.5",
    "process_video_v6_4_5_1_single": "v6.4.5.1",
    "process_video_v6_4_5_2_single": "v6.4.5.2",
}


# ═══════════════════════════════════════════════
# 0. GPU 基础设施与错误关键词（移植自 analyze_video_pipeline_v3.py）
# ═══════════════════════════════════════════════

# GPU 解码器错误关键词（与软件解码器错误字符串不同，需显式加入）
# 注意: 禁用裸设备名 'nvenc'/'nvdec'/'cuvid' —— 会误匹配 ffmpeg 信息行中的
# 编码器/解码器名称 (如 "h264 (h264_nvenc)" stream mapping 行、
# "encoder : LavcXX h264_nvenc" 元数据行)，导致每个文件恒定 errors=2 假阳性。
# 必须使用只在真实错误中出现、不会出现在设备名提及中的具体短语。
_GPU_DECODE_ERROR_KW = [
    'failed to decode',              # NVDEC: "hardware accelerator failed to decode picture"
    'hardware accelerator',          # Cuvid/NVDEC 硬件解码器报错前缀
    'cuda device',                   # CUDA 设备错误
    'cannot init cuvid',             # NVDEC 初始化失败
    'cannot load libnvcuvid',        # NVDEC 运行时库加载失败
    'no nvenc capable devices',      # NVENC 设备不可用 ("No NVENC capable devices found")
    'openencodesessionex failed',    # NVENC 会话创建失败 (OOM/会话耗尽)
    'encodepicture failed',          # NVENC 编码单帧失败
    'nvenc api version',             # NVENC API/驱动版本不兼容
    'vdpau',                         # VDPAU backend error
]


def _build_ffmpeg_decode_err_kw() -> list:
    """合并软件解码器 + GPU 解码器错误关键词列表。"""
    base = [
        'illegal', 'error', 'missing', 'corrupt',
        'duplicated', 'duplicate', 'decoder',
        'non-existing pps', 'no frame',
        'more than 1000 frames duplicated',
    ]
    return base + _GPU_DECODE_ERROR_KW


# ═══════════════════════════════════════════════════════════════════════════
# 系统资源探测（移植自 analyze_video_pipeline_v3.py）
# ═══════════════════════════════════════════════════════════════════════════

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
    os.cpu_count() 在容器里通常返回宿主机核心数，因此优先读 cgroup。
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


def _detect_cgroup_memory_gb():
    """
    读取 cgroup 内存限制与已用量（GB），返回 (limit_gb, usage_gb)。
    如果未设置限制，limit_gb 为 None；'max' 或极大值也表示无限制。
    """
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


def _detect_host_ram_gb():
    """
    跨平台探测物理内存，返回 (total_gb, avail_gb)；失败返回 (0.0, 0.0)。
    优先 psutil；Windows 回退 GlobalMemoryStatusEx；Linux 回退 /proc/meminfo → sysconf。
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        pass
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


def detect_ram_gb():
    """
    跨平台 + 容器感知探测内存，返回 (total_gb, avail_gb)；失败返回 (0.0, 0.0)。
      - 宿主机：psutil 优先；Windows 回退 GlobalMemoryStatusEx；Linux 回退 /proc/meminfo/sysconf
      - 容器：再用 cgroup v1/v2 的 memory.limit/usage 修正，避免读到宿主机全部内存
    """
    total, avail = _detect_host_ram_gb()
    cg_limit, cg_usage = _detect_cgroup_memory_gb()
    if cg_limit is not None:
        # 如果 limit 极大（如 2^63-1），说明未限制，仍用宿主机值
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
    ffprobe/ffmpeg 解码子进程本身吃 CPU，故不超过 CPU 核数；RAM 不足时自动缩减。
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


# GPU 并发信号量 (thread 模式下生效，限制同时跑 GPU 的任务数)
_GPU_SEMAPHORE = None         # 在 main() 中按 --verify-workers 初始化


def _ffmpeg_decode(video_path: str, use_image2pipe: bool = False, timeout: int = 300,
                   hwaccel: str = 'off') -> dict:
    """ffmpeg 解码（image2pipe/NVENC-null/null 三种管线），流式捕获帧数/dup/drop/错误。

    hwaccel='cuda':
      - image2pipe 模式 → NVDEC 解码 + h264_nvenc 编码 → -f null（全 GPU 管线）
      - null 模式     → NVDEC 解码 + -f null（自动 hwdownload）
    hwaccel='off': 纯 CPU 软件解码（原有行为）。
    返回码非 0 时自动回退 CPU 重跑（如 10-bit 输入 h264_nvenc 不支持）。
    """
    result = {'frame_count': None, 'errors': [], 'dup_count': None, 'drop_count': None,
              'decoder': 'cpu', 'exit_code': None}
    gpu_attempted = False
    cmd = None

    if hwaccel == 'cuda':
        gpu_attempted = True
        result['decoder'] = 'gpu'
        if use_image2pipe:
            # 全 GPU 管线：NVDEC decode + NVENC encode + null mux
            # dup/drop 由 NVENC 编码路径产生，等同于 image2pipe 语义
            cmd = ["ffmpeg", "-hide_banner",
                   "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                   "-i", str(video_path),
                   "-c:v", "h264_nvenc", "-f", "null", "-"]
        else:
            # GPU 解码 + null mux（自动 hwdownload）
            cmd = ["ffmpeg", "-hide_banner",
                   "-hwaccel", "cuda",
                   "-i", str(video_path), "-f", "null", "-"]

    if not gpu_attempted:
        # 纯 CPU 路径
        if use_image2pipe:
            cmd = ["ffmpeg", "-hide_banner", "-i", str(video_path),
                   "-f", "image2pipe", "-vcodec", "png", "-"]
        else:
            cmd = ["ffmpeg", "-hide_banner", "-i", str(video_path), "-f", "null", "-"]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        try:
            for line in proc.stderr:
                stripped = line.strip()
                m = re.search(r'frame=\s*(\d+)', stripped)
                if m:
                    result['frame_count'] = int(m.group(1))
                m_dup = re.search(r'dup=(\d+)', stripped)
                if m_dup:
                    result['dup_count'] = int(m_dup.group(1))
                m_drop = re.search(r'drop=(\d+)', stripped)
                if m_drop:
                    result['drop_count'] = int(m_drop.group(1))
                lower = stripped.lower()
                if any(kw in lower for kw in _build_ffmpeg_decode_err_kw()):
                    result['errors'].append(stripped)
            proc.wait(timeout=timeout)
            result['exit_code'] = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result['errors'].append("[TIMEOUT] ffmpeg 解码超时")
            result['exit_code'] = -1
    except FileNotFoundError:
        result['errors'].append("[SKIP] ffmpeg 不可用")
        result['exit_code'] = -2

    # GPU 失败自动回退 CPU 重跑
    if gpu_attempted and result['exit_code'] is not None and result['exit_code'] != 0:
        if use_image2pipe:
            cmd2 = ["ffmpeg", "-hide_banner", "-i", str(video_path),
                    "-f", "image2pipe", "-vcodec", "png", "-"]
        else:
            cmd2 = ["ffmpeg", "-hide_banner", "-i", str(video_path), "-f", "null", "-"]
        result2 = {'frame_count': None, 'errors': [], 'dup_count': None, 'drop_count': None,
                   'decoder': 'cpu', 'exit_code': None}
        result2['errors'].append("[GPU-FALLBACK] GPU 解码失败，回退 CPU 软件解码重跑")
        try:
            proc2 = subprocess.Popen(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
            try:
                for line in proc2.stderr:
                    stripped = line.strip()
                    m = re.search(r'frame=\s*(\d+)', stripped)
                    if m:
                        result2['frame_count'] = int(m.group(1))
                    m_dup = re.search(r'dup=(\d+)', stripped)
                    if m_dup:
                        result2['dup_count'] = int(m_dup.group(1))
                    m_drop = re.search(r'drop=(\d+)', stripped)
                    if m_drop:
                        result2['drop_count'] = int(m_drop.group(1))
                    lower = stripped.lower()
                    if any(kw in lower for kw in _build_ffmpeg_decode_err_kw()):
                        result2['errors'].append(stripped)
                proc2.wait(timeout=timeout)
                result2['exit_code'] = proc2.returncode
            except subprocess.TimeoutExpired:
                proc2.kill()
                proc2.wait()
                result2['errors'].append("[TIMEOUT] ffmpeg 解码超时(CPU回退)")
        except Exception:
            result2['errors'].append("[SKIP] CPU 回退也失败")
        # 保留 GPU 阶段的错误信息，追加入 CPU 结果前面
        gpu_errs_saved = [e for e in result['errors'] if not e.startswith('[GPU-FALLBACK]')]
        result = result2
        if gpu_errs_saved:
            fallback_idx = next((i for i, e in enumerate(result['errors'])
                                 if e.startswith('[GPU-FALLBACK]')), 0)
            for e in reversed(gpu_errs_saved):
                result['errors'].insert(fallback_idx + 1, f"[GPU] {e}")

    return result


def _ffprobe_count(video_path: str, count_mode: str = 'packets', timeout: int = 60,
                   hwaccel: str = 'off') -> str:
    """获取帧/包计数。count_mode: 'packets' | 'frames'

    CPU 路径: ffprobe -count_packets / -count_frames (packets 读包头 O(1); frames 逐帧软解)
    GPU 路径 (frames 模式, hwaccel='cuda'):
        ffprobe 本身不支持 -hwaccel，改用等效的 ffmpeg -hwaccel cuda -f null - 管线，
        解析 stderr 终末行的 frame= N 得到精确解码帧数（等价于 nb_read_frames），
        真正走 NVDEC 解码。GPU 失败时自动回退 CPU ffprobe。
    """
    gpu_attempted = (count_mode == 'frames' and hwaccel == 'cuda')

    if gpu_attempted:
        # ffprobe 无 -hwaccel 选项 → 用 ffmpeg NVDEC null 管线等效替代
        cmd = ["ffmpeg", "-hide_banner",
               "-hwaccel", "cuda",
               "-i", str(video_path),
               "-an", "-f", "null", "-"]
    else:
        flag = '-count_packets' if count_mode == 'packets' else '-count_frames'
        entry = 'stream=nb_read_packets' if count_mode == 'packets' else 'stream=nb_read_frames'
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", flag,
               "-show_entries", entry, "-of", "csv=p=0", str(video_path)]

    def _parse_output(result) -> str:
        """从 ffprobe stdout / ffmpeg stderr 提取帧数。
        优先级: stdout 纯数字 > stderr 'frame=N' 尾值（ffmpeg null 管线）> stdout strip > [ERROR]
        """
        stdout = (result.stdout or '').strip()
        stderr = (result.stderr or '').strip()
        # ffprobe 正常: stdout 为纯数字
        if stdout and re.fullmatch(r'\d+', stdout):
            return stdout
        # ffmpeg null 管线: stderr 末行含 frame= N
        for line in reversed(stderr.splitlines()):
            m = re.search(r'frame=\s*(\d+)', line)
            if m:
                return m.group(1)
        # 兜底: 返回非空文本（含错误提示），不把 stderr 垃圾数字误当帧数
        return stdout or "[ERROR]"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        val = _parse_output(result)

        # GPU 失败 → 回退 CPU ffprobe
        if gpu_attempted and (val == "[ERROR]" or result.returncode != 0):
            flag_fb = '-count_frames'
            entry_fb = 'stream=nb_read_frames'
            cmd2 = ["ffprobe", "-v", "error", "-select_streams", "v:0", flag_fb,
                    "-show_entries", entry_fb, "-of", "csv=p=0", str(video_path)]
            try:
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=timeout)
                val2 = _parse_output(result2)
                return val2 if val2 != "[ERROR]" else "[ERROR]"
            except Exception:
                return val  # CPU 回退也失败，返回 GPU 错误描述
        return val
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except FileNotFoundError:
        return "[SKIP: ffprobe/ffmpeg not found]"


def detect_gpu_hwaccel(ffmpeg_bin: str = 'ffmpeg') -> dict:
    """GPU 硬件加速能力自检。

    返回 {'cuda': True/False, 'nvenc': True/False, 'hint': str}
      - cuda: ffmpeg 列出 cuda hwaccel + NVDEC 解码 1 帧成功
      - nvenc: h264_nvenc 编码器可用
      任一 False → 全局回退纯 CPU。

    自检策略（无临时文件）：
      - NVDEC: lavfi → libx264 → -f h264 pipe（Annex B 比特流自带帧边界，
        无 rawvideo pipe 截断问题）
      - NVENC: lavfi → h264_nvenc → null（直接测试编码器，单条命令）
    """
    result = {'cuda': False, 'nvenc': False, 'hint': ''}

    # 1. Check hwaccel support
    try:
        r = subprocess.run([ffmpeg_bin, '-hide_banner', '-hwaccels'],
                           capture_output=True, text=True, timeout=10)
        if 'cuda' not in r.stdout.lower():
            result['hint'] = 'ffmpeg 未编译 CUDA hwaccel 支持'
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result['hint'] = 'ffmpeg 不可用'
        return result

    # 2. Micro self-test: NVDEC decode 1 black frame
    #    使用 -f h264 pipe（Annex B 比特流带 start code 帧边界），
    #    避免 -f rawvideo pipe 截断导致 packet corrupt / Invalid buffer size 假阴性。
    try:
        r = subprocess.run([
            ffmpeg_bin, '-hide_banner', '-y',
            '-f', 'lavfi', '-i', 'color=c=black:s=160x160:d=1:r=10',
            '-frames:v', '10', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-f', 'h264', 'pipe:1'
        ], capture_output=True, text=False, timeout=20)
        if r.returncode != 0:
            result['hint'] = 'ffmpeg 无法生成测试流'
            return result

        # NVDEC decode from H.264 pipe (stdin)
        r2 = subprocess.run([
            ffmpeg_bin, '-hide_banner', '-hwaccel', 'cuda',
            '-f', 'h264', '-i', 'pipe:0', '-frames:v', '1', '-f', 'null', '-'
        ], input=r.stdout, capture_output=True, text=False, timeout=20)
        if r2.returncode != 0:
            result['hint'] = 'NVDEC 自检失败（驱动/CUDA 可能不可用）'
            return result
        result['cuda'] = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        result['hint'] = 'NVDEC 自检超时或 ffmpeg 缺失'
        return result

    # 3. nvenc self-test: 直接测试 h264_nvenc（lavfi → NVENC → null）
    #    步骤 2 已验证 NVDEC，此处仅需验证 NVENC 编码器即可覆盖全 GPU 管线。
    #    单条命令，无 pipe、无临时文件 — 比原 rawvideo pipe 方案更简洁可靠。
    try:
        r = subprocess.run([
            ffmpeg_bin, '-hide_banner',
            '-f', 'lavfi', '-i', 'color=c=black:s=160x160:d=1:r=10',
            '-frames:v', '10', '-c:v', 'h264_nvenc', '-f', 'null', '-'
        ], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            result['nvenc'] = True
        else:
            result['hint'] = 'NVENC 编码器自检失败 (GeForce 可能会话已耗尽、容器缺设备)'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if result['cuda'] and not result['hint']:
        #  仅 ffmpeg; ffprobe 不支持 hwaccel，frames 统计改用等效 ffmpeg NVDEC null 管线
        result['hint'] = 'CUDA' + ('+NVENC' if result['nvenc'] else '') + ' GPU 加速可用'
    return result


def _to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _execute_task(task: dict):
    """执行单个探测/解码任务。

    GPU 任务受 _GPU_SEMAPHORE 闸门限制 (thread 模式)。
    """
    gpu_sem = _GPU_SEMAPHORE
    gpu_task = task.get('hwaccel', 'off') == 'cuda'
    if gpu_task and gpu_sem is not None:
        gpu_sem.acquire()

    start = time.time()
    try:
        hw = task.get('hwaccel', 'off')
        if task['kind'] in ('ffprobe', 'ffprobe_recheck'):
            return _ffprobe_count(task['path'], task['mode'], task['timeout'],
                                  hwaccel=hw), start
        # ffmpeg / ffmpeg_recheck
        return _ffmpeg_decode(task['path'],
                              use_image2pipe=task.get('image2pipe', False),
                              timeout=task.get('timeout', 300),
                              hwaccel=hw), start
    finally:
        if gpu_task and gpu_sem is not None:
            gpu_sem.release()


def _task_error_result(task: dict, exc: Exception):
    """任务异常时生成与正常结果同结构的兜底值。"""
    if task['kind'] == 'ffprobe':
        return f"[ERROR: {exc}]"
    return {'frame_count': None, 'errors': [f"[ERROR] 并行执行异常: {exc}"],
            'dup_count': None, 'drop_count': None}


def _print_task_progress(done: int, total: int, task: dict, result,
                         task_duration: float):
    """逐任务进度打印。"""
    name = Path(task['path']).name
    if task['kind'] == 'ffprobe':
        print(f"  [{done}/{total}] ffprobe/{task['mode']} {name}: {result}  (用时 {task_duration:.1f}s)")
    elif isinstance(result, dict):
        print(f"  [{done}/{total}] ffmpeg {name}: frame={result.get('frame_count')}, "
              f"dup={result.get('dup_count')}, drop={result.get('drop_count')}, "
              f"errors={len(result.get('errors', []))}  (用时 {task_duration:.1f}s)")
    else:
        print(f"  [{done}/{total}] ffmpeg {name}: {result}  (用时 {task_duration:.1f}s)")


def run_tasks_parallel(tasks: list, workers: int, phase_label: str = '') -> dict:
    """并行执行任务列表，返回 {task['key']: result}（固定 thread 模式）。"""
    results = {}
    total = len(tasks)
    if total == 0:
        return results
    workers = max(1, min(workers, total))
    t_batch = time.time()

    if workers == 1:
        for i, t in enumerate(tasks, 1):
            t0 = time.time()
            try:
                res, _ = _execute_task(t)
                results[t['key']] = res
            except Exception as e:
                results[t['key']] = _task_error_result(t, e)
            _print_task_progress(i, total, t, results[t['key']], time.time() - t0)
        return results

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2task = {}
        for t in tasks:
            fut2task[ex.submit(_execute_task, t)] = t
        for fut in as_completed(fut2task):
            t = fut2task[fut]
            try:
                res, real_start = fut.result()
                results[t['key']] = res
                task_dur = time.time() - real_start
            except Exception as e:
                results[t['key']] = _task_error_result(t, e)
                task_dur = time.time() - t_batch
            done += 1
            _print_task_progress(done, total, t, results[t['key']], task_dur)
    return results


def _collect_anomaly_outputs_for_recheck(results: list, prefetched: dict,
                                         use_image2pipe: bool = False) -> list:
    """对 GPU 初筛后含异常的输出视频生成 CPU 软件解码复检任务。

    异常判定：errors 非空 / dup > 100 / |帧差| > 2
    """
    recheck_tasks = []
    for r in results:
        if not r.get('success'):
            continue
        v = r.get('verify')
        if not v:
            continue
        path = r.get('output_path', '')
        if not path or not Path(path).exists():
            continue
        dup = v.get('dup')
        delta = v.get('frame_delta')
        errors = v.get('errors', [])
        if errors or (dup is not None and dup > 100) or (delta is not None and abs(delta) > 2):
            resolved_path = str(Path(path).resolve())
            key = ('ffmpeg_recheck', resolved_path, use_image2pipe)
            if key not in prefetched:
                recheck_tasks.append({
                    'kind': 'ffmpeg_recheck',
                    'key': key,
                    'path': resolved_path,
                    'image2pipe': use_image2pipe,
                    'timeout': 300,
                    'hwaccel': 'off',
                })
    return recheck_tasks


def _run_verify(results: list, input_path: str, scale: float,
                hwaccel: str = 'auto', workers=None,
                cpu_recheck: bool = False,
                task_ram_mb: int = 300,
                reserve_ratio: float = 0.10) -> dict:
    """运行精准检测阶段，返回 {path: verify_dict}。

    输入视频只做一次 frames 精准统计，各版本输出分别做 frames + image2pipe。
    workers=None 时按容器感知 CPU/内存自动计算；GPU 任务额外受 _GPU_SEMAPHORE 限制。
    """
    if not results:
        return {}

    ok_results = [r for r in results if r.get('success')]
    if not ok_results:
        return {}

    # 系统资源探测
    cpu_count = detect_cpu_count()
    cpu_host = os.cpu_count() or 1
    ram_total, ram_avail = detect_ram_gb()
    ram_desc = f"总计 {ram_total:.1f}GB / 可用 {ram_avail:.1f}GB" if ram_total > 0 else "未知"

    # workers 自动计算
    if workers is None or workers == 'auto':
        workers = compute_auto_workers(task_ram_mb, reserve_ratio)
    else:
        workers = max(1, int(workers))

    # GPU 自检与 GPU workers 动态上限
    gpu_caps = {'cuda': False, 'nvenc': False, 'hint': '未检测'}
    if hwaccel != 'off':
        gpu_caps = detect_gpu_hwaccel()
    use_gpu = (hwaccel == 'cuda' and gpu_caps['cuda']) or (
        hwaccel == 'auto' and gpu_caps['cuda']
    )
    hwaccel_mode = 'cuda' if use_gpu else 'off'

    # GPU 并发上限：根据 NVDEC 能力动态调整
    if use_gpu:
        # GeForce 消费卡 NVDEC 会话通常 1-2；Tesla T4/A100/L4 可更高
        gpu_workers_max = 8
        try:
            hint_l = gpu_caps.get('hint', '').lower()
            if any(x in hint_l for x in ('geforce', 'rtx', 'gtx')):
                gpu_workers_max = 2
            elif any(x in hint_l for x in ('tesla', 'a100', 'a6000', 'l4', 'l40')):
                gpu_workers_max = 8
        except Exception:
            pass
        gpu_workers = max(1, min(workers, gpu_workers_max))
    else:
        gpu_workers = 0

    cpu_note = f" (宿主机 {cpu_host})" if cpu_count != cpu_host else ""
    print(f"   系统资源: CPU x{cpu_count}{cpu_note} | RAM {ram_desc}")
    gpu_info = ""
    if use_gpu:
        gpu_info = f" | GPU {gpu_caps.get('hint', 'CUDA')} (gpu_workers={gpu_workers})"
    print(f"   并行配置: workers={workers} | 单任务内存估计={task_ram_mb}MB | 预留={reserve_ratio*100:.0f}%{gpu_info}")

    if use_gpu:
        print(f"   GPU 检测可用: {gpu_caps.get('hint', 'CUDA')}")
    else:
        hint = gpu_caps.get('hint', '')
        print(f"   使用 CPU 软件解码检测" + (f" ({hint})" if hint else ""))

    # 收集任务
    tasks = []
    seen = set()

    def add_task(kind, path, mode=None, image2pipe=None):
        # key 统一使用 resolve 后的绝对路径，避免相对/绝对路径混用导致查不到结果
        resolved_path = str(Path(path).resolve()) if path else ''
        if not resolved_path or not Path(resolved_path).exists():
            return
        if kind == 'ffprobe':
            key = ('ffprobe', resolved_path, mode)
            if key in seen:
                return
            seen.add(key)
            tasks.append({
                'kind': 'ffprobe',
                'key': key,
                'path': resolved_path,
                'mode': mode,
                'timeout': 900 if mode == 'frames' else 60,
                'hwaccel': hwaccel_mode,
            })
        else:
            key = ('ffmpeg', resolved_path, image2pipe)
            if key in seen:
                return
            seen.add(key)
            tasks.append({
                'kind': 'ffmpeg',
                'key': key,
                'path': resolved_path,
                'image2pipe': image2pipe,
                'timeout': 900 if image2pipe else 300,
                'hwaccel': hwaccel_mode,
            })

    # 输入视频 frames 精准统计
    add_task('ffprobe', input_path, mode='frames')

    for r in ok_results:
        out_path = r.get('output_path', '')
        if not out_path:
            continue
        add_task('ffprobe', out_path, mode='frames')
        add_task('ffmpeg', out_path, image2pipe=True)

    if not tasks:
        return {}

    print(f"\n[VERIFY] 精准检测阶段 ({len(tasks)} 任务, workers={workers}, hwaccel={hwaccel_mode}) ...")
    t_phase = time.time()
    prefetched = run_tasks_parallel(tasks, workers, phase_label='精准检测')
    print(f"   [OK] 精准检测完成，用时 {time.time() - t_phase:.1f}s")

    # CPU 复检
    cpu_recheck_results = {}
    if cpu_recheck:
        recheck_tasks = _collect_anomaly_outputs_for_recheck(
            results, prefetched, use_image2pipe=True)
        if recheck_tasks:
            print(f"\n[RECHECK] CPU 软件解码复检 ({len(recheck_tasks)} 个异常输出, workers={workers}) ...")
            t_recheck = time.time()
            cpu_recheck_results = run_tasks_parallel(
                recheck_tasks, workers, phase_label='CPU 复检')
            print(f"   [OK] CPU 复检完成，用时 {time.time() - t_recheck:.1f}s")
        else:
            print(f"\n[OK] 无需 CPU 复检")

    # 汇总 verify 信息
    verify_map = {}
    input_key = ('ffprobe', str(Path(input_path).resolve()), 'frames')
    input_decoded_str = prefetched.get(input_key, '')
    input_decoded = _to_int(input_decoded_str)
    input_packets = None
    # 尝试用已有结果中的 input_frames（packets）作为 fallback
    for r in ok_results:
        if r.get('input_frames') is not None:
            input_packets = r['input_frames']
            break

    effective_input = input_decoded if input_decoded is not None else input_packets
    expected_frames = None
    if effective_input is not None:
        # interpolate 期望: (输入帧 - 1) * scale + 1
        expected_frames = int((effective_input - 1) * scale + 1)

    for r in ok_results:
        out_path = r.get('output_path', '')
        if not out_path:
            continue
        v = {
            'input_frames_decoded': input_decoded,
            'input_frames_packets': input_packets,
            'expected_frames': expected_frames,
            'output_frames_decoded': None,
            'frame_delta': None,
            'dup': None,
            'drop': None,
            'errors': [],
            'decoder': 'cpu',
            'cpu_recheck': None,
        }

        # ffprobe frames
        pk = ('ffprobe', str(Path(out_path).resolve()), 'frames')
        decoded_str = prefetched.get(pk, '')
        decoded = _to_int(decoded_str)
        v['output_frames_decoded'] = decoded

        # ffmpeg image2pipe
        fk = ('ffmpeg', str(Path(out_path).resolve()), True)
        ff_r = prefetched.get(fk)
        if isinstance(ff_r, dict):
            frame_count = ff_r.get('frame_count')
            dup = ff_r.get('dup_count')
            drop = ff_r.get('drop_count')
            # ffmpeg 无 dup/drop 输出时通常实际为 0，避免显示 "?"
            if frame_count is not None:
                dup = dup if dup is not None else 0
                drop = drop if drop is not None else 0
            v['dup'] = dup
            v['drop'] = drop
            v['errors'] = list(ff_r.get('errors', []))
            v['decoder'] = ff_r.get('decoder', 'cpu')

        # 帧差
        if decoded is not None and expected_frames is not None:
            v['frame_delta'] = decoded - expected_frames

        # CPU 复检
        ck = ('ffmpeg_recheck', str(Path(out_path).resolve()), True)
        cpu_r = cpu_recheck_results.get(ck)
        if isinstance(cpu_r, dict):
            cpu_frame_count = cpu_r.get('frame_count')
            cpu_dup = cpu_r.get('dup_count')
            cpu_drop = cpu_r.get('drop_count')
            if cpu_frame_count is not None:
                cpu_dup = cpu_dup if cpu_dup is not None else 0
                cpu_drop = cpu_drop if cpu_drop is not None else 0
            v['cpu_recheck'] = {
                'frame_count': cpu_frame_count,
                'dup': cpu_dup,
                'drop': cpu_drop,
                'errors': list(cpu_r.get('errors', [])),
                'decoder': cpu_r.get('decoder', 'cpu'),
            }
            v['decoder'] = 'GPU+CPU复检'

        verify_map[out_path] = v
        r['verify'] = v

    return verify_map


# ═══════════════════════════════════════════════
# 1. 模型路径与版本运行（与 v1 一致）
# ═══════════════════════════════════════════════

def find_model_path(model_name: str) -> str:
    model_file = MODELS_DIR / f"{model_name}.pth"
    if model_file.exists():
        return str(model_file)
    candidates = list(MODELS_DIR.glob(f"{model_name}*.pth"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        f"Model not found: {model_name}.pth in {MODELS_DIR}"
    )


def run_version(version_module: str, input_path: str, output_path: str,
                model_path: str, scale: float, batch_size: int,
                trt_cache_dir: str, codec: str | None = None,
                crf: int = 23, preset: str = "medium") -> dict:
    """Run a single IFRNet version in a subprocess and return timing results."""
    inner_script = f'''
import importlib
import json
import os
import subprocess as _sp
import sys
import threading
import time

_ifrnet_dir = {str(IFRNET_DIR)!r}
if _ifrnet_dir not in sys.path:
    sys.path.insert(0, _ifrnet_dir)

mod = importlib.import_module({version_module!r})

# 自动选择编码器，支持显式指定
# 参照底层 best_encoder()：libx264 → h264_nvenc，libx265 → hevc_nvenc
_codec_explicit = {codec!r}
if _codec_explicit:
    codec = _codec_explicit
else:
    codec = "libx264"
    try:
        codec = mod.HardwareCapability.best_encoder("libx264")
    except Exception:
        pass

proc = mod.IFRNetVideoProcessor(
    model_path={model_path!r},
    device="cuda",
    batch_size={batch_size},
    max_batch_size={batch_size * 2},
    use_fp16=True,
    use_compile=False,
    use_cuda_graph=False,
    use_tensorrt=True,
    trt_cache_dir={trt_cache_dir!r},
    use_hwaccel=True,
    codec=codec,
    crf={crf},
    x264_preset={preset!r},
    keep_audio=False,
    ffmpeg_bin="ffmpeg",
    quiet=True,
)

# ── 后台 GPU 监控（处理期间每秒采样，取平均值 + 峰值显存） ──────
_gpu_samples = []
_gpu_stop = threading.Event()

def _monitor_gpu():
    while not _gpu_stop.is_set():
        try:
            _p = _sp.run([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            if _p.returncode == 0:
                parts = _p.stdout.strip().split(",")
                if len(parts) >= 2:
                    _gpu_samples.append((
                        int(parts[0].strip()),
                        int(parts[1].strip()),
                    ))
        except Exception:
            pass
        _gpu_stop.wait(1.0)

_gpu_thread = threading.Thread(target=_monitor_gpu, daemon=True)
_gpu_thread.start()

t0 = time.perf_counter()

# v6.4.x 新增 total_segments/segment_index 参数（有默认值，向后兼容）
ok = proc.process_video(
    input_path={input_path!r},
    output_path={output_path!r},
    scale={scale},
    preview=False,
    preview_interval=30,
)

elapsed = time.perf_counter() - t0

# 停止 GPU 监控，汇总采样数据
_gpu_stop.set()
_gpu_thread.join(timeout=3)

result = {{
    "version": {version_module!r},
    "label": {VERSION_LABELS.get(version_module, version_module)!r},
    "success": ok,
    "elapsed_sec": round(elapsed, 3),
    "input_path": {input_path!r},
    "output_path": {output_path!r},
    "scale": {scale},
    "codec": getattr(proc, '_last_used_codec', codec),
    "model": {os.path.basename(model_path)!r},
}}

# 诊断数据：pipeline 各阶段帧/批次数（用于帧丢失排查）
_diag_keys = [
    '_diag_reader_pairs', '_diag_infer_batches',
    '_diag_infer_pairs', '_diag_writer_frames',
    '_diag_gpu_stay_batches', '_diag_nvenc_frames', '_diag_empty_h264',
    '_diag_lock_fallback',
    '_diag_active_level',
    '_diag_external_first_frame',
]
for _dk in _diag_keys:
    _dv = getattr(proc, _dk, None)
    if _dv is not None:
        result[_dk] = _dv

# [PHASE4] 额外从 NVENC encoder 收集 fallback 计数器 + 码率控制模式
_nvenc = getattr(proc, '_cached_nvenc_encoder', None)
if _nvenc is not None:
    _lfb = getattr(_nvenc, '_diag_lock_fallback', 0)
    if _lfb > 0:
        result.setdefault('_diag_lock_fallback', 0)
        result['_diag_lock_fallback'] = max(result['_diag_lock_fallback'], _lfb)
    # NVENC 码率控制模式（权威来源，不依赖 ffprobe 推测）
    _rm = getattr(_nvenc, '_rate_mode', None)
    if _rm:
        _label_map = {{"constqp": "CONSTQP", "vbr_hq": "VBR_HQ", "qvbr": "QVBR"}}
        result["output_rc_mode"] = _label_map.get(_rm, _rm.upper())
    else:
        result["output_rc_mode"] = "CONSTQP"
else:
    # ctypes NVENC encoder 引用不可达时的回退检测链
    # 1) 尝试从模块级常量 _NVENC_LEVEL1_RATE_MODE 获取 (v6.4.3.1+)
    _rm = getattr(mod, '_NVENC_LEVEL1_RATE_MODE', None)
    if _rm:
        _label_map = {{"constqp": "CONSTQP", "vbr_hq": "VBR_HQ", "qvbr": "QVBR"}}
        result["output_rc_mode"] = _label_map.get(_rm, _rm.upper())
    else:
        # 2) ffprobe 回退：区分 NVENC 管道 vs 软件编码
        _codec_str = getattr(proc, '_last_used_codec', codec)
        if 'nvenc' in (_codec_str or '').lower():
            result["output_rc_mode"] = "NVENC"
        else:
            result["output_rc_mode"] = "SW"

# GPU 利用率（处理期间平均）和峰值显存
if _gpu_samples:
    avg_util = sum(s[0] for s in _gpu_samples) / len(_gpu_samples)
    peak_mem = max(s[1] for s in _gpu_samples)
    result["gpu_util_pct"] = round(avg_util, 1)
    result["gpu_mem_used_mb"] = peak_mem
    result["gpu_samples"] = len(_gpu_samples)
else:
    # nvidia-smi 不可用时的回退快照
    try:
        _p = _sp.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=10)
        if _p.returncode == 0:
            parts = _p.stdout.strip().split(",")
            if len(parts) >= 1:
                result["gpu_util_pct"] = int(parts[0].strip())
            if len(parts) >= 2:
                result["gpu_mem_used_mb"] = int(parts[1].strip())
    except Exception:
        pass

# 收集输出文件信息
if ok and os.path.exists({output_path!r}):
    try:
        st = os.stat({output_path!r})
        result["output_size_mb"] = round(st.st_size / (1024 * 1024), 2)
        result["output_size_bytes"] = st.st_size
    except OSError:
        pass

    # 用 ffprobe 获取输出帧数（packet 级，可靠的主计数）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=30)
        if _p.returncode == 0 and _p.stdout.strip():
            val = _p.stdout.strip()
            if val != "N/A":
                result["output_frames"] = int(val)
    except Exception:
        pass

    # 容器级帧数（nb_frames 优先，无需解码；不可用时由帧率×时长推算）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=15)
        if _p.returncode == 0 and _p.stdout.strip():
            val = _p.stdout.strip()
            if val not in ("N/A", "", "0"):
                result["output_frames_nf"] = int(val)
    except Exception:
        pass

    # nb_frames 不可用时的回退：r_frame_rate × duration
    if "output_frames_nf" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip():
                parts = _p.stdout.strip().split(",")
                if len(parts) >= 2:
                    from fractions import Fraction as _Frac
                    try:
                        _rfr = float(_Frac(parts[0].strip()))
                        _dur = float(parts[1].strip())
                        if _rfr > 0 and _dur > 0:
                            result["output_frames_nf"] = int(round(_rfr * _dur))
                    except (ValueError, ZeroDivisionError):
                        pass
        except Exception:
            pass

    # 输出视频平均码率（stream 级优先，format 级回退）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=15)
        if _p.returncode == 0 and _p.stdout.strip():
            _val = _p.stdout.strip()
            if _val not in ("N/A", "", "0"):
                result["output_bitrate_kbps"] = round(int(_val) / 1000, 1)
    except Exception:
        pass

    if "output_bitrate_kbps" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=bit_rate",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip():
                _val = _p.stdout.strip()
                if _val not in ("N/A", "", "0"):
                    result["output_bitrate_kbps"] = round(int(_val) / 1000, 1)
        except Exception:
            pass

    # 码率控制模式回退：仅当无 NVENC encoder 时尝试 ffprobe
    if "output_rc_mode" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate_mode",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip() in ("0", "1", "2"):
                _mode_map = {{"0": "CBR", "1": "VBR", "2": "ABR"}}
                result["output_rc_mode"] = _mode_map[_p.stdout.strip()]
        except Exception:
            pass

# 收集输入帧数
try:
    _p = _sp.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", {input_path!r}
    ], capture_output=True, text=True, timeout=30)
    if _p.returncode == 0 and _p.stdout.strip():
        result["input_frames"] = int(_p.stdout.strip())
except Exception:
    pass

# 计算 FPS（基于 packet 级帧数，后续主进程可能用精准帧数覆盖）
if ok and elapsed > 0 and "output_frames" in result:
    result["fps"] = round(result["output_frames"] / elapsed, 1)

print("__BENCHMARK_RESULT__", json.dumps(result), flush=True)
'''
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.run(
        [sys.executable, "-c", inner_script],
        stdout=subprocess.PIPE,       # 只捕获 stdout（JSON 结果路径）
        stderr=None,                  # stderr 直通终端，使 tqdm 能正常用 \r 刷新
        text=True, timeout=3600,
        env=env, cwd=str(PROJECT_ROOT),
    )

    # 解析输出中的 JSON 结果
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("__BENCHMARK_RESULT__"):
            try:
                return json.loads(line.split("__BENCHMARK_RESULT__", 1)[1].strip())
            except json.JSONDecodeError:
                pass

    # 未找到 JSON 结果
    return {
        "version": version_module,
        "label": VERSION_LABELS.get(version_module, version_module),
        "success": False,
        "error": "no_json_output",
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]) if proc.stderr else "",
    }


# ═══════════════════════════════════════════════
# 2. 控制台输出（v2 增加精准检测列）
# ═══════════════════════════════════════════════

def _display_width(s: str) -> int:
    """字符串终端显示宽度（CJK 字符计 2，其余计 1）。"""
    w = 0
    for ch in s:
        if '一' <= ch <= '鿿' or '　' <= ch <= '〿' or '＀' <= ch <= '￯':
            w += 2
        else:
            w += 1
    return w


def _pad(s: str, width: int, align: str = '<') -> str:
    """按终端显示宽度填充字符串。"""
    dw = _display_width(s)
    pad = max(0, width - dw)
    if align == '<':
        return s + ' ' * pad
    elif align == '>':
        return ' ' * pad + s
    else:
        left = pad // 2
        return ' ' * left + s + ' ' * (pad - left)


def _anomaly_status_str(v: dict, lang: str = 'zh') -> str:
    """根据 verify 结果返回状态字符串。"""
    lang = _normalize_lang(lang)
    if not v:
        return _t('not_verified', lang)
    errors = v.get('errors', [])
    delta = v.get('frame_delta')
    dup = v.get('dup')

    has_bad = any('illegal' in e.lower() or 'pps' in e.lower() or 'no frame' in e.lower()
                  for e in errors)
    if has_bad:
        return f"[ERR] {_t('error', lang)}"
    if errors or (dup is not None and dup > 100) or (delta is not None and abs(delta) > 2):
        return f"[WARN] {_t('warn', lang)}"
    return f"[OK] {_t('normal', lang)}"


def print_results(results: list[dict], input_path: str, scale: float,
                  batch_size: int, model: str, crf: int = 23, preset: str = "medium",
                  lang: str = 'zh'):
    """Print comparison table with precise verification columns."""
    lang = _normalize_lang(lang)
    H = [
        _t('version', lang), _t('time', lang), _t('input_frames', lang), _t('output_frames', lang),
        _t('decoded_frames', lang), _t('expected_frames', lang), _t('frame_delta', lang),
        _t('dup', lang), _t('drop', lang), _t('errors', lang), _t('container_frames', lang),
        _t('fps', lang), _t('file_size', lang), _t('rc_mode', lang), _t('bitrate', lang),
        _t('gpu_util', lang), _t('gpu_mem', lang), _t('encoder', lang)
    ]
    W = [9, 7, 7, 7, 7, 7, 6, 5, 5, 5, 7, 7, 9, 8, 8, 5, 5, 12]

    print()
    print("=" * 130)
    print(_t('title', lang))
    print("=" * 130)
    print(f"  {_t('input', lang)}: {input_path}  |  {_t('scale', lang)}: {scale}x  |  batch-size: {batch_size}  |  {_t('model', lang)}: {model}  |  crf: {crf}  |  preset: {preset}")
    print()

    header = "  " + "  ".join(_pad(h, w, '<' if h in (_t('version', lang), _t('encoder', lang)) else '>') for h, w in zip(H, W))
    print(header)
    _main_sep_len = sum(W) + 2 * (len(W) - 1)
    print("  " + "-" * _main_sep_len)

    best_fps = 0.0
    best_label = ""

    for r in results:
        label = r.get("label", "?")
        if not r.get("success"):
            err = r.get("error", "unknown")
            print(f"  {label:<9} {'-- FAILED --':>8}  ({err})")
            continue

        elapsed = r.get("elapsed_sec", 0)
        in_frames = r.get("input_frames", "?")
        out_frames = r.get("output_frames", "?")
        container_frames = r.get("output_frames_nf", "?")
        v = r.get("verify")
        decoded = v.get('output_frames_decoded') if v else None
        expected = v.get('expected_frames') if v else None
        delta = v.get('frame_delta') if v else None
        dup = v.get('dup') if v else None
        drop = v.get('drop') if v else None
        err_count = len(v.get('errors', [])) if v else 0
        # FPS 优先使用精准帧数
        fps_base = decoded if decoded is not None else (out_frames if out_frames != "?" else None)
        fps = round(fps_base / elapsed, 1) if elapsed and fps_base else r.get("fps", 0)
        # 把精准 FPS 写回 result 供 PDF/JSON 使用
        if fps:
            r['fps_precise'] = fps
        size_mb = r.get("output_size_mb", 0)
        bitrate_kbps = r.get("output_bitrate_kbps")
        rc_mode = r.get("output_rc_mode", "?")
        gpu_util = r.get("gpu_util_pct", "?")
        gpu_mem = r.get("gpu_mem_used_mb", "?")
        codec = r.get("codec", "?")

        elapsed_str = f"{elapsed:.1f}s" if elapsed else "?"
        in_str = str(in_frames) if in_frames else "?"
        out_str = str(out_frames) if out_frames else "?"
        decoded_str = str(decoded) if decoded is not None else "?"
        expected_str = str(expected) if expected is not None else "?"
        delta_str = f"{delta:+d}" if delta is not None else "?"
        dup_str = str(dup) if dup is not None else "?"
        drop_str = str(drop) if drop is not None else "?"
        err_str = str(err_count) if err_count is not None else "?"
        container_str = str(container_frames) if container_frames else "?"
        fps_str = f"{fps:.1f}" if fps else "?"
        size_str = f"{size_mb:.1f} MB" if size_mb else "?"
        rc_mode_str = str(rc_mode) if rc_mode != "?" else "?"
        bitrate_str = f"{bitrate_kbps:.0f}kbps" if bitrate_kbps else "?"
        gpu_util_str = f"{gpu_util}%" if gpu_util != "?" else "?"
        gpu_mem_str = f"{gpu_mem}M" if isinstance(gpu_mem, (int, float)) else "?"

        cols = [label, elapsed_str, in_str, out_str, decoded_str, expected_str, delta_str,
                dup_str, drop_str, err_str, container_str, fps_str, size_str,
                rc_mode_str, bitrate_str, gpu_util_str, gpu_mem_str, codec]
        aligns = ['<', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '<']
        row = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(cols, W, aligns))
        print(row)

        if isinstance(fps, (int, float)) and fps > best_fps:
            best_fps = fps
            best_label = label

    # 帧数交叉验证提示（packets / decoded / expected 三方）
    ok_results = [r for r in results if r.get("success")]
    for r in ok_results:
        label = r.get("label", "?")
        pkts = r.get("output_frames")
        nf = r.get("output_frames_nf")
        v = r.get("verify")
        decoded = v.get('output_frames_decoded') if v else None
        expected = v.get('expected_frames') if v else None
        alerts = []
        if pkts and decoded and pkts != decoded:
            alerts.append(f"packets({pkts})≠decoded({decoded})")
        if decoded is not None and expected is not None and decoded != expected:
            alerts.append(f"decoded({decoded})≠expected({expected}, Δ{decoded-expected:+d})")
        if pkts and nf and pkts != nf:
            alerts.append(f"packets({pkts})≠container({nf})")
        if alerts:
            print(f"  [WARN] {label}: " + " | ".join(alerts))

    print()
    if best_label:
        print(f"  {_t('fastest_version', lang)}: {best_label} ({best_fps:.1f} FPS)")

    # 版本间对比
    if len(ok_results) >= 2:
        fps_values = [r.get("fps_precise", r.get("fps", 0)) for r in ok_results]
        slowest_fps = min(fps_values)
        if slowest_fps > 0:
            spread_pct = (best_fps - slowest_fps) / slowest_fps * 100
            print(f"  {_t('performance_spread', lang)}: {spread_pct:.1f}% {_t('fastest_vs_slowest', lang)}")
        times = [r.get("elapsed_sec", 0) for r in ok_results]
        print(f"  {_t('time_range', lang)}: {min(times):.1f}s – {max(times):.1f}s")

    # 编码器一致性检查
    codecs = set(r.get("codec", "") for r in ok_results)
    if len(codecs) > 1:
        print(f"  [WARN] Encoder mismatch: {codecs}")

    # ── 帧完整性验证表 ──
    _verify_results = [r for r in ok_results if r.get("verify")]
    if _verify_results:
        print()
        print(f"  ── {_t('frame_integrity', lang)} ({_t('frame_integrity_hint', lang)}) ──")
        _vH = [
            _t('version', lang), _t('input_frames', lang), _t('expected_frames', lang),
            _t('decoded_frames', lang), _t('frame_delta', lang), _t('dup', lang), _t('drop', lang),
            _t('errors', lang), _t('decoder', lang), _t('status', lang)
        ]
        _vW = [9, 8, 8, 8, 7, 6, 6, 7, 10, 8]
        _vA = ['<', '>', '>', '>', '>', '>', '>', '>', '<', '<']
        _vh = "  " + "  ".join(_pad(h, w, a) for h, w, a in zip(_vH, _vW, _vA))
        print(_vh)
        _vsep_len = sum(_vW) + 2 * (len(_vW) - 1)
        print("  " + "-" * _vsep_len)
        for r in _verify_results:
            v = r['verify']
            label = r.get('label', '?')
            cols = [
                label,
                str(v.get('input_frames_decoded') or v.get('input_frames_packets') or '?'),
                str(v.get('expected_frames', '?')),
                str(v.get('output_frames_decoded', '?')),
                f"{v.get('frame_delta'):+d}" if v.get('frame_delta') is not None else '?',
                str(v.get('dup', '?')),
                str(v.get('drop', '?')),
                str(len(v.get('errors', []))),
                str(v.get('decoder', '?')),
                _anomaly_status_str(v),
            ]
            print("  " + "  ".join(_pad(c, w, a) for c, w, a in zip(cols, _vW, _vA)))

    # ── 诊断数据: pipeline 帧计数交叉验证 (reader → infer → writer) ──
    _diag_versions = [r for r in ok_results if r.get("_diag_reader_pairs")]
    if _diag_versions:
        print()
        print(f"  ── {_t('pipeline_diagnosis', lang)} ──")
        print(f"  {_t('pipeline_hint', lang)}\n")

        _pH = [
            _t('version', lang), _t('reader_pairs', lang), _t('infer_batches', lang),
            _t('infer_pairs', lang), _t('writer_frames', lang), _t('pipe_expected', lang),
            _t('pipe_delta', lang), _t('total_expected', lang), "ffprobe", _t('file_delta', lang)
        ]
        _pW = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        _pA = ['<', '>', '>', '>', '>', '>', '>', '>', '>', '<']
        _dh = "  " + "  ".join(_pad(h, w, a) for h, w, a in zip(_pH, _pW, _pA))
        print(_dh)
        _sep_len = sum(_pW) + 2 * (len(_pW) - 1)
        print("  " + "-" * _sep_len)
        for r in _diag_versions:
            _label = r.get("label", "?")
            _rp = r.get("_diag_reader_pairs", 0)
            _ib = r.get("_diag_infer_batches", 0)
            _ip = r.get("_diag_infer_pairs", 0)
            _wf = r.get("_diag_writer_frames", 0)
            _scale_int = int(r.get("scale", 2.0))
            _out_pkts = r.get("output_frames")

            _pipe_expected = _ip * _scale_int if _ip > 0 else 0
            _pipe_delta = _wf - _pipe_expected if _pipe_expected > 0 else 0

            # 外部首帧：f0 不在 pipeline._written 内，需单独计入总期望
            _ext_first = r.get('_diag_external_first_frame', 1)
            _ext_default = '_diag_external_first_frame' not in r
            _total_expected = _ext_first + _pipe_expected if _pipe_expected > 0 else 0

            # 文件级校验: ffprobe 实际帧数 vs 总期望
            if _out_pkts is not None:
                _file_delta = _out_pkts - _total_expected
                _verify = "[OK]" if _file_delta == 0 else f"[WARN] {_file_delta:+d}"
            else:
                _file_delta = None
                _verify = "?"

            _out_str = str(_out_pkts) if _out_pkts is not None else "?"
            _total_str = f"{_total_expected}" + ("[d]" if _ext_default else "")
            _dcols = [_label, str(_rp), str(_ib), str(_ip), str(_wf),
                      str(_pipe_expected), f"{_pipe_delta:+d}", _total_str, _out_str, _verify]
            _drow = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(_dcols, _pW, _pA))
            print(_drow)

        # ── NVENC 编码精度诊断 (精确数值, 不设阈值) ──
        _gpu_stay = [r for r in _diag_versions if r.get("_diag_gpu_stay_batches")]
        if _gpu_stay:
            print()
            print(f"  ── {_t('nvenc_diagnosis', lang)} ──")
            print(f"  {_t('pipeline_hint', lang)}\n")

            _nH = [
                _t('version', lang), _t('gpu_batches', lang), _t('encoded_frames', lang),
                _t('empty_h264', lang), _t('pipe_delta', lang), _t('lock_fallback', lang),
                _t('description', lang)
            ]
            _nW = [10, 10, 10, 10, 10, 10, 30]
            _nA = ['<', '>', '>', '>', '>', '>', '<']
            _dh2 = "  " + "  ".join(_pad(h, w, a) for h, w, a in zip(_nH, _nW, _nA))
            print(_dh2)
            _sep2_len = sum(_nW) + 2 * (len(_nW) - 1)
            print("  " + "-" * _sep2_len)

            # 检测所有版本管內偏移是否一致 (一致=可能 LA 正常行为, 非 bug)
            _all_pipe_deltas = []
            for _r in _gpu_stay:
                __ip = _r.get("_diag_infer_pairs", 0)
                __wf = _r.get("_diag_writer_frames", 0)
                __sc = int(_r.get("scale", 2.0))
                __pe = __ip * __sc
                __pd = __wf - __pe if __pe > 0 else 0
                if __pd < 0:
                    _all_pipe_deltas.append(__pd)
            _all_same_delta = len(_all_pipe_deltas) >= 2 and len(set(_all_pipe_deltas)) == 1

            for r in _gpu_stay:
                _label = r.get("label", "?")
                _gs = r.get("_diag_gpu_stay_batches", 0)
                _nv = r.get("_diag_nvenc_frames", 0)
                _eh = r.get("_diag_empty_h264", 0)
                _lf = r.get("_diag_lock_fallback", 0)
                _scale_int2 = int(r.get("scale", 2.0))
                _ip2 = r.get("_diag_infer_pairs", 0)
                _wf2 = r.get("_diag_writer_frames", 0)
                _pipe_expected2 = _ip2 * _scale_int2
                _pipe_delta2 = _wf2 - _pipe_expected2 if _pipe_expected2 > 0 else 0

                # 构造精确说明
                _notes = []
                if _eh > 0:
                    _notes.append(f"空H264={_eh}")
                if _pipe_delta2 < 0:
                    if _all_same_delta:
                        _notes.append(f"管內缺{abs(_pipe_delta2)}帧(LA可能)")
                    else:
                        _notes.append(f"丢{abs(_pipe_delta2)}帧")
                if _lf > 0:
                    _notes.append(f"降级{_lf}")
                if not _notes:
                    _notes.append("正常")
                _note_str = ", ".join(_notes)

                # 附加诊断: NVENC调用 与 管內期望 是否一致
                _nv_vs_expected = _nv - _pipe_expected2 if _pipe_expected2 > 0 else 0
                if _nv_vs_expected != 0:
                    _note_str += f" (调用Δ{_nv_vs_expected:+d})"

                # 文件级交叉校验: ffprobe vs 总期望（NVENC→Writer→File 全链路）
                _out_pkts2 = r.get("output_frames")
                _ext_first2 = r.get('_diag_external_first_frame', 1)
                _total_expected2 = _ext_first2 + _pipe_expected2 if _pipe_expected2 > 0 else 0
                if _out_pkts2 is not None and _total_expected2 > 0:
                    _file_delta2 = _out_pkts2 - _total_expected2
                    if _file_delta2 != 0:
                        _note_str += f" [文件Δ{_file_delta2:+d}]"

                _ncols = [_label, str(_gs), str(_nv), str(_eh),
                          f"{_pipe_delta2:+d}", str(_lf), _note_str]
                _nrow = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(_ncols, _nW, _nA))
                print(_nrow)

    # ── 版本间输出帧数一致性 ──
    _out_frames_list = [r.get("verify", {}).get("output_frames_decoded")
                        for r in ok_results
                        if r.get("verify", {}).get("output_frames_decoded")]
    if not _out_frames_list:
        _out_frames_list = [r.get("output_frames") for r in ok_results if r.get("output_frames")]
    if len(set(_out_frames_list)) > 1:
        print()
        print(f"  [WARN] {_t('output_frame_mismatch', lang)}")
        _ref = max(_out_frames_list)
        for r in ok_results:
            _of = r.get("verify", {}).get("output_frames_decoded") or r.get("output_frames")
            if _of is not None:
                _label = r.get("label", "?")
                _delta = _of - _ref
                _pct = f" ({_delta:+d}, {_delta/_ref*100:+.1f}%)" if _delta != 0 else " (baseline)"
                print(f"      {_label}: {_of} frames{_pct}")

    # 编码级别诊断：独立于 Pipeline diag，即使缺少 _diag_reader_pairs 也显示
    _level_versions = [r for r in ok_results if r.get("_diag_active_level")]
    if _level_versions:
        print()
        print(f"  ── {_t('encoder_level', lang)} ──")
        _dh_lv = "  {:<10} {:>5}  {}".format(_t('version', lang), _t('level', lang), _t('description', lang))
        print(_dh_lv)
        print("  " + "-" * 50)
        for r in _level_versions:
            _label = r.get("label", "?")
            _lv = r.get("_diag_active_level", "?")
            _desc = _t_level(_lv, lang)
            _rm = r.get("output_rc_mode", "")
            if _lv == 1 and _rm:
                _desc += f" (rc={_rm})"
            elif _lv != 1 and _rm:
                _desc += f" (fallback, expected rc={_rm})"
            print(f"  {_label:<10} {_lv:>5}  {_desc}")

    print("=" * 130)


# ═══════════════════════════════════════════════
# 3. PDF 报告生成（reportlab）
# ═══════════════════════════════════════════════

def _register_pdf_font() -> str:
    """跨平台注册 CJK 字体，返回可用的字体名。失败则回退 Helvetica。

    注意：reportlab 的 TTFont 仅支持 TrueType(glyf) 轮廓，不支持 CFF/PostScript
    轮廓。Debian/Ubuntu 的 fonts-noto-cjk 包即为 CFF 格式（OTTO），注册会报
    "postscript outlines are not supported"，装了也用不了。因此查找顺序：
      1) 系统中的 TrueType 轮廓 CJK 字体（文泉驿 / Windows 字体，可嵌入 PDF）；
      2) 回退 reportlab 内置 CID 字体 STSong-Light —— 无需任何字体文件，
         主流 PDF 阅读器均可正常显示中文（字形不嵌入，依赖阅读器替换）。
    """
    if not HAS_REPORTLAB:
        return "Helvetica"

    candidates = [
        # Windows（均为 TrueType 轮廓）
        ("C:/Windows/Fonts/msyh.ttc", 0),
        ("C:/Windows/Fonts/simsun.ttc", 0),
        ("C:/Windows/Fonts/simhei.ttf", None),
        ("C:/Windows/Fonts/msyhbd.ttc", 0),
        # Linux — TrueType 轮廓，reportlab 可直接嵌入
        ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
        ("/usr/share/fonts/truetype/arphic/uming.ttc", 0),
        ("/usr/share/fonts/truetype/arphic/ukai.ttc", 0),
        # Linux — Noto CJK：仅当为 TrueType 轮廓版本时可用
        # （Debian/Ubuntu 官方包是 CFF 轮廓，会注册失败并走 CID 回退）
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ]
    errors = []
    for path, idx in candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            if idx is not None:
                font = TTFont("CJKFont", str(p), subfontIndex=idx)
            else:
                font = TTFont("CJKFont", str(p))
            pdfmetrics.registerFont(font)
            return "CJKFont"
        except Exception as e:
            errors.append(f"{path}: {e}")
            continue

    if errors:
        print("[INFO] 以下 TrueType CJK 字体注册失败（reportlab 不支持 CFF/PostScript 轮廓）:")
        for err in errors:
            print(f"       - {err}")

    # 零依赖回退：reportlab 内置 CID 字体，无需字体文件即可显示中文
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        print("[INFO] 使用 reportlab 内置 CID 字体 STSong-Light（中文字形不嵌入，依赖阅读器字体替换）")
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def _pdf_font_install_hint() -> str:
    """返回 CJK 字体缺失时的修复安装提示。"""
    return (
        "reportlab can only embed TrueType-outline CJK fonts. Note that the\n"
        "Debian/Ubuntu 'fonts-noto-cjk' package ships CFF-outline fonts which\n"
        "reportlab does NOT support. To get an embeddable CJK font, install:\n"
        "  Ubuntu/Debian: sudo apt-get install fonts-wqy-zenhei\n"
        "  CentOS/RHEL:   sudo yum install wqy-zenhei-fonts\n"
        "  Windows:       ensure C:/Windows/Fonts/msyh.ttc or simsun.ttc exists\n"
        "  Fallback:      reportlab's built-in CID font STSong-Light is used\n"
        "                 automatically (no font file needed, viewer-dependent)."
    )


def _make_styles(font_name: str):
    """创建 PDF 段落样式。"""
    styles = getSampleStyleSheet()
    base = {
        'fontName': font_name,
        'leading': 14,
    }
    title = ParagraphStyle(
        'CJKTitle', parent=styles['Title'],
        fontName=font_name, fontSize=18, leading=24,
        alignment=1, spaceAfter=14,
    )
    heading = ParagraphStyle(
        'CJKHeading', parent=styles['Heading2'],
        fontName=font_name, fontSize=13, leading=18,
        spaceBefore=14, spaceAfter=8,
    )
    normal = ParagraphStyle(
        'CJKNormal', parent=styles['Normal'],
        fontName=font_name, fontSize=9, leading=13,
    )
    small = ParagraphStyle(
        'CJKSmall', parent=styles['Normal'],
        fontName=font_name, fontSize=8, leading=11,
    )
    return title, heading, normal, small


def _build_table(data: list, col_widths: list, font_name: str,
                 first_row_header: bool = True, first_col_header: bool = False) -> Table:
    """辅助：生成统一样式的 reportlab Table。"""
    t = Table(data, colWidths=col_widths, repeatRows=1 if first_row_header else 0)
    style_commands = [
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]
    if first_row_header:
        style_commands.extend([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2F5496')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
        ])
    if first_col_header:
        style_commands.extend([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#D6E4F0')),
            ('FONTNAME', (0, 0), (0, -1), font_name),
        ])
    t.setStyle(TableStyle(style_commands))
    return t


def _safe_val(val, fmt=None):
    """安全格式化数值/缺失值。"""
    if val is None:
        return "—"
    if fmt:
        try:
            return fmt.format(val)
        except Exception:
            return str(val)
    return str(val)


def _bar_chart(labels: list, values: list, title: str, font_name: str,
               width: float = 16 * cm, height: float = 7 * cm) -> Drawing:
    """使用 reportlab.graphics 绘制简单矢量条形图。"""
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 40
    chart.height = height - 70
    chart.width = width - 80
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = font_name
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontName = font_name
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor('#2F5496')
    chart.barWidth = 12
    d.add(chart)
    # 标题
    lab = Label()
    lab.setText(title)
    lab.fontName = font_name
    lab.fontSize = 10
    lab.x = width / 2
    lab.y = height - 15
    lab.textAnchor = 'middle'
    d.add(lab)
    return d


def generate_pdf_report(results: list[dict], input_path: str, output_path: str,
                        scale: float, batch_size: int, model: str,
                        crf: int, preset: str, total_elapsed: float, lang: str = 'zh'):
    """生成 PDF 基准测试报告。"""
    lang = _normalize_lang(lang)
    if not HAS_REPORTLAB:
        print("[SKIP] reportlab 未安装，跳过 PDF 报告生成")
        return

    font_name = _register_pdf_font()
    if font_name == "Helvetica":
        print("[WARN] CJK font not found. PDF report will use English text with Helvetica.")
        print(_pdf_font_install_hint())
        # 无 CJK 字体时无法渲染中文，自动回退英文
        lang = 'en'

    title_style, heading_style, normal_style, small_style = _make_styles(font_name)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=pagesizes.A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    story = []

    # ── Cover / Summary ──
    story.append(Paragraph(_t('pdf_title', lang), title_style))
    story.append(Spacer(1, 0.4 * cm))
    info_lines = [
        f"{_t('input_video', lang)}: {input_path}",
        f"{_t('interp_scale', lang)}: {scale}x",
        f"Model: {model}",
        f"Batch Size: {batch_size}",
        f"CRF/QP: {crf}",
        f"Preset: {preset}",
        f"{_t('generated_at', lang)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{_t('total_elapsed', lang)}: {total_elapsed:.1f}s",
    ]
    for line in info_lines:
        story.append(Paragraph(line, normal_style))
    story.append(Spacer(1, 0.6 * cm))

    ok_results = [r for r in results if r.get("success")]

    # ── Main Comparison Table ──
    story.append(Paragraph(_t('main_comparison', lang), heading_style))
    main_headers = [
        _t('version', lang), _t('time_s', lang), _t('input_frames', lang), _t('output_frames', lang),
        _t('decoded_frames', lang), _t('expected_frames', lang), _t('frame_delta', lang), _t('dup', lang),
        _t('drop', lang), _t('fps', lang), _t('size_mb', lang), _t('rc_mode', lang), _t('bitrate_kbps', lang),
        _t('gpu_util', lang), _t('vram_mb', lang), _t('encoder', lang)
    ]
    main_data = [main_headers]
    for r in ok_results:
        v = r.get('verify')
        decoded = v.get('output_frames_decoded') if v else None
        expected = v.get('expected_frames') if v else None
        delta = v.get('frame_delta') if v else None
        fps_base = decoded if decoded is not None else r.get('output_frames')
        elapsed = r.get('elapsed_sec', 0)
        fps = round(fps_base / elapsed, 1) if elapsed and fps_base else r.get('fps', 0)
        main_data.append([
            r.get('label', '?'),
            _safe_val(r.get('elapsed_sec'), "{:.1f}"),
            _safe_val(r.get('input_frames')),
            _safe_val(r.get('output_frames')),
            _safe_val(decoded),
            _safe_val(expected),
            _safe_val(delta, "{:+d}"),
            _safe_val(v.get('dup') if v else None),
            _safe_val(v.get('drop') if v else None),
            _safe_val(fps, "{:.1f}"),
            _safe_val(r.get('output_size_mb'), "{:.1f}"),
            _safe_val(r.get('output_rc_mode')),
            _safe_val(r.get('output_bitrate_kbps'), "{:.0f}"),
            _safe_val(r.get('gpu_util_pct')),
            _safe_val(r.get('gpu_mem_used_mb')),
            _safe_val(r.get('codec')),
        ])
    cw = [1.0 * cm] * len(main_headers)
    cw[0] = 1.3 * cm
    cw[-1] = 1.8 * cm
    story.append(_build_table(main_data, cw, font_name))
    story.append(Spacer(1, 0.5 * cm))

    # ── Frame Integrity Verification Table ──
    story.append(Paragraph(_t('sec_frame_integrity', lang), heading_style))
    story.append(Paragraph(_t('frame_integrity_hint', lang), small_style))
    verify_headers = [
        _t('version', lang), _t('input_frames', lang), _t('expected_frames', lang), _t('decoded_frames', lang),
        _t('frame_delta', lang), _t('dup', lang), _t('drop', lang), _t('errors', lang), _t('decoder', lang),
        _t('status', lang)
    ]
    verify_data = [verify_headers]
    for r in ok_results:
        v = r.get('verify')
        if not v:
            continue
        status = _anomaly_status_str(v, lang)
        # Use plain ASCII markers instead of emoji to avoid missing glyphs
        status_text = status.replace("[OK]", "[OK]").replace("[WARN]", "[WARN]").replace("[ERR]", "[ERR]")
        verify_data.append([
            r.get('label', '?'),
            _safe_val(v.get('input_frames_decoded') or v.get('input_frames_packets')),
            _safe_val(v.get('expected_frames')),
            _safe_val(v.get('output_frames_decoded')),
            _safe_val(v.get('frame_delta'), "{:+d}"),
            _safe_val(v.get('dup')),
            _safe_val(v.get('drop')),
            str(len(v.get('errors', []))),
            _safe_val(v.get('decoder')),
            status_text,
        ])
    if len(verify_data) > 1:
        story.append(_build_table(verify_data, [1.2 * cm] * len(verify_headers), font_name))
    else:
        story.append(Paragraph(_t('no_verify_data', lang), normal_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── Diagnostic Tables ──
    _diag_versions = [r for r in ok_results if r.get("_diag_reader_pairs")]
    if _diag_versions:
        story.append(Paragraph(_t('sec_pipeline_diagnosis', lang), heading_style))
        story.append(Paragraph(_t('pipeline_hint', lang), small_style))
        p_headers = [
            _t('version', lang), _t('reader_pairs', lang), _t('infer_batches', lang), _t('infer_pairs', lang),
            _t('writer_frames', lang), _t('pipe_expected', lang), _t('pipe_delta', lang),
            _t('total_expected', lang), "ffprobe", _t('file_delta', lang)
        ]
        p_data = [p_headers]
        for r in _diag_versions:
            _rp = r.get("_diag_reader_pairs", 0)
            _ib = r.get("_diag_infer_batches", 0)
            _ip = r.get("_diag_infer_pairs", 0)
            _wf = r.get("_diag_writer_frames", 0)
            _scale_int = int(r.get("scale", 2.0))
            _pipe_expected = _ip * _scale_int if _ip > 0 else 0
            _pipe_delta = _wf - _pipe_expected if _pipe_expected > 0 else 0
            _ext_first = r.get('_diag_external_first_frame', 1)
            _total_expected = _ext_first + _pipe_expected if _pipe_expected > 0 else 0
            _out_pkts = r.get("output_frames")
            _file_delta = (_out_pkts - _total_expected) if _out_pkts is not None else None
            p_data.append([
                r.get('label', '?'),
                str(_rp), str(_ib), str(_ip), str(_wf),
                str(_pipe_expected), f"{_pipe_delta:+d}", str(_total_expected),
                _safe_val(_out_pkts), _safe_val(_file_delta, "{:+d}"),
            ])
        story.append(_build_table(p_data, [1.1 * cm] * len(p_headers), font_name))
        story.append(Spacer(1, 0.5 * cm))

        _gpu_stay = [r for r in _diag_versions if r.get("_diag_gpu_stay_batches")]
        if _gpu_stay:
            story.append(Paragraph(_t('sec_nvenc_diagnosis', lang), heading_style))
            n_headers = [
                _t('version', lang), _t('gpu_batches', lang), _t('encoded_frames', lang), _t('empty_h264', lang),
                _t('pipe_delta', lang), _t('lock_fallback', lang)
            ]
            n_data = [n_headers]
            for r in _gpu_stay:
                _ip = r.get("_diag_infer_pairs", 0)
                _wf = r.get("_diag_writer_frames", 0)
                _scale_int = int(r.get("scale", 2.0))
                _pipe_expected = _ip * _scale_int
                _pipe_delta = _wf - _pipe_expected if _pipe_expected > 0 else 0
                n_data.append([
                    r.get('label', '?'),
                    str(r.get('_diag_gpu_stay_batches', 0)),
                    str(r.get('_diag_nvenc_frames', 0)),
                    str(r.get('_diag_empty_h264', 0)),
                    f"{_pipe_delta:+d}",
                    str(r.get('_diag_lock_fallback', 0)),
                ])
            story.append(_build_table(n_data, [1.3 * cm] * len(n_headers), font_name))
            story.append(Spacer(1, 0.5 * cm))

    _level_versions = [r for r in ok_results if r.get("_diag_active_level")]
    if _level_versions:
        story.append(Paragraph(_t('sec_encoder_level', lang), heading_style))
        l_headers = [_t('version', lang), _t('level', lang), _t('description', lang)]
        l_data = [l_headers]
        for r in _level_versions:
            _lv = r.get("_diag_active_level", "?")
            _desc = _t_level(_lv, lang)
            _rm = r.get("output_rc_mode", "")
            if _rm:
                _desc += f" (rc={_rm})"
            l_data.append([r.get('label', '?'), str(_lv), _desc])
        story.append(_build_table(l_data, [2 * cm, 1.5 * cm, 10 * cm], font_name))
        story.append(Spacer(1, 0.5 * cm))

    # ── Charts ──
    if len(ok_results) >= 1:
        story.append(PageBreak())
        story.append(Paragraph(_t('sec_performance_charts', lang), heading_style))

        labels = [r.get('label', '?') for r in ok_results]

        # FPS
        fps_vals = [round(r.get('fps_precise', r.get('fps', 0)) or 0, 1) for r in ok_results]
        story.append(_bar_chart(labels, fps_vals, _t('fps_chart', lang), font_name))
        story.append(Spacer(1, 0.6 * cm))

        # Time
        time_vals = [round(r.get('elapsed_sec', 0) or 0, 1) for r in ok_results]
        story.append(_bar_chart(labels, time_vals, _t('time_chart', lang), font_name))
        story.append(Spacer(1, 0.6 * cm))

        # File size
        size_vals = [round(r.get('output_size_mb', 0) or 0, 1) for r in ok_results]
        story.append(_bar_chart(labels, size_vals, _t('size_chart', lang), font_name))
        story.append(Spacer(1, 0.6 * cm))

        # GPU%
        gpu_vals = [r.get('gpu_util_pct', 0) or 0 for r in ok_results]
        story.append(_bar_chart(labels, gpu_vals, _t('gpu_chart', lang), font_name))
        story.append(Spacer(1, 0.6 * cm))

    # ── Conclusion ──
    story.append(Paragraph(_t('sec_conclusion', lang), heading_style))
    fps_values = [(r.get('fps_precise', r.get('fps', 0)) or 0, r.get('label', '?')) for r in ok_results]
    if fps_values:
        best_fps, best_label = max(fps_values)
        slowest_fps, _ = min(fps_values)
        spread = ((best_fps - slowest_fps) / slowest_fps * 100) if slowest_fps > 0 else 0
        conclusion = [
            f"{_t('fastest_version', lang)}: {best_label} ({best_fps:.1f} FPS)",
            f"{_t('performance_spread', lang)}: {spread:.1f}% {_t('fastest_vs_slowest', lang)}",
        ]
    else:
        conclusion = [_t('no_success', lang)]

    # Frame integrity summary
    verified = [r for r in ok_results if r.get('verify')]
    bad = [r for r in verified if '[ERR]' in _anomaly_status_str(r['verify'])]
    warn = [r for r in verified if '[WARN]' in _anomaly_status_str(r['verify'])]
    good = [r for r in verified if '[OK]' in _anomaly_status_str(r['verify'])]
    conclusion.append(
        f"{_t('frame_integrity_summary', lang)}: {_t('normal', lang)} {len(good)} / "
        f"{_t('warn', lang)} {len(warn)} / {_t('error', lang)} {len(bad)}"
    )

    for line in conclusion:
        story.append(Paragraph(line, normal_style))

    doc.build(story)
    print(f"[OK] PDF report saved: {output_path}")


# ═══════════════════════════════════════════════
# 4. 主函数
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IFRNet 版本对比基准测试 v2（精准帧数检测 + PDF 报告）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用版本: {', '.join(VERSION_LABELS.values())}

示例:
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/ --scale 4
    python benchmark_ifrnet_versions_v2.py -i test.mp4 -o benchmark_output/ --versions v6.4.5.1 --no-warmup
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="输入视频路径")
    parser.add_argument("-o", "--output-dir", default="benchmark_output",
                        help="中间输出目录（默认: benchmark_output/）")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="插帧倍数（默认: 2）")
    parser.add_argument("--model", default="IFRNet_S_Vimeo90K",
                        choices=["IFRNet_Vimeo90K", "IFRNet_S_Vimeo90K", "IFRNet_L_Vimeo90K"],
                        help="模型名称（默认: IFRNet_S_Vimeo90K）")
    parser.add_argument("--versions", nargs="+",
                        default=["v6.3.5", "v6.4.1", "v6.4.2", "v6.4.3", "v6.4.3.1",
                                 "v6.4.3.2", "v6.4.4", "v6.4.4.1", "v6.4.4.2",
                                 "v6.4.5", "v6.4.5.1", "v6.4.5.2"],
                        help="要测试的版本列表（默认: 全部）")
    parser.add_argument("--cleanup", action="store_true",
                        help="测试完成后删除中间输出文件")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="保留所有版本的输出视频")
    parser.add_argument("--json-report", metavar="PATH",
                        help="保存 JSON 格式详细结果到指定路径")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过 GPU 预热（默认会先用短视频预热）")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="TRT 推理 batch_size（默认 24）")
    parser.add_argument("--warmup-duration", type=int, default=10,
                        help="预热视频时长（秒，默认 10）")
    parser.add_argument("--codec", type=str, default=None,
                        choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                        help="编码器（默认: None=自动选择最佳编码器）。与 --codecs 互斥")
    parser.add_argument("--codecs", nargs="+",
                        help="多编码器轮次测试，如 --codecs libx264 h264_nvenc。与 --codec 互斥")
    parser.add_argument("--crf", type=int, default=23,
                        help="编码质量 / QP 值（默认: 23）。libx264/x265: CRF; NVENC CONSTQP: QP; NVENC VBR_HQ: QP→targetQuality 映射")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="编码速度预设（默认: medium）。x264: x264 预设体系; NVENC: p1-p7 体系（自动映射）")

    # ── v2 新增参数 ──
    parser.add_argument("--no-verify", action="store_true",
                        help="关闭精准解码检测（默认开启）")
    parser.add_argument("--verify-hwaccel", choices=['auto', 'cuda', 'off'], default='auto',
                        help="精准检测硬件加速: auto=自动检测 (默认) | cuda=强制 CUDA | off=纯 CPU")
    parser.add_argument("--verify-workers", type=str, default='auto',
                        help="检测任务并发数: 'auto'=按 CPU/内存自动计算 (默认) | 正整数=固定值")
    parser.add_argument("--verify-task-ram-mb", type=int, default=300,
                        help="自动 workers 时单任务内存估计 MB (默认 300; image2pipe 任务自动按 600 估算)")
    parser.add_argument("--verify-reserve-ratio", type=float, default=0.10,
                        help="自动 workers 时系统预留内存比例 0-1 (默认 0.10)")
    parser.add_argument("--cpu-recheck", action="store_true",
                        help="开启 CPU 软件解码异常复检（默认关闭）")
    parser.add_argument("--pdf-report", metavar="PATH",
                        help="PDF 报告输出路径（默认: <output-dir>/benchmark_report_<输入stem>.pdf）")
    parser.add_argument("--no-pdf", action="store_true",
                        help="关闭 PDF 报告生成")
    parser.add_argument("--language", choices=['chinese', 'english', 'cn', 'en'], default='chinese',
                        help="输出语言: chinese/cn (默认) | english/en")

    args = parser.parse_args()

    # 环境检查
    if not shutil.which('ffmpeg'):
        print("[WARN] PATH 中未找到 ffmpeg，检测任务将全部标记 [SKIP]")
    if not shutil.which('ffprobe'):
        print("[WARN] PATH 中未找到 ffprobe，检测任务将全部标记 [SKIP]")

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        return 1

    # --codec 与 --codecs 互斥
    if args.codecs and args.codec is not None:
        print("错误: --codec 与 --codecs 互斥，请只指定其中一个")
        return 1

    codec_list = args.codecs if args.codecs else [args.codec]

    try:
        model_path = find_model_path(args.model)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    print(f"模型: {model_path}")
    print(f"输入: {args.input}")
    print(f"插帧: {args.scale}x")
    print(f"batch-size: {args.batch_size}")
    print(f"crf: {args.crf}  |  preset: {args.preset}")
    print(f"版本: {', '.join(args.versions)}")
    if args.no_verify:
        print("[WARN] 已关闭精准解码检测（--no-verify）")
    else:
        workers_desc = args.verify_workers
        if workers_desc == 'auto':
            workers_desc = f"auto (CPUx{detect_cpu_count()}, RAM检测后自动)"
        print(f"精准检测: hwaccel={args.verify_hwaccel}, workers={workers_desc}, cpu-recheck={args.cpu_recheck}")
    if args.no_pdf:
        print("[WARN] 已关闭 PDF 报告生成（--no-pdf）")
    print()

    version_map = {v: k for k, v in VERSION_LABELS.items()}
    selected = []
    for v in args.versions:
        if v in version_map:
            selected.append(version_map[v])
        else:
            print(f"警告: 未知版本 '{v}'，已跳过")

    if not selected:
        print("错误: 没有有效的版本")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem

    # ── GPU 预热 ──────────────────────────────────────────────────────────────
    if not args.no_warmup:
        warmup_video = output_dir / f"_warmup_{input_stem}.mp4"
        warmup_output = output_dir / f"_warmup_{input_stem}_output.mp4"
        warmup_version = selected[0]
        warmup_label = VERSION_LABELS[warmup_version]

        print("=" * 60)
        print("GPU 预热阶段")
        print("=" * 60)
        try:
            _p = subprocess.run([
                "ffmpeg", "-y",
                "-i", os.path.abspath(args.input),
                "-t", str(args.warmup_duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(warmup_video),
            ], capture_output=True, text=True, timeout=60)
            if _p.returncode == 0 and warmup_video.exists():
                print(f"  预热视频: {warmup_video} ({args.warmup_duration}s)")
            else:
                # ffmpeg -c copy 失败时回退到重新编码
                _p2 = subprocess.run([
                    "ffmpeg", "-y",
                    "-i", os.path.abspath(args.input),
                    "-t", str(args.warmup_duration),
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-an",
                    str(warmup_video),
                ], capture_output=True, text=True, timeout=120)
                if _p2.returncode != 0 or not warmup_video.exists():
                    print(f"  预热视频生成失败，跳过预热")
                    warmup_video = None

            if warmup_video and warmup_video.exists():
                print(f"  预热版本: {warmup_label}")
                print(f"  预热中...", end=" ", flush=True)
                t_warmup = time.perf_counter()
                _result = run_version(
                    version_module=warmup_version,
                    input_path=str(warmup_video),
                    output_path=str(warmup_output),
                    model_path=model_path,
                    scale=args.scale,
                    batch_size=args.batch_size,
                    trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
                    codec=codec_list[0],
                    crf=args.crf,
                    preset=args.preset,
                )
                t_warmup_elapsed = time.perf_counter() - t_warmup
                if _result.get("success"):
                    print(f"完成 ({t_warmup_elapsed:.1f}s)")
                    print(f"  TRT Engine 已构建/加载，GPU 已进入稳定状态")
                else:
                    print(f"失败，继续基准测试")
        except Exception as e:
            print(f"  预热异常: {e}，跳过")
        finally:
            # 清理预热文件
            for _f in (warmup_video, warmup_output):
                if _f and _f.exists():
                    try:
                        _f.unlink()
                    except OSError:
                        pass

        print("=" * 60)
        print()
        # 预热后短暂冷却，让 GPU 回到基准温度/频率
        time.sleep(3)
    else:
        print("[WARN] 已跳过 GPU 预热（--no-warmup）")
        print()

    # ── 正式基准测试 ──────────────────────────────────────────────────────────
    results = []
    t_test_start = time.perf_counter()

    for idx, version_module in enumerate(selected):
        label = VERSION_LABELS[version_module]
        output_path = output_dir / f"{input_stem}_{label}_x{args.scale:.0f}.mp4"

        print(f"[{idx+1}/{len(selected)}] 测试 {label} ...\n", end=" ", flush=True)

        try:
            result = run_version(
                version_module=version_module,
                input_path=os.path.abspath(args.input),
                output_path=str(output_path),
                model_path=model_path,
                scale=args.scale,
                batch_size=args.batch_size,
                trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
                codec=codec_list[0],
                crf=args.crf,
                preset=args.preset,
            )
        except subprocess.TimeoutExpired:
            result = {
                "version": version_module,
                "label": label,
                "success": False,
                "error": "timeout",
            }
        except Exception as e:
            result = {
                "version": version_module,
                "label": label,
                "success": False,
                "error": str(e),
            }

        results.append(result)

        if result.get("success"):
            fps = result.get("fps", "?")
            elapsed = result.get("elapsed_sec", "?")
            print(f"OK  ({elapsed}s, {fps} FPS)")
        else:
            err = result.get("error", "unknown")
            print(f"FAILED  ({err})")

        # 版本间有 GPU 内存残留风险，短暂暂停
        if idx < len(selected) - 1:
            time.sleep(2)

    test_elapsed = time.perf_counter() - t_test_start

    # ── 精准检测阶段（默认开启，计时隔离） ──
    if not args.no_verify:
        print("\n" + "=" * 60)
        print("精准解码检测阶段")
        print("=" * 60)

        # workers 解析：auto 或正整数
        if args.verify_workers.lower() == 'auto':
            verify_workers = None
        else:
            try:
                verify_workers = max(1, int(args.verify_workers))
            except ValueError:
                print(f"[WARN] --verify-workers 值 '{args.verify_workers}' 无效，回退 auto")
                verify_workers = None

        # GPU 能力预检，用于设置动态 GPU 并发上限
        gpu_caps_local = {'cuda': False, 'hint': ''}
        if args.verify_hwaccel != 'off':
            gpu_caps_local = detect_gpu_hwaccel()
        use_gpu_local = (args.verify_hwaccel == 'cuda' and gpu_caps_local['cuda']) or (
            args.verify_hwaccel == 'auto' and gpu_caps_local['cuda'])

        global _GPU_SEMAPHORE
        if use_gpu_local:
            gpu_workers_max = 8
            try:
                hint_l = gpu_caps_local.get('hint', '').lower()
                if any(x in hint_l for x in ('geforce', 'rtx', 'gtx')):
                    gpu_workers_max = 2
                elif any(x in hint_l for x in ('tesla', 'a100', 'a6000', 'l4', 'l40')):
                    gpu_workers_max = 8
            except Exception:
                pass
            # 若用户显式指定整数，GPU 闸门取用户值与上限的较小者；auto 时由 _run_verify 内部再算
            if verify_workers is not None:
                gpu_workers = max(1, min(verify_workers, gpu_workers_max))
            else:
                # 临时用自动 workers 估计值设置上限
                auto_est = compute_auto_workers(args.verify_task_ram_mb * 2,
                                                args.verify_reserve_ratio)
                gpu_workers = max(1, min(auto_est, gpu_workers_max))
            _GPU_SEMAPHORE = threading.BoundedSemaphore(gpu_workers)

        _run_verify(
            results,
            input_path=os.path.abspath(args.input),
            scale=args.scale,
            hwaccel=args.verify_hwaccel,
            workers=verify_workers,
            cpu_recheck=args.cpu_recheck,
            task_ram_mb=args.verify_task_ram_mb,
            reserve_ratio=args.verify_reserve_ratio,
        )
        _GPU_SEMAPHORE = None

    # 汇总
    print_results(results, os.path.abspath(args.input), args.scale,
                  args.batch_size, args.model, args.crf, args.preset, lang=args.language)

    total_elapsed = time.perf_counter() - t_test_start

    # JSON 详细报告
    if args.json_report:
        report = {
            "input": os.path.abspath(args.input),
            "scale": args.scale,
            "model": args.model,
            "model_path": model_path,
            "batch_size": args.batch_size,
            "crf": args.crf,
            "preset": args.preset,
            "results": results,
        }
        with open(args.json_report, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON 报告已保存: {args.json_report}")

    # PDF 报告
    if not args.no_pdf:
        pdf_path = Path(args.pdf_report) if args.pdf_report else (
            output_dir / f"benchmark_report_{input_stem}.pdf"
        )
        print()
        generate_pdf_report(
            results,
            input_path=os.path.abspath(args.input),
            output_path=pdf_path,
            scale=args.scale,
            batch_size=args.batch_size,
            model=args.model,
            crf=args.crf,
            preset=args.preset,
            total_elapsed=total_elapsed,
            lang=args.language,
        )

    # 清理
    if args.cleanup and not args.keep_outputs:
        for r in results:
            out = r.get("output_path", "")
            if out and os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass

    # 全部成功？
    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} 个版本测试失败")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
