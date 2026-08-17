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

v4 新增 (decode-merge):
  - check_decode_integrity(): 合并 check_frames_packets + check_pts_anomaly
    为单次 ffmpeg -v verbose -f null 解码，节省一次全量解码（CPU 软解场景 ~50% 时间，
    GPU 路径同样受益）。
  - frames 来源: ffmpeg stderr 末行 frame=N（解码器输出帧数）
  - packets 来源: ffprobe -count_packets（O(1) 读包头，始终独立提取）
  - pts_issues 来源: 同一次 verbose stderr 提取 anomaly/error 行
  - 检测灵敏度微量变化: frames 从 ffprobe nb_read_frames（demuxer→decoder）
    变为 ffmpeg decoder output frame=N。正常码流下二者相等；损坏码流下
    decoder output ≤ demuxer frames，仍能被 frames ≠ packets 检测到。

v4.1 新增 (hwaccel 失败误报修复):
  - 移植 analyze_video_pipeline_v3.py 的 NVDEC hwaccel 初始化失败过滤
    （_HWACCEL_INIT_FAILURE_KW / _is_hwaccel_init_failure）: cuvidCreateDecoder
    失败 / "Failed setup for format cuda: hwaccel initialisation returned error"
    是环境告警（driver/NVDEC SDK 版本不匹配等），ffmpeg 已内部回退软解且帧数
    正确，不再计入 pts_issues 误报 FAIL。
  - GPU→CPU 回退判断升级: 原仅凭 'frame=' 有无判断，但 hwaccel 初始化失败时
    ffmpeg 内部回退软解后 frame=N 仍会输出导致判断失效；现显式检测 NVDEC 失败
    特征行，命中即丢弃 GPU 轮 stderr 重跑纯 CPU 软解并打印 [WARN]。

v5 新增 (多编码探测):
  - 新增 probe_video_codec(): ffprobe 读 v:0 的 codec_name，识别 h264 / mpeg4
    家族 (mpeg4/msmpeg4v2/msmpeg4v3) / hevc / 其他编码，失败返回 None。
  - extract_annexb_es() 按编码分支提取 Annex B ES: h264 → h264_mp4toannexb；
    mpeg4 家族 → -f m4v (AVI 中 mpeg4 已是 Annex B，无需 bsf)；
    hevc → hevc_mp4toannexb (bsf 存在时，HEVC 做基础 NAL/IDR 分析)；
    其他编码 → 返回 None 显示 N/A。
  - 新增 MPEG-4 Part 2 深度 VOP 分析 (对应 H.264 三项检查):
      parse_mpeg4_es()   VOL resolution → vop_time_increment_bits + VOP header
                         (vop_coding_type + vop_time_increment + mtb)
      check_vop_stats()  I-VOP 计数 / 最长连续 I-VOP 块 / vop_time_increment 时间回退
                         (mpeg4 正常 GOP 周期性出 I 帧，段首窗口判据不适用；
                          最长连续 I 块 > 3 才是修复工具重编码特征)
  - 修复 mpeg4 视频「检查2 NAL 解析」直接崩溃 (h264_mp4toannexb 不支持 mpeg4)，
    并输出 codec 标识 + VOP 统计 + VOP 总数交叉核对 (用户判断容器假象还是真丢帧)。
  - 容错: 非支持编码 (vp9/av1 等) 跳过专项分析显示 N/A，不判失败；脚本对任意
    输入保持可用。

v6 新增 (色度花屏检测 + 连IDR误报阈值修正):
  - 修正检查 2「段首连 IDR」阈值: > 0 改为 >= 3。SPS/PPS 冗余重注入
    (FIX-SPS-PPS-V2) 会产生良性段首双 IDR（首个 IDR → 冗余 SPS/PPS/AUD →
    恢复点 IDR），原阈值将其误判为修复前 per-slot IDR 异常（后者为 6-16+ 连 IDR）。
  - 新增检查 4 check_chroma_corruption(): 解码 yuv420p 逐帧计算 U/V 平面 std，
    自校准相对阈值 (median*1.6, 下限 12) 识别周期性色度污染坏帧，坏帧数 >= 3
    判 FAIL。捕获「布纹花屏/帧闪烁」等纯像素域缺陷（不产生解码错误/PTS 异常/
    frame_num 回退/frames≠packets，原三项检查全部漏检的 v6.4.3 类缺陷）。
    坏帧索引按间距分簇计数（簇间距 >= 8 帧算新簇，仅簇内首帧保留），避免
    单簇多帧刷屏。

v6.1 新增 (chroma 加速 + 并发修正):
  - check_chroma_corruption(): 逐帧 std 改为分块批量归约 (_CHROMA_CHUNK_FRAMES=128)，
    消除逐帧 Python/NumPy 调用开销，数值结果不变（同一两遍 std，仅批量化）。
  - 检查 4 移至 GPU 并发闸门外执行 (FIX-CHROMA-GPU-GATE): chroma 为纯 CPU
    软解 + NumPy，不占用 NVDEC 会话，原实现持有 GPU 信号量导致批处理下
    CPU 任务被 GPU 队列串行化。
  - 新增 --chroma-hwaccel: 可选 NVDEC 解码加速 chroma 检查（需配合
    --hwaccel cuda/auto 才生效），默认关闭——NVDEC 错误隐藏行为与软解
    不一致，可能掩盖花屏特征，故不默认启用。

用法:
  单文件: python tests/verify_segment_bitstream_v3.py <video.mp4> [--dump-nal temp/nal.txt]
  多文件: python tests/verify_segment_bitstream_v3.py a.mp4 b.mp4 c.mp4 --workers 4
  文件夹: python tests/verify_segment_bitstream_v3.py ./segments/ --hwaccel cuda --gpu-workers 2
  文件夹: python tests/verify_segment_bitstream_v3.py ./segments/ --hwaccel cuda --chroma-hwaccel
  混合:   python tests/verify_segment_bitstream_v3.py a.mp4 ./dir1/ b.mp4
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

import numpy as np


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

# [FIX-NVDEC-THREAD-CAP] NVDEC 解码线程钳位：
# ffmpeg 6.1.1 的 NVDEC 解码 surface 计算公式：
#   ulNumDecodeSurfaces = ref_frame_count + num_reorder_frames
#                         + 2(deinterlace) + thread_count + 3(基础工作 surface)
# 32 是驱动硬上限（cudaVideoDecoder 拒绝），ffmpeg 6.1.1 无 FFMIN(pool,32) 钳位。
# 实测：-threads 8 → 32 surfaces 成功；-threads 9 → 33 surfaces 被驱动拒绝。
_MAX_DECODE_THREADS = 8
_DECODE_THREAD_HINT_SHOWN = False

def _clamp_decode_threads(requested=None) -> int:
    """钳位 NVDEC 解码线程数（驱动 32-surface 硬上限），超限时打印一次提示。"""
    global _DECODE_THREAD_HINT_SHOWN
    n = int(requested) if requested else (os.cpu_count() or 4)
    if n > _MAX_DECODE_THREADS:
        if not _DECODE_THREAD_HINT_SHOWN:
            print(f"[decode] threads={n} exceeds max {_MAX_DECODE_THREADS}, "
                  f"clamped (NVDEC 32-surface limit)", flush=True)
            _DECODE_THREAD_HINT_SHOWN = True
        n = _MAX_DECODE_THREADS
    return n

# NVDEC hwaccel 初始化失败特征（ffmpeg 已内部回退软件解码，帧数仍有效，非真实解码错误）
# v4.1: 与 analyze_video_pipeline_v3.py 保持一致，用于 issue 过滤与 GPU→CPU 回退判断。
# 注意: 关键词为小写（调用处传 lower 行）；覆盖 cuvidCreateDecoder 失败 /
# "Failed setup for format cuda: hwaccel initialisation returned error" 两行误报文本。
_HWACCEL_INIT_FAILURE_KW = [
    'failed setup for format cuda',
    'hwaccel initialisation returned error',
    'cuvidcreatedecoder',
    'decode surfaces',  # "Using more than 32 (N) decode surfaces" 超 surface 上限告警
]


def _is_hwaccel_init_failure(lower_line):
    """判定某行是否为 NVDEC hwaccel 初始化失败告警（而非真实解码错误）。

    这类失败是运行时环境问题（driver/NVDEC SDK 版本不匹配、容器 GPU 未透传等），
    ffmpeg 会静默回退到软件解码，frame 计数仍然正确，不应当作码流异常。
    """
    return any(kw in lower_line for kw in _HWACCEL_INIT_FAILURE_KW)


# ═══════════════════════════════════════════════
# 1. 核心检查函数
# ═══════════════════════════════════════════════

def check_decode_integrity(path, hwaccel='off'):
    """v4 合并检查: 一次 ffmpeg -v verbose -f null → (frames, packets, pts_issues, error).

    一次解码产出三项数据:
      - frames:    从 stderr 末行 frame= N 提取（解码器输出帧数）
      - packets:   ffprobe -count_packets（O(1) 读容器包头，始终独立提取，无需解码）
      - pts_issues: 从同次 verbose stderr 提取 anomaly/error/解码错误行

    对比旧版 check_frames_packets + check_pts_anomaly:
      frames 来源从 ffprobe nb_read_frames（demuxer→decoder 送入帧数）
      变为 ffmpeg decoder output frame=N（解码器实际输出帧数）。
      正常码流下二者相等；损坏码流下 decoder output ≤ demuxer frames，
      仍能被 frames ≠ packets 检测到（静默丢帧会同时反映为两者差异）。
      节省 CPU 软解场景下一次完整解码（~50% 时间），GPU 路径同样受益
      （旧版 GPU 路径也是两次解码: 一次 -f null 取帧数 + 一次 -v verbose 取异常）。
    """
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')

    # ── packets: ffprobe -count_packets (O(1)，只读容器包头，不涉及解码) ──
    packets = None
    pkt_err = None
    if ffprobe:
        r_pkts = _run([ffprobe, '-v', 'error', '-count_packets',
                       '-select_streams', 'v:0',
                       '-show_entries', 'stream=nb_read_packets',
                       '-of', 'csv=p=0', str(path)])
        m_pkts = re.match(r'(\d+)', r_pkts.stdout.strip())
        if m_pkts:
            packets = int(m_pkts.group(1))
        else:
            pkt_err = 'ffprobe packets 解析失败: %r' % r_pkts.stdout[:200]

    if not ffmpeg:
        return None, packets, None, 'ffmpeg 不可用（PATH 缺失）'

    # ── frames + pts_issues: 一次 ffmpeg -v verbose -f null 完成 ──
    # [FIX-PTS-SHOWINFO] 统一加 -vsync 0 -vf showinfo：解析逐帧 pts 检测原始
    # dup/drop/backward 异常（对齐 analyze_video_pipeline_v3 与 verify_segment
    # _bitstream_v2）。旧版用 'pts_anomaly' 关键字在 ffmpeg stderr 里查找，但该串
    # 是自定义日志格式，ffmpeg 从不输出 → pts 检查恒空转 OK（与 v3 矛盾）。
    _showinfo_flags = ['-vsync', '0', '-vf', 'showinfo']
    hw_flags = None
    if hwaccel == 'cuda':
        hw_flags = ['-hwaccel', 'cuda']
    elif hwaccel == 'auto':
        hw_flags = _HWACCEL_AUTO_FLAGS

    stderr_data = None
    frames = None

    if hw_flags:
        # GPU 路径先尝试硬件解码
        r = _run([ffmpeg, '-hide_banner', '-v', 'verbose'] + hw_flags +
                 # [FIX-NVDEC-THREAD-CAP] 解码线程钳位 ≤8，避免多核机上
                 # NVDEC 解码 surface 超过驱动 32 上限。
                 ['-threads', str(_clamp_decode_threads()),
                  '-i', str(path), '-an'] + _showinfo_flags + ['-f', 'null', '-'])
        stderr_data = r.stderr
        # 回退 CPU 软解的两个条件:
        #  1) hwaccel 初始化失败（如 cuvidCreateDecoder 报错）——此时 ffmpeg 已内部
        #     回退软解，frame=N 依然输出，仅凭 'frame=' 判断会失效；显式检测
        #     NVDEC 失败特征行，命中即丢弃 GPU 轮 stderr 重跑，取干净日志。
        #  2) stderr 中无 frame=N（无实际解码输出）——原兜底判断。
        hw_init_fail = bool(stderr_data) and any(
            _is_hwaccel_init_failure(line.lower())
            for line in stderr_data.splitlines())
        if hw_init_fail or 'frame=' not in (stderr_data or ''):
            if hw_init_fail:
                print("[WARN] ffmpeg NVDEC hwaccel 初始化失败，已回退 CPU 软解: %s" % path)
            r = _run([ffmpeg, '-hide_banner', '-v', 'verbose',
                      '-threads', str(_clamp_decode_threads()),
                      '-i', str(path), '-an'] + _showinfo_flags + ['-f', 'null', '-'])
            stderr_data = r.stderr
    else:
        # CPU 路径
        r = _run([ffmpeg, '-hide_banner', '-v', 'verbose',
                  '-threads', str(_clamp_decode_threads()),
                  '-i', str(path), '-an'] + _showinfo_flags + ['-f', 'null', '-'])
        stderr_data = r.stderr

    # 提取 decoded frames: stderr 末行 frame= N
    if stderr_data:
        for line in reversed((stderr_data or '').splitlines()):
            m = re.search(r'frame=\s*(\d+)', line)
            if m:
                frames = int(m.group(1))
                break

    # [FIX-PTS-SHOWINFO] 解析逐帧 pts 检测原始异常（对齐 analyze_video_pipeline_v3
    # 的 _parse_showinfo_line 算法与 verify_segment_bitstream_v2）：pts 回退/重复/
    # 跳变 → 计入 issues。旧版用 'pts_anomaly' 关键字在 ffmpeg stderr 里查找，但
    # 该串是自定义格式，ffmpeg 从不输出 → 检查恒空转 OK。现改为真实解析。
    _MAX_PTS_ISSUES = 200
    issues = []
    if stderr_data:
        _pts_last = None
        _pts_interval = None
        _pts_threshold = 2.0
        _pts_total_issues = 0
        for line in stderr_data.splitlines():
            if len(issues) >= _MAX_PTS_ISSUES and _pts_total_issues >= _MAX_PTS_ISSUES:
                break
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
            # 真实解码错误仍计入（非 pts 空转）。
            if 'non-existing' in low or ('error' in low and 'parsed_showinfo' not in low):
                if len(issues) < _MAX_PTS_ISSUES:
                    issues.append(line.strip()[:200])
                _pts_total_issues += 1
            # ── showinfo 逐帧 pts 异常检测（对齐 v3 _parse_showinfo_line）──
            if 'parsed_showinfo' in low:
                m = re.search(r'n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([\d.]+)', line)
                if not m:
                    m = re.search(r'n:\s*(\d+)\s+pts:\s*(-?\d+)', line)
                if not m:
                    continue
                _n = int(m.group(1))
                _pts = int(m.group(2))
                _pts_t = float(m.group(3)) if len(m.groups()) >= 3 else None
                if _pts_last is not None:
                    _diff = _pts - _pts_last
                    if _diff == 0:
                        _rec = 'pts dup: frame=%d pts=%d（与上一帧相同）' % (_n, _pts)
                        _pts_total_issues += 1
                        if len(issues) < _MAX_PTS_ISSUES:
                            issues.append(_rec)
                    elif _diff < 0:
                        _rec = ('pts backward: frame=%d pts=%d（回退 %d，%d->%d）'
                                % (_n, _pts, _diff, _pts_last, _pts))
                        _pts_total_issues += 1
                        if len(issues) < _MAX_PTS_ISSUES:
                            issues.append(_rec)
                    else:
                        if _pts_interval is None:
                            _pts_interval = _diff
                        elif _diff > _pts_threshold * _pts_interval:
                            _est = max(1, int(round(_diff / _pts_interval)) - 1)
                            _rec = ('pts drop: frame=%d pts=%d diff=%d interval≈%d est_missing=%d'
                                    % (_n, _pts, _diff, _pts_interval, _est))
                            _pts_total_issues += 1
                            if len(issues) < _MAX_PTS_ISSUES:
                                issues.append(_rec)
                        else:
                            _pts_interval = (_pts_interval * 3 + _diff) // 4
                _pts_last = _pts
        if _pts_total_issues > len(issues):
            issues.append('...（共 %d 条 pts/解码异常，仅显示前 %d 条）'
                          % (_pts_total_issues, len(issues)))

    # 综合错误: packets 解析失败优先, 其次 frames 提取失败
    err = None
    if pkt_err:
        err = pkt_err
    elif frames is None:
        err = 'ffmpeg 解码失败（无法提取帧数）'

    return frames, packets, issues, err


# ── [v5] 视频编码探测 ──

def probe_video_codec(path):
    """探测视频流编码名称，返回 codec_name 小写（如 'h264'/'mpeg4'/'hevc'），失败返回 None。

    用 ffprobe 读 v:0 流的 codec_name（O(1) 读容器包头，不涉及解码）。
    DivX/Xvid 在 ffprobe 里 codec_name 统一为 'mpeg4'（MPEG-4 Part 2）。
    失败（ffprobe 缺失/无视频流/读取异常）返回 None，不中断流程——
    上层按未知编码处理，跳过专项码流分析显示 N/A。
    """
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None
    try:
        r = _run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'stream=codec_name',
                  '-of', 'csv=p=0', str(path)])
        name = (r.stdout or '').strip().lower()
        if name and r.returncode == 0:
            return name
    except Exception:
        pass
    return None


# MPEG-4 Part 2 家族 codec_name（全部按 mpeg4 VOP 深度分析处理）
# 注: msmpeg4v2/v3 (DivX 老格式) 结构同为 MPEG-4 Part 2 衍生，start code 兼容。
_MPEG4_FAMILY = {'mpeg4', 'msmpeg4v2', 'msmpeg4v3'}

# [v6] 段首连 IDR 异常阈值（与 mpeg4 连 I-VOP 判据 max_consec_i_vop > 3 对齐）。
# SPS/PPS 冗余重注入 (FIX-SPS-PPS-V2) 的良性段首双 IDR 模式只产生 1 个
# 「首个 IDR 之后 32 NAL 窗口内」的额外 IDR；修复前 per-slot IDR 异常为 6-16+ 个。
_CONSEC_IDR_THRESHOLD = 3

# [v6] 色度坏帧判 FAIL 的最小数量（防单帧合法高色度画面如闪光误报）。
_CHROMA_BAD_FRAME_MIN = 3

# [v6.1] chroma 检查分块帧数：一次读入 N 帧做批量 std 归约（axis=1），
# 消除逐帧 Python/NumPy 调用开销；内存 O(N*frame_size) 有界，仍为流式。
_CHROMA_CHUNK_FRAMES = 128


def extract_annexb_es(path, codec=None):
    """按编码分支提取 Annex B 结构码流字节（v5: 支持 h264/mpeg4 家族/hevc）。

    返回 (es_bytes, codec_ok):
      - es_bytes: 提取到的 Annex B ES 字节；非支持编码返回 None
      - codec_ok: True 表示该编码有专项码流分析；False 表示 N/A（不判失败）

    各编码分支:
      - h264 → ffmpeg -c:v copy -bsf:v h264_mp4toannexb -f h264（原 v2 逻辑）
      - mpeg4 家族 → ffmpeg -c:v copy -f m4v（AVI 中 mpeg4 已是 Annex B，
        无需 bitstream filter；m4v muxer 直接输出原始 VOP start code 结构）
      - hevc → -bsf:v hevc_mp4toannexb（bsf 存在时）；否则 None
      - 其他 (vp9/av1/...) → None（无专项分析，显示 N/A）

    提取失败抛 RuntimeError（含 stderr 前 500 字符），与 v2 错误报告模式一致。
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg 不可用（PATH 缺失）')

    if codec is None:
        codec = probe_video_codec(path)
    codec = (codec or '').lower()

    # 未知编码或明确不支持专项分析的编码 → N/A（不抛错，上层显示 N/A）
    if codec not in ('h264',) and codec not in _MPEG4_FAMILY and codec != 'hevc':
        return None, False

    if codec == 'h264':
        cmd = [ffmpeg, '-v', 'error', '-i', str(path),
               '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb',
               '-f', 'h264', 'pipe:1']
    elif codec in _MPEG4_FAMILY:
        # mpeg4: -f m4v 输出 Annex B；本机 ffmpeg 无 mpeg4_mp4toannexb bsf
        # （实测 -bsfs 只有 mpeg4_unpack_bframes），直接 copy + m4v muxer。
        cmd = [ffmpeg, '-v', 'error', '-i', str(path),
               '-c:v', 'copy', '-f', 'm4v', 'pipe:1']
    else:  # hevc
        # hevc_mp4toannexb bsf 需存在才可用，否则 N/A
        if not _bsf_available('hevc_mp4toannexb'):
            return None, False
        cmd = [ffmpeg, '-v', 'error', '-i', str(path),
               '-c:v', 'copy', '-bsf:v', 'hevc_mp4toannexb',
               '-f', 'hevc', 'pipe:1']

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg 提取 ES 失败: %s' % r.stderr.decode('utf-8', 'replace')[:500])
    return r.stdout, True


_BSF_CACHE = None  # 全局缓存一次 ffmpeg -bsfs 结果


def _bsf_available(name):
    """探测 ffmpeg 是否存在指定 bitstream filter（全局缓存一次，避免重复调用）。"""
    global _BSF_CACHE
    if _BSF_CACHE is None:
        try:
            r = _run(['ffmpeg', '-hide_banner', '-bsfs'])
            _BSF_CACHE = (r.stdout or '').split()
        except Exception:
            _BSF_CACHE = []
    return name in _BSF_CACHE


# ═══════════════════════════════════════════════
# H.264 NAL 解析（v2 原有逻辑，v5 保留不变）
# ═══════════════════════════════════════════════

class _BitReader:
    """RBSP 位读取器（MSB-first），按需去除 emulation prevention bytes（0x000003）。

    v4 懒加载版: 不在构造时预处理全部 RBSP 字节，而是在 read_bit() 时按需填充。
    对仅需读 slice header 前若干位的场景（frame_num 位于 header 开头约 2-3 字节后），
    每个 slice 仅处理约 5-10 字节而非整个 RBSP（可能 50-200KB），千倍加速。
    """
    def __init__(self, data):
        self.raw = data
        self.rbsp = bytearray()  # 已处理的 RBSP 缓冲区（按需增长）
        self.raw_pos = 0         # 在 raw 中的当前位置
        self.bit_pos = 0         # 在 rbsp 中的当前位位置
        self._zeros = 0          # 连续零字节计数（用于 0x000003 检测）

    def _fill_bytes(self, need_bytes):
        """确保 rbsp 缓冲区至少有 need_bytes 字节（不足时从 raw 继续处理）。"""
        while len(self.rbsp) < need_bytes and self.raw_pos < len(self.raw):
            b = self.raw[self.raw_pos]
            self.raw_pos += 1
            # emulation prevention: 检测 0x000003 → 跳过 0x03
            if self._zeros >= 2 and b == 0x03:
                self._zeros = 0
                continue
            if b == 0:
                self._zeros += 1
            else:
                self._zeros = 0
            self.rbsp.append(b)

    def read_bit(self):
        byte_idx = self.bit_pos >> 3
        self._fill_bytes(byte_idx + 1)
        if byte_idx >= len(self.rbsp):
            raise ValueError('RBSP 位越界')
        bit = (self.rbsp[byte_idx] >> (7 - (self.bit_pos & 7))) & 1
        self.bit_pos += 1
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
    """解析 SPS RBSP → (frame_num_bits, separate_colour_plane_flag)。

    [FIX-HIGH-PROFILE-FN-BITS] High/Extended profile (100/110/122/244 等)
    在 seq_parameter_set_id 之后还有 chroma_format_idc / bit_depth 等扩展字段，
    旧实现漏读直接取 log2_max_frame_num_minus4，把 chroma_format_idc 的值
    误当 log2（NVENC High profile 输出实测 log2_max_frame_num_minus4=4 →
    frame_num_bits 应为 8，旧解析得 4 → 大量伪"frame_num 回退"误报）。
    """
    br = _BitReader(payload)
    profile_idc = br.read_bits(8)
    br.read_bits(8)  # constraint_set0..5 + reserved_zero_2bits
    br.read_bits(8)  # level_idc
    br.read_ue()     # seq_parameter_set_id
    separate_colour_plane = False
    high_profile = profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 134, 135, 138, 139)
    if high_profile:
        chroma_format_idc = br.read_ue()
        if chroma_format_idc == 3:
            separate_colour_plane = bool(br.read_bit())
        br.read_ue()  # bit_depth_luma_minus8
        br.read_ue()  # bit_depth_chroma_minus8
        br.read_bit()  # qpprime_y_zero_transform_bypass_flag
        if br.read_bit():  # seq_scaling_matrix_present_flag
            n_scaling = 8 if chroma_format_idc != 3 else 12
            for _i in range(n_scaling):
                if br.read_bit():  # seq_scaling_list_present_flag
                    _skip_scaling_list(br)
    log2_max_frame_num_minus4 = br.read_ue()
    return log2_max_frame_num_minus4 + 4, separate_colour_plane


def _skip_scaling_list(br):
    """跳过 H.264 scaling list (8x8 与 4x4 的 delta 序列)。"""
    size = 16 if br.read_ue() == 0 else 64
    last_scale = 8
    next_scale = 8
    for _j in range(size):
        if next_scale != 0:
            delta_scale = br.read_se()
            next_scale = (last_scale + delta_scale + 256) % 256
        last_scale = (next_scale if next_scale != 0 else last_scale)


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
    """解析 Annex B ES → [(nal_type, frame_num), ...]（仅 VCL/SPS/PPS/IDR）。

    v4 优化: 使用 bytes.find() (C 级) 代替逐字节 NAL 边界扫描；
    配合 _BitReader 懒加载，整体解析从 O(ES_size × N_slices) 降为 O(ES_size + N_slices×10)。
    """
    nals = []
    n = len(es)
    frame_num_bits = 8  # 默认值；遇到 SPS 后按 log2_max_frame_num_minus4 更新
    separate_colour_plane = False

    offset = 0
    while offset < n:
        # C-level scan for next start code prefix \x00\x00\x01
        next_sc3 = es.find(b'\x00\x00\x01', offset)
        if next_sc3 == -1:
            break

        # 判断是 3 字节 (\x00\x00\x01) 还是 4 字节 (\x00\x00\x00\x01) 起始码
        if next_sc3 >= 1 and es[next_sc3 - 1] == 0x00:
            sc_pos = next_sc3 - 1  # 4-byte: 0x00000001
            sc_len = 4
        else:
            sc_pos = next_sc3
            sc_len = 3

        payload_start = sc_pos + sc_len

        # 找下一个起始码作为当前 NAL 的结束边界
        next_sc3 = es.find(b'\x00\x00\x01', payload_start)
        if next_sc3 == -1:
            payload_end = n
        else:
            if next_sc3 >= 1 and es[next_sc3 - 1] == 0x00:
                payload_end = next_sc3 - 1
            else:
                payload_end = next_sc3

        payload = es[payload_start:payload_end]
        offset = payload_end  # 继续从当前 NAL 末尾搜索

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

    return nals, frame_num_bits


# NOTE: frame_num 严格按 SPS log2_max_frame_num_minus4 位宽解 slice header。
# 早期版本曾用 RBSP 第 2 字节近似——但该字节实际是 first_mb_in_slice/slice_type/
# pps_id 的位域，frame_num 并不在其中，对部分码流会产生大量误报回退；
# 现改为完整 slice header 解析（含 emulation prevention 去除与
# separate_colour_plane_flag 处理）。
def check_nal_stats(nals, frame_num_bits, dump_path=None):
    # 将 IDR slice 按帧聚类: 同一帧的多个 IDR slice 计为 1 个 IDR 帧。
    # [FIX-SAME-FRAME-IDR-SLICES] NVENC 在较大分辨率下每帧编码为多个 slice
    # （如 1280x720 为 4 slice/帧，2560x1440 更多），同一帧的各 IDR slice 在
    # NAL 流中连续出现（NAL 索引紧邻且 frame_num 相同）；旧逻辑按 slice 统计
    # 会把同帧 slice 误计为"连 IDR"（正常码流报 连IDR=3 的误报来源）。
    # 判据: 与上一个 IDR slice 索引相邻（中间无其他 NAL）且 frame_num 相同
    #       → 同一帧；跨帧 IDR 之间必有 AUD/SEI/P slice/SPS 等分隔 NAL，
    #       索引必然不相邻。frame_num 相同不是充分判据（异常码流中连续多帧
    #       IDR 的 frame_num 都可能为 0），故必须以索引相邻为主。
    idr_frames = []  # [(nal_idx, frame_num), ...]: 每项代表 1 个不同的 IDR 帧
    prev_slice_idx = -2   # 上一个 IDR slice 的 NAL 索引（含被聚类跳过的同帧 slice）
    prev_slice_fn = None
    for i, (t, fn) in enumerate(nals):
        if t != 5:
            continue
        # 同帧判据必须与"上一个 IDR slice"比较而非"上一帧起始 slice"，
        # 否则同帧第 3+ 个 slice 会因与帧起始 slice 不相邻而误判为新帧
        # （对照基准: 4 slice/帧码流的 IDR slice 索引 2,3,4,5 应聚类为 1 帧）。
        if i == prev_slice_idx + 1 and fn == prev_slice_fn:
            prev_slice_idx = i  # 同帧 slice: 只推进 prev，不新增 IDR 帧
            continue
        idr_frames.append((i, fn))
        prev_slice_idx = i
        prev_slice_fn = fn
    idr_count = len(idr_frames)  # 5: IDR（帧级计数）
    first_idr_at = idr_frames[0][0] if idr_frames else None
    # 段首连 IDR 检查（帧级）：首个 IDR 帧后 32 NAL 窗口内是否又出现新的 IDR 帧。
    # 窗口保持 32 NAL 不变（覆盖约 8 个 4-slice 帧），统计对象从 IDR slice 改为
    # 不同 IDR 帧 —— 同帧 slice 不再计入，而真异常（如修复前 per-slot IDR 的
    # 22 连 IDR 帧）因 IDR 帧之间总有分隔 NAL 仍能检出。
    consecutive_idr_after_first = 0
    if idr_frames:
        window_end = first_idr_at + 32
        for idx, _fn in idr_frames[1:]:
            if idx <= window_end:
                consecutive_idr_after_first += 1
            else:
                break  # idr_frames 按 NAL 索引递增，超出窗口即结束
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


# ═══════════════════════════════════════════════
# [v5] HEVC (H.265) 基础 NAL 分析
# ═══════════════════════════════════════════════
# HEVC 无 H.264 的 frame_num 概念（用 POC），此处只做 NAL 类型统计与
# IDR 帧/连 IDR 检查（对应 H.264 检查 2 的可比部分）；vp9/av1 等编码
# 无 Annex B start code 结构，直接由上层提示跳过。

_HEVC_IDR_TYPES = {19, 20}  # IDR_W_RADL / IDR_N_LP


def parse_hevc_es(es):
    """解析 HEVC Annex B ES → [(nal_type, 0), ...]。

    HEVC NAL header 为 2 字节:
      forbidden_zero_bit(1) + nal_unit_type(6) + nuh_layer_id(6)
      + nuh_temporal_id_plus1(3)
    扫描复用 H.264 的 C 级 bytes.find start code 框架（O(ES_size)）。
    """
    nals = []
    n = len(es)
    offset = 0
    while offset < n:
        next_sc3 = es.find(b'\x00\x00\x01', offset)
        if next_sc3 == -1:
            break
        if next_sc3 >= 1 and es[next_sc3 - 1] == 0x00:
            sc_pos = next_sc3 - 1
            sc_len = 4
        else:
            sc_pos = next_sc3
            sc_len = 3
        payload_start = sc_pos + sc_len
        next_sc3 = es.find(b'\x00\x00\x01', payload_start)
        if next_sc3 == -1:
            payload_end = n
        else:
            if next_sc3 >= 1 and es[next_sc3 - 1] == 0x00:
                payload_end = next_sc3 - 1
            else:
                payload_end = next_sc3
        payload = es[payload_start:payload_end]
        offset = payload_end
        if len(payload) >= 2:
            # nal_unit_type = 6 位，位于第 1 字节 bit1-6
            nal_type = (payload[0] >> 1) & 0x3F
            nals.append((nal_type, 0))
    return nals


def check_hevc_stats(nals, dump_path=None):
    """HEVC 基础 NAL 统计（对应 H.264 的 IDR/连 IDR 检查；无 frame_num 回退）。

    返回:
      nal_count                   NAL 单元总数
      idr_count                   IDR 帧数（type 19/20，多 slice 聚类）
      first_idr_at                段首 IDR NAL 索引（0-based，None=无 IDR）
      idr_within_32_after_first   段首 IDR 后 32 NAL 窗口内新 IDR 帧数
      frame_num_regress           恒 0（HEVC 无 frame_num，保留字段对齐）
    """
    # IDR 帧聚类: 同一帧的多 IDR slice NAL 在流中连续（索引紧邻），计 1 帧
    idr_frames = []  # [nal_idx, ...]
    prev_slice_idx = -2
    for i, (t, _x) in enumerate(nals):
        if t not in _HEVC_IDR_TYPES:
            continue
        if i == prev_slice_idx + 1:
            prev_slice_idx = i  # 同帧多 slice: 只推进，不新增帧
            continue
        idr_frames.append(i)
        prev_slice_idx = i
    idr_count = len(idr_frames)
    first_idr_at = idr_frames[0] if idr_frames else None
    consecutive_idr_after_first = 0
    if idr_frames:
        window_end = first_idr_at + 32
        for idx in idr_frames[1:]:
            if idx <= window_end:
                consecutive_idr_after_first += 1
            else:
                break
    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as f:
            for i, (t, _x) in enumerate(nals):
                f.write('%d,%d,0\n' % (i, t))
    return {'nal_count': len(nals),
            'idr_count': idr_count,
            'first_idr_at': first_idr_at,
            'idr_within_32_after_first': consecutive_idr_after_first,
            'frame_num_regress': 0}


# ═══════════════════════════════════════════════
# [v5] MPEG-4 Part 2 VOP 解析与统计
# ═══════════════════════════════════════════════

# MPEG-4 Part 2 start code（Annex B: 前缀 0x000001 + 1 字节 code）。
# 注意: 以下常量均为「code 字节」（紧跟 00 00 01 前缀后的 1 字节，0x00-0xFF），
#       非完整 4 字节 start code 值（0x0001xx）——解析时 sc 为单字节比较。
#       MPEG-4 无 H.264 的 emulation prevention（0x000003），异常文件压缩数据
#       内部可能含伪 0x000001B6 start code，需 header 解析过滤。
_VOS_SC = 0xB0   # visual_object_sequence_start
_VOL_SC_MIN, _VOL_SC_MAX = 0x20, 0x2F  # video_object_layer_start（vol_id 1-32）
_GOV_SC = 0xB3   # group_of_vop_start
_VOP_SC = 0xB6   # vop_start（注意 code 0xB6 同时是 video_object_start，
                 #     但 video_object_start 只出现在 VOL 之前，VOL 后即 VOP）
_EOS_SC = 0xB7   # end_of_video_object_sequence
_VOP_TYPE = {0: 'I', 1: 'P', 2: 'B', 3: 'S'}  # vop_coding_type

# [FIX-PSEUDO-STARTCODE] MPEG-4 Part 2 无 emulation prevention，异常/损坏文件
# 压缩数据内部可能出现伪 0x000001B6 start code。真实 VOP 识别仅依赖
# VOP header 解析成功（modulo_time_base + vop_time_increment + vop_coding_type）。
# 注: 前导字节不可作过滤依据——GOV/user_data payload 可能以 0x00 结尾，
#     紧邻真实 VOP 前导字节恰为 0x00；4 字节前缀由 sc_len 判断处理。
# 实测: 干净 mpeg4 (S12E16, 14924帧) 扫描数==解码帧数==14924 完全吻合；
#       异常文件 (112 Max Bed Time.avi) 扫描出 19825 但解码仅 14799，
#       VOP 总数 > packets/frames 的差异即码流异常告警（展示层交叉核对）。


def _parse_mpeg4_vol(payload):
    """解析 MPEG-4 VOL header → vop_time_increment_bits（None=解析失败）。

    按 ISO/IEC 14496-2 video_object_layer() 语法顺序读取:
      random_accessible_vol(1) + video_object_type_indication(8)
      + is_object_layer_identifier(1) [+ video_object_layer_verid(4) + priority(3)]
      + aspect_ratio_info(4) [+ par_width(8) + par_height(8) 当 ari==3]
      + vol_control_parameters(1) [+ chroma_format(2) + low_delay(1)
          + vbv_parameters(1) [+ bit_rate 15+marker 两段 + vbv_buffer 15+marker 两段
                               + vbv_occupancy 11+marker 两段]]
      + video_object_layer_shape(2)
      + marker(1) + vop_time_increment_resolution(16) + marker(1)
      + fixed_vop_rate(1)
    vop_time_increment_bits = ceil(log2(resolution))；resolution ≤ 1 时 bits=0
    （时间戳恒 0，上层跳过时间回退检查）。
    实测: resolution=2997/4123 与 ffprobe r_frame_rate(2997/100≈29.97fps) 精确吻合。
    """
    try:
        br = _BitReader(payload)
        br.read_bit()       # random_accessible_vol
        br.read_bits(8)     # video_object_type_indication
        is_oid = br.read_bit()  # is_object_layer_identifier
        if is_oid:
            br.read_bits(4)  # video_object_layer_verid
            br.read_bits(3)  # video_object_layer_priority
        ari = br.read_bits(4)  # aspect_ratio_info
        if ari == 3:
            br.read_bits(8)   # par_width
            br.read_bits(8)   # par_height
        if br.read_bit():    # vol_control_parameters
            br.read_bits(2)  # chroma_format
            br.read_bit()    # low_delay
            if br.read_bit():  # vbv_parameters
                for _ in range(3):
                    br.read_bits(15)
                    br.read_bit()  # marker
                for _ in range(3):
                    br.read_bits(15)
                    br.read_bit()  # marker
                for _ in range(2):
                    br.read_bits(11)
                    br.read_bit()  # marker
        shape = br.read_bits(2)  # video_object_layer_shape
        # shape 非矩形（binary/gray/bitmap）时还有额外 shape 参数——
        # 但常见 mpeg4 (XVID) 均为矩形 shape=0，直接跳过；异常则 catch 外层 ValueError。
        if shape != 0:
            if shape in (1, 3):
                br.read_bit()  # binary shape: intra_slice_permitted(1)
            if shape == 2:
                br.read_bit()  # gray shape: volumetric_vol(1)
            # 完整 shape 参数较复杂，此处只读前 1-2 位；非矩形文件会在此近似解析，
            # 若后续 resolution 读偏，由外层 try/except 降级为 N/A。
        br.read_bit()          # marker_bit(1) 应=1
        resolution = br.read_bits(16)  # vop_time_increment_resolution
        br.read_bit()          # marker_bit(1) 应=1
        br.read_bit()          # fixed_vop_rate(1)
        if resolution <= 1:
            return 0
        return int(math.ceil(math.log2(resolution)))
    except ValueError:
        return None


def _parse_mpeg4_vop(payload, time_bits):
    """解析单个 MPEG-4 VOP header → (vop_coding_type, vop_time_increment, mtb)。

    实测 ffmpeg mpeg4 编码器（mpeg4videoenc.c mpeg4_encode_picture_header）位序:
      vop_coding_type(2 位: 0=I,1=P,2=B,3=S)
      → modulo_time_base(连续读 1 计数，读到 0 结束并消费)
      → marker_bit(1)
      → vop_time_increment(vop_time_increment_bits 位)
      → marker_bit(1)
      → vop_coded(1)
    （与 ISO 标准书面的 mod/时间戳/type 顺序不同，以实际编码器为准；
      实测 ffmpeg 生成样本 + XVID 文件均按此顺序。）
    解析失败抛 ValueError（由调用方跳过该单元——伪 start code 或损坏）。
    """
    br = _BitReader(payload)
    ctype = br.read_bits(2)
    mtb = 0
    while br.read_bit() == 1:
        mtb += 1
    br.read_bit()  # marker_bit
    ti = br.read_bits(time_bits) if time_bits else 0
    br.read_bit()  # marker_bit
    br.read_bit()  # vop_coded
    return ctype, ti, mtb


def parse_mpeg4_es(es):
    """解析 MPEG-4 Annex B ES → (vops, time_bits)。

    vops: [(vop_coding_type, vop_time_increment, modulo_time_base), ...]，
          按出现顺序。mtb 用于时间回退检查的周期重置判定。
    time_bits: vop_time_increment_bits（None=VOL 解析失败，时间回退检查降级 N/A，
               但 VOP 类型计数仍可用；此时 vop_time_increment 不解析置 0）。

    扫描复用 H.264 的 C 级 bytes.find start code 框架（O(ES_size)）；
    用前导字节非 0x00 + VOP header 解析成功过滤伪 start code。
    """
    vops = []
    n = len(es)
    time_bits = None
    have_vol = False

    offset = 0
    while offset < n:
        next_sc = es.find(b'\x00\x00\x01', offset)
        if next_sc == -1:
            break
        # start code 长度判断: 4 字节 (00 00 00 01) vs 3 字节 (00 00 01)
        if next_sc >= 1 and es[next_sc - 1] == 0x00:
            sc_pos = next_sc - 1
            sc_len = 4
        else:
            sc_pos = next_sc
            sc_len = 3
        code = es[sc_pos + sc_len] if sc_pos + sc_len < n else -1
        payload_start = sc_pos + sc_len + 1

        # 找下一个 start code 作为结束边界
        next_sc = es.find(b'\x00\x00\x01', payload_start)
        if next_sc == -1:
            payload_end = n
        else:
            if next_sc >= 1 and es[next_sc - 1] == 0x00:
                payload_end = next_sc - 1
            else:
                payload_end = next_sc
        payload = es[payload_start:payload_end]
        offset = payload_end

        if code == -1:
            continue

        if _VOL_SC_MIN <= code <= _VOL_SC_MAX:
            # VOL: 更新 vop_time_increment_bits（多 VOL 时以后者为准）
            have_vol = True
            tb = _parse_mpeg4_vol(payload)
            if tb is not None:
                time_bits = tb
        elif code == _VOP_SC and have_vol:
            # [FIX-PSEUDO-STARTCODE] MPEG-4 无 emulation prevention，异常/损坏文件
            # 压缩数据内部可能含伪 0x000001B6 start code。真实 VOP 过滤:
            #   仅依赖 VOP header 解析成功（前导字节不可作过滤依据——
            #   GOV/user_data 等 payload 可能以 0x00 结尾，紧邻的真实 VOP
            #   前导字节恰好为 0x00；4 字节前缀 0x00000001 已由 sc_len 判断处理）。
            try:
                ctype, ti, mtb = _parse_mpeg4_vop(payload, time_bits)
            except (ValueError, IndexError):
                offset = payload_end
                continue
            if time_bits is None:
                ti = 0  # VOL 解析失败时时间戳不可用
            vops.append((ctype, ti, mtb))
        # 其他 start code (VOS/VO/GOV/user_data/EOS) 跳过

    return vops, time_bits


def check_vop_stats(vops, time_bits, dump_path=None):
    """mpeg4 VOP 统计（对应 H.264 三项检查）。

    vops: [(vop_coding_type, vop_time_increment, modulo_time_base), ...]
    time_bits: vop_time_increment_bits（None → 时间回退检查返回 N/A）

    返回:
      vop_count_total                VOP 总数（供与 frames/packets 交叉核对）
      i_vop_count                    I-VOP 帧数（vop_coding_type=0）
      first_i_vop_at                段首 I-VOP 在 VOP 序列中的序号（1-based，None=无 I）
      max_consec_i_vop              最长连续 I-VOP 块长度（相邻索引差==1 的连续 I 序列，
                                    修复工具重编码特征；对应 H.264 连 IDR 检查）
      vop_time_regress              时间回退次数（None=时间信息不可用/N/A）

    连 I-VOP 检查: MPEG-4 一帧一个 VOP（无 slice 概念，无需聚类）。
    注: 不能沿用 H.264 的「段首窗口内新 IDR」判据——mpeg4 正常 GOP 周期性
    出现 I 帧（如每 12 帧一个），段首 I 后 32 VOP 窗口内必然有新 I 帧。
    异常特征 = 连续 I-VOP 块（如 fixed.avi 21032 连 I）；段首 1-2 个 I 预热
    帧是 XVID 正常行为（如 S12E16 VOP#1/#2 双 I），判定用最长连续 I 块 > 3。

    时间回退检查（仅 I/P-VOP，type 0/1，B-VOP 天然时间回退需过滤）:
      modulo_time_base 递增（满位宽回绕）时 time 重新从低位递增，视为合法重置；
      同一 mtb 周期内 time 单调，回退量 0 < diff < 半程(1<<(bits-1)) 计 regress
      （复用 H.264 frame_num 半程判据思路）。
    """
    # 连 I-VOP 检查: 最长连续 I-VOP 块长度（VOP 序列 1-based 序号）
    i_idx = [i + 1 for i, (ct, _t, _m) in enumerate(vops) if ct == 0]
    first_i = i_idx[0] if i_idx else None
    max_consec_i_vop = 0
    if i_idx:
        cur = 1
        max_consec_i_vop = 1
        for _a, _b in zip(i_idx, i_idx[1:]):
            if _b - _a == 1:
                cur += 1
            else:
                cur = 1
            if cur > max_consec_i_vop:
                max_consec_i_vop = cur

    # 时间回退检查（仅 I/P，且仅当 time_bits 有效；mtb 变化视为周期重置）
    regress = None
    if time_bits is not None and time_bits >= 1:
        half = 1 << (time_bits - 1)
        regress = 0
        prev_ti = None
        prev_mtb = None
        for ct, ti, mtb in vops:
            if ct not in (0, 1):  # 仅 I/P-VOP；B-VOP 时间戳天然回退，跳过
                continue
            if mtb != prev_mtb:
                # 进入新的 modulo_time_base 周期，time 重新从低位递增，重置锚点
                prev_ti = ti
                prev_mtb = mtb
                continue
            if prev_ti is not None:
                diff = prev_ti - ti
                if 0 < diff < half:
                    regress += 1
            prev_ti = ti

    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as f:
            for idx, (ct, ti, _mtb) in enumerate(vops):
                f.write('%d,%s,%d\n' % (idx, _VOP_TYPE.get(ct, '?'), ti))

    return {'vop_count_total': len(vops),
            'i_vop_count': len(i_idx),
            'first_i_vop_at': first_i,
            'max_consec_i_vop': max_consec_i_vop,
            'vop_time_regress': regress}


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
    # 方式 1: pynvml (nvidia-ml-py)
    try:
        import pynvml
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
# 4b. 色度花屏检测（[v6] 像素域检查，与码流结构检查正交）
# ═══════════════════════════════════════════════

def check_chroma_corruption(path, hwaccel=False):
    """[v6] 解码 yuv420p 逐帧计算 U/V 平面 std，自校准阈值识别色度花屏坏帧。

    原理: 正常视频 U/V 平面 std 稳定 (v6.4.4/5 全程 U≈35、V≈25)；
    批量编码跨流竞态导致的 NV12 撕裂/色度污染会使部分帧 U/V std 飙升至
    2-3 倍基线（v6.4.3 实测坏帧 73-90 vs 基线 36）。这是像素域信号，
    能捕获 frames/packets、连 IDR、frame_num、pts_anomaly 全部漏检的
    「布纹花屏/帧闪烁」缺陷。

    流式处理: ffmpeg rawvideo 输出按 _CHROMA_CHUNK_FRAMES 帧分块读取，
    每块一次批量 std 归约（axis=1），内存占用 O(chunk*frame_size) 有界，
    不随帧数增长（720×576×1373 帧 ≈ 850MB pipe 全流式）。
    坏帧索引按间距分簇（间距 >= 8 帧算新簇，仅保留簇内首帧），避免单簇多帧刷屏。

    hwaccel: True 时用 NVDEC 解码（帧自动回拷主机，无需 hwdownload），
    加速解码但 NVDEC 错误隐藏与软解不一致，可能掩盖花屏特征，默认 False；
    NVDEC 不可用/解码失败时自动回退 CPU 软解（对齐 v4.1 GPU→CPU 回退）。

    返回 dict（ffprobe/ffmpeg 失败或解码异常返回 None，调用方不判失败）:
        frame_count: 解码帧数
        median_u / median_v: 各帧 U/V std 的中位数
        bad_frames: 坏帧索引列表（分簇去重后）
        bad_count:   坏帧簇总数（调用方以 >= _CHROMA_BAD_FRAME_MIN 判 FAIL）
    """
    if shutil.which('ffprobe') is None or shutil.which('ffmpeg') is None:
        return None
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=30)
        if probe.returncode != 0 or not probe.stdout.strip():
            return None
        parts = probe.stdout.strip().replace('\r', '').split('\n')[-1].split(',')
        width, height = int(parts[0]), int(parts[1])
        if width <= 0 or height <= 0:
            return None
    except Exception:
        return None

    frame_size = width * height * 3 // 2
    y_size = width * height
    uv_size = width * height // 4

    # 默认 CPU 软解（参考级/确定性解码）；hwaccel=True 时用 NVDEC 解码，
    # 帧自动回拷主机（无 -hwaccel_output_format，无需 hwdownload 滤镜）。
    cmd = ['ffmpeg', '-v', 'error']
    if hwaccel:
        cmd += ['-hwaccel', 'cuda']
    cmd += ['-i', str(path), '-an', '-vsync', '0',
            '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    chunk_bytes = frame_size * _CHROMA_CHUNK_FRAMES

    def _stream_uv_std(cmd):
        """按 chunk 分块读取 rawvideo，批量 std 归约（内存 O(chunk*frame_size)）。"""
        us, vs = [], []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, shell=False) as proc:
            while True:
                buf = proc.stdout.read(chunk_bytes)
                if not buf:
                    break
                n = len(buf) // frame_size
                if n == 0:
                    break
                # 截断至整帧边界（管道末尾不完整帧丢弃，与原逐帧读取语义一致）
                data = buf if len(buf) == n * frame_size else buf[:n * frame_size]
                planes = np.frombuffer(data, dtype=np.uint8).reshape(n, frame_size)
                # 每帧 U/V 平面 → (n, uv_size) 视图，axis=1 批量归约（uint8→float64）
                us.append(planes[:, y_size:y_size + uv_size].std(axis=1))
                vs.append(planes[:, y_size + uv_size:].std(axis=1))
        return us, vs

    us = vs = None
    try:
        us, vs = _stream_uv_std(cmd)
    except Exception:
        if not hwaccel:
            return None
    # [v6.1] NVDEC 不可用/解码失败（如无 nvcuda.dll，ffmpeg 无 rawvideo 输出）
    # 或抛出异常 → 回退 CPU 软解（对齐 v4.1 检查 1 的 GPU→CPU 回退哲学）
    if (not us) and hwaccel:
        try:
            us, vs = _stream_uv_std(['ffmpeg', '-v', 'error', '-i', str(path),
                                     '-an', '-vsync', '0', '-pix_fmt', 'yuv420p',
                                     '-f', 'rawvideo', 'pipe:1'])
        except Exception:
            return None

    if len(us) == 0:
        return None
    us = np.concatenate(us)
    vs = np.concatenate(vs)
    median_u = float(np.median(us))
    median_v = float(np.median(vs))
    # 自校准相对阈值: 中位数 * 1.6，绝对下限 12（防灰白/黑屏视频 median≈0 误报）
    thr_u = max(median_u * 1.6, 12.0)
    thr_v = max(median_v * 1.6, 12.0)
    bad_u = np.where(us > thr_u)[0]
    bad_v = np.where(vs > thr_v)[0]
    bad_all = np.unique(np.concatenate([bad_u, bad_v]))

    # 分簇去重: 相邻坏帧 (间距 < 8) 视为同一污染事件，仅保留簇内首帧
    clustered = []
    for idx in bad_all:
        idx = int(idx)
        if not clustered or idx - clustered[-1] >= 8:
            clustered.append(idx)

    return {
        'frame_count': len(us),
        'median_u': round(median_u, 2),
        'median_v': round(median_v, 2),
        'bad_frames': clustered,
        'bad_count': len(clustered),
    }


# ═══════════════════════════════════════════════
# 5. 单文件验收（模块级函数，ThreadPoolExecutor 可 pickle）
# ═══════════════════════════════════════════════

def _verify_one_video(video_path, hwaccel='auto', dump_nal=None, skip_chroma=False,
                      chroma_hwaccel=False):
    """对单个视频执行全部 3 项检查，返回结构化结果。

    v4 变更: 检查 1 (frames/packets) 与检查 3 (pts_anomaly) 合并为
    一次 ffmpeg -v verbose -f null 解码 (check_decode_integrity)，
    节省 CPU 软解与 GPU 路径下各一次完整解码。

    v5 变更: 新增 codec 探测，检查 2 按编码分支:
      - h264        → extract_annexb_es + parse_h264_es + check_nal_stats（原逻辑不变）
      - mpeg4 家族  → extract_annexb_es(-f m4v) + parse_mpeg4_es + check_vop_stats
                      （I-VOP 计数/连 I-VOP/vop_time 回退，对应 H.264 三项）
      - 其他编码    → 无专项码流分析，显示 N/A 不判失败
    非 h264 的 frames/packets 守恒仍严格判定（frames != packets 判 FAIL），
    mpeg4 时 VOP 总数在展示层输出供交叉核对（区分容器假象与真丢帧）。

    GPU 任务受 _GPU_SEMAPHORE 闸门限制 (thread 模式)；
    process 模式信号量不可跨进程共享，由 --gpu-workers 上限约束。

    返回 dict:
        path:           输入路径字符串
        label:          显示标签（传参传入，无则用文件名）
        codec:          视频编码名（probe_video_codec 结果，None=未知）
        frames:         nb_read_frames
        packets:        nb_read_packets
        fp_match:       frames == packets
        fp_err:         检查 1 错误描述 (None=通过)
        nal_stats:      check_nal_stats / check_vop_stats 返回的 dict
                        (None=解析失败或 N/A)
        nal_err:        检查 2 错误描述 (None=通过)
        pts_issues:     check_pts_anomaly 返回的 list (None=跳过/失败)
        pts_err:        检查 3 错误描述 (None=通过)
        chroma_stats:   [v6] check_chroma_corruption 返回的 dict (None=跳过/失败)
        chroma_err:     [v6] 检查 4 错误描述 (None=通过)
        elapsed:        总耗时 (秒)
        pass_all:       全部 4 项检查通过
        fails:          失败项列表 ['frames/packets', 'nal', 'pts', 'chroma']
    """
    gpu_sem = _GPU_SEMAPHORE
    gpu_task = hwaccel in ('cuda', 'auto')
    # [v6.1] chroma 检查是否走 NVDEC（仅 --chroma-hwaccel 且总 hwaccel 开启时生效）
    chroma_gpu = chroma_hwaccel and gpu_task
    if gpu_task and gpu_sem is not None:
        gpu_sem.acquire()

    t0 = time.monotonic()
    result = {
        'path': video_path,
        'label': Path(video_path).name,
        'codec': None,
        'frames': None,
        'packets': None,
        'fp_match': None,
        'fp_err': None,
        'nal_stats': None,
        'nal_err': None,
        'pts_issues': None,
        'pts_err': None,
        'chroma_stats': None,   # [v6] 检查 4: 色度平面异常
        'chroma_err': None,
        'chroma_skipped': False,  # [v6] --skip-chroma 跳过检查 4
        'fails': [],
        'pass_all': False,
    }

    try:
        # ── 1+3) v4 合并检查: 一次解码产出 frames + pts_issues ──
        frames, packets, issues, fp_err = check_decode_integrity(video_path, hwaccel)
        result['frames'] = frames
        result['packets'] = packets
        if fp_err:
            result['fp_err'] = fp_err
            result['fails'].append('frames/packets')
        elif frames is not None and packets is not None:
            result['fp_match'] = (frames == packets)
            if not result['fp_match']:
                result['fp_err'] = 'frames(%d) != packets(%d)' % (frames, packets)
                result['fails'].append('frames/packets')

        result['pts_issues'] = issues
        if issues is None:
            result['pts_err'] = 'ffmpeg 不可用（无法执行 pts 检查）'
            result['fails'].append('pts')
        elif len(issues) > 0:
            result['pts_err'] = 'ffmpeg verbose 检出 %d 条异常' % len(issues)
            result['fails'].append('pts')

        # ── 2) 码流统计（按编码分支）──
        # [v5] 先探测编码，决定专项分析走 H.264 NAL / HEVC NAL / mpeg4 VOP；
        #      非支持编码 (vp9/av1 等) 显示 N/A，不判失败。
        codec = probe_video_codec(video_path)
        result['codec'] = codec
        try:
            if codec in _MPEG4_FAMILY:
                # mpeg4 家族: VOP 深度分析（对应 H.264 三项检查）
                es, ok = extract_annexb_es(video_path, codec)
                if ok and es:
                    vops, time_bits = parse_mpeg4_es(es)
                    stats = check_vop_stats(vops, time_bits, dump_nal)
                    result['nal_stats'] = stats
                    # VOP 全解析失败（0 个）→ N/A，不判失败
                    if stats['vop_count_total'] == 0:
                        result['nal_err'] = None
                    else:
                        # 连 I-VOP 检查（对应 H.264 连 IDR）:
                        # 最长连续 I-VOP 块 > 3 才是异常（段首 1-2 个 I 预热帧正常）
                        if stats['max_consec_i_vop'] > 3:
                            result['nal_err'] = ('连续 I-VOP 块最长 %d 个（大量重编码为 I，'
                                                 '修复工具特征)') % stats['max_consec_i_vop']
                            result['fails'].append('nal')
                        # vop_time_increment 回退检查（时间戳逆序）
                        vr = stats['vop_time_regress']
                        if vr is not None and vr > 0:
                            msg = 'vop_time 回退 %d 次（时间戳逆序特征）' % vr
                            if result['nal_err']:
                                result['nal_err'] += '; ' + msg
                            else:
                                result['nal_err'] = msg
                            if 'nal' not in result['fails']:
                                result['fails'].append('nal')
                else:
                    result['nal_err'] = None  # 提取失败/空 → N/A 不判失败
            elif codec == 'h264':
                # H.264: NAL 专项分析（原 v2 逻辑不变）
                es, ok = extract_annexb_es(video_path, codec)
                if not (ok and es):
                    result['nal_err'] = 'H.264 ES 提取为空'
                    result['fails'].append('nal')
                else:
                    nals, frame_num_bits = parse_h264_es(es)
                    stats = check_nal_stats(nals, frame_num_bits, dump_nal)
                    result['nal_stats'] = stats
                    # [v6] 阈值 >0 → >=3: 良性段首双 IDR（SPS/PPS 冗余重注入
                    # 恢复点）不再误报，只抓修复前 per-slot IDR 异常 (6-16+ 个)。
                    if stats['idr_within_32_after_first'] >= _CONSEC_IDR_THRESHOLD:
                        result['nal_err'] = ('段首出现连续 %d 个 IDR 帧（修复前 '\
                                             'per-slot IDR 特征)') % stats['idr_within_32_after_first']
                        result['fails'].append('nal')
                    if stats['frame_num_regress'] > 0:
                        msg = 'frame_num 回退 %d 次（LA 重路由错位特征）' % stats['frame_num_regress']
                        if result['nal_err']:
                            result['nal_err'] += '; ' + msg
                        else:
                            result['nal_err'] = msg
                        if 'nal' not in result['fails']:
                            result['fails'].append('nal')
            elif codec == 'hevc':
                # H.265: 基础 NAL/IDR 分析（无 frame_num 概念，与 v2 一致）
                es, ok = extract_annexb_es(video_path, codec)
                if ok and es:
                    nals = parse_hevc_es(es)
                    stats = check_hevc_stats(nals, dump_nal)
                    result['nal_stats'] = stats
                    # [v6] 与 h264 分支同阈值策略（良性双 IDR 不误报）
                    if stats['idr_within_32_after_first'] >= _CONSEC_IDR_THRESHOLD:
                        result['nal_err'] = ('段首出现连续 %d 个 IDR 帧（修复前 '\
                                             'per-slot IDR 特征)') % stats['idr_within_32_after_first']
                        result['fails'].append('nal')
                else:
                    # hevc_mp4toannexb bsf 不可用 → 无专项分析，显示 N/A 不判失败
                    result['nal_stats'] = None
                    result['nal_err'] = None
            else:
                # 非支持编码 (vp9/av1 等): N/A，不判失败
                result['nal_stats'] = None
                result['nal_err'] = None
        except Exception as e:
            result['nal_err'] = '码流解析失败: %s' % e
            result['fails'].append('nal')

    finally:
        if gpu_task and gpu_sem is not None:
            gpu_sem.release()

    # ── 4) [v6] 色度平面异常检查（像素域，与结构检查正交）──
    # [v6.1][FIX-CHROMA-GPU-GATE] 默认纯 CPU 软解 + NumPy，不占用 NVDEC 会话，
    # 置于 GPU 闸门外执行，批处理下与其它文件的 GPU 检查并行；
    # --chroma-hwaccel 时走 NVDEC（chroma_gpu），需重新进入闸门。
    # --skip-chroma 时跳过该步骤（默认包含该检测）。
    if skip_chroma:
        result['chroma_skipped'] = True
    else:
        if chroma_gpu and gpu_sem is not None:
            gpu_sem.acquire()
        try:
            chroma_stats = check_chroma_corruption(video_path, hwaccel=chroma_gpu)
            result['chroma_stats'] = chroma_stats
            if chroma_stats is not None and chroma_stats['bad_count'] >= _CHROMA_BAD_FRAME_MIN:
                result['chroma_err'] = ('色度平面异常 %d 簇（周期性色度污染/花屏特征，'\
                                        '坏帧索引 %s)') % (chroma_stats['bad_count'], chroma_stats['bad_frames'])
                result['fails'].append('chroma')
        except Exception:
            result['chroma_stats'] = None
        finally:
            if chroma_gpu and gpu_sem is not None:
                gpu_sem.release()

    result['pass_all'] = len(result['fails']) == 0
    result['elapsed'] = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════
# 6. 并行执行引擎
# ═══════════════════════════════════════════════

def run_verify_parallel(video_files, hwaccel='auto', dump_nal=None,
                         workers=1, parallel_mode='thread',
                         gpu_workers=0, skip_chroma=False,
                         chroma_hwaccel=False) -> list:
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
            'skip_chroma': skip_chroma,
            'chroma_hwaccel': chroma_hwaccel,
        })

    results_map = {}  # path -> result
    workers = max(1, min(workers, total))
    t_batch = time.time()

    if workers == 1:
        # 串行模式（保持原有进度打印风格）
        for i, t in enumerate(tasks, 1):
            print(f"\n── [{i}/{total}] {t['label']} ──")
            res = _verify_one_video(t['path'], hwaccel=t['hwaccel'],
                                     dump_nal=t['dump_nal'],
                                     skip_chroma=t['skip_chroma'],
                                     chroma_hwaccel=t['chroma_hwaccel'])
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
                                   t['hwaccel'], t['dump_nal'],
                                   t['skip_chroma'], t['chroma_hwaccel'])] = t
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
                        'codec': None,
                        'frames': None, 'packets': None, 'fp_match': None,
                        'fp_err': '并行执行异常: %s' % e,
                        'nal_stats': None, 'nal_err': None,
                        'pts_issues': None, 'pts_err': None,
                        'chroma_stats': None, 'chroma_err': None,
                        'chroma_skipped': False,
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
    codec_tag = ' (%s)' % res.get('codec') if res.get('codec') else ''
    if res['fp_match']:
        ch1 = ' [1] frames=packets=%d OK' % res['frames']
    elif res['fp_err']:
        ch1 = ' [1] %s' % res['fp_err'][:80]
    else:
        ch1 = ' [1] frames=%s packets=%s' % (res['frames'], res['packets'])

    s = res.get('nal_stats')
    if s and res.get('codec') in _MPEG4_FAMILY:
        # [v5] mpeg4: VOP 统计展示（含总数供与 frames/packets 交叉核对）
        vr = s['vop_time_regress']
        ch2 = ' [2] VOP=%d I-VOP=%d 连I块=%d 时间回退=%s' % (
            s['vop_count_total'], s['i_vop_count'],
            s['max_consec_i_vop'],
            vr if vr is not None else 'N/A')
    elif s:
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

    cs = res.get('chroma_stats')
    if res.get('chroma_skipped'):
        ch4 = ' [4] 跳过'
    elif cs is not None:
        ch4 = ' [4] 色度坏帧簇=%d (U中位=%.1f/V中位=%.1f)' % (
            cs['bad_count'], cs['median_u'], cs['median_v'])
    elif res.get('chroma_err'):
        ch4 = ' [4] %s' % res['chroma_err'][:60]
    else:
        ch4 = ' [4] N/A'

    status = 'PASS' if res['pass_all'] else 'FAIL'
    print(f"[{idx}/{total}] {status} {label}{codec_tag} ({elapsed})")
    print(f"      {ch1}{ch2}{ch3}{ch4}")


def _print_summary_table(results: list):
    """打印终末汇总表。"""
    n_total = len(results)
    n_pass = sum(1 for r in results if r['pass_all'])
    n_fail = n_total - n_pass

    # 按失败类型统计
    fp_fail = sum(1 for r in results if 'frames/packets' in r['fails'])
    nal_fail = sum(1 for r in results if 'nal' in r['fails'])
    pts_fail = sum(1 for r in results if 'pts' in r['fails'])
    chroma_fail = sum(1 for r in results if 'chroma' in r['fails'])

    print("\n" + "=" * 72)
    print("  验收汇总")
    print("=" * 72)
    print("  文件总数: %d  |  通过: %d  |  失败: %d" % (n_total, n_pass, n_fail))
    if fp_fail or nal_fail or pts_fail or chroma_fail:
        print("  失败明细:")
        if fp_fail:
            print("    检查1 [frames/packets 不匹配]: %d 文件" % fp_fail)
        if nal_fail:
            print("    检查2 [NAL/VOP 异常/连IDR/帧号回退]: %d 文件" % nal_fail)
        if pts_fail:
            print("    检查3 [pts_anomaly/解码错误]: %d 文件" % pts_fail)
        if chroma_fail:
            print("    检查4 [色度平面异常/花屏特征]: %d 文件" % chroma_fail)

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
                if r.get('chroma_err'):
                    print("        CHROMA: %s" % r['chroma_err'])

    print("=" * 72)


# ═══════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='插帧段码流完整性验收 v5（多编码: H.264 NAL / mpeg4 VOP / 其他 N/A）',
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

v4 变更说明：
  check_decode_integrity() 合并原 check_frames_packets + check_pts_anomaly
  为单次 ffmpeg -v verbose -f null 解码，同时产出 frames / packets / pts_issues。
  原版 CPU 软解需两次全量解码（共三层检查）；v4 减为一次解码 + 一次 O(1) 包头读。
  检测灵敏度微量变化: frames 从 ffprobe nb_read_frames 变为 ffmpeg decoder output
  frame=N，两者在正常码流下相等，损坏码流下仍能被 frames ≠ packets 检出。

v5 变更说明：
  多编码探测（--hwaccel 等参数语义不变）:
    - probe_video_codec() 用 ffprobe 读 v:0 codec_name，识别 h264 / mpeg4 家族
      (mpeg4/msmpeg4v2/msmpeg4v3) / hevc / 其他编码。
    - 检查 2 按编码分支: h264 → NAL 分析；mpeg4 家族 → VOP 深度分析
      (I-VOP 计数/段首连 I-VOP/vop_time 时间回退)；其他编码 → N/A 不判失败。
    - 非 h264 的 frames/packets 守恒仍严格判定；mpeg4 额外输出 VOP 总数
      供交叉核对（区分容器假象与真丢帧）。
    - --dump-nal 对 mpeg4 导出 (vop_idx, vop_type, time_increment) 列表。

v6 变更说明：
  检查 4 check_chroma_corruption(): 解码 yuv420p 逐帧计算 U/V std 识别色度花屏。
  新增 --skip-chroma 跳过检查 4（默认包含该检测）。

示例：
  python tests/verify_segment_bitstream_v3.py a.mp4
  python tests/verify_segment_bitstream_v3.py ./segments --hwaccel cuda --gpu-workers 4
  python tests/verify_segment_bitstream_v3.py ./segments --hwaccel cuda --chroma-hwaccel
  python tests/verify_segment_bitstream_v3.py a.avi ./dir1/ b.mp4 --workers 8 --hwaccel auto
''',
    )
    ap.add_argument('paths', nargs='+',
                    help='待验收的视频文件或文件夹（支持混合传入）')
    ap.add_argument('--dump-nal', default=None,
                    help='导出 NAL (nal_type,frame_num) 或 VOP (vop_type,time) 到文件'
                         '（仅单文件模式有效；多文件时会被忽略以免覆盖）')
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
    ap.add_argument('--skip-chroma', action='store_true',
                    help='跳过检查4「色度坏帧簇」检测（默认包含该检测）')
    ap.add_argument('--chroma-hwaccel', action='store_true',
                    help='检查4 色度检测用 NVDEC 解码加速（需配合 --hwaccel cuda/auto 生效；'
                         '默认关闭：NVDEC 错误隐藏与软解不一致，可能弱化花屏检测灵敏度）')
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

        res = _verify_one_video(str(fpath), hwaccel=hwaccel_mode, dump_nal=dump_nal,
                                skip_chroma=args.skip_chroma,
                                chroma_hwaccel=args.chroma_hwaccel)
        res['label'] = label

        # [v5] 编码标识
        print('[codec] %s' % (res.get('codec') or 'N/A'))

        # 详细输出（兼容原来风格）
        if res['fp_err']:
            print('[1] frames/packets 检查失败: %s' % res['fp_err'])
        else:
            match_str = 'OK' if res['fp_match'] else 'MISMATCH'
            print('[1] frames=%d packets=%d %s' % (res['frames'] or 0, res['packets'] or 0, match_str))

        s = res.get('nal_stats')
        if s and res.get('codec') in _MPEG4_FAMILY:
            # [v5] mpeg4: VOP 统计 + VOP 总数与 frames/packets 交叉核对
            vr = s['vop_time_regress']
            print('[2] VOP总数=%d I-VOP=%d 首个I-VOP@%s 连I块=%d 时间回退=%s' % (
                s['vop_count_total'], s['i_vop_count'],
                s['first_i_vop_at'] if s['first_i_vop_at'] is not None else -1,
                s['max_consec_i_vop'],
                vr if vr is not None else 'N/A'))
            if s['vop_count_total'] > 0 and res['frames'] is not None:
                print('  (交叉核对: VOP总数=%d, frames=%s, packets=%s)' % (
                    s['vop_count_total'], res['frames'], res['packets']))
        elif s:
            print('[2] IDR=%d 首个IDR@%d 其后32NAL内IDR=%d frame_num回退=%d' % (
                s['idr_count'], s['first_idr_at'] if s['first_idr_at'] is not None else -1,
                s['idr_within_32_after_first'], s['frame_num_regress']))
        if res['nal_err']:
            print('[2] NAL/VOP 异常: %s' % res['nal_err'])

        if res['pts_issues'] is not None and len(res['pts_issues']) == 0:
            print('[3] 无 pts_anomaly / 解码错误 OK')
        elif res['pts_err']:
            print('[3] %s' % res['pts_err'])
            if res['pts_issues']:
                for line in res['pts_issues'][:5]:
                    print('     ' + line)

        # [v6] 检查 4: 色度平面异常详细输出
        cs = res.get('chroma_stats')
        if res.get('chroma_skipped'):
            print('[4] 色度检查已跳过 (--skip-chroma)')
        elif cs is not None:
            print('[4] 帧数=%d U中位=%.2f V中位=%.2f 坏帧簇=%d %s' % (
                cs['frame_count'], cs['median_u'], cs['median_v'],
                cs['bad_count'],
                ('(索引: %s)' % cs['bad_frames']) if cs['bad_count'] else 'OK'))
        elif res.get('chroma_err'):
            print('[4] 色度检查异常: %s' % res['chroma_err'])
        else:
            print('[4] 色度检查 N/A')

        total = time.monotonic() - t_start
        if res['pass_all']:
            print('\n✅ 验收通过 (总用时 %s): 帧数守恒 / 段首无连IDR / 帧号单调 / 无 pts_anomaly / 色度正常' % _fmt_elapsed(total))
        else:
            print('\n❌ 验收未通过 (总用时 %s):' % _fmt_elapsed(total))
            if res['fp_err']:
                print('  - %s' % res['fp_err'])
            if res['nal_err']:
                print('  - %s' % res['nal_err'])
            if res['pts_err']:
                print('  - %s' % res['pts_err'])
            if res.get('chroma_err'):
                print('  - %s' % res['chroma_err'])
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
                                       gpu_workers=gpu_workers,
                                       skip_chroma=args.skip_chroma,
                                       chroma_hwaccel=args.chroma_hwaccel)
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
