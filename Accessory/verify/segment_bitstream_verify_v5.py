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
  - 移植 video_pipeline_analyzer_v3.py 的 NVDEC hwaccel 初始化失败过滤
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

v4 新增 (检查间并行，本文件；由 v3 复制而来，v3 保持不变):
  - _verify_one_video() 将检查 2 (ES 提取 + NAL/VOP/HEVC 解析) 抽为独立
    线程 _check2_worker，与检查 1+3 (ffmpeg -f null 全量解码，单文件大视频
    的主瓶颈) 并行重叠执行，缩短单文件墙钟时间。检查 2 的 ES 提取为
    -c copy + bitstream filter，不占 NVDEC 解码会话，故移出 GPU 并发闸门，
    进一步降低 GPU 任务串行化。
  - 分析结论（详见 memory/verify-bitstream-large-file-parallel.md）:
    检查 1 (frames==packets 守恒) 与检查 3 (pts 连续性) 为全局/跨段性质，
    按时间或字节分片会破坏语义（段切点即新 GOP/IDR，-reset_timestamps 使
    pts 归零），故对主瓶颈「分片并行」不可行；检查 4 (chroma 像素域) 是
    embarrassingly parallel 的局部性质，为唯一安全分片项，但日志场景均
    --skip-chroma，收益有限。v4 仅落地「检查间并行」这一零语义风险优化，
    chroma 分片与多机分布式批处理留作进阶方案（见上述 memory 文档）。

v5 新增 (资源自适应 + 检查 1+3 策略化/分块并行，本文件原地升级):
  - 资源自动探测最大化: 复用容器感知的 CPU/RAM 探测，新增显存探测
    (detect_vram_gb)，输出完整资源计划 (CPU/RAM/VRAM/workers/gpu-workers/
    decode-chunks/chroma-shards)。
  - 检查 1+3 策略化解码 (--decode-strategy):
      * auto: 统一走 framecrc。v5 原设「GPU→gpu_dual / CPU→showinfo」，
        实测两者均劣于单次 framecrc，v10 已改（详见 v10 说明）。
      * showinfo: 原 v4 单次 -vf showinfo 解码（保留作兼容/对照）。
      * gpu_dual: 显式 GPU 双进程路径（不可用自动回退 showinfo）。
      * framecrc: 单次 -f framecrc 解码（stdout 逐帧含 pts，无 filtergraph），
        供 A/B 实测对比。
  - 检查 1+3 分块并行解码 (--decode-chunks auto/N，仅 CPU 软解路径生效):
      ffprobe -show_packets（不解码）构造「关键帧对齐分块计划」，各分块
      ffmpeg -copyts -ss <关键帧pts> -i in -frames:v <块帧数> 并行解码，
      按 pts 修剪 B 帧重排边界泄漏，段内 pts 连续性 + 段间缝合检查；
      Σ帧数 != 容器帧数 或任一分块失败 → 自动回退整文件单次解码（正确性兜底）。
      实测: 120 帧 GOP=30 样本 3 块解码 Σ=120、边界 pts 间隔 0.04s 精确。
  - 检查 4 色度分块并行 (--chroma-shards auto/N，仅 CPU 软解): 复用同一
    分块计划，各分块 rawvideo 解码 + NumPy std 归约，坏帧索引偏移聚合；
    任一分块失败自动回退单流。
  - 检查 2 ES 提取内存有界化: extract_annexb_es 由 capture_output=True 全量
    驻留改为流式写临时文件 + mmap 只读映射（页缓存背衬，峰值堆内存 O(1)）。
  - packets/decode-errors 来源: 从同一次解码 stderr 汇总行解析
    ("N packets read; N frames decoded; N decode errors")，ffprobe
    -count_packets 降级为 fallback（该命令本身是一次全文件 demux 扫描，
    并非 O(1) 读包头）。
  - 新增 --timing: 逐检查分项耗时输出，为后续优化提供证据基线。

v7 新增 (2026-08-20，卡死防御，FIX-PIPE-DEADLOCK):
  - 修复 v4 引入的 ES 提取管道死锁: extract_annexb_es 原「先排 stdout 再读
    stderr」在 ffmpeg stderr 超过管道缓冲（Linux 64KB）时互相阻塞，copyfileobj
    永久卡死（多主机/多输入复现，主线程卡在 nal_thread.join()，GPU 归零后
    静默）。现改为双 daemon 线程并发排空两条管道 + wait(timeout)，超时 kill。
  - 全部子进程/线程等待加兜底超时（V4_TIMEOUT_META/DECODE/EXTRACT/GPU_WAIT
    环境变量可调），停滞不再无限阻塞，而是 kill 并报错。
  - 单文件模式逐阶段输出 [v7] 进度（probe/decode/check2/chroma），静默期消失。
  - 看门狗线程: 阶段停滞超过 V4_WATCHDOG_STALL（默认 600s）时 dump 全部线程
    堆栈到 stderr 与 ./verify_segment_bitstream_v4_stuck.log，便于定位残留卡点。
  - 修复色度 rawvideo 分块读取丢弃帧尾余量导致的 U/V 平面错位（跨块累积余量）。

v8 新增 (2026-08-20，色度多点 ROI 探测，局部静态画面精度提升):
  - check_chroma_corruption 由「整平面 U/V std」升级为「4 角 + 中心 + 全局」
    共 6 个 ROI 多点探测 (_build_roi_list + _compute_roi_uv_std): 角块边长 =
    chroma 平面 1/4（≈ luma ¼×¼，锚定帧四角），中心块 = chroma 1/2 居中
    （≈ luma ½×½），参数见 _CHROMA_ROI_CFG（corner_div/center_div/
    min_block_px，角块样本 <1500 时降级为仅中心+全局并告警一次）。
  - 判据升级为双通道 (_chroma_postprocess):
      1) 绝对水平: 逐 ROI 独立自校准 std > max(median_roi*1.6, floor)，
         局部 ROI 下限降为 6（防小区域信号被整平面稀释淹没），全局 ROI
         下限保持 12（兼容 v6 旧语义）;
      2) 时间域跳变: 帧间差分 > max(median_diff*8, 1.5)——对画面局部
         长时间无变化的动画，合法静态帧差分≈0，污染帧出现尖峰，阈值只
         校准像素噪声不受画面内容影响;
      3) 剪辑帧豁免: 单帧 >70% ROI 同时跳变且下一跳变未持续 → 视为场景
         切变不计坏帧（持续跳变风暴仍正常检出，不会把整帧污染误豁免）。
  - 返回新增 n_rois / roi_names / cluster_regions（逐簇命中区域，如 tl/U）。
  - 计算量 ≈1.5× 原整平面 std（角块合计 1/4 + 中心 1/4 + 全局 1，每平面），
    仍为逐 chunk 向量化批量归约；分块并行 (_chroma_shard_worker) 同步升级。

v9 新增 (2026-08-22，资源感知自动分块最大化):
  - _auto_parallel_counts 改按「实际活跃文件数」n_active_files =
    min(文件总数, workers) 均分 CPU/RAM 预算。旧版按批处理 workers 均分，
    单文件验收（最常见场景）误得 per_file_cpu = cpu//workers = 1，
    decode-chunks / chroma-shards 恒为 1，多核全部空转。
    现单文件时全资源归该文件: 8 核机 chroma-shards=8、CPU 机
    decode-chunks=min(cpu, RAM 上限, 8)；GPU 主机 decode-chunks 维持 1
    （检查 1+3 走 gpu_dual/NVDEC 单次解码），chroma 默认纯 CPU 软解故
    shards 在 GPU 主机同样生效。
  - chroma 分片线程钳位 (_chroma_shard_worker 新增 threads 参数，
    = max(1, cpu//shards)): 旧版每个分片 ffmpeg 默认开满自动解码线程，
    N 分片互相争抢导致线性扩展失效；现 N 进程合计 ≈ 占满可用核数。
  - 单流路径 reader/compute 双缓冲 (_stream_uv_std): 管道读与 NumPy
    ROI std 归约拆为两线程经有界队列(maxsize=2)衔接——旧实现同线程串行，
    归约期间 ffmpeg 阻塞在管道写、解码器空转；现两者真重叠（NumPy 释放
    GIL），总耗时 ≈ max(解码+传输, 归约) 而非两者之和。分片失败回退与
    --chroma-hwaccel 路径同样受益。进度输出/看门狗心跳/超时 kill 语义不变。
  - 数值语义零变化: 仅调度层并行化，像素判定算法不动；分片/双缓冲路径
    与单流结果逐位一致（同一两遍 std，仅批量化）。

v10 新增 (2026-09-14，GPU DEC 利用率与验收解码路径修复):
  - [FIX-GPU-DUAL-FFPROBE] 修复 _decode_single_gpu_dual 恒返回 None 的两个
    独立缺陷，二者任一命中都会让 `--decode-strategy auto` 在 GPU 上静默
    退化到 showinfo（实测单会话 NVDEC 20.7% / 11.9s）:
      1) ffprobe 不接受 `-hwaccel`（option 表只有 `-hwaccel_flags`），实测报
         "Failed to set value 'cuda' for option 'hwaccel': Option not found"
         且 rc=1；而 -show_frames 只读元数据、不下载像素，本不需要 hwaccel。
      2) `-of csv=p=0` 会省略行首 section 名（输出裸值 "0.000000"），与解析器
         的 `frame,` 前缀过滤冲突 → 恒 0 行；改回 `-of csv`，并把取值从
         split(',',1)[-1] 改为字段 1（首帧带 SEI side data 追加列，末字段
         不是 pts_time）。
  - [PERF-DECODE-STRATEGY] `--decode-strategy auto` 改为统一走 framecrc。
    实测（墙钟 | NVDEC 均值）:
      4K30s : showinfo 11.9s(20.7%) | gpu_dual 11.6s(21.6%) | framecrc 5.9s(35.6%)
      1080p : showinfo  3.4s(16.9%) | gpu_dual  4.5s(16.9%) | framecrc 1.9s(32.0%)
      SD    : showinfo  0.9s(14.5%) | gpu_dual  1.1s(13.8%) | framecrc 0.8s(19.2%)
      CPU 4K: showinfo 12.5s        | —                     | framecrc 4.1s
    framecrc 在三种分辨率与 GPU/CPU 上全面最优。showinfo 的开销大头是逐帧
    文本写 stderr；gpu_dual 是「-f null NVDEC」+「ffprobe -show_frames」
    （后者自身要软件全解码）双进程重复解码，墙钟恒等于较慢一侧。
    语义等价性: framecrc 行 (stream,dts,pts,duration,size,crc) 的 pts 即解码器
    输出显示时间戳；逐帧 pts_time 实测与 showinfo 在 4K/1080p 完全相同，带
    非零 start_time 的 VFR 源存在恒定 offset（实测 0.0130135s），而
    _analyze_pts_sequence 的全部判据基于相邻帧差值，offset 自动抵消。
    验收回归（24 个样本: baseline_pre/post 16 个 + 截断/字节翻转/HEVC 段 +
    人工 pts 跳变 2 个）: 两侧 verdict、frames、packets、issue 条数、异常帧号、
    est_missing 全部一致，0 处不一致。唯一差异是 issue 文本里的原始 pts 数值
    单位不同（showinfo 以 1001/帧 为单位 → pts=348300，framecrc 以 tb 为单位
    → pts=348），对同一跳变给出相同的 diff/interval 比值与 est_missing=48。
  - [FIX-FRAMECRC-HW-RETRY] framecrc 路径补齐 NVDEC 初始化失败 → CPU 软解
    重跑（与 showinfo 路径对齐），避免 hwaccel 告警污染 issues 判据。

用法:
  单文件: python Accessory/verify/segment_bitstream_verify_v3.py <video.mp4> [--dump-nal temp/nal.txt]
  多文件: python Accessory/verify/segment_bitstream_verify_v3.py a.mp4 b.mp4 c.mp4 --workers 4
  文件夹: python Accessory/verify/segment_bitstream_verify_v3.py ./segments/ --hwaccel cuda --gpu-workers 2
  文件夹: python Accessory/verify/segment_bitstream_verify_v3.py ./segments/ --hwaccel cuda --chroma-hwaccel
  混合:   python Accessory/verify/segment_bitstream_verify_v3.py a.mp4 ./dir1/ b.mp4
"""
import argparse
import math
import mmap
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import numpy as np


def _fmt_elapsed(seconds):
    """秒 → 人类可读耗时字符串。"""
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1000)
    return '%.2f s' % seconds


def _fmt_hms(seconds):
    """秒 → mm:ss 或 h:mm:ss（进度条 [elapsed<eta] 用）。"""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    return '%02d:%02d' % (m, s)


def _fmt_chroma_progress(processed, total, t0):
    """chroma 进度行: 已累计 processed/total 帧 [pct% elapsed<eta]。"""
    if total and total > 0:
        pct = 100.0 * processed / total
    else:
        pct = 0.0
    elapsed = time.monotonic() - t0
    if total and total > 0 and processed > 0:
        eta = elapsed / processed * (total - processed)
    else:
        eta = 0.0
    if total and total > 0:
        return '[verify] chroma: 处理中，已累计 %d/%d 帧 [%.1f%% %s<%s]' % (
            processed, total, pct, _fmt_hms(elapsed), _fmt_hms(eta))
    return '[verify] chroma: 处理中，已累计 %d 帧' % processed

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


# ═══════════════════════════════════════════════════════════════════
# [v7] 卡死防御（FIX-PIPE-DEADLOCK 配套加固）
#   v4 将检查 2 抽为后台线程后，extract_annexb_es 改为 Popen 流式排水，
#   但「先排 stdout 再读 stderr」会在 ffmpeg stderr 超过管道缓冲（Linux
#   64KB）时互相阻塞 → copyfileobj 永久卡死（2026-08-20 多主机复现）。
#   v7 统一策略:
#     - 所有子进程 wait/read/join 增加兜底超时（超时 = 停滞检测，非性能
#       上限；默认值远大于正常耗时，超时后 kill 并报错，绝不无限阻塞）;
#     - stdout/stderr 一律并发排空（双 daemon 线程），消除管道死锁;
#     - 单文件模式逐阶段打印进度，配合看门狗在阶段停滞时 dump 全部线程
#       堆栈到 stderr 与 ./verify_segment_bitstream_v4_stuck.log。
#   超时值可用环境变量覆盖: V4_TIMEOUT_META / V4_TIMEOUT_DECODE /
#   V4_TIMEOUT_EXTRACT / V4_TIMEOUT_GPU_WAIT / V4_WATCHDOG_STALL。
# ═══════════════════════════════════════════════════════════════════
_SUB_TIMEOUT_META = int(os.environ.get('V4_TIMEOUT_META', '600'))
_SUB_TIMEOUT_DECODE = int(os.environ.get('V4_TIMEOUT_DECODE', '3600'))
_SUB_TIMEOUT_EXTRACT = int(os.environ.get('V4_TIMEOUT_EXTRACT', '1800'))
_GPU_WAIT_TIMEOUT = int(os.environ.get('V4_TIMEOUT_GPU_WAIT', '900'))
_WATCHDOG_STALL_S = int(os.environ.get('V4_WATCHDOG_STALL', '600'))

_WATCHDOG_STATE = {
    'last_progress': None,
    'last_dump': 0.0,
    'stop': threading.Event(),
    'thread': None,
}
_WATCHDOG_LOG = Path.cwd() / 'verify_segment_bitstream_v4_stuck.log'


def _watchdog_tick():
    """各阶段进度标记：记录最近一次进度时间（看门狗据此判断停滞）。"""
    _WATCHDOG_STATE['last_progress'] = time.monotonic()


def _watchdog_start():
    """启动看门狗线程（幂等）。阶段停滞超过 _WATCHDOG_STALL_S 时，
    dump 全部线程堆栈到 stderr 与日志文件（不退出进程，可重复诊断）。"""
    if _WATCHDOG_STATE['thread'] is not None:
        return

    def _body():
        interval = max(5, min(60, _WATCHDOG_STALL_S // 4))
        while not _WATCHDOG_STATE['stop'].is_set():
            time.sleep(interval)
            if _WATCHDOG_STATE['stop'].is_set():
                break
            now = time.monotonic()
            last_prog = _WATCHDOG_STATE['last_progress']
            if last_prog is None or now - last_prog < _WATCHDOG_STALL_S:
                continue
            if now - _WATCHDOG_STATE['last_dump'] < _WATCHDOG_STALL_S:
                continue
            _WATCHDOG_STATE['last_dump'] = now
            frames = sys._current_frames()
            lines = [
                '\n===== [v7 watchdog] 疑似卡死: 阶段停滞 > %ds (%s) ====='
                % (_WATCHDOG_STALL_S, time.strftime('%Y-%m-%d %H:%M:%S')),
            ]
            for tid, frm in sorted(frames.items()):
                lines.append('Thread %d (0x%x):' % (tid, tid))
                for fn, lineno, name, line in traceback.extract_stack(frm):
                    lines.append('  %s:%d in %s: %s' % (
                        fn, lineno, name, (line or '').strip()))
            msg = '\n'.join(lines) + '\n'
            sys.stderr.write(msg)
            sys.stderr.flush()
            try:
                with open(str(_WATCHDOG_LOG), 'a', encoding='utf-8') as f:
                    f.write(msg)
            except Exception:
                pass

    t = threading.Thread(target=_body, daemon=True, name='v4-watchdog')
    t.start()
    _WATCHDOG_STATE['thread'] = t
    _WATCHDOG_STATE['last_progress'] = time.monotonic()


def _run(cmd, timeout=_SUB_TIMEOUT_META):
    """subprocess.run 带兜底超时；超时时 kill 子进程并以 rc=-1 返回，
    绝不无限等待（此前 ffprobe -count_packets / -bsfs 等均为无界等待）。"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              encoding='utf-8', errors='replace',
                              timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ''
        err = e.stderr or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        if isinstance(err, bytes):
            err = err.decode('utf-8', 'replace')
        return subprocess.CompletedProcess(
            cmd, -1, stdout=out,
            stderr=(err + '\n[超时 kill] 命令停滞 >%ds' % timeout).strip())
    except Exception as e:
        return subprocess.CompletedProcess(cmd, -1, stdout='',
                                           stderr='命令异常: %s' % e)


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
# v4.1: 与 video_pipeline_analyzer_v3.py 保持一致，用于 issue 过滤与 GPU→CPU 回退判断。
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



# ── [v5] 策略化解码与分块并行（Phase 0/1 落地）────────────────────────────

_STATS_LINE_RE = re.compile(
    r'(\d+)\s+packets read\b.*?;\s*(\d+)\s+frames decoded;\s*(\d+)\s+decode errors')


def _parse_stats_line(line):
    """从 ffmpeg 汇总行解析 (packets, frames_decoded, decode_errors)。"""
    m = _STATS_LINE_RE.search(line or '')
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _classify_error_line(line):
    """将 stderr 行分类为真实解码/码流异常记录；正常统计/环境告警返回 None。"""
    low = (line or '').lower()
    if 'packets read' in low and 'frames decoded' in low:
        return None
    if any(kw in low for kw in ('avhwdevicecontext', 'instance creation failure')):
        return None
    if 'error' in low and any(kw in low for kw in ('opening output', 'filtergraph')):
        return None
    if 'non-existing' in low or ('error' in low and 'parsed_showinfo' not in low):
        return line.strip()[:200]
    return None


def _analyze_pts_sequence(pts_rows, max_issues=200):
    """对 [(n, pts, pts_time), ...] 序列做 pts 连续性分析（v4 算法原样抽取）。

    返回 (issues, total_issues)。dup / backward / drop 判据与 v4 完全一致。
    """
    issues = []
    total = 0
    pts_last = None
    pts_interval = None
    threshold = 2.0
    for _n, _pts, _pts_t in pts_rows:
        if len(issues) >= max_issues and total >= max_issues:
            break
        if pts_last is not None:
            diff = _pts - pts_last
            if diff == 0:
                rec = 'pts dup: frame=%d pts=%d（与上一帧相同）' % (_n, _pts)
                total += 1
                if len(issues) < max_issues:
                    issues.append(rec)
            elif diff < 0:
                rec = 'pts backward: frame=%d pts=%d（回退 %d，%d->%d）' % (_n, _pts, diff, pts_last, _pts)
                total += 1
                if len(issues) < max_issues:
                    issues.append(rec)
            else:
                if pts_interval is None:
                    pts_interval = diff
                elif diff > threshold * pts_interval:
                    est = max(1, int(round(diff / pts_interval)) - 1)
                    rec = 'pts drop: frame=%d pts=%d diff=%d interval≈%d est_missing=%d' % (
                        _n, _pts, diff, pts_interval, est)
                    total += 1
                    if len(issues) < max_issues:
                        issues.append(rec)
                else:
                    pts_interval = (pts_interval * 3 + diff) // 4
        pts_last = _pts
    if total > len(issues):
        issues.append('...（共 %d 条 pts/解码异常，仅显示前 %d 条）' % (total, len(issues)))
    return issues, total


def _parse_decode_stderr(lines, max_issues=200):
    """流式解析 ffmpeg verbose stderr（内存 O(行)），返回 dict:

      pts_rows       [(n, pts, pts_time), ...] showinfo 帧行
      issues         真实解码错误行（按首见去重，保持顺序）
      packets / frames_decoded / decode_errors  汇总行解析值（无则 None）
      frame_line     最后 'frame= N' 进度值
      hw_failed      NVDEC hwaccel 初始化失败特征是否出现
      tail_lines     最后 6 行原始文本（诊断用）
    """
    pts_rows = []
    issues = []
    seen = set()
    packets = frames_decoded = decode_errors = None
    frame_line = None
    hw_failed = False
    tail_lines = []
    re_pts1 = re.compile(r'n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([\d.]+)')
    re_pts2 = re.compile(r'n:\s*(\d+)\s+pts:\s*(-?\d+)')
    for raw in lines:
        line = (raw or '').rstrip('\r\n')
        if not line:
            continue
        tail_lines.append(line)
        if len(tail_lines) > 6:
            tail_lines.pop(0)
        low = line.lower()
        if _is_hwaccel_init_failure(low):
            hw_failed = True
            continue
        st = _parse_stats_line(line)
        if st[0] is not None:
            packets, frames_decoded, decode_errors = st
            continue
        m = re.search(r'frame=\s*(\d+)', line)
        if m:
            frame_line = int(m.group(1))
            continue
        err_rec = _classify_error_line(line)
        if err_rec:
            if err_rec not in seen:
                seen.add(err_rec)
                issues.append(err_rec)
            continue
        if 'parsed_showinfo' in low:
            m = re_pts1.search(line)
            if not m:
                m = re_pts2.search(line)
            if not m:
                continue
            n = int(m.group(1))
            pts = int(m.group(2))
            pts_t = float(m.group(3)) if m.lastindex >= 3 and m.group(3) else None
            pts_rows.append((n, pts, pts_t))
    return {
        'pts_rows': pts_rows,
        'issues': issues,
        'packets': packets,
        'frames_decoded': frames_decoded,
        'decode_errors': decode_errors,
        'frame_line': frame_line,
        'hw_failed': hw_failed,
        'tail_lines': tail_lines,
    }


def _parse_framecrc_rows(stdout_lines):
    """解析 `-f framecrc` stdout 为 [(n, pts, pts_time), ...]。

    行格式（ffmpeg 7.x）:
        #tb 0: 1001/24000            ← 流 0 的时间基
        0, <dts>, <pts>, <dur>, <size>, <crc>
    取字段 2 为 pts，乘以 `#tb` 时间基换算成秒；与 showinfo 的 pts_time 在
    逐帧差值上完全一致（见 v10 变更说明）。
    """
    tb = None
    rows = []
    for raw in (stdout_lines or []):
        raw = raw.rstrip('\r\n')
        if not raw:
            continue
        if raw.startswith('#tb'):
            m = re.search(r'(\d+)/(\d+)', raw)
            if m:
                tb = (int(m.group(1)), int(m.group(2)))
            continue
        if raw.startswith('#'):
            continue
        parts = raw.split(',')
        if len(parts) < 6:
            continue
        try:
            pts = int(parts[2].strip())
        except ValueError:
            continue
        rows.append((len(rows), pts, pts * tb[0] / tb[1] if tb else None))
    return rows


def _run_decode_stream(cmd, capture_stdout=False):
    """运行 ffmpeg 子进程并流式解析 stderr；返回 {**解析 dict, rc, err_tail}。

    [v7] 防卡死: stderr 由 daemon 线程持续排空，主线程 wait(timeout)；
    超时 kill 进程后以 rc=-1 + 错误文本返回，绝不无限阻塞。

    [P1-1] capture_stdout=True 时同时排空 stdout（`-f framecrc` 的逐帧 pts
    走 stdout），结果放在返回 dict 的 'stdout_lines'。两条管道各由独立
    daemon 线程排空，避免任一管道写满导致死锁（与 v7 的 stderr 处理同源）。
    """
    _empty = {'pts_rows': [], 'issues': [], 'packets': None,
              'frames_decoded': None, 'decode_errors': None, 'frame_line': None,
              'hw_failed': False, 'tail_lines': [], 'stdout_lines': []}
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False, bufsize=1, text=True, encoding='utf-8', errors='replace')
    except Exception as e:
        return dict(_empty, rc=-1, err_tail='启动失败: %s' % e)
    stderr_lines = []
    stdout_lines = []

    def _drain(pipe, sink, tick):
        try:
            for raw in pipe:
                sink.append(raw)
                if tick:
                    # [v7] 心跳: stderr 有输出即视为有进展，重置看门狗停滞计时
                    _watchdog_tick()
        except Exception:
            pass

    t_drain = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, True),
                               daemon=True)
    t_drain.start()
    t_out = None
    if capture_stdout:
        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines, False),
                                 daemon=True)
        t_out.start()
    try:
        rc = proc.wait(timeout=_SUB_TIMEOUT_DECODE)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            proc.kill()
            rc = proc.wait(timeout=15)
        except Exception:
            rc = -9
    t_drain.join(timeout=30)
    if t_out is not None:
        t_out.join(timeout=30)
    if timed_out:
        parsed = {'pts_rows': [], 'issues': [],
                  'packets': None, 'frames_decoded': None,
                  'decode_errors': None, 'frame_line': None,
                  'hw_failed': False, 'tail_lines': stderr_lines[-6:],
                  'stdout_lines': stdout_lines}
        return dict(parsed, rc=-1,
                    err_tail='\n'.join(stderr_lines[-6:])
                             + '\n[超时 kill] ffmpeg 解码停滞 >%ds'
                             % _SUB_TIMEOUT_DECODE)
    parsed = _parse_decode_stderr(stderr_lines)
    return dict(parsed, rc=rc, err_tail='\n'.join(parsed['tail_lines']),
                stdout_lines=stdout_lines)


def _get_packets_ffprobe(path):
    """ffprobe -count_packets 统计视频包数（全文件 demux 扫描，无解码）。

    返回 (packets, err)。
    """
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None, 'ffprobe 不可用（PATH 缺失）'
    try:
        r = _run([ffprobe, '-v', 'error', '-count_packets',
                  '-select_streams', 'v:0',
                  '-show_entries', 'stream=nb_read_packets',
                  '-of', 'csv=p=0', str(path)])
        m = re.match(r'(\d+)', r.stdout.strip())
        if m:
            return int(m.group(1)), None
        return None, 'ffprobe packets 解析失败: %r' % r.stdout[:200]
    except Exception as e:
        return None, 'ffprobe packets 异常: %s' % e


def _decode_single_showinfo(path, hwaccel='off'):
    """v4 语义单次解码: -vf showinfo -f null，一次产出 frames/pts/issues/packets。

    GPU 初始化失败自动重跑 CPU 软解（对齐 v4.1 回退哲学）。
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return None
    base = [ffmpeg, '-hide_banner', '-v', 'verbose']
    hw_flags = []
    if hwaccel == 'cuda':
        hw_flags = ['-hwaccel', 'cuda']
    elif hwaccel == 'auto':
        hw_flags = list(_HWACCEL_AUTO_FLAGS)
    tail = ['-i', str(path), '-an', '-fps_mode', 'passthrough',
            '-vf', 'showinfo', '-f', 'null', '-']
    cmd = base + hw_flags + ['-threads', str(_clamp_decode_threads())] + tail
    if hw_flags:
        res = _run_decode_stream(cmd)
        if res['hw_failed'] or (res['frames_decoded'] is None and not res['pts_rows']):
            if res['hw_failed']:
                print('[WARN] ffmpeg NVDEC hwaccel 初始化失败，已回退 CPU 软解: %s' % path)
            cmd = base + ['-threads', str(_clamp_decode_threads())] + tail
            res = _run_decode_stream(cmd)
    else:
        res = _run_decode_stream(cmd)
    return res


def _decode_single_framecrc(path, hwaccel='off'):
    """单次 -f framecrc 解码: stdout 逐帧含 pts，无 filtergraph（A/B 用）。"""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return None
    def _run_once(use_hw):
        cmd = [ffmpeg, '-hide_banner', '-v', 'verbose']
        if use_hw:
            cmd += ['-hwaccel', 'cuda']
        cmd += ['-threads', str(_clamp_decode_threads()),
                '-i', str(path), '-an', '-fps_mode', 'passthrough',
                '-f', 'framecrc', '-']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='replace',
                               timeout=_SUB_TIMEOUT_DECODE)
        except subprocess.TimeoutExpired:
            return {'pts_rows': [], 'issues': [], 'packets': None,
                    'frames_decoded': None, 'decode_errors': None,
                    'frame_line': None, 'hw_failed': False, 'tail_lines': [],
                    'rc': -1, 'err_tail': '[超时 kill] ffmpeg framecrc 解码停滞 >%ds'
                                         % _SUB_TIMEOUT_DECODE}
        except Exception as e:
            return {'pts_rows': [], 'issues': [], 'packets': None,
                    'frames_decoded': None, 'decode_errors': None,
                    'frame_line': None, 'hw_failed': False, 'tail_lines': [],
                    'rc': -1, 'err_tail': '启动失败: %s' % e}
        parsed = _parse_decode_stderr(r.stderr.splitlines())
        pts_rows = _parse_framecrc_rows((r.stdout or '').splitlines())
        return dict(parsed, pts_rows=pts_rows, rc=r.returncode,
                    err_tail=(r.stderr or '')[-500:])

    res = _run_once(hwaccel != 'off')
    # [FIX-FRAMECRC-HW-RETRY] 与 _decode_single_showinfo 对齐：NVDEC 初始化失败
    # （driver/NVDEC SDK 版本不匹配、容器 GPU 未透传等）时 ffmpeg 会静默软解，
    # 帧数仍有效但 stderr 含失败特征 → 丢弃该轮并重跑纯 CPU 软解，避免
    # hwaccel 告警污染 issues 判据。
    if hwaccel != 'off' and (res['hw_failed']
                             or (res['frames_decoded'] is None and not res['pts_rows'])):
        if res['hw_failed']:
            print('[WARN] ffmpeg NVDEC hwaccel 初始化失败，已回退 CPU 软解: %s' % path)
        res = _run_once(False)
    return res


def _decode_single_gpu_dual(path):
    """GPU 双进程并发: -f null（无 filter 不下载像素）取 frames/packets/errors
    + ffprobe -show_frames（帧元数据不下载像素）取逐帧 pts；任一失败返回 None。
    """
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        return None
    out = {}

    def _null_stats():
        cmd = [ffmpeg, '-hide_banner', '-v', 'verbose', '-hwaccel', 'cuda',
               '-threads', str(_clamp_decode_threads()),
               '-i', str(path), '-an', '-fps_mode', 'passthrough', '-f', 'null', '-']
        res = _run_decode_stream(cmd)
        if res['hw_failed'] or (res['frames_decoded'] is None and res['rc'] != 0):
            res['retry'] = True
        out['null'] = res

    def _show_frames():
        # [FIX-GPU-DUAL-FFPROBE] 两处修正，缺一即导致本路径恒返回 None →
        # `--decode-strategy auto` 在 GPU 上静默退化到 showinfo（实测单会话
        # NVDEC 26.5% / 11.7s vs framecrc 35.2% / 5.4s）:
        #   1) 不再传 `-hwaccel cuda`：ffprobe 的 option 表里并无 `hwaccel`
        #      （只有 `-hwaccel_flags`），实测报
        #      "Failed to set value 'cuda' for option 'hwaccel': Option not found"
        #      且 rc=1；而 -show_frames 只读容器/解码器元数据、不下载像素，
        #      本就不需要 hwaccel（实测 720 帧 pts_time 与 NVDEC 轮完全一致）。
        #   2) `-of csv=p=0` 改为 `-of csv`：p=0 会省略行首的 section 名，
        #      输出裸值 "0.000000"，而下方解析器按 `frame,` 前缀过滤 → 0 行。
        #      保留前缀后首行形如 "frame,0.000000,H.26[45] ... SEI"（带 side
        #      data 追加列），故取值必须用字段 1 而非 split(',',1)[-1]。
        cmd = [ffprobe, '-v', 'error',
               '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'frame=pts_time', '-of', 'csv', str(path)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='replace',
                               timeout=_SUB_TIMEOUT_DECODE)
        except subprocess.TimeoutExpired:
            out['frames_probe'] = {'pts_rows': [], 'rc': -1,
                                   'err_tail': '[超时 kill] ffprobe -show_frames 停滞 >%ds'
                                               % _SUB_TIMEOUT_DECODE}
            return
        except Exception as e:
            out['frames_probe'] = {'pts_rows': [], 'rc': -1, 'err_tail': str(e)}
            return
        pts_rows = []
        for raw in (r.stdout or '').splitlines():
            if not raw.startswith('frame,'):
                continue
            # 字段 1 = pts_time。首帧可能追加 side data 文本
            # （"frame,0.000000,H.26[45] User Data ..."），不能用末字段。
            parts = raw.split(',')
            try:
                pts_t = float(parts[1].strip())
            except (IndexError, ValueError):
                continue
            pts_rows.append((len(pts_rows), int(round(pts_t * 1000000)), pts_t))
        out['frames_probe'] = {'pts_rows': pts_rows, 'rc': r.returncode,
                               'err_tail': (r.stderr or '')[-300:]}

    t1 = threading.Thread(target=_null_stats)
    t2 = threading.Thread(target=_show_frames)
    t1.start()
    t2.start()
    t1.join(timeout=_SUB_TIMEOUT_DECODE + 120)
    t2.join(timeout=_SUB_TIMEOUT_DECODE + 120)
    if t1.is_alive() or t2.is_alive():
        print('[WARN] gpu_dual 双进程停滞超时，放弃 GPU 双进程路径（回退 showinfo）',
              flush=True)
        return None
    if 'null' not in out or 'frames_probe' not in out:
        return None
    null_res = out['null']
    fp_res = out['frames_probe']
    if null_res.get('retry'):
        return None
    if fp_res['rc'] != 0 or not fp_res['pts_rows']:
        return None
    frames = null_res['frames_decoded']
    if frames is None:
        frames = len(fp_res['pts_rows'])
    if frames != len(fp_res['pts_rows']):
        return None
    issues, _total = _analyze_pts_sequence(fp_res['pts_rows'])
    issues = issues + null_res['issues']
    return {
        'pts_rows': fp_res['pts_rows'],
        'issues': issues,
        'packets': null_res['packets'],
        'frames_decoded': frames,
        'decode_errors': null_res['decode_errors'],
        'frame_line': null_res['frame_line'],
        'rc': null_res['rc'],
        'err_tail': null_res['err_tail'] or fp_res['err_tail'],
        'hw_failed': null_res['hw_failed'],
        'strategy': 'gpu_dual',
    }


def _build_chunk_plan(path, n_chunks):
    """ffprobe -show_packets（不解码）构造关键帧对齐分块计划。

    返回 dict {total, chunks: [(start_pts, count), ...], duration}；失败 None。
    """
    ffprobe = shutil.which('ffprobe')
    if not ffprobe or n_chunks < 2:
        return None
    try:
        r = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=600)
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    pts_list = []
    key_idxs = []
    for raw in r.stdout.splitlines():
        raw = raw.strip()
        if not raw or raw.lower() in ('n/a', 'nan'):
            continue
        parts = raw.split(',')
        if len(parts) < 2:
            continue
        try:
            pts_t = float(parts[0])
        except ValueError:
            continue
        idx = len(pts_list)
        pts_list.append(pts_t)
        if 'K' in parts[1].upper():
            key_idxs.append(idx)
    total = len(pts_list)
    if total < 100 or not key_idxs:
        return None
    n = max(2, min(n_chunks, total))
    target = total / float(n)
    boundaries = [0]
    cur = 0
    while len(boundaries) < n:
        goal = cur + target
        best = None
        for ki in key_idxs:
            if ki <= cur:
                continue
            if best is None or abs(ki - goal) < abs(best - goal):
                best = ki
        if best is None:
            break
        boundaries.append(best)
        cur = best
        if cur >= total - 1:
            break
    boundaries = sorted(set(boundaries))
    # [FIX-HEVC-REORDER] 按「显示 pts 区间」计数而非包索引区间:
    # HEVC b-pyramid 重排使关键帧之后的包可能含上一 GOP 尾部 B 帧
    # （pts < 本块起点），索引区间计数会高估/低估实际可解码帧数
    # （实测尾部块 363 vs 360），导致分块校验失败回退。
    starts = [pts_list[s] for s in boundaries]
    chunks = []
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else None
        if end is not None and end <= start:
            continue
        count = 0
        for p in pts_list:
            if p >= start - 1e-6 and (end is None or p < end - 1e-6):
                count += 1
        if count <= 0:
            continue
        chunks.append((start, count))
    if len(chunks) < 2:
        return None
    if chunks[-1][1] < 10:
        prev = chunks[-2]
        chunks[-2] = (prev[0], prev[1] + chunks[-1][1])
        chunks.pop()
    return {'total': total, 'chunks': chunks,
            'duration': pts_list[-1] if pts_list else 0.0}


# 分块解码额外余量帧数: 线程级解码在 -frames:v 截止处存在非确定性
# （实测同一命令有时多 1 帧、有时漏 1 帧），余量保证窗口内帧必然解码，
# 再由上层按 pts 窗口过滤，杜绝边界误判。
_CHUNK_MARGIN = 8

# [P1-1] GPU 分块并发上限（NVDEC 会话数）= 默认分块数上限。
#
# 实测 Tesla T4 解 4K H.264「同一个文件分块并行」（关键帧对齐切分）:
#   3 分钟素材 4318 帧:  chunks 1 → 33.75s / dec 均值 35.0%
#                        chunks 2 → 14.01s / dec 均值 86.3%   ← 2.41x，最优
#                        chunks 3 → 16.36s / dec 均值 73.3%
#   30 秒素材  720 帧:   chunks 1 →  5.70s / 34.7%
#                        chunks 2 →  3.64s / 52.3%
#                        chunks 3 →  3.63s / 52.1%
#   1080p 25s / SD 25s: chunks 2 亦为最优（1.37s / 0.83s）
# 结论: 2 路即达吞吐平台（长素材 dec 86%），再增加分块数无收益且在
# 关键帧切分不均时分块墙钟反升。故默认与上限均取 2。
#
# 对照：多文件并发（每文件各跑一路整文件解码）时 2 会话仅 64.5%、
# 3 会话 78.7%，与「同文件分块」不是同一场景，勿混用该数字。
_NVDEC_CHUNK_PARALLEL = 2


def _decode_chunk_worker(path, start_pts, count, threads, hwaccel=False):
    """解码单个分块（输入 -ss + -copyts + -frames:v + -f framecrc），
    返回原始 pts 行与错误行。

    [P1-1][PERF-CHUNK-FRAMECRC] 由 `-vf showinfo -f null` 改为 `-f framecrc`，
    并新增 hwaccel 参数以支持 GPU 分块：
      · 两者都产出逐帧 pts，但 showinfo 逐帧写大段文本到 stderr（4K 实测
        11.9s vs framecrc 5.9s），framecrc 每帧仅一行短文本到 stdout；
      · GPU 分块是本项的核心目的——单会话 NVDEC 在 4K 上只到 ~33% 利用率，
        实测 2 并发 64.5% / 3 并发 78.7%，吞吐在 3~4 会话饱和。
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return {'rc': -1, 'pts_rows': [], 'issues': [], 'err_tail': 'ffmpeg 不可用'}
    cmd = [ffmpeg, '-hide_banner', '-v', 'verbose']
    if hwaccel:
        cmd += ['-hwaccel', 'cuda']
    cmd += ['-threads', str(threads), '-copyts',
            '-ss', '%.6f' % start_pts, '-i', str(path),
            '-frames:v', str(count), '-an', '-fps_mode', 'passthrough',
            '-f', 'framecrc', '-']
    res = _run_decode_stream(cmd, capture_stdout=True)
    pts_rows = _parse_framecrc_rows(res.get('stdout_lines'))
    return {'rc': res['rc'], 'pts_rows': pts_rows,
            'issues': res['issues'], 'err_tail': res['err_tail'],
            'hw_failed': res.get('hw_failed', False)}


def _check_decode_chunked(path, plan, cpu_count=None, hwaccel=False,
                          max_parallel=None):
    """分块并行解码检查 1+3（CPU 软解 / NVDEC 均可）；任何校验失败返回 None 由上层回退。

    边界处理: B 帧重排会导致每个分块多解出 1~2 帧（pts 属下一分块），
    按计划帧数从头部保留并截断；Σ 帧数 != 容器帧数即判失败回退。

    [P1-1] 新增 hwaccel / max_parallel:
      · hwaccel=True 时各分块走 NVDEC（`_decode_chunk_worker(..., hwaccel=True)`），
        使单个大文件也能吃满多会话 NVDEC —— 原实现仅 CPU 路径可分块，
        GPU 路径恒单会话，实测 4K 上 NVDEC 利用率只有 ~33%。
      · max_parallel 限制同时在跑的分块数。GPU 上并非越多越好：实测 4K H.264
        单会话 33%、2 会话 64.5%、3 会话 78.7%、4 会话 80.3% 后吞吐饱和
        （8 会话 86% 但墙钟 13.5s，反而劣于 3 会话的 5.2s）。故 GPU 侧默认
        钳到 3；CPU 侧保持「分块数=并发数」（各分块按核数均分线程）。
    """
    chunks = plan['chunks']
    n = len(chunks)
    cpu = cpu_count or os.cpu_count() or 4
    if hwaccel:
        # NVDEC 单会话无法占满解码引擎，靠多会话并行提升利用率；
        # 线程数对 NVDEC 解码本身无并行意义，统一用钳位后的保守值。
        threads = _clamp_decode_threads(max(1, cpu // max(1, n)))
        pool = max(1, min(n, max_parallel or _NVDEC_CHUNK_PARALLEL))
    else:
        threads = _clamp_decode_threads(max(1, cpu // n))
        pool = n

    def _window_rows(res, i):
        rows = res['pts_rows']
        start = chunks[i][0]
        end = chunks[i + 1][0] if i + 1 < n else None
        expect = chunks[i][1]
        # [FIX-BOUNDARY-LEAK] 泄漏帧在分块输出中的位置不稳定（实测有时是第
        # N+1 帧、有时顶替第 N 帧），不能按数量修剪；改为按显示 pts 窗口
        # [start, end) 精确过滤。
        kept = [r for r in rows
                if r[2] is not None
                and r[2] >= start - 1e-6
                and (end is None or r[2] < end - 1e-6)]
        return kept, expect

    def _norm_error(e):
        # 归一化错误行: 去掉 ffmpeg 对象指针（同一错误在不同 chunk 输出不同地址）
        return re.sub(r'@\s*0x[0-9a-fA-F]+', '@', e)

    results = [None] * n
    with ThreadPoolExecutor(max_workers=pool) as ex:
        futs = [ex.submit(_decode_chunk_worker, str(path), s, c + _CHUNK_MARGIN,
                          threads, hwaccel)
                for s, c in chunks]
        for i, f in enumerate(futs):
            try:
                results[i] = f.result()
            except Exception as e:
                results[i] = {'rc': -1, 'pts_rows': [], 'issues': [],
                              'err_tail': str(e), 'hw_failed': False}

    # [FIX-DECODE-RACE] 线程化解码在边界/EOF 偶发少解 1 帧（实测 ~5%），
    # 失败块用单线程确定性重试（threads=1），重试仍失败才整体回退。
    # [P1-1] NVDEC 初始化失败同样重试（改走 CPU 软解），避免环境问题被当作
    # 码流缺陷；重试后仍失败才整体回退到整文件单次解码。
    for i, res in enumerate(results):
        if res['rc'] != 0:
            return None
        kept, expect = _window_rows(res, i)
        if len(kept) != expect or res.get('hw_failed'):
            retry = _decode_chunk_worker(str(path), chunks[i][0],
                                         chunks[i][1] + _CHUNK_MARGIN, 1,
                                         hwaccel=False)
            if retry['rc'] != 0:
                return None
            results[i] = retry
            kept, expect = _window_rows(retry, i)
            if len(kept) != expect:
                return None

    kept_all = []
    issues = []
    seen = set()
    total_frames = 0
    for i, res in enumerate(results):
        kept, expect = _window_rows(res, i)
        kept_all.append(kept)
        total_frames += len(kept)
        for e in res['issues']:
            norm = _norm_error(e)
            if norm not in seen:
                seen.add(norm)
                issues.append(e)
    if total_frames != plan['total']:
        return None
    for rows in kept_all:
        sub, _cnt = _analyze_pts_sequence(rows)
        issues.extend(sub)
    # 段间缝合: 下块首帧 pts_time 应紧邻上块末帧
    interval = None
    for rows in kept_all:
        for j in range(1, len(rows)):
            d = rows[j][2] - rows[j - 1][2]
            if d and d > 0:
                interval = d if interval is None else (interval * 3 + d) / 4.0
    for i in range(1, len(kept_all)):
        prev_last = kept_all[i - 1][-1][2]
        cur_first = kept_all[i][0][2]
        diff = cur_first - prev_last
        if diff <= 0:
            issues.append('pts boundary: chunk%d 首帧 pts_time=%.4f 不晚于 chunk%d 末帧 %.4f'
                          % (i + 1, cur_first, i, prev_last))
        elif interval is not None and diff > interval * 2.5:
            issues.append('pts boundary: chunk%d 首帧 pts_time=%.4f 与 chunk%d 末帧 %.4f 间隔 %.4f（估计 %.4f）'
                          % (i + 1, cur_first, i, prev_last, diff, interval))
    return {'frames': total_frames, 'issues': issues, 'n_chunks': n}


def check_decode_integrity(path, hwaccel='off', strategy='auto', chunks=1,
                           cpu_count=None, ram_avail_gb=None, batch_workers=1,
                           chunk_plan=None):
    """v5 策略化检查 1+3 入口，返回 dict:

      frames / packets / issues / decode_errors / err / strategy / chunked
      n_chunks / timing
    """
    t0 = time.monotonic()

    def _mk(frames, packets, issues, decode_errors, err, strategy, chunked, n_chunks=1):
        return {'frames': frames, 'packets': packets, 'issues': issues,
                'decode_errors': decode_errors, 'err': err, 'strategy': strategy,
                'chunked': chunked, 'n_chunks': n_chunks,
                'timing': {'decode': time.monotonic() - t0}}

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return _mk(None, None, None, None, 'ffmpeg 不可用（PATH 缺失）', 'none', False)

    # ── 分块路径（CPU 软解 / NVDEC 均可）────────────────────────────────
    # [P1-1] 原实现用 `hwaccel == 'off'` 把 GPU 排除在分块之外，导致「单个大
    # 文件」在 GPU 主机上永远只有 1 个 NVDEC 会话 —— 而单会话 4K H.264 实测
    # 天花板就是 ~33% 利用率（`-f null` 完全无消费端时同样如此），30% 并非
    # 可调参数问题而是单流结构上限。放开后由 _NVDEC_CHUNK_PARALLEL 把并发
    # 钳到 3（吞吐甜点），既不越过饱和点、也不让 8 会话把墙钟拖长。
    if chunks > 1 and strategy in ('auto', 'framecrc', 'showinfo'):
        plan = chunk_plan
        if plan is None:
            plan = _build_chunk_plan(path, chunks)
        if plan is not None:
            res = _check_decode_chunked(path, plan, cpu_count,
                                        hwaccel=(hwaccel != 'off'))
            if res is not None:
                packets, pkt_err = _get_packets_ffprobe(path)
                err = None
                if pkt_err:
                    err = pkt_err
                elif packets is not None and res['frames'] != packets:
                    err = 'frames(%d) != packets(%d)' % (res['frames'], packets)
                return _mk(res['frames'], packets, res['issues'], 0, err,
                           'chunked', True, res['n_chunks'])
            print('[WARN] 分块解码校验失败，回退整文件单次解码: %s' % path)

    # ── 策略化单次解码 ──
    # [PERF-DECODE-STRATEGY] auto 默认改走 framecrc（单次 `-f framecrc` 解码）。
    #
    # 实测（20~30s 素材，GPU=cuda / CPU=off，墙钟 | NVDEC 利用率均值）:
    #   4K30s : showinfo 11.9s(20.7%) | gpu_dual 11.6s(21.6%) | framecrc 5.9s(35.6%)
    #   1080p : showinfo  3.4s(16.9%) | gpu_dual  4.5s(16.9%) | framecrc 1.9s(32.0%)
    #   SD    : showinfo  0.9s(14.5%) | gpu_dual  1.1s(13.8%) | framecrc 0.8s(19.2%)
    #   CPU 4K: showinfo 12.5s        | (n/a)                | framecrc 4.1s
    # framecrc 在三种分辨率、GPU/CPU 上全面最优；showinfo 的大头是逐帧文本
    # 写 stderr，gpu_dual 则是「-f null NVDEC」+「ffprobe -show_frames」(后者
    # 自身要软件全解码) 双进程重复解码，墙钟恒等于较慢的软解那一侧。
    #
    # 语义等价性（为何可以替换 showinfo）:
    #   framecrc 行 "stream,dts,pts,duration,size,crc" 的 pts 即解码器输出显示
    #   时间戳。逐帧 pts_time 实测: 4K/1080p 与 showinfo 完全相同；带非零
    #   start_time 的 VFR 源存在一个恒定 offset（clipsd: 0.0130135s），而
    #   _analyze_pts_sequence 的全部判据（dup / backward / drop）都基于相邻
    #   帧差值，恒定 offset 自动抵消，异常检出结果逐项一致。
    used = strategy
    if strategy == 'auto':
        r = _decode_single_framecrc(path, hwaccel)
        used = 'framecrc'
    elif strategy == 'framecrc':
        r = _decode_single_framecrc(path, hwaccel)
    elif strategy == 'showinfo':
        r = _decode_single_showinfo(path, hwaccel)
    elif strategy == 'gpu_dual' and hwaccel != 'off':
        # 显式请求才走双进程（较慢，仅用于 A/B 对照）
        r = _decode_single_gpu_dual(path)
        if r is None:
            r = _decode_single_showinfo(path, hwaccel)
            used = 'showinfo'
    else:
        r = _decode_single_showinfo(path, hwaccel)
        if strategy == 'gpu_dual':
            used = 'showinfo'

    if r is None:
        return _mk(None, None, None, None, '解码启动失败', used, False)

    issues, _total = _analyze_pts_sequence(r['pts_rows'])
    for e in r.get('issues', []):
        if e not in issues:
            issues.append(e)
    dec_err = r.get('decode_errors') or 0
    if dec_err > 0:
        issues.append('ffmpeg 汇总 %d 条 decode errors' % dec_err)
    frames = r.get('frames_decoded')
    if frames is None:
        frames = r.get('frame_line')
    if frames is None:
        frames = len(r['pts_rows']) or None
    packets = r.get('packets')
    pkt_err = None
    if packets is None:
        packets, pkt_err = _get_packets_ffprobe(path)
    err = None
    if r.get('rc', 0) != 0:
        err = 'ffmpeg 解码失败（rc=%d）: %s' % (r.get('rc'), (r.get('err_tail') or '')[:200])
    elif pkt_err:
        err = pkt_err
    elif frames is None:
        err = 'ffmpeg 解码失败（无法提取帧数）'
    return _mk(frames, packets, issues, dec_err, err, used, False)




# ═══════════════════════════════════════════════
# 1. 核心检查函数
# ═══════════════════════════════════════════════

# [v5] 以下为 v4 原 check_decode_integrity 实现（保留作参考，未再被调用；
# v5 入口 check_decode_integrity 负责策略化/分块解码，并在此函数语义基础上
# 增加 packets/decode-errors 汇总行解析与 stderr 流式解析）。
def _check_decode_integrity_v4(path, hwaccel='off'):
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

# [v8] 多点 ROI 探测参数（4 角 + 中心 + 全局，chroma 平面坐标系）:
#   corner_div=4  → 角块边长 = chroma 平面 1/4（≈ luma ¼×¼ 画面，锚定帧四角）
#   center_div=2  → 中心块边长 = chroma 平面 1/2（≈ luma ½×½，居中对称）
#   min_block_px  → 角块最小 chroma 样本数（不足则降级为仅中心+全局，并告警一次）
#   abs_mult/abs_floor_roi/abs_floor_global: 绝对水平判据
#     std > max(median_roi * abs_mult, floor)，局部 ROI 下限 6（防小信号被全局稀释
#     淹没），全局 ROI 下限 12（兼容 v6 旧语义）。
#   diff_mult/diff_floor: 时间域跳变判据（对长时间无变化的静态画面有决定性分辨力）
#     |std[i]-std[i-1]| > max(median_diff * diff_mult, diff_floor)
#   cut_ratio: 剪辑帧豁免——单帧足够高比例的 ROI（任一 U/V 分量）同时跳变
#     且下一跳变未持续，且前后帧均未绝对超标 → 视为场景切变，不计坏帧。
_CHROMA_ROI_CFG = {
    'corner_div': 4,
    'center_div': 2,
    'min_block_px': 1500,
    'abs_mult': 3.0,
    'abs_floor_roi': 20.0,
    'abs_floor_global': 25.0,
    'diff_mult': 8.0,
    'diff_floor': 1.5,
    'cut_ratio': 0.7,
}

_ROI_TINY_WARNED = [False]


def extract_annexb_es(path, codec=None):
    """按编码分支提取 Annex B 结构码流字节（v5: 支持 h264/mpeg4 家族/hevc）。

    返回 (es_file, codec_ok):
      - es_file: _EsFile 包装（.data 为 mmap，内存有界）；非支持编码/空 ES 返回 None
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

    # [v7][FIX-PIPE-DEADLOCK] stdout/stderr 必须并发排空：旧实现「先排 stdout
    # 再读 stderr」会在 ffmpeg stderr 超过管道缓冲（Linux 64KB）时互相阻塞 →
    # copyfileobj 永久卡死（多主机实测复现）。现改为双 daemon 线程同时排空
    # 两条管道 + 主线程 wait(timeout)，超时 kill 并报错，绝不无限阻塞。
    fd, tmp_path = tempfile.mkstemp(prefix='v4es_', suffix='.es')
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=False)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
    out_result = {}
    err_chunks = []

    def _drain_out():
        try:
            with os.fdopen(fd, 'wb') as out_f:
                while True:
                    chunk = proc.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    _watchdog_tick()
            out_result['ok'] = True
        except Exception as e:
            out_result['exc'] = e
            try:
                os.close(fd)
            except Exception:
                pass

    def _drain_err():
        try:
            while True:
                chunk = proc.stderr.read(65536)
                if not chunk:
                    break
                err_chunks.append(chunk)
                _watchdog_tick()
        except Exception:
            pass

    t_out = threading.Thread(target=_drain_out, daemon=True)
    t_err = threading.Thread(target=_drain_err, daemon=True)
    t_out.start()
    t_err.start()
    try:
        rc = proc.wait(timeout=_SUB_TIMEOUT_EXTRACT)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            rc = proc.wait(timeout=15)
        except Exception:
            rc = -9
        stderr_data = b''.join(err_chunks)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(
            'ffmpeg 提取 ES 停滞超时(>%ds): %s'
            % (_SUB_TIMEOUT_EXTRACT,
               stderr_data.decode('utf-8', 'replace')[:500]))
    t_out.join(timeout=30)
    t_err.join(timeout=30)
    stderr_data = b''.join(err_chunks)
    if 'exc' in out_result:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError('ffmpeg 提取 ES 失败: %s' % out_result['exc'])
    if rc != 0:
        raise RuntimeError('ffmpeg 提取 ES 失败: %s'
                           % stderr_data.decode('utf-8', 'replace')[:500])
    if os.path.getsize(tmp_path) == 0:
        os.remove(tmp_path)
        return None, True  # 空 ES：调用方按既有语义处理（h264 判空 / hevc|mpeg4 N/A）
    f = open(tmp_path, 'rb')
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        os.remove(tmp_path)
        raise
    return _EsFile(tmp_path, f, mm), True


class _EsFile:
    """内存有界 Annex B ES 包装: mmap 只读映射（页缓存背衬，不占堆内存）。

    .data 为 mmap 对象（支持 .find / 切片 / len，与 bytes 接口兼容）；
    close() 关闭映射与文件句柄并删除临时文件。
    """

    def __init__(self, tmp_path, file_handle, mm):
        self._tmp_path = tmp_path
        self._file = file_handle
        self._mmap = mm
        self.data = mm

    def close(self):
        mm, f = self._mmap, self._file
        self._mmap = self._file = None
        self.data = None
        for closer in (getattr(mm, 'close', None), getattr(f, 'close', None)):
            try:
                if closer is not None:
                    closer()
            except Exception:
                pass
        if self._tmp_path:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass
            self._tmp_path = None


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

    def read_se(self):
        """Exp-Golomb se(v)：ue(v) 后按 (v+1)//2 符号展开。"""
        v = self.read_ue()
        if v & 1:
            return (v + 1) // 2
        return -(v // 2)

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


def _max_nvdec_sessions(gpu_name):
    """NVDEC 解码会话[硬件预算]，单一来源: src/utils/system_resources.GPUInfo。

    [P1-3] 原实现自带一套 High/Mid/Standard 名称表，与生产侧的
    system_resources.GPUInfo.max_nvdec_sessions 口径不一致（T4 分别得 2 与 4）。
    现改为惰性导入生产侧实现，两者共用同一张表；导入失败（脚本被拷到仓库外
    单独运行等）时退回本地镜像表，数值与生产侧逐项一致。
    """
    try:
        import os as _os
        import sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        _utils = _os.path.join(_root, 'src', 'utils')
        if _os.path.isdir(_utils) and _utils not in _sys.path:
            _sys.path.insert(0, _utils)
        from system_resources import GPUInfo
        return GPUInfo(index=0, name=gpu_name or '', vram_total_gb=0.0,
                       vram_free_gb=0.0, compute_capability='').max_nvdec_sessions
    except Exception:
        name_l = (gpu_name or '').lower()
        if any(x in name_l for x in ('a100', 'a6000', 'l40', 'h100', 'h800')):
            return 8
        if any(x in name_l for x in ('a10', 'l4', 't4', 'v100', 'p100')):
            return 4
        return 2


def compute_gpu_workers(gpu_name=None, hwaccel_mode='auto'):
    """根据 GPU 型号自动计算 --gpu-workers 建议值（同时用作分块并发预算）。

    原则: 取硬件 NVDEC 会话预算（单一来源，见 _max_nvdec_sessions），
          再留 ~25% 余量给其他 GPU 任务（CUDA compute / NVENC 编码 /
          显存与驱动队列争用），并保底 2 以免浪费双会话机型。

      会话预算 8 → 6 workers    (A100/H100/L40/A6000 等)
      会话预算 4 → 3 workers    (T4/V100/P100/A10/L4 等)
      会话预算 2 → 2 workers    (未知/消费级)

    [P1-3] 原实现为 6 / 4 / 2，其中 T4 取 2 —— 与生产侧 max_nvdec_sessions
    给出的 4 不一致，且实测偏低：同一 4K H.264 文件并发 1/2/3 会话时
    NVDEC 聚合利用率 33%/64.5%/78.7%、吞吐 1.0/2.0/2.4x，3 会话仍是甜点，
    故 T4 取 3（会话 4 时利用率 80.3% 但墙钟开始回升，留作余量）。

    仅在 hwaccel=cuda/auto 时有效；hwaccel=off 时返回 0（GPU 不参与）。
    用户可通过 --gpu-workers N 显式覆盖。
    """
    if hwaccel_mode not in ('cuda', 'auto'):
        return 0

    if gpu_name is None:
        gpu_name = detect_gpu_name() or ''

    sessions = max(1, _max_nvdec_sessions(gpu_name))
    return max(2, int(round(sessions * 0.75)))


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

def _build_roi_list(width, height):
    """构造 chroma 平面 ROI 列表: [(name, y0, x0, bh, bw), ...]（y=x 行优先）。

    4 角块锚定帧四角（边长 = chroma 1/4 ≈ luma ¼×¼），中心块居中对称
    （边长 = chroma 1/2 ≈ luma ½×½），最后追加全局整平面兜底。
    角块样本数不足 min_block_px 时降级（仅中心+全局），小分辨率告警一次。
    分辨率过小（chroma < 4×4）返回 []。
    """
    cw, ch = width // 2, height // 2
    if cw < 4 or ch < 4:
        return []
    cfg = _CHROMA_ROI_CFG
    rois = []
    qw = max(cw // cfg['corner_div'], 1)
    qh = max(ch // cfg['corner_div'], 1)
    if qw * qh >= cfg['min_block_px']:
        rois.append(('tl', 0, 0, qh, qw))
        rois.append(('tr', 0, cw - qw, qh, qw))
        rois.append(('bl', ch - qh, 0, qh, qw))
        rois.append(('br', ch - qh, cw - qw, qh, qw))
    elif not _ROI_TINY_WARNED[0]:
        _ROI_TINY_WARNED[0] = True
        print('[WARN] chroma: 分辨率过小，角块 (%dx%d) 不足 %d 样本，'
              '降级为仅中心+全局 ROI'
              % (qw, qh, cfg['min_block_px']), flush=True)
    hw = max(cw // cfg['center_div'], 1)
    hh = max(ch // cfg['center_div'], 1)
    rois.append(('center', (ch - hh) // 2, (cw - hw) // 2, hh, hw))
    rois.append(('global', 0, 0, ch, cw))
    return rois


def _compute_roi_uv_std(planes, width, height, rois):
    """批量计算各帧各 ROI 的 U/V 平面 std。

    planes: (n_frames, frame_size) uint8；返回 (us, vs) 均为 (n_frames, len(rois))。
    行连续 + 零拷贝切片，一次向量化归约；计算量 ≈ 1.5× 原整平面 std。
    """
    n = planes.shape[0]
    y_size = width * height
    uv_size = y_size // 4
    ch, cw = height // 2, width // 2
    u3 = planes[:, y_size:y_size + uv_size].reshape(n, ch, cw)
    v3 = planes[:, y_size + uv_size:].reshape(n, ch, cw)
    us = np.empty((n, len(rois)), dtype=np.float64)
    vs = np.empty((n, len(rois)), dtype=np.float64)
    for j, (_name, y0, x0, bh, bw) in enumerate(rois):
        us[:, j] = u3[:, y0:y0 + bh, x0:x0 + bw].std(axis=(1, 2))
        vs[:, j] = v3[:, y0:y0 + bh, x0:x0 + bw].std(axis=(1, 2))
    return us, vs


def _chroma_postprocess(us, vs, roi_names=None):
    """[v8] 多点 ROI 色度坏帧判定（单流/分块聚合共用）。

    输入 us/vs: (n_frames, n_rois)，每帧每 ROI 的 U/V 平面 std。
    判据:
      1) 坏帧判据 (bad frame): 绝对水平 — 逐 ROI 自校准:
         std > max(median_roi * 1.6, floor)，局部 ROI 下限 6.0，
         全局 ROI 下限 12.0。对应持续性色度污染（U/V std 飙升至
         2-3 倍基线），而非瞬态噪声波动。
      2) 剪辑帧豁免 (scene-cut exemption): 时间域跳变
         (max(median_diff * 8, 1.5)) 仅用于**检测场景切割**，
         不作为坏帧信号。当 ≥70% ROI 同时跳变（up）且跳变
         未持续（down）且 该帧及前后帧均未绝对超标 → 豁免该帧
         不计入坏帧。污染突发自带 down 跳变回基线或自身绝对
         超标，不被豁免。
      [FIX-CHROMA-FA1] 2026-08-20: 修正了 v8 引入的严重误报:
         跳变掩码不再并入 bad_any，仅用于剪辑豁免判断。
    坏帧索引按间距 >= 8 分簇（仅保留簇首帧），调用方以簇数 >= 3 判 FAIL。
    """
    cfg = _CHROMA_ROI_CFG
    n_frames, n_rois = us.shape
    if roi_names is None or len(roi_names) != n_rois:
        roi_names = ['roi%d' % j for j in range(n_rois)]
    med_u = np.median(us, axis=0)
    med_v = np.median(vs, axis=0)
    floors = np.full(n_rois, cfg['abs_floor_roi'])
    for j, name in enumerate(roi_names):
        if name == 'global':
            floors[j] = cfg['abs_floor_global']
    bad_u = us > np.maximum(med_u * cfg['abs_mult'], floors)
    bad_v = vs > np.maximum(med_v * cfg['abs_mult'], floors)
    diff_u = diff_v = None
    if n_frames >= 2:
        du = np.abs(np.diff(us, axis=0))
        dv = np.abs(np.diff(vs, axis=0))
        md_u = np.median(du, axis=0)
        md_v = np.median(dv, axis=0)
        diff_u = du > np.maximum(md_u * cfg['diff_mult'], cfg['diff_floor'])
        diff_v = dv > np.maximum(md_v * cfg['diff_mult'], cfg['diff_floor'])
    cut = np.zeros(n_frames, dtype=bool)
    if diff_u is not None:
        # 按 ROI 计数（任一 U/V 分量跳变即计）: 切变若只有单分量内容变化
        # （如 V 平面本就接近平坦），2*rois 的平面计数会把 frac 稀释到
        # < 0.7 而漏豁免；ROI 计数面对整帧切变所有 ROI 的 U 或 V 必跳变 → 1.0
        jump_frac = (diff_u | diff_v).sum(axis=1) / float(n_rois)
        abs_any = (bad_u | bad_v).any(axis=1)
        for f in range(1, n_frames):
            up_ok = jump_frac[f - 1] >= cfg['cut_ratio']
            down_ok = (f >= n_frames - 1) or (jump_frac[f] < cfg['cut_ratio'])
            # 豁免条件: 全画面瞬态跳入（up 跳变）+ 未持续（下一跳变<cut_ratio，
            # 污染突发会紧随 down 跳变回基线故不被豁免）+ 该帧及前后帧均未
            # 绝对超标（排除污染帧自身超标但跳变刚好落空的组合）
            if up_ok and down_ok and not abs_any[f] and \
                    not abs_any[f - 1] and \
                    (f + 1 >= n_frames or not abs_any[f + 1]):
                cut[f] = True
    # [FIX-CHROMA-FA1] 2026-08-20: 跳变（temporal jump）不应直接并入
    # bad_any，从而判定坏帧。跳变阈值 (max(median_diff * 8, 1.5)) 在
    # median_diff 极小时（平画面/低噪声），下限 1.5 会把像素级噪声
    # 波动误判为"跳变"，继而误标正常帧为坏帧，引发严重误报。
    #
    # 正确语义: 坏帧判据仅基于「绝对水平」(std > max(median*1.6, floor))，
    # 对应的是持续性色度污染（U/V std 飙升至 2-3 倍基线）。
    # 跳变仅用于剪辑豁免（scene-cut exemption），不作为独立坏帧信号。
    bad_any = bad_u | bad_v
    bad_any[cut] = False
    bad_idx = np.where(bad_any.any(axis=1))[0]
    clustered = []
    cluster_regions = []
    for idx in bad_idx:
        idx = int(idx)
        if not clustered or idx - clustered[-1] >= 8:
            clustered.append(idx)
            hits = []
            for j, name in enumerate(roi_names):
                # [FIX-CHROMA-FA1] 仅报告绝对阈值命中 ROI（std > max(median*1.6, floor)），
                # 不再将跳变(diff)作为命中依据——跳变不参与坏帧判定。
                if bad_u[idx, j]:
                    hits.append('%s/U' % name)
                if bad_v[idx, j]:
                    hits.append('%s/V' % name)
            cluster_regions.append('/'.join(sorted(set(hits))[:4]) or '?')
    g_idx = roi_names.index('global') if 'global' in roi_names else -1
    med_u_g = float(med_u[g_idx]) if g_idx >= 0 else float(np.median(us))
    med_v_g = float(med_v[g_idx]) if g_idx >= 0 else float(np.median(vs))
    return {
        'frame_count': n_frames,
        'median_u': round(med_u_g, 2),
        'median_v': round(med_v_g, 2),
        'bad_frames': clustered,
        'bad_count': len(clustered),
        'n_rois': n_rois,
        'roi_names': list(roi_names),
        'cluster_regions': cluster_regions,
    }


def _chroma_shard_worker(path, width, height, start_pts, end_pts, count, rois,
                         threads=1, hwaccel=False):
    """解码单个分块 rawvideo + showinfo 逐帧 pts → (frame_count, us, vs)。

    us/vs 为 (n_frames, len(rois)) 逐帧逐 ROI 的 U/V std（v8 多点探测）。
    stdout 像素帧与 stderr showinfo 行同序（显示序），按 pts 窗口
    [start_pts, end_pts) 过滤，免疫 B 帧重排与线程级截止非确定性；
    任何不一致返回 (0, [], [])，由上层回退单流。
    threads: [v9] 解码线程数（分片时 = cpu//shards），避免多 ffmpeg
    进程各自默认开满自动线程导致 CPU 过订阅、分片线性扩展失效。
    hwaccel: [P1-2] 为 True 时该分块走 NVDEC（不指定 hwaccel_output_format，
    由 ffmpeg 自动回拷并转 yuv420p，与 check_chroma_corruption 的
    hwaccel 分支同构），使 --chroma-hwaccel 也能分片并行。
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg or count <= 0 or not rois:
        return (0, [], [])
    frame_size = width * height * 3 // 2
    cmd = [ffmpeg, '-v', 'verbose', '-threads', str(max(1, int(threads)))]
    if hwaccel:
        cmd += ['-hwaccel', 'cuda']
    cmd += ['-copyts', '-ss', '%.6f' % start_pts,
            '-i', str(path), '-frames:v', str(count + _CHUNK_MARGIN), '-an',
            '-fps_mode', 'passthrough', '-vf', 'showinfo', '-pix_fmt', 'yuv420p',
            '-f', 'rawvideo', 'pipe:1']
    chunk_bytes = frame_size * min(64, max(1, count))
    re_pts = re.compile(r'n:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:\s*([\d.]+)')
    pts_rows = []
    us_all, vs_all = [], []

    def _drain_stderr():
        try:
            for raw in proc.stderr:
                line = raw.decode('utf-8', 'replace')
                m = re_pts.search(line)
                if m:
                    pts_rows.append(float(m.group(2)))
        except Exception:
            pass

    def _drain_stdout():
        # [v7] 跨块保留帧尾余量字节，避免管道短读导致平面错位；
        # 不足一帧时继续累积而不是提前 break。
        leftover = b''
        try:
            while True:
                buf = proc.stdout.read(chunk_bytes)
                if not buf:
                    break
                data = leftover + buf
                n = len(data) // frame_size
                if n == 0:
                    leftover = data
                    continue
                use = data[:n * frame_size]
                leftover = data[n * frame_size:]
                planes = np.frombuffer(use, dtype=np.uint8).reshape(n, frame_size)
                u2, v2 = _compute_roi_uv_std(planes, width, height, rois)
                us_all.append(u2)
                vs_all.append(v2)
                _watchdog_tick()
        except Exception:
            pass

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=False)
    except Exception:
        return (0, [], [])
    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_err.start()
    t_out.start()
    try:
        proc.wait(timeout=_SUB_TIMEOUT_DECODE)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=15)
        except Exception:
            pass
        return (0, [], [])
    t_err.join(timeout=30)
    t_out.join(timeout=30)
    if not us_all:
        return (0, [], [])
    us = np.concatenate(us_all, axis=0)
    vs = np.concatenate(vs_all, axis=0)
    n_pix = us.shape[0]
    # 解码器 EOF flush 时 showinfo 可能比 rawvideo 像素多 1 行（无像素帧），
    # 取两者最小长度前缀（同序显示序），窗口过滤后再校验数量。
    if len(pts_rows) < n_pix or len(pts_rows) - n_pix > _CHUNK_MARGIN + 2:
        return (0, [], [])
    keep = [i for i, p in enumerate(pts_rows[:n_pix])
            if p >= start_pts - 1e-6 and (end_pts is None or p < end_pts - 1e-6)]
    if len(keep) != count:
        return (0, [], [])
    return (count, us[keep], vs[keep])


def _check_chroma_sharded(path, width, height, plan, cpu_count=None,
                          hwaccel=False, max_parallel=None):
    """色度分块并行驱动：各分块独立解码 + 逐 ROI std 归约，失败返回 None（上层回退单流）。

    [v9] 每 shard 解码线程数 = max(1, cpu//shards)，N 个 ffmpeg 进程合计
    ≈ 占满可用核数，消除自动线程过订阅。

    [P1-2] hwaccel=True 时分块走 NVDEC（每个分块一个会话），
    max_parallel 限制同时会话数（NVDEC 并发甜点为 2~3，见
    _NVDEC_CHUNK_PARALLEL）；CPU 路径保持「分块数 = 并发数」。
    """
    chunks = plan['chunks']
    n = len(chunks)
    rois = _build_roi_list(width, height)
    if not rois:
        return None
    cpu = cpu_count or detect_cpu_count()
    threads = max(1, cpu // n)
    pool = n
    if hwaccel:
        pool = max(1, min(n, max_parallel or _NVDEC_CHUNK_PARALLEL))
    results = [None] * n
    with ThreadPoolExecutor(max_workers=pool) as ex:
        futs = []
        for i, (s, c) in enumerate(chunks):
            end = chunks[i + 1][0] if i + 1 < n else None
            futs.append(ex.submit(_chroma_shard_worker, str(path), width, height,
                                  s, end, c, rois, threads, hwaccel))
        for i, f in enumerate(futs):
            try:
                results[i] = f.result()
            except Exception:
                results[i] = (0, [], [])
    if any(r[0] == 0 for r in results):
        return None
    us = np.concatenate([r[1] for r in results], axis=0)
    vs = np.concatenate([r[2] for r in results], axis=0)
    return _chroma_postprocess(us, vs, [r[0] for r in rois])


def check_chroma_corruption(path, hwaccel=False, n_shards=1, shard_plan=None,
                            progress=False, total_frames_fallback=None,
                            cpu_count=None):
    """[v6][v8] 解码 yuv420p 多点 ROI 探测色度花屏坏帧。

    原理: 正常视频 U/V 平面 std 稳定 (v6.4.4/5 全程 U≈35、V≈25)；
    批量编码跨流竞态导致的 NV12 撕裂/色度污染会使部分帧 U/V std 飙升至
    2-3 倍基线（v6.4.3 实测坏帧 73-90 vs 基线 36）。这是像素域信号，
    能捕获 frames/packets、连 IDR、frame_num、pts_anomaly 全部漏检的
    「布纹花屏/帧闪烁」缺陷。

    v8 多点探测: 4 角 + 中心 + 全局共 6 个 ROI（_build_roi_list），每 ROI
    独立自校准「绝对水平」阈值 (std > max(median*1.6, floor))。坏帧判据
    仅基于绝对水平，对应持续性色度污染 (U/V std 飙升 2-3 倍基线)。

    [FIX-CHROMA-FA1] 2026-08-20: 修正 v8 引入的严重误报 —— 时间域跳变
    (temporal jump) 仅用于场景切割豁免 (cut exemption)，不作为坏帧信号。
    原实现将跳变掩码 (trans) 并入 bad_any，导致正常像素噪声波动被误判为
    坏帧 (median_diff 极小时 floor=1.5 主导，正常 1-3 点波动均触发)。

    流式处理: ffmpeg rawvideo 输出按 _CHROMA_CHUNK_FRAMES 帧分块读取，
    每块一次批量 ROI std 归约（axis=(1,2)），内存占用 O(chunk*frame_size)
    有界，不随帧数增长（720×576×1373 帧 ≈ 850MB pipe 全流式）。
    坏帧索引按间距分簇（间距 >= 8 帧算新簇，仅保留簇内首帧），避免单簇多帧刷屏。

    hwaccel: True 时用 NVDEC 解码（帧自动回拷主机，无需 hwdownload），
    加速解码但 NVDEC 错误隐藏与软解不一致，可能掩盖花屏特征，默认 False；
    NVDEC 不可用/解码失败时自动回退 CPU 软解（对齐 v4.1 GPU→CPU 回退）。

    返回 dict（ffprobe/ffmpeg 失败或解码异常返回 None，调用方不判失败）:
        frame_count: 解码帧数
        median_u / median_v: 全局 ROI 各帧 U/V std 的中位数
        bad_frames: 坏帧索引列表（分簇去重后）
        bad_count:   坏帧簇总数（调用方以 >= _CHROMA_BAD_FRAME_MIN 判 FAIL）
        n_rois / roi_names / cluster_regions: v8 逐簇命中区域（如 tl/U）
    """
    if shutil.which('ffprobe') is None or shutil.which('ffmpeg') is None:
        return None
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=30)
        if probe.returncode != 0 or not probe.stdout.strip():
            return None
        parts = probe.stdout.strip().replace('\r', '').split('\n')[-1].split(',')
        width, height = int(parts[0]), int(parts[1])
        total_frames = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
        # [FIX-CHROMA-PROGRESS-FMT] HEVC 流 ffprobe nb_read_frames 常返回 "N/A"
        # （不能一次性计数），导致 total_frames=None，进度栏回退到旧格式
        # '[verify] chroma: 处理中，已累计 %d 帧'。
        # 回退使用检查 1 步（ffmpeg -f null）已得到的 decoder 输出帧数；
        # 该值来自 _verify_one_video 传入的 total_frames_fallback，
        # 对 HEVC/HEVC 等容器不支持 nb_read_frames 的编码均有效。
        if total_frames is None:
            total_frames = total_frames_fallback
        if width <= 0 or height <= 0:
            return None
    except Exception:
        return None

    # [v5] 色度分块并行: 复用关键帧分块计划，各分块 rawvideo 解码 + 逐 ROI
    # std 归约，坏帧索引全局聚合；失败回退单流。
    # [P1-2] 原限定 `not hwaccel`（NVDEC 路径恒单流），现放开 NVDEC 分片，
    # 由 _NVDEC_CHUNK_PARALLEL 限制会话数（并发甜点，避免越过饱和点）。
    if n_shards > 1:
        plan = shard_plan
        if plan is None:
            plan = _build_chunk_plan(path, n_shards)
        if plan is not None:
            total_frames = plan['total']
            sharded = _check_chroma_sharded(path, width, height, plan,
                                            cpu_count=cpu_count,
                                            hwaccel=hwaccel)
            if sharded is not None:
                return sharded
    # 单流路径进度已在 ffprobe 探测阶段通过 `-show_entries stream=nb_read_frames` 获得，
    # 无需额外调用；`total_frames` 已在探测时从 `parts[2]` 直接传入后续流程。
    rois = _build_roi_list(width, height)
    if not rois:
        return None
    frame_size = width * height * 3 // 2
    cmd = ['ffmpeg', '-v', 'error']
    if hwaccel:
        cmd += ['-hwaccel', 'cuda']
    cmd += ['-i', str(path), '-an', '-vsync', '0',
            '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    chunk_bytes = frame_size * _CHROMA_CHUNK_FRAMES

    def _stream_uv_std(cmd, progress=False, total=None):
        """按 chunk 分块读取 rawvideo，批量 std 归约（内存 O(chunk*frame_size)）。

        [v7] 修复: (1) 超时兜底——stdout 由 daemon 线程排空，主线程
        wait(timeout)，停滞超时 kill 后视为失败（调用方回退/置 None）；
        (2) 跨块保留帧尾余量字节，避免管道短读导致 U/V 平面错位；
        不足一帧时继续累积而不是提前 break 丢掉剩余码流。
        [v9] reader/compute 双缓冲重叠（Queue maxsize=2）：旧实现读管道与
        NumPy 归约同线程串行，计算期间 ffmpeg 阻塞在管道写、解码器空转；
        拆分后解码/传输与 std 归约真重叠（NumPy 大数组运算释放 GIL），
        总耗时 ≈ max(解码+传输, 归约) 而非两者之和。
        """
        us, vs = [], []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, shell=False)
        except Exception:
            return [], []

        processed = [0]
        last_print = [time.monotonic()]
        t0 = [time.monotonic()]
        buf_q = Queue(maxsize=2)
        _EOF = object()

        def _reader():
            """只做管道读 + 帧对齐切块，入队后立即继续读（不计算）。"""
            leftover = b''
            try:
                while True:
                    buf = proc.stdout.read(chunk_bytes)
                    if not buf:
                        break
                    data = leftover + buf
                    n = len(data) // frame_size
                    if n == 0:
                        leftover = data
                        continue
                    use = data[:n * frame_size]
                    leftover = data[n * frame_size:]
                    planes = np.frombuffer(use, dtype=np.uint8).reshape(n, frame_size)
                    buf_q.put((n, planes))
            except Exception:
                pass
            finally:
                try:
                    buf_q.put(_EOF)
                except Exception:
                    pass

        def _compute():
            """出队做 ROI std 批量归约 + 进度输出；异常时 kill 解码进程自解阻塞。"""
            try:
                while True:
                    item = buf_q.get()
                    if item is _EOF or item is None:
                        break
                    n, planes = item
                    # 每帧每 ROI U/V 平面 → (n, len(rois)) 批量归约（v8 多点探测）
                    u2, v2 = _compute_roi_uv_std(planes, width, height, rois)
                    us.append(u2)
                    vs.append(v2)
                    processed[0] += n
                    # [v7] 心跳: 每处理一个 chunk 即视为有进展，重置看门狗
                    _watchdog_tick()
                    if progress and time.monotonic() - last_print[0] >= 30:
                        last_print[0] = time.monotonic()
                        print(_fmt_chroma_progress(processed[0], total, t0[0]),
                              flush=True)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        t_read = threading.Thread(target=_reader, daemon=True)
        t_calc = threading.Thread(target=_compute, daemon=True)
        t_read.start()
        t_calc.start()
        try:
            proc.wait(timeout=_SUB_TIMEOUT_DECODE)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=15)
            except Exception:
                pass
            return [], []
        t_read.join(timeout=30)
        # EOF 后队列中可能仍有 ≤2 个待归约 chunk（128 帧 1080p ≈ 10s/chunk）
        t_calc.join(timeout=300)
        return us, vs

    us = vs = None
    try:
        us, vs = _stream_uv_std(cmd, progress=progress, total=total_frames)
    except Exception:
        if not hwaccel:
            return None
    # [v6.1] NVDEC 不可用/解码失败（如无 nvcuda.dll，ffmpeg 无 rawvideo 输出）
    # 或抛出异常 → 回退 CPU 软解（对齐 v4.1 检查 1 的 GPU→CPU 回退哲学）
    if (not us) and hwaccel:
        try:
            us, vs = _stream_uv_std(['ffmpeg', '-v', 'error', '-i', str(path),
                                     '-an', '-vsync', '0', '-pix_fmt', 'yuv420p',
                                     '-f', 'rawvideo', 'pipe:1'],
                                    progress=progress, total=total_frames)
        except Exception:
            return None

    if len(us) == 0:
        return None
    us = np.concatenate(us, axis=0)
    vs = np.concatenate(vs, axis=0)
    return _chroma_postprocess(us, vs, [r[0] for r in rois])


# ═══════════════════════════════════════════════
# 5. 单文件验收（模块级函数，ThreadPoolExecutor 可 pickle）
# ═══════════════════════════════════════════════

def _check2_worker(video_path, codec, dump_nal, out):
    """[v4] 检查 2（码流统计）独立线程体，与检查 1+3 并行重叠执行。

    ES 提取为 -c copy + bitstream filter，不占 NVDEC 解码会话，纯 CPU/I/O。
    结果写入可变 dict `out`（通过 join 同步，无需显式锁）:
      out['codec']: 视频编码名（回显）
      out['stats']: check_nal_stats / check_vop_stats / check_hevc_stats 返回 dict
      out['err']:   检查 2 错误描述（None=通过/N/A）
    聚合时 err 非 None 即判 'nal' 失败，与原 v3 逐分支 fails.append('nal') 语义等价。
    """
    out['codec'] = codec
    out['stats'] = None
    out['err'] = None
    try:
        if codec in _MPEG4_FAMILY:
            # mpeg4 家族: VOP 深度分析（对应 H.264 三项检查）
            es_obj, ok = extract_annexb_es(video_path, codec)
            try:
                if ok and es_obj is not None:
                    es = es_obj.data
                    vops, time_bits = parse_mpeg4_es(es)
                    stats = check_vop_stats(vops, time_bits, dump_nal)
                    out['stats'] = stats
                    # VOP 全解析失败（0 个）→ N/A，不判失败
                    if stats['vop_count_total'] == 0:
                        out['err'] = None
                    else:
                        # 连 I-VOP 检查（对应 H.264 连 IDR）:
                        # 最长连续 I-VOP 块 > 3 才是异常（段首 1-2 个 I 预热帧正常）
                        if stats['max_consec_i_vop'] > 3:
                            out['err'] = ('连续 I-VOP 块最长 %d 个（大量重编码为 I，'
                                          '修复工具特征)') % stats['max_consec_i_vop']
                        # vop_time_increment 回退检查（时间戳逆序）
                        vr = stats['vop_time_regress']
                        if vr is not None and vr > 0:
                            msg = 'vop_time 回退 %d 次（时间戳逆序特征）' % vr
                            out['err'] = (out['err'] + '; ' + msg) if out['err'] else msg
                else:
                    out['err'] = None  # 提取失败/空 → N/A 不判失败
            finally:
                if es_obj is not None:
                    es_obj.close()
        elif codec == 'h264':
            # H.264: NAL 专项分析（原 v2 逻辑不变）
            es_obj, ok = extract_annexb_es(video_path, codec)
            try:
                if not (ok and es_obj is not None):
                    out['err'] = 'H.264 ES 提取为空'
                else:
                    es = es_obj.data
                    nals, frame_num_bits = parse_h264_es(es)
                    stats = check_nal_stats(nals, frame_num_bits, dump_nal)
                    out['stats'] = stats
                    # [v6] 阈值 >0 → >=3: 良性段首双 IDR（SPS/PPS 冗余重注入
                    # 恢复点）不再误报，只抓修复前 per-slot IDR 异常 (6-16+ 个)。
                    if stats['idr_within_32_after_first'] >= _CONSEC_IDR_THRESHOLD:
                        out['err'] = ('段首出现连续 %d 个 IDR 帧（修复前 '
                                      'per-slot IDR 特征)') % stats['idr_within_32_after_first']
                    if stats['frame_num_regress'] > 0:
                        msg = 'frame_num 回退 %d 次（LA 重路由错位特征）' % stats['frame_num_regress']
                        out['err'] = (out['err'] + '; ' + msg) if out['err'] else msg
            finally:
                if es_obj is not None:
                    es_obj.close()
        elif codec == 'hevc':
            # H.265: 基础 NAL/IDR 分析（无 frame_num 概念，与 v2 一致）
            es_obj, ok = extract_annexb_es(video_path, codec)
            try:
                if ok and es_obj is not None:
                    es = es_obj.data
                    nals = parse_hevc_es(es)
                    stats = check_hevc_stats(nals, dump_nal)
                    out['stats'] = stats
                    # [v6] 与 h264 分支同阈值策略（良性双 IDR 不误报）
                    if stats['idr_within_32_after_first'] >= _CONSEC_IDR_THRESHOLD:
                        out['err'] = ('段首出现连续 %d 个 IDR 帧（修复前 '
                                      'per-slot IDR 特征)') % stats['idr_within_32_after_first']
                else:
                    # hevc_mp4toannexb bsf 不可用 → 无专项分析，显示 N/A 不判失败
                    out['stats'] = None
                    out['err'] = None
            finally:
                if es_obj is not None:
                    es_obj.close()
        else:
            # 非支持编码 (vp9/av1 等): N/A，不判失败
            out['stats'] = None
            out['err'] = None
    except Exception as e:
        out['err'] = '码流解析失败: %s' % e


def _verify_one_video(video_path, hwaccel='auto', dump_nal=None, skip_chroma=False,
                      chroma_hwaccel=False, decode_strategy='auto', decode_chunks=1,
                      chroma_shards=1, cpu_count=None, ram_avail_gb=None,
                      batch_workers=1, chunk_plan=None, progress=False):
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

    t0 = time.monotonic()
    timing = {}

    def _phase(msg):
        """[v7] 单文件模式逐阶段进度输出 + 看门狗心跳（批处理仅心跳）。"""
        if progress:
            print('[verify] %s' % msg, flush=True)
        _watchdog_tick()

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

    # [v4] 提前探测 codec（O(1) ffprobe 读 v:0 codec_name，不占 NVDEC），
    # 决定检查 2 是否可并行；codec 立即回填 result（与原 v3 语义一致）。
    _t_probe = time.monotonic()
    codec = probe_video_codec(video_path)
    timing['probe'] = time.monotonic() - _t_probe
    result['codec'] = codec
    _phase('probe: codec=%s (%.1fs)'
           % (codec or 'N/A', timing['probe']))

    # [v4][FIX-CHECK-PARALLEL] 检查间并行：检查 2（ES 提取 -c copy + 纯 Python
    # NAL/VOP/HEVC 解析）为 CPU/I/O 任务、不占 NVDEC 解码会话，与检查 1+3
    # （ffmpeg -f null 全量解码，单文件大视频的主瓶颈）无数据依赖，可并行重叠
    # 缩短墙钟时间。检查 2 移出 GPU 闸门（原 v3 其位于 try 信号量块内，但
    # -c copy 不占 NVDEC，持有信号量反而让 CPU 任务被 GPU 队列串行化，与
    # FIX-CHROMA-GPU-GATE 同理）。daemon 线程承载检查 2，主线程执行检查 1+3。
    nal_result = {}
    nal_thread = None
    _t2_start = time.monotonic()
    if codec == 'h264' or codec == 'hevc' or codec in _MPEG4_FAMILY:
        nal_thread = threading.Thread(
            target=_check2_worker,
            args=(video_path, codec, dump_nal, nal_result),
            daemon=True,
        )
        nal_thread.start()

    _phase('decode: 检查1+3 开始 (strategy=%s, chunks=%d)'
           % (decode_strategy, decode_chunks))
    gpu_acquired = False
    if gpu_task and gpu_sem is not None:
        try:
            gpu_sem.acquire(timeout=_GPU_WAIT_TIMEOUT)
            gpu_acquired = True
        except Exception:
            gpu_acquired = False
        if not gpu_acquired:
            print('[WARN] GPU 信号量等待超时(%ds)，继续执行（可能超出 NVDEC 并发上限）'
                  % _GPU_WAIT_TIMEOUT, flush=True)

    try:
        # ── 1+3) v5 策略化/分块检查: 一次（或多块并行）解码产出 frames + pts_issues ──
        dec = check_decode_integrity(video_path, hwaccel,
                                     strategy=decode_strategy,
                                     chunks=decode_chunks,
                                     cpu_count=cpu_count,
                                     ram_avail_gb=ram_avail_gb,
                                     batch_workers=batch_workers,
                                     chunk_plan=chunk_plan)
        timing['decode'] = dec['timing']['decode']
        result['strategy'] = dec['strategy']
        result['chunked'] = dec['chunked']
        result['n_chunks'] = dec.get('n_chunks', 1)
        result['frames'] = dec['frames']
        result['packets'] = dec['packets']
        if dec['err']:
            result['fp_err'] = dec['err']
            result['fails'].append('frames/packets')
        elif dec['frames'] is not None and dec['packets'] is not None:
            result['fp_match'] = (dec['frames'] == dec['packets'])
            if not result['fp_match']:
                result['fp_err'] = 'frames(%d) != packets(%d)' % (dec['frames'], dec['packets'])
                result['fails'].append('frames/packets')

        result['pts_issues'] = dec['issues']
        if dec['issues'] is None:
            result['pts_err'] = 'ffmpeg 不可用（无法执行 pts 检查）'
            result['fails'].append('pts')
        elif len(dec['issues']) > 0:
            result['pts_err'] = 'ffmpeg verbose 检出 %d 条异常' % len(dec['issues'])
            result['fails'].append('pts')
        _phase('decode: 完成 frames=%s packets=%s (%.1fs)'
               % (dec.get('frames'), dec.get('packets'), timing['decode']))

    finally:
        if gpu_acquired and gpu_task and gpu_sem is not None:
            gpu_sem.release()

    # [v4] join 检查 2 线程并聚合结果。err 非 None 即判 'nal' 失败（与原 v3
    # 逐分支 fails.append('nal') 语义等价）；N/A（err=None）不判失败。
    if nal_thread is not None:
        _nal_timeout = _SUB_TIMEOUT_DECODE + _SUB_TIMEOUT_EXTRACT + 120
        nal_thread.join(timeout=_nal_timeout)
        timing['check2'] = time.monotonic() - _t2_start
        if nal_thread.is_alive():
            print('[WARN] 检查 2 线程停滞超时(>%ds)，NAL 检查标记失败'
                  % _nal_timeout, flush=True)
            result['nal_stats'] = None
            result['nal_err'] = '检查 2 停滞超时（>%ds）' % _nal_timeout
            result['fails'].append('nal')
        else:
            result['nal_stats'] = nal_result.get('stats')
            result['nal_err'] = nal_result.get('err')
            if nal_result.get('err') is not None:
                result['fails'].append('nal')
        _phase('check2: 码流统计完成 (%.1fs)' % timing['check2'])

    # ── 4) [v6] 色度平面异常检查（像素域，与结构检查正交）──
    # [v6.1][FIX-CHROMA-GPU-GATE] 默认纯 CPU 软解 + NumPy，不占用 NVDEC 会话，
    # 置于 GPU 闸门外执行，批处理下与其它文件的 GPU 检查并行；
    # --chroma-hwaccel 时走 NVDEC（chroma_gpu），需重新进入闸门。
    # --skip-chroma 时跳过该步骤（默认包含该检测）。
    if skip_chroma:
        result['chroma_skipped'] = True
    else:
        _phase('chroma: 检查4 开始 (%s)'
               % ('NVDEC' if chroma_gpu else 'CPU 软解'))
        # [P1-2] 原实现 NVDEC 路径强制单流（"尊重 GPU 闸门"）。实测单会话
        # NVDEC 在 4K 上只有 ~33% 利用率，故放开为最多 _NVDEC_CHUNK_PARALLEL
        # 个分片，并按实际分片数逐个申请闸门许可（原来只申请 1 个，
        # 若直接分片会绕过 gpu-workers 限流）。
        # CPU 软解路径维持原语义（分片数 = chroma_shards）。
        shards_eff = chroma_shards
        if chroma_gpu:
            shards_eff = max(1, min(chroma_shards, _NVDEC_CHUNK_PARALLEL))
        shard_plan = None
        if shards_eff > 1 and chunk_plan is None:
            shard_plan = _build_chunk_plan(video_path, shards_eff)
        chroma_gpu_acquired = 0
        if chroma_gpu and gpu_sem is not None:
            need = shards_eff if shards_eff > 1 else 1
            for _ in range(need):
                try:
                    gpu_sem.acquire(timeout=_GPU_WAIT_TIMEOUT)
                    chroma_gpu_acquired += 1
                except Exception:
                    break
            if chroma_gpu_acquired == 0:
                print('[WARN] chroma GPU 信号量等待超时(%ds)，按 CPU 软解继续'
                      % _GPU_WAIT_TIMEOUT, flush=True)
                chroma_gpu = False
            elif chroma_gpu_acquired < need:
                # 只拿到部分许可：降级为对应分片数，避免超出 gpu-workers 限流
                print('[WARN] chroma GPU 仅取得 %d/%d 个闸门许可，分片数降级为 %d'
                      % (chroma_gpu_acquired, need, chroma_gpu_acquired), flush=True)
                shards_eff = chroma_gpu_acquired
                shard_plan = None
                if shards_eff > 1:
                    shard_plan = _build_chunk_plan(video_path, shards_eff)
        _t_chroma = time.monotonic()
        try:
            chroma_stats = check_chroma_corruption(video_path, hwaccel=chroma_gpu,
                                                   n_shards=shards_eff,
                                                   shard_plan=shard_plan,
                                                   progress=progress,
                                                   total_frames_fallback=result.get('frames'),
                                                   cpu_count=cpu_count)
            result['chroma_stats'] = chroma_stats
            if chroma_stats is not None and chroma_stats['bad_count'] >= _CHROMA_BAD_FRAME_MIN:
                result['chroma_err'] = ('色度平面异常 %d 簇（周期性色度污染/花屏特征）'
                                        % chroma_stats['bad_count'])
                result['fails'].append('chroma')
        except Exception:
            result['chroma_stats'] = None
        finally:
            timing['chroma'] = time.monotonic() - _t_chroma
            # [P1-2] 按实际取得的许可数逐个归还（分片路径可能持有多个）
            if chroma_gpu_acquired and gpu_sem is not None:
                for _ in range(chroma_gpu_acquired):
                    try:
                        gpu_sem.release()
                    except Exception:
                        break
            _phase('chroma: 完成 (%.1fs)' % timing['chroma'])

    result['timing'] = timing
    result['pass_all'] = len(result['fails']) == 0
    result['elapsed'] = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════
# 6. 并行执行引擎
# ═══════════════════════════════════════════════

def run_verify_parallel(video_files, hwaccel='auto', dump_nal=None,
                         workers=1, parallel_mode='thread',
                         gpu_workers=0, skip_chroma=False,
                         chroma_hwaccel=False, decode_strategy='auto',
                         decode_chunks=1, chroma_shards=1,
                         cpu_count=None, ram_avail_gb=None) -> list:
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
            'decode_strategy': decode_strategy,
            'decode_chunks': decode_chunks,
            'chroma_shards': chroma_shards,
            'cpu_count': cpu_count,
            'ram_avail_gb': ram_avail_gb,
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
                                     chroma_hwaccel=t['chroma_hwaccel'],
                                     decode_strategy=t['decode_strategy'],
                                     decode_chunks=t['decode_chunks'],
                                     chroma_shards=t['chroma_shards'],
                                     cpu_count=t['cpu_count'],
                                     ram_avail_gb=t['ram_avail_gb'],
                                     batch_workers=workers)
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
                                   t['skip_chroma'], t['chroma_hwaccel'],
                                   t['decode_strategy'], t['decode_chunks'],
                                   t['chroma_shards'], t['cpu_count'],
                                   t['ram_avail_gb'], workers)] = t
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

def detect_vram_gb():
    """探测首个 NVIDIA GPU 显存 (total_gb, free_gb)；失败返回 (0.0, 0.0)。

    优先 pynvml；其次 nvidia-smi --query-gpu=memory.total,memory.free。
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total / 1e9, info.free / 1e9
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().splitlines()[0].replace(' ', '').split(',')
            if len(parts) == 2:
                return int(parts[0]) / 1024.0, int(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0, 0.0


def _parse_count_arg(value):
    """解析 --decode-chunks / --chroma-shards 参数（auto 或正整数），返回 int 或 None。"""
    if isinstance(value, int):
        return max(1, value)
    s = str(value or '').strip().lower()
    if s in ('auto', ''):
        return None
    try:
        return max(1, int(s))
    except ValueError:
        return None


def _auto_parallel_counts(cpu_count, ram_avail_gb, n_active_files, gpu_enabled,
                          gpu_conc=None):
    """按容器感知资源自动计算 (decode_chunks, chroma_shards) 建议值。

    [v9] n_active_files = 实际并发处理的文件数 min(文件总数, workers)。
      单文件验收时为 1 —— 全部 CPU/RAM 预算归该文件的分块并行
      （旧版按批处理 workers 均分，单文件场景误得 cap=1，分块恒关闭，
       8 核机也只会单流跑满 1 核）。

    [P1-1] GPU 主机不再无条件 decode_chunks=1。原设定基于「GPU 走 gpu_dual
      单次解码」的假设，实测该路径恒劣于单次 framecrc，且单会话 NVDEC 在 4K
      上只有 ~33% 利用率。现按可用 NVDEC 会话数（gpu_conc，缺省取
      _NVDEC_CHUNK_PARALLEL）在活跃文件间均分，单文件场景即可拿到 2~3 路
      并行（实测 4K 吞吐 2.0~2.4x）。

    decode_chunks / chroma_shards 均为 per-file 分块数；CPU 侧受核数与可用
    RAM（每解码 ~300MB）共同约束，GPU 侧受 NVDEC 会话数约束。
    """
    active = max(1, n_active_files)
    per_file_cpu = max(1, cpu_count // active)
    per_file_ram = 1
    if ram_avail_gb > 0:
        usable_mb = ram_avail_gb * 1024 * 0.7
        per_file_ram = max(1, int(usable_mb // (300 * active)))
    cap = max(1, min(per_file_cpu, per_file_ram))
    if gpu_enabled:
        total_sessions = max(1, gpu_conc or _NVDEC_CHUNK_PARALLEL)
        decode_chunks = max(1, min(total_sessions // active, _NVDEC_CHUNK_PARALLEL))
    else:
        decode_chunks = max(1, min(cap, 8))
    chroma_shards = max(1, min(cap, 8))
    return decode_chunks, chroma_shards


def main():
    # [FIX-STDIN-TTOU] 后台进程组 + tty stdin 时，本脚本拉起的子 ffmpeg 会对 fd0
    # 调 ioctl(TCSETS) 触发 SIGTTOU 被停住 —— 表现是"验收脚本秒卡、0% CPU、
    # 无任何输出"，极易误判为码流损坏或 NVDEC 挂死。入口处把 fd0 换成
    # /dev/null 即可根除（详见 src/utils/stdin_hardening.py）。
    # 脚本要求可脱离仓库单独运行，故导入失败时静默跳过。
    try:
        import os as _os
        import sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        _utils = _os.path.join(_root, 'src', 'utils')
        if _os.path.isdir(_utils) and _utils not in _sys.path:
            _sys.path.insert(0, _utils)
        from stdin_hardening import detach_background_stdin
        detach_background_stdin()
    except Exception:
        pass

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
  python Accessory/verify/segment_bitstream_verify_v3.py a.mp4
  python Accessory/verify/segment_bitstream_verify_v3.py ./segments --hwaccel cuda --gpu-workers 4
  python Accessory/verify/segment_bitstream_verify_v3.py ./segments --hwaccel cuda --chroma-hwaccel
  python Accessory/verify/segment_bitstream_verify_v3.py a.avi ./dir1/ b.mp4 --workers 8 --hwaccel auto
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
    ap.add_argument('--decode-chunks', default='auto', metavar='N|auto',
                    help='检查1+3 单文件分块解码数 (auto=按CPU/RAM自动计算; 仅CPU软解路径生效; 1=整文件单次)')
    ap.add_argument('--chroma-shards', default='auto', metavar='N|auto',
                    help='检查4 色度分块并行数 (auto=自动; CPU 软解与 --chroma-hwaccel 均生效，'
                         'GPU 侧并发钳到 %d; 1=单流)' % _NVDEC_CHUNK_PARALLEL)
    ap.add_argument('--decode-strategy', choices=['auto', 'showinfo', 'gpu_dual', 'framecrc'],
                    default='auto',
                    help='检查1+3 解码策略: auto(默认; 统一走 framecrc，实测 GPU/CPU 均最优) | '
                         'showinfo(兼容旧语义，慢~2x) | gpu_dual(双进程重复解码，仅A/B对照) | framecrc')
    ap.add_argument('--timing', action='store_true',
                    help='输出逐检查分项耗时与所用解码策略')
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

    # ── [v5] VRAM 探测与分块/分片并发计划 ──
    vram_total, vram_free = detect_vram_gb()
    vram_note = ''
    if gpu_enabled and vram_total > 0:
        vram_note = ' | VRAM %.1fGB/%.1fGB free' % (vram_free, vram_total)
    decode_chunks = _parse_count_arg(args.decode_chunks)
    chroma_shards = _parse_count_arg(args.chroma_shards)
    # [v9] 实际并发文件数: 单文件验收=1（全资源归该文件的分块并行）；
    # 批处理时 = min(文件总数, workers)，各活跃文件均分 CPU/RAM 预算。
    n_active_files = min(len(video_files), workers)
    if decode_chunks is None or chroma_shards is None:
        auto_chunks, auto_shards = _auto_parallel_counts(cpu_count, ram_avail,
                                                         n_active_files, gpu_enabled,
                                                         gpu_conc=gpu_workers)
        if decode_chunks is None:
            decode_chunks = auto_chunks
        if chroma_shards is None:
            chroma_shards = auto_shards
    decode_chunks = max(1, decode_chunks)
    chroma_shards = max(1, chroma_shards)
    print("分块计划: decode-chunks=%d%s | chroma-shards=%d | decode-strategy=%s | 活跃文件=%d%s" % (
        decode_chunks,
        (' (GPU 分块并发上限 %d)' % _NVDEC_CHUNK_PARALLEL if gpu_enabled else ''),
        chroma_shards, args.decode_strategy, n_active_files, vram_note))

    # ── 多文件模式下禁用 --dump-nal（避免相互覆盖）──
    dump_nal = args.dump_nal if single_file else None
    if not single_file and args.dump_nal:
        print("[WARN] 多文件模式下 --dump-nal 已忽略（避免覆盖），单文件模式可用")

    # ── 执行验收 ──
    # [v7] 看门狗: 阶段停滞超时自动 dump 全部线程堆栈到
    # ./verify_segment_bitstream_v4_stuck.log（静默卡死可定位）
    _watchdog_start()
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
                                chroma_hwaccel=args.chroma_hwaccel,
                                decode_strategy=args.decode_strategy,
                                decode_chunks=decode_chunks,
                                chroma_shards=chroma_shards,
                                cpu_count=cpu_count,
                                ram_avail_gb=ram_avail,
                                batch_workers=workers,
                                progress=True)
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

        if args.timing:
            t = res.get('timing') or {}
            print('[timing] probe=%.2fs decode=%s check2=%s chroma=%s' % (
                t.get('probe', 0), _fmt_elapsed(t.get('decode', 0)),
                _fmt_elapsed(t.get('check2', 0)), _fmt_elapsed(t.get('chroma', 0))))
            print('          strategy=%s chunked=%s chunks=%d' % (
                res.get('strategy', '?'), res.get('chunked', False),
                res.get('n_chunks', 1)))

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
                                       chroma_hwaccel=args.chroma_hwaccel,
                                       decode_strategy=args.decode_strategy,
                                       decode_chunks=decode_chunks,
                                       chroma_shards=chroma_shards,
                                       cpu_count=cpu_count,
                                       ram_avail_gb=ram_avail)
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
