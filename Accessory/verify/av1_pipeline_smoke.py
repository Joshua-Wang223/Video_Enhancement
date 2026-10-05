#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AV1 NVENC 端到端冒烟 + 验收（constqp / vbr 双 rate_mode）—— 可重复执行入口。

为什么需要它
------------
2026-09-30 在 L40 上第一次跑「长视频 AV1 冒烟」时**开箱即失败**，一次性暴露三处
AV1 路径缺陷（方案 §8.6）：cv2 验收层把完好 AV1 产物判成损坏、ESRGan 侧 writer
静默改用 libx264、AV1 的 `vbr_hq` 未降级导致命令失败。三者都**只在下发真实命令
并解码产物时**才暴露 —— 此前 AC1~AC7 全是「下发单条 ffmpeg 命令」的微观测试。
本脚本把那次一次性验证固化成一条命令，供每次改动 AV1 路径后复跑。

    python3 Accessory/verify/av1_pipeline_smoke.py --src <真实素材> < /dev/null
    python3 Accessory/verify/av1_pipeline_smoke.py --src <真实素材> \
        --codec h264_nvenc --rate-modes constqp,vbr_hq < /dev/null   # T4 上复现 S8 机制

覆盖
----
| 阶段 | 内容 |
|---|---|
| 前置 | 目标编码器（默认 `av1_nvenc`）**实跑一帧**（不是 `-h encoder=`）；不可用 ⇒ exit 2（环境前置不成立） |
| 跑批 | 对每个 rate_mode 调 `src/main_video_optimized.py`，两侧 `--codec-* <CODEC>` |
| 采样 | 整棵进程树 RSS/PSS + **主进程 vs 子进程分组斜率** + 按 pid 归属的 GPU 显存（判泄漏：后半程斜率 + 峰值上界）|
| 验收 | ① 退出码 ② 段级日志 `decoded == expected` ③ 产物 `ffprobe -count_frames` 帧数守恒 ④ `validate_decodable_video(count_mode='decode')` ⑤ `segment_bitstream_verify_v5 --skip-chroma` 硬指标 ⑥ QA sidecar 字段完整 |

⚠ **显存维度可能「不可归属」而不是 0**（`[FIX-B3-GPU-UNATTRIBUTABLE]`）：
`nvidia-smi --query-compute-apps=pid` 在容器里返回**宿主机命名空间 pid**，容器 `/proc` 下
不存在 ⇒ 无法按 pid 归属。本脚本此时记 `None` + 状态串（报告里渲染为「不可归属」并附原因），
**绝不填 0**——0 会被读成「确实没用显存」，而 T4 实测整卡 7698 MiB / 100% 时该字段也是 0。
这只影响显存维度；RSS/PSS 走 `/proc`，口径正确。判泄漏仍以 RSS 斜率为主判据。

⚠ **必须 `< /dev/null`**：这些脚本在「后台进程组 + tty stdin」下会被 SIGTTOU 整组停住。
⚠ 色度检查（检查 4）在真实素材上是**内容相关假阳性**（方案 §8.5：AV1 长片 113 簇，
   而源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。

`--codec`（S8 机制复现用）
------------------------
`constqp` 强制 `LA=0`（硬件静默禁用）⇒ 走 `encode_frames_batch_ce_pipeline`；
其它 rate_mode 保留配置 LA ⇒ 走 `encode_frames_stream` 分块累积。**两条路径与编码器无关**
（`main.py` 的 Level 1 直通判据是 `'nvenc' in use_codec`），故 **T4 上用 `h264_nvenc`/
`hevc_nvenc`跑 constqp vs vbr_hq，可复现 AV1/L40 上 S8 观测到的同一机制**（方案 §9.5 D项：
降本验证，省掉等 L40）。Turing 无 AV1 NVENC ⇒ 本机须显式 `--codec h264_nvenc`。

⚠ **`h264_nvenc`/`hevc_nvenc` 的对照臂必须用 `vbr_hq`，不能用 `vbr`**（[FIX-B2-VBR-CBR-REJECT]）：
`vbr`/`cbr` 在 NVENC SDK 直通（Level 1）上**未实现** —— `nvenc_sdk._build_encoder_config`
只有 `vbr_hq`/`qvbr` 两个 CQ 分支，`vbr` 落 `else` 兜底 ⇒ `rc_ptr[1]=0`（真 CONSTQP）且
LA 门控不含它 ⇒ 硬件 LA 也不使能。即 `vbr` 臂拿到的仍是 CONSTQP，**与constqp 臂同路径**，
A/B 失去意义。故 `main_video_optimized.py` 对非 AV1 的 NVENC 编码器**直接拒绝** `vbr`/`cbr`
（AV1 豁免：AV1 硬件本就只支持 constqp/vbr/cbr）。
⇒ 脚本在 `--codec` 为 h264/hevc NVENC 时，若 `--rate-modes` 含 `vbr`/`cbr` **提前报错退出**，
   避免跑完 10 分钟才发现臂失效。`av1_nvenc`（本脚本默认）不受影响。

`--batch-size`（T4 默认 8）
--------------------------
config 默认 `batch_size=24`。T4 上实测（358.76s 素材切 40s 段，插帧阶段，2 轮交替 A/B）：

| bs | 墙钟（中位） | 单批 ms | pinned result pool |
|---|---|---|---|
| 24（config 默认） | 33.45 s | 343 / 363 | 305 MB |
| **8** | **29.14 s（快 12.9%）** | **90 / 90** | **102 MB** |

两轮各差 <0.5% ⇒ 可复现。**双重收益**：
① 吞吐更高（单批 3.9× 快 ⇒ 同样 24 帧只要 90ms 而非 353ms）；
② **S8 的锯齿幅度由 batch_size 驱动** —— pinned result pool 按 `n × 单帧字节` 线性分配，
bs=24 时逐段从 305 MB 涨到 1251 MB（实测 HEVC constqp 臂 12 段），而这正是让后半程 OLS
斜率在 **−786 ~ +1434 MB/min** 之间跳变的噪声源 ⇒ 小 bs 让 S8 斜率可信。
`--batch-size 0` = 不下发，沿用 config。

⚠ **换 batch_size 会破坏与历史 S8 读数的可比性**（L40 的 +149.5 是 bs=24 下测的）。
跨批次比较必须同 bs。

退出码：0 = 无 FAIL；1 = 有 FAIL；2 = 环境前置不成立（无 ffmpeg / 编码器不可用）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "src" / "main_video_optimized.py"
VERIFY5 = ROOT / "Accessory" / "verify" / "segment_bitstream_verify_v5.py"

# 帧守恒口径：插帧 2× ⇒ 每段 (n-1)*2+1 = 2n-1；最终产物 = 各段之和（不重建公式）
QA_REQUIRED_FIELDS = ("encoding_generation", "generated_by", "source", "mode",
                      "ifrnet_factor", "realesrgan_factor", "codec_hint",
                      "rate_mode_ifrnet", "lookahead_depth_ifrnet",
                      "fixes_applied", "validated_at_utc")


# ── 进程与度量 ────────────────────────────────────────────────────────────────
def run(cmd: List[str], timeout: int = 3600) -> Tuple[int, str, str]:
    """跑外部命令。stdin 固定 DEVNULL（后台进程组 + tty 下会被 SIGTTOU 整组停住）。"""
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)


def probe_int(ffprobe: str, path: Path, entry: str) -> int:
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                      "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(path)], 900)
    try:
        return int((out or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def probe_stream_codec(ffprobe: str, path: Path) -> str:
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)], 120)
    return (out or "").strip().splitlines()[0] if out else ""


def has_audio(ffprobe: str, path: Path) -> bool:
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "a",
                      "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)], 120)
    return bool((out or "").strip())


def encoder_available(ffmpeg: str, codec: str) -> Tuple[bool, str]:
    """**实跑一帧**判编码器能力（`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样打印选项表，
    构建里有 ≠ 硬件编得动 —— 本仓 memory 有实证）。"""
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
           "-frames:v", "1", "-c:v", codec, "-f", "null", "-"]
    rc, _, err = run(cmd, 180)
    if rc == 0:
        return True, "实跑一帧成功"
    first = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return False, (first[0] if first else f"rc={rc}")[:200]


#: ffmpeg 编码器名 → 产物 ffprobe `codec_name`（S7 断言用）。
#: 不在表内的编码器不做断言（codec 名通常去掉 `_nvenc` 后缀，但不必猜）。
EXPECTED_CODEC_NAME = {
    "av1_nvenc": "av1",
    "h264_nvenc": "h264",
    "hevc_nvenc": "hevc",
}


def source_frame_count(ffprobe: str, path: Path) -> int:
    """源片段真实解码帧数（用于核对 2n-1 口径；容器元数据不可信时以解码为准）。"""
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                      "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                      str(path)], 1800)
    try:
        return int((out or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


# ── 内存采样（判泄漏）────────────────────────────────────────────────────────
def _read_pss_kb(pid: int) -> int:
    """Pss（smaps_rollup）比 RSS 更适合跨进程求和：RSS 把共享页（libcuda 等）
    在每个进程里各算一份，进程数越多越虚高。读不到（非 Linux/无权限）返回 0。"""
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _gpu_mem_by_pid() -> Tuple[Optional[Dict[int, float]], str]:
    """本机各进程占用的显存（MiB）。整卡 `memory.used` 在共享 GPU 主机上会被
    其他会话污染（本仓 memory 有实证），必须按 pid 归属后再由调用方筛进程树。

    [FIX-B3-GPU-UNATTRIBUTABLE] 返回 `(表, 状态)`：
      · `({}, "empty")`     —— 驱动侧没有 compute app（确实没进程在用 GPU）；
      · `({pid: mib}, "ok")` —— 表可用；
      · `(None, 原因)`       —— **测不到**（不是 0）。

    ⚠ **不可用 `0` 冒充「没占显存」**（B3，2026-10-04 T4 实测推翻旧归因）：
    `nvidia-smi --query-compute-apps=pid` 返回的是**宿主机命名空间 pid**
    （实测 725115 / 1071006），在容器 `/proc` 下不存在；容器内 `NSpid` 只有一层
    ⇒ 驱动侧与容器 PID namespace 不通，pid 查表永远 miss ⇒ 整卡实测 7698 MiB / 100%
    利用率的同时，本字段恒为 0。`0` 会被读成「确实没用显存」，只有 `None` 才能表达
    「测不到」。**状态串**会原样进入 S8 detail 与报告，避免无声降级。
    """
    saw_field = False
    for field in ("used_gpu_memory", "used_memory"):      # 不同 driver/版本字段名不同
        try:
            p = subprocess.run(
                ["nvidia-smi", f"--query-compute-apps=pid,{field}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return None, f"nvidia-smi 调用失败（{type(exc).__name__}）"
        if p.returncode != 0:
            return None, f"nvidia-smi rc={p.returncode}（{(p.stderr or '').strip()[:80]}）"
        out = (p.stdout or "").strip()
        saw_field = True
        if not out:
            continue
        got: Dict[int, float] = {}
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                got[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
        if got:
            # 驱动侧 pid 在本容器 /proc 下**全部不存在** ⇒ PID namespace 不通，
            # 无法把显存归属到进程树（宿主 pid 725115/1071006 是本次实测的证据）。
            if not any(os.path.isdir(f"/proc/{pid}") for pid in got):
                sample = next(iter(got))
                return None, (f"驱动侧 pid 与容器 PID namespace 不通"
                              f"（如 {sample} 在 /proc 下不存在）⇒ 不可归属")
            return got, "ok"
    if not saw_field:
        return None, "nvidia-smi 无 compute-app 字段（无 GPU 或驱动不支持）"
    return {}, "empty"


def _gpu_total_mib() -> Optional[float]:
    """整卡已用显存（MiB）——只作参考，不作判据（见 _gpu_mem_by_pid 注释）。
    测不到返回 `None`（同 [FIX-B3-GPU-UNATTRIBUTABLE]：0 会被读成「整卡没占显存」）。"""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        return float(q.splitlines()[0])
    except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
        return None


class MemWatcher:
    """盯住匹配 `--match` 的进程树，周期采样内存/显存。

    [FIX-S8-ATTRIB] 原实现只累加进程树 RSS 求和后存 `(t, rss, gpu)`，
    **算出来的进程数 `n` 直接丢弃** ⇒ 上机复现出斜率后无法判读是「主进程泄漏」
    还是「子进程（ffmpeg 读帧器/muxer）累积」。本版每次采样额外记录：
      · `n`：进程树进程数（判断子进程是否随时间增加）；
      · 主进程 / 子进程**分组** RSS 与 PSS（泄漏归属，一眼可分）；
      · 每进程明细（top-N RSS + 全量 pid/rss/pss/tag）落盘成 JSONL，
        事后可离线重分析，不必再花一次 GPU 上机；
      · 显存按 **pid 归属**统计（`_gpu_mem_by_pid`），整卡值仅作参考
        ——共享 GPU 主机上整卡值会把他人的任务算进来。
    PSS 用 `/proc/<pid>/smaps_rollup`，读不到则为 0（不影响 RSS 判据）。
    """

    #: 进程角色分类：主进程 = 跑 main_video_optimized.py 的那个 python；
    #: ffmpeg = 读帧器 / muxer 子进程；其余归 other。
    @staticmethod
    def _tag(args: str, main_marker: str) -> str:
        base = args.split()[0] if args.split() else ""
        if main_marker in args:
            return "main"
        if "ffmpeg" in base or "ffmpeg" in args.split()[0:2]:
            return "ffmpeg"
        if base.endswith("ffprobe"):
            return "ffprobe"
        return "other"

    def __init__(self, match: str, interval: int = 10, dump_path: Optional[Path] = None):
        self.match = match
        self.interval = interval
        self.dump_path = Path(dump_path) if dump_path else None
        self.rows: List[Dict[str, Any]] = []
        self._dump_fh = None

    def _sample(self) -> Dict[str, Any]:
        out = subprocess.run(["ps", "-eo", "pid,ppid,rss,args", "--no-headers"],
                             capture_output=True, text=True).stdout
        kids: Dict[int, List[int]] = {}
        rss: Dict[int, int] = {}
        args: Dict[int, str] = {}
        for line in out.splitlines():
            p = line.split(None, 3)
            if len(p) < 4:
                continue
            pid, ppid, r = int(p[0]), int(p[1]), int(p[2])
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = r
            args[pid] = p[3]
        roots = [p for p, a in args.items() if self.match in a and "ps -eo" not in a]
        seen: set = set()
        procs: List[Dict[str, Any]] = []
        for root in roots:
            stack = [root]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                procs.append({
                    "pid": x,
                    "rss_mb": rss.get(x, 0) / 1024.0,
                    "pss_mb": _read_pss_kb(x) / 1024.0,
                    "tag": self._tag(args.get(x, ""), self.match),
                })
                stack += kids.get(x, [])
        procs.sort(key=lambda d: d["rss_mb"], reverse=True)

        def _grp(tag: str, key: str = "rss_mb") -> float:
            return sum(d[key] for d in procs if d["tag"] == tag)

        gpu_by_pid, gpu_status = _gpu_mem_by_pid()
        # [FIX-B3-GPU-UNATTRIBUTABLE] 不可归属时 gpu_tree_mib = None（**不是 0**）：
        # 0 会被读成「本进程树确实没用显存」，而实测整卡满载时本字段也是 0。
        gpu_tree = (None if gpu_by_pid is None
                    else sum(gpu_by_pid.get(d["pid"], 0.0) for d in procs))
        return {
            "n": len(procs),
            "rss_mb": sum(d["rss_mb"] for d in procs),
            "pss_mb": sum(d["pss_mb"] for d in procs),
            "rss_main_mb": _grp("main"),
            "rss_child_mb": sum(d["rss_mb"] for d in procs if d["tag"] != "main"),
            "pss_main_mb": _grp("main", "pss_mb"),
            "pss_child_mb": sum(d["pss_mb"] for d in procs if d["tag"] != "main"),
            "gpu_tree_mib": gpu_tree,
            "gpu_status": gpu_status,
            "gpu_total_mib": _gpu_total_mib(),
            "procs": procs,
        }

    def run(self, proc: subprocess.Popen) -> None:
        if self.dump_path:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            self._dump_fh = self.dump_path.open("w", encoding="utf-8")
            self._dump_fh.write("# t_s\tn_proc\trss_mb\tpss_mb\trss_main_mb\t"
                                "rss_child_mb\tgpu_tree_mib\tprocs_json\n")
        t0 = time.time()
        try:
            while proc.poll() is None:
                s = self._sample()
                if s["n"]:
                    t = time.time() - t0
                    self.rows.append({"t": t, **{k: v for k, v in s.items() if k != "procs"}})
                    if self._dump_fh:
                        import json as _json
                        # [FIX-B3-GPU-UNATTRIBUTABLE] 显存不可归属时写 `NA` 而非 `0.0`
                        # （0.0 会被后续离线复算读成「确实没用显存」）。
                        _gpu = ("NA" if s["gpu_tree_mib"] is None
                                else f"{s['gpu_tree_mib']:.1f}")
                        self._dump_fh.write(
                            f"{t:.1f}\t{s['n']}\t{s['rss_mb']:.1f}\t{s['pss_mb']:.1f}\t"
                            f"{s['rss_main_mb']:.1f}\t{s['rss_child_mb']:.1f}\t"
                            f"{_gpu}\t"
                            f"{_json.dumps(s['procs'], ensure_ascii=False)}\n")
                        self._dump_fh.flush()
                time.sleep(self.interval)
        finally:
            if self._dump_fh:
                self._dump_fh.close()
                self._dump_fh = None

    @staticmethod
    def _slope(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        """后半程线性斜率（MB/min）；样本不足 8 或 x 无方差时返回 None。"""
        if len(rows) < 8:
            return None
        half = rows[len(rows) // 2:]
        xs = [r["t"] for r in half]
        ys = [r[key] for r in half]
        m = len(xs)
        mx, my = sum(xs) / m, sum(ys) / m
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 60.0

    def slope_mb_per_min(self) -> Optional[float]:
        """后半程 **RSS** 斜率（MB/min）——S8 判据口径（与历史记录可比，不改）。"""
        return self._slope(self.rows, "rss_mb")

    def slope_breakdown(self) -> Dict[str, Optional[float]]:
        """泄漏归属：主进程 / 子进程 / PSS 各自的斜率（诊断列，不作判据）。"""
        return {
            "rss_main": self._slope(self.rows, "rss_main_mb"),
            "rss_child": self._slope(self.rows, "rss_child_mb"),
            "pss_all": self._slope(self.rows, "pss_mb"),
        }

    def peak(self) -> Tuple[float, Optional[float]]:
        """(RSS 峰值 MB, 本进程树显存峰值 MiB)。

        [FIX-B3-GPU-UNATTRIBUTABLE] 显存全样本不可归属时返回 `None`（测不到），
        区别于「峰值 = 0」（确实没占）。
        """
        if not self.rows:
            return 0.0, None
        gpu_vals = [r["gpu_tree_mib"] for r in self.rows
                    if r.get("gpu_tree_mib") is not None]
        return (max(r["rss_mb"] for r in self.rows),
                max(gpu_vals) if gpu_vals else None)

    def gpu_status(self) -> str:
        """显存归属状态（末次采样）——「ok」/「empty」/ 不可归属原因。"""
        if not self.rows:
            return "未采样"
        return str(self.rows[-1].get("gpu_status", "未知"))

    def _fmt_gpu_peak(self, gpu_peak: Optional[float]) -> str:
        """显存峰值的报告文本：`None` 渲染为「不可归属（原因）」而非 `0 MiB`。"""
        if gpu_peak is None:
            return f"不可归属（{self.gpu_status()}）"
        return f"{gpu_peak:.0f} MiB"

    def n_range(self) -> Tuple[int, int]:
        if not self.rows:
            return 0, 0
        return min(r["n"] for r in self.rows), max(r["n"] for r in self.rows)


# ── 单轮冒烟 ─────────────────────────────────────────────────────────────────
def run_one(ffmpeg: str, ffprobe: str, src: Path, out: Path, rate_mode: str,
            seg_dur: int, extra: List[str], keep: bool,
            mem_interval: int, checks_only: bool = False,
            log_text: str = "", mem_dump: Optional[Path] = None,
            mem_peak_mb: float = 0.0, mem_min_samples: int = 8,
            codec: str = "av1_nvenc", batch_size: int = 0) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"rate_mode": rate_mode, "codec": codec,
                           "batch_size": batch_size,
                           "out": str(out), "checks": []}
    watcher: Optional[MemWatcher] = None
    if checks_only:
        # 复验既有产物：跳过跑批，只跑验收项（GPU 不可用时也能核对历史产物）
        rec["rc"] = 0
        rec["elapsed_s"] = 0.0
        rec["checks_only"] = True
        log_path = Path("")
    else:
        cmd = [sys.executable, "-u", str(MAIN),
               "-c", str(ROOT / "config" / "default_config.json"),
               "-i", str(src), "-o", str(out),
               "--codec-ifrnet", codec, "--codec-esrgan", codec,
               "--rate-mode-ifrnet", rate_mode, "--rate-mode-esrgan", rate_mode,
               "--segment-duration", str(seg_dur)]
        # [S8-BS8] batch_size 显式下发：T4 上 bs=8 推理 FPS 更高，且 S8 内存锯齿
        # 幅度由 pinned result pool（∝ batch_size）驱动 ⇒ 小 bs 让斜率可信。
        if batch_size > 0:
            cmd += ["--batch-size-ifrnet", str(batch_size),
                    "--batch-size-esrgan", str(batch_size)]
        cmd += extra
        rec["cmd"] = " ".join(cmd)
        print(f"\n=== [{rate_mode}] {' '.join(cmd)}")
        log_path = out.with_suffix(out.suffix + f".{rate_mode}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT)
            watcher = MemWatcher(match=str(MAIN), interval=mem_interval,
                                 dump_path=mem_dump)
            watcher.run(proc)
            rc = proc.wait()
        rec["rc"] = rc
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["log"] = str(log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

    def check(cid: str, name: str, ok: Optional[bool], detail: str) -> None:
        rec["checks"].append({"id": cid, "name": name,
                              "status": "SKIP" if ok is None else ("PASS" if ok else "FAIL"),
                              "detail": detail})

    # ① 退出码
    if checks_only:
        check("S1", "管线退出码", None, "--checks-only：未跑批")
    else:
        check("S1", "管线退出码", rc == 0, f"rc={rc}，耗时 {rec['elapsed_s']}s")
        if rc != 0:
            tail = [ln for ln in log_text.splitlines() if ln.strip()][-6:]
            rec["checks"][-1]["detail"] += " | 尾部: " + " / ".join(tail)[:400]
            return rec

    # ② 段级解码级守恒（管线自己的门禁日志）
    # ⚠ [FIX-S3-STAGE-DEDUP] 两阶段管线（Step 1/2 IFRNet + Step 2/2 ESRGAN）**各校验一次**
    #   同一批分段 ⇒ 全量 findall 会把每段计两次（S3 得到 2×产物帧的假 FAIL）。
    #   只取**最后一个阶段**（Step 2/2，即最终产物来源）的分段验收行；单阶段管线无该标记
    #   ⇒ 退回全量（逐字节等价）。
    _s2 = log_text.rfind("Step 2/2")
    seg_region = log_text[_s2:] if _s2 >= 0 else log_text
    seg_ok = re.findall(r"解码级验收通过: decoded=(\d+) expected=(\d+)", seg_region)
    seg_bad = len(re.findall(r"解码级验收失败", seg_region))
    seg_sum = sum(int(a) for a, _ in seg_ok)
    if not seg_ok:
        check("S2", "段级解码级门禁（decoded==expected）", None,
              "无跑批日志可比对（--checks-only 或日志缺失）")
    else:
        check("S2", "段级解码级门禁（decoded==expected）", seg_bad == 0,
              f"{len(seg_ok)} 次通过 / 失败 {seg_bad} 次；分段帧数合计 {seg_sum}")
    rec["segments"] = len(seg_ok)
    rec["segment_frames_sum"] = seg_sum

    # ③ 产物帧数守恒（解码级计数，不是容器元数据）
    frames = probe_int(ffprobe, out, "nb_read_frames")
    src_frames = source_frame_count(ffprobe, src)
    if seg_sum:
        check("S3", "产物可解码帧数 = 各段之和", frames == seg_sum,
              f"产物 {frames} 帧 vs 分段合计 {seg_sum} 帧；源 {src_frames} 帧")
    else:
        check("S3", "产物可解码帧数 = 各段之和", None if frames > 0 else False,
              f"产物 {frames} 帧（无跑批日志，跳过与分段合计的等式核对）；源 {src_frames} 帧")
    rec["frames"] = frames

    # ④ 解码级门禁（复用生产同一函数）
    verdict = "(未执行)"
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from utils.video_utils import validate_decodable_video  # noqa: PLC0415
        ok, rep = validate_decodable_video(str(out), count_mode="decode")
        verdict = (f"ok={ok} reason={rep.get('reason')} frames={rep.get('decoded_frames')} "
                   f"errors={rep.get('decode_errors')} path={rep.get('hwaccel_path')}")
        check("S4", "validate_decodable_video(count_mode=decode)", bool(ok), verdict)
    except Exception as exc:  # noqa: BLE001
        check("S4", "validate_decodable_video(count_mode=decode)", None,
              f"不可用：{type(exc).__name__}: {exc}")
    rec["validate"] = verdict

    # ⑤ 段级码流硬指标（--skip-chroma：色度检查对本素材是内容相关假阳性）
    vrc, vout, _ = run([sys.executable, str(VERIFY5), str(out), "--skip-chroma"], 3600)
    hard = [ln.strip() for ln in vout.splitlines()
            if re.search(r"\[1\]|\[2\]|\[3\]|验收", ln)]
    check("S5", "segment_bitstream_verify_v5（帧守恒/IDR/frame_num/pts）",
          vrc == 0, (hard[-1] if hard else f"rc={vrc}")[:300])

    # ⑥ QA sidecar
    qa = Path(str(out) + ".qa.json")
    if qa.exists():
        try:
            payload = json.loads(qa.read_text(encoding="utf-8"))
            missing = [f for f in QA_REQUIRED_FIELDS if f not in payload]
            ok = not missing and payload.get("codec_hint") == codec
            check("S6", "QA sidecar 字段完整", ok,
                  f"缺字段 {missing or '无'}；codec_hint={payload.get('codec_hint')} "
                  f"rate_mode={payload.get('rate_mode_ifrnet')}")
        except (OSError, ValueError) as exc:
            check("S6", "QA sidecar 字段完整", False, f"解析失败：{exc}")
    else:
        check("S6", "QA sidecar 字段完整", False, f"未生成：{qa}")

    # ⑦ 产物编码器/音轨（确认没被静默换编码器 —— 曾经的缺陷 ②）
    want_codec = EXPECTED_CODEC_NAME.get(codec)
    codec_actual = probe_stream_codec(ffprobe, out)
    audio_note = f"音轨={'有' if has_audio(ffprobe, out) else '无'}"
    if want_codec is None:
        check("S7", f"产物编码器确为 {codec}", None,
              f"codec={codec_actual or '未知'}；无 {codec} 的期望 codec_name 映射，跳过断言"
              f"；{audio_note}")
    else:
        check("S7", f"产物编码器确为 {want_codec}", codec_actual == want_codec,
              f"codec={codec_actual}（期望 {want_codec}）；{audio_note}")

    # ⑧ 内存
    # [FIX-S8-CRITERIA] 判据从「单一斜率」扩为「斜率 + 峰值上界 + 样本充分性」三项：
    #   · 斜率（后半程 RSS OLS，≤ +50 MB/min）——**口径与历史记录一致，未改**，
    #     保证与 2026-09-30 的 +149.5 / −21.4 直接可比；
    #   · 峰值上界（默认 16 GB，可用 --mem-peak-mb 调）——短跑（11 min）里
    #     OLS 斜率对「何时进段 / 段内缓存台阶 / 平台噪声」敏感，峰值是独立
    #     的第二道判据：斜率勉强过线但峰值失控仍应暴露；
    #   · 样本充分性（≥ mem_min_samples 个采样点，默认 8）——样本太少时
    #     斜率不可信，明确报 SKIP 而不是给一个假 PASS/FAIL。
    #   斜率超阈值时 detail 附**主进程 / 子进程分组斜率**（[FIX-S8-ATTRIB]），
    #   使「主进程泄漏 vs ffmpeg 子进程累积」在报告里就能直接读出，无需再上机。
    #
    # [FIX-S8-PEAK-CALIBRATION] 峰值上界默认 **12000 → 16000**（原值是拍脑袋估值，
    # 已被 T4 实测否决）。依据（`verification_report/s8_20261004_raw/`，bs=24、
    # 358.76s 素材 / 720×576 / 12 段，三条**已知无泄漏**的臂）：
    #     h264 constqp 12393 / hevc constqp 13877 / hevc vbr_hq 13578 MB
    # 三者斜率分别为 −55.6 / +3.1 / +35.8 MB/min（全部判定为无泄漏），
    # **却全部超出 12000 上界 ⇒ 12000 只会产生假 FAIL，不具备判别力**。
    # 取 16000 = 最坏实测 13877 × 1.15（留 15% 余量），仍是有限值。
    #   ⚠ 峰值是**结构性锯齿**而非泄漏征兆，实测特征：峰值出现在全程 80~87% 处
    #     （非启动爬坡），全程有 6~10 次 2~3 GB 级别的下跌（段切换清缓存），
    #     峰值/中位数 = 1.17~1.35×，且 **RSS ≈ PSS**（差 <0.3%）⇒ 真实占用，
    #     不是共享页虚高。峰值几乎全部在**主进程**（子进程仅 85~113 MB）。
    #   ⚠ **换素材/分辨率/卡型/段长必须重新标定**：本值只对上述 T4 配置有效。
    #     bs 是关键变量（pinned pool ∝ bs：bs=24 起点 305 MB vs bs=8 的 102 MB），
    #     脚本现默认 bs=8 ⇒ 峰值预计低于上表（外推 ≈13.2 GB，未实测）。
    #   ⚠ **本判据优先级低于斜率**：S8 判读以斜率为准，峰值只在斜率勉强过线时
    #     提供独立佐证；上界失效（误报）时先看 detail 里的分组斜率与 PSS，
    #     **别直接改管线**（见 Plan §9.5 读数判读表第 3、4 行）。
    if watcher is not None and watcher.rows:
        rss_peak, gpu_peak = watcher.peak()
        slope = watcher.slope_mb_per_min()
        br = watcher.slope_breakdown()
        n_lo, n_hi = watcher.n_range()
        rec["rss_peak_mb"] = round(rss_peak, 1)
        rec["gpu_peak_mib"] = None if gpu_peak is None else round(gpu_peak, 1)
        rec["gpu_status"] = watcher.gpu_status()
        rec["rss_slope_mb_per_min"] = None if slope is None else round(slope, 1)
        rec["rss_slope_main_mb_per_min"] = (None if br["rss_main"] is None
                                            else round(br["rss_main"], 1))
        rec["rss_slope_child_mb_per_min"] = (None if br["rss_child"] is None
                                             else round(br["rss_child"], 1))
        rec["pss_slope_mb_per_min"] = (None if br["pss_all"] is None
                                       else round(br["pss_all"], 1))
        rec["proc_count_range"] = [n_lo, n_hi]
        rec["mem_samples"] = len(watcher.rows)
        if mem_dump:
            rec["mem_dump"] = str(mem_dump)

        n_s = len(watcher.rows)
        if slope is None or n_s < mem_min_samples:
            check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）", None,
                  f"样本不足（{n_s} 点 < 门槛 {mem_min_samples}，"
                  f"RSS 峰值 {rss_peak:.0f} MB）——提高 --mem-interval 采样密度或延长素材")
        else:
            attrib = (f"主 {br['rss_main']:+.1f} / 子 {br['rss_child']:+.1f} MB/min"
                      if br["rss_main"] is not None and br["rss_child"] is not None
                      else "分组斜率不可用")
            peak_ok = rss_peak <= mem_peak_mb if mem_peak_mb > 0 else True
            peak_note = ("" if mem_peak_mb <= 0 else
                         f"，峰值 {rss_peak:.0f}/{mem_peak_mb:.0f} MB "
                         f"{'✓' if peak_ok else '✗ 超上界'}")
            check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）",
                  bool(slope <= 50.0 and peak_ok),
                  f"RSS 峰值 {rss_peak:.0f} MB，斜率 {slope:+.1f} MB/min"
                  f"（{attrib}；PSS {br['pss_all']:+.1f}），显存峰值 "
                  f"{watcher._fmt_gpu_peak(gpu_peak)}，"
                  f"进程数 {n_lo}~{n_hi}，样本 {n_s}{peak_note}；"
                  f"明细 {rec.get('mem_dump', '未落盘')}")
    else:
        check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）", None, "未采样（--checks-only）")

    if not keep and not checks_only:
        for f in out.parent.glob(out.name + "*"):
            if f.suffix in (".mp4", ".mkv", ".mov"):
                f.unlink(missing_ok=True)
    return rec


# ── 报告 ─────────────────────────────────────────────────────────────────────
def render_md(result: Dict[str, Any]) -> str:
    L = ["# NVENC 端到端冒烟 + 验收报告", "",
         f"- 生成时间：{result['generated']}",
         f"- 素材：`{result['src']}`",
         f"- 编码器：`{result['codec']}`",
         f"- batch_size：{result.get('batch_size') or '沿用 config（默认 24）'}",
         f"- rate_mode：{', '.join(result['rate_modes'])}",
         f"- 主机：{result['host']}",
         f"- 环境前置：{result['precondition']}",
         "", f"**汇总：PASS={result['summary']['pass']} / FAIL={result['summary']['fail']}"
         f" / SKIP={result['summary']['skip']}**", ""]
    for rec in result["runs"]:
        L += [f"## rate_mode = {rec['rate_mode']}", "",
              f"- 退出码：{rec['rc']}（耗时 {rec['elapsed_s']}s）",
              f"- 命令：`{rec.get('cmd', '（--checks-only，未跑批）')}`", "",
              "| 项 | 结论 | 说明 |", "|---|:--:|---|"]
        for c in rec["checks"]:
            L.append(f"| {c['id']} {c['name']} | {c['status']} | {c['detail']} |")
        L.append("")
        if rec.get("mem_samples"):
            L += [f"- 内存采样：{rec['mem_samples']} 点，进程数 "
                  f"{rec.get('proc_count_range', ['?', '?'])[0]}~"
                  f"{rec.get('proc_count_range', ['?', '?'])[1]}，"
                  f"RSS 峰值 {rec.get('rss_peak_mb')} MB，"
                  f"本进程树显存峰值 "
                  f"{rec.get('gpu_peak_mib') if rec.get('gpu_peak_mib') is not None else '不可归属'}"
                  f"（{rec.get('gpu_status', '未知')}）",
                  f"- 斜率（全树 / 主进程 / 子进程 / PSS）："
                  f"{rec.get('rss_slope_mb_per_min')} / {rec.get('rss_slope_main_mb_per_min')} / "
                  f"{rec.get('rss_slope_child_mb_per_min')} / {rec.get('pss_slope_mb_per_min')} MB/min",
                  f"- 逐进程明细：`{rec.get('mem_dump', '未落盘')}`", ""]
    L += ["> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**",
          "> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。", ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="NVENC 端到端冒烟 + 验收（constqp / vbr 双 rate_mode）")
    ap.add_argument("--codec", default="av1_nvenc",
                    help="两侧编码器（默认 av1_nvenc；T4 上复现 S8 机制用 h264_nvenc/hevc_nvenc）")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="两阶段 batch_size（默认 8：T4 上推理 FPS 更高，且 S8 内存锯齿幅度 "
                         "∝ pinned result pool ∝ batch_size；0 = 沿用 config 的 24）")
    ap.add_argument("--src", required=True, help="真实素材（建议 ≥1 min，含音轨）")
    ap.add_argument("--out-dir", help="产物目录（默认 <src 同级>/<codec>_smoke_out）")
    ap.add_argument("--rate-modes", default="constqp,vbr_hq",
                    help="逗号分隔（默认 constqp,vbr_hq）。"
                         "非AV1 的 NVENC 编码器请用 vbr_hq：vbr/cbr 在Level 1 SDK "
                         "未实现（落 CONSTQP 且 LA 不生效 ⇒ 与 constqp 臂同路径，"
                         "A/B 空转），传了会被 [FIX-B2-ARMS] 提前拒；"
                         "av1_nvenc 保留 vbr/cbr（那是 AV1 的合法 RC）")
    ap.add_argument("--segment-duration", type=int, default=30, help="分段秒数（默认 30）")
    ap.add_argument("--timeout", type=int, default=7200, help="单轮超时秒（默认 7200）")
    ap.add_argument("--mem-interval", type=int, default=10, help="内存采样间隔秒（默认 10）")
    ap.add_argument("--mem-peak-mb", type=float, default=16000.0,
                    help="S8 RSS 峰值上界 MB（默认 16000，按 T4 实测重标定，见下注；"
                         "0 = 不判峰值只看斜率）")
    ap.add_argument("--mem-min-samples", type=int, default=8,
                    help="S8 斜率所需最少采样点（默认 8，不足则报 SKIP）")
    ap.add_argument("--mem-dump-dir",
                    help="内存采样明细落盘目录（每轮一个 <rate_mode>.mem.tsv，含逐进程明细）")
    ap.add_argument("--keep", action="store_true", help="保留产物（默认跑完删除视频）")
    ap.add_argument("--checks-only", metavar="VIDEO", default="",
                    help="只对既有产物跑验收项（不跑批、不需要 GPU）")
    ap.add_argument("--report", help="Markdown 报告路径")
    ap.add_argument("--json", dest="json_path", help="JSON 结果路径")
    ap.add_argument("extra", nargs="*", help="透传给主入口的额外参数")
    args = ap.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg") or ""
    ffprobe = shutil.which("ffprobe") or ""
    if not ffmpeg or not ffprobe:
        print("❌ 缺少 ffmpeg/ffprobe")
        return 2
    src = Path(args.src).resolve()
    if not src.exists():
        print(f"❌ 素材不存在：{src}")
        return 2

    # [FIX-B2-ARMS] 非 AV1 的 NVENC 编码器禁用 vbr/cbr 对照臂（提前拦截，见模块 docstring）。
    # 主入口 `main_video_optimized.py [FIX-B2-VBR-CBR-REJECT]` 会拒绝启动，但那要等
    # 模型加载完才报错（白等数分钟）⇒ 在此提前 exit 2。
    if "nvenc" in args.codec and "av1" not in args.codec:
        _bad = [r.strip() for r in args.rate_modes.split(",")
                if r.strip() in ("vbr", "cbr")]
        if _bad:
            print(f"❌ --rate-modes 含 {_bad}，但 codec={args.codec} 在 NVENC SDK 直通上"
                  f"未实现该档位（会静默落 CONSTQP ⇒ 与 constqp 臂同路径，A/B 失效）。")
            print(f"   请改用 vbr_hq（质量优先）或 qvbr，例如："
                  f"--rate-modes constqp,vbr_hq")
            print(f"   （av1_nvenc 保留 vbr/cbr：那是 AV1 的合法 RC）")
            return 2

    ok_codec, why = encoder_available(ffmpeg, args.codec)
    print("=" * 78)
    print(f"  {args.codec} 端到端冒烟 + 验收")
    print("=" * 78)
    print(f"素材: {src}")
    if args.checks_only:
        target = Path(args.checks_only).resolve()
        if not target.exists():
            print(f"❌ --checks-only 目标不存在：{target}")
            return 2
        print(f"模式: --checks-only（只验收既有产物，不跑批、不需要 GPU）→ {target}")
        rec = run_one(ffmpeg, ffprobe, src, target,
                      args.rate_modes.split(",")[0].strip(), args.segment_duration,
                      [], True, args.mem_interval, checks_only=True,
                      codec=args.codec, batch_size=args.batch_size)
        runs = [rec]
        for c in rec["checks"]:
            mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[c["status"]]
            print(f"  {mark} [{c['id']}] {c['name']} — {c['detail']}")
    else:
        print(f"前置: {args.codec} {'✅' if ok_codec else '⏭️ 不可用'} — {why}")
        if not ok_codec:
            print(f"⏭️ 环境前置不成立（{args.codec} 实跑一帧失败）⇒ exit 2")
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else src.parent / f"{args.codec}_smoke_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for rate in [r.strip() for r in args.rate_modes.split(",") if r.strip()]:
            out = out_dir / f"{args.codec}_{rate}{src.suffix or '.mp4'}"
            dump = (Path(args.mem_dump_dir) / f"{rate}.mem.tsv"
                    if args.mem_dump_dir else None)
            rec = run_one(ffmpeg, ffprobe, src, out, rate, args.segment_duration,
                          list(args.extra), args.keep, args.mem_interval,
                          mem_dump=dump, mem_peak_mb=args.mem_peak_mb,
                          mem_min_samples=args.mem_min_samples, codec=args.codec,
                          batch_size=args.batch_size)
            runs.append(rec)
            for c in rec["checks"]:
                mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[c["status"]]
                print(f"  {mark} [{c['id']}] {c['name']} — {c['detail']}")

    n_pass = sum(1 for r in runs for c in r["checks"] if c["status"] == "PASS")
    n_fail = sum(1 for r in runs for c in r["checks"] if c["status"] == "FAIL")
    n_skip = sum(1 for r in runs for c in r["checks"] if c["status"] == "SKIP")
    print("\n" + "=" * 78)
    print(f"  汇总：PASS={n_pass} / FAIL={n_fail} / SKIP={n_skip}")
    print("=" * 78)

    result = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "src": str(src), "rate_modes": args.rate_modes.split(","),
        "codec": args.codec,
        "batch_size": args.batch_size,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "precondition": why, "runs": runs,
        "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
    }
    md = render_md(result)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md, encoding="utf-8")
        print(f"📄 报告: {args.report}")
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        print(f"🧾 JSON: {args.json_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
