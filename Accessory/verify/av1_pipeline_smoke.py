#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AV1 NVENC 端到端冒烟 + 验收（constqp / vbr_hq 双 rate_mode）—— 可重复执行入口。

为什么需要它
------------
2026-09-30 在 L40 上第一次跑「长视频 AV1 冒烟」时**开箱即失败**，一次性暴露三处
AV1 路径缺陷（方案 §8.6）：cv2 验收层把完好 AV1 产物判成损坏、ESRGan 侧 writer
静默改用 libx264、AV1 的 `vbr_hq` 未降级导致命令失败。三者都**只在下发真实命令
并解码产物时**才暴露 —— 此前 AC1~AC7 全是「下发单条 ffmpeg 命令」的微观测试。
本脚本把那次一次性验证固化成一条命令，供每次改动 AV1 路径后复跑。

    python3 Accessory/verify/av1_pipeline_smoke.py --src <真实素材> < /dev/null
    python3 Accessory/verify/av1_pipeline_smoke.py --src <真实素材> \
        --codec h264_nvenc --rate-modes constqp,vbr_hq < /dev/null  # T4 上复现 S8 机制

覆盖
----
| 阶段 | 内容 |
|---|---|
| 前置 | 目标编码器（默认 `av1_nvenc`）**实跑一帧**（不是 `-h encoder=`）；不可用 ⇒ exit 2（环境前置不成立） |
| 跑批 | 对每个 rate_mode 调 `src/main_video_optimized.py`，两侧 `--codec-* <CODEC>` |
| 采样 | 整棵进程树 RSS/PSS + **主进程 vs 子进程分组斜率** + 按 pid 归属的 GPU 显存（判泄漏：后半程斜率 + 峰值上界）|
| 验收 | ① 退出码 ② 段级日志 `decoded == expected` ③ 产物 `ffprobe -count_frames` 帧数守恒 ④ `validate_decodable_video(count_mode='decode')` ⑤ `segment_bitstream_verify_v5 --skip-chroma` 硬指标 ⑥ QA sidecar 字段完整 |

⚠ **必须 `< /dev/null`**：这些脚本在「后台进程组 + tty stdin」下会被 SIGTTOU 整组停住。
⚠ 色度检查（检查 4）在真实素材上是**内容相关假阳性**（方案 §8.5：AV1 长片 113 簇，
   而源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。

`--codec`（S8 机制复现用）
------------------------
`constqp` 强制 `LA=0`（硬件静默禁用）⇒ 走 `encode_frames_batch_ce_pipeline`；
其它 rate_mode 保留配置 LA ⇒ 走 `encode_frames_stream` 分块累积。**两条路径与编码器无关**
（`main.py` 的 Level 1 直通判据是 `'nvenc' in use_codec`），故 **T4 上用 `h264_nvenc`/
`hevc_nvenc` 跑 constqp vs vbr_hq，可复现 AV1/L40 上 S8 观测到的同一机制**（方案 §9.5 D 项：
降本验证，省掉等 L40）。Turing 无 AV1 NVENC ⇒ 本机须显式 `--codec h264_nvenc`。

⚠ **对照臂用 `vbr_hq` 而不是 `vbr`**（[FIX-B2-RC-MODE-REJECT]，2026-10-04）：
`vbr` / `cbr` 在 **SDK 直通层未实现** —— `nvenc_sdk._build_encoder_config()` 只分支
constqp / vbr_hq / qvbr，传 `vbr` 会**静默**落 CONSTQP（LA 被硬件禁用而代码仍认为 LA=8，
且绕过 per-frame completionEvent）。该缺陷已在两侧 `nvenc_sdk.__init__` 改为**显式 raise**，
故本脚本默认对照臂同步改为 `vbr_hq`（生产默认值，唯一真 VBR_HQ + LA>0 路径）。

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
    """本机各进程占用的显存（MiB），**按 pid 归属**。整卡 `memory.used` 在共享 GPU
    主机上会被其他会话污染（本仓 memory 有实证），必须按 pid 归属后再由调用方筛进程树。

    [FIX-B3-GPU-UNATTRIBUTABLE] 返回 `(by_pid, status)`：
      · `("ok", dict)`         —— 可归属，dict 非空；
      · `("pid_ns_mismatch",)` —— **nvidia-smi 报的是宿主 pid，容器 `/proc` 下不存在**
        ⇒ 容器内按 pid 归属**结构性不可行**，必须报「测不到」；
      · `("no_smi"|"query_failed"|"empty",)` —— 同样测不到。

    ⚠ **不可归属时必须报 None 而不是 0.0**：0.0 会被读成「确实没用显存」。
    2026-10-04 T4 实测反证 —— 跑批时整卡 7698 MiB / 100% 利用率，而按 pid 归属恒为 0
    （`--query-compute-apps` 返回宿主 pid 725115 / 1071006，容器 `/proc` 下查不到）。
    这与 GPU 是否挂载**无关**，旧归因「本容器无 GPU 挂载 ⇒ 显存记 0」已被推翻。
    """
    if shutil.which("nvidia-smi") is None:
        return None, "no_smi"
    for field in ("used_gpu_memory", "used_memory"):      # 不同 driver/版本字段名不同
        try:
            p = subprocess.run(
                ["nvidia-smi", f"--query-compute-apps=pid,{field}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        out = (p.stdout or "").strip()
        if p.returncode != 0:
            continue
        if not out:
            return None, "empty"
        got: Dict[int, float] = {}
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                got[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
        if not got:
            return None, "empty"
        if not any(Path(f"/proc/{pid}").exists() for pid in got):
            return None, "pid_ns_mismatch"       # 宿主 pid，容器 /proc 下不存在
        return got, "ok"
    return None, "query_failed"


def _gpu_total_mib() -> float:
    """整卡已用显存（MiB）——只作参考，不作判据（见 _gpu_mem_by_pid 注释）。"""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        return float(q.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0.0


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
        ——共享 GPU 主机上整卡值会把他人的任务算进来；
        **[FIX-B3-GPU-UNATTRIBUTABLE]** 归属不可行时（容器 PID namespace 与驱动侧
        不通，`--query-compute-apps` 返回宿主 pid）报 **None**（"测不到"）而非 0.0，
        TSV 落盘写 `NA` —— 0.0 会被读成「确实没用显存」。
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
            # [FIX-B3-GPU-UNATTRIBUTABLE] None = 测不到（不可归属），0.0 = 确实为 0
            "gpu_tree_mib": gpu_tree,
            "gpu_attrib": gpu_status,
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
                        # [FIX-B3-GPU-UNATTRIBUTABLE] 不可归属写 "NA"，不写 0.0
                        gpu_col = ("NA" if s["gpu_tree_mib"] is None
                                   else f"{s['gpu_tree_mib']:.1f}")
                        self._dump_fh.write(
                            f"{t:.1f}\t{s['n']}\t{s['rss_mb']:.1f}\t{s['pss_mb']:.1f}\t"
                            f"{s['rss_main_mb']:.1f}\t{s['rss_child_mb']:.1f}\t"
                            f"{gpu_col}\t"
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

        [FIX-B3-GPU-UNATTRIBUTABLE] 不可归属时第二项为 **None**（不是 0.0）。
        """
        if not self.rows:
            return 0.0, None
        vals = [r["gpu_tree_mib"] for r in self.rows
                if r.get("gpu_tree_mib") is not None]
        return max(r["rss_mb"] for r in self.rows), (max(vals) if vals else None)

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
    #   · 峰值上界（[FIX-S8-PEAK-BS-AWARE] 按 batch_size 自动标定，bs=8 基准
    #     7694 MB × bs/8 × 1.25 余量；旧的固定 12000 是 bs=24 下的估值，实测
    #     bs=24 三臂峰值 12393~13877 **全超上界而斜率正常** ⇒ 必然误报）——短跑里
    #     OLS 斜率对「何时进段 / 段内缓存台阶 / 平台噪声」敏感，峰值是独立
    #     的第二道判据：斜率勉强过线但峰值失控仍应暴露；
    #   · 样本充分性（≥ mem_min_samples 个采样点，默认 8）——样本太少时
    #     斜率不可信，明确报 SKIP 而不是给一个假 PASS/FAIL。
    #   斜率超阈值时 detail 附**主进程 / 子进程分组斜率**（[FIX-S8-ATTRIB]），
    #   使「主进程泄漏 vs ffmpeg 子进程累积」在报告里就能直接读出，无需再上机。
    if watcher is not None and watcher.rows:
        rss_peak, gpu_peak = watcher.peak()
        slope = watcher.slope_mb_per_min()
        br = watcher.slope_breakdown()
        n_lo, n_hi = watcher.n_range()
        rec["rss_peak_mb"] = round(rss_peak, 1)
        # [FIX-B3-GPU-UNATTRIBUTABLE] None = 不可归属（容器 PID namespace 不通等）
        rec["gpu_peak_mib"] = None if gpu_peak is None else round(gpu_peak, 1)
        rec["gpu_attrib"] = watcher.rows[-1].get("gpu_attrib")
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
            # [FIX-B3-GPU-UNATTRIBUTABLE] 不可归属时明说「测不到」，不报 0 MiB
            gpu_note = (f"{gpu_peak:.0f} MiB" if gpu_peak is not None else
                        f"不可归属（{rec.get('gpu_attrib')}）")
            check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）",
                  bool(slope <= 50.0 and peak_ok),
                  f"RSS 峰值 {rss_peak:.0f} MB，斜率 {slope:+.1f} MB/min"
                  f"（{attrib}；PSS {br['pss_all']:+.1f}），显存峰值 {gpu_note}，"
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
            # [FIX-B3-GPU-UNATTRIBUTABLE] 不可归属写「不可归属（原因）」而非 0 MiB
            gpu_txt = (f"{rec.get('gpu_peak_mib')} MiB"
                       if rec.get("gpu_peak_mib") is not None
                       else f"不可归属（{rec.get('gpu_attrib')}，非 0）")
            L += [f"- 内存采样：{rec['mem_samples']} 点，进程数 "
                  f"{rec.get('proc_count_range', ['?', '?'])[0]}~"
                  f"{rec.get('proc_count_range', ['?', '?'])[1]}，"
                  f"RSS 峰值 {rec.get('rss_peak_mb')} MB，"
                  f"本进程树显存峰值 {gpu_txt}",
                  f"- 斜率（全树 / 主进程 / 子进程 / PSS）："
                  f"{rec.get('rss_slope_mb_per_min')} / {rec.get('rss_slope_main_mb_per_min')} / "
                  f"{rec.get('rss_slope_child_mb_per_min')} / {rec.get('pss_slope_mb_per_min')} MB/min",
                  f"- 逐进程明细：`{rec.get('mem_dump', '未落盘')}`", ""]
    L += ["> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**",
          "> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。", ""]
    return "\n".join(L)


def _auto_peak_mb(batch_size: int) -> float:
    """[FIX-S8-PEAK-BS-AWARE] 按 batch_size 标定 S8 峰值上界。

    旧默认是一个常数 12000 MB，那是 **bs=24 下估的值**，实测已被否决：
    bs=24 的三臂峰值 12393/ 13578 / 13877 MB **全都超上界而斜率正常**
    ⇒ 峰值判据在 bs=24 下必然误报。

    实测峰值随 bs 近似线性（标准素材 `01 the race to mystery island.fixed.mp4`
    358.76s / 12 段 / 720x576，T4）：

    | bs | constqp | vbr_hq | hevc constqp | hevc vbrhq |
    |---|---|---|---|---|
    | 8 | 6730 | 7694 | — | — |
    | 24 | 12393 | — | 13877 | 13578 |

    ⇒ 以 bs=8 实测上界 7694 MB 为基准，按 `bs/8` 线性外推，再留**余量**。
    固定 bs（冒烟默认 8）下该值不会误报；换 bs 时自动跟随。
    ⚠ **仍不能跨素材通用**（素材分辨率/时长影响工作集）—— 换素材时若 S8 报
    「峰值超上界」而斜率正常，应先确认是不是该重新标定，而非直接判定失控。
    """
    _BASE_BS, _BASE_PEAK_MB = 8, 7694.0
    return max(2000.0, _BASE_PEAK_MB * max(1, batch_size) / _BASE_BS * 1.25)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="NVENC 端到端冒烟 + 验收（constqp / vbr_hq 双 rate_mode）")
    ap.add_argument("--codec", default="av1_nvenc",
                    help="两侧编码器（默认 av1_nvenc；T4 上复现 S8 机制用 h264_nvenc/hevc_nvenc）")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="两阶段 batch_size（默认 8：T4 上推理 FPS 更高，且 S8 内存锯齿幅度 "
                         "∝ pinned result pool ∝ batch_size；0 = 沿用 config 的 24）")
    ap.add_argument("--src", required=True, help="真实素材（建议 ≥1 min，含音轨）")
    ap.add_argument("--out-dir", help="产物目录（默认 <src 同级>/<codec>_smoke_out）")
    ap.add_argument("--rate-modes", default="constqp,vbr_hq",
                    help="逗号分隔（默认 constqp,vbr_hq）。⚠ 对照臂必须用 vbr_hq 而非 vbr：\n"
                         "SDK 直通层未实现 vbr/cbr，传了会被 [FIX-B2-RC-MODE-REJECT] 拒绝")
    ap.add_argument("--segment-duration", type=int, default=30, help="分段秒数（默认 30）")
    ap.add_argument("--timeout", type=int, default=7200, help="单轮超时秒（默认 7200）")
    ap.add_argument("--mem-interval", type=int, default=10, help="内存采样间隔秒（默认 10）")
    ap.add_argument("--mem-peak-mb", type=float, default=0.0,
                    help="S8 RSS 峰值上界 MB；0 = 按 --batch-size 自动标定（推荐，"
                         "见 [FIX-S8-PEAK-BS-AWARE]）；正数 = 手动覆盖（会失去 bs 自适应）")
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
    # [FIX-S8-PEAK-BS-AWARE] 峰值上界随 bs 标定（--mem-peak-mb 显式给值则不自动）
    peak_mb = args.mem_peak_mb if args.mem_peak_mb > 0 else _auto_peak_mb(args.batch_size)
    print(f"S8 峰值上界: {peak_mb:.0f} MB"
          + ("（手动指定 --mem-peak-mb）" if args.mem_peak_mb > 0
             else f"（按 bs={args.batch_size} 自动标定）"))

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
                          mem_dump=dump, mem_peak_mb=peak_mb,
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
