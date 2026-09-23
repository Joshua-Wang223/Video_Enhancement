#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[DIAG-READER-RGB] NVDEC 硬解 vs CPU 软解：`nv12→rgb24` 为何产出不同 RGB。

立项：`Plan/NVDEC与软解RGB一致性_立项Prompt.md`
（本脚本是该立项 §3「建议的实施步骤」的可执行版本；立项文档里所有数字来自
 2026-09-14/15 的 Tesla T4 会话，**接手方必须先在本机复跑再下结论**。）

## 它做什么

按立项文档的四个判据逐条跑**可判定的实验**，并把结论落成 JSON + Markdown：

| 判据 | 实验 | 期望 |
|---|---|---|
| 1 | 两条路径各导出 `yuv420p`，**逐平面**比对 | Y/UV 平面 0 差异 ⇒ 差异 100% 在 `nv12→rgb24` |
| 1 | 两条路径各导出 `rgb24`，逐像素比对 | 有差异（幅度/占比以本环境实测为准） |
| 2 | reader 产出 vs **同参数**参考解码 | 0 不一致 ⇒ reader 对其自身契约是忠实的 |
| 3 | 用**输入侧颜色元数据覆盖**穷举「矩阵 × 范围」，找出与硬件路径**逐字节相等**的那组 | 命中 ⇒ 钉住硬件路径实际用的 range/matrix（`--sweep`） |
| 4 | 输出「对齐 / 声明不对齐」的判定模板 | 由人依据上面证据填写 |

## 关键设计（为什么这样写）

* **两条路径锁步流式比对**，不落盘：578 帧 × 640×360 rgb24 ≈ 400MB，落盘既慢
  又会掩盖"边读边比"的真实行为。
* **所有 ffmpeg 子进程都带 `stdin=DEVNULL`**：本仓库实测过「后台进程组里 ffmpeg
  启动阶段对 fd0 调 ioctl(TCSETS) 被 SIGTTOU 停住 → 秒卡 + 0% CPU + 无输出」，
  现象极像码流/GPU 故障（见 src/utils/stdin_hardening.py）。
* **硬件路径必须点名探测，不能只看 `-hwaccel auto` 的返回码**：实测在**没有
  NVIDIA GPU** 的 Windows 机器上 `-hwaccel auto` 会静默退到 `dxva2` 并返回 rc=0
  （假阳性），而 `-hwaccel cuda` 才正确报 `Device creation failed`。
  故本脚本逐个显式试 `cuda/vaapi/d3d11va/dxva2/qsv` 并检查 stderr 失败标记，
  且**只有当可用后端确实是 `cuda` 时才称其为 NVDEC**。
* **判据 3 用输入侧选项而不是 `zscale`**：立项 §3 假设"必须换带 libzimg 的
  build 才能控制矩阵/范围"，实测 `-color_range` / `-colorspace` 放在 `-i` 之前
  **就能真实改变 RGB 输出**（见 `_sweep_candidates` 的 docstring 有哈希证据），
  因此任何 ffmpeg build 都能跑，不必先解决环境。

## 用法

    # 最小复测（判据 1 + 2）
    python tests/diagnose_reader_rgb_path_consistency.py CLIP.mp4

    # 加参数穷举（判据 3，需带 libzimg 的 ffmpeg：zscale 滤镜）
    python tests/diagnose_reader_rgb_path_consistency.py CLIP.mp4 --sweep

    # 落报告
    ... --json out.json --report out.md

    # 冻结路径做 A/B 对照时（立项文档 §2.5 的坑）
    READER_HWACCEL=on  ...   # 强制 NVDEC
    READER_HWACCEL=off ...   # 强制软解

不需要 GPU 的是「脚本自身的语法/流程」与判据 1/2 的**软解臂**；
**判据 3 与 NVDEC 臂必须有 NVIDIA GPU**。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_SAFE = {"stdin": subprocess.DEVNULL}          # 见模块 docstring「关键设计」
PIX_FMT_RGB = "rgb24"
PLANE_ORDER = ("Y", "U", "V")


# ──────────────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────────────
def _run(cmd: List[str], timeout: int = 120) -> Tuple[int, bytes, bytes]:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, **_SAFE)
    return p.returncode, p.stdout, p.stderr


def _ffprobe_frames(clip: str, ffprobe: str = "ffprobe") -> Optional[int]:
    rc, out, _ = _run([ffprobe, "-v", "error", "-select_streams", "v:0",
                       "-count_frames", "-show_entries", "stream=nb_read_frames",
                       "-of", "csv=p=0", clip])
    if rc != 0:
        return None
    try:
        return int(out.decode().strip())
    except ValueError:
        return None


def _ffprobe_meta(clip: str, ffprobe: str = "ffprobe") -> Dict[str, object]:
    rc, out, _ = _run([ffprobe, "-v", "error", "-select_streams", "v:0",
                       "-show_entries",
                       "stream=width,height,pix_fmt,color_range,color_space,"
                       "color_primaries,color_transfer",
                       "-of", "json", clip])
    if rc != 0:
        return {}
    try:
        return (json.loads(out.decode()).get("streams") or [{}])[0]
    except Exception:
        return {}


def _has_filter(name: str, ffmpeg: str = "ffmpeg") -> bool:
    rc, out, _ = _run([ffmpeg, "-hide_banner", "-filters"])
    if rc != 0:
        return False
    text = out.decode(errors="replace")
    return any(line.split()[1:2] == [name] for line in text.splitlines()
               if len(line.split()) > 1)


def _hwaccels(ffmpeg: str = "ffmpeg") -> List[str]:
    rc, out, _ = _run([ffmpeg, "-hide_banner", "-hwaccels"])
    if rc != 0:
        return []
    return [l.strip() for l in out.decode(errors="replace").splitlines()
            if l.strip() and not l.endswith(":")]


def _decode_cmd(clip: str, pix_fmt: str, hwaccel: Optional[str],
                vf: Optional[str] = None, ffmpeg: str = "ffmpeg",
                in_opts: Tuple[str, ...] = ()) -> List[str]:
    """`hwaccel=None` → 纯软解；否则 `-hwaccel <hwaccel>`。

    `in_opts` 是**输入侧选项**，必须出现在 `-i` 之前才生效
    （本仓库已有同类教训：`-fflags +genpts` 放在 `-i` 后无效）。
    """
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-noautorotate"]
    cmd += list(in_opts)
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", clip]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-an", "-fps_mode", "passthrough", "-f", "rawvideo",
            "-pix_fmt", pix_fmt, "pipe:1"]
    return cmd


def _plane_sizes(w: int, h: int, pix_fmt: str) -> Tuple[int, ...]:
    if pix_fmt == PIX_FMT_RGB:
        return (w * h * 3,)
    if pix_fmt == "yuv420p":
        cw, ch = (w + 1) // 2, (h + 1) // 2
        return (w * h, cw * ch, cw * ch)
    raise ValueError("未支持的 pix_fmt: %s" % pix_fmt)


_HW_FAIL_MARKS = ("Device creation failed", "No device available",
                  "Hardware device setup failed", "not compiled",
                  "Cannot load", "Failed to find")


def _detect_hwaccel(clip: str, ffmpeg: str,
                    requested: str = "auto") -> Tuple[Optional[str], str]:
    """**逐个显式试**，返回 (真正可用的 hwaccel 名 or None, 说明)。

    为什么不能只看 `-hwaccels` 列表、也不能只看 `-hwaccel auto` 的返回码：

      · `-hwaccels` 列出的是**编译进来的后端**，与"本机有没有对应硬件"无关；
      · `-hwaccel auto` 在**没有 NVIDIA GPU** 的机器上会静默退到别的后端
        （Windows 实测退到 `dxva2`，由 WARP 软渲染支撑）并返回 **rc=0** ——
        于是 `auto` 会给出"NVDEC 可用"的**假阳性**，让两条路径的比对结论
        与 NVDEC 立项完全无关。

    实测（2026-09-15，Windows 无 GPU）：
      `-hwaccel cuda`  → `Device creation failed: -1`（正确判为不可用）
      `-hwaccel auto`  → `Using auto hwaccel type dxva2`（rc=0，假阳性）
    故这里改为**点名逐个试 + 检查 stderr 的失败标记**。
    """
    if requested and requested != "auto":
        cands = [requested]
    else:
        # cuda 排第一：本立项关心的就是 NVDEC。
        cands = ["cuda", "vaapi", "d3d11va", "dxva2", "qsv", "auto"]
    notes = []
    for name in cands:
        p = subprocess.run(
            [ffmpeg, "-hide_banner", "-v", "info", "-noautorotate",
             "-hwaccel", name, "-i", clip, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=300, **_SAFE)
        err = p.stderr.decode(errors="replace")
        bad = [m for m in _HW_FAIL_MARKS if m in err]
        if p.returncode == 0 and not bad:
            used = name
            if name == "auto":
                for line in err.splitlines():
                    if "Using auto hwaccel type" in line:
                        used = line.split("Using auto hwaccel type")[1].strip()
                        used = used.split()[0] if used.split() else "auto"
                        break
            notes.append("%s: rc=0 可用%s"
                         % (name, "（auto 实际选中 %s）" % used
                            if name == "auto" else ""))
            return used, "; ".join(notes)
        notes.append("%s: rc=%d %s" % (name, p.returncode,
                                       ("拒绝(%s)" % bad[0]) if bad else "失败"))
    return None, "; ".join(notes)



# ──────────────────────────────────────────────────────────────────────────
# 锁步流式比对
# ──────────────────────────────────────────────────────────────────────────
class _Lockstep:
    """同时喂两个 ffmpeg 管道，按帧读满即比，比完继续 —— 全程不落盘。"""

    def __init__(self, cmd_a: List[str], cmd_b: List[str], frame_bytes: int,
                 plane_sizes: Tuple[int, ...]):
        self.frame_bytes = frame_bytes
        self.plane_sizes = plane_sizes
        self.pa = subprocess.Popen(cmd_a, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, **_SAFE)
        self.pb = subprocess.Popen(cmd_b, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, **_SAFE)

    def _read_frame(self, p) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < self.frame_bytes:
            chunk = p.stdout.read(self.frame_bytes - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf) if len(buf) == self.frame_bytes else None

    def compare(self, max_frames: int = 0):
        """返回 (frames, per_plane_stats) 或抛 RuntimeError（帧数不齐）。"""
        stats = [{"max_diff": 0, "diff_pixels": 0, "total_bytes": 0,
                  "sum_diff": 0} for _ in self.plane_sizes]
        n = 0
        while True:
            fa = self._read_frame(self.pa)
            fb = self._read_frame(self.pb)
            if fa is None and fb is None:
                break
            if fa is None or fb is None:
                raise RuntimeError(
                    "两条路径帧数不齐：在读到第 %d 帧时一侧提前 EOF" % n)
            if fa != fb:
                off = 0
                for i, size in enumerate(self.plane_sizes):
                    sa = fa[off:off + size]
                    sb = fb[off:off + size]
                    s = stats[i]
                    s["total_bytes"] += size
                    for x, y in zip(sa, sb):
                        d = abs(x - y)
                        if d:
                            s["diff_pixels"] += 1
                            s["sum_diff"] += d
                            if d > s["max_diff"]:
                                s["max_diff"] = d
                    off += size
            else:
                for i, size in enumerate(self.plane_sizes):
                    stats[i]["total_bytes"] += size
            n += 1
            if max_frames and n >= max_frames:
                break
        return n, stats

    def close(self):
        errs = []
        for p in (self.pa, self.pb):
            try:
                if p.stdout:
                    p.stdout.close()
            except Exception:
                pass
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            try:
                e = (p.stderr.read() or b"").decode(errors="replace").strip()
                if e:
                    errs.append(e[:400])
            except Exception:
                pass
            try:
                p.stderr.close()
            except Exception:
                pass
        return errs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _pct(s: Dict[str, int]) -> float:
    return (100.0 * s["diff_pixels"] / s["total_bytes"]) if s["total_bytes"] else 0.0


# ──────────────────────────────────────────────────────────────────────────
# 判据 1：平面级 + RGB 级
# ──────────────────────────────────────────────────────────────────────────
def check_planes(clip, w, h, hw, max_frames, ffmpeg="ffmpeg",
                 ffprobe="ffprobe") -> Dict[str, object]:
    fb = sum(_plane_sizes(w, h, "yuv420p"))
    with _Lockstep(_decode_cmd(clip, "yuv420p", None, ffmpeg=ffmpeg),
                   _decode_cmd(clip, "yuv420p", hw, ffmpeg=ffmpeg),
                   fb, _plane_sizes(w, h, "yuv420p")) as ls:
        n, stats = ls.compare(max_frames)
    res = {"frames": n, "planes": {}}
    for name, s in zip(PLANE_ORDER, stats):
        res["planes"][name] = {"max_diff": s["max_diff"],
                               "diff_ratio_pct": round(_pct(s), 4),
                               "diff_pixels": s["diff_pixels"]}
    res["identical"] = all(v["diff_pixels"] == 0 for v in res["planes"].values())
    print("  [判据1a] 平面级比对（%d 帧）: %s" % (
        n, "Y/UV 全部 0 差异 ⇒ 差异 100%% 在 nv12→rgb24 转换"
        if res["identical"] else "⚠️ 平面级就有差异 ⇒ 差异发生在**解码**层，"
        "与立项文档 §2.1 的定性不符，需重新归因"))
    if not res["identical"]:
        for name in PLANE_ORDER:
            p = res["planes"][name]
            print("           %s: 最大差=%d 不同像素=%.4f%%"
                  % (name, p["max_diff"], p["diff_ratio_pct"]))
    return res


def check_rgb(clip, w, h, hw, max_frames, ffmpeg="ffmpeg") -> Dict[str, object]:
    with _Lockstep(_decode_cmd(clip, PIX_FMT_RGB, None, ffmpeg=ffmpeg),
                   _decode_cmd(clip, PIX_FMT_RGB, hw, ffmpeg=ffmpeg),
                   w * h * 3, _plane_sizes(w, h, PIX_FMT_RGB)) as ls:
        n, stats = ls.compare(max_frames)
    s = stats[0]
    out = {"frames": n, "max_diff": s["max_diff"],
           "diff_ratio_pct": round(_pct(s), 2),
           "mean_diff_when_diff": (round(s["sum_diff"] / s["diff_pixels"], 3)
                                   if s["diff_pixels"] else 0.0)}
    print("  [判据1b] RGB 级比对（%d 帧）: 最大逐像素差=%s，不同像素=%.2f%%"
          % (n, out["max_diff"], out["diff_ratio_pct"]))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 判据 2：reader 忠实性
# ──────────────────────────────────────────────────────────────────────────
def check_reader_fidelity(clip, w, h, max_frames, root) -> Dict[str, object]:
    """用**真实读帧器**消费，与「同参数参考解码」逐帧比对。

    缺 torch / ffmpeg-python 时明确 SKIP（readers 依赖它们），不伪装成 PASS。
    """
    import os
    for p in (os.path.join(root, "external"),
              os.path.join(root, "external", "ifrnet_video"),
              os.path.join(root, "src", "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from unittest import mock
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
            sys.modules.setdefault("torch.nn", mock.MagicMock(name="torch.nn"))
            sys.modules.setdefault("torch.nn.functional",
                                   mock.MagicMock(name="torch.nn.functional"))
        import importlib
        io = importlib.import_module("ifrnet_video.ffmpeg_io")
    except Exception as e:
        print("  [判据2] SKIP：读帧器不可用（%s: %s）" % (type(e).__name__, e))
        return {"skipped": True, "reason": "%s: %s" % (type(e).__name__, e)}

    ref = subprocess.Popen(_decode_cmd(clip, PIX_FMT_RGB, None, ffmpeg="ffmpeg"),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           **_SAFE)
    rd = io.FFmpegFrameReader(clip, prefetch=8, use_hwaccel=False)
    fb = w * h * 3
    n = mism = 0
    try:
        while True:
            pair = rd.read(timeout=60.0)
            if pair is None:
                break
            chunk = ref.stdout.read(fb)
            if len(chunk) != fb:
                break
            n += 1
            if bytes(pair[0].tobytes()) != chunk:
                mism += 1
            if max_frames and n >= max_frames:
                break
    finally:
        rd.close()
        try:
            ref.wait(timeout=30)
        except subprocess.TimeoutExpired:
            ref.kill()
        ref.stdout.close()

    out = {"frames": n, "mismatch": mism, "faithful": mism == 0}
    print("  [判据2] reader vs 同参数参考解码: %d 帧中 %d 帧不一致 ⇒ %s"
          % (n, mism, "reader 对其自身契约忠实" if mism == 0
             else "⚠️ reader 与参考解码不一致，需先修 reader"))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 判据 3：用 zscale / colorspace 穷举，钉住 NVDEC 实际用的 range/matrix
# ──────────────────────────────────────────────────────────────────────────
_MATRIX_IN = ("bt709", "smpte170m", "bt470bg", "smpte240m")
_RANGE_IN = ("tv", "pc")


def _sweep_candidates(ffmpeg: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """返回 [(描述, 输入侧选项元组)] —— 用 **输入侧颜色元数据覆盖** 钉住转换参数。

    为什么不用 `zscale`（立项文档 §3 步骤 2 的建议）：

      · 立项假设"必须换带 libzimg 的 ffmpeg build 才能控制矩阵/范围"，
        并把 `scale` 滤镜的 `in_range`/`in_color_matrix` 判为 no-op —— 这两点都成立；
      · 但**还有一条不需要任何滤镜的路**：把 `-color_range` / `-colorspace`
        放在 `-i` **之前**，直接覆盖解码器输出的颜色元数据，随后 ffmpeg 自动
        插入的 scaler 就会按这组参数做 `yuv→rgb`。实测（2026-09-15，
        Windows / ffmpeg N-122480）**它是真的生效的**，不是 no-op：

            -color_range tv   → sha=23cb17e921f8f3e2
            -color_range pc   → sha=211007eee3bb31b1   ← 输出确实变了
            -colorspace bt709 → sha=c2b681994c1e5255

      · 因而判据 3 **不再依赖 libzimg**，任何 ffmpeg build 都能跑
        （本 build 的 zscale 只暴露整数型 `range/primaries/transfer`，
        没有经典的 `matrixin=`/`matrixout=` 字符串选项，照抄立项 §3 的写法会直接
        "Option not found"）。

    注意 `-colorspace` 的取值是 ffmpeg 的枚举名：**没有 `bt601`**，
    601 系要用 `smpte170m`（NTSC）或 `bt470bg`（PAL）。
    """
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for m in _MATRIX_IN:
        for r in _RANGE_IN:
            out.append(("-colorspace %s -color_range %s" % (m, r),
                        ("-colorspace", m, "-color_range", r)))
    return out


def sweep_matrix(clip, w, h, hw, max_frames, ffmpeg="ffmpeg") -> Dict[str, object]:
    cands = _sweep_candidates(ffmpeg)
    print("  [判据3] 输入侧颜色元数据穷举（每候选最多 %d 帧，与 %s 输出逐字节比对）"
          % (max_frames, hw))
    hits: List[str] = []
    for desc, in_opts in cands:
        cmd_soft = _decode_cmd(clip, PIX_FMT_RGB, None, ffmpeg=ffmpeg,
                               in_opts=in_opts)
        cmd_hw = _decode_cmd(clip, PIX_FMT_RGB, hw, ffmpeg=ffmpeg)
        try:
            with _Lockstep(cmd_soft, cmd_hw, w * h * 3,
                           _plane_sizes(w, h, PIX_FMT_RGB)) as ls:
                n, stats = ls.compare(max_frames)
            s = stats[0]
        except Exception as e:
            print("         %-56s ERROR %s" % (desc, e))
            continue
        if s["total_bytes"] == 0:
            print("         %-56s ERROR 该组合无输出（取值非法？）" % desc)
            continue
        mark = "★ 逐字节相等" if s["diff_pixels"] == 0 else \
            "最大差=%-3d 不同=%.2f%%" % (s["max_diff"], _pct(s))
        if s["diff_pixels"] == 0:
            hits.append(desc)
        print("         %-56s %s" % (desc, mark))
    if hits:
        print("  ⇒ 硬件路径（%s）实际等价于：%s" % (hw, "；".join(hits)))
    else:
        print("  ⇒ 没有一个候选与硬件路径逐字节相等：说明其转换参数不在本次"
              "穷举范围内。可扩 _MATRIX_IN / _RANGE_IN，并加 `-color_primaries` / "
              "`-color_trc`，或给硬件路径加 -loglevel debug 看 swscale 协商到的 "
              "SWS_CS_*。")
    return {"hits": hits, "candidates": [d for d, _ in cands]}


# ──────────────────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────────────────
def render_md(res: Dict[str, object]) -> str:
    L = []
    L.append("# NVDEC vs 软解 RGB 一致性 —— 诊断报告（模板，需人填结论）\n")
    L.append("> 生成时间：%s\n" % res["generated_at"])
    L.append("## 0. 环境\n")
    for k, v in res["env"].items():
        L.append("- %s: `%s`" % (k, v))
    L.append("\n## 1. 素材\n")
    for k, v in res["clip"].items():
        L.append("- %s: `%s`" % (k, v))
    L.append("\n## 2. 判据 1a —— 平面级（差异是否在解码层）\n")
    p = res["planes"]
    L.append("| 平面 | 最大差 | 不同像素占比 |")
    L.append("|---|---:|---:|")
    for name in PLANE_ORDER:
        L.append("| %s | %d | %.4f%% |" % (name, p["planes"][name]["max_diff"],
                                           p["planes"][name]["diff_ratio_pct"]))
    L.append("\n**结论**：%s\n" % ("Y/UV 逐字节相同 ⇒ 差异 100% 在 `nv12→rgb24`"
                                   if p["identical"] else
                                   "⚠️ 平面级即有差异 ⇒ 差异在解码层，"
                                   "**与立项文档 §2.1 的定性不符，需重新归因**"))
    L.append("## 3. 判据 1b —— RGB 级幅度\n")
    r = res["rgb"]
    L.append("- 帧数：%s\n- 最大逐像素差：**%s**\n- 不同像素占比：**%.2f%%**\n"
             "- 差异像素的平均差：%s\n"
             % (r["frames"], r["max_diff"], r["diff_ratio_pct"],
                r["mean_diff_when_diff"]))
    L.append("## 4. 判据 2 —— reader 忠实性\n")
    f = res["reader_fidelity"]
    if f.get("skipped"):
        L.append("SKIP：%s\n" % f.get("reason"))
    else:
        L.append("reader vs 同参数参考解码：**%d 帧中 %d 帧不一致** ⇒ %s\n"
                 % (f["frames"], f["mismatch"],
                    "reader 忠实" if f["faithful"] else "reader 需先修"))
    L.append("## 5. 判据 3 —— 元凶（矩阵/范围）\n")
    sw = res["sweep"]
    if sw.get("skipped"):
        L.append("SKIP：%s\n" % sw.get("reason"))
    else:
        L.append("候选数：%d；逐字节与 NVDEC 相等的组合：**%s**\n"
                 % (len(sw.get("candidates", [])),
                    "、".join(sw["hits"]) if sw["hits"] else "无"))
    L.append("\n## 6. 判据 4 —— 结论（二选一，需人填）\n")
    L.append("- [ ] **不强制对齐**：在文档/日志里显式声明「路径影响像素值」，"
             "并把「跨路径产物不可逐字节比对」写进验收说明。\n")
    L.append("- [ ] **要对齐**：在读帧器侧把范围/矩阵显式钉死（`-vf` 或 "
             "`-color_range`/`-colorspace`），使两条路径产出同一结果；"
             "再回到 `Plan/Video_Enhancement_color_range_强制转换_立项Prompt.md` "
             "做 CLI 暴露。\n")
    L.append("\n**理由与证据**：（在此填写）\n")
    L.append("\n---\n\n⚠️ 做 A/B 对照前必须先冻结路径（立项文档 §2.5）：\n"
             "`READER_HWACCEL=on` 强制 NVDEC（与旧行为一致）／"
             "`READER_HWACCEL=off` 强制软解。\n")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="待诊断的素材（立项文档用 /tmp/clip_sd_25s.mp4）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="判据 1/2 最多比对帧数（0=全片）")
    ap.add_argument("--sweep", action="store_true",
                    help="执行判据 3：zscale/colorspace 参数穷举")
    ap.add_argument("--sweep-frames", type=int, default=30,
                    help="判据 3 每候选比对帧数（默认 30，足够判定逐字节相等）")
    ap.add_argument("--skip-reader", action="store_true",
                    help="跳过判据 2（避免 import 读帧器）")
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--hwaccel", default="auto",
                    help="硬件路径用的 hwaccel（默认 auto：逐个显式试 cuda/vaapi/"
                         "d3d11va/dxva2/qsv，取第一个真正可用的）")
    args = ap.parse_args()

    for exe in (args.ffmpeg, args.ffprobe):
        if not shutil.which(exe):
            print("找不到可执行文件：%s" % exe)
            return 2

    meta = _ffprobe_meta(args.clip, args.ffprobe)
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    if not (w and h):
        print("无法从素材取到宽高：%s" % meta)
        return 2

    hw, hw_note = _detect_hwaccel(args.clip, args.ffmpeg, args.hwaccel)
    is_nvdec = (hw == "cuda")
    env = {
        "ffmpeg": subprocess.run([args.ffmpeg, "-hide_banner", "-version"],
                                 capture_output=True, text=True,
                                 **_SAFE).stdout.splitlines()[0].strip(),
        "ffmpeg_filters": ",".join(
            f for f in ("zscale", "colorspace", "scale", "format")
            if _has_filter(f, args.ffmpeg)),
        "hwaccel_methods_compiled": ",".join(_hwaccels(args.ffmpeg)),
        "hwaccel_used": hw or "(none)",
        "hwaccel_probe": hw_note,
        "is_nvdec": is_nvdec,
    }
    clip_info = {"path": args.clip, "width": w, "height": h,
                 "frames_count_frames": _ffprobe_frames(args.clip, args.ffprobe),
                 "pix_fmt": meta.get("pix_fmt"),
                 "color_range": meta.get("color_range"),
                 "color_space": meta.get("color_space"),
                 "color_primaries": meta.get("color_primaries"),
                 "color_transfer": meta.get("color_transfer")}

    print("=" * 74)
    print(" 硬件解码 vs 软解 RGB 一致性诊断")
    print("=" * 74)
    print(" 素材   : %s (%dx%d, %s 帧)" % (args.clip, w, h,
                                           clip_info["frames_count_frames"]))
    print(" 色彩元数据: range=%s space=%s primaries=%s transfer=%s"
          % (clip_info["color_range"], clip_info["color_space"],
             clip_info["color_primaries"], clip_info["color_transfer"]))
    print(" ffmpeg : %s" % env["ffmpeg"])
    print(" 滤镜   : %s" % env["ffmpeg_filters"])
    print(" 编译进来的 hwaccel: %s" % env["hwaccel_methods_compiled"])
    print(" 实际可用 hwaccel  : %s" % env["hwaccel_used"])
    print(" 探测细节          : %s" % hw_note)
    if hw is None:
        print(" ❌ 本机没有任何可用的硬件解码路径，判据 1/3 不成立，全部 SKIP。")
        print("    请在有 NVIDIA GPU（NVDEC）的机器上重跑。")
    elif not is_nvdec:
        print(" ⚠️ 可用的是 **%s**，**不是 NVDEC**。下面的数字对该后端成立，"
              "但不能直接当作" % hw)
        print("    NVDEC 立项的结论（立项文档针对的是 NVIDIA NVDEC）。")
    print("-" * 74)

    res: Dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": env, "clip": clip_info,
        "planes": {"skipped": True, "reason": "无可用硬件解码路径"},
        "rgb": {"skipped": True, "reason": "无可用硬件解码路径"},
        "reader_fidelity": {"skipped": True, "reason": "未执行"},
        "sweep": {"skipped": True, "reason": "未启用 --sweep"},
    }

    if hw is None:
        if not args.skip_reader:
            res["reader_fidelity"] = check_reader_fidelity(
                args.clip, w, h, args.max_frames,
                str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    else:
        res["planes"] = check_planes(args.clip, w, h, hw,
                                     args.max_frames, args.ffmpeg, args.ffprobe)
        res["rgb"] = check_rgb(args.clip, w, h, hw, args.max_frames, args.ffmpeg)
        if not args.skip_reader:
            res["reader_fidelity"] = check_reader_fidelity(
                args.clip, w, h, args.max_frames,
                str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        if args.sweep:
            res["sweep"] = sweep_matrix(args.clip, w, h, hw,
                                        args.sweep_frames, args.ffmpeg)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("\nJSON 已写入 %s" % args.json_out)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(render_md(res))
        print("Markdown 报告已写入 %s（结论小节需人填）" % args.report)

    print("=" * 74)
    print(" 判据 4（对齐 / 声明不对齐）需依据上面证据人工判定；"
          "若判定『不对齐』，必须把")
    print(" 「路径影响像素值、跨路径产物不可逐字节比对」写进验收说明并更新 memory。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
