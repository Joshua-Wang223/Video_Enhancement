#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AV1 / VP9 画质族矩阵 —— L40(Ada) / 任意机的**可执行**验证入口。

用途
----
`SIZE_MAP` 里 AV1 / VP9 家族共 7 个编码器，它们分布在三种不同的前置条件下，
在 T4 上一次跑不完。本脚本把这一族收成**一条命令**，并在任何机器上给出：
「哪些能跑、跑出来是否达标、不能跑的原因是什么」。

    python3 Accessory/probe/av1_vp9_quality_matrix.py \
        --src /workspace/input_videos/word_world_2.mp4 \
        --report verification_report/av1_vp9_matrix.md < /dev/null

覆盖的 7 个编码器与前置条件
--------------------------
| 编码器 | 前置 | T4 状态 |
|---|---|---|
| `av1_nvenc`   | Ada 及以上（L40/A10/RTX40） | ❌ 无 AV1 NVENC |
| `av1_qsv`     | Intel QSV（Arc / 新 iGPU）  | ❌ |
| `av1_amf`     | AMD AMF（RDNA3+）           | ❌ |
| `libsvtav1`   | **ffmpeg 构建**含该编码器   | ❌ 本 build 无 |
| `libaom-av1`  | 同上                        | ❌ 本 build 无 |
| `librav1e`    | 同上                        | ❌ 本 build 无 |
| `libvpx-vp9`  | 同上                        | ✅ 可跑 |

⚠ 两种「不可用」必须分开判：**构建有没有**用 `ffmpeg -encoders`；**硬件编不编得动**
用**实跑一帧**（`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样打印选项表，不可作依据）。

两组测试
--------
* **A 组 · 质量族矩阵（CQ 轴）**：对每个可用编码器，比较「表值下发」（`quality_map`
  换算出的等效质量）与「朴素下发」（基准轴数字原样下发）相对 `libx264 crf21` 的
  码率比与 ΔPSNR。判据沿用判据脚本同一套容忍带。
* **B 组 · AV1 CONSTQP 的 QP 尺度（AC1）**：仅 `av1_nvenc` 可用时执行，扫
  `-qp {21, <表值>, 84, 105}`，用于给方案 §7 的 AC1（QP 尺度倍率是否成立）**定案**。
  判读锚点由 `quality_map` 现场推导（`resolve_quality` → `to_constqp_qp`），不写死 ——
  表从 ×4 改为 ×3（2026-09-29 L40 实测）后，写死的 `84` 会给出误导性判读。

度量口径（与 `Accessory/verify/crf_cq_unification_verify.py` 严格同源）
--------------------------------------------------------------------
码率取 `ffprobe format=bit_rate`；PSNR 取 `-v info` + **显式 `[0:v][1:v]psnr`**
并解析 `average:`。

⚠ 两个已踩过的坑（详见 memory/ffmpeg-metric-measurement-traps.md）：
  1. 裸 `-lavfi psnr`（不带 `[0:v][1:v]`）走出不同结果（实测差 3 dB），不可用；
  2. `-v error` 会压掉 psnr 滤镜 INFO 级汇总行 ⇒ ΔPSNR 恒为 0.00 的假象。

退出码：0 = 无 FAIL（SKIP 不算失败）；1 = 有 FAIL；2 = 环境前置不成立（无 ffmpeg）。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]          # Accessory/probe/x.py → 项目根
UTILS = ROOT / "src" / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import quality_map as Q                              # noqa: E402  (纯 stdlib，无 cv2/torch)

# ── 判据容忍带：与 Accessory/verify/crf_cq_unification_verify.py 同一套 ──────────
TOL_PSNR_DB = 1.5
TOL_PSNR_WARN = 3.0
RATE_PASS = (0.65, 1.50)
RATE_WARN = (0.55, 1.65)
REF_CRF = 21

# ── 覆盖对象：SIZE_MAP 里全部 AV1 / VP9 编码器 ──────────────────────────────
AV1_VP9_CODECS: Tuple[str, ...] = (
    "av1_nvenc", "av1_qsv", "av1_amf",
    "libsvtav1", "libaom-av1", "librav1e", "libvpx-vp9",
)
# 「需什么」——用于 SKIP 原因的可读描述
_NEEDS = {
    "av1_nvenc": "Ada 及以上 GPU（L40/A10/RTX40）",
    "av1_qsv": "Intel QSV（Arc / 新 iGPU）",
    "av1_amf": "AMD AMF（RDNA3+）",
    "libsvtav1": "ffmpeg 构建含 libsvtav1",
    "libaom-av1": "ffmpeg 构建含 libaom-av1",
    "librav1e": "ffmpeg 构建含 librav1e",
    "libvpx-vp9": "ffmpeg 构建含 libvpx-vp9",
}
# 除质量参数外必须补的参数（会在报告里注明，避免"结果与标定口径不符"的误解）
# ⚠ libvpx-vp9 的 `-b:v 0` 由 `resolve_quality` 的 extra 给出，此处不重复列。
_EXTRA_ARGS: Dict[str, List[str]] = {
    "libsvtav1": ["-preset", "8"],                       # E6 标定所用档位
    "libaom-av1": ["-cpu-used", "5"],                    # 默认档极慢，探针提速
    "libvpx-vp9": ["-cpu-used", "4", "-row-mt", "1"],
}

# NVENC 的**生产 rc 口径**（跨仓契约 CR-2）——探针必须与生产/harness 一致，
# 否则 A 组的 `-cq` 结论不代表生产实际等效点：
#   · h264/hevc → **裸 `vbr`**（FFmpeg 9.0 CLI 移除 `vbr_hq`/`qvbr` ⇒ 统一映射为 vbr；
#     `-tune hq` 是 ffmpeg 默认值、固定 CQ 下 multipass 不升 VMAF ⇒ 均不下发）
#   · av1_nvenc → plain `vbr`（VE 生产 av1 由 `vbr_hq` 降级为 `vbr`）
# ⚠ 与 `Accessory/probe/calibrate_equal_quality.BASE_LOCK`、`ffmpeg_io` writer 的
#   `_rc_v_map` 同源；软编不用 `-rc`，qsv/amf 不在 VE 生产路径故不猜。
# ⚠ VU 侧同步（跨仓 CR-2 handoff）：VU 若仍下发 `-rc:v vbr_hq` 会被 FFmpeg 9.0 拒绝。
_PROD_RC: Dict[str, List[str]] = {
    "h264_nvenc": ["-rc:v", "vbr"],
    "hevc_nvenc": ["-rc:v", "vbr"],
    "av1_nvenc":  ["-rc:v", "vbr"],
}


# =============================================================================
# 进程与度量
# =============================================================================

def run(cmd: List[str], timeout: int) -> Tuple[int, str, str]:
    """跑外部命令。stdin 固定 DEVNULL（后台进程组 + tty 下会被 SIGTTOU 整组停住）。"""
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)


def available_encoders(ffmpeg: str) -> set:
    """本机 ffmpeg **构建**里存在的编码器名集合（与硬件能力无关）。"""
    rc, out, _ = run([ffmpeg, "-hide_banner", "-encoders"], 120)
    names = set()
    if rc != 0:
        return names
    for line in (out or "").splitlines():
        m = re.match(r"^\s*[VAS][.A-Z]*\s+(\S+)", line)
        if m:
            names.add(m.group(1))
    return names


def smoke_encode(ffmpeg: str, codec: str, timeout: int = 180) -> Tuple[bool, str]:
    """**实跑一帧**探测该编码器是否真的可用（构建 + 硬件一起判）。

    这是唯一可靠的硬件能力判据：`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样
    打印完整选项表，只有真编码才会报 `No capable devices found`。
    """
    cmd = [ffmpeg, "-hide_banner", "-v", "error",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
           "-frames:v", "1", "-c:v", codec]
    cmd += _EXTRA_ARGS.get(codec, [])
    cmd += ["-f", "null", "-"]
    rc, _, err = run(cmd, timeout)
    if rc == 0:
        return True, "实跑一帧成功"
    first = [l for l in (err or "").strip().splitlines() if l.strip()]
    return False, (first[0] if first else f"rc={rc}")[:180]


def bitrate_bps(ffprobe: str, path: Path) -> int:
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "format=bit_rate",
                      "-of", "csv=p=0", str(path)], 60)
    try:
        return int((out or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def psnr_avg(ffmpeg: str, enc: Path, ref: Path, nframes: int, timeout: int) -> Optional[float]:
    """对源 PSNR —— 严格镜像判据脚本 `Ctx._metric` 的口径。

    ⚠ 必须 `-v info` + **显式 `[0:v][1:v]`**；裸 `psnr` 与 `-v error` 都会给出错值。
    """
    cmd = [ffmpeg, "-hide_banner", "-v", "info", "-i", str(enc), "-i", str(ref)]
    if nframes > 0:
        cmd += ["-frames:v", str(nframes)]
    cmd += ["-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]
    _, _, err = run(cmd, timeout)
    hits = re.findall(r"average:\s*([0-9.]+|inf)", err or "")
    if not hits:
        return None
    if hits[-1] == "inf":
        return float("inf")
    try:
        return float(hits[-1])
    except ValueError:
        return None


def probe_frames(ffprobe: str, path: Path) -> Tuple[int, str]:
    """取帧数（用于 `-frames:v`）与可读描述。

    用 JSON 输出解析：csv + 多个 `-show_entries` 段的行序在实测中不稳定
    （曾把 nb_frames 解析成 0），JSON 取键更可靠。
    """
    rc, out, _ = run([ffprobe, "-v", "error", "-select_streams", "v:0",
                      "-show_entries",
                      "stream=nb_frames,avg_frame_rate,r_frame_rate",
                      "-show_entries", "format=duration",
                      "-of", "json", str(path)], 60)
    nbf, fps, dur = 0, 0.0, 0.0
    try:
        d = json.loads(out or "{}")
        st = (d.get("streams") or [{}])[0]
        nbf = int(st.get("nb_frames") or 0)
        for k in ("avg_frame_rate", "r_frame_rate"):
            v = st.get(k) or ""
            if "/" in v:
                a, b = v.split("/")
                if float(b):
                    fps = float(a) / float(b)
                    break
        dur = float((d.get("format") or {}).get("duration") or 0.0)
    except (ValueError, IndexError, TypeError):
        pass
    # 容器元数据 nb_frames 与实际时长不符（-c copy 分段常见）⇒ 取较大者更安全
    n = max(nbf, int(fps * dur))
    return n, f"nb_frames={nbf}/fps={fps:.3g}/duration={dur:.2f}s → 采用 {n} 帧"


# =============================================================================
# 编码
# =============================================================================

def encode_quality(ffmpeg: str, src: Path, out: Path, codec: str, value: int,
                   param: str, extra: List[str], timeout: int) -> Tuple[bool, str]:
    """按 `quality_map.resolve_quality` 给出的 (param, value, extra) 下发。

    ⚠ NVENC 额外补 `_PROD_RC`（生产 rc 口径，跨仓契约 CR-2）——否则 `-cq` 会落在
    ffmpeg preset 默认 rc 上，与生产不一致（h264/hevc 尤其：默认 VBR ≠ 生产 `vbr_hq`）。
    """
    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error", "-i", str(src),
           "-c:v", codec, param, str(value)]
    cmd += _PROD_RC.get(codec, [])          # 生产 rc（NVENC 显式下发，CR-2）
    cmd += list(extra)
    cmd += _EXTRA_ARGS.get(codec, [])
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    rc, _, err = run(cmd, timeout)
    if rc != 0:
        return False, (err or "").strip().splitlines()[-1][:200] if err.strip() else f"rc={rc}"
    return True, ""


def encode_constqp(ffmpeg: str, src: Path, out: Path, codec: str, qp: int,
                   timeout: int) -> Tuple[bool, str]:
    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error", "-i", str(src),
           "-c:v", codec, "-preset", "p4", "-rc:v", "constqp", "-qp", str(qp),
           "-bf", "0", "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = run(cmd, timeout)
    if rc != 0:
        return False, (err or "").strip().splitlines()[-1][:200] if err.strip() else f"rc={rc}"
    return True, ""


def make_synthetic(ffmpeg: str, out: Path, timeout: int = 300) -> Path:
    """默认素材：无损合成片段（`--src` 未给时使用）。"""
    if out.exists():
        return out
    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error",
           "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
           "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
           "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = run(cmd, timeout)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"合成素材失败：{(err or '')[-300:]}")
    return out


# =============================================================================
# 判定
# =============================================================================

def verdict(d_psnr: Optional[float], ratio: float) -> str:
    """与判据脚本 `_rate_verdict` 同义：质量单向下探容差 + 码率带。"""
    if d_psnr is None or ratio <= 0:
        return "SKIP"
    if ratio < RATE_WARN[0] or ratio > RATE_WARN[1] or d_psnr < -TOL_PSNR_WARN:
        return "FAIL"
    if RATE_PASS[0] <= ratio <= RATE_PASS[1] and d_psnr >= -TOL_PSNR_DB:
        return "PASS"
    return "WARN"


def av1_expected_qp(ref_crf: int) -> int:
    """当前 `quality_map` 表把基准轴换算成 AV1 CONSTQP QP 后的**期望值**。

    走生产同一条链（``resolve_quality`` → ``to_constqp_qp``），因此 AC1 的判读
    锚点永远跟随表，而不是像旧版那样把 ``84``（×4 假设）写死在脚本里 ——
    表已按 L40 实测改为 ×3（QP 63）后，写死的判读会给出误导性结论。
    """
    _, cq_value, _, _ = Q.resolve_quality("av1_nvenc", default_ref=ref_crf)
    return int(Q.to_constqp_qp("av1_nvenc", cq_value))


def scan_av1_qp(ffmpeg: str, ffprobe: str, src: Path, tmp: Path, n: int,
                soft_psnr: Optional[float], soft_bps: int, timeout: int,
                points: Tuple[int, ...]) -> List[dict]:
    """AC1：AV1 CONSTQP 的 QP 尺度扫描（需 av1_nvenc 可用）。"""
    rows = []
    for qp in points:
        out = tmp / f"ac1_qp{qp}.mp4"
        ok, why = encode_constqp(ffmpeg, src, out, "av1_nvenc", qp, timeout)
        if not ok:
            rows.append({"qp": qp, "ok": False, "why": why})
            continue
        ps = psnr_avg(ffmpeg, out, src, n, timeout)
        bps = bitrate_bps(ffprobe, out)
        d = (ps - soft_psnr) if (ps is not None and soft_psnr is not None) else None
        r = (bps / soft_bps) if soft_bps else 0.0
        rows.append({"qp": qp, "ok": True, "psnr": ps, "ratio": r, "d_psnr": d,
                     "verdict": verdict(d, r)})
    return rows


# =============================================================================
# main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="AV1/VP9 画质族矩阵（L40/Ada 与任意机的可执行验证入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="真实素材（默认合成 testsrc2 无损片段）")
    ap.add_argument("--only", help="只跑指定编码器（逗号分隔）")
    ap.add_argument("--skip-qp-scan", action="store_true",
                    help="跳过 B 组（AC1 的 AV1 constqp QP 扫描）")
    ap.add_argument("--ref-crf", type=int, default=REF_CRF, help=f"软编基准 CRF（默认 {REF_CRF}）")
    ap.add_argument("--quality-mode", choices=["size", "quality"], default="quality",
                    help="换算口径：size=等体积（文件大小优先）；quality=等质量（VMAF 定标，画质优先，默认）")
    ap.add_argument("--timeout", type=int, default=1800, help="单条编码/度量超时秒")
    ap.add_argument("--report", help="Markdown 报告输出路径")
    ap.add_argument("--json", dest="json_path", help="JSON 结果输出路径")
    ap.add_argument("--keep-temp", action="store_true", help="保留临时产物")
    args = ap.parse_args(argv)

    # 选定换算口径（等体积/等质量）；默认 volume，旧行为逐字不变。
    if hasattr(Q, "set_quality_mode"):
        Q.set_quality_mode(args.quality_mode)

    ffmpeg = shutil.which("ffmpeg") or ""
    ffprobe = shutil.which("ffprobe") or ""
    if not ffmpeg or not ffprobe:
        print("❌ 缺少 ffmpeg/ffprobe")
        return 2

    tmp = ROOT / "temp" / "av1_vp9_matrix"
    tmp.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  AV1 / VP9 画质族矩阵")
    print("=" * 78)

    src = Path(args.src) if args.src else make_synthetic(ffmpeg, tmp / "src_synthetic.mp4")
    n_def, desc = probe_frames(ffprobe, src)
    build = available_encoders(ffmpeg)
    print(f"素材: {src}")
    print(f"      {desc}（用于 -frames:v {n_def}）")
    print(f"构建编码器总数: {len(build)}")

    codecs = list(AV1_VP9_CODECS)
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        codecs = [c for c in codecs if c in want]
    _qm = Q.get_quality_map() if hasattr(Q, "get_quality_map") else Q.SIZE_MAP
    unknown = [c for c in codecs if c not in _qm]
    if unknown:
        print(f"⚠ 不在换算表（{args.quality_mode} 口径），忽略: {unknown}")

    # ── 0. 可用性（构建 + 实跑）────────────────────────────────────────────
    status: Dict[str, dict] = {}
    for c in codecs:
        if c not in build:
            status[c] = {"ok": False, "why": f"本机 ffmpeg 构建不含该编码器（需 {_NEEDS[c]}）"}
            continue
        ok, why = smoke_encode(ffmpeg, c)
        status[c] = {"ok": ok, "why": why if ok else f"实跑一帧失败（需 {_NEEDS[c]}）: {why}"}
    for c in codecs:
        mark = "✅ 可用" if status[c]["ok"] else "⏭️ SKIP"
        print(f"  {mark:8} {c:12} {status[c]['why']}")

    # ── 1. 软编基准 ───────────────────────────────────────────────────────
    soft = tmp / "soft.mp4"
    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.ref_crf),
           "-pix_fmt", "yuv420p", str(soft)]
    rc, _, err = run(cmd, args.timeout)
    if rc != 0:
        print(f"❌ 软编基准失败：{(err or '')[-300:]}")
        return 1
    soft_bps = bitrate_bps(ffprobe, soft)
    soft_psnr = psnr_avg(ffmpeg, soft, src, n_def, args.timeout)
    print(f"\n软编基准 libx264 crf{args.ref_crf}: {soft_bps/1000:.0f} kbps / "
          f"PSNR {soft_psnr if soft_psnr is None else round(soft_psnr, 3)} dB")

    # ── 2. A 组：质量族矩阵 ───────────────────────────────────────────────
    print("\n【A 组】AV1/VP9 质量族矩阵（表值 vs 朴素，基准 = libx264 crf%d）" % args.ref_crf)
    rows: List[dict] = []
    for c in codecs:
        if not status[c]["ok"]:
            rows.append({"codec": c, "verdict": "SKIP", "why": status[c]["why"]})
            continue
        param, val, extra, note = Q.resolve_quality(c, default_ref=args.ref_crf)
        # 「朴素下发」= 修复前的行为：把基准轴数字原样当该编码器的质量值
        naive = args.ref_crf
        rec = {"codec": c, "param": param, "value": val, "naive": naive, "note": note,
               "extra_args": _PROD_RC.get(c, []) + list(extra) + _EXTRA_ARGS.get(c, [])}
        ok, why = encode_quality(ffmpeg, src, tmp / f"c_{c}.mp4", c, val,
                                 param, extra, args.timeout)
        if not ok:
            rec.update({"verdict": "FAIL", "why": f"编码失败: {why}"})
            rows.append(rec)
            continue
        ok2, why2 = encode_quality(ffmpeg, src, tmp / f"n_{c}.mp4", c, naive,
                                   param, extra, args.timeout)
        ps_c = psnr_avg(ffmpeg, tmp / f"c_{c}.mp4", src, n_def, args.timeout)
        bps_c = bitrate_bps(ffprobe, tmp / f"c_{c}.mp4")
        d = (ps_c - soft_psnr) if (ps_c is not None and soft_psnr is not None) else None
        ratio = (bps_c / soft_bps) if soft_bps else 0.0
        rec.update({"psnr": ps_c, "kbps": round(bps_c / 1000),
                    "ratio": round(ratio, 3),
                    "d_psnr": None if d is None else round(d, 3),
                    "verdict": verdict(d, ratio)})
        if ok2:
            bps_n = bitrate_bps(ffprobe, tmp / f"n_{c}.mp4")
            rec["naive_ratio"] = round((bps_n / soft_bps) if soft_bps else 0.0, 3)
        else:
            rec["naive_error"] = why2
        rows.append(rec)

    for r in rows:
        if r["verdict"] == "SKIP":
            print(f"  ⏭️ SKIP  {r['codec']:12} {r.get('why', '')}")
            continue
        d = r.get("d_psnr")
        dtxt = "n/a" if d is None else f"{d:+.2f}"
        ntxt = "" if "naive_ratio" not in r else f"（朴素 {r['naive_ratio']:.2f}×）"
        print(f"  {r['verdict']:5} {r['codec']:12} {r['param']} {r['value']}  "
              f"码率比 {r['ratio']:.2f}×{ntxt}  ΔPSNR {dtxt} dB")

    # ── 3. B 组：AV1 constqp QP 尺度（AC1）─────────────────────────────────
    qp_rows: List[dict] = []
    exp_qp = av1_expected_qp(args.ref_crf)
    ac1_text = "未执行（--skip-qp-scan 或 av1_nvenc 不可用）"
    if args.skip_qp_scan:
        print("\n【B 组】AV1 CONSTQP QP 扫描：已按 --skip-qp-scan 跳过")
    elif not status.get("av1_nvenc", {}).get("ok"):
        ac1_text = "SKIP：av1_nvenc 不可用（AC1 需 Ada/L40）"
        print(f"\n【B 组】AV1 CONSTQP QP 扫描：⏭️ SKIP（av1_nvenc 不可用，AC1 需 Ada/L40）")
    else:
        # 扫描点 = 旧实现的错值(21) + 当前表期望值 + 84(×4 旧假设) + 105(×5 方向对照)
        points = tuple(dict.fromkeys([21, exp_qp, 84, 105]))
        print("\n【B 组】AV1 CONSTQP QP 尺度扫描"
              f"（方案 §7 AC1；当前表期望 -qp {exp_qp}）")
        qp_rows = scan_av1_qp(ffmpeg, ffprobe, src, tmp, n_def, soft_psnr,
                              soft_bps, args.timeout, points)
        for r in qp_rows:
            if not r["ok"]:
                print(f"  ⏭️ -qp {r['qp']:<4} 编码失败: {r['why']}")
                continue
            mark = " ←表值" if r["qp"] == exp_qp else ""
            print(f"  {r['verdict']:5} -qp {r['qp']:<4} 码率比 {r['ratio']:.2f}×  "
                  f"ΔPSNR {r['d_psnr']:+.2f} dB{mark}")
        hit = any(x.get("ok") and x["qp"] == exp_qp and x["verdict"] == "PASS"
                  for x in qp_rows)
        all_fail = bool(qp_rows) and all(
            x.get("ok") and x["verdict"] == "FAIL" for x in qp_rows)
        if hit:
            ac1 = (f"表值 -qp {exp_qp} 落带内 ⇒ AC1 PASS。该值由 quality 口径的 "
                   f"`QUALITY_MAP_QP['av1_nvenc']`（L40 17 素材仿射标定）换算得到"
                   f"（旧 ×3 近似已由标定表取代）")
        elif all_fail:
            ac1 = ("所有扫描点全出带 ⇒ AV1 的 -qp 与基准轴非线性，应记为「不支持」，"
                   "生产改走 -cq/VBR")
        else:
            inband = [x for x in qp_rows
                      if x.get("ok") and RATE_PASS[0] <= x["ratio"] <= RATE_PASS[1]
                      and x["d_psnr"] is not None and x["d_psnr"] >= -TOL_PSNR_DB]
            if inband:
                best = min(inband, key=lambda x: abs(x["qp"] - args.ref_crf))
                ac1 = (f"表值 -qp {exp_qp} 未落带内，落带点为 {best['qp']} ⇒ 应更新 "
                       f"`QUALITY_MAP_QP['av1_nvenc']`（quality 口径 QP 轴），"
                       f"并同步判据 G3-7 / G6-7")
            else:
                ac1 = (f"表值 -qp {exp_qp} 未落带内且无落带点 ⇒ 重扫更密的 QP 网格，"
                       f"再决定 a（不要凭单调性外推）")
        print(f"\n  ⇒ AC1 判读：{ac1}")
        ac1_text = ac1

    # ── 4. 产出 ───────────────────────────────────────────────────────────
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL") + \
        sum(1 for r in qp_rows if r.get("ok") and r["verdict"] == "FAIL")
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_warn = sum(1 for r in rows if r["verdict"] == "WARN")
    n_skip = sum(1 for r in rows if r["verdict"] == "SKIP")
    print("\n" + "=" * 78)
    print(f"  汇总：PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skip}"
          f"（B 组：{'未跑' if not qp_rows else len(qp_rows)} 点）")
    print("=" * 78)

    result: Dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "src": str(src), "src_frames": n_def, "ref_crf": args.ref_crf,
        "soft": {"kbps": round(soft_bps / 1000),
                 "psnr": None if soft_psnr is None else round(soft_psnr, 6)},
        "tolerances": {"RATE_PASS": RATE_PASS, "RATE_WARN": RATE_WARN,
                       "TOL_PSNR_DB": TOL_PSNR_DB, "TOL_PSNR_WARN": TOL_PSNR_WARN},
        "matrix": rows, "av1_qp_scan": qp_rows,
        "av1_qp_expected": av1_expected_qp(args.ref_crf),
        "ac1_verdict": ac1_text,
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail, "skip": n_skip},
    }
    md = _render_md(result)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md, encoding="utf-8")
        print(f"📄 报告: {args.report}")
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🧾 JSON: {args.json_path}")
    if not args.keep_temp:
        for f in tmp.glob("*.mp4"):
            if f != src:
                f.unlink(missing_ok=True)
    return 1 if n_fail else 0


def _render_md(r: Dict[str, Any]) -> str:
    L = ["# AV1 / VP9 画质族矩阵", "",
         f"- 生成时间：{r['generated']}",
         f"- 素材：`{r['src']}`（{r['src_frames']} 帧）",
         f"- 软编基准：libx264 crf{r['ref_crf']} = {r['soft']['kbps']} kbps / "
         f"PSNR {r['soft']['psnr']} dB",
         f"- 容忍带：RATE_PASS={r['tolerances']['RATE_PASS']}、"
         f"RATE_WARN={r['tolerances']['RATE_WARN']}、"
         f"TOL_PSNR_DB={r['tolerances']['TOL_PSNR_DB']}、"
         f"TOL_PSNR_WARN={r['tolerances']['TOL_PSNR_WARN']}",
         "",
         f"**汇总：PASS={r['summary']['pass']} / WARN={r['summary']['warn']} / "
         f"FAIL={r['summary']['fail']} / SKIP={r['summary']['skip']}**", "",
         "## A 组 · 质量族矩阵", "",
         "| 编码器 | 结论 | 下发 | 码率比 | 朴素比 | ΔPSNR | 说明 |",
         "|---|:--:|---|---:|---:|---:|---|"]
    for x in r["matrix"]:
        if x["verdict"] == "SKIP":
            L.append(f"| `{x['codec']}` | ⏭️ SKIP | — | — | — | — | {x.get('why','')} |")
            continue
        d = x.get("d_psnr")
        d_txt = "n/a" if d is None else f"{d:+.2f} dB"
        nv = x.get("naive_ratio")
        nv_txt = "—" if nv is None else f"{nv:.2f}×"
        note = x.get("why") or x.get("note", "")
        L.append(f"| `{x['codec']}` | {x['verdict']} | `{x['param']} {x['value']}` | "
                 f"{x['ratio']:.2f}× | {nv_txt} | {d_txt} | {note} |")
    L += ["", "## B 组 · AV1 CONSTQP QP 尺度（方案 §7 AC1）", ""]
    if not r["av1_qp_scan"]:
        L.append(f"未执行 —— {r.get('ac1_verdict', 'av1_nvenc 不可用或已跳过')}。")
    else:
        exp = r.get("av1_qp_expected")
        L += [f"当前 `quality_map` 表期望值：**`-qp {exp}`**", "",
              "| `-qp` | 结论 | 码率比 | ΔPSNR | |", "|---:|:--:|---:|---:|---|"]
        for q in r["av1_qp_scan"]:
            if not q.get("ok"):
                L.append(f"| {q['qp']} | 编码失败 | — | — | |")
                continue
            mark = " ←表值" if q["qp"] == exp else ""
            L.append(f"| {q['qp']} | {q['verdict']} | {q['ratio']:.2f}× | "
                     f"{q['d_psnr']:+.2f} dB |{mark} |")
        L += ["", f"**AC1 判读**：{r.get('ac1_verdict', '')}"]
    L += ["", "> 度量口径与 `Accessory/verify/crf_cq_unification_verify.py` 同源：",
          "> 码率 = `ffprobe format=bit_rate`；PSNR = `-v info` + 显式 `[0:v][1:v]psnr`。",
          "> ⚠ 裸 `-lavfi psnr` 或 `-v error` 都会给出错值。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
