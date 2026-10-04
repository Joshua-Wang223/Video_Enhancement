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

覆盖
----
| 阶段 | 内容 |
|---|---|
| 前置 | `av1_nvenc` **实跑一帧**（不是 `-h encoder=`）；不可用 ⇒ exit 2（环境前置不成立） |
| 跑批 | 对每个 rate_mode 调 `src/main_video_optimized.py`，两侧 `--codec-* av1_nvenc` |
| 采样 | 整棵进程树 RSS + `nvidia-smi` 显存（判泄漏：后半程斜率） |
| 验收 | ① 退出码 ② 段级日志 `decoded == expected` ③ 产物 `ffprobe -count_frames` 帧数守恒 ④ `validate_decodable_video(count_mode='decode')` ⑤ `segment_bitstream_verify_v5 --skip-chroma` 硬指标 ⑥ QA sidecar 字段完整 |

⚠ **必须 `< /dev/null`**：这些脚本在「后台进程组 + tty stdin」下会被 SIGTTOU 整组停住。
⚠ 色度检查（检查 4）在真实素材上是**内容相关假阳性**（方案 §8.5：AV1 长片 113 簇，
   而源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。

退出码：0 = 无 FAIL；1 = 有 FAIL；2 = 环境前置不成立（无 ffmpeg / 无 av1_nvenc）。
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


def av1_nvenc_available(ffmpeg: str) -> Tuple[bool, str]:
    """**实跑一帧**判 AV1 NVENC 能力（`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样打印）。"""
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
           "-frames:v", "1", "-c:v", "av1_nvenc", "-f", "null", "-"]
    rc, _, err = run(cmd, 180)
    if rc == 0:
        return True, "实跑一帧成功"
    first = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return False, (first[0] if first else f"rc={rc}")[:200]


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
class MemWatcher:
    """盯住匹配 `--match` 的进程树，周期采样 RSS 与 GPU 显存。"""

    def __init__(self, match: str, interval: int = 10):
        self.match = match
        self.interval = interval
        self.rows: List[Tuple[float, float, float]] = []

    def _sample(self) -> Tuple[float, float, int]:
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
        seen, total, n = set(), 0, 0
        for root in roots:
            stack = [root]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                n += 1
                total += rss.get(x, 0)
                stack += kids.get(x, [])
        gpu = 0.0
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip()
            gpu = float(q.splitlines()[0])
        except Exception:  # noqa: BLE001
            pass
        return total / 1024.0, gpu, n

    def run(self, proc: subprocess.Popen) -> None:
        t0 = time.time()
        while proc.poll() is None:
            rss, gpu, n = self._sample()
            if n:
                self.rows.append((time.time() - t0, rss, gpu))
            time.sleep(self.interval)

    def slope_mb_per_min(self) -> Optional[float]:
        """后半程 RSS 线性斜率（MB/min）；负值/接近 0 ⇒ 无单调泄漏。"""
        if len(self.rows) < 8:
            return None
        half = self.rows[len(self.rows) // 2:]
        xs = [r[0] for r in half]
        ys = [r[1] for r in half]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 60.0

    def peak(self) -> Tuple[float, float]:
        if not self.rows:
            return 0.0, 0.0
        return (max(r[1] for r in self.rows), max(r[2] for r in self.rows))


# ── 单轮冒烟 ─────────────────────────────────────────────────────────────────
def run_one(ffmpeg: str, ffprobe: str, src: Path, out: Path, rate_mode: str,
            seg_dur: int, extra: List[str], keep: bool,
            mem_interval: int, checks_only: bool = False,
            log_text: str = "") -> Dict[str, Any]:
    rec: Dict[str, Any] = {"rate_mode": rate_mode, "out": str(out), "checks": []}
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
               "--codec-ifrnet", "av1_nvenc", "--codec-esrgan", "av1_nvenc",
               "--rate-mode-ifrnet", rate_mode, "--rate-mode-esrgan", rate_mode,
               "--segment-duration", str(seg_dur)] + extra
        rec["cmd"] = " ".join(cmd)
        print(f"\n=== [{rate_mode}] {' '.join(cmd)}")
        log_path = out.with_suffix(out.suffix + f".{rate_mode}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT)
            watcher = MemWatcher(match=str(MAIN), interval=mem_interval)
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
            ok = not missing and payload.get("codec_hint") == "av1_nvenc"
            check("S6", "QA sidecar 字段完整", ok,
                  f"缺字段 {missing or '无'}；codec_hint={payload.get('codec_hint')} "
                  f"rate_mode={payload.get('rate_mode_ifrnet')}")
        except (OSError, ValueError) as exc:
            check("S6", "QA sidecar 字段完整", False, f"解析失败：{exc}")
    else:
        check("S6", "QA sidecar 字段完整", False, f"未生成：{qa}")

    # ⑦ 产物编码器/音轨（确认没被静默换编码器 —— 曾经的缺陷 ②）
    codec = probe_stream_codec(ffprobe, out)
    check("S7", "产物编码器确为 av1", codec == "av1",
          f"codec={codec}；音轨={'有' if has_audio(ffprobe, out) else '无'}")

    # ⑧ 内存
    if watcher is not None and watcher.rows:
        rss_peak, gpu_peak = watcher.peak()
        slope = watcher.slope_mb_per_min()
        rec["rss_peak_mb"] = round(rss_peak, 1)
        rec["gpu_peak_mib"] = round(gpu_peak, 1)
        rec["rss_slope_mb_per_min"] = None if slope is None else round(slope, 1)
        check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）",
              None if slope is None else (slope <= 50.0),
              f"RSS 峰值 {rss_peak:.0f} MB，斜率 {slope:+.1f} MB/min，显存峰值 {gpu_peak:.0f} MiB"
              if slope is not None else f"样本不足（RSS 峰值 {rss_peak:.0f} MB）")
    else:
        check("S8", "无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min）", None, "未采样（--checks-only）")

    if not keep and not checks_only:
        for f in out.parent.glob(out.name + "*"):
            if f.suffix in (".mp4", ".mkv", ".mov"):
                f.unlink(missing_ok=True)
    return rec


# ── 报告 ─────────────────────────────────────────────────────────────────────
def render_md(result: Dict[str, Any]) -> str:
    L = ["# AV1 NVENC 端到端冒烟 + 验收报告", "",
         f"- 生成时间：{result['generated']}",
         f"- 素材：`{result['src']}`",
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
    L += ["> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**",
          "> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。", ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="AV1 NVENC 端到端冒烟 + 验收（constqp / vbr 双 rate_mode）")
    ap.add_argument("--src", required=True, help="真实素材（建议 ≥1 min，含音轨）")
    ap.add_argument("--out-dir", help="产物目录（默认 <src 同级>/av1_smoke_out）")
    ap.add_argument("--rate-modes", default="constqp,vbr", help="逗号分隔（默认 constqp,vbr）")
    ap.add_argument("--segment-duration", type=int, default=30, help="分段秒数（默认 30）")
    ap.add_argument("--timeout", type=int, default=7200, help="单轮超时秒（默认 7200）")
    ap.add_argument("--mem-interval", type=int, default=10, help="内存采样间隔秒（默认 10）")
    ap.add_argument("--keep", action="store_true", help="保留产物（默认跑完删除视频）")
    ap.add_argument("--checks-only", metavar="VIDEO", default="",
                    help="只对既有 AV1 产物跑验收项（不跑批、不需要 GPU）")
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

    ok_av1, why = av1_nvenc_available(ffmpeg)
    print("=" * 78)
    print("  AV1 NVENC 端到端冒烟 + 验收")
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
                      [], True, args.mem_interval, checks_only=True)
        runs = [rec]
        for c in rec["checks"]:
            mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[c["status"]]
            print(f"  {mark} [{c['id']}] {c['name']} — {c['detail']}")
    else:
        print(f"前置: av1_nvenc {'✅' if ok_av1 else '⏭️ 不可用'} — {why}")
        if not ok_av1:
            print("⏭️ 环境前置不成立（需 Ada 及以上 + ffmpeg 含 av1_nvenc）⇒ exit 2")
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else src.parent / "av1_smoke_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for rate in [r.strip() for r in args.rate_modes.split(",") if r.strip()]:
            out = out_dir / f"av1_{rate}{src.suffix or '.mp4'}"
            rec = run_one(ffmpeg, ffprobe, src, out, rate, args.segment_duration,
                          list(args.extra), args.keep, args.mem_interval)
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
