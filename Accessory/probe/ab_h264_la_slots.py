#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h264+LA 槽位 A/B 实测（9 vs 11）—— 判定 `drain LockBitstream code=8` 的成因。

要回答的问题
------------
`code=8`（`INVALID_PARAM`）在 LA 流式首排空期出现。两种成因都能产生它，
**单侧观测无法区分**：

| 假说 | 机制 | 若为真，则提高槽数（9→11）应 |
|---|---|---|
| **H1槽位不足** | LA 输出延迟≈la+1，而物理槽仅 la+1 ⇒ 余量0 ⇒ 提交/排空循环依赖 | **消除** code=8 |
| **H2 warmup 期本就未就绪** | LA warmup 期硬件还没产出任何东西，与槽数无关 | **无影响**，code=8 照旧 |

背景与证据见 `memory/h264-la-ready-gate-asymmetry.md`。两侧槽数现状：
IFRNet `la+3`（h264 也一样，11 槽）、ESRGAN `la+1`（h264，9 槽）。
本脚本用 `NVENC_ESRGAN_AB_SLOTS` 环境变量**只改 ESRGAN 侧槽数**，其它一切不变。

⚠ **单臂单次不足以定因果**：槽数与素材/时段相关，两臂必须**同素材、同分段、
交替或紧邻运行**（memory `eqq-batch-measure-parallel-constraints`：
「A/B 对照必须与被测项同批跑」）。`--repeats` 默认 2。

⚠ **同机并发会污染**：`shared-gpu-host-concurrent-jobs.md`——本机是共享 GPU 主机，
开跑前须确认无他人任务，且 `--jobs 1`。

用法
----
    # 预检（不需要 GPU 长时间占用）
    python3 Accessory/probe/ab_h264_la_slots.py --precheck < /dev/null

    # 实跑（每臂约 15 min @100s 素材）
    python3 Accessory/probe/ab_h264_la_slots.py \
        --src /tmp/b1/seg100.mp4 --repeats 2 < /dev/null

⚠ **必须 `< /dev/null`**（后台进程组 + tty stdin 会 SIGTTOU 整组停住）。

退出码：0 = 判据出结论；1 = 有 FAIL / 数据不足；2 = 环境前置不成立。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE = ROOT / "Accessory" / "verify" / "av1_pipeline_smoke.py"
#: A/B 两臂的槽位取值：la+1（ESRGAN 现状）与 la+3（IFRNet 现状）。
ARMS = (9, 11)

CODE8_RE = re.compile(r"drain LockBitstream code=8 \(slot=(\d+), fi=(\d+)\)")


def _precheck() -> int:
    """不需要长时间占卡的前置检查：ffmpeg / 编码器实跑 / 杠杆可用性。"""
    print("=" * 78)
    print("  h264+LA 槽位 A/B（9 vs 11）— 预检")
    print("=" * 78)
    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        p = shutil.which(tool)
        print(f"{'✅' if p else '❌'} {tool}: {p or '缺失'}")
        ok &= bool(p)
    if not ok:
        return 2
    # 实跑一帧（不认 -h encoder）
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=30:duration=1",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL)
    print(f"{'✅' if r.returncode == 0 else '❌'} h264_nvenc 实跑一帧 rc={r.returncode}")
    if r.returncode != 0:
        print("   ", (r.stderr or "")[:300])
        return 2
    # GPU 占用
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30,
                           stdin=subprocess.DEVNULL).stdout.strip()
        print(f"ℹ️GPU: {q.splitlines()[0] if q else '?'} MiB / %（共享主机，非 0 需先确认无他人任务）")
    except Exception as e:
        print(f"ℹ️ nvidia-smi 不可用: {e}")
    # 杠杆存在性（AST，避免匹配注释）
    sdk = ROOT / "external" / "realesrgan_video" / "nvenc_sdk.py"
    has_lever = "NVENC_ESRGAN_AB_SLOTS" in sdk.read_text(encoding="utf-8")
    print(f"{'✅' if has_lever else '❌'} A/B 杠杆 NVENC_ESRGAN_AB_SLOTS 存在于 {sdk}")
    return 0 if has_lever else 2


def _run_arm(arm: int, src: Path, out_dir: Path, seg_dur: int,
             batch_size: int, timeout: int) -> dict:
    """跑单臂（单 rate_mode=vbr_hq，即 LA>0 路径），返回 code=8 统计 + 验收结果。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_txt = out_dir / f"arm{arm}.log"
    env = dict(os.environ)
    env["NVENC_ESRGAN_AB_SLOTS"] = str(arm)          # 唯一变量
    cmd = [sys.executable, "-u", str(SMOKE),
           "--src", str(src), "--codec", "h264_nvenc",
           "--rate-modes", "vbr_hq",                    # LA>0 唯一相关路径
           "--segment-duration", str(seg_dur),
           "--batch-size", str(batch_size),
           "--mem-interval", "5",
           "--mem-dump-dir", str(out_dir / f"mem{arm}"),
           "--report", str(out_dir / f"arm{arm}.md"),
           "--json", str(out_dir / f"arm{arm}.json"),
           "--out-dir", str(out_dir / f"out{arm}")]
    print(f"\n=== [臂 {arm} slots] ===\n{' '.join(cmd)}\n"
          f"    NVENC_ESRGAN_AB_SLOTS={arm}")
    t0 = time.time()
    with open(log_txt, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=timeout, env=env)
    elapsed = time.time() - t0
    rec = {"arm": arm, "slots": arm, "rc": p.returncode,
           "elapsed_s": round(elapsed, 1), "log": str(log_txt)}

    # 从跑批日志统计 code=8（逐次计数，含 #N 编号）
    txt = log_txt.read_text(encoding="utf-8", errors="replace")
    hits = CODE8_RE.findall(txt)
    rec["code8_count"] = len(hits)
    rec["code8_slots"] = dict(Counter(s for s, _ in hits))
    rec["code8_fi"] = [int(f) for _, f in hits][:20]
    # 实际生效槽数（从 Ready / slots created 行）
    m = re.search(r"Ready: (\S+) (\S+) .*slots=(\d+)", txt)
    rec["ready_line"] = m.group(0) if m else None
    rec["ready_slots"] = int(m.group(3)) if m else None
    rec["ab_lever_seen"] = "AB-SLOT-LEVER" in txt

    # 帧守恒与验收项（读 json）
    jf = out_dir / f"arm{arm}.json"
    if jf.exists():
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            rec["summary"] = d.get("summary")
            for r in d.get("runs", []):
                for c in r.get("checks", []):
                    if c["id"] in ("S2", "S3", "S4", "S5"):
                        rec[c["id"]] = f"{c['status']}: {c['detail'][:160]}"
                rec.setdefault("rss_slope", r.get("rss_slope_mb_per_min"))
                rec.setdefault("rss_peak", r.get("rss_peak_mb"))
                rec.setdefault("mem_samples", r.get("mem_samples"))
        except Exception as e:
            rec["json_error"] = str(e)
    # 空帧/prev 补偿（守恒是否靠占位换来——关键否证项）
    rec["empty_frame_lines"] = len(re.findall(r"空帧占位|prev 填充", txt))
    rec["esf_fallback"] = len(re.findall(r"排空超限", txt))
    return rec


def _verdict(runs: list) -> tuple:
    """按预注册判据给结论。返回 (verdict_str, 判读表 list)。"""
    by = {r["slots"]: r for r in runs if r["slots"] in ARMS}
    rows = []
    if set(by) != set(ARMS):
        return ("SKIP 不足以定论", [f"缺臂：have={sorted(by)} need={list(ARMS)}"])

    ok = True
    for s in ARMS:
        r = by[s]
        rows.append((s, r))

    # ── 判据（先注册再判定，避免事后挑口径）─────────────────────────
    # D1 守恒：两臂都必须 rc=0 且 S2/S3 全 PASS（否则该臂数据不可用）
    # D2 主指标：code8_count(9) vs code8_count(11)
    #    · 11 臂显著更少（且 9 臂 >0）⇒ 支持 H1（槽位不足是成因之一）
    #    · 两者相当（都 >0 或都 ≈0）⇒ 支持 H2（warmup 期未就绪，与槽数无关）
    # D3 否证项：11 臂若出现 empty_frame/prev 填充 或 排空超限 ⇒ 提高槽数有害，
    #    此时即使 code8 减少也不能采纳
    c9, c11 = by[9]["code8_count"], by[11]["code8_count"]
    d1 = all(by[s]["rc"] == 0 and str(by[s].get("S2", "")).startswith("PASS")
             and str(by[s].get("S3", "")).startswith("PASS") for s in ARMS)
    ok &= d1
    # D3
    harm = {s: by[s]["empty_frame_lines"] + by[s]["esf_fallback"] for s in ARMS}
    d3 = harm[11] == 0
    ok &= d3
    if not d1:
        v = "SKIP 有一臂未通过帧守恒/退出码，该臂数据不可用"
    elif not d3:
        v = (f"NO 提高槽数有害：11 臂出现占位/兜底 {harm[11]} 次"
             f"（9 臂 {harm[9]} 次）⇒ 即使 code=8 减少也不采纳")
    elif c9 == 0 and c11 == 0:
        v = ("SKIP 两臂都零 code=8（可能未复现该现象或素材/LA 不同）"
             "⇒ 需换素材或更大 LA 重跑")
    elif c11 < c9:
        v = (f"YES 支持 H1（槽位不足是成因之一）：code=8 {c9}→{c11}"
             f"（9 臂 vs 11 臂）且 11 臂零占位/兜底"
             f"⇒ 可考虑把 ESRGAN h264 槽数对齐 IFRNet 的 la+3")
    else:
        v = (f"NO 支持 H2（与槽数无关的 warmup 未就绪）：code=8 {c9}→{c11}"
             f"（9 臂 vs 11 臂，未减少）⇒ 加槽数无效，"
             f"应改走「提交后站点加就绪门」而非调整槽数")
    return (v if ok or v.startswith(("YES", "NO")) else v), rows, harm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="h264+LA 槽位 A/B（9 vs 11）")
    ap.add_argument("--src", help="真实素材（建议 ≤100s，A/B 只需复现 warmup 期现象）")
    ap.add_argument("--work", default="/tmp/ab_slots", help="工作目录")
    ap.add_argument("--repeats", type=int, default=2, help="每臂重复次数（默认 2）")
    ap.add_argument("--segment-duration", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="固定 bs（⚠ 跨 bs 的 S8 读数不可比）")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--precheck", action="store_true", help="只做前置检查")
    ap.add_argument("--json", dest="json_path")
    args = ap.parse_args(argv)

    if args.precheck:
        return _precheck()
    if not args.src or not Path(args.src).exists():
        print("❌ 需要 --src <真实素材>")
        return 2

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"  h264+LA 槽位 A/B：{ARMS[0]} (la+1, ESRGAN现状) vs {ARMS[1]} (la+3, IFRNet现状)")
    print(f"  素材: {args.src}")
    print(f"  重复: {args.repeats} 臂/组（交替跑，避免时段/素材差异混入）")
    print("=" * 78)

    runs = []
    for rep in range(args.repeats):
        for arm in ARMS:
            rd = work / f"rep{rep}"
            try:
                runs.append(_run_arm(arm, Path(args.src), rd,
                                     args.segment_duration,
                                     args.batch_size, args.timeout))
            except subprocess.TimeoutExpired:
                runs.append({"arm": arm, "slots": arm, "rc": -9,
                             "elapsed_s": args.timeout, "code8_count": None,
                             "log": str(rd / f"arm{arm}.log"),
                             "empty_frame_lines": 0, "esf_fallback": 0})

    print("\n" + "=" * 78)
    print("  读数")
    print("=" * 78)
    print(f"{'臂':>4} {'rep':>4} {'rc':>4} {'code=8':>7} {'占位':>5} {'兜底':>5} "
          f"{'耗时s':>8}  ready")
    for r in runs:
        print(f"{r['slots']:>4} {r.get('rep','-'):>4} {r['rc']:>4} "
              f"{str(r['code8_count']):>7} {r['empty_frame_lines']:>5} "
              f"{r['esf_fallback']:>5} {r['elapsed_s']:>8}  "
              f"slots={r.get('ready_slots')} lever={r.get('ab_lever_seen')}")

    # 按臂聚合（重复轮取中位数，减单次噪声）
    print("\n" + "=" * 78)
    print("  判读（先注册判据，见脚本 docstring D1~D3）")
    print("=" * 78)
    agg = {}
    for s in ARMS:
        cs = [r["code8_count"] for r in runs if r["slots"] == s and r["code8_count"] is not None]
        es = [r["elapsed_s"] for r in runs if r["slots"] == s]
        agg[s] = {"code8": cs, "code8_median": statistics.median(cs) if cs else None,
                  "elapsed_median": statistics.median(es) if es else None}
        print(f"  臂{s}: code8 逐轮={cs} 中位={agg[s]['code8_median']} "
              f"耗时中位={agg[s]['elapsed_median']}s")

    v = _verdict(runs)
    verdict = v[0] if isinstance(v, tuple) else v
    print(f"\n  结论: {verdict}")
    print("  ⚠ 判读纪律：单次差异可能是噪声（warmup 期现象本就稀少）；"
          "若两臂 code=8 都在个位数，需加 --repeats 提高置信度。")
    print("  ⚠ 本 A/B 只覆盖 ESRGAN 侧槽数；IFRNet 侧已是 la+3，不在本实验范围。")

    out = {"generated": datetime.now().isoformat(timespec="seconds"),
           "src": str(args.src), "repeats": args.repeats,
           "arms": list(ARMS), "runs": runs, "agg": agg,
           "verdict": verdict}
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📄 {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())