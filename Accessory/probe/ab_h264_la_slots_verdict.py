#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 结果判读器 —— 独立复算预注册判据 D1/D2/D3，并给出判读表。

与 `ab_h264_la_slots.py` 的判据保持同一口径，但**从原始日志重新提取**，
不信任 A/B 脚本自己汇总的字段（避免"报告这么说"式自证）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

CODE8_RE = re.compile(r"drain LockBitstream code=8 \(slot=(\d+), fi=(\d+)\)")
#: Ready 行：`Ready: 720x576@50.0fps H.264 VBR_HQ ... slots=11 la=8 ...`
#: ⚠ **必须按分辨率分组**：同一次跑批有两个编码器实例 ——
#: 插帧段 720x576（IFRNet 侧，**恒为 la+3=11**，与本实验无关）
#: 超分段 1440x1152（ESRGAN 侧 = 本实验变量，9 vs 11）
#: 混在一起看会得出「两臂都是 11 ⇒ 杠杆没生效」的错误结论。
READY_RE = re.compile(r"Ready: (\d+x\d+)@([\d.]+)fps .*?slots=(\d+) la=(\d+)")
LEVRE_RE = re.compile(r"\[AB-SLOT-LEVER\].*?强制为 (\d+)")
#: 占位/兜底—— D3 否证项的原始信号
LOSSY_RE = re.compile(r"空帧占位|prev 填充|排空超限")
#: 帧守恒原始信号
GUARD_RE = re.compile(r"decoded=(\d+) expected=(\d+)")


def scan(log: Path) -> dict:
    txt = log.read_text(encoding="utf-8", errors="replace")
    c8 = CODE8_RE.findall(txt)
    ready = READY_RE.findall(txt)
    # 按分辨率归类 slots：esrgan = 超分段（本实验变量），ifrnet = 插帧段（对照，恒 11）
    slots_by_res = {r[0]: int(r[2]) for r in ready}
    # 本实验变量 = **超分段**（分辨率较大者 = 插帧输出 2x 超分 ⇒ 面积最大）的槽数。
    # ⚠ 不能用 max(slots_by_res.values()) —— 插帧段恒为 11（IFRNet 本就 la+3），
    # max() 会永远返回 11从而掩盖两臂差异（本轮踩过）。
    _res_sorted = sorted(slots_by_res, key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1]))
    esrgan_res = _res_sorted[-1] if _res_sorted else None
    return {
        "log": str(log),
        "code8_total": len(c8),
        "code8_by_res": {},
        "ready": [f"{r[0]}@{r[1]}fps slots={r[2]} la={r[3]}" for r in ready],
        "slots_by_res": slots_by_res,
        "esrgan_res": esrgan_res,
        "esrgan_slots": slots_by_res.get(esrgan_res) if esrgan_res else None,
        "ifrnet_slots": slots_by_res.get(_res_sorted[0]) if len(_res_sorted) > 1 else None,
        "lever": sorted(set(LEVRE_RE.findall(txt))),
        "lossy_lines": LOSSY_RE.findall(txt),
        "lossy_total": len(LOSSY_RE.findall(txt)),
        "conservation": GUARD_RE.findall(txt),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B 结果判读（独立复算）")
    ap.add_argument("--work", default="/tmp/ab_slots", help="A/B 工作目录")
    ap.add_argument("--expect", default="9,11", help="期望的两臂槽位")
    args = ap.parse_args(argv)
    work = Path(args.work)
    arms = [int(x) for x in args.expect.split(",")]

    # 收集 rep*/out{arm}/*.log（**管线日志**才是 code=8 的来源；
    # rep*/arm{arm}.log 只是冒烟脚本自身 stdout，不含管线输出 —— 别读错文件）
    # ⚠ **丢弃「空/未完成」的日志**：跑批进行中时管线日志尚未落盘（或只有表头），
    # 此时 code8_count 恒为 0 —— 若当成真实 0 计入，会凭空造出「11 臂 code8 更少」
    # 的假 YES（2026-10-05 实测踩到：臂11 rep1 仍在跑→计0→中位被拉低）。
    # 判据：日志里必须同时出现 Ready 行（证明管线真的启动过）与至少一条段级守恒记录。
    raw = {}
    found = {}
    for arm_dir in sorted(work.glob("rep*/out*")):
        m = re.match(r"out(\d+)$", arm_dir.name)
        if not m:
            continue
        arm = int(m.group(1))
        for log in sorted(arm_dir.glob("*.log")):
            s = scan(log)
            complete = bool(s["ready"]) and bool(s["conservation"])
            s["complete"] = complete
            raw.setdefault(arm, []).append((log.parent.parent.name, s))
    for arm, items in raw.items():
        kept = [(r, s) for r, s in items if s["complete"]]
        for r, s in items:
            if not s["complete"]:
                print(f"  ⚠ 跳过未完成/空日志: {s['log']}"
                      f"（ready={len(s['ready'])} 守恒记录={len(s['conservation'])}）"
                      f"—— 多为跑批仍在进行")
        if kept:
            found[arm] = [s for _, s in kept]
    if not found:      # 回退：兼容只有 arm*.log 的旧布局
        for log in sorted(work.glob("rep*/arm*.log")):
            m = re.match(r"arm(\d+)\.log$", log.name)
            if m:
                found.setdefault(int(m.group(1)), []).append(scan(log))

    print("=" * 84)
    print("  A/B 判读（从原始日志独立复算）")
    print("=" * 84)
    if not found:
        print(f"❌ {work} 下无 arm*.log")
        return 2

    for arm in sorted(found):
        print(f"\n【臂 {arm}】{len(found[arm])} 轮")
        for i, s in enumerate(found[arm]):
            print(f"  rep{i}: code8={s['code8_total']:>3}  "
                  f"超分段slots={s['esrgan_slots']}  lever={s['lever']}  "
                  f"占位/兜底={s['lossy_total']}")
            print(f"        Ready 明细: {s['ready']}")
            if s["code8_by_res"]:
                print(f"        code8 分布: {s['code8_by_res']}")
            bad = [c for c in s["conservation"] if c[0] != c[1]]
            print(f"        段级守恒: {len(s['conservation'])} 条, 不守恒 {len(bad)} 条"
                  + (f" {bad[:3]}" if bad else " ✓"))

    # ── 预注册判据 ────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  预注册判据复算")
    print("=" * 84)
    agg = {}
    for arm in arms:
        ss = found.get(arm) or []
        if not ss:
            print(f"臂 {arm}: 数据缺失 ⇒ SKIP")
            continue
        c8 = [s["code8_total"] for s in ss]
        ls = [s["lossy_total"] for s in ss]
        cons_bad = sum(1 for s in ss for c in s["conservation"] if c[0] != c[1])
        eff = sorted({x for s in ss for x in [s["esrgan_slots"]] if x})
        agg[arm] = {"code8": c8, "code8_med": statistics.median(c8) if c8 else None,
                    "lossy": ls, "lossy_med": statistics.median(ls) if ls else None,
                    "cons_bad": cons_bad, "eff_slots": eff}
        print(f"臂 {arm}: code8 逐轮={c8} 中位={agg[arm]['code8_med']} | "
              f"占位/兜底 逐轮={ls} | 段级不守恒={cons_bad} | 超分段生效slots={eff}")

    if len(agg) != 2:
        print("\n结论: SKIP 两臂数据不完整")
        return 1
    a, b = arms
    A, B = agg[a], agg[b]

    # D1：①两臂段级守恒无异常 ②两臂超分段槽数**确实不同**（否则杠杆未生效，实验无效）
    d1_cons = (A["cons_bad"] == 0 and B["cons_bad"] == 0)
    sa = A["eff_slots"][0] if len(A["eff_slots"]) == 1 else None
    sb = B["eff_slots"][0] if len(B["eff_slots"]) == 1 else None
    d1_diff = (sa is not None and sb is not None and sa != sb)
    d1 = d1_cons and d1_diff
    print(f"\nD1 守恒: {'✓' if d1_cons else '✗'} (A={A['cons_bad']} B={B['cons_bad']} 条不守恒)")
    print(f"D1 杠杆确已改变超分段槽数: {'✓' if d1_diff else '✗'} "
          f"(臂{a}={sa} vs 臂{b}={sb}；期望 {a} vs {b})")
    if not d1_cons:
        print("   ⚠ 存在段级不守恒 ⇒ 该臂数据不可用")
    if not d1_diff:
        print("   ⚠ 两臂超分段槽数相同或缺失 ⇒ 杠杆未生效/ 日志不全，实验无效"
              "（先查 AB-SLOT-LEVER 日志与 out*/ 管线日志是否落盘）")
    # D3（否证项，优先于 D2）
    d3 = (B["lossy_med"] == 0)
    print(f"D3 否证项（11 臂零占位/兜底）: {'✓ 通过' if d3 else '✗ 违反'}"
          f" (11 臂逐轮={B['lossy']})")
    # D2
    print(f"D2 主指标 code8: 臂{a}={A['code8_med']} → 臂{b}={B['code8_med']}")

    print("\n" + "-" * 84)
    if not d1:
        v = "SKIP 实验无效（超分段槽数未真正改变，或存在段级不守恒）"
    elif not d3:
        v = ("NO 提高槽数有害：11 臂出现占位/兜底 ⇒ 即使 code=8 减少也不采纳")
    elif A["code8_med"] and B["code8_med"] is not None and B["code8_med"] < A["code8_med"]:
        v = (f"YES 支持『槽位不足是成因之一』: code8 {A['code8_med']}→{B['code8_med']} "
             f"且 11 臂零占位 ⇒ 可考虑对齐 IFRNet 的 la+3")
    elif A["code8_med"] == 0 and B["code8_med"] == 0:
        v = "SKIP 两臂都零 code=8 ⇒ 现象未复现（换素材/更大 LA），本轮无结论"
    else:
        v = (f"NO 支持『与槽数无关的 warmup 未就绪』: code8 {A['code8_med']}→"
             f"{B['code8_med']} 未减少 ⇒ 加槽数无效，应走提交后就绪门")
    print(f"结论: {v}")
    print("-" * 84)
    print("⚠ 纪律：code=8 是 warmup 期稀有现象，个位数差异可能是噪声；")
    print("  若两臂都在个位数，应加 --repeats 提高置信度后再下结论。")
    print("⚠ 本实验只覆盖 ESRGAN 侧槽数；IFRNet 侧本就是 la+3，不在范围内。")
    return 0


if __name__ == "__main__":
    sys.exit(main())