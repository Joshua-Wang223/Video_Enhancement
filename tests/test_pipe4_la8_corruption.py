#!/usr/bin/env python3
"""
pipe=4 + LA=8: Production vs Test-Script Differential Analysis
===============================================================
已知:
  ✅ test_nvenc_completion_event_v4.py: pipe=4, LA=8, VBR_HQ → 零空帧, 零丢帧
    配置: 纯 ce_pipeline, fi==0 force_idr, 无 encode_frame, 无 SPS/PPS

  ❌ production v6.4.3.1/4.1/5.1: pipe=4, LA=8, VBR_HQ/QVBR → 丢31-57帧, 花屏
    配置: encode_frame + ce_pipeline + per-slot IDR + SPS/PPS V3

  ❌ production v6.4.3.1/4.1/5.1: pipe=1, LA=8, VBR_HQ/QVBR → 丢5-7帧, 视频正常
    配置: encode_frame + ce_pipeline + per-slot IDR (pipe=1 只有1槽)

本脚本不模拟 NVENC 内部行为，而是对比以下差异的生产代码语义:

  差异 1 (Slot0 handle alias):
    encode_frame() 使用 _input_buf_handle (= slots[0].input_buf)
    ce_pipeline()  也使用 slots[0].input_buf
    即使在正常操作下, 两个调用者交替使用同一 handle 也增加了
    NVENC 内部状态同步的复杂度, 特别是 LA buffering 后.

  差异 2 (Per-slot force_idr):
    _slots_warmed 在 pipe=4 下产生 4 个 IDR, pipe=1 下仅 1 个.
    每个 IDR 触发 LA 帧类型重决策.
    4× IDR → 4× LA 开销 → 更多空帧.

  差异 3 (SPS/PPS 重复):
    Per-slot IDR 每槽可能触发 _cached_sps_pps 操作.
    由于 ctypes repeatSPSPPS bug, NVENC 不输出 SPS+PPS 到后续 IDR.
    Per-slot prepend 将 slot[0] 的 SPS+PPS 注入到其他槽的 IDR →
    可能造成参数不匹配 (不同槽的 DPB 状态不同).

模拟方法: 纯状态机追踪.
  每条 EncodePicture 提交 → 记录 (caller, slot, force_idr, SPS action)
  对比各场景的: handle 共享次数, IDR 分布, SPS 注入模式

Usage: python tests/test_pipe4_la8_corruption.py
"""

import sys, io
from typing import Set, Dict, List, Tuple

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ------------------------------------------------------------------
# Frame-level trace
# ------------------------------------------------------------------
Frame = Dict  # {fid, slot, caller, force_idr, sps_action, handle_shared}

class Tracer:
    """追踪每条 EncodePicture 的关键属性"""

    def __init__(self, pipe_depth: int = 4, la_depth: int = 8):
        self.pd = pipe_depth
        self.la = la_depth
        self.fi = 0
        self.frames: List[Frame] = []
        self.slots_warmed: Set[int] = set()
        self._cached_sps_pps = None
        self._sps_pps_injected = False
        self.sps_actions: List[dict] = []

        # Legacy ref (slot 0)
        self.legacy_slot = 0

    # -- encode_frame --
    def encode_frame(self, force_idr: bool = False):
        fid = self.fi; self.fi += 1
        slot = self.legacy_slot
        caller = "enc_frame"
        sps = self._sps(force_idr, fid, slot)
        self.frames.append({
            'fid': fid, 'slot': slot, 'caller': caller,
            'force_idr': force_idr, 'sps': sps,
            'handle_shared_with_ce': True,  # slot 0 is shared
        })

    # -- ce_pipeline batch --
    def encode_batch_ce(self, bs: int = 24, force_idr_first: bool = False):
        batch = []
        for _ in range(bs):
            si = self.fi % self.pd
            fid = self.fi; self.fi += 1
            caller = f"ce_s{si}"

            force_idr = force_idr_first and (si not in self.slots_warmed)
            if force_idr:
                self.slots_warmed.add(si)

            sps = self._sps(force_idr, fid, si)
            self.frames.append({
                'fid': fid, 'slot': si, 'caller': caller,
                'force_idr': force_idr, 'sps': sps,
                'handle_shared_with_ce': (si == self.legacy_slot),
            })
            batch.append(fid)
        return batch

    def _sps(self, force_idr: bool, fid: int, si: int) -> str:
        if not force_idr:
            return "none"
        if self._cached_sps_pps is None:
            self._cached_sps_pps = f"sps_pps(fid={fid})"
            self.sps_actions.append({'action': 'cache', 'fid': fid, 'slot': si})
            return "cache_new"
        if not self._sps_pps_injected:
            self._sps_pps_injected = True
            self.sps_actions.append({'action': 'prepend_inject', 'fid': fid, 'slot': si})
            return "prepend"
        return "already_cached"

    # -- 分析 --
    def analyze(self) -> dict:
        frames = self.frames

        # 1. Slot0 handle sharing instances
        slot0_calls = [f for f in frames if f['slot'] == 0]
        enc_calls_on_slot0 = [f for f in slot0_calls if f['caller'] == 'enc_frame']
        ce_calls_on_slot0  = [f for f in slot0_calls if f['caller'].startswith('ce')]

        # 2. IDR distribution
        idrs = [f for f in frames if f['force_idr']]
        idr_by_slot = {}
        for f in idrs:
            sl = f['slot']
            idr_by_slot[sl] = idr_by_slot.get(sl, 0) + 1

        # 3. SPS action distribution
        sps_by_slot = {}
        for a in self.sps_actions:
            sl = a['slot']
            sps_by_slot[sl] = sps_by_slot.get(sl, 0) + 1

        # 4. Estimate LA overhead: each IDR needs la_depth frames of buffering
        # In production: LA re-fills on each IDR (frame type re-decision)
        # With per-slot IDR ×4 → 4 separate LA fill cycles
        la_overhead_estimate = len(idrs) * self.la if idrs else 0

        return {
            'fi': self.fi,
            'total_frames': len(frames),
            'slot0_total_calls': len(slot0_calls),
            'slot0_enc_calls': len(enc_calls_on_slot0),
            'slot0_ce_calls': len(ce_calls_on_slot0),
            'idr_total': len(idrs),
            'idr_by_slot': idr_by_slot,
            'sps_by_slot': sps_by_slot,
            'sps_actions': self.sps_actions,
            'la_overhead': la_overhead_estimate,
        }


# ------------------------------------------------------------------
# 场景
# ------------------------------------------------------------------

def run(label, pd, la, use_enc_frame, per_slot_idr):
    t = Tracer(pipe_depth=pd, la_depth=la)
    if use_enc_frame:
        t.encode_frame(force_idr=True)
    for bn in range(29):
        t.encode_batch_ce(24, force_idr_first=(bn == 0 and per_slot_idr))
    return t

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("pipe=4 + LA=8: Production vs Test-Script Differential Analysis")
    print("=" * 72)

    scenarios = [
        ("T0 BUG",       4, 8, True,  True,  "enc_frame + per-slot IDR"),
        ("T1",            4, 8, True,  False, "enc_frame + fi==0 only"),
        ("T2",            4, 8, False, True,  "no enc_frame + per-slot IDR"),
        ("T3 GOAL",      4, 8, False, False, "no enc_frame + fi==0 only"),
        ("T4 pipe=1",    1, 8, True,  True,  "pipe=1 baseline"),
        ("T5 REFERENCE", 4, 8, False, False, "test script (known-good)"),
    ]

    all_tracers = {}
    all_stats = {}

    for key, pd, la, ef, psi, label in scenarios:
        t = run(label, pd, la, ef, psi)
        all_tracers[key] = t
        s = t.analyze()
        all_stats[key] = s

        # Show trace of first 16 frames
        print(f"\n  {label} (pipe={pd}, la={la})")
        header = f"  {'fid':>4} {'slot':>4} {'caller':<12} {'IDR?':>4} {'SPS':<15} {'shared?'}"
        print(header)
        print("  " + "-" * 58)
        for f in t.frames[:16]:
            sps = f['sps'] if f['sps'] != 'none' else '-'
            shared = 'Y' if f['handle_shared_with_ce'] else '-'
            idr = 'IDR' if f['force_idr'] else '-'
            print(f"  {f['fid']:>4} {f['slot']:>4} {f['caller']:<12} {idr:>4} {sps:<15} {shared:>4}")
        if len(t.frames) > 16:
            print(f"  ... ({len(t.frames) - 16} more frames)")

    # ═══ 决策矩阵 ═══
    print()
    print("=" * 72)
    print("Decision Matrix")
    print("=" * 72)
    print(f"  {'Key':<12} {'slot0 calls':>11} {'enc on slot0':>12} {'IDRs':>5} "
          f"{'IDR/slot':>8} {'SPS actions':>11} {'Est LA cost':>11}")
    print("  " + "-" * 72)
    for key, s in all_stats.items():
        idr_dist = str(dict(sorted(s['idr_by_slot'].items())))
        sps_dist = str(dict(sorted(s['sps_by_slot'].items())))
        print(f"  {key:<12} {s['slot0_total_calls']:>11} {s['slot0_enc_calls']:>12} "
              f"{s['idr_total']:>5} {idr_dist:<8} {sps_dist:<11} {s['la_overhead']:>11}")

    # ═══ 分析 ═══
    print()
    print("── Analysis ──")
    t0 = all_stats["T0 BUG"]
    t3 = all_stats["T3 GOAL"]
    t5 = all_stats["T5 REFERENCE"]

    print(f"  T0 BUG: slot0 shared by enc_frame + ce_pipeline = {t0['slot0_enc_calls']} + {t0['slot0_ce_calls']} calls")
    print(f"          per-slot IDR: {t0['idr_total']} IDs across {t0['idr_by_slot']}")
    print(f"          SPS: {t0['sps_actions']}")
    print(f"          Est LA overhead: {t0['la_overhead']} buffered frames")

    print(f"\n  T3 GOAL: slot0 shared by enc_frame + ce_pipeline = 0 + {t3['slot0_ce_calls']} (no enc_frame)")
    print(f"           per-slot IDR: {t3['idr_total']} (fi==0 only)")
    print(f"           SPS: {t3['sps_actions']}")
    print(f"           Est LA overhead: {t3['la_overhead']}")

    print(f"\n  T5 test script (known-good):")
    print(f"           slot0 shared = 0 enc_frame calls")
    print(f"           IDR: {t5['idr_total']} (fi==0 only)")
    print(f"           SPS: {t5['sps_actions']}")

    print()
    print("  Differential from T0→T3:")
    s0 = t0['slot0_enc_calls']  # enc_frame on slot0
    idr_diff = t0['idr_total'] - t3['idr_total']
    la_diff = t0['la_overhead'] - t3['la_overhead']
    print(f"    - Eliminates {s0} encode_frame calls on slot0 handle")
    print(f"    - Eliminates {idr_diff} per-slot IDRs")
    print(f"    - Reduces LA overhead by {la_diff} buffered frames")
    print(f"    - T5 test script proves: fi==0 only IDR + no enc_frame = SAFE")

    print()
    print("  ★ Recommended fix: align production code with test script behavior")
    print("    1. Remove encode_frame() in _process_segment")
    print("    2. _slots_warmed → fi==0 only force_idr")
    print("    3. Comment out pipe forced 4→1 guard")
    print()
    print("done.")


if __name__ == "__main__":
    main()
