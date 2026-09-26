#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe_Optimisation_batch_size —— 搜索逻辑的纯 CPU 自测（不需要 GPU / torch / numpy）。

GPU 环境才能跑真正的探测（见文件末尾「GPU 验证清单」），但「粗扫找拐点 →
步进折半精调 → 回归」这套搜索逻辑是纯 Python，用桩对象即可离线验证：
  · 拐点由 FPS 增益阈值触发
  · 拐点由 GPU 利用率饱和触发
  · 拐点由 OOM / 显存护栏触发（并回退到上一个安全点）
  · 精调能在拐点附近收敛到真正的最优点（含向下回归）
  · 平手时按 GPU 利用率 → 显存 → 更小 batch_size 取舍

运行：python3 Accessory/test/test_probe_batch_size_search_logic.py
"""

import sys
import types

# numpy / torch 只在本模块被 import 时需要，桩掉即可加载搜索逻辑部分
for _name in ("numpy", "torch"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

_HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, _HERE)
# 被测脚本在同级 Accessory/probe/ 下（分类目录不同，需显式加路径）
sys.path.insert(0, _HERE.rsplit("/", 1)[0] + "/probe")
import batch_size_optimizer_probe as probe  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.start_bs = 8
        self.coarse_step = 8
        self.max_bs = 96
        self.gain_threshold = 0.02
        self.util_gain_threshold = 1.0
        self.min_improve = 0.005
        self.max_moves = 3
        self.vram_limit = 0.85
        self.decide_by = "e2e"
        self.__dict__.update(kw)


class _StubProber:
    """按给定的 fps/util 映射伪造 measure()；未列出的 bs 视为 OOM。

    util 默认 -1（未采到）→ 不触发利用率饱和判据，便于单独验证其它判据。
    """

    def __init__(self, table, total_vram_gb=16.0, util_table=None):
        self.table = table
        self.util_table = util_table or {}
        self.results = {}
        self.failed = set()
        self.total_vram_gb = total_vram_gb

    def measure(self, bs, fresh=False):
        if bs in self.results and not fresh:
            return self.results[bs]
        if bs in self.table:
            fps = self.table[bs]
            res = probe.ProbeResult(batch_size=bs, fps_e2e=fps, fps_pure=fps,
                                    gpu_util_avg=self.util_table.get(bs, -1.0),
                                    proc_per_video_sec=24.0 / fps)
        else:
            res = probe.ProbeResult(batch_size=bs, status="oom", note="stub OOM")
        if res.status != "ok":
            self.failed.add(bs)
        self.results[bs] = res
        return res


def _run(fn):
    """静默执行（搜索函数的 print 用 builtins，这里临时替换）。"""
    import builtins
    out = []
    old = builtins.print
    builtins.print = lambda *a, **k: out.append(" ".join(str(x) for x in a))
    try:
        return fn(), out
    finally:
        builtins.print = old


def case_coarse_gain_knee():
    """FPS 增益跌破阈值 → 拐点。bs=32 只比 24 快 0.5%。"""
    table = {8: 100.0, 16: 190.0, 24: 270.0, 32: 271.4, 40: 300.0}
    p = _StubProber(table)
    knee, _ = probe.coarse_scan(p, _Args())
    assert knee == 32, knee
    return knee


def case_coarse_util_saturation():
    """FPS 仍在涨，但 GPU 利用率已饱和（+0.2pp < 1pp）→ 判拐点，不再往上探。"""
    table = {8: 100.0, 16: 190.0, 24: 270.0, 32: 340.0, 40: 420.0}
    util = {8: 60.0, 16: 80.0, 24: 97.0, 32: 97.2, 40: 97.4}
    p = _StubProber(table, util_table=util)
    knee, _ = probe.coarse_scan(p, _Args())
    assert knee == 32, knee
    assert 40 not in p.results, "利用率饱和后不应再往上测"
    return knee


def case_coarse_oom_fallback():
    """bs=24 OOM → 拐点回退到上一个安全点 16。"""
    p = _StubProber({8: 100.0, 16: 190.0})
    knee, _ = probe.coarse_scan(p, _Args())
    assert knee == 16, knee
    assert 24 in p.failed
    return knee


def case_refine_finds_true_optimum():
    """真最优在 29（粗扫点都不覆盖）：粗扫拐点 32 → 精调逐级折半向下回归到 29。"""
    table = {
        8: 100.0, 16: 190.0, 24: 270.0, 32: 271.0,   # 粗扫点：32 处增益仅 0.37%
        28: 278.0, 36: 272.0,                        # 步进 4
        26: 272.0, 30: 285.0,                        # 步进 2
        29: 300.0, 31: 280.0,                        # 步进 1
    }
    p = _StubProber(table)
    knee, _ = probe.coarse_scan(p, _Args())
    assert knee == 32, knee
    best = probe.refine(p, _Args(), knee)
    assert best == 29, best
    return best


def case_pick_best_tiebreak():
    """FPS 平手（差距 < min_improve）→ 取 GPU 利用率更高者。"""
    p = _StubProber({}, util_table={})
    p.results = {
        16: probe.ProbeResult(batch_size=16, fps_e2e=200.0, gpu_util_avg=80.0,
                              peak_alloc_gb=4.0),
        24: probe.ProbeResult(batch_size=24, fps_e2e=200.5, gpu_util_avg=95.0,
                              peak_alloc_gb=6.0),
    }
    assert probe.pick_best(p, [16, 24], _Args()) == 24
    # 利用率都相同时 → 取显存更低 / batch_size 更小者
    for r in p.results.values():
        r.gpu_util_avg = 90.0
    assert probe.pick_best(p, [16, 24], _Args()) == 16
    return True


def main() -> int:
    cases = [
        ("粗扫-FPS增益拐点", case_coarse_gain_knee),
        ("粗扫-利用率饱和拐点", case_coarse_util_saturation),
        ("粗扫-OOM回退", case_coarse_oom_fallback),
        ("精调-向下回归到真最优", case_refine_finds_true_optimum),
        ("平手-利用率/显存/bs 取舍", case_pick_best_tiebreak),
    ]
    failed = 0
    for name, fn in cases:
        try:
            val, _logs = _run(fn)
            print(f"  PASS  {name}  → {val}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}  → {exc}")
        except Exception as exc:  # 不吞上下文
            failed += 1
            print(f"  ERROR {name}  → {type(exc).__name__}: {exc}")
    print(f"\nCPU 逻辑自测：{len(cases) - failed}/{len(cases)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Probe_Optimisation_batch_size —— 搜索逻辑 CPU 自测")
    sys.exit(main())
