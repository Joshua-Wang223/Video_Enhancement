#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVENC `qp=0`（CRF=0 → 强制 CONSTQP）段错误复现与定位 —— 可复用诊断。

## 为什么有这个脚本

`Plan/NVENC硬件测试隔离_立项Prompt.md` 原本把 `pytest tests/` 的 SIGSEGV 归因为
**跨测试类的状态污染**，并以「逐文件隔离」为方案 A。2026-09-16 在 Linux + T4
（驱动 580.65.06 / CUDA 13.0）实测**推翻**了该归因：

- `tests/test_nvenc_sdk_realesrgan.py` **单独**跑也会崩（3/8），
  连**单个测试**独立循环也崩（3/10）；
- 崩溃与测试顺序无关，与前置文件无关。

真正的触发条件是**编码器配置 `NVENCEncoder(qp=0, rate_mode='constqp',
la_depth=0)` 再编码**（即 `crf=0` 无损路径，见 `memory/crf0-la-depth-cli-ignore-fix.md`）。
本脚本用二分把该结论固定下来，便于换机器/换驱动后复验。

## 三种模式（每次都在**独立子进程**里跑，因为 SEGV 会杀掉进程）

| 模式 | 内容 | 实测（T4/CUDA13.0，15 次） |
|---|---|---|
| `ctor_only` | 建会话 + `close()`，**不编码** | 段错误 **0/15** |
| `ce_pipeline` | 建会话 + `encode_frames_batch_ce_pipeline`（**生产 LA=0 入口**） | 段错误 **10/15** |
| `batch_direct` | 建会话 + `encode_frames_batch`（生产 LA>0 入口，此处 LA=0） | 段错误 **4/10** |

⇒ 故障在**编码阶段**，与建会话无关、与入口函数无关。

## 对照：同一路径、只改 qp

| qp | rate_mode | la | 实测 |
|---|---|---|---|
| 0 | constqp | 0 | 段错误 3/10（另一次 rc=1） |
| 23 | constqp | 0 | **0/10** |

⇒ 触发条件是 **`qp=0`（无损）**，不是 constqp / LA=0 本身。

## 崩溃现场特征（堆损坏，非单点）

- `faulthandler` 在不同次运行落到不同帧：`_drain_outputs_blocking` 的
  `lock_bs_fn(...)`（驱动调用）、`_is_legal_bitstream_size` 之后读结果、
  pytest 的 `code.py`；
- `gdb` 原生栈另见 `gc_collect_main → subtract_refs → visit_decref →
  _PyObject_IS_GC(obj=<坏指针>)`，即**GC 遍历时碰到已损坏对象**；
- 崩溃点还出现在 `close()` 之后的 `del`/`gc.collect()` 与解释器退出
  （`atexit`）阶段 —— 说明是**进程内堆/驱动状态被破坏**，只是显现位置随机。

## 已排除的假设（避免重复调查）

| 假设 | 排除依据 |
|---|---|
| `_NvEncPresetConfig` / `_NvEncConfig` 尺寸偏小越界 | 已修正为真实布局（5128 / 3584），崩溃率**未改善**（2-3/10 → 4/10，同量级）。驱动实际只写 `presetCfg`（@8 起 3584B，止于 3592 < 5128），从未越界 |
| `_FUNC_IDX` 索引错位 | 逐条比对 `/usr/include/ffnvcodec/nvEncodeAPI.h` 的 `NV_ENCODE_API_FUNCTION_LIST` 偏移，21 条**全部正确** |
| `LockBitstream` 垃圾 size → `from_address` 越界读 | 5 处 `from_address` 站点**全部**有 `_is_legal_bitstream_size` 前置 |
| `_slot_pending` 元组形状不一致 | 4 元组（`encode_frames_batch`）与 5 元组（`ce_pipeline`）**按路径各自自洽**，仅跨路径混用才会错 |
| `close()` 后释放 CUDA 张量引发 | 不释放张量/不主动 GC 也照样崩（含 2 次在 `atexit`） |
| 缓冲区尺寸类（`NV_ENC_PIC_PARAMS` 3360、`NV_ENC_LOCK_BITSTREAM` 1544、`NV_ENC_CREATE_*` 776、`CUDA_MEMCPY2D` 128） | 逐个 gcc `sizeof` 实测，与代码**完全一致** |

## 有用的诊断杠杆

`MALLOC_CHECK_=3 MALLOC_PERTURB_=165 PYTHONMALLOC=malloc` 下实测 **0/5 崩溃**
（仅 5 次样本，不能作定论），可作为后续定位堆损坏的入口。

## 跑法

    python tests/diagnose_nvenc_qp0_segv.py                 # 三模式各 15 次
    python tests/diagnose_nvenc_qp0_segv.py --iters 30
    python tests/diagnose_nvenc_qp0_segv.py --modes ctor_only ce_pipeline
    python tests/diagnose_nvenc_qp0_segv.py --qp 23         # 对照组

退出码：0 = 未观测到段错误；1 = 观测到（表示缺陷在本次环境仍可复现）。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 子进程里跑的一端：一次完整的「建会话 → [编码] → close」
_CHILD_SRC = r'''
import os, sys, torch, importlib.util
sys.path.insert(0, os.path.join(%(root)r, "external"))
sys.path.insert(0, os.path.join(%(root)r, "external", "realesrgan_video"))
spec = importlib.util.spec_from_file_location(
    "nvenc_sdk", os.path.join(%(root)r, "external", "realesrgan_video", "nvenc_sdk.py"))
m = importlib.util.module_from_spec(spec); sys.modules["nvenc_sdk"] = m
spec.loader.exec_module(m)

MODE = os.environ["MODE"]
QP = int(os.environ["QP"])
N = int(os.environ.get("NFRAMES", "16"))

def mk(s):
    print("MARK:" + s, flush=True)

mk("import_ok")
enc = m.NVENCEncoder(176, 144, 30.0, qp=QP, rate_mode="constqp", la_depth=0)
mk("ctor_ok")
if MODE != "ctor_only":
    frames = [torch.randint(0, 255, (216, 176), dtype=torch.uint8, device="cuda")
              for _ in range(N)]
    mk("frames_ok")
    if MODE == "ce_pipeline":
        res = enc.encode_frames_batch_ce_pipeline(frames, True)   # 生产 LA=0 入口
    else:
        res = enc.encode_frames_batch(frames, force_idr_first=True)
    mk("encode_ok len=%%d empty=%%d none=%%d" %% (
        len(res), sum(1 for r in res if not r), sum(1 for r in res if r is None)))
enc.close()
mk("close_ok")
mk("end_of_script")
''' % {"root": str(ROOT)}

_MODES = ("ctor_only", "ce_pipeline", "batch_direct")


def _run_once(mode: str, qp: int, timeout: int) -> dict:
    env = os.environ.copy()
    env.update({"MODE": mode, "QP": str(qp)})
    try:
        p = subprocess.run([sys.executable, "-c", _CHILD_SRC],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT), env=env)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, out = None, ""
    marks = [ln for ln in out.splitlines() if ln.startswith("MARK:")]
    last = marks[-1][5:] if marks else None
    enc = next((x for x in marks if x.startswith("MARK:encode_ok")), None)
    # subprocess 对被信号杀死的子进程返回**负信号号**（SIGSEGV → -11）；
    # 经 shell 包装时可能看到 139。两者都归为 SEGV。
    if rc in (-11, 139):
        state = "SEGV"
    elif rc is None:
        state = "HANG"
    elif rc < 0:
        state = "signal%d" % (-rc)
    elif rc == 0:
        state = "ok"
    else:
        state = "rc=%s" % rc
    return {"state": state, "last_mark": last,
            "encode_line": enc[5:] if enc else None, "rc": rc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=15, help="每种模式迭代次数（默认 15）")
    ap.add_argument("--qp", type=int, default=0, help="QP（默认 0=触发条件；23=对照）")
    ap.add_argument("--modes", nargs="+", default=list(_MODES), choices=_MODES)
    ap.add_argument("--timeout", type=int, default=120, help="单次子进程超时秒数")
    args = ap.parse_args()

    print("NVENC qp=%d 段错误复现诊断  (%d 次/模式)" % (args.qp, args.iters))
    print("  python=%s" % sys.version.split()[0])
    if not os.path.isdir(str(ROOT / "external" / "realesrgan_video")):
        print("  ⚠️ 找不到 external/realesrgan_video，路径假设不成立"); return 2

    total_segv = 0
    for mode in args.modes:
        tally = {}
        empties = []
        for i in range(1, args.iters + 1):
            r = _run_once(mode, args.qp, args.timeout)
            tally[r["state"]] = tally.get(r["state"], 0) + 1
            if r["encode_line"]:
                empties.append(r["encode_line"])
            if r["state"] in ("SEGV", "HANG"):
                print("  [%s] run%-3d %-5s 最后标记=%s"
                      % (mode, i, r["state"], r["last_mark"]))
        segv = tally.get("SEGV", 0) + tally.get("HANG", 0)
        total_segv += segv
        print("\n%-13s 崩溃=%d/%d   分布=%s"
              % (mode, segv, args.iters,
                 ", ".join("%s=%d" % (k, v) for k, v in sorted(tally.items()))))
        if empties:
            uniq = sorted(set(empties))
            print("              帧守恒样本: %s" % " | ".join(uniq[:4]))
            if any("empty=0 none=0" not in e for e in empties):
                print("              ⚠️ 存在非零空帧/None —— 同配置下还伴随帧丢失")

    print("\n%s: 段错误合计 %d 次" % ("FAIL(缺陷可复现)" if total_segv else "PASS(未复现)",
                                      total_segv))
    print("判读：qp=0 下 ctor_only 应 0 崩溃、编码模式应显著崩溃；"
          "qp=23 应全 0 —— 符合则该归因成立。")
    return 1 if total_segv else 0


if __name__ == "__main__":
    sys.exit(main())
