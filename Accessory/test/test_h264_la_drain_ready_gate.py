#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h264+LA>0 排空就绪门在 ESRGAN 侧的**否证测试** + 判据 helper 的边界测试。

结论（2026-10-05 T4 实测，本文件是该结论的回归锁）
----------------------------------------------------
把 IFRNet 的 `[FIX-H264-LA-DRAIN-READY]` 门移植到 ESRGAN 的 `_drain_outputs_blocking`
**会引入丢帧回归**，已移除该调用点。本文件锁住三件事：

1. **调用点必须不存在**（否证断言，含「不得挪进 `_ensure_slot_free`」）；
2. 移除处必须留有**证伪记录**（含实测数字），避免后人「好心」再搬一次；
3. 判据 helper `_h264_target_slot_ready()` 的边界语义本身是正确的
   （与 IFRNet 逐字一致），保留供将来统一记账语义后复用。

实测证据（同素材 100s / 5 段 / h264_nvenc / vbr_hq / bs=8）
----------------------------------------------------------
| 指标 | 移植前（A/B 4 轮） | 移植后 |
|---|---|---|
| code=8 | 3（仅 warmup） | 0 |
| 门命中 | — | **196**（frame_idx 1→1024 全程） |
| 段 1 帧守恒 | 4999 == 4999 | **decoded=1024 expected=1025** |
| 结果 | rc=0，8/0/0 | rc=1，片段 1 失败终止 |

根因：判据依赖 `_frame_idx - _oldest_gfi` 单调增大，但 ESRGAN 的
`_apply_drained_entries` 在辅助块 / size 钳制等多条路径也推进 `_output_slot_idx`，
使 `_slot_pending` 的 4 元组 gfi 与 `_output_slot_idx` 不同步 ⇒ 目标槽常驻
「看起来未就绪」的旧 entry ⇒ 每轮 drain 都 break ⇒ 排空停滞 → 反压堆积 → 丢帧。
⇒ **这是记账语义差异，不是判据写错**；要真正消除 code=8 须先统一两侧记账（架构级）。

⚠ 与「槽数」无关：已 A/B 排除（9 vs 11 槽 code=8 均为 3 次，见
`Accessory/probe/ab_h264_la_slots.py` 与 memory `h264-la-ready-gate-asymmetry.md`）。
"""
from __future__ import annotations

import ast
import importlib
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SDK_PATH = ROOT / "external" / "realesrgan_video" / "nvenc_sdk.py"
if str(ROOT / "external") not in sys.path:
    sys.path.insert(0, str(ROOT / "external"))


def _sdk():
    return importlib.import_module("realesrgan_video.nvenc_sdk")


def _src() -> str:
    return SDK_PATH.read_text(encoding="utf-8")


def _body(src: str, header: str) -> str:
    i = src.index(header)
    j = src.index("\n    def ", i + 10)
    return src[i:j]


def _strip_comments(text: str) -> str:
    return "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())


def _entry4(gfi):
    """LA>0（`encode_frames_batch`）4 元组：(gfi, bs_buf, force_idr, ep_status)。"""
    return (gfi, None, False, 0)


def _entry5(ce, fi):
    """LA=0（`encode_frames_batch_ce_pipeline`）5 元组：(_ce, fi, ep, idr, bs)。"""
    return (ce, fi, 0, False, None)


def _enc(codec="h264", la_depth=8, slot_count=9, frame_idx=0, pending=None):
    sdk = _sdk()
    e = sdk.NVENCEncoder.__new__(sdk.NVENCEncoder)
    e._codec = codec
    e._la_depth = la_depth
    e._slot_count = slot_count
    e._frame_idx = frame_idx
    e._output_slot_idx = 0
    e._slot_pending = {} if pending is None else pending
    return e


# ══════════════════════════════════════════════════════════════════════
# 1. 否证断言：调用点必须不存在
# ══════════════════════════════════════════════════════════════════════
def test_gate_call_site_absent_in_drain():
    """**核心否证**：`_drain_outputs_blocking` 内不得调用就绪门 helper。

    移植它会丢帧（实测 decoded=1024 expected=1025 / rc=1），详见模块 docstring。
    ⚠ 判定必须**剥注释**：移除处的证伪记录里就写着 helper 名，文本匹配会假失败。
    """
    drain_code = _strip_comments(_body(_src(), "def _drain_outputs_blocking"))
    assert "_h264_target_slot_ready(" not in drain_code, \
        "ESRGAN 的 drain 循环不得调用 h264 就绪门（实测丢帧回归，见本文件 docstring）"


def test_gate_not_in_ensure_slot_free():
    """也不得挪进 `_ensure_slot_free`：那里未就绪会落 `[FIX-SLOT-BACKPRESSURE-B]`
    用 prev 帧顶替真实码流（静默丢帧），且该函数在锁内、`_frame_idx += 1` 之前
    调用 ⇒ 等待永不满足（纯自旋）。"""
    esf_code = _strip_comments(_body(_src(), "def _ensure_slot_free"))
    assert "_h264_target_slot_ready(" not in esf_code, \
        "就绪门不得加在 _ensure_slot_free（prev 帧占位丢帧 + 锁内自旋）"


def test_removal_site_keeps_falsification_record():
    """移除处必须留证伪记录（含实测数字），否则后人会「好心」再搬一次。"""
    drain = _body(_src(), "def _drain_outputs_blocking")
    assert "已实测证伪并移除" in drain, "移除处缺少证伪记录"
    for token in ("196", "1024", "expected=1025"):
        assert token in drain, f"证伪记录应含实测数字 {token}"


# ══════════════════════════════════════════════════════════════════════
# 2. helper 边界语义（本身正确，留作将来统一记账后复用）
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("frame_idx,expect_ready", [
    (0, False), (1, False), (5, False), (9, False),      # fi-gfi <= 9 ⇒ 未就绪
    (10, True), (11, True), (50, True),                # ≥10 ⇒ 已就绪
])
def test_helper_readiness_boundary(frame_idx, expect_ready):
    e = _enc(frame_idx=frame_idx, pending={0: deque([_entry4(0)])})
    assert e._h264_target_slot_ready(0) is expect_ready


@pytest.mark.parametrize("la_depth", [1, 8, 16, 32])
def test_helper_matches_ifrnet_formula(la_depth):
    """helper 判据必须逐字等于 IFRNet 的 `frame_idx - oldest_gfi <= la_depth + 1`。

    ⚠ `la_depth=0` 不适用：helper 对 LA<=0 直接返回 True（由 completionEvent 判就绪）。
    ⇒ **helper 本身没写错**；移植失败源于两侧 drain 记账语义不同（见 docstring）。
    """
    for gfi in (0, 3, 17):
        for fi in range(gfi, gfi + la_depth + 4):
            e = _enc(la_depth=la_depth, frame_idx=fi,
                     pending={0: deque([_entry4(gfi)])})
            assert e._h264_target_slot_ready(0) is (not (fi - gfi <= la_depth + 1)), \
                f"la={la_depth} fi={fi} gfi={gfi} 与 IFRNet 口径不一致"


@pytest.mark.parametrize("codec,la_depth", [
    ("hevc", 8), ("av1", 8), ("hevc", 0), ("av1", 0), ("h264", 0),
])
def test_helper_is_identity_for_other_paths(codec, la_depth):
    """hevc/av1 由 `_hevc_ready_count` 单独门控；LA=0 靠 completionEvent 判就绪。
    三条路径互不重叠 ⇒ helper 对这些组合必须恒等 True。"""
    e = _enc(codec=codec, la_depth=la_depth, frame_idx=0,
             pending={0: deque([_entry4(0)])})
    assert e._h264_target_slot_ready(0) is True


def test_helper_empty_and_malformed_pending():
    e = _enc(pending={})
    assert e._h264_target_slot_ready(0) is True
    e2 = _enc(pending={0: deque([(0,)])})          # 畸形 1 元素 entry
    assert e2._h264_target_slot_ready(0) is True, "非预期形状不应干预"


def test_helper_uses_oldest_entry_not_newest():
    """必须看**队首**（最旧）帧 —— per-slot FIFO 里队首才是该 slot 的占用者。"""
    e = _enc(la_depth=8, frame_idx=3,
             pending={0: deque([_entry4(0), _entry4(100)])})
    assert e._h264_target_slot_ready(0) is False


def test_entry_shape_first_field_semantics_differ():
    """记录易错点：LA>0 的 `[0]` 是全局帧号，LA=0 的 `[0]` 是 CUDA event。

    这就是 helper 必须靠 `la_depth > 0` 限定、**不能**靠「长度 < 4」排除 LA=0 路径
    的原因 —— 5 元组同样满足 `len >= 4`，靠长度排除会静默把 ce_handle 当帧号。
    """
    e4, e5 = _entry4(7), _entry5(ce=12345, fi=7)
    assert len(e4) == 4 and e4[0] == 7
    assert len(e5) == 5 and e5[0] == 12345 and e5[1] == 7
    e = _enc(la_depth=0, frame_idx=100, pending={0: deque([e5])})
    assert e._h264_target_slot_ready(0) is True, "LA=0 必须恒等，不得取 [0]"


def test_helper_is_side_effect_free():
    """helper 必须是纯判定：不消耗 pending、不推进指针、不写计数。"""
    e = _enc(frame_idx=2, pending={0: deque([_entry4(0)])})
    before, idx = list(e._slot_pending[0]), e._output_slot_idx
    e._h264_target_slot_ready(0)
    assert list(e._slot_pending[0]) == before
    assert e._output_slot_idx == idx
    assert not hasattr(e, "_diag_h264_not_ready")


# ══════════════════════════════════════════════════════════════════════
# 3. 记账语义差异（本回归的真正根因，锁住事实）
# ══════════════════════════════════════════════════════════════════════
def test_output_slot_idx_advance_paths_differ_from_ifrnet():
    """**本回归的机制根因**：两侧 `_drain_outputs_blocking` 里`_output_slot_idx += 1`
    的**处数不同** —— IFRNet 仅 1 处（成功 Unlock 之后），ESRGAN 有 2 处
    （多一条 `[P3-FIX-LockBitstream-SizeCap]` 强制消费分支）。
    指针推进语义不同 ⇒ IFRNet 的就绪判据（依赖指针与 pending 同步）在 ESRGAN 不成立。
    用 AST 计数锁住该事实。"""
    def _count_advance(path):
        src = Path(path).read_text(encoding="utf-8")
        i = src.index("def _drain_outputs_blocking")
        j = src.index("\n    def ", i + 10)
        tree = ast.parse(src[i:j])
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) \
                    and isinstance(node.target, ast.Attribute) \
                    and node.target.attr == "_output_slot_idx":
                n += 1
        return n

    esr = _count_advance(SDK_PATH)
    ifr = _count_advance(ROOT / "external" / "ifrnet_video" / "nvenc_sdk.py")
    assert ifr == 1, f"IFRNet 侧应只有 1 处推进（实测 {ifr}）—— 判据成立的前提"
    assert esr >= 2, \
        f"ESRGAN 侧应有≥2 处推进（sizecap 强制消费分支），实测 {esr} —— " \
        "这正是就绪判据不能直接搬来的机制根因"