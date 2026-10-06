#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[FIX-ESF-FILL-ADVANCE] ESRGAN 排空超限兜底分支的指针推进量测试（纯 CPU）。

缺陷
----
`external/realesrgan_video/nvenc_sdk.py::_ensure_slot_free` 的兜底分支
（`[FIX-SLOT-BACKPRESSURE-B]`）把目标槽 per-slot FIFO 的**全部**条目以 prev 帧占位消费
（`while _dq: _dq.popleft()` + `del _slot_pending[slot_idx]`），
但指针原先硬编码 `self._output_slot_idx += 1`。

per-slot FIFO 存 N>1 条是**常态**（跨 chunk 累积，见 `[FIX-SLOT-DEQUE]`：
append 永不覆盖），于是 `+= 1` 只推进 1 ⇒ **指针落后 N-1** ⇒ 后续 drain 从错误物理槽
起轮转 ⇒ 相位漂移 / 帧错位。IFRNet 同分支是 `self._output_slot_idx += _n_fill`
（`_n_fill = len(_dq)`，消费前取），本次修复即对齐该语义。

为什么能用纯 CPU 测
------------------
该分支的前置是「guard 耗尽或 hevc 未就绪」，但**真正的触发条件可以构造**：
把 `_lock_bitstream_blocking` 打桩返回空 ⇒ 排空探测必然失败 ⇒ 直接进入兜底。
不需要真实编码器，也不需要 GPU。

⚠ 生产可达性：该分支在 8 份存档生产日志中命中数为 **0**（属**潜伏**缺陷，
非活跃故障）。修复是「消除潜在错误」而非「修当前可见故障」。
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


def _entry4(gfi):
    """LA>0（`encode_frames_batch`）4 元组：(gfi, bs_buf, force_idr, ep_status)。"""
    return (gfi, None, False, 0)


def _make(slot_pending, slot_count=4, output_slot_idx=0):
    sdk = _sdk()
    e = sdk.NVENCEncoder.__new__(sdk.NVENCEncoder)
    e._codec = "h264"
    e._la_depth = 0# 走 h264 分支（hevc 需 _hevc_ready_count）
    e._slot_count = slot_count
    e._frame_idx = 99
    e._output_slot_idx = output_slot_idx
    e._slot_pending = slot_pending
    e._prev_stream_h264 = b"\x00\x00\x00\x01\x65PREV"   # prev 帧占位内容
    e._strict_eos = False
    # 槽数须≥ 测试用的 slot_idx+1（_ensure_slot_free 会索引 _slots[slot_idx]）
    e._slots = [{"bs_buf": None} for _ in range(slot_count)]
    # 打桩：阻塞 Lock 一律失败（模拟「硬件无数据」⇒ 必然进兜底）
    e._lock_bitstream_blocking = lambda bs, timeout_ms=500: (b"", 0)
    # 打桩：per-frame drain 一律取不到东西（让 guard 走到耗尽）
    e._drain_outputs_blocking = lambda max_slots=None: []
    return e


# ── 判据 1：指针推进量 == 实际消费条目数 ─────────────────────────────────
@pytest.mark.parametrize("n_entries", [1, 2, 3, 5, 9])
def test_pointer_advances_by_entries_consumed(n_entries):
    """核心判据：N 条 pending 被消费 ⇒ 指针必须推进 **N**（原实现恒推进 1）。"""
    slot, start = 0, 40
    dq = deque(_entry4(start + i) for i in range(n_entries))
    e = _make({slot: dq}, output_slot_idx=start)
    results = [None] * 64
    prev = []
    e._ensure_slot_free(slot, start, 64, results, prev)

    assert slot not in e._slot_pending, "兜底后应删除该槽 pending"
    assert e._output_slot_idx == start + n_entries, \
        f"指针应推进 {n_entries}（实际消费条目数），实得 {e._output_slot_idx - start}"


def test_placeholder_written_for_every_consumed_frame():
    """被消费的每一条都要写prev 帧占位（帧数守恒靠占位，不靠丢帧）。"""
    slot, start, n = 2, 10, 3
    dq = deque(_entry4(start + i) for i in range(n))
    e = _make({slot: dq}, output_slot_idx=start)
    results = [None] * 64
    e._ensure_slot_free(slot, start, 64, results, [])

    filled = [i for i, v in enumerate(results) if v == e._prev_stream_h264]
    assert filled == [0, 1, 2], \
        f"本 chunk 内3 条 pending 应各写一处占位，实得 {filled}"


# ── 判据 2：负向 —— 不得越界写 / 不得少写 ───────────────────────────────
def test_no_advance_beyond_consumed():
    """指针不得超推进：消费 N 条就只推进 N（防「多推」造成反向漂移）。"""
    slot, start, n = 1, 0, 4
    dq = deque(_entry4(start + i) for i in range(n))
    e = _make({slot: dq}, output_slot_idx=start)
    e._ensure_slot_free(slot, start, 64, [None] * 64, [])
    assert e._output_slot_idx == start + n


def test_empty_pending_no_advance():
    """pending 为空时 while 不进入，指针**不得**推进（原实现也正确，保持）。"""
    e = _make({}, output_slot_idx=7)
    e._ensure_slot_free(0, 7, 64, [None] * 64, [])
    assert e._output_slot_idx == 7, "无 pending 时不得推进指针"


def test_out_of_range_gfi_not_written():
    """gfi 落在本 chunk 之外（`_actual_fi_f` 越界）时不写 results，但仍计数推进。"""
    slot, chunk_start, n = 0, 0, 3
    dq = deque(_entry4(100 + i) for i in range(n))   # 全在本 chunk 之外
    e = _make({slot: dq}, output_slot_idx=chunk_start)
    results = [None] * 64
    e._ensure_slot_free(slot, chunk_start, 64, results, [])
    assert all(v is None for v in results), "越界 gfi 不得写 results"
    assert e._output_slot_idx == chunk_start + n, "但记账仍须按消费数推进"


# ── 判据 3：源码层 —— 推进量必须来自 len()，不得硬编码 ──────────────────
def test_source_uses_n_fill_not_literal():
    """源码层锁住：兜底分支的推进量必须来自 `_n_fill`（= 消费前 len），非字面量 1。

    ⚠ 判定用 AST（取 AugAssign 的 value 并看是否 Name `_n_fill`），
    不用文本匹配 —— 注释里也会出现 `+= 1`。
    """
    src = SDK_PATH.read_text(encoding="utf-8")
    i = src.index("def _ensure_slot_free")
    j = src.index("\n    def ", i + 10)
    fn = next(n for n in ast.walk(ast.parse(src[i:j]))
              if isinstance(n, ast.FunctionDef) and n.name == "_ensure_slot_free")
    targets = []
    for node in ast.walk(fn):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) \
                and isinstance(node.target, ast.Attribute) \
                and node.target.attr == "_output_slot_idx":
            targets.append(node)
    assert targets, "未找到 _output_slot_idx += ... 语句"
    # 至少有一处的增量是 `_n_fill`（Name 节点）
    assert any(isinstance(t.value, ast.Name) and t.value.id == "_n_fill" for t in targets), \
        "兜底分支的推进量应使用 _n_fill（实际消费条目数），不得硬编码"
    # 兜底分支里不应再有裸常量增量（1）
    assert not any(isinstance(t.value, ast.Constant) and t.value.value == 1 for t in targets), \
        "兜底分支不得硬编码 += 1（per-slot FIFO 可有 N>1 条）"


def test_n_fill_captured_before_popping():
    """`_n_fill` 必须在 popleft 循环**之前**取（否则 len 恒为 0）。

    ⚠ 必须在**剥掉注释**后再比位置：`_dq.popleft()` 这个词在紧邻上方的
    [FIX-ESF-FILL-ADVANCE] 注释里就出现过（解释为什么要先取长度），
    直接文本 index 会命中注释而误判顺序（2026-10-05 实测踩到）。
    """
    src = SDK_PATH.read_text(encoding="utf-8")
    i = src.index("def _ensure_slot_free")
    j = src.index("\n    def ", i + 10)
    body = "\n".join(ln.split("#", 1)[0] for ln in src[i:j].splitlines())
    assert "_n_fill = len(_dq)" in body, "缺少 _n_fill = len(_dq)"
    assert body.index("_n_fill = len(_dq)") < body.index("_dq.popleft()"), \
        "_n_fill 必须在 popleft 之前取长度"
