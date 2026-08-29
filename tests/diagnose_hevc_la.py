#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEVC + LA 最小化 GPU 诊断/验证脚本（tests/ 临时诊断脚本，非 pytest）。

背景
----
生产 VBR_HQ + LA=8 + HEVC（NVENC SDK 13.0, T4）在 LA 预热期后出现两条故障：
  · 阻塞排空死锁：辅助块（VPS/SPS/PPS）之后首个 VCL 未就绪，blocking
    LockBitstream 永不返回（output2.err: Cached SPS+PPS 后零帧写出）
  · 非阻塞排空 segfault：doNotWait=1 在 T4 驱动上直接段错误（output3.err）
H.264 同参数实机正常（辅助块+轮转+FIFO 记账已验证），HEVC 的驱动输出
时序/槽位路由与 H.264 不同，需要先隔离测量再定生产修复。

本脚本参照 tests/test_nvenc_la_frame_conservation.py 的 MinimalTestEncoder，
用最小编码器逐 drain 记录 (slot_idx, est_fi, outputTimeStamp, size, NAL 类型)，
对比 5 种排空策略的帧数守恒结果：
  · baseline          : 生产同构（每帧提交后轮转阻塞排空 + FIFO 记账 + 辅助块不占帧槽）
  · delayed           : 预热期（LA+2 帧内）不排空，之后每帧提交后正常排空
  · aux_stay          : HEVC 辅助块不推进轮转指针（帧紧跟同一槽）
  · delayed_aux_stay  : delayed + aux_stay 组合
  · free_pool         : 扩容槽位(LA+2) + 空闲槽池 + ts@40 重关联映射
  · ce_pipeline       : LA=0 CE 流水线（生产 ce_pipeline 同构）——2026-08-18
    output4 实测该路径在段末 writer/encode 线程 30s 未退出（疑似死锁）。
    本变体复现并定位：Phase1/3 的 cuEventSynchronize 与 blocking LockBitstream
    均可能无界等待（HEVC 驱动在"帧未就绪"时阻塞而不返回 NEED_MORE_INPUT）。
  · ce_pipeline_fix   : ce_pipeline 同构 + 修复版 flush（FIX-HEVC-EOS-FLUSH）——
    生产 flush() 补丁的参考实现：HEVC/AV1 发送 EOS 后只锁仍有 pending 帧的槽，
    空槽一律不锁（驱动对空槽 blocking Lock 永久阻塞，test4 实测；EOS 后 pending
    槽帧已就绪，test5 实测 vcl=700 守恒）。H.264 走原有 _flush_eos_prod 全槽轮转。
    本变体复现 codec-esrgan-hevc_nvenc_test1.err：LA=0（FIX-HIGHRES-RC 降级）
    ce_pipeline 全部 harvest 后 flush() 锁空槽 → 修复前 TIMEOUT / 修复后 PASS。
  · multi_segment     : 同一 encoder 实例连续 N 段（默认 2），每段 =
    ce_pipeline + 修复版 flush——对齐生产 [FIX-SKIP-REOPEN] 跨段复用场景：
    EOS 排空 → 下一段复用同一驱动会话（_frame_idx 跨段单调不重置），逐段校验
    帧数守恒，最后合并 ES 做 ffmpeg decode。复现 test1 第二段 1% 冻结的前提
    （段 1 flush 卡死 → 段 2 复用被锁会话）。
  · nonblocking       : 修复方向验证——与 baseline 同构但 LockBitstream 用
    doNotWait=1（非阻塞），未就绪帧立即跳过、由后续提交推进后再取回。
    diagnose_hevc_la.log 实测：HEVC 首 9 帧（gfi0-8）窗口填满时批量涌出、
    gfi9（首个槽位复用帧）在 sub17 后未就绪 → blocking Lock 死锁；
    非阻塞是否安全（output3 的 segfault 仅限 LA 预热 buffer）由本变体界定。
  · counted           : 修复方向验证 2——提交计数驱动的有界排空：
    gfi k 在提交 sub(k+LA) 后就绪（7 变体日志实测），每帧提交后最多取回
    submitted - LA 个帧，永不锁未就绪帧 → 无死锁；帧数守恒由 EOS 兜底。
  · eos_probe         : EOS 专用探针——counted 中途 + EOS 后 doNotWait=1
    全槽多轮扫描。2026-08-18 test2 实测：counted/free_pool 中途全通
    （vcl=692/690），均挂在 EOS 后 blocking Lock（_flush_eos_diag）——
    EOS 是 HEVC 最后障碍。本变体揭示 EOS 后驱动把滞留帧输出到哪些槽、
    doNotWait=1 在 EOS 后是否安全。

2026-08-18 全变体实测结论（temp/hevc_la_diag_8/*.log）：
  · 所有 blocking 变体（baseline/delayed/aux_stay/delayed_aux_stay）都挂在
    锁"未就绪的下一帧"；delayed（10 槽）最远跑到 gfi9。
  · nonblocking：doNotWait=1 在数据未就绪时返回 SUCCESS + 垃圾 size
    （7.2MB 误读），且下一次锁同样挂起 → 非阻塞方向彻底否决。
  · free_pool（10 槽）：中途全程无死锁（700 帧提交完 vcl=690），只在
    EOS 前的末尾排空挂起（脚本缺陷，已改为 EOS 优先）。
  · 修复 = counted 有界排空（或 free_pool）+ EOS 兜底；
    段尾 EOS 排空 = 只锁 pending 槽（FIX-HEVC-EOS），LA=0 flush() 同构
    （ce_pipeline_fix / multi_segment 变体，2026-08-18 生产补丁参考实现）。

诊断手段
--------
  · 逐 drain 记录 (slot_idx, est_fi, ts@40, size, NAL 类型, VCL 有无)
  · 每次 blocking LockBitstream 计时：单次 >0.5s 打印 WARN
  · watchdog 线程：编码期间每 2s dump 主线程 Python 栈（ctypes 调用释放 GIL，
    卡死时仍能看到卡在哪一行），前 3 次

用法（生产 Linux GPU，需 torch + ffmpeg）
-----------------------------------------
  # 单策略（可能死锁，配 timeout 防挂死；日志每行 flush）
  timeout 90 python tests/diagnose_hevc_la.py --variant delayed \
      --frames 700 --la-depth 8 --decode-check

  # 全策略回归对比：每个变体按 VARIANT_CANONICAL 的规范配置独立子进程运行
  # （= 各变体单独验证通过时的配置，见 run_all）；VARIANT_EXPECTED 声明
  # 期望结果（旧轮转模型复现器 = 预期 TIMEOUT/FAIL），与期望不符时 exit=1。
  # --stall-timeout 按 vcl 心跳检测“真死锁”，慢速但仍在推进的运行不误杀。
  python tests/diagnose_hevc_la.py --run-all --frames 700 --timeout 75

通过判据：VCL 输出帧数 == 提交帧数（含 EOS flush），且可 ffmpeg 解码。
"""

import argparse
import ctypes
import os
import pathlib
import re
import subprocess
import sys
import time
from collections import deque
from ctypes import (POINTER, byref, c_size_t, c_uint32, c_uint64, c_uint8,
                    c_void_p, cast)
from typing import List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    import test_nvenc_la_frame_conservation as ref
    MinimalTestEncoder = ref.MinimalTestEncoder
    generate_synthetic_nv12_frames = ref.generate_synthetic_nv12_frames
    _REF_LOADED = True
except Exception as _e:  # 通常因本机无 torch（测试文件类型注解依赖 torch）
    ref = None
    MinimalTestEncoder = None
    generate_synthetic_nv12_frames = None
    _REF_LOADED = False
    _REF_ERROR = _e


# ============================================================================
# NAL 解析（H.264/HEVC，与生产 _nal_first_vcl_type 同源逻辑）
# ============================================================================

def nal_first_vcl(data: bytes, codec: str) -> Optional[int]:
    """返回第一个 VCL NAL 类型；无 VCL（辅助块）返回 None。
    H.264: VCL=1/5（1 字节头）；HEVC: VCL=0..31（2 字节头，(b0>>1)&0x3F）。"""
    pos, n = 0, len(data)
    while pos < n - 3:
        if data[pos:pos + 4] == b'\x00\x00\x00\x01':
            pos += 4
        elif data[pos:pos + 3] == b'\x00\x00\x01':
            pos += 3
        else:
            pos += 1
            continue
        if pos >= n:
            break
        t = (data[pos] & 0x1f) if codec == "h264" else ((data[pos] >> 1) & 0x3f)
        if codec == "h264":
            if t in (1, 5):
                return t
        else:
            if t in range(0, 32):
                return t
        while pos < n - 3:
            if data[pos:pos + 4] == b'\x00\x00\x00\x01' or \
                    data[pos:pos + 3] == b'\x00\x00\x01':
                break
            pos += 1
    return None


def nal_types(data: bytes, codec: str) -> List[int]:
    """返回码流块内全部 NAL 类型（诊断用）。"""
    types = []
    pos, n = 0, len(data)
    while pos < n - 3:
        if data[pos:pos + 4] == b'\x00\x00\x00\x01':
            pos += 4
        elif data[pos:pos + 3] == b'\x00\x00\x01':
            pos += 3
        else:
            pos += 1
            continue
        if pos >= n:
            break
        b0 = data[pos]
        types.append((b0 & 0x1f) if codec == "h264" else ((b0 >> 1) & 0x3f))
        while pos < n - 3:
            if data[pos:pos + 4] == b'\x00\x00\x00\x01' or \
                    data[pos:pos + 3] == b'\x00\x00\x01':
                break
            pos += 1
    return types


# ============================================================================
# 诊断编码器（继承已验证的 MinimalTestEncoder，注入 FIFO/ts/变体逻辑）
# ============================================================================

class DiagEncoder(MinimalTestEncoder if MinimalTestEncoder is not None else object):
    """MinimalTestEncoder + 逐 drain 诊断 + 多种排空策略。"""

    def __init__(self, width: int, height: int, fps: float,
                 preset: str = "veryslow", qp: int = 23,
                 rate_mode: str = "vbr_hq", la_depth: int = 8,
                 codec: str = "hevc", pipeline_depth: int = 10,
                 variant: str = "baseline", diag_limit: int = 60):
        super().__init__(width, height, fps, preset=preset, qp=qp,
                         rate_mode=rate_mode, la_depth=la_depth,
                         codec=codec, pipeline_depth=pipeline_depth)
        self._variant = variant
        self._diag_limit = diag_limit
        self._diag_n = 0
        self._slot_pending: dict = {}     # slot -> deque[(gfi, bs_buf, fidr, status)]
        self._slot_owner: dict = {}       # slot -> gfi（free_pool 用）
        self._free_pool: Optional[deque] = None
        self._results: dict = {}          # gfi -> bytes（VCL 帧）
        self._aux_count = 0
        self._vcl_count = 0
        self._ts_hits = 0
        self._ts_misses = 0
        self._mismatch = 0
        self._submitted = 0
        self._last_aux_slot = None
        self._eos_leftover = 0
        self._slow_lock_warns = 0

    # ── 提交一帧（返回 gfi, ep_status）────────────────────────────────────
    def _submit_frame(self, frame, force_idr: bool = False,
                      slot_idx: Optional[int] = None,
                      ce=None) -> Tuple[int, int, object]:
        if slot_idx is None:
            slot_idx = self._frame_idx % self._pipeline_depth
        slot = self._slots[slot_idx]
        pitch = self._copy_frame_to_input_buffer(frame, slot['input_buf'])

        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, POINTER(c_uint32))[0] = ref.NV_ENC_PIC_PARAMS_VER
        cast(byref(pic_buf, 4), POINTER(c_uint32))[0] = self._width
        cast(byref(pic_buf, 8), POINTER(c_uint32))[0] = self._height
        cast(byref(pic_buf, 12), POINTER(c_uint32))[0] = (
            pitch if pitch > 0 else self._width)
        cast(byref(pic_buf, 24), POINTER(c_uint64))[0] = self._frame_idx
        cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = slot['input_buf']
        cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = slot['bs_buf']
        if ce is not None and ce.value is not None:
            cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = ce
        cast(byref(pic_buf, 64), POINTER(c_uint32))[0] = ref.NV_ENC_BUFFER_FORMAT_NV12
        cast(byref(pic_buf, 68), POINTER(c_uint32))[0] = ref.NV_ENC_PIC_STRUCT_FRAME
        if force_idr:
            cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x2

        _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
            self._func_ptrs[ref._FUNC_IDX["EncodePicture"]])
        status = _EncodePicture(self._encoder, pic_buf)
        gfi = self._frame_idx
        self._frame_idx += 1
        self.total_encoded += 1
        self._submitted += 1
        self._slot_pending.setdefault(slot_idx, deque()).append(
            (gfi, slot['bs_buf'], force_idr, status))
        self._slot_owner[slot_idx] = gfi
        if self._diag_n <= max(3, self._diag_limit // 4):
            print(f"[submit] gfi={gfi} slot={slot_idx} ep_status={status} "
                  f"(NMI={ref.NV_ENC_ERR_NEED_MORE_INPUT})", flush=True)
        return gfi, status, ce

    # ── blocking LockBitstream + ts@40 读取（含单次耗时诊断）─────────────
    def _lock_bs_diag(self, bs_handle, do_not_wait: int = 0) -> \
            Tuple[bytes, int, Optional[int]]:
        _t0 = time.time()
        lock_raw = (c_uint8 * 1544)()
        ctypes.memset(lock_raw, 0, 1544)
        cast(lock_raw, POINTER(c_uint32))[0] = ref.NV_ENC_LOCK_BITSTREAM_VER
        cast(byref(lock_raw, 4), POINTER(c_uint32))[0] = do_not_wait
        cast(byref(lock_raw, 8), POINTER(c_void_p))[0] = bs_handle
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))(
            self._func_ptrs[ref._FUNC_IDX["LockBitstream"]])
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
            self._func_ptrs[ref._FUNC_IDX["UnlockBitstream"]])
        bs_status = _LockBS(self._encoder, lock_raw)
        if bs_status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
            return b"", bs_status, None
        if bs_status != ref.NV_ENC_SUCCESS:
            return b"", bs_status, None
        size = cast(byref(lock_raw, 36), POINTER(c_uint32))[0]
        ts = int(cast(byref(lock_raw, 40), POINTER(c_uint64))[0])
        ptr = cast(byref(lock_raw, 56), POINTER(c_void_p))[0]
        ptr_val = ptr if isinstance(ptr, int) else (ptr.value or 0)
        data = b""
        if size > 0 and ptr_val:
            buf_type = c_uint8 * size
            data = bytes(buf_type.from_address(ptr_val))
        _UnlockBS(self._encoder, bs_handle)
        _el = time.time() - _t0
        if _el > 0.5:
            self._slow_lock_warns += 1
            if self._slow_lock_warns <= 10:
                print(f"[lock] ⚠️ LockBitstream(bs={bs_handle}) 单次耗时 "
                      f"{_el:.2f}s doNotWait={do_not_wait}（HEVC 驱动可能阻塞等待）",
                      flush=True)
        return data, bs_status, ts

    # ── CE 辅助（ce_pipeline 变体用）─────────────────────────────────────
    def _ce_create(self):
        _ce = c_void_p(None)
        self._libcuda.cuEventCreate.restype = c_uint32
        self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
        self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)
        return _ce

    def _ce_destroy(self, ce):
        if ce is not None and ce.value is not None:
            self._libcuda.cuEventDestroy.restype = c_uint32
            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
            self._libcuda.cuEventDestroy(ce)

    def _ce_wait(self, ce, where: str):
        """cuEventSynchronize 计时（无界等待；若 HEVC CE 不触发，watchdog
        会 dump 出此处栈）。"""
        _t0 = time.time()
        self._libcuda.cuEventSynchronize.restype = c_uint32
        self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
        self._libcuda.cuEventSynchronize(ce)
        _el = time.time() - _t0
        if _el > 1.0:
            print(f"[ce_pipeline] ⚠️ cuEventSynchronize({where}) 耗时 {_el:.2f}s",
                  flush=True)

    def _harvest_pending(self, pending, slot_idx: int):
        """ce_pipeline Phase1/3 harvest：CE 等待 + blocking 取回（生产同构）。"""
        _ce, _fi, _ep, _idr, _bs = pending
        if _ce is not None and _ce.value is not None:
            self._ce_wait(_ce, f"fi={_fi} slot={slot_idx}")
            self._ce_destroy(_ce)
        data, status, ts = self._lock_bs_diag(_bs)
        if data:
            gfi = _fi
            self._results[gfi] = data
            self._vcl_count += 1
            self._output_slot_idx += 1
            self._diag_drain(slot_idx, _fi, ts, data, f"→ gfi={gfi} (ce-pipe)")
            return True
        if _ep == ref.NV_ENC_ERR_NEED_MORE_INPUT:
            # 生产同构：5s blocking retry（驱动阻塞不返回时，watchdog 显示卡点）
            _t0 = time.time()
            _h264_blk, _ = self._lock_bs_blocking_timeout(_bs, timeout_ms=5000)
            _el = time.time() - _t0
            if _h264_blk:
                self._results[_fi] = _h264_blk
                self._vcl_count += 1
                self._output_slot_idx += 1
                self._diag_drain(slot_idx, _fi, ts, _h264_blk,
                                 f"→ gfi={_fi} (ce-pipe blk {_el:.1f}s)")
                return True
            self._diag_drain(slot_idx, _fi, ts, b"",
                             f"⚠️ NEED_MORE_INPUT 5s retry 无数据 ({_el:.1f}s)")
            self._results[_fi] = b""
        return False

    def _lock_bs_blocking_timeout(self, bs_handle, timeout_ms: int = 5000) -> \
            Tuple[bytes, int]:
        """生产 _lock_bitstream_blocking 同构：doNotWait=0 轮询 + 截止时间。
        注意：若驱动对未就绪 buffer 直接阻塞不返回，本函数同样无法超时，
        此时由 watchdog / 子进程 timeout 兜底。"""
        _deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < _deadline:
            data, status, _ts = self._lock_bs_diag(bs_handle)
            if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                time.sleep(0.001)
                continue
            return data, status
        return b"", ref.NV_ENC_ERR_NEED_MORE_INPUT

    # ── 单条 drain 记录诊断 ──────────────────────────────────────────────
    def _diag_drain(self, slot_idx: int, est_fi: int, ts, data: bytes,
                    note: str = ""):
        self._diag_n += 1
        if self._diag_n > self._diag_limit:
            return
        tps = nal_types(data, self._codec)
        print(f"[drain#{self._diag_n}] slot={slot_idx} est_fi={est_fi} "
              f"ts={ts} size={len(data)} nals={tps} "
              f"vcl={nal_first_vcl(data, self._codec)} {note}", flush=True)

    @staticmethod
    def _pending_gfis(slot_pending: dict) -> list:
        """取当前 pending 全部条目的帧号（4-tuple: gfi 在 0；5-tuple: 在 1）。
        诊断用：EOS 前后快照，精确定位丢失帧。"""
        out = []
        for dq in slot_pending.values():
            for ent in dq:
                out.append(ent[1] if len(ent) >= 5 else ent[0])
        return sorted(out)

    # ── VCL/辅助块分类消费（含变体语义）─────────────────────────────────
    def _consume_drained(self, slot_idx: int, est_fi: int, ts, data: bytes) -> \
            Optional[int]:
        """处理一个已锁定的 drain 块。返回取回帧的 gfi（辅助块返回 None）。"""
        hv = nal_first_vcl(data, self._codec)
        if hv is None:
            # 无 VCL 辅助块：仅缓存（最小脚本不做 SPS 提取），不占 fi
            self._aux_count += 1
            note = "aux"
            if self._variant in ("aux_stay", "delayed_aux_stay", "baseline",
                                 "delayed", "counted", "eos_probe"):
                # [FIX-AUX-NO-DRIFT] 辅助块不占帧序列：若推进轮转指针会造成
                # 相位漂移——pending 达到 slot_count（槽位全满），下一次提交
                # 复用满槽时驱动丢弃未读输出，该帧在 EOS 前已永久丢失
                # （test6 baseline h264 实测 aux=1 时 vcl=699 leftover=1，
                # 且 EOS leftover 扫描也取不回）。帧紧跟同一槽位产出 →
                # 回退指针即对齐（aux_stay 语义推广到全部轮转变体）。
                self._output_slot_idx -= 1
                note += "+stay"
            self._diag_drain(slot_idx, est_fi, ts, data, note)
            return None
        # VCL：优先 FIFO 队首映射（生产同构）；free_pool 优先 ts 重关联
        gfi = None
        if self._variant == "free_pool":
            gfi = self._pop_by_ts(ts)
        if gfi is None:
            dq = self._slot_pending.get(slot_idx)
            if dq:
                gfi, _, _, _ = dq[0]
                dq.popleft()
                if not dq:
                    del self._slot_pending[slot_idx]
        if gfi is None:
            self._mismatch += 1
            self._diag_drain(slot_idx, est_fi, ts, data, "⚠️ VCL 无标签")
            return None
        self._results[gfi] = data
        self._vcl_count += 1
        self._diag_drain(slot_idx, est_fi, ts, data,
                         f"→ gfi={gfi}{' ts-hit' if ts == gfi else ''}")
        return gfi

    def _pop_by_ts(self, ts: Optional[int]):
        """free_pool：用 outputTimeStamp@40 精确查找 pending 帧（tt6/tt7 双射验证）。"""
        if ts is None:
            return None
        for s, dq in list(self._slot_pending.items()):
            for i, ent in enumerate(dq):
                if ent[0] == ts:
                    dq.remove(ent)
                    if not dq:
                        del self._slot_pending[s]
                    self._ts_hits += 1
                    return ent[0]
        self._ts_misses += 1
        return None

    # ── 轮转排空（baseline/delayed/aux_stay 共用）───────────────────────
    def _drain_rotation(self, max_slots: Optional[int] = None):
        if max_slots is None:
            max_slots = self._pipeline_depth
        for _ in range(max_slots):
            slot_idx = self._output_slot_idx % self._pipeline_depth
            data, status, ts = self._lock_bs_diag(self._slots[slot_idx]['bs_buf'])
            if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if status != ref.NV_ENC_SUCCESS or not data:
                break
            est_fi = self._output_slot_idx
            self._output_slot_idx += 1
            self._consume_drained(slot_idx, est_fi, ts, data)

    def _drain_rotation_nonblocking(self, max_slots: Optional[int] = None):
        """[FIX-HEVC-NONBLOCK] 修复方向验证：与 _drain_rotation 同构，但
        LockBitstream 用 doNotWait=1（非阻塞）。未就绪帧立即 NMI break，
        由后续提交推进后再取回；帧数守恒由 EOS 兜底。
        风险：output3 曾见 doNotWait=1 对 LA 预热 buffer segfault；本变体
        在子进程内运行，若 segfault 会留下 core + 日志，正好界定安全边界。"""
        for _ in range(max_slots or self._pipeline_depth):
            slot_idx = self._output_slot_idx % self._pipeline_depth
            data, status, ts = self._lock_bs_diag(
                self._slots[slot_idx]['bs_buf'], do_not_wait=1)
            if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if status != ref.NV_ENC_SUCCESS or not data:
                break
            est_fi = self._output_slot_idx
            self._output_slot_idx += 1
            self._consume_drained(slot_idx, est_fi, ts, data)

    def _drain_rotation_counted(self):
        """[FIX-HEVC-COUNTED] 提交计数驱动的有界排空（2026-08-18 实测验证）：
        gfi k 的输出在提交 sub(k+LA) 后就绪。每帧提交后最多取回
        `submitted - la_depth` 个帧（已取回数 = _output_slot_idx），
        永不锁未就绪帧 → HEVC 驱动不阻塞。帧数守恒由 EOS 排空兜底。
        H.264 路径不动（有独立辅助块 + 不同节奏）。"""
        submitted = self._frame_idx
        drained = self._output_slot_idx
        max_ready = submitted - self._la_depth
        budget = min(self._pipeline_depth, max_ready - drained)
        for _ in range(max(0, budget)):
            slot_idx = self._output_slot_idx % self._pipeline_depth
            data, status, ts = self._lock_bs_diag(
                self._slots[slot_idx]['bs_buf'])
            if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if status != ref.NV_ENC_SUCCESS or not data:
                break
            est_fi = self._output_slot_idx
            self._output_slot_idx += 1
            self._consume_drained(slot_idx, est_fi, ts, data)

    # ── ensure_slot_free（delayed 变体复用槽位前）────────────────────────
    def _ensure_slot_free(self, slot_idx: int):
        guard = 0
        while self._slot_pending.get(slot_idx):
            before = self._output_slot_idx
            self._drain_rotation(max_slots=1)
            if self._output_slot_idx != before or not self._slot_pending.get(slot_idx):
                guard = 0
                continue
            guard += 1
            if guard > self._pipeline_depth * 4:
                # 直探目标槽（bounded：最多 ~2s）
                import time as _t
                deadline = _t.monotonic() + 2.0
                got = False
                while _t.monotonic() < deadline:
                    data, status, ts = self._lock_bs_diag(self._slots[slot_idx]['bs_buf'])
                    if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                        _t.sleep(0.001)
                        continue
                    if data:
                        self._consume_drained(slot_idx, slot_idx, ts, data)
                        got = True
                    break
                if got:
                    guard = 0
                    continue
                print(f"[ensure_slot_free] ⚠️ slot={slot_idx} 直探超时，"
                      f"pending={len(self._slot_pending.get(slot_idx, ()))}, "
                      f"以空帧占位（帧数守恒将失败）", flush=True)
                self._slot_pending[slot_idx].clear()
                del self._slot_pending[slot_idx]
                break

    # ── free_pool：排空最旧 pending 槽 ───────────────────────────────────
    def _drain_oldest_pending(self):
        if not self._slot_pending:
            return
        oldest_slot = min(self._slot_pending,
                          key=lambda s: self._slot_pending[s][0][0])
        while self._slot_pending.get(oldest_slot):
            data, status, ts = self._lock_bs_diag(self._slots[oldest_slot]['bs_buf'])
            if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if status != ref.NV_ENC_SUCCESS or not data:
                break
            self._consume_drained(oldest_slot, oldest_slot, ts, data)
        if self._slot_pending.get(oldest_slot):
            print(f"[free_pool] ⚠️ slot={oldest_slot} 排空后仍有 "
                  f"{len(self._slot_pending[oldest_slot])} 帧 pending（驱动未就绪），"
                  f"仍回收入池（帧将丢失）", flush=True)
        self._free_pool.append(oldest_slot)

    # ── EOS flush（ts 映射版，帧数守恒必需）──────────────────────────────
    def _flush_eos_diag(self):
        """发送 EOS 并按轮转顺序逐槽 while-True 排空（同测试 flush_eos），
        但每个 drain 块读取 ts@40 并用 ts 映射回 pending 帧（tt6/tt7 验证
        @40 双射），保证 LA 滞留帧计入 self._results。"""
        # [FIX-HEVC-EOS-FLUSH] HEVC/AV1：驱动对空槽/未就绪槽的 blocking Lock
        # 永久阻塞（test4 实测）→ EOS 排空必须 pending-only（生产 FIX-HEVC-EOS
        # 同构；eos_probe 实测 vcl=700 守恒）。counted/free_pool 等 LA>0 变体
        # 复用此路径，避免 test6 复现的 `_flush_eos_diag` EOS 挂死。
        if self._codec in ("hevc", "av1"):
            _pend_before = self._pending_gfis(self._slot_pending)
            print(f"[eos] pending_before={_pend_before} "
                  f"output_idx={self._output_slot_idx} vcl={self._vcl_count}",
                  flush=True)
            self._flush_eos_pending()
            self._eos_leftover = sum(len(dq) for dq in self._slot_pending.values())
            if self._eos_leftover:
                print(f"[eos] ⚠️ 仍有 {self._eos_leftover} 帧未取回（帧数守恒将失败）: "
                      f"{self._pending_gfis(self._slot_pending)}", flush=True)
            return
        _pend_before = self._pending_gfis(self._slot_pending)
        print(f"[eos] pending_before={_pend_before} "
              f"output_idx={self._output_slot_idx} vcl={self._vcl_count}",
              flush=True)
        _eos_recovered = []
        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, POINTER(c_uint32))[0] = ref.NV_ENC_PIC_PARAMS_VER
        cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8  # EOS flag
        cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
        cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
        _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
            self._func_ptrs[ref._FUNC_IDX["EncodePicture"]])
        _EncodePicture(self._encoder, pic_buf)

        start_slot = self._output_slot_idx % self._pipeline_depth
        drain_order = [(start_slot + i) % self._pipeline_depth
                       for i in range(self._pipeline_depth)]
        for slot_idx in drain_order:
            bs = self._slots[slot_idx]['bs_buf']
            while True:
                data, status, ts = self._lock_bs_diag(bs)
                if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                    break
                if status != ref.NV_ENC_SUCCESS or not data:
                    break
                hv = nal_first_vcl(data, self._codec)
                if hv is None:
                    self._aux_count += 1
                    self._diag_drain(slot_idx, self._output_slot_idx, ts, data,
                                     "aux(eos)")
                    continue
                gfi = self._pop_by_ts(ts)
                if gfi is None:
                    dq = self._slot_pending.get(slot_idx)
                    if dq:
                        gfi, _, _, _ = dq[0]
                        dq.popleft()
                        if not dq:
                            del self._slot_pending[slot_idx]
                if gfi is None:
                    self._mismatch += 1
                    self._diag_drain(slot_idx, self._output_slot_idx, ts, data,
                                     "⚠️ eos VCL 无标签")
                    continue
                self._results[gfi] = data
                self._vcl_count += 1
                self._output_slot_idx += 1
                _eos_recovered.append(gfi)
                self._diag_drain(slot_idx, self._output_slot_idx - 1, ts, data,
                                 f"→ gfi={gfi} (eos){' ts-hit' if ts == gfi else ''}")
        # [FIX-EOS-LEFTOVER-SCAN] h264 轮转排空后仍可能有 pending：辅助块推进
        # 轮转指针造成相位漂移、或尾帧就绪晚于该槽被排空的时刻（test6 baseline
        # 实测 leftover=1）。对剩余 pending 槽做有界多轮 doNotWait=1 扫描
        # （eos_probe 同款策略；h264 空槽返回 SUCCESS+size=0 不挂）。
        no_progress = 0
        for _rnd in range(50):
            got = 0
            for _s in sorted(self._slot_pending.keys()):
                data, status, ts = self._lock_bs_diag(
                    self._slots[_s]['bs_buf'], do_not_wait=1)
                if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                    continue
                if status != ref.NV_ENC_SUCCESS or not data:
                    continue
                hv = nal_first_vcl(data, self._codec)
                if hv is None:
                    self._aux_count += 1
                    got += 1
                    continue
                gfi = self._pop_by_ts(ts)
                if gfi is None:
                    dq = self._slot_pending.get(_s)
                    if dq:
                        gfi, _, _, _ = dq[0]
                        dq.popleft()
                        if not dq:
                            del self._slot_pending[_s]
                if gfi is None:
                    self._mismatch += 1
                    continue
                self._results[gfi] = data
                self._vcl_count += 1
                self._output_slot_idx += 1
                _eos_recovered.append(gfi)
                got += 1
            if got == 0:
                no_progress += 1
                if no_progress >= 3:
                    break
            else:
                no_progress = 0
        print(f"[eos] recovered={sorted(_eos_recovered)}", flush=True)
        self._eos_leftover = sum(len(dq) for dq in self._slot_pending.values())
        if self._eos_leftover:
            print(f"[eos] ⚠️ 仍有 {self._eos_leftover} 帧未取回（帧数守恒将失败）: "
                  f"{self._pending_gfis(self._slot_pending)}", flush=True)

    # ── ce_pipeline（LA=0）变体 ───────────────────────────────────────────
    def _flush_eos_prod(self):
        """生产 flush() EOS 排空同构：逐槽 while-True，首锁 blocking
        (doNotWait=0)、后续 non-blocking (doNotWait=1)；取回块按 ts 映射
        回 pending 帧（映射失败仅计数，不崩脚本）。"""
        _t0 = time.time()
        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, POINTER(c_uint32))[0] = ref.NV_ENC_PIC_PARAMS_VER
        cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8
        cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
        cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
        _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
            self._func_ptrs[ref._FUNC_IDX["EncodePicture"]])
        _EncodePicture(self._encoder, pic_buf)

        start_slot = self._output_slot_idx % self._pipeline_depth
        drain_order = [(start_slot + i) % self._pipeline_depth
                       for i in range(self._pipeline_depth)]
        for slot_idx in drain_order:
            bs = self._slots[slot_idx]['bs_buf']
            _first = True
            while True:
                data, status, ts = self._lock_bs_diag(bs, do_not_wait=(0 if _first else 1))
                _first = False
                if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                    break
                if status != ref.NV_ENC_SUCCESS or not data:
                    break
                hv = nal_first_vcl(data, self._codec)
                if hv is None:
                    self._aux_count += 1
                    continue
                gfi = self._pop_by_ts(ts)
                if gfi is None:
                    dq = self._slot_pending.get(slot_idx)
                    if dq:
                        gfi, _, _, _ = dq[0]
                        dq.popleft()
                        if not dq:
                            del self._slot_pending[slot_idx]
                if gfi is None:
                    self._mismatch += 1
                    continue
                self._results[gfi] = data
                self._vcl_count += 1
                self._output_slot_idx += 1
        print(f"[flush_prod] EOS 排空耗时 {time.time() - _t0:.1f}s，"
              f"vcl_total={self._vcl_count}", flush=True)

    def _flush_eos_pending(self):
        """[FIX-HEVC-EOS-FLUSH] 修复版 EOS 排空参考实现（生产 flush() 补丁同构）。

        与 _flush_eos_prod 的唯一差异：HEVC/AV1 发送 EOS 后只锁仍有 pending 帧的槽
        ——驱动对空槽的 blocking LockBitstream 永久阻塞（test4 实测），而 EOS 后
        pending 槽帧已就绪、按 pending 条数取回（test5 实测 vcl=700 守恒）。
        H.264 保持原全槽轮转（空槽返回 SUCCESS+size=0，无死锁）。
        """
        _t0 = time.time()
        _pend_before = self._pending_gfis(self._slot_pending)
        _eos_recovered = []
        if _pend_before:
            print(f"[flush_pending] pending_before={_pend_before} "
                  f"output_idx={self._output_slot_idx} vcl={self._vcl_count}",
                  flush=True)
        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, POINTER(c_uint32))[0] = ref.NV_ENC_PIC_PARAMS_VER
        cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8
        cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
        cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
        _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
            self._func_ptrs[ref._FUNC_IDX["EncodePicture"]])
        _EncodePicture(self._encoder, pic_buf)

        _hevc = self._codec in ("hevc", "av1")
        if _hevc:
            # 只锁仍有 pending 的槽；空槽 Lock 会永久阻塞（test4 实测）
            drain_order = sorted(self._slot_pending.keys())
        else:
            start_slot = self._output_slot_idx % self._pipeline_depth
            drain_order = [(start_slot + i) % self._pipeline_depth
                           for i in range(self._pipeline_depth)]
        for slot_idx in drain_order:
            bs = self._slots[slot_idx]['bs_buf']
            _first = True
            while True:
                if _hevc and not self._slot_pending.get(slot_idx):
                    break  # pending 已取完，不再锁空槽
                data, status, ts = self._lock_bs_diag(bs, do_not_wait=(0 if _first else 1))
                _first = False
                if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                    break
                if status != ref.NV_ENC_SUCCESS or not data:
                    break
                hv = nal_first_vcl(data, self._codec)
                if hv is None:
                    self._aux_count += 1
                    continue
                gfi = self._pop_by_ts(ts)
                if gfi is None:
                    dq = self._slot_pending.get(slot_idx)
                    if dq:
                        ent = dq[0]
                        # ce_pipeline 5-tuple: (ce, fi, ep, idr, bs)；其余 4-tuple
                        gfi = ent[1] if len(ent) >= 5 else ent[0]
                        dq.popleft()
                        if not dq:
                            del self._slot_pending[slot_idx]
                if gfi is None:
                    self._mismatch += 1
                    continue
                self._results[gfi] = data
                self._vcl_count += 1
                self._output_slot_idx += 1
                _eos_recovered.append(gfi)
                self._diag_drain(slot_idx, self._output_slot_idx - 1, ts, data,
                                 f"→ gfi={gfi} (eos-pending){' ts-hit' if ts == gfi else ''}")
        if _eos_recovered:
            print(f"[flush_pending] recovered={sorted(_eos_recovered)}", flush=True)
        print(f"[flush_pending] EOS 排空耗时 {time.time() - _t0:.1f}s，"
              f"vcl_total={self._vcl_count} "
              f"pending_slots={sorted(self._slot_pending.keys())}", flush=True)

    def encode_frames_ce_pipeline(self, frames: List[object], flush_fn=None):
        """LA=0 CE 流水线（生产 ce_pipeline 同构，复现 output4 段末死锁）。
        Phase1 harvest → Phase2 提交（per-frame CE）→ Phase3 排空 → flush EOS。"""
        if flush_fn is None:
            flush_fn = self._flush_eos_prod
        n = len(frames)
        pd = self._pipeline_depth
        _slot_pending = [None] * pd
        self._output_slot_idx = 0  # 生产 ce_pipeline 每批次重置输出指针
        for i, frame in enumerate(frames):
            slot_idx = self._frame_idx % pd
            if _slot_pending[slot_idx] is not None:
                self._harvest_pending(_slot_pending[slot_idx], slot_idx)
                _slot_pending[slot_idx] = None
            _ce = self._ce_create()
            gfi, status, _ = self._submit_frame(frame, force_idr=(i == 0),
                                                slot_idx=slot_idx, ce=_ce)
            _slot_pending[slot_idx] = (
                # [FIX-MULTI-SEGMENT] 存局部 fi（enumerate 索引）而非全局 gfi：
                # 每段 _results 按 0..n-1 局部索引读取，multi_segment 第 2 段
                # 的 _frame_idx 已前移（全局单调），存全局 gfi 会写入段外键。
                _ce, i, status, (i == 0), self._slots[slot_idx]['bs_buf'])
            if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                print(f"[progress-ce] {i + 1}/{n} submitted "
                      f"vcl={self._vcl_count} "
                      f"pending={sum(1 for p in _slot_pending if p is not None)}",
                      flush=True)
        # Phase 3: 排空剩余 pending
        for slot_idx in range(pd):
            if _slot_pending[slot_idx] is not None:
                self._harvest_pending(_slot_pending[slot_idx], slot_idx)
                _slot_pending[slot_idx] = None
        # [FIX-HEVC-EOS-FLUSH] 清空 _submit_frame 在实例 _slot_pending 中的
        # 记账残留：ce_pipeline 流程用上方局部 _slot_pending 消费驱动输出，
        # 实例 dict 里的 4-tuple 从未被 pop（仅其他变体用它做权威记账）。
        # 若不清理，修复版 flush 会误判"仍有 pending"而锁已排空槽 → HEVC
        # 空槽 blocking Lock 死锁无法被本变体正确验证（对齐生产 Phase 3
        # popleft 后 _slot_pending 为空的状态）。
        self._slot_pending.clear()
        # flush EOS（生产 flush() 同构；ce_pipeline_fix 变体用修复版）
        flush_fn()
        return n

    def encode_frames_ce_pipeline_fix(self, frames: List[object]):
        """[FIX-HEVC-EOS-FLUSH] ce_pipeline 同构 + 修复版 flush：
        HEVC/AV1 只锁 pending 槽，空槽不锁（生产 flush() 补丁参考实现）。"""
        return self.encode_frames_ce_pipeline(frames, flush_fn=self._flush_eos_pending)

    def encode_frames_multi_segment(self, frames: List[object],
                                    segments: int = 2):
        """[FIX-HEVC-EOS-FLUSH] 跨段复用最小复现：同一 encoder 实例连续
        segments 段，每段 = ce_pipeline + 修复版 flush。

        对齐生产 [FIX-SKIP-REOPEN] 场景（段尾 EOS 排空 → 下一段复用同一驱动
        会话）：_frame_idx 跨段单调递增（严禁重置，对齐生产 FIX-GLOBAL-FI）；
        _slot_pending 每段结束必须为空（对齐 _stream_begin(force=True) 语义）。
        每段首帧强制 IDR（encode_frames_ce_pipeline 内 fi==0 判定，与生产一致）。
        """
        n = len(frames)
        all_outputs: List[bytes] = []
        ok = True
        for seg in range(segments):
            _sub_start = self._submitted
            _vcl_start = self._vcl_count
            _aux_start = self._aux_count
            self._results = {}
            self._slot_pending = {}
            self.encode_frames_ce_pipeline(frames, flush_fn=self._flush_eos_pending)
            _sub_seg = self._submitted - _sub_start
            _vcl_seg = self._vcl_count - _vcl_start
            out_seg = [self._results.get(i, b"") for i in range(n)]
            non_empty = sum(1 for x in out_seg if x)
            conserved = (non_empty == _sub_seg)
            left = sum(len(dq) for dq in self._slot_pending.values())
            print(f"[multi_segment] seg{seg + 1}/{segments}: submitted={_sub_seg} "
                  f"vcl={_vcl_seg} aux={self._aux_count - _aux_start} "
                  f"conserved={conserved} pending_left={left}", flush=True)
            ok = ok and conserved and left == 0
            all_outputs.extend(out_seg)
        # 汇总为单段语义，供 summary() / decode-check 复用
        self._results = {i: d for i, d in enumerate(all_outputs)}
        self._submitted = n * segments
        print(f"[multi_segment] 完成：总提交 {self._submitted} "
              f"vcl={self._vcl_count} conserved={ok}", flush=True)
        return self._submitted

    def encode_frames_eos_probe(self, frames: List[object]):
        """EOS 探针：counted 有界排空中途（已验证无死锁）+ EOS 后
        doNotWait=1 全槽多轮扫描，测 EOS 后非阻塞锁安全性并定位滞留帧槽位。"""
        n = len(frames)
        for i, frame in enumerate(frames):
            self._submit_frame(frame, force_idr=(i == 0))
            self._drain_rotation_counted()
            if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                print(f"[progress-eos] {i + 1}/{n} submitted, "
                      f"vcl={self._vcl_count} output_idx={self._output_slot_idx}",
                      flush=True)
        print(f"[eos_probe] 提交完成 pending={sum(len(d) for d in self._slot_pending.values())} "
              f"pending_slots={sorted(self._slot_pending.keys())} -> 发送 EOS", flush=True)
        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, POINTER(c_uint32))[0] = ref.NV_ENC_PIC_PARAMS_VER
        cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8
        cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
        cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
        # EOS 帧也带 CE + 记录返回码：验证 EOS 是否被驱动真正接受/完成
        _eos_ce = self._ce_create()
        cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _eos_ce
        _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
            self._func_ptrs[ref._FUNC_IDX["EncodePicture"]])
        _eos_status = _EncodePicture(self._encoder, pic_buf)
        print(f"[eos_probe] EOS EncodePicture status={_eos_status} "
              f"(0=SUCCESS, 17=NEED_MORE_INPUT)", flush=True)
        if _eos_ce.value is not None:
            _t0 = time.time()
            self._libcuda.cuEventSynchronize.restype = c_uint32
            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
            _rc = self._libcuda.cuEventSynchronize(_eos_ce)
            print(f"[eos_probe] EOS CE synchronize rc={_rc} "
                  f"耗时 {time.time() - _t0:.2f}s", flush=True)
            self._ce_destroy(_eos_ce)

        # EOS 后：只锁仍有 pending 的槽（test4 实测：HEVC 空槽 Lock 永久阻塞，
        # h264 空槽返回 SUCCESS size=0；EOS 后 pending 槽帧已就绪可直接取回）
        no_progress = 0
        for rnd in range(300):
            got = 0
            for slot_idx in sorted(self._slot_pending.keys()):
                data, status, ts = self._lock_bs_diag(
                    self._slots[slot_idx]['bs_buf'], do_not_wait=1)
                if rnd < 2:
                    print(f"[eos_probe] rnd={rnd} slot={slot_idx} "
                          f"status={status} size={len(data)} "
                          f"pending={len(self._slot_pending.get(slot_idx, ()))}",
                          flush=True)
                if status == ref.NV_ENC_ERR_NEED_MORE_INPUT:
                    continue
                if status != ref.NV_ENC_SUCCESS or not data:
                    continue
                hv = nal_first_vcl(data, self._codec)
                if hv is None:
                    self._aux_count += 1
                    got += 1
                    continue
                gfi = self._pop_by_ts(ts)
                if gfi is None:
                    self._mismatch += 1
                    continue
                self._results[gfi] = data
                self._vcl_count += 1
                got += 1
                self._diag_drain(slot_idx, self._output_slot_idx, ts, data,
                                 f"→ gfi={gfi} (eos-probe rnd={rnd})")
                self._output_slot_idx += 1
            if got == 0:
                no_progress += 1
                if no_progress >= 5:
                    break
                time.sleep(0.05)
            else:
                no_progress = 0
        self._eos_leftover = sum(len(d) for d in self._slot_pending.values())
        print(f"[eos_probe] 完成：vcl={self._vcl_count} "
              f"leftover={self._eos_leftover} mismatch={self._mismatch} "
              f"aux={self._aux_count}", flush=True)
        return n

    # ── 主循环（按变体）──────────────────────────────────────────────────
    def encode_frames(self, frames: List[object]):
        n = len(frames)
        v = self._variant
        if v == "ce_pipeline":
            return self.encode_frames_ce_pipeline(frames)
        if v == "ce_pipeline_fix":
            return self.encode_frames_ce_pipeline_fix(frames)
        if v == "eos_probe":
            return self.encode_frames_eos_probe(frames)
        if v == "free_pool":
            self._free_pool = deque(range(self._pipeline_depth))
            for i, frame in enumerate(frames):
                if not self._free_pool:
                    self._drain_oldest_pending()
                slot_idx = self._free_pool.popleft()
                self._submit_frame(frame, force_idr=(i == 0), slot_idx=slot_idx)
                if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                    print(f"[progress] {i + 1}/{n} submitted, vcl={self._vcl_count} "
                          f"aux={self._aux_count} pending_slots={len(self._slot_pending)}",
                          flush=True)
            # 末尾排空全部
            # [FIX-HEVC-EOS-FIRST] EOS 之前不排空未就绪帧（实测会挂起）：
            # 直接发送 EOS 后由 _flush_eos_diag 排空全部（EOS 后驱动完成所有帧）
            self._flush_eos_diag()
            print(f"[free_pool] EOS 排空完成 vcl={self._vcl_count} "
                  f"leftover={self._eos_leftover}", flush=True)
            return n
        elif v in ("delayed", "delayed_aux_stay"):
            for i, frame in enumerate(frames):
                self._ensure_slot_free(self._frame_idx % self._pipeline_depth)
                self._submit_frame(frame, force_idr=(i == 0))
                if self._frame_idx >= self._la_depth + 2:
                    self._drain_rotation()
                if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                    print(f"[progress] {i + 1}/{n} submitted, vcl={self._vcl_count} "
                          f"aux={self._aux_count} pending_slots={len(self._slot_pending)}",
                          flush=True)
            self._drain_rotation()
        elif v == "nonblocking":
            # 修复方向验证：每帧提交后非阻塞排空（不等待未就绪帧）
            for i, frame in enumerate(frames):
                self._submit_frame(frame, force_idr=(i == 0))
                self._drain_rotation_nonblocking()
                if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                    print(f"[progress-nb] {i + 1}/{n} submitted, "
                          f"vcl={self._vcl_count} aux={self._aux_count} "
                          f"pending_slots={len(self._slot_pending)}", flush=True)
            self._drain_rotation_nonblocking()
        elif v == "counted":
            # 提交计数驱动的有界排空（修复方向验证 2）
            for i, frame in enumerate(frames):
                self._submit_frame(frame, force_idr=(i == 0))
                self._drain_rotation_counted()
                if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                    print(f"[progress-ct] {i + 1}/{n} submitted, "
                          f"vcl={self._vcl_count} aux={self._aux_count} "
                          f"output_idx={self._output_slot_idx}", flush=True)
            self._drain_rotation_counted()
        else:  # baseline / aux_stay（生产同构：每帧提交后立即轮转排空）
            for i, frame in enumerate(frames):
                self._submit_frame(frame, force_idr=(i == 0))
                self._drain_rotation()
                if (i + 1) % 100 == 0 or i < 3 or i >= n - 3:
                    print(f"[progress] {i + 1}/{n} submitted, vcl={self._vcl_count} "
                          f"aux={self._aux_count} pending_slots={len(self._slot_pending)}",
                          flush=True)
            self._drain_rotation()

        # EOS flush（ts 映射版，LA 滞留帧计入结果 → 帧数守恒）
        self._flush_eos_diag()
        print(f"[eos] vcl_total={self._vcl_count} "
              f"leftover={self._eos_leftover}", flush=True)
        return n

    # ── 汇总与结果 ───────────────────────────────────────────────────────
    def summary(self) -> dict:
        out = [self._results.get(i, b"") for i in range(self._submitted)]
        non_empty = sum(1 for x in out if x)
        ordered = all(nal_first_vcl(x, self._codec) is not None for x in out if x)
        return {
            "variant": self._variant,
            "submitted": self._submitted,
            "vcl": self._vcl_count,
            "aux": self._aux_count,
            "non_empty": non_empty,
            "conserved": non_empty == self._submitted,
            "ordered": ordered,
            "ts_hits": self._ts_hits,
            "ts_misses": self._ts_misses,
            "mismatch": self._mismatch,
            "outputs": out,
        }


# ============================================================================
# 主流程
# ============================================================================

VARIANTS = ["baseline", "delayed", "aux_stay", "delayed_aux_stay",
            "free_pool", "ce_pipeline", "ce_pipeline_fix", "multi_segment",
            "nonblocking", "counted", "eos_probe"]


def _start_watchdog(interval: float = 2.0, max_dumps: int = 3, beat=None):
    """后台线程每 interval 秒 dump 主线程 Python 栈（ctypes 调用释放 GIL，
    即使主线程卡在 blocking LockBitstream / cuEventSynchronize 也能看到卡点）。
    卡死时栈显示调用行，结合外层子进程 timeout 定位。

    beat 传 DiagEncoder 时（--run-all 子进程自动开启），每次唤醒额外打印
    `[beat] ... vcl=N` 心跳行：父进程据此区分“仍在推进（慢）”与
    “vcl 停滞（真死锁）”，避免把慢速运行误报为 TIMEOUT；并且首次检测到
    vcl 停滞时补 dump 一次主线程栈（test10 nonblocking 实测：前 3 次 dump
    落在 21MB 垃圾块解析的 nal_types 上，真正的挂点在之后的 _lock_bs_diag），
    让 hang 证据指向最终卡点；vcl 恢复推进后重新武装。"""
    import threading
    import traceback
    main_tid = threading.main_thread().ident
    dumped = [0]
    _last_vcl = [None]
    _stall_dumped = [False]
    _t0 = time.time()

    def _run():
        while True:
            time.sleep(interval)
            if beat is not None:
                _v = beat._vcl_count
                print(f"[beat] t={time.time() - _t0:6.1f}s "
                      f"submitted={beat._submitted} vcl={_v} "
                      f"aux={beat._aux_count} output_idx={beat._output_slot_idx}",
                      flush=True)
                if _last_vcl[0] is None:
                    _last_vcl[0] = _v
                elif _v != _last_vcl[0]:
                    _last_vcl[0] = _v
                    _stall_dumped[0] = False
                elif not _stall_dumped[0]:
                    _stall_dumped[0] = True
                    cf = sys._current_frames()
                    f = cf.get(main_tid)
                    print(f"\n[watchdog] ⚠️ vcl 停滞在 {_v}（候选死锁）"
                          f"主线程栈:", flush=True)
                    if f is not None:
                        traceback.print_stack(f, file=sys.stdout)
            if dumped[0] >= max_dumps:
                continue
            cf = sys._current_frames()
            f = cf.get(main_tid)
            print(f"\n[watchdog] 主线程栈 dump #{dumped[0] + 1}:", flush=True)
            if f is not None:
                traceback.print_stack(f, file=sys.stdout)
            dumped[0] += 1

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def self_test() -> int:
    """无 GPU 自检：验证 NAL 分类 / FIFO / ts 映射 / aux_stay 指针逻辑。
    通过 object.__new__ 构造 DiagEncoder，不触碰 NVENC 初始化。"""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [self-test] [{'OK' if cond else 'FAIL'}] {name} {detail}",
              flush=True)

    # 构造 HEVC NAL 字节（2 字节头: type=(b0>>1)&0x3F）
    def sc(t):
        return b"\x00\x00\x00\x01" + bytes([(t << 1) | 0]) + b"\x01\x02\x03"

    aux_vps = sc(32) + sc(33) + sc(34)   # VPS/SPS/PPS → 无 VCL
    vcl_idr = sc(20) + sc(39) + b"\xaa" * 8  # IDR(20) + SEI(39) → VCL
    vcl_p = sc(1) + b"\xbb" * 8              # TRAIL_R(1) → VCL

    check("HEVC 辅助块识别为无 VCL", nal_first_vcl(aux_vps, "hevc") is None)
    check("HEVC IDR 识别为 VCL", nal_first_vcl(vcl_idr, "hevc") == 20)
    check("HEVC P 帧识别为 VCL", nal_first_vcl(vcl_p, "hevc") == 1)
    check("HEVC NAL 类型解析",
          nal_types(aux_vps, "hevc") == [32, 33, 34] and
          nal_types(vcl_idr, "hevc") == [20, 39])
    check("H.264 分支不受影响",
          nal_first_vcl(b"\x00\x00\x00\x01\x65" + b"\x00" * 4, "h264") == 5)

    # 记账逻辑（构造 DiagEncoder 而不调 __init__）
    enc = object.__new__(DiagEncoder)
    enc._codec = "hevc"
    enc._slot_pending = {0: deque([(0, None, True, 0)]), 1: deque([(1, None, False, 0)])}
    enc._results = {}
    enc._aux_count = 0
    enc._vcl_count = 0
    enc._ts_hits = 0
    enc._ts_misses = 0
    enc._mismatch = 0
    enc._diag_n = 0
    enc._diag_limit = 100
    enc._output_slot_idx = 0

    # free_pool: ts 映射
    enc._variant = "free_pool"
    gfi = enc._consume_drained(0, 0, 0, vcl_idr)   # ts=0 → 命中 gfi 0
    check("free_pool ts=0 命中 gfi 0", gfi == 0 and enc._results.get(0) == vcl_idr
          and enc._ts_hits == 1 and not enc._slot_pending.get(0))
    gfi = enc._consume_drained(1, 1, 999, vcl_p)   # ts=999 → 无命中 → FIFO 队首 gfi 1
    check("free_pool ts 未命中回退 FIFO", gfi == 1 and enc._ts_misses == 1
          and enc._results.get(1) == vcl_p)

    # aux 分支：不占 fi、不进 results
    enc._slot_pending = {0: deque([(0, None, True, 0)])}
    enc._results = {}
    enc._variant = "baseline"
    before = enc._aux_count
    r = enc._consume_drained(0, 0, 0, aux_vps)
    check("aux 仅缓存不占 fi", r is None and enc._aux_count == before + 1
          and not enc._results.get(0) and len(enc._slot_pending[0]) == 1)

    # aux_stay：指针回退
    enc._variant = "aux_stay"
    enc._output_slot_idx = 5
    enc._consume_drained(0, 5, 0, aux_vps)
    check("aux_stay 指针回退", enc._output_slot_idx == 4)

    print(f"\n[self-test] {'ALL PASS' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


def run_single(args) -> int:
    print(f"\n{'=' * 72}\n[DIAG] codec={args.codec} rate={args.rate_mode} "
          f"LA={args.la_depth} variant={args.variant} frames={args.frames} "
          f"slots={args.slots}\n{'=' * 72}", flush=True)

    slots = args.slots
    if slots <= 0:
        # baseline 用 LA+1 忠实复现生产配置（9 槽 → 预期死锁）；
        # 其余变体用 LA+2 隔离"槽位不足"变量
        if args.variant in ("ce_pipeline", "ce_pipeline_fix", "multi_segment"):
            slots = 4  # 生产 LA=0 ce_pipeline 用 pipe=4
        elif args.variant == "baseline":
            slots = args.la_depth + 1 if args.la_depth > 0 else 4
        elif args.variant in ("nonblocking", "counted", "eos_probe"):
            slots = args.la_depth + 1 if args.la_depth > 0 else 4
        else:
            slots = args.la_depth + 2 if args.la_depth > 0 else 4
    if args.variant in ("ce_pipeline_fix", "multi_segment") and args.la_depth > 0:
        print(f"[diag] ⚠️ {args.variant} 是 LA=0 专用变体（生产 FIX-HIGHRES-RC "
              f"降级路径）；当前 LA={args.la_depth} 不代表生产配置，结果仅参考",
              flush=True)
    print(f"[diag] 槽位数={slots}（variant={args.variant}；"
          f"LA={args.la_depth} → baseline=LA+1 / 其余=LA+2 / "
          f"ce_pipeline=4）", flush=True)

    enc = DiagEncoder(args.width, args.height, args.fps,
                      preset=args.preset, qp=args.qp,
                      rate_mode=args.rate_mode, la_depth=args.la_depth,
                      codec=args.codec, pipeline_depth=slots,
                      variant=args.variant, diag_limit=args.diag_limit)
    print(f"[diag] encoder ready: {enc._codec.upper()} "
          f"{args.width}x{args.height}@{args.fps} "
          f"{enc._rate_mode} LA={enc._la_depth} slots={enc._pipeline_depth}",
          flush=True)

    frames = generate_synthetic_nv12_frames(args.frames, args.width, args.height)
    t0 = time.time()
    _watchdog = _start_watchdog(beat=enc if args.beat else None)
    if args.variant == "multi_segment":
        enc.encode_frames_multi_segment(frames, args.segments)
    else:
        enc.encode_frames(frames)
    elapsed = time.time() - t0

    s = enc.summary()
    print(f"\n[RESULT] variant={s['variant']} submitted={s['submitted']} "
          f"vcl={s['vcl']} aux={s['aux']} non_empty={s['non_empty']} "
          f"conserved={s['conserved']} ordered={s['ordered']} "
          f"ts_hits={s['ts_hits']} ts_misses={s['ts_misses']} "
          f"mismatch={s['mismatch']} elapsed={elapsed:.1f}s", flush=True)

    if args.out:
        es_path = pathlib.Path(args.out)
        es_path.parent.mkdir(parents=True, exist_ok=True)
        es_path.write_bytes(b"".join(s["outputs"]))
        print(f"[out] ES 已写入 {es_path}（{es_path.stat().st_size} bytes）", flush=True)

    if args.decode_check and s["conserved"]:
        if args.out:
            es_path = pathlib.Path(args.out)
        else:
            # 默认写到 CWD temp/ 下（避免 /tmp 权限/路径解析问题）
            es_path = pathlib.Path("temp") / f"hevc_la_out_{args.variant}.es"
            es_path.parent.mkdir(parents=True, exist_ok=True)
            es_path.write_bytes(b"".join(s["outputs"]))
        print(f"[decode] ES: {es_path} ({es_path.stat().st_size} bytes)",
              flush=True)
        fmt = {"h264": "h264", "hevc": "hevc", "av1": "obu"}.get(args.codec, "hevc")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", fmt,
             "-framerate", str(args.fps), "-i", str(es_path),
             "-c:v", "copy", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        dec_ok = (r.returncode == 0)
        print(f"[decode] ffmpeg decode {'OK' if dec_ok else 'FAIL'}: "
              f"{r.stderr.strip()[:300]}", flush=True)
        if not dec_ok:
            return 3

    return 0 if s["conserved"] else 2


# ============================================================================
# run-all 回归矩阵：规范配置 + 期望结果
# ============================================================================
# test9 误报根因（2026-08-18）：
#   旧 run-all 把所有变体都按全局 codec=hevc + LA=8 跑。而各变体“单独验证通过”
#   时的配置并不一致：baseline 验证的是 h264（test8 全矩阵）；ce_pipeline_fix /
#   multi_segment 是 LA=0 专用变体（FIX-HIGHRES-RC 降级路径）；delayed / aux_stay /
#   delayed_aux_stay / ce_pipeline 在 hevc+LA=8 下是旧轮转模型的“预期死锁复现器”
#   （blocking Lock 未就绪帧 / EOS 锁空槽，test9 各变体日志精确卡在 drain#9/#10
#   与 EOS flush）；nonblocking 是 doNotWait=1 segfault 复现器。全局一刀切 +
#   无期望结果表 + 固定退出 0，把预期复现器死锁渲染成“TIMEOUT(deadlock?)”误报。
#   修复：每个变体用 VARIANT_CANONICAL 的规范配置运行；VARIANT_EXPECTED 声明
#   期望结果；结果与期望不符才标记 UNEXPECTED 并以 exit=1 报告回归。
VARIANT_CANONICAL = {
    # 生产 H.264 路径回归（baseline 的 HEVC 版是 bug 复现器，验证配置为 h264）
    "baseline":         {"codec": "h264", "la": 8},
    # 旧轮转模型 bug 复现器（hevc+LA=8 预期死锁，守护旧 bug 不复发）
    "delayed":          {"codec": "hevc", "la": 8},
    "aux_stay":         {"codec": "hevc", "la": 8},
    "delayed_aux_stay": {"codec": "hevc", "la": 8},
    "ce_pipeline":      {"codec": "hevc", "la": 8},
    "nonblocking":      {"codec": "hevc", "la": 8},
    # LA=0 专用变体（FIX-HIGHRES-RC 降级后的生产 LA=0 路径）
    "ce_pipeline_fix":  {"codec": "hevc", "la": 0},
    "multi_segment":    {"codec": "hevc", "la": 0},
    # 已修复的 HEVC+LA 变体（生产 FIX-HEVC-COUNTED / FIX-HEVC-EOS 对应）
    "free_pool":        {"codec": "hevc", "la": 8},
    "counted":          {"codec": "hevc", "la": 8},
    "eos_probe":        {"codec": "hevc", "la": 8},
}

# 期望结果：PASS / TIMEOUT / FAIL（按状态前缀匹配；集合 = 任一均可）。
# delayed/aux_stay/delayed_aux_stay/ce_pipeline = hevc+LA 下旧轮转模型预期死锁；
# nonblocking = doNotWait=1 在 T4 上 segfault(-11, test9) 或垃圾 size 误读后
# 挂起（test10：21MB 垃圾块解析后锁 gfi9 停滞，line 255）——两种均为已知失败模式。
VARIANT_EXPECTED = {
    "baseline":         "PASS",
    "delayed":          "TIMEOUT",
    "aux_stay":         "TIMEOUT",
    "delayed_aux_stay": "TIMEOUT",
    "free_pool":        "PASS",
    "ce_pipeline":      "TIMEOUT",
    "ce_pipeline_fix":  "PASS",
    "multi_segment":    "PASS",
    "nonblocking":      {"FAIL", "TIMEOUT"},
    "counted":          "PASS",
    "eos_probe":        "PASS",
}

# 预期非 PASS 的复现器（--skip-reproducers 时跳过）
REPRODUCERS = {v for v, exp in VARIANT_EXPECTED.items() if exp != "PASS"}


def _run_variant_proc(cmd, wall_timeout: int, stall_timeout: int) -> \
        Tuple[Optional[int], str, str]:
    """运行单个变体子进程并实时分析（替换 subprocess.run 的纯墙钟超时）：

    - stdout+stderr 合并实时读取，同时解析 `[beat] ... vcl=N` 心跳；
    - vcl 停止推进连续 stall_timeout 秒（且已运行 >10s 越过启动期）→ kill，
      归类 TIMEOUT(stall)：真死锁提前终止（复现器从 60s 降到 ~20s），
      慢速但仍在推进的运行不会被误杀（test9 误报根因之一）；
    - 否则 wall_timeout 硬超时兜底 → TIMEOUT(wall)。

    返回 (returncode, kind, 全部输出文本)；kind ∈ {ok, stall, wall}。"""
    import threading
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    out_lines = []
    state = {"max_vcl": -1, "last_vcl_t": t0}
    lock = threading.Lock()

    def _read():
        try:
            for line in proc.stdout:
                with lock:
                    out_lines.append(line)
                    m = re.search(r"vcl=(\d+)", line)
                    if m:
                        v = int(m.group(1))
                        if v > state["max_vcl"]:
                            state["max_vcl"] = v
                            state["last_vcl_t"] = time.monotonic()
        except Exception:
            pass

    rt = threading.Thread(target=_read, daemon=True)
    rt.start()
    kind = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        el = time.monotonic() - t0
        if el >= wall_timeout:
            kind = "wall"
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            break
        # vcl 停滞判定：先跑过启动期（>10s）再启用，避免预热期误杀
        if stall_timeout > 0 and el > 10.0 and state["max_vcl"] >= 0:
            if time.monotonic() - state["last_vcl_t"] >= stall_timeout:
                kind = "stall"
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                break
        time.sleep(0.05)
    rt.join(timeout=5)
    return rc, kind, "".join(out_lines)


def _norm_status(status: str) -> str:
    if status.startswith("PASS"):
        return "PASS"
    if status.startswith("TIMEOUT"):
        return "TIMEOUT"
    if status.startswith("FAIL"):
        return "FAIL"
    return status


def run_all(args) -> int:
    out_dir = pathlib.Path(args.tmpdir) / f"hevc_la_diag_{args.la_depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in VARIANTS
                if not (args.skip_reproducers and v in REPRODUCERS)]
    print(f"\n[RUN-ALL] frames={args.frames} rate={args.rate_mode} "
          f"timeout={args.timeout}s/variant stall={args.stall_timeout}s"
          f" （各变体按 VARIANT_CANONICAL 规范配置；expected=符合期望，"
          f"UNEXPECTED=回归信号）\n", flush=True)
    rows = []
    for v in variants:
        cfg = VARIANT_CANONICAL[v]
        cfg_desc = f"{cfg['codec']}/la={cfg['la']}"
        log_path = out_dir / f"{v}.log"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               "--variant", v, "--codec", cfg["codec"],
               "--frames", str(args.frames), "--la-depth", str(cfg["la"]),
               "--rate-mode", args.rate_mode,
               "--slots", str(args.slots),
               "--diag-limit", str(args.diag_limit),
               "--out", str(out_dir / f"{v}.es"),
               "--beat"]
        if v == "multi_segment":
            cmd += ["--segments", str(args.segments)]
        t0 = time.time()
        rc, kind, out_text = _run_variant_proc(cmd, args.timeout,
                                               args.stall_timeout)
        log_path.write_text(out_text)
        el = time.time() - t0
        if kind:
            status = f"TIMEOUT({kind})"
            _hang = ""
            for _line in reversed(out_text.splitlines()):
                if '", line ' in _line and " in " in _line:
                    _hang = _line.strip()
                    break
            _last_prog = ""
            for _line in reversed(out_text.splitlines()):
                if _line.startswith("[progress") or _line.startswith("[drain#"):
                    _last_prog = _line.strip()
                    break
            tail = [f"hang: {_hang}", f"last: {_last_prog}"]
        else:
            status = "PASS" if rc == 0 else f"FAIL({rc})"
            tail = out_text.strip().splitlines()[-3:]
        exp = VARIANT_EXPECTED[v]
        exp_set = {exp} if isinstance(exp, str) else set(exp)
        ok = _norm_status(status) in exp_set
        tag = "expected" if ok else "UNEXPECTED"
        print(f"  {v:<18} {cfg_desc:<11} {status:<20} {el:6.1f}s  [{tag}]",
              flush=True)
        for t in tail:
            print(f"      {t}", flush=True)
        rows.append((v, cfg_desc, status, el, tag))

    print("\n[RUN-ALL SUMMARY]")
    for v, cfg_desc, status, el, tag in rows:
        print(f"  {v:<18} {cfg_desc:<11} {status:<20} {el:6.1f}s  [{tag}]",
              flush=True)
    print(f"  日志目录: {out_dir}", flush=True)
    bad = [r for r in rows if r[4] == "UNEXPECTED"]
    if bad:
        print(f"[RUN-ALL] {len(bad)} 个变体结果与期望不符: "
              f"{', '.join(r[0] for r in bad)}"
              f"（期望见 VARIANT_EXPECTED）", flush=True)
        return 1
    print("[RUN-ALL] 全部变体结果符合期望（PASS 回归 + 预期复现器）", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description="HEVC+LA 最小化 GPU 诊断/验证脚本")
    p.add_argument("--codec", default="hevc", choices=["h264", "hevc", "av1"])
    p.add_argument("--rate-mode", default="vbr_hq",
                   choices=["constqp", "vbr_hq", "qvbr"])
    p.add_argument("--la-depth", type=int, default=8)
    p.add_argument("--frames", type=int, default=700)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=float, default=48.0)
    p.add_argument("--preset", default="veryslow")
    p.add_argument("--qp", type=int, default=23)
    p.add_argument("--variant", default="delayed", choices=VARIANTS)
    p.add_argument("--segments", type=int, default=2,
                   help="multi_segment 变体连续段数（默认 2，跨段复用验证）")
    p.add_argument("--slots", type=int, default=0,
                   help="槽位数（0=自动 LA+2）")
    p.add_argument("--diag-limit", type=int, default=60,
                   help="详细 drain 日志条数上限")
    p.add_argument("--out", default="",
                   help="ES 输出路径（.hevc/.h264/.obu）")
    p.add_argument("--tmpdir", default=os.environ.get("TMPDIR",
                   os.path.join(os.environ.get("TEMP", "/tmp"))))
    p.add_argument("--decode-check", action="store_true",
                   help="帧数守恒后用 ffmpeg 解码验证")
    p.add_argument("--beat", action="store_true",
                   help="watchdog 每 2s 输出 [beat] vcl= 心跳（--run-all 子进程自动启用）")
    p.add_argument("--run-all", action="store_true",
                   help="全变体回归矩阵：按 VARIANT_CANONICAL 规范配置独立子进程运行")
    p.add_argument("--timeout", type=int, default=75,
                   help="--run-all 每个变体硬超时（秒）")
    p.add_argument("--stall-timeout", type=int, default=15,
                   help="--run-all 判定真死锁的 vcl 停滞秒数（0=关闭；仅统计心跳时有效）")
    p.add_argument("--skip-reproducers", action="store_true",
                   help="--run-all 跳过预期失败复现器（delayed/aux_stay/"
                        "delayed_aux_stay/ce_pipeline/nonblocking），只跑回归 PASS 集")
    p.add_argument("--self-test", action="store_true",
                   help="无 GPU 自检（验证脚本自身记账逻辑）")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not _REF_LOADED:
        print(f"[FATAL] 无法导入 test_nvenc_la_frame_conservation.py "
              f"（本脚本复用其 MinimalTestEncoder / SDK 常量；"
              f"通常因本机未安装 PyTorch，GPU 测试机需要 torch）。\n"
              f"  错误: {_REF_ERROR}", flush=True)
        return 2
    if args.run_all:
        return run_all(args)
    return run_single(args)


if __name__ == "__main__":
    sys.exit(main())
