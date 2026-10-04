#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test: `quality_map._qp_model()` 的**口径感知**（为 M4/D2b 的 NVENC 等质量 QP 行铺路）。

背景
----
`to_constqp_qp()` 的 QP 轴模型由 `_qp_model()` 给出。改造前它在**任何口径**下都
**无条件**优先 `_QP_MAP_OVERRIDE`（硬编显式偏移），因此一旦把标定得到的 NVENC 行写进
`QUALITY_MAP_QP`（D2b），会被 override **永久遮蔽**——标了也不生效。

改造（2026-10-04）按口径分流：
  * **size 口径**：`_QP_MAP_OVERRIDE` → 活动表（**完全保持改造前行为**，判据 G3/G6 钉 size，
    期望值不变 ⇒ 零侵入）；
  * **quality 口径**：`QUALITY_MAP_QP`（标定表）优先 → `_QP_MAP_OVERRIDE`（未标定回退）→ 活动表。

本测试锁四件事（纯 CPU，不需 GPU）：
  1. size 口径：NVENC 走 `_QP_MAP_OVERRIDE`（h264 26→21、av1 27→63）；
  2. quality 口径：**T4 标定后** h264/hevc 命中 `QUALITY_MAP_QP` 标定行（≠ size 的
     override 值）；**未标定**的 av1_nvenc 仍回退 override（与 size 同值）；
  3. 前向兼容：注入覆盖行后，quality 用它、size 仍不用（**须保存/还原原行**）；
  4. 回归守卫：size 口径对软编仍走 `SIZE_MAP`，**不得**被 `QUALITY_MAP_QP` 影响。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quality_map as Q  # noqa: E402


def _set(mode):
    Q.set_quality_mode(mode)


def test_size_mode_uses_override():
    """size 口径：h264_nvenc 走基准轴直取（26→21）；av1_nvenc 走 ×3（27→63）。"""
    try:
        _set('size')
        assert Q.to_constqp_qp('h264_nvenc', 26) == 21
        assert Q.to_constqp_qp('av1_nvenc', 27) == 63
    finally:
        _set('quality')


def test_quality_mode_calibrated_nvenc_rows():
    """quality 口径：h264/hevc 命中 QUALITY_MAP_QP 标定行（≠ size 的 override）；
    av1_nvenc 未标定 ⇒ quality 也回退 override（与 size 同值）。"""
    assert 'h264_nvenc' in Q.QUALITY_MAP_QP, "h264_nvenc 标定行应已落表"
    assert 'hevc_nvenc' in Q.QUALITY_MAP_QP, "hevc_nvenc 标定行应已落表"
    assert 'av1_nvenc' not in Q.QUALITY_MAP_QP, "av1_nvenc 未标定，不应有行"
    try:
        _set('quality')
        a, b, lo, hi = Q.QUALITY_MAP_QP['h264_nvenc']
        ref = Q.to_x264_crf('h264_nvenc', 26)          # quality 口径基准轴
        exp = int(max(lo, min(hi, round(a * ref + b))))
        q_quality = Q.to_constqp_qp('h264_nvenc', 26)
        a_quality = Q.to_constqp_qp('av1_nvenc', 27)
        _set('size')
        q_size = Q.to_constqp_qp('h264_nvenc', 26)
        a_size = Q.to_constqp_qp('av1_nvenc', 27)
        # size 口径保持改造前行为（override 基准轴直取）
        assert q_size == 21
        assert a_quality == a_size == 63
        # quality 口径命中标定行，且与 size 的 21 可区分
        assert q_quality == exp and q_quality != q_size, \
            f"quality 应命中标定行 {exp}（≠ size {q_size}），实得 {q_quality}"
    finally:
        _set('quality')


def test_quality_mode_prefers_calibrated_row():
    """前向兼容：注入覆盖行后，quality 用它、size 仍走 override。"""
    injected = (1.0, 5.0, 0, 255)          # ref 21 → 26（与 override 的 21 可区分）
    orig = Q.QUALITY_MAP_QP.get('h264_nvenc')   # ⚠ 标定后已有真实行：须保存并还原
    Q.QUALITY_MAP_QP['h264_nvenc'] = injected
    try:
        _set('quality')
        assert Q.to_constqp_qp('h264_nvenc', 26) == 26, "quality 口径应优先 QUALITY_MAP_QP"
        _set('size')
        assert Q.to_constqp_qp('h264_nvenc', 26) == 21, "size 口径不得被 QUALITY_MAP_QP 影响"
    finally:
        if orig is None:
            Q.QUALITY_MAP_QP.pop('h264_nvenc', None)
        else:
            Q.QUALITY_MAP_QP['h264_nvenc'] = orig
        _set('quality')


def test_size_mode_ignores_quality_qp_for_soft():
    """回归守卫：size 口径对软编仍取 SIZE_MAP，不受 QUALITY_MAP_QP 影响。"""
    injected = (1.0, 5.0, 0, 63)
    orig = Q.QUALITY_MAP_QP.get('libx265')
    Q.QUALITY_MAP_QP['libx265'] = injected
    try:
        _set('size')
        sz = Q.to_constqp_qp('libx265', 26)
        _set('quality')
        ql = Q.to_constqp_qp('libx265', 26)
        # size 走 SIZE_MAP 的 libx265 行（往返恒等 ⇒ 26）
        # quality 走注入行 (1.0, 5.0)：按 quality 口径的基准轴反解后 +5
        ref = Q.to_x264_crf('libx265', 26)
        exp_ql = int(max(0, min(63, round(1.0 * ref + 5.0))))
        assert sz == 26, f"size 口径 libx265 应为 26，实得 {sz}"
        assert ql == exp_ql and ql != sz, \
            f"quality 口径 libx265 应命中注入行 {exp_ql}（≠ size {sz}），实得 {ql}"
    finally:
        if orig is None:
            Q.QUALITY_MAP_QP.pop('libx265', None)
        else:
            Q.QUALITY_MAP_QP['libx265'] = orig
        _set('quality')


if __name__ == "__main__":
    for fn in (test_size_mode_uses_override,
               test_quality_mode_calibrated_nvenc_rows,
               test_quality_mode_prefers_calibrated_row,
               test_size_mode_ignores_quality_qp_for_soft):
        fn()
        print("PASS %s" % fn.__name__)
    print("模式感知 4/4 通过")
