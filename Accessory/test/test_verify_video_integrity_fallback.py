#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test: `verify_video_integrity()` 的 cv2 → ffmpeg 回退（[FIX-AV1-CV2]）。

背景（2026-09-30 L40 实测，方案 §8.6-①）
------------------------------------------
`src/utils/video_utils.py::verify_video_integrity()` 原本用 OpenCV 读首帧判定
"文件是否完好"。OpenCV 4.13 自带的 FFmpeg **不含 AV1 解码**，对 NVENC 直出的
AV1 分段 `cap.read()` 返回 False，而同一个文件用系统 ffmpeg 解码完全正常
（1437/1437 帧、rc=0）。于是**完好的 AV1 产物被判成损坏**，上层随即把它
`unlink` 掉并终止整条流水线 ⇒ 本仓 AV1 端到端完全跑不通。

本测试锁三件事：
  1. cv2 判失败时，**必须**走 ffmpeg 探针复核（不再直接返回 False）；
  2. ffmpeg 能解出 1 帧 ⇒ 通过；解不出（垃圾文件）⇒ 仍判失败（**不放宽**）；
  3. 走的是 ffmpeg 而不是"无脑返回 True"——用垃圾文件反证。

不依赖 AV1 硬件：用 monkeypatch 模拟"cv2 读不出帧"，素材用 libx264 生成的
小文件即可（真实 AV1 场景已由 L40 冒烟覆盖，见 memory/l40-av1-pipeline-verification.md）。
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import video_utils as VU  # noqa: E402


def _make_sample(path, seconds=1):
    """生成一个 libx264 小样本（cv2 与 ffmpeg 都能解）。"""
    cmd = ["ffmpeg", "-hide_banner", "-y", "-v", "error", "-nostdin",
           "-f", "lavfi", "-i",
           "testsrc2=size=160x120:rate=15:duration=%d" % seconds,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
           "-pix_fmt", "yuv420p", path]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    return r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1024


class _BlindCap:
    """模拟"cv2 打不开/读不出帧"的 VideoCapture（如 OpenCV 无 AV1 解码器）。"""

    def __init__(self, *_a, **_k):
        pass

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


def test_cv2_failure_falls_back_to_ffmpeg():
    """cv2 读不出帧时，好文件必须仍判为完好（这是 AV1 能跑通的前提）。"""
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.mp4")
        if not _make_sample(good):
            print("[SKIP] ffmpeg 不可用，无法生成样本")
            return
        orig_cap = VU.cv2.VideoCapture
        VU.cv2.VideoCapture = _BlindCap
        try:
            assert VU.verify_video_integrity(good) is True, \
                "cv2 读失败后应回退 ffmpeg 探针，而不是直接判坏"
        finally:
            VU.cv2.VideoCapture = orig_cap


def test_fallback_still_rejects_broken_file():
    """回退**不放宽**判定：ffmpeg 解不出的垃圾文件仍必须判坏。"""
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.mp4")
        with open(bad, "wb") as fh:
            fh.write(b"\x00" * 4096)          # > 1KB，绕过体积下限这道闸
        orig_cap = VU.cv2.VideoCapture
        VU.cv2.VideoCapture = _BlindCap
        try:
            assert VU.verify_video_integrity(bad) is False, \
                "ffmpeg 探针解不出帧时不得放行（否则损坏文件会混过验收层）"
        finally:
            VU.cv2.VideoCapture = orig_cap


def test_missing_and_tiny_files_rejected():
    """缺文件 / <1KB 的老闸门不受影响。"""
    with tempfile.TemporaryDirectory() as td:
        assert VU.verify_video_integrity(os.path.join(td, "nope.mp4")) is False
        tiny = os.path.join(td, "tiny.mp4")
        with open(tiny, "wb") as fh:
            fh.write(b"x" * 100)
        assert VU.verify_video_integrity(tiny) is False


if __name__ == "__main__":
    for fn in (test_cv2_failure_falls_back_to_ffmpeg,
               test_fallback_still_rejects_broken_file,
               test_missing_and_tiny_files_rejected):
        fn()
        print("PASS %s" % fn.__name__)
