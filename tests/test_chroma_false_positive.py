#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: verify chroma checking in verify_segment_bitstream_v5.py.

Tests two scenarios:
1. Normal video with random noise -> should NOT trigger false positive (bad_count < 3)
2. Video with actual chroma corruption -> SHOULD trigger (bad_count >= 3)

Before fix [FIX-CHROMA-FA1]: Scenario 1 triggered false positives because
temporal jump mask was OR'd into bad_any, catching normal noise fluctuations.
"""
import os
import sys
import subprocess
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_chroma_check():
    """[FIX-VERIFY-RENAME] 取 check_chroma_corruption。

    2026-09-15 起该验收脚本在生产侧已改名为 `verify_segment_bitstream_v5.py`
    （内容与旧 `..._v4.py` 逐字节相同，见 Plan/门禁与测试资产纳管清理_立项Prompt.md）。
    这里做**双名兼容**，避免改名期间任一侧缺失导致本测试 ModuleNotFoundError。
    """
    import importlib
    last = None
    for _name in ("verify_segment_bitstream_v5", "verify_segment_bitstream_v4"):
        try:
            return getattr(importlib.import_module(_name), "check_chroma_corruption")
        except (ImportError, AttributeError) as e:
            last = e
    raise ImportError("找不到 verify_segment_bitstream_v5 / _v4 里的 "
                      "check_chroma_corruption: %s" % last)


def make_yuv420_frame(width, height, frame_idx, rng, chroma_std=25.0):
    """Generate one YUV420 frame with gradient + realistic noise."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base_u = 90.0 + (frame_idx % 5) * 3.0
    base_v = 120.0 + (frame_idx % 7) * 2.0

    y_plane = (xx + yy) * 0.5
    y_plane += rng.normal(0, 15, y_plane.shape)
    y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)

    xu = np.linspace(0, 80, width // 2, dtype=np.float32)
    yu = np.linspace(0, 80, height // 2, dtype=np.float32)
    xxu, yyu = np.meshgrid(xu, yu)
    u_plane = base_u + xxu * 0.3 + yyu * 0.1
    u_plane += rng.normal(0, chroma_std, u_plane.shape)
    u_plane = np.clip(u_plane, 0, 255).astype(np.uint8)

    xv = np.linspace(0, 80, width // 2, dtype=np.float32)
    yv = np.linspace(0, 80, height // 2, dtype=np.float32)
    xxv, yyv = np.meshgrid(xv, yv)
    v_plane = base_v + xxv * 0.2 + yyv * 0.4
    v_plane += rng.normal(0, chroma_std, v_plane.shape)
    v_plane = np.clip(v_plane, 0, 255).astype(np.uint8)

    return y_plane.tobytes() + u_plane.tbytes() + v_plane.tobytes()


def make_yuv420_frame(y_plane, u_plane, v_plane):
    """Assemble YUV420 frame from separate planes."""
    return y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()


def generate_normal_frame(width, height, frame_idx, rng):
    """Generate a frame with natural noise (std ~25)."""
    y = np.full((height, width), 128, dtype=np.float32)
    y += np.linspace(0, 50, width, dtype=np.float32)
    y += rng.normal(0, 8, (height, width))
    y = np.clip(y, 0, 255).astype(np.uint8)

    cw, ch = width // 2, height // 2
    u = np.full((ch, cw), 90.0, dtype=np.float32)
    u += np.linspace(0, 30, cw, dtype=np.float32)
    u += rng.normal(0, 25, (ch, cw))  # natural chroma noise
    u = np.clip(u, 0, 255).astype(np.uint8)

    v = np.full((ch, cw), 120.0, dtype=np.float32)
    v += np.linspace(0, 20, cw, dtype=np.float32)
    v += rng.normal(0, 25, (ch, cw))
    v = np.clip(v, 0, 255).astype(np.uint8)

    return make_yuv420_frame(y, u, v)


def generate_corrupted_frame(width, height, frame_idx, rng):
    """Generate a frame with chroma corruption at certain frames.

    Corruption: U plane std suddenly jumps to 200 (vs baseline ~30).
    This simulates NVENC cross-stream race condition causing color bleeding.
    """
    y = np.full((height, width), 128, dtype=np.float32)
    y += np.linspace(0, 50, width, dtype=np.float32)
    y += rng.normal(0, 8, (height, width))
    y = np.clip(y, 0, 255).astype(np.uint8)

    cw, ch = width // 2, height // 2
    # Frames 40-42, 70-72, 100-102 have corrupted chroma
    if frame_idx in range(40, 43) or frame_idx in range(70, 73) or frame_idx in range(100, 103):
        u = np.full((ch, cw), 90.0, dtype=np.float32)
        u += rng.normal(0, 200, (ch, cw))  # massive corruption
        u = np.clip(u, 0, 255).astype(np.uint8)
        v = np.full((ch, cw), 120.0, dtype=np.float32)
        v += rng.normal(0, 200, (ch, cw))
        v = np.clip(v, 0, 255).astype(np.uint8)
    else:
        u = np.full((ch, cw), 90.0, dtype=np.float32)
        u += np.linspace(0, 30, cw, dtype=np.float32)
        u += rng.normal(0, 25, (ch, cw))
        u = np.clip(u, 0, 255).astype(np.uint8)
        v = np.full((ch, cw), 120.0, dtype=np.float32)
        v += np.linspace(0, 20, cw, dtype=np.float32)
        v += rng.normal(0, 25, (ch, cw))
        v = np.clip(v, 0, 255).astype(np.uint8)

    return make_yuv420_frame(y, u, v)


def write_yuv_then_encode(raw_path, mp4_path, n_frames, width, height, frame_gen, rng):
    """Write raw YUV420 frames then encode to MP4."""
    with open(raw_path, 'wb') as f:
        for i in range(n_frames):
            f.write(frame_gen(width, height, i, rng))
    cmd = [
        'ffmpeg', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
        '-s', f'{width}x{height}', '-r', '30',
        '-i', raw_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        mp4_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_normal_video_no_false_positive():
    """A normal video with natural chroma noise should NOT trigger chroma FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, 'normal.mp4')
        raw_path = os.path.join(tmpdir, 'normal.yuv')
        rng = np.random.default_rng(42)
        write_yuv_then_encode(raw_path, video_path, 120, 640, 360, generate_normal_frame, rng)
        os.unlink(raw_path)

        check_chroma_corruption = _load_chroma_check()
        result = check_chroma_corruption(video_path, hwaccel=False, n_shards=1)
        assert result is not None, "check_chroma_corruption returned None"

        print(f"\n  Normal video:")
        print(f"    frame_count: {result['frame_count']}")
        print(f"    median_u: {result['median_u']}")
        print(f"    median_v: {result['median_v']}")
        print(f"    bad_count: {result['bad_count']}")
        print(f"    bad_frames: {result['bad_frames']}")

        assert result['bad_count'] < 3, (
            f"False positive! Normal video triggered chroma FAIL: "
            f"bad_count={result['bad_count']}, bad_frames={result['bad_frames']}"
        )
        print("    [OK] No false positive — chroma check passed.")


def test_corrupted_video_detected():
    """A video with real chroma corruption SHOULD trigger chroma FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, 'corrupted.mp4')
        raw_path = os.path.join(tmpdir, 'corrupted.yuv')
        rng = np.random.default_rng(99)
        # 3 clusters of corruption (at frames ~41, ~71, ~101)
        write_yuv_then_encode(raw_path, video_path, 120, 640, 360, generate_corrupted_frame, rng)
        os.unlink(raw_path)

        check_chroma_corruption = _load_chroma_check()
        result = check_chroma_corruption(video_path, hwaccel=False, n_shards=1)
        assert result is not None, "check_chroma_corruption returned None"

        print(f"\n  Corrupted video:")
        print(f"    frame_count: {result['frame_count']}")
        print(f"    median_u: {result['median_u']}")
        print(f"    median_v: {result['median_v']}")
        print(f"    bad_count: {result['bad_count']}")
        print(f"    bad_frames: {result['bad_frames']}")

        assert result['bad_count'] >= 3, (
            f"False negative! Corrupted video NOT detected: "
            f"bad_count={result['bad_count']}"
        )
        print("    [OK] Corruption detected — chroma check correctly flagged.")


if __name__ == '__main__':
    test_normal_video_no_false_positive()
    test_corrupted_video_detected()
    print("\nAll tests passed.")
