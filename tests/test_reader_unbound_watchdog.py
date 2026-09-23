#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[FIX-READER-UNBOUND] IFRNet 读帧器 `read()` 有界化 —— 纯 CPU 回归测试。

背景（详见 Plan/IFRNet读帧器无界阻塞_立项Prompt.md 与
memory/ifrnet-reader-unbounded-queue-get.md）：
`FFmpegFrameReader.read()` 原先是裸 `self._queue.get()`，只要读线程不再投递
任何东西（子 ffmpeg 被 SIGTTOU 停住 / 死锁而不退出），调用方就永久静默挂起。

本文件覆盖三类断言，全部**不需要 GPU、不需要 torch、不需要 ffmpeg-python**：
  A. 语义分支（合成桩，无子进程）：死亡判据、观察窗、哨兵/异常透传、看门狗开关；
  B. 真实 ffmpeg 正常路径：帧数守恒 + 逐字节与参考解码一致 + 反压不丢帧；
  C. 真实 ffmpeg 注入路径：复现「线程静默死亡」与「线程活着但永不产出」两种
     挂死签名，断言 `read()` 在限额内抛 `RuntimeError` 而非挂死。

跑法：
    python tests/test_reader_unbound_watchdog.py            # 直接跑
    python -m pytest tests/test_reader_unbound_watchdog.py  # pytest 亦可

⚠️ 本机（Windows，无 GPU）已实测通过 A/B/C；判据 3 在 Linux 上的**真 SIGSTOP**
   变体仍应在 GPU 环境补跑一次（见返回的 GPU 待验证清单）。
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "external", ROOT / "external" / "ifrnet_video",
           ROOT / "src" / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------------------------
# 模块加载：本机可能没有 torch（Windows 开发树），ifrnet_utils 顶层 import torch，
# 故仅在缺失时补一个 MagicMock 桩 —— 只要不真的跑推理，桩不影响被测逻辑。
# --------------------------------------------------------------------------
def _load_ffmpeg_io():
    try:
        import torch  # noqa: F401
    except ImportError:
        from unittest import mock
        sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
        sys.modules.setdefault("torch.nn", mock.MagicMock(name="torch.nn"))
        sys.modules.setdefault(
            "torch.nn.functional", mock.MagicMock(name="torch.nn.functional"))
    import importlib
    return importlib.import_module("ifrnet_video.ffmpeg_io")


io = _load_ffmpeg_io()
FFmpegFrameReader = io.FFmpegFrameReader


# --------------------------------------------------------------------------
# 合成桩：绕过 __init__（它要 ffprobe + ffmpeg），只装配被 read() 依赖的字段。
# --------------------------------------------------------------------------
class _FakeProc:
    """只有 poll()/terminate() 的假子进程。rc=None 表示"仍在运行"。"""

    def __init__(self, rc=None):
        self._rc = rc
        self.terminated = False

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True


def _orphan_reader(timeout=0.3, frames_read=0, thread=None, rc=None):
    rd = FFmpegFrameReader.__new__(FFmpegFrameReader)
    rd._queue = queue.Queue(maxsize=4)
    rd._read_timeout = timeout
    rd._frames_read = frames_read
    rd._proc = _FakeProc(rc)
    rd._thread = thread if thread is not None else _live_thread()
    return rd


def _live_thread():
    t = threading.Thread(target=lambda: time.sleep(30), daemon=True)
    t.start()
    return t


def _dead_thread():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


# ══════════════════════════════════════════════════════════════════════════
# A. 语义分支（合成桩）
# ══════════════════════════════════════════════════════════════════════════
def test_env_resolution():
    """IFRNET_READER_TIMEOUT 解析：缺省/非法→120s；0/负数→关闭（0.0）。"""
    key = io._READER_TIMEOUT_ENV
    saved = os.environ.pop(key, None)
    try:
        assert io._reader_timeout_from_env() == io._READER_DEFAULT_TIMEOUT
        for raw, want in (("0", 0.0), ("-1", 0.0), ("90", 90.0),
                          ("abc", io._READER_DEFAULT_TIMEOUT),
                          ("  ", io._READER_DEFAULT_TIMEOUT)):
            os.environ[key] = raw
            got = io._reader_timeout_from_env()
            assert got == want, f"{raw!r} -> {got}, 期望 {want}"
    finally:
        os.environ.pop(key, None)
        if saved is not None:
            os.environ[key] = saved
    print("  [A1] 环境变量解析 OK")


def test_dead_thread_raises_within_one_timeout():
    """线程已停（未投递哨兵）→ ≤1×T 抛 RuntimeError，消息含四要素现场。"""
    rd = _orphan_reader(timeout=0.2, frames_read=7, thread=_dead_thread())
    t0 = time.monotonic()
    try:
        rd.read()
        raise AssertionError("线程已死却未抛异常")
    except RuntimeError as e:
        msg = str(e)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"耗时 {elapsed:.2f}s 超出限额"
    for token in ("[FIX-READER-UNBOUND]", "reader_thread_dead",
                  "thread_alive=", "child_poll=", "queue=", "frame=7"):
        assert token in msg, f"异常信息缺少 {token!r}: {msg}"
    print(f"  [A2] 线程死亡路径 OK（{elapsed:.2f}s）")


def test_child_exited_without_sentinel_raises():
    """子进程已退出却没有哨兵 → ≤1×T 抛，依据为 child_exited。"""
    rd = _orphan_reader(timeout=0.2, rc=137)
    t0 = time.monotonic()
    try:
        rd.read()
        raise AssertionError("子进程已退出却未抛异常")
    except RuntimeError as e:
        msg = str(e)
    assert time.monotonic() - t0 < 1.0
    assert "child_exited(rc=137" in msg, msg
    print("  [A3] 子进程退出路径 OK")


def test_alive_but_silent_is_bounded_at_two_timeouts():
    """线程活着 + 子进程活着但永不产出（SIGTTOU 停住/死锁签名）→ 2×T 内抛。"""
    rd = _orphan_reader(timeout=0.2)
    t0 = time.monotonic()
    try:
        rd.read()
        raise AssertionError("永久静默却未抛异常")
    except RuntimeError as e:
        msg = str(e)
    elapsed = time.monotonic() - t0
    assert 0.2 <= elapsed < 1.5, f"耗时 {elapsed:.2f}s 不符合 2×T 预期"
    assert "producer_alive_but_silent_2xT" in msg, msg
    assert "child_running" in msg, msg
    print(f"  [A4] 活着但静默路径 OK（{elapsed:.2f}s ≈ 2×T）")


def test_sentinel_exception_and_value_passthrough():
    """返回契约不变：哨兵→None；异常对象→原样 raise；帧元组→原样返回。"""
    rd = _orphan_reader(timeout=0.5)
    rd._queue.put(FFmpegFrameReader._SENTINEL)
    assert rd.read() is None
    rd._queue.put(ValueError("boom"))
    try:
        rd.read()
        raise AssertionError("异常对象未被 raise")
    except ValueError as e:
        assert str(e) == "boom"
    rd._queue.put((1, 2))
    assert rd.read() == (1, 2)
    print("  [A5] 哨兵/异常/数据透传契约 OK")


def test_timeout_zero_disables_watchdog():
    """timeout<=0 → 关闭看门狗：队列空时不抛（退回原无界阻塞语义）。"""
    rd = _orphan_reader(timeout=0.1, thread=_dead_thread())
    got = []
    th = threading.Thread(target=lambda: got.append(rd.read(timeout=0)),
                          daemon=True)
    th.start()
    th.join(timeout=0.6)
    assert th.is_alive(), "timeout=0 本应无界阻塞，却提前返回/抛异常"
    rd._queue.put((7, 7))          # 解除阻塞
    th.join(timeout=1.0)
    assert got == [(7, 7)], got
    print("  [A6] 看门狗可关闭（回滚开关）OK")


# ══════════════════════════════════════════════════════════════════════════
# B/C. 真实 ffmpeg 路径
# ══════════════════════════════════════════════════════════════════════════
N_FRAMES, W, H, FPS = 24, 320, 240, 10
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


def _make_clip(path: Path):
    """合成 24 帧确定性素材（testsrc，10fps × 2.4s）。"""
    subprocess.run(
        [_FFMPEG, "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate={FPS}:duration=2.4",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-g", "10", str(path)],
        check=True, stdin=subprocess.DEVNULL, capture_output=True)


def _count_frames(path: Path) -> int:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, stdin=subprocess.DEVNULL, capture_output=True,
        text=True).stdout.strip()
    return int(out)


def _reference_rgb(path: Path) -> bytes:
    """与读帧器同参数的参考解码（软解 + 单线程，保证确定性）。"""
    sync = (["-fps_mode", "passthrough"] if io._ffmpeg_has_fps_mode(_FFMPEG)
            else ["-vsync", "0"])
    return subprocess.run(
        [_FFMPEG, "-hide_banner", "-v", "error", "-noautorotate",
         "-threads", "1", "-i", str(path)] + sync
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, stdin=subprocess.DEVNULL, capture_output=True).stdout


def _drain(reader, slow_ms=0.0, limit=N_FRAMES * 4):
    frames = []
    while True:
        pair = reader.read(timeout=10.0)
        if pair is None:
            break
        frames.append(pair)
        if slow_ms:
            time.sleep(slow_ms / 1000.0)
        assert len(frames) <= limit, "读出的帧数超过素材总帧数，疑似未收敛"
    return frames


def test_real_normal_path_frame_conservation_and_bytes():
    """判据 1：逐帧返回；帧数 == ffprobe -count_frames；逐帧字节与参考解码一致。"""
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        _make_clip(clip)
        expected = _count_frames(clip)
        assert expected == N_FRAMES, f"素材帧数异常: {expected}"

        rd = FFmpegFrameReader(str(clip), prefetch=4, use_hwaccel=False)
        try:
            frames = _drain(rd)
        finally:
            rd.close()
        assert len(frames) == expected, f"帧数 {len(frames)} != {expected}"
        assert rd._frames_read == expected, f"_frames_read={rd._frames_read}"

        got = b"".join(bytes(f[0].tobytes()) for f in frames)
        ref = _reference_rgb(clip)
        assert len(got) == len(ref), f"字节数 {len(got)} != 参考 {len(ref)}"
        assert got == ref, "逐帧字节与参考解码不一致"
        for raw, _padded in frames:
            assert raw.shape == (H, W, 3), raw.shape
    print(f"  [B1] 正常路径 OK（{expected} 帧，逐字节与参考解码一致）")


def test_real_backpressure_slow_consumer():
    """判据 4：消费速率 << 生产速率时，不抛异常、无丢帧、内存不增长。"""
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        _make_clip(clip)
        expected = _count_frames(clip)
        rd = FFmpegFrameReader(str(clip), prefetch=2, use_hwaccel=False)
        try:
            frames = _drain(rd, slow_ms=15.0)
        finally:
            rd.close()
        assert len(frames) == expected, f"反压下丢帧: {len(frames)} != {expected}"
        assert rd._queue.maxsize == 4, "prefetch=2 时应为 max(prefetch,4)=4"
    print(f"  [B2] 反压路径 OK（慢消费仍读满 {expected} 帧）")


def test_real_injection_dead_loop_raises():
    """判据 2（真实子进程）：`_read_loop` 直接 return 且不投哨兵 → 限额内抛。"""
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        _make_clip(clip)
        with mock.patch.object(FFmpegFrameReader, "_read_loop",
                               lambda self: None):
            rd = FFmpegFrameReader(str(clip), prefetch=4, use_hwaccel=False)
            rd._read_timeout = 0.3
            try:
                t0 = time.monotonic()
                try:
                    rd.read()
                    raise AssertionError("静默死亡却未抛异常")
                except RuntimeError as e:
                    msg = str(e)
                elapsed = time.monotonic() - t0
            finally:
                rd.close()
        assert elapsed < 2.0, f"耗时 {elapsed:.2f}s 超出限额"
        assert "reader_thread_dead" in msg, msg
    print(f"  [C1] 注入·线程静默死亡 OK（{elapsed:.2f}s）")


def test_real_injection_stalled_loop_raises():
    """判据 3（真实子进程）：读线程活着但永不产出（SIGTTOU/死锁签名）→ 限额内抛。"""
    import unittest.mock as mock
    stop = threading.Event()
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        _make_clip(clip)
        with mock.patch.object(FFmpegFrameReader, "_read_loop",
                               lambda self: stop.wait(30)):
            rd = FFmpegFrameReader(str(clip), prefetch=4, use_hwaccel=False)
            rd._read_timeout = 0.3
            try:
                t0 = time.monotonic()
                try:
                    rd.read()
                    raise AssertionError("活着的静默生产者却未抛异常")
                except RuntimeError as e:
                    msg = str(e)
                elapsed = time.monotonic() - t0
                assert rd._thread.is_alive(), "线程本应仍在（模拟停住）"
                assert rd._proc.poll() is None, "子进程本应仍在（模拟停住）"
            finally:
                stop.set()
                rd.close()
        assert elapsed < 3.0, f"耗时 {elapsed:.2f}s 超出 2×T 预期"
        assert "producer_alive_but_silent_2xT" in msg, msg
    print(f"  [C2] 注入·活着但静默 OK（{elapsed:.2f}s ≈ 2×T）")


def test_real_sigstop_child_ffmpeg_raises():
    """判据 3（Linux 真变体）：`kill -STOP` 子 ffmpeg → read() 在限额内抛而非挂死。

    与 C1/C2 的合成注入不同，这里停的是**真实子 ffmpeg 进程**（正是
    `memory/env-ffmpeg-ffprobe-gotchas.md` 里 SIGTTOU 停住的那条路径）：
    线程活着、子进程未退出、队列却再也不来数据 —— 看门狗必须判「活着但静默」
    并在 2×T 内抛出，而不是无限等下去。
    """
    if os.name == "nt" or not hasattr(os, "kill"):
        print("  [C3] SKIP（需 POSIX 信号）")
        return
    import signal as _signal
    if not _FFMPEG:
        print("  [C3] SKIP（需 ffmpeg）")
        return

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "long.mp4"
        # 长片（20s）：确保 SIGSTOP 时 ffmpeg 仍在产出，而不是早已把整段写完
        subprocess.run(
            [_FFMPEG, "-hide_banner", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate={FPS}:duration=20",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-g", "10", str(clip)],
            check=True, stdin=subprocess.DEVNULL, capture_output=True)

        rd = FFmpegFrameReader(str(clip), prefetch=4, use_hwaccel=False)
        pid = rd._proc.pid
        elapsed = None
        try:
            rd._read_timeout = 5.0                 # 预热宽限
            for _ in range(3):
                rd.read()
            assert rd._proc.poll() is None, "预热后子进程已退出，素材过长假设不成立"

            rd._read_timeout = 0.5                 # 2×T = 1.0s
            os.kill(pid, _signal.SIGSTOP)
            assert rd._proc.poll() is None, "SIGSTOP 后 poll() 应为 None（未退出）"

            raised, elapsed = None, None
            t_start = time.monotonic()
            for _ in range(100000):
                t_call = time.monotonic()
                try:
                    rd.read()
                except RuntimeError as e:
                    raised = str(e)
                    elapsed = time.monotonic() - t_call
                    break
                if time.monotonic() - t_start > 60:
                    raise AssertionError("SIGSTOP 后 read() 超 60s 仍未抛（仍会挂死）")

            assert raised is not None, "SIGSTOP 后 read() 未抛异常"
            assert "producer_alive_but_silent_2xT" in raised, raised
            assert "thread_alive=True" in raised, raised
            assert "child_poll=None" in raised, raised
            assert rd._thread.is_alive(), "读线程本应仍活着（被停住≠已死）"
            assert 0.5 <= elapsed <= 2.0, \
                f"最后一次 read() 耗时 {elapsed:.2f}s 不在 2×T≈1.0s 附近"
        finally:
            # 必须 SIGCONT：SIGTERM 不会终止 STOPPED 进程（会一直挂着）
            try:
                os.kill(pid, _signal.SIGCONT)
            except OSError:
                pass
            rd.close()

    print(f"  [C3] 真 SIGSTOP OK（最后一帧 read() 于 {elapsed:.2f}s 抛出 ≈ 2×T，"
          f"子进程仍 STOPPED 时已被判为「活着但静默」）")


# ══════════════════════════════════════════════════════════════════════════
def _main():
    if not _FFMPEG or not _FFPROBE:
        print("SKIP: 未找到 ffmpeg/ffprobe，仅跑 A 组语义分支")
    tests = [test_env_resolution,
             test_dead_thread_raises_within_one_timeout,
             test_child_exited_without_sentinel_raises,
             test_alive_but_silent_is_bounded_at_two_timeouts,
             test_sentinel_exception_and_value_passthrough,
             test_timeout_zero_disables_watchdog]
    if _FFMPEG and _FFPROBE:
        tests += [test_real_normal_path_frame_conservation_and_bytes,
                  test_real_backpressure_slow_consumer,
                  test_real_injection_dead_loop_raises,
                  test_real_injection_stalled_loop_raises,
                  test_real_sigstop_child_ffmpeg_raises]
    failures = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if not failures else 'FAILURES'}: "
          f"{len(tests) - failures}/{len(tests)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
