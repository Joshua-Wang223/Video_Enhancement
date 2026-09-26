#!/usr/bin/env python
"""test_frame_count_probe.py —— 校验 [PROBE-OPT] 帧数探测优化（P0~P4）。

纯 ffprobe/ffmpeg 调用，不占用 GPU，可安全单独运行。

用法:
  python Accessory/test/test_frame_count_probe.py [视频路径 ...]
不传参数时使用仓库内的既有素材（若存在）。
"""
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "utils"))

from video_utils import (  # noqa: E402
    count_decoded_video_frames, count_frames_parallel, probe_stats,
)


def check_strict_cache_provenance():
    """[FIX-GATE-STRICT-COUNT] 缓存「来源可信度」回归 —— 不需要 ffmpeg / GPU。

    背景：`_PROBE_FRAME_CACHE` 的 key 只有「路径+大小+mtime」，不带 mode。于是
    「先用 auto 命中容器元数据」写下的低可信值，会被之后要求 mode='decode' 的
    验收门调用直接复用 —— decode 语义被静默降级（验收盲区）。上游
    count_frames_parallel() 预热默认就是 auto，所以这个组合是真实路径。

    本检查用桩替换三个计数器，断言：
      ① auto 首次 → 命中元数据（来源标记 'metadata'）
      ② decode 必须绕开元数据缓存，拿到真解码值
      ③ decode 的真值会升级缓存（来源标记变 'decode'）
      ④ 之后的 auto 调用复用升级后的严格值（不再回落到元数据）
    """
    import shutil as _shutil
    import tempfile
    import video_utils as vu

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    p = Path(tmp)
    saved = (vu._read_nb_frames_metadata, vu._count_frames_ffprobe,
             vu._count_frames_nvdec, _shutil.which)
    # 桩：元数据说 100，真解码是 98，无 NVDEC（'which' 伪造为可用）
    vu._read_nb_frames_metadata = lambda path, ffprobe: 100
    vu._count_frames_ffprobe = lambda path, ffprobe: 98
    vu._count_frames_nvdec = lambda path, ffmpeg: None
    _shutil.which = lambda name: name
    try:
        vu._PROBE_FRAME_CACHE.clear()
        n_auto = vu.count_decoded_video_frames(p, mode="auto", use_hwaccel=False)
        k_auto = vu._PROBE_FRAME_CACHE[vu._probe_cache_key(p)][1]
        n_dec = vu.count_decoded_video_frames(p, mode="decode", use_hwaccel=False)
        k_dec = vu._PROBE_FRAME_CACHE[vu._probe_cache_key(p)][1]
        n_auto2 = vu.count_decoded_video_frames(p, mode="auto", use_hwaccel=False)
    finally:
        (vu._read_nb_frames_metadata, vu._count_frames_ffprobe,
         vu._count_frames_nvdec, _shutil.which) = saved
        try:
            os.unlink(tmp)
        except OSError:
            pass

    rows = [
        ("auto 首次命中元数据（100/来源 metadata）",
         n_auto == 100 and k_auto == "metadata", f"{n_auto} / {k_auto}"),
        ("decode 必须绕开元数据缓存（应得 98）", n_dec == 98, str(n_dec)),
        ("decode 真值升级缓存（来源 decode）", k_dec == "decode", k_dec),
        ("其后 auto 复用严格值（应得 98）", n_auto2 == 98, str(n_auto2)),
    ]
    print("\n■ [FIX-GATE-STRICT-COUNT] 缓存来源可信度")
    ok_all = True
    for name, ok, got in rows:
        ok_all = ok_all and ok
        print(f"  {'✅' if ok else '❌'} {name} → {got}")
    return ok_all


def main():
    strict_ok = check_strict_cache_provenance()

    args = [a for a in sys.argv[1:] if a]
    if args:
        paths = [Path(a) for a in args]
    else:
        cands = [
            _ROOT / "temp" / "wws3e02_26s.mp4",
            _ROOT / "temp" / "retest" / "esr_test_60s.mp4",
            _ROOT / "temp" / "retest" / "fake_upscaled_25min.mp4",
        ]
        paths = [c for c in cands if c.exists()]
    if not paths:
        print("未找到测试视频，请显式传入路径")
        return 1

    print("=" * 68)
    print("帧数探测优化校验")
    print("=" * 68)

    for p in paths:
        print(f"\n■ {p.name}")
        # 1) metadata 模式（最快）
        t0 = time.perf_counter()
        m = count_decoded_video_frames(p, mode="metadata", use_cache=False)
        t_meta = time.perf_counter() - t0

        # 2) decode 模式（最严格，等价原实现）
        t0 = time.perf_counter()
        d = count_decoded_video_frames(p, mode="decode", use_cache=False)
        t_dec = time.perf_counter() - t0

        # 3) auto 模式（默认）
        t0 = time.perf_counter()
        a = count_decoded_video_frames(p, mode="auto", use_cache=False)
        t_auto = time.perf_counter() - t0

        print(f"  metadata={m}  ({t_meta:.2f}s)")
        print(f"  decode  ={d}  ({t_dec:.2f}s)")
        print(f"  auto    ={a}  ({t_auto:.2f}s)")
        ok = (d is None) or (m == d == a)
        print(f"  一致性: {'✅ 三种模式一致' if ok else '⚠️ 不一致（auto 可能用了元数据）'}")
        if d is not None and t_dec > 0.5:
            print(f"  提速比 (decode/auto) = {t_dec / max(t_auto, 1e-6):.1f}x")

    # 缓存命中校验
    p0 = paths[0]
    print(f"\n■ 缓存校验 ({p0.name})")
    count_decoded_video_frames(p0)   # 首次（写入缓存）
    h0 = probe_stats()["cache_hits"]
    t0 = time.perf_counter()
    count_decoded_video_frames(p0)   # 二次（应命中）
    t_hit = time.perf_counter() - t0
    h1 = probe_stats()["cache_hits"]
    print(f"  二次调用耗时 {t_hit * 1000:.2f} ms，cache_hits {h0} → {h1} "
          f"{'✅ 命中' if h1 > h0 else '❌ 未命中'}")

    # 并行预热校验
    if len(paths) > 1:
        print(f"\n■ 并行预热校验（{len(paths)} 个文件）")
        t0 = time.perf_counter()
        res = count_frames_parallel(paths, max_workers=min(4, len(paths)))
        print(f"  并行耗时 {time.perf_counter() - t0:.2f}s，结果 {res}")

    print("\n■ 探测统计")
    for k, v in probe_stats().items():
        print(f"  {k} = {v}")
    return 0 if strict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
