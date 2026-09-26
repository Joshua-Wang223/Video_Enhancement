#!/usr/bin/env python
"""[PROBE-CACHE-PERSIST] / [SCENE-CUT-PERSIST] 回归测试。

验证 [P3-2] 切镜预扫描与 [PROBE-OPT] 帧数预热的缓存落盘（断点恢复）：
  ① 成功结果落盘后，清空进程内缓存（模拟重启）再加载，调用直接命中、不重算；
  ② 失败结果（帧数 None / 扫描返回 None / 抛异常）**绝不落盘**；
  ③ 低可信来源（metadata）不落盘；
  ④ 解码证据（detail）只落「rc==0 且无错误行」的成功项；
  ⑤ 切镜 key 携带阈值 —— 改阈值必须 miss；
  ⑥ E2E（真实 ffmpeg + process 模式）：子进程各自持久化必须**累积**而非
     「最后写者胜」，且重启后可全部命中（零解码）。

①～⑤ 全程用桩替换真实探测，不调 ffmpeg/GPU；⑥ 需要 ffmpeg（缺失则跳过）。
用法: python Accessory/test/test_prescan_cache_persistence.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "utils"))
sys.path.insert(0, str(_ROOT / "external"))

import video_utils as vu  # noqa: E402
from ifrnet_video import pipeline as pl  # noqa: E402

_RESULTS = []


def _check(name, ok, got=""):
    _RESULTS.append(bool(ok))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" → {got}" if got else ""))


def _make_file(dirpath, name):
    p = os.path.join(dirpath, name)
    with open(p, "wb") as f:
        f.write(b"\x00" * 4096)
    return p


def test_probe_persist_roundtrip():
    """帧数缓存：成功落盘 + 重启命中 + 失败/元数据不落盘。"""
    print("\n■ [PROBE-CACHE-PERSIST] 帧数缓存落盘 / 重启命中")
    saved = (vu._read_nb_frames_metadata, vu._count_frames_ffprobe,
             vu._count_frames_nvdec, shutil.which)
    calls = {"decode": 0}
    meta_value = {"v": None}          # None → 元数据不可信，走解码
    decode_value = {"v": 123}

    with tempfile.TemporaryDirectory() as tmp:
        sidecar = os.path.join(tmp, "probe_cache.json")
        f_ok = _make_file(tmp, "seg_ok.mp4")
        f_fail = _make_file(tmp, "seg_fail.mp4")
        f_meta = _make_file(tmp, "seg_meta.mp4")

        def _dec(path, ffprobe):
            calls["decode"] += 1
            return decode_value["v"]

        vu._read_nb_frames_metadata = lambda path, ffprobe: meta_value["v"]
        vu._count_frames_ffprobe = _dec
        vu._count_frames_nvdec = lambda path, ffmpeg: None
        shutil.which = lambda name: name
        try:
            vu._PROBE_FRAME_CACHE.clear()
            vu._PROBE_DETAIL_CACHE.clear()
            vu.set_probe_cache_file(sidecar)

            # ① 成功解码 → 落盘
            v1 = vu.count_decoded_video_frames(f_ok, mode="decode",
                                               use_hwaccel=False)
            _check("首次解码取值正确", v1 == 123, str(v1))
            _check("sidecar 已写入", os.path.exists(sidecar),
                   os.path.basename(sidecar))
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("sidecar 含该段的成功项", len(data.get("frames", {})) == 1,
                   f"frames={len(data.get('frames', {}))}")

            # ② 模拟进程重启：清空内存缓存后重新加载 → 不再解码
            before = calls["decode"]
            vu._PROBE_FRAME_CACHE.clear()
            vu._PROBE_DETAIL_CACHE.clear()
            vu.set_probe_cache_file(sidecar)
            v2 = vu.count_decoded_video_frames(f_ok, mode="decode",
                                               use_hwaccel=False)
            _check("重启后命中落盘缓存（免重复解码）",
                   v2 == 123 and calls["decode"] == before,
                   f"v={v2}, 新增解码={calls['decode'] - before}")

            # ③ 失败（解码返回 None）不落盘
            decode_value["v"] = None
            v3 = vu.count_decoded_video_frames(f_fail, mode="decode",
                                               use_hwaccel=False)
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("失败结果不落盘",
                   v3 is None and len(data.get("frames", {})) == 1,
                   f"v={v3}, frames={len(data.get('frames', {}))}")

            # ④ 低可信 metadata 来源不落盘
            decode_value["v"] = 123
            meta_value["v"] = 456
            v4 = vu.count_decoded_video_frames(f_meta, mode="auto",
                                               use_hwaccel=False)
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            kinds = {e.get("kind") for e in data.get("frames", {}).values()}
            _check("metadata 来源不落盘",
                   v4 == 456 and kinds == {"decode"}, f"kinds={kinds}")

            # ⑤ detail：只有成功证据落盘
            vu._PROBE_DETAIL_CACHE.clear()
            k_ok = (f_ok, os.stat(f_ok).st_size, os.stat(f_ok).st_mtime_ns)
            k_bad = (f_fail, os.stat(f_fail).st_size, os.stat(f_fail).st_mtime_ns)
            vu._PROBE_DETAIL_CACHE[k_ok] = {
                "frames": 123, "error_lines": [], "hw_failed": False,
                "rc": 0, "stderr_tail": ""}
            vu._PROBE_DETAIL_CACHE[k_bad] = {
                "frames": 9, "error_lines": ["boom"], "hw_failed": False,
                "rc": 0, "stderr_tail": "boom"}
            vu._persist_probe_cache()
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("detail 只落成功证据（失败证据被剔除）",
                   len(data.get("detail", {})) == 1,
                   f"detail={len(data.get('detail', {}))}")
        finally:
            (vu._read_nb_frames_metadata, vu._count_frames_ffprobe,
             vu._count_frames_nvdec, shutil.which) = saved
            vu.set_probe_cache_file(None)
            vu._PROBE_FRAME_CACHE.clear()
            vu._PROBE_DETAIL_CACHE.clear()


def test_scene_cut_persist_roundtrip():
    """切镜缓存：成功落盘 + 重启命中 + 失败不落盘 + 阈值入 key。"""
    print("\n■ [SCENE-CUT-PERSIST] 切镜缓存落盘 / 重启命中")
    saved_scan = pl._scan_scene_cuts
    calls = {"n": 0}
    scan_ret = {"v": {3, 7}}

    def _fake_scan(path, threshold=None, threads=1, timeout=1800):
        calls["n"] += 1
        ret = scan_ret["v"]
        if isinstance(ret, Exception):
            raise ret
        return ret

    with tempfile.TemporaryDirectory() as tmp:
        sidecar = os.path.join(tmp, "scene_cuts.json")
        f_ok = _make_file(tmp, "segment_000.mp4")
        f_fail = _make_file(tmp, "segment_001.mp4")
        f_raise = _make_file(tmp, "segment_002.mp4")
        f_empty = _make_file(tmp, "segment_003.mp4")
        pl._scan_scene_cuts = _fake_scan
        try:
            pl._SCENE_CUT_CACHE.clear()
            pl._SCENE_CUT_CACHE_OK.clear()
            pl.set_scene_cut_cache_file(sidecar)

            # ① 预扫描成功 → 落盘
            done = pl.prescan_scene_cuts([f_ok], quiet=True)
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("预扫描成功并落盘",
                   done == 1 and len(data.get("entries", {})) == 1,
                   f"done={done}, entries={len(data.get('entries', {}))}")

            # ② 模拟重启：清空内存后重载 → 命中，不再扫描
            before = calls["n"]
            pl._SCENE_CUT_CACHE.clear()
            pl._SCENE_CUT_CACHE_OK.clear()
            pl.set_scene_cut_cache_file(sidecar)
            got = pl.detect_scene_cut_pairs(f_ok)
            _check("重启后切镜集合命中落盘缓存（免重复解码）",
                   got == {3, 7} and calls["n"] == before,
                   f"got={sorted(got)}, 新增扫描={calls['n'] - before}")

            # ③ 真无切镜的空集应落盘（与失败区分）
            scan_ret["v"] = set()
            pl.prescan_scene_cuts([f_empty], quiet=True)
            pl._SCENE_CUT_CACHE.clear()
            pl._SCENE_CUT_CACHE_OK.clear()
            pl.set_scene_cut_cache_file(sidecar)
            before = calls["n"]
            got_empty = pl.detect_scene_cut_pairs(f_empty)
            _check("真空集落盘且重启命中（扫描计数不增）",
                   got_empty == set() and calls["n"] == before,
                   f"got={got_empty}, 新增扫描={calls['n'] - before}")

            # ④ 扫描返回 None（ffmpeg 不可用）不落盘
            scan_ret["v"] = None
            pl.prescan_scene_cuts([f_fail], quiet=True)
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("扫描不可用的空集不落盘",
                   len(data.get("entries", {})) == 2,
                   f"entries={len(data.get('entries', {}))}")

            # ⑤ 扫描抛异常不落盘
            scan_ret["v"] = RuntimeError("boom")
            pl.prescan_scene_cuts([f_raise], quiet=True)
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            _check("扫描异常不落盘",
                   len(data.get("entries", {})) == 2,
                   f"entries={len(data.get('entries', {}))}")

            # ⑥ 阈值是 key 的一部分
            _check("阈值并入缓存 key",
                   pl._scene_cut_cache_key(f_ok, 0.25) !=
                   pl._scene_cut_cache_key(f_ok, 0.50))
            # ⑦ 判据版本并入 key
            _check("判据版本并入缓存 key",
                   "v=%d" % pl._SCENE_CUT_CRITERIA_VERSION in
                   pl._scene_cut_cache_key(f_ok, 0.25))
        finally:
            pl._scan_scene_cuts = saved_scan
            pl.set_scene_cut_cache_file(None)
            pl._SCENE_CUT_CACHE.clear()
            pl._SCENE_CUT_CACHE_OK.clear()


def test_process_mode_persist_e2e():
    """⑥ E2E：process 模式下子进程各自持久化必须累积，重启后可全部命中。

    这是 [PROBE-CACHE-PERSIST-PROC] 的回归：进程池用 fork 时子进程继承父进程
    缓存的**快照**，若各自直接覆盖写同一 sidecar，则「最后写者胜」——实测 8 段
    只留 1 条。修复（flock 串行「读-合并-写」+ 带 pid 的临时文件名）后应累积到 N 条。
    """
    print("\n■ [PROBE-CACHE-PERSIST] process 模式 E2E（真实 ffmpeg）")
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  ⏭️  未找到 ffmpeg，跳过 process 模式 E2E")
        _RESULTS.append(True)
        return

    n = 4
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i in range(n):
            p = os.path.join(tmp, f"c{i}.mp4")
            subprocess.run(
                [ffmpeg, "-hide_banner", "-v", "error", "-y",
                 "-f", "lavfi",
                 "-i", f"testsrc=size=160x120:rate=10:duration={1 + i * 0.5}",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "5", p],
                check=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append(p)

        sidecar = os.path.join(tmp, "probe_cache.json")
        vu.set_probe_cache_file(sidecar)
        # 模拟「未预热」：清空父进程内存缓存 → 子进程必须各自解码并持久化
        vu._PROBE_FRAME_CACHE.clear()
        vu._PROBE_DETAIL_CACHE.clear()

        results = vu.validate_decodable_video_batch(
            clips, expected_frames_list=[None] * n,
            workers=n, parallel_mode="process", gpu_workers=0)

        ok_n = sum(1 for ok, _ in results if ok)
        decoded = [r.get("decoded_frames") for _, r in results]
        _check("process 模式验收全部通过",
               ok_n == n and all(d for d in decoded),
               f"ok={ok_n}/{n}, frames={decoded}")

        # 关键：父进程内存缓存仍为空 ⇒ 工作确在独立子进程完成（未退化成 thread）
        _check("确为独立子进程执行（父进程内存缓存未被填充）",
               len(vu._PROBE_FRAME_CACHE) == 0,
               f"parent_cache={len(vu._PROBE_FRAME_CACHE)}")

        entries, valid = -1, False
        try:
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            valid = True
            entries = len(data.get("frames", {}))
        except Exception as e:
            _check("sidecar JSON 有效", False, str(e))
        if valid:
            _check("子进程结果全部累积落盘（非最后写者胜）",
                   entries == n, f"entries={entries}/{n}")

        # 重启：清内存后从 sidecar 重载，必须全部命中且零解码
        vu.set_probe_cache_file(None)
        vu._PROBE_FRAME_CACHE.clear()
        vu._PROBE_DETAIL_CACHE.clear()
        vu.set_probe_cache_file(sidecar)

        calls = {"n": 0}
        saved = (vu._count_frames_nvdec, vu._count_frames_ffprobe,
                 vu._read_nb_frames_metadata)

        def _boom(*_a, **_k):
            calls["n"] += 1
            raise AssertionError("命中缓存时不应触发任何解码")

        vu._count_frames_nvdec = _boom
        vu._count_frames_ffprobe = _boom
        vu._read_nb_frames_metadata = _boom
        try:
            vals = [vu.count_decoded_video_frames(p, mode="decode",
                                                  use_hwaccel=False)
                    for p in clips]
        finally:
            (vu._count_frames_nvdec, vu._count_frames_ffprobe,
             vu._read_nb_frames_metadata) = saved
            vu.set_probe_cache_file(None)
            vu._PROBE_FRAME_CACHE.clear()
            vu._PROBE_DETAIL_CACHE.clear()

        _check("重启后 process 模式落盘结果全部命中（零解码）",
               vals == decoded and calls["n"] == 0,
               f"hits={sum(1 for v in vals if v is not None)}/{n}, "
               f"解码调用={calls['n']}")


def main():
    print("=" * 68)
    print("预扫描缓存断点恢复（[PROBE-CACHE-PERSIST]/[SCENE-CUT-PERSIST]）")
    print("=" * 68)
    test_probe_persist_roundtrip()
    test_scene_cut_persist_roundtrip()
    test_process_mode_persist_e2e()
    total = len(_RESULTS)
    passed = sum(_RESULTS)
    print(f"\n{'=' * 68}\n结果: {passed}/{total} "
          f"{'✅ 全部通过' if passed == total else '❌ 存在失败'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
