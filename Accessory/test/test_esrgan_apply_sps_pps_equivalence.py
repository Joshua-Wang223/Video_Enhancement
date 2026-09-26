#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[P2.4c-LADDER-ESRGAN] `_apply_sps_pps` 与「改造前三段式内联阶梯」的等价性回归。

背景：`Plan/ESRGAN_LA二次排空安全网_立项Prompt.md` 的前置条件 1 要求把散布在
`external/realesrgan_video/nvenc_sdk.py` 里的 SPS/PPS 注入阶梯收敛到一个统一入口
`_apply_sps_pps()`，供后续 `[FIX-LA-REDRAIN]` 二次排空兜底复用。

实测（2026-09-15）该文件共 **11** 个内联阶梯站点，但只有 **3 处**（Phase1 harvest /
Phase3 drain / encode_frame）是同一个「三段式」变体，与 `_apply_sps_pps` 语义逐字
等价；其余 8 处语义不同（非 IDR 也注入 muxer / 只 prepend 不缓存 / 只缓存不注入），
故**只收敛了那 3 处**。

本测试用**改造前的内联阶梯**作为 oracle（逐字冻结），在完整真值表上比对：
  返回数据 / 缓存值 / `_sps_pps_injected` / muxer 调用次数 / 打印次数。
全部相等才说明"收敛是纯重构"。

不需要 GPU、不需要 torch（缺失时补 MagicMock 桩）。

跑法：
    python Accessory/test/test_esrgan_apply_sps_pps_equivalence.py
    python -m pytest Accessory/test/test_esrgan_apply_sps_pps_equivalence.py
"""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "external") not in sys.path:
    sys.path.insert(0, str(ROOT / "external"))


def _load_nvenc():
    try:
        import torch  # noqa: F401
    except ImportError:
        from unittest import mock
        sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
        sys.modules.setdefault("torch.nn", mock.MagicMock(name="torch.nn"))
    import importlib
    return importlib.import_module("realesrgan_video.nvenc_sdk")


nvenc = _load_nvenc()
NVENCEncoder = nvenc.NVENCEncoder


# ── 合成 Annex B 素材 ────────────────────────────────────────────────────
SPS = b"\x00\x00\x00\x01\x67" + b"\xaa" * 8      # H.264 type 7
PPS = b"\x00\x00\x00\x01\x68" + b"\xbb" * 4      # H.264 type 8
PARAMS = SPS + PPS
IDR = b"\x00\x00\x00\x01\x65" + b"\xcc" * 16     # type 5
PB = b"\x00\x00\x00\x01\x41" + b"\xdd" * 16      # type 1


def _old_ladder(self, h264_data, is_idr):
    """改造前的内联三段式阶梯（逐字冻结自 nvenc_sdk.py，未做任何"改进"）。"""
    if is_idr and self._cached_sps_pps is not None and \
            not self._has_sps_pps(h264_data):
        h264_data = self._cached_sps_pps + h264_data
    elif is_idr and self._cached_sps_pps is None and h264_data:
        self._cached_sps_pps = self._extract_sps_pps(h264_data)
        if self._cached_sps_pps:
            print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes"
                  % len(self._cached_sps_pps), flush=True)
            if self._muxer_ref is not None:
                try:
                    self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                    self._sps_pps_injected = True
                except Exception:
                    pass
    elif not is_idr and self._cached_sps_pps is None and h264_data:
        self._cached_sps_pps = self._extract_sps_pps(h264_data)
        if self._cached_sps_pps:
            print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes"
                  % len(self._cached_sps_pps), flush=True)
    return h264_data


class _Muxer:
    def __init__(self):
        self.calls = []

    def write_sps_pps(self, data):
        self.calls.append(bytes(data))


def _mk(codec, cached, with_muxer):
    rd = NVENCEncoder.__new__(NVENCEncoder)
    rd._codec = codec
    rd._cached_sps_pps = cached
    rd._sps_pps_injected = False
    rd._muxer_ref = _Muxer() if with_muxer else None
    return rd


def _state(rd):
    return (rd._cached_sps_pps, rd._sps_pps_injected,
            len(rd._muxer_ref.calls) if rd._muxer_ref else None)


def _run(fn, rd, data, is_idr):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(rd, data, is_idr)
    return bytes(out), _state(rd), len(buf.getvalue().splitlines())


def test_equivalence_over_truth_table():
    datas = {
        "empty": b"",
        "vcl_only": PB,
        "params_only": PARAMS,
        "params+vcl": PARAMS + PB,
        "idr_vcl": IDR,
    }
    cacheds = {"none": None, "params": PARAMS, "other": b"\x00\x00\x00\x01\x67XYZ"}
    rows = 0
    diffs = []
    for codec in ("h264", "hevc"):
        for cname, cached in cacheds.items():
            for dname, data in datas.items():
                for is_idr in (False, True):
                    for with_muxer in (False, True):
                        a = _mk(codec, cached, with_muxer)
                        b = _mk(codec, cached, with_muxer)
                        ra = _run(_old_ladder, a, data, is_idr)
                        rb = _run(NVENCEncoder._apply_sps_pps, b, data, is_idr)
                        rows += 1
                        if ra != rb:
                            diffs.append(
                                f"{codec}/{cname}/{dname}/idr={is_idr}/"
                                f"mux={with_muxer}:\n    old={ra}\n    new={rb}")
    assert not diffs, ("统一入口与改造前内联阶梯行为不一致：\n  "
                       + "\n  ".join(diffs))
    print(f"  [E1] 真值表等价 OK（{rows} 组合，含 h264/hevc × 缓存 3 态 × "
          f"数据 5 态 × is_idr 2 × muxer 2）")


def test_semantics_pinned():
    """把语义钉死成断言，防止后人"顺手改好"时无人察觉。

    注意：这些是**现状语义**，不是"正确语义"。是否要改（例如补 SPS/PPS 漂移
    检测、让非 IDR 也注入 muxer）属行为改动，须带 GPU 四组合回归。
    """
    # 1) IDR + 已缓存 + 原生缺参数集 → 预挂
    rd = _mk("h264", None, True)
    rd._cached_sps_pps = PARAMS
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, IDR, True)
    assert out.startswith(PARAMS), "IDR 应预挂缓存参数集"
    assert st[0] == PARAMS, "预挂路径不应改动缓存值"

    # 2) IDR + 未缓存 → 缓存 + 预注入 muxer
    rd = _mk("h264", None, True)
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, PARAMS + IDR, True)
    assert st[0] == PARAMS, f"应缓存提取到的参数集，实际 {st[0]!r}"
    assert st[1] is True, "IDR 首次缓存应预注入 muxer"
    assert st[2] == 1, "muxer 应被调用 1 次"
    assert out == PARAMS + IDR, "原生已含参数集时不应重复预挂"

    # 3) 非 IDR + 未缓存 → 只缓存，**不**注入 muxer
    rd = _mk("h264", None, True)
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, PARAMS + PB, False)
    assert st[0] == PARAMS
    assert st[1] is False and st[2] == 0, "非 IDR 不应注入 muxer"

    # 4) 空数据 → 原样返回、不缓存
    rd = _mk("h264", None, True)
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, b"", True)
    assert out == b"" and st == (None, False, 0)

    # 5) muxer 为 None → 不炸、不置 injected
    rd = _mk("h264", None, False)
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, PARAMS + IDR, True)
    assert st == (PARAMS, False, None)

    # 6) av1 → 提取恒为 None，永不缓存（与 _extract_sps_pps 的 av1 分支一致）
    rd = _mk("av1", None, True)
    out, st, _ = _run(NVENCEncoder._apply_sps_pps, rd, PARAMS + IDR, True)
    assert st == (None, False, 0), f"av1 不应缓存参数集，实际 {st}"
    print("  [E2] 语义钉死 OK（6 条）")


def _main():
    tests = [test_equivalence_over_truth_table, test_semantics_pinned]
    fails = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if not fails else 'FAILURES'}: "
          f"{len(tests) - fails}/{len(tests)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
