#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVENC `rateControlMode` 取值真值探针 —— 用**驱动 runtime** 裁决 `rc_ptr[1]` 写入值。

背景（2026-10-05 静态审计，**结论待GPU 核实，代码暂不动**）
--------------------------------------------------------
生产代码 `external/ifrnet_video/nvenc_sdk.py:_build_encoder_config` 写入：

| 内部 rate_mode | rc_ptr[1] 写入值 | 状态 |
|---|---|---|
| `constqp`       | 0  | 恒定 QP，各来源一致 ✅ |
| `vbr_hq`        | 32 | **待核实** ⚠ |
| `qvbr`          | 64 | **待核实** ⚠ |

⚠ **本探针的存在理由是「静态查表不可靠」**。初版曾据
`FFmpeg/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h`（含`NVENCAPI_MAJOR_VERSION 13`）
断言「32/64 是非法枚举」，**该断言已撤回** —— 判定该头文件为**按 FFmpeg 需求裁剪的子集**：

| 检验项 | 结果 | 含义 |
|---|---|---|
| `grep -cE 'VBR_HQ|QVBR'` | **0** | 完整 SDK 头文件必然含这两个模式 |
| RC 枚举项数 | 恰为 3（CONSTQP/VBR/CBR） | 与 FFmpeg `nvenc.c` 实际用到的完全相同 |
| `NV_ENC_PIC_PARAMS_V2` / `NvEncGetEncodeVersion` | 0 命中 | 完整 SDK 应有 |

⇒ **「我查到的文件里没有 32/64」≠「枚举里不存在 32/64」**。
且仓库内两处记录互相矛盾（memory `nvenc_ctypes_verified_layouts.md:78` 写 32/64，
而 `nvenc_vbr_hq_offsets_probe.py:17,532` 注释写 4/32）⇒ 缺乏可靠单一真源。

✅ **已确立的 runtime 事实（T4 实机 GPU 直通 ctypes 实测，权重最高）**：
`rc_ptr[1]=32` 被驱动接受，且 vbr_hq/constqp/qvbr **三者输出互异**（非静默钳制）。
⚠ 这只证明「32 与 0/64 行为不同」，**不证明「32 的语义 == VBR_HQ」**。

本探针用**官方 caps API**直接问驱动：`NvEncGetEncodeCaps`(
`NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES`)，官方文档明确其返回值是
`NV_ENC_PARAMS_RC_MODE` 值的**位掩码**。⇒ 若返回值含 `bit5`/`bit6`，
则 32/64 是该驱动**声明支持**的模式，生产代码正确。

用法（T4 / L40，Linux + NVIDIA GPU）
------------------------------------
    # ① 只问 caps（最快，~30 秒出结论）
    python3 Accessory/probe/nvenc_rc_enum_truth.py --caps-only --verbose < /dev/null

    # ② caps + 逐值实编验证
    python3 Accessory/probe/nvenc_rc_enum_truth.py --try-values 0,1,2,32,64 < /dev/null

退出码：0 = 未发现非法写入 / 3 = 发现非法写入（**需评估修生产代码**）/ 2 = 环境不成立。

判读表（**以 runtime 事实为准**）
--------------------------------
| caps 观测 | 结论 | 动作 |
|---|---|---|
| 含 `bit5(32)` | 驱动**声明支持** 32 | **初版断言推翻**；核对是否即 VBR_HQ |
| 含 `bit6(64)` | 驱动声明支持 64 | 同上（可能为 QVBR 或厂商扩展） |
| 只含 `bit0/1/2` | 仅支持 CONSTQP/VBR/CBR | 32/64 为未定义值 ⇒ 评估改`RC_VBR(1)+targetQuality` |
| caps 查询失败 | 该 API 不可用 | 退回 `--try-values` 实编 + 读驱动实际行为 |
| `--try-values 32` 构造失败 | 驱动拒绝该值 | 必修（当前生产在该驱动上会崩） |
| `--try-values 32` 成功 | 驱动**容忍**该值 | 结合 caps 位判定语义是否正确 |

⚠ **本探针只回答「驱动是否声明/接受该值」**，不回答「该值的画质语义是否等于
VBR_HQ」—— 后者需另跑质量 A/B（VMAF/PSNR + 码率），见 Plan §0.2。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── 已知枚举（⚠ 来自 FFmpeg 裁剪版头文件，**仅作参考，不作判据**）─────────────
# 该头文件（nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h）只含 FFmpeg 用到的
# 3 个模式，**不能**据此断定 32/64 是否合法。真正的判据是驱动的 caps 位掩码。
RC_CONSTQP = 0x0
RC_VBR = 0x1
RC_CBR = 0x2
RC_LEGAL = {RC_CONSTQP: "CONSTQP", RC_VBR: "VBR", RC_CBR: "CBR"}

#: 生产代码当前写入值（external/*/nvenc_sdk.py:_build_encoder_config）。
PRODUCTION_WRITES = {"constqp": 0x0, "vbr_hq": 0x20, "qvbr": 0x40}

# ── [FIX-B3-CAPS-INDEX] NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES 的真实序号 ──
# 权威依据（2026-10-05 实测解析 nvEncodeAPI.h 的 NV_ENC_CAPS 枚举块，共 60 项）：
#     index 0 = NV_ENC_CAPS_NUM_MAX_BFRAMES
#     index 1 = NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES← 本项
#     index 2 = NV_ENC_CAPS_SUPPORT_FIELD_ENCODING
# 即「紧跟 NUM_MAX_BFRAMES 之后的那一项」，序号 = 1。
# ⚠ **查错序号会得到「不支持」的错误位掩码，从而闭环「证实」静态推断**（假阳性）。
#   故此值**必须**由下方 _verify_caps_index_against_header() 在运行时自查，
#   而非依赖注释。旧版本此处曾手抄为 8（错误），且该函数定义了却从无调用点。
CAPS_SUPPORTED_RATECONTROL_MODES = 1
CAPS_NUM_MAX_BFRAMES = 0  # 紧邻其前一项，用作序号自检的锚


def _verify_caps_index_against_header() -> Optional[str]:
    """运行时自查 caps 序号：若本机能取到头文件，则解析并与常量比对。

    返回 `None` 表示「无法校验」（无网络/无本地头文件）——此时**必须在输出里显式
    标注序号未经独立校验**，不得让读者以为已核对。
    """
    import re
    urls = (
        "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/"
        "include/ffnvcodec/nvEncodeAPI.h",
    )
    for url in urls:
        try:
            p = subprocess.run(["curl", "-sL", "--max-time", "20", url],
                               capture_output=True, text=True, timeout=30)
            body = p.stdout or ""
            if "NV_ENC_CAPS_PARAM" not in body:
                continue
            m = re.search(r"typedef enum _NV_ENC_CAPS\s*\{(.*?)\}\s*NV_ENC_CAPS;",
                          body, re.S)
            if not m:
                continue
            names = []
            for line in m.group(1).splitlines():
                line = line.split("/**")[0].strip()
                mm = re.match(r"^(NV_ENC_CAPS_[A-Z0-9_]+)\s*,", line)
                if mm:
                    names.append(mm.group(1))
            idx = names.index("NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES")
            prev = names[idx - 1] if idx > 0 else "(无)"
            if idx != CAPS_SUPPORTED_RATECONTROL_MODES:
                return (f"❌ 序号不符：头文件解析得 {idx}，本探针用 "
                        f"{CAPS_SUPPORTED_RATECONTROL_MODES}"
                        f"（capsToQuery 查错字段会得到假『不支持』）")
            return (f"✅ 序号已核对：头文件 {len(names)} 项中，"
                    f"SUPPORTED_RATECONTROL_MODES={idx}，前一项={prev}")
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return None


def _bits(mask: int) -> List[str]:
    """把 caps 位掩码解成RC 名称列表。"""
    names = []
    for val, nm in sorted(RC_LEGAL.items()):
        if mask & (1 << val):
            names.append(f"{nm}(bit{val})")
    return names


def _unknown_bits(mask: int) -> List[int]:
    """位掩码里超出 0/1/2 的置位 —— 即生产代码写入的那些值。"""
    return [b for b in range(3, 32) if mask & (1 << b)]


def run(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """跑外部命令；stdin 固定 DEVNULL（后台进程组 + tty 下会被 SIGTTOU 整组停住）。"""
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)


def env_precheck() -> Tuple[bool, str]:
    """GPU / NVENC 库前置体检（失败则 exit 2）。"""
    if not any(Path("/dev").glob("nvidia*")):
        return False, "无 /dev/nvidia*（容器未挂载 GPU）"
    rc, out, _ = run(["nvidia-smi", "-L"], 30)
    if rc != 0:
        return False, f"nvidia-smi rc={rc}（无驱动或未挂载）"
    try:
        ctypes.CDLL("libnvidia-encode.so.1")
    except OSError as exc:
        return False, f"libnvidia-encode.so.1 不可加载：{exc}"
    return True, out.strip().splitlines()[0] if out.strip() else "OK"


def query_caps(verbose: bool = False) -> Tuple[Optional[int], str]:
    """问驱动支持哪些 RC 模式（`NvEncGetEncodeCaps`）。

    返回 `(bitmask, 说明)`；bitmask 为 None 表示查询不可用（调用方须退回实编 A/B）。
    """
    try:
        from external.ifrnet_video.nvenc_sdk import (
            NVENCEncoder, _NvEncCapsParam, _FUNC_IDX, _sdk13_ver,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"无法导入 nvenc_sdk：{type(exc).__name__}: {exc}"
    # [FIX-B3-CAPS-INDEX] 键名无前导下划线（`_FUNC_IDX` 的键风格）。
    # 旧代码写的是 "_GetEncodeCaps"，带前导下划线 ⇒ 恒不匹配 ⇒ caps 查询永远返回 None，
    # 却仍以 exit 0 结束（假 PASS）。
    if "GetEncodeCaps" not in _FUNC_IDX:
        return None, ("_FUNC_IDX 缺 GetEncodeCaps 条目（应为 index 7，"
                      "见 nvenc_sdk.py [FIX-B3-CAPS-INDEX] 的来源说明）")
    enc = None
    try:
        # 用最小尺寸建一个仅用于查询的编码器。
        enc = NVENCEncoder(width=320, height=240, fps=30, qp=23,
                           preset="p4", rate_mode="constqp", la_depth=0,
                           pipeline_depth=2, codec="h264")
        addr = enc._get_func(_FUNC_IDX["GetEncodeCaps"])
        if addr is None:
            return None, "GetEncodeCaps 符号不可用"
        # 官方签名(NV_ENC_CAPS_PARAM_VER=1)：
        #   NVENCSTATUS NvEncGetEncodeCaps(void *encoder, NV_ENC_CAPS_PARAM *p,
        #                                   uint32_t *valueToRead);
        fn = ctypes.CFUNCTYPE(
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(_NvEncCapsParam), ctypes.POINTER(ctypes.c_uint32),
        )(addr)
        param = _NvEncCapsParam()
        ctypes.memset(byref(param), 0, sizeof(param))
        param.version = _sdk13_ver(1)
        param.capsToQuery = CAPS_SUPPORTED_RATECONTROL_MODES
        val = ctypes.c_uint32(0)
        st = fn(enc._encoder, byref(param), byref(val))
        if st != 0:
            return None, f"NvEncGetEncodeCaps 返回 {st}（该 caps 不可用）"
        if verbose:
            print(f"  [caps] paramVersion={_sdk13_ver(1)} "
                  f"capsToQuery={CAPS_SUPPORTED_RATECONTROL_MODES} "
                  f"value=0b{val.value:b} ({val.value})")
        return int(val.value), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if enc is not None:
            try:
                enc.close()
            except Exception:  # noqa: BLE001
                pass


def try_rc_values(values: List[int], frames: int) -> Tuple[bool, List[str]]:
    """逐个值**真的写进 `rc_ptr[1]`** 实编，回报驱动是否接受 + 输出是否可解。

    [FIX-M4-FALSE-PASS] 旧实现按**名字**构造（`{0:"constqp",1:"vbr_hq",2:"qvbr"}.get(v,"constqp")`），
    从不写请求的数值 ⇒ `--try-values 32/64` 实际落`constqp` 写 0，
    而 `exit 3` 永不触发 ⇒ **exit 0「全部合法」是假 PASS**。
    本版改为：monkeypatch `_build_encoder_config` 的写值点，**注入并回读实际值**，
    回读与请求不符即判FAIL（宁可不给结论，也不给假结论）。
    """
    try:
        from external.ifrnet_video import nvenc_sdk as _sdk
    except Exception as exc:  # noqa: BLE001
        return False, [f"无法导入 nvenc_sdk：{exc}"]

    lines: List[str] = []
    all_ok = True
    orig_build = _sdk.NVENCEncoder._build_encoder_config

    for v in values:
        tag = RC_LEGAL.get(v, "非0/1/2")
        written: List[int] = []          # 记录本轮实际写入 rc_ptr[1] 的值

        # 用 constqp 分支进入（la=0 不会被清零逻辑干扰），再把 rc 覆写成请求值。
        def _patched(self, codec_guid, preset_guid, width, height, fps, qp):
            cfg = orig_build(self, codec_guid, preset_guid, width, height, fps, qp)
            # rc_ptr 基址 = preset_config + 8 + 40（见 nvenc_sdk 的 _build_encoder_config）
            rc_ptr = ctypes.cast(ctypes.byref(cfg, 8 + 40), ctypes.POINTER(ctypes.c_uint32))
            written.append(int(rc_ptr[1]))
            rc_ptr[1] = v                 # ← 真正写入请求的值
            return cfg

        enc = None
        _sdk.NVENCEncoder._build_encoder_config = _patched
        try:
            enc = _sdk.NVENCEncoder(width=320, height=240, fps=30, qp=23, preset="p4",
                                    rate_mode="constqp", la_depth=0,
                                    pipeline_depth=2, codec="h264")
            got = written[-1] if written else None
            if got is None:
                all_ok = False
                lines.append(f"  rc={v:3d} ({tag:6s}) → ❌ **未捕获写入值**（探针失效）")
            elif got != v:
                all_ok = False
                lines.append(f"  rc={v:3d} ({tag:6s}) → ❌ 写入回读不符："
                             f"实际写入 {got} ≠ 请求 {v}")
            else:
                lines.append(f"  rc={v:3d} ({tag:6s}) → ✅ 确认写入 rc_ptr[1]={v}"
                             f"，驱动接受（InitializeEncoder rc=0）")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            lines.append(f"  rc={v:3d} ({tag:6s}) → ❌ **失败** "
                         f"{type(exc).__name__}: {str(exc)[:120]}")
        finally:
            _sdk.NVENCEncoder._build_encoder_config = orig_build
            if enc is not None:
                try:
                    enc.close()
                except Exception:  # noqa: BLE001
                    pass
    _ = frames  # 实编测码率尚未实现；当前只验证「能否构造成功（驱动是否接受该值）」
    return all_ok, lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="NVENC rateControlMode 枚举真值探针（裁定 rc_ptr[1]=32/64 是否合法）")
    ap.add_argument("--caps-only", action="store_true",
                    help="**默认行为即只问 caps**（此flag 保留兼容，实际不再改变行为；"
                         "要跳过 caps 用 --no-caps）")
    ap.add_argument("--no-caps", dest="no_caps", action="store_true",
                    help="跳过 caps 查询（只用 --try-values 实编）")
    ap.add_argument("--try-values", dest="try_values", default="",
                    help="逗号分隔的 rc_ptr[1] 候选值，逐个实编验证（如 0,1,2,32,64）"
                         "（--try 是 Python 关键字，故用 --try-values）")
    ap.add_argument("--frames", type=int, default=120,
                    help="实编帧数（默认 120）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("  NVENC rateControlMode 枚举真值探针")
    print("=" * 74)
    ok, why = env_precheck()
    print(f"环境前置：{'✅' if ok else '❌'} — {why}")
    if not ok:
        print("⇒ 环境不成立（本容器无 GPU），exit 2。请在 T4 / L40 上运行。")
        return 2

    print("\n已知枚举（⚠ 来自 FFmpeg 裁剪版头文件，仅供参考，不作判据）：")
    for v, nm in sorted(RC_LEGAL.items()):
        print(f"  {nm:8s} = 0x{v:X} ({v})")
    print("  ⚠ 该表只含 FFmpeg 用到的 3 个模式，**不能**据此断定 32/64 是否合法。")

    print("\n生产代码写入值（external/ifrnet_video/nvenc_sdk.py）：")
    for rm, v in PRODUCTION_WRITES.items():
        print(f"  {rm:8s} → rc_ptr[1]={v:3d}（0x{v:X}）")

    # [FIX-B3-CAPS-INDEX] caps 序号自查：查错字段会得到「不支持」的假位掩码，
    # 从而闭环「证实」静态推断 ⇒ 必须让读者看到序号是否被独立核对过。
    chk = _verify_caps_index_against_header()
    print(f"\n[自检] capsToQuery 序号 = {CAPS_SUPPORTED_RATECONTROL_MODES}")
    if chk is None:
        print("  ⚠️ 未能取到头文件核对 ⇒ **序号未经独立校验**，")
        print("     解读 caps 结果时须人工确认（查错字段会得到假『不支持』）。")
    else:
        print(f"  {chk}")

    exit_code = 0
    mask = None
    # [FIX-B3-CAPS-POLARITY] 旧代码是 `if not args.caps_only:` —— 传 `--caps-only`
    # 反而**跳过**查询，且随后以 exit 0 静默结束（假 PASS）。现改为默认查询、
    # 显式 `--no-caps` 才跳过。
    if not args.no_caps:
        mask, note = query_caps(verbose=args.verbose)
        print(f"\n① 驱动 caps（NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES）：")
        if mask is None:
            print(f"  ⚠️ 查询不可用：{note}")
            print("  ⇒ 退回 --try 实编 A/B 判定（caps 不是必需项）")
        else:
            names = _bits(mask)
            extra = _unknown_bits(mask)
            print(f"  bitmask = 0b{mask:08b} ({mask})")
            print(f"  合法位：{', '.join(names) if names else '（无）'}")
            if extra:
                print(f"  ✅ 驱动**声明支持**超出 0/1/2 的值：{extra}")
                for b in extra:
                    for rm, v in PRODUCTION_WRITES.items():
                        if v == b:
                            print(f"     · 含生产写入的 bit{b} = {v}（rate_mode={rm}）"
                                  f" ⇒ 该驱动**认可**此值，非未定义行为")
                print("  ⇒ 静态表说『没有』不足以判非法；**以本行 runtime 观测为准**。")
            else:
                print("  ℹ️  驱动仅声明 0/1/2，未声明 32/64。")
                print("  ⚠ 「未声明」≠「非法」：可能是厂商扩展未在caps 中暴露，")
                print("    也可能是真未定义 ⇒ 结合 --try-values 实编与质量 A/B 定夺。")

    if args.try_values:
        vals = [int(x) for x in args.try_values.split(",") if x.strip()]
        print(f"\n② 实编验证 {vals}：")
        all_ok, lines = try_rc_values(vals, args.frames)
        for ln in lines:
            print(ln)
        if not all_ok:
            exit_code = 3

    # ── 结论：只陈述 runtime 观测，不对「合法/非法」下静态判决 ──
    if mask is not None and _unknown_bits(mask):
        print("\n" + "=" * 74)
        print("  ✅ 观测：驱动 caps 声明支持生产代码写入的值")
        print("=" * 74)
        for b in _unknown_bits(mask):
            for rm, v in PRODUCTION_WRITES.items():
                if v == b:
                    print(f"  · rate_mode={rm!r} → rc_ptr[1]={v}（bit{b}）**在 caps 内**")
        print()
        print("  ⇒ 初版「32/64 是非法枚举」的断言**被本行观测推翻**（该断言基于"
              "FFmpeg 裁剪版头文件，见 docstring）。")
        print("  ⇒ 生产代码无需因「枚举非法」而改动。")
        print()
        print("  ⚠ 本探针**只回答「驱动是否声明/接受该值」**，不回答")
        print("    「该值的画质语义是否等于 VBR_HQ」。若需确认语义，")
        print("    另跑质量 A/B（VMAF/PSNR + 码率对比，见 Plan §0.2.7）。")
        return 0

    if args.try_values and exit_code == 3:
        print("\n" + "=" * 74)
        print("  ⚠️ 观测：部分候选值实编失败（驱动拒绝）")
        print("=" * 74)
        print("  ⇒ 该驱动上生产代码会崩，需评估修生产代码（见 Plan §0.2.7）。")
        return 3

    print("\n" + "=" * 74)
    print("  ℹ️ 观测：caps 未声明 32/64，且未做实编验证或实编通过")
    print("=" * 74)
    print("  · 「caps 未声明」**不等于**「枚举非法」—— 见 §0.2.3 裁剪证据。")
    print("  · 已知 runtime 事实：T4 实测 32 被接受且三模式输出互异。")
    print("  · 若要进一步定论，请跑：--try-values 0,1,2,32,64 --verbose")
    print("  · 本探针**不产出『生产代码有 bug』的结论**，只报告驱动观测。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
