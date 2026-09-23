# -*- coding: utf-8 -*-
"""编码质量参数（CRF / CQ / QP）解析 —— 换算表的唯一消费层。

换算表本身在 ``src/utils/convert_crf.py``（移植自 VidUtils，全项目唯一真源）：

    value = a × x264_crf + b        （再夹到 [lo, hi]）
    src_value --to_x264_crf--> x264_crf --from_x264_crf--> dst_value

本模块只做 convert_crf 没覆盖的两件事：
  1. 把"某编码器用哪个 FFmpeg 参数"这件事集中成表（-crf / -cq:v / -qp + 配套参数）；
  2. 提供 resolve_quality()：把四类用户输入（字面量 crf/cq、基准 crf_ref/cq_ref）
     统一解析成"针对实际生效编码器的 (参数名, 值, 配套参数, 说明)"。

为什么必须有这一层
------------------
libx264 的 ``-crf`` 与 NVENC 的 ``-cq:v`` 都号称"0-51，越小越好"，但刻度并不
等价：等效关系约为 h264_nvenc = crf + 5、hevc_nvenc = crf + 7.5。此前本项目把
同一个 crf 数字原样下发给软编与硬编，导致同一条命令在不同编码器下画质漂移约
±7 个档位（详见 Plan/Video_Enhancement_crf_cq统一优化_对比分析报告.md）。

使用约定（与 VidUtils 一致）
----------------------------
* 字面量参数（``crf`` / ``cq``）：用户明确按某类编码器的刻度给的值，同族时
  原样下发；跨族（如给 GPU 的 cq 却落到 CPU 编码器）才做等效换算，并说明。
* 基准参数（``crf_ref`` / ``cq_ref``）：以统一轴给出质量，按表换算到任意目标
  编码器。与字面量参数互斥。
* ``crf_ref == 0`` / ``cq_ref == 0`` 表示"无损意图"，直接返回 0（不套线性
  映射），以复用各后端已有的无损分支（libx265 → lossless=1、
  nvenc → -qp 0、libx264 → -qp 0）。
"""

from typing import Dict, List, Optional, Tuple

# 换算表与基础换算函数：唯一来源，避免各处硬编码偏移量互相矛盾
from convert_crf import (                     # noqa: F401  (re-export)
    QUALITY_MAP,
    from_x264_crf,
    to_x264_crf,
    convert_quality,
)

# ─────────────────────────────────────────────────────────────────────────────
# 统一质量基准（libx264 CRF 轴）。改这一个常量即可整体调整画质基线。
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REF: int = 21

# ── CONSTQP 轴 ──────────────────────────────────────────────────────────────
# NVENC 的 targetQuality / QVBR 的 qvbrQuality（即 FFmpeg 的 -cq:v）与 CONSTQP 的
# QP（FFmpeg 的 -qp）是**两条刻度**：
#   · CQ/targetQuality 相对 x264 CRF 有 +5(h264) / +7.5(hevc) 的偏移 —— 见 QUALITY_MAP
#   · CONSTQP 直接下发量化步长，与 x264 的 QP 同量纲（x264 的 CRF 在中段近似
#     等于其平均 QP），故应落在**基准轴**上，不再叠加上述偏移
#
# FFmpeg 自身的选项说明也印证两者不可混用：
#   -cq  "Set target quality level (0 to 51, 0 means automatic) for constant
#         quality mode in VBR rate control"          ← 仅 VBR 有效
#   -qp  "Constant quantization parameter rate control method"  ← CONSTQP 专用
#
# 实测佐证：同数值下 constqp 产物体积小于 vbr_hq(CQ)
# （memory/constqp-fast-path.md），方向与"等效质量下 constqp QP 应低于 CQ 值"一致。
#
# 该偏移留作实测微调口：0 = 基准轴直取（当前模型）。
CONSTQP_QP_OFFSET: int = 0

# 各编码器的质量参数名（FFmpeg 私有选项名）
#   · librav1e  不认 -crf（实测会被静默忽略），必须用 -qp
#   · 硬件编码器（NVENC / QSV / AMF / VAAPI / VideoToolbox）用 -cq:v
#   · 其余用 -crf
_CQ_CODECS = {
    'h264_nvenc', 'hevc_nvenc', 'av1_nvenc',
    'h264_qsv', 'hevc_qsv', 'av1_qsv',
    'h264_amf', 'hevc_amf', 'av1_amf',
    'h264_vaapi', 'hevc_vaapi',
    'h264_videotoolbox', 'hevc_videotoolbox',
}

# 需要配套 -b:v 0 才是"纯恒定质量"的编码器：
#   VP8/VP9 的 -crf 不配 -b:v 0 会退化成 constrained quality（受码率上限约束）
_NEEDS_ZERO_BITRATE = {'libvpx', 'libvpx-vp9'}

# ── CQ 轴可调偏移 ────────────────────────────────────────────────────────────
# 各硬件编码器 -cq:v（targetQuality / CQ 轴）的微调量，单位与 CQ 同刻度。
#   · 默认**全 0** ⇒ resolve_quality 的输出严格等于 QUALITY_MAP（= 设计文档 §4.3）。
#   · QUALITY_MAP 是固定线性模型，存在内容相关误差（合成素材偏过配、真实素材偏
#     欠配约 2 dB），单一常量无法同时消除；需要时按内容在此微调该编码器。
#   · 仅作用于"换算 / 基准轴 / 默认基准"得到的值，**不影响**用户显式 --cq 字面量
#     （字面量语义是"原样下发"，见 resolve_quality 的同族分支）。
CQ_OFFSET: Dict[str, int] = {_c: 0 for _c in _CQ_CODECS}


def _apply_cq_offset(codec: str, value: float) -> float:
    """给派生出的 CQ 轴数值叠加该编码器的可调偏移（仅 -cq:v 类编码器）。"""
    c = str(codec).lower()
    if c in _CQ_CODECS:
        return float(value) + CQ_OFFSET.get(c, 0)
    return float(value)


def supports_cq(codec: str) -> bool:
    """编码器是否用 -cq:v（硬件编码器）而非 -crf。"""
    return str(codec).lower() in _CQ_CODECS


def supports_crf(codec: str) -> bool:
    """编码器是否用 -crf（软件编码器）。"""
    c = str(codec).lower()
    return c in QUALITY_MAP and c not in _CQ_CODECS and c != 'librav1e'


def literal_range(codec: str, kind: str = 'crf') -> Tuple[int, int]:
    """字面量质量参数在给定生效编码器下的**技术规范可用范围** ``(lo, hi)``。

    用于输入校验：超出该范围的值必须被拒绝，而不是静默截断/放行。

    刻度归属（与 ``resolve_quality`` 的字面量分支一致）：
      · ``kind='crf'`` —— 值按 libx264 CRF 轴解释。软编编码器直接使用其自身
        量程（如 libvpx-vp9 为 0~63）；硬件编码器不认 CRF，值会经换算，
        故以 libx264 的 0~51 为准。
      · ``kind='cq'``  —— 值按 h264_nvenc CQ 轴解释。硬件编码器直接使用其自身
        量程（如 h264_qsv 为 1~51）；软件编码器不认 CQ，值会经换算，
        故以 h264_nvenc 的 0~51 为准。
    """
    c = str(codec).lower()
    if kind == 'cq':
        key = c if supports_cq(c) else 'h264_nvenc'
    else:
        key = c if supports_crf(c) else 'libx264'
    m = QUALITY_MAP.get(key) or QUALITY_MAP['h264_nvenc' if kind == 'cq' else 'libx264']
    return int(m[2]), int(m[3])


def _quality_param(codec: str) -> Tuple[str, List[str]]:
    """返回 (质量参数名, 配套参数列表)。"""
    c = str(codec).lower()
    if c == 'librav1e':
        return '-qp', []
    if c in _CQ_CODECS:
        return '-cq:v', ['-b:v', '0']
    if c in _NEEDS_ZERO_BITRATE:
        return '-crf', ['-b:v', '0']
    return '-crf', []


def _clamp_int(value: int, codec: str) -> int:
    """把已换算好的值夹到该编码器的合法量程内；未知编码器返回原值。"""
    m = QUALITY_MAP.get(str(codec).lower())
    if m is None:
        return value
    lo, hi = m[2], m[3]
    return int(max(lo, min(hi, value)))


def _finish(codec: str, value: float, note: str) -> Tuple[str, int, List[str], str]:
    """收尾：取整 → clamp 到编码器量程 → 配参数名。

    注意：value 已是**目标编码器自身刻度**上的值（QUALITY_MAP 的 librav1e 行
    直接输出 qp，无需二次换算）。
    """
    c = str(codec).lower()
    param, extra = _quality_param(c)
    return param, _clamp_int(int(round(value)), c), extra, note


def to_constqp_qp(codec: str, value: int) -> int:
    """把 CQ/targetQuality 轴上的值换算为该编码器 CONSTQP 的 QP。

    码率控制模式在运行中从 vbr_hq/qvbr 切到 constqp 时（如 Real-ESRGAN 的
    [FIX-HIGHRES-RC] ≥2160p 自动切换），已算好的值是 targetQuality 刻度；若原样
    落到 CONSTQP 会被当作真实 QP 使用，画质偏松。换算路径：

        value --to_x264_crf--> 基准轴 --(+CONSTQP_QP_OFFSET)--> QP

    未知编码器原样返回（不做猜测）。
    """
    c = str(codec).lower()
    # value 位于 CQ 轴，可能已包含 CQ_OFFSET；先扣除再回溯基准轴，保证与
    # resolve_quality 的偏移语义一致（CQ_OFFSET 默认 0 时此处为恒等）。
    _v = float(value) - (CQ_OFFSET.get(c, 0) if c in _CQ_CODECS else 0)
    ref = to_x264_crf(c, _v)
    if ref is None:
        return int(value)
    return _clamp_int(int(round(ref)) + CONSTQP_QP_OFFSET, c)


def resolve_quality(codec: str, *,
                    crf: Optional[int] = None,
                    cq: Optional[int] = None,
                    crf_ref: Optional[int] = None,
                    cq_ref: Optional[int] = None,
                    default_ref: int = DEFAULT_REF,
                    ) -> Tuple[str, int, List[str], str]:
    """把四类质量输入统一解析成"针对 codec 的 (参数名, 值, 配套参数, 说明)"。

    判定顺序（与 VidUtils ``_resolve_quality_params`` 一致）：
      1. crf_ref  —— libx264 CRF 基准，按表换算
      2. cq_ref   —— h264_nvenc CQ 基准，先归一到基准轴再按表换算
      3. cq       —— 字面量，codec 支持 cq 时原样下发
      4. crf      —— 字面量，codec 支持 crf 时原样下发
      5. cq 但 codec 是软编 —— 按 h264_nvenc 量纲换算（并说明）
      6. crf 但 codec 是硬编 —— 按 libx264 量纲换算（并说明）
      7. 均未给   —— 用 default_ref 作为基准换算

    第 5/6 步是"编码器自动升级 / 降级"场景的保障：用户按某一族刻度给的字面量，
    在实际 codec 换族后被等效换算，而不是被当作同刻度数值静默下发。

    Args:
        codec: FFmpeg 编码器名（如 'libx264' / 'hevc_nvenc'）。必须是**实际生效**
               的编码器——即自动升级/降级之后的值，否则换算方向会错。
        crf / cq: 字面量质量值（None 表示未指定）
        crf_ref / cq_ref: 基准轴质量值（None 表示未指定）
        default_ref: 四类输入都为空时使用的 libx264 CRF 基准

    Returns:
        (参数名, 参数值, 配套参数列表, 人类可读说明)
    """
    c = str(codec).lower()

    # ── 1/2. 基准轴输入（0 视为无损意图，直接下发 0 走各后端无损分支）────────
    if crf_ref is not None:
        note = f'--crf-ref {crf_ref}（libx264 CRF 基准）'
        if int(crf_ref) == 0:
            return _finish(c, 0, note)
        value = from_x264_crf(c, crf_ref)
    elif cq_ref is not None:
        note = f'--cq-ref {cq_ref}（h264_nvenc CQ 基准）'
        if int(cq_ref) == 0:
            return _finish(c, 0, note)
        value = from_x264_crf(c, to_x264_crf('h264_nvenc', cq_ref))
    else:
        # ── 3/4. 字面量同族：原样下发 ────────────────────────────────────────
        if cq is not None and supports_cq(c):
            return _finish(c, cq, f'--cq {cq}（原样下发）')
        if crf is not None and supports_crf(c):
            return _finish(c, crf, f'--crf {crf}（原样下发）')

        # ── 5/6. 字面量跨族：按原量纲等效换算 ────────────────────────────────
        if cq is not None:
            value = convert_quality('h264_nvenc', cq, c)
            note = (f'编码器 {c} 不支持 -cq，已将 --cq {cq}'
                    f'（h264_nvenc 量纲）映射为等效值')
        elif crf is not None:
            value = convert_quality('libx264', crf, c)
            note = (f'编码器 {c} 不使用 -crf 刻度，已将 --crf {crf}'
                    f'（libx264 量纲）映射为等效值')
        # ── 7. 均未给：用全局基准 ────────────────────────────────────────────
        else:
            value = from_x264_crf(c, default_ref)
            note = f'默认基准 CRF {default_ref}'

    if value is None:       # 未知编码器：不做猜测，原样下发
        fallback = cq if cq is not None else (crf if crf is not None else default_ref)
        return _finish(c, fallback, f'编码器 {c} 无等效表，原样下发')

    # 派生值（基准轴 / 跨族 / 默认）落到 CQ 轴编码器时叠加可调偏移；
    # 字面量同族分支已在上面提前 return，故不受偏移影响。默认全 0 ⇒ 无变化。
    value = _apply_cq_offset(c, value)

    return _finish(c, value, note)


if __name__ == '__main__':
    # 自测：与报告 §4.3 的默认值映射表对齐
    print(f'基准 libx264 CRF {DEFAULT_REF} 的等效值：\n')
    print(f"{'编码器':<14}{'参数':<8}{'值':>5}   配套参数")
    print('-' * 40)
    for _codec in ('libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc',
                   'av1_nvenc', 'libsvtav1', 'libvpx-vp9', 'librav1e'):
        _p, _v, _e, _n = resolve_quality(_codec)
        print(f'{_codec:<14}{_p:<8}{_v:>5}   {" ".join(_e)}')
