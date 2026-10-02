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
* ``crf_ref == 0`` / ``cq_ref == 0`` 表示"最高质量意图"，直接返回 0（不套线性
  映射），以复用各后端已有的 0 号分支。按编码器**数学无损与否不同**：

  ==========  ==========================================================
  编码器      ``0`` 的语义
  ==========  ==========================================================
  libx265     **数学无损**（后端走 ``lossless=1``）
  libx264     **数学无损**（``-qp 0``）
  NVENC 族    ⚠ **仅"最高质量档"，不是逐位无损** —— H.264/HEVC NVENC 按
             NVIDIA 规范**没有**无损模式。VidUtils `probe/probe_lossless_qp0.sh`
             实测：像素恒等装置下 constqp ``-qp 0`` **561/561 帧与源不同**
             （负向对照 ``-qp 18`` 亦全帧不同，证明装置有分辨力）。
             注意本仓 0 号档走的是 **CQ 轴**（``-cq:v 0 -b:v 0``），
             constqp 路径则由 ``to_constqp_qp()`` 另行给出 ``-qp 0``；
             两者都只是"最高质量档"。
  其余        按各编码器 0 号档的实际语义，不宣称无损。
  ==========  ==========================================================

  ⚠ 本容器 **无法实跑 NVENC**（无 ``/dev/nvidia*``、``libcuda.so.1`` 不可加载），
    上述 NVENC 结论引自 VidUtils 的实测记录，本仓未复现；复现需带 NVIDIA 卡的机器。
"""

import os
from typing import Dict, List, Optional, Tuple

# 换算表与基础换算函数：唯一来源，避免各处硬编码偏移量互相矛盾
from convert_crf import (                     # noqa: F401  (re-export)
    SIZE_MAP,
    QUALITY_MAP,
    get_quality_map,
    get_quality_mode,
    set_quality_mode,
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
#   · CQ/targetQuality 相对 x264 CRF 有 +5(h264) / +7.5(hevc) 的偏移 —— 见 SIZE_MAP
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
#   · 硬件编码器（NVENC / QSV / AMF / VideoToolbox）用 -cq:v
#   · 其余用 -crf
#   · VAAPI 族只有 `-qp`（无 -cq/-crf），不在本表内，见 _QP_ONLY_CODECS
_CQ_CODECS = {
    'h264_nvenc', 'hevc_nvenc', 'av1_nvenc',
    'h264_qsv', 'hevc_qsv', 'av1_qsv',
    'h264_amf', 'hevc_amf', 'av1_amf',
    'h264_videotoolbox', 'hevc_videotoolbox',
}

# 只认 `-qp` 的编码器（VAAPI 族）。
# 实测 `ffmpeg -h encoder=h264_vaapi`：`-qp (0 to 52)`，且**没有** -cq/-crf/-global_quality。
# 其 `-qp` 与 x264 的 QP 同尺度（≠ CQ/targetQuality 轴的 +5/+7.5 偏移），
# 故任意质量输入都先归一到基准轴再下发（与 VidUtils 的 V6 同源同判）。
# 此前 VAAPI 被误列在 _CQ_CODECS 里，会下发 `-cq:v N -b:v 0` ——
# 那是 h264_vaapi 不存在的选项，ffmpeg 直接报错（不是"静默忽略"）。
_QP_ONLY_CODECS = {'h264_vaapi', 'hevc_vaapi'}
_QP_ONLY_RANGE = (0, 52)    # VAAPI 的 -qp 量程（ffmpeg 实测 0 to 52）

# 需要配套 -b:v 0 才是"纯恒定质量"的编码器：
#   VP8/VP9 的 -crf 不配 -b:v 0 会退化成 constrained quality（受码率上限约束）
_NEEDS_ZERO_BITRATE = {'libvpx', 'libvpx-vp9'}

# ── librav1e 的 `-speed`（性能 / 画质取舍开关）───────────────────────────────
# 实测（2026-09-29，方案 §6.11）：2s / 1280×720 / 8 核
#   · 不传 `-speed`（rav1e 原生默认档）= 179.2 s  ≈ **0.011× 实时**（长视频不可用）
#   · `-speed 10`               =  39.0 s  ≈ 0.051× 实时（快 4.6×）
#   · `-tile-rows/-tile-columns 1` = **no-op**（输出字节与不分 tile 完全相同）⇒ 不下发
#
# ⚠ **`-speed` 不是免费的**（2026-09-30 在门禁素材 word_world_2 上实测）：
#   同码率比（≈0.91~0.99）下，`-speed 10` 比原生默认档**多掉约 1.9 dB**：
#       原生默认档  -qp 66  ratio 0.91   ΔPSNR **-1.21 dB**  → AC7 PASS
#       -speed 10   -qp 85  ratio 0.91   ΔPSNR **-3.12 dB**  → AC7 **FAIL**（阈值 -1.5）
#   即：**在 speed 10 下"等体积"与"等质量"两个判据无法同时满足** ——
#       要守住质量（ΔPSNR ≥ -1.5）必须用 -qp 55，代价是码率比 **1.303（体积 +30%）**。
#   ⇒ 视频增强管线以画质为产品，**默认保持 rav1e 原生档**（AC7 绿），
#     `-speed 10` 作为**显式可选项**暴露（性能换体积/画质，由调用方决策）。
#
# ⚠ SIZE_MAP['librav1e'] 的 a/b **只对已声明的 speed 档成立**（`-speed` 整体平移
#   码率曲线）。若启用 `-speed 10`，必须换成 speed 10 的标定值：
#     原生档 → (7.0032, -80.993)  crf21 ⇒ qp 66
#     speed10 → (6.8159, -66.093) crf21 ⇒ qp 77（等体积）／等质量需 qp ≈ 55
RAV1E_SPEED_DEFAULT = 0        # 0 = 不下发 -speed，用 rav1e 原生默认档
RAV1E_SPEED = int(os.environ.get('VIDEO_RAV1E_SPEED', '') or RAV1E_SPEED_DEFAULT)

#: ``-speed 10`` 下的**等体积**标定值（与 _RAV1E_EQVOL 原生档同一口径，只是曲线平移）。
#: 实测（2026-09-30，门禁素材 word_world_2，qp 扫 55~105 六点 + libx264 锚点 18~30）：
#:   等体积拟合 a=6.8159, b=−66.093 ⇒ crf 21 → qp 77（落点码率比 0.986）。
#: ⚠ 该点 ΔPSNR ≈ −2.7 dB，**超出 AC7 的 −1.5 dB 质量地板** —— 这是 `-speed 10`
#:   的固有代价（详见 RAV1E_SPEED 上方注释），不是标定误差。
_EQVOL_SPEED_OVERRIDE: Dict[str, tuple] = {
    'librav1e': (6.8159, -66.093, 0, 255),   # 仅当 RAV1E_SPEED > 0 时生效
}

#: ``-speed 10`` 下的**等质量**标定值（2026-10-01 落表）。
#: ``QUALITY_MAP['librav1e']`` 只对 **native 档**成立（``-speed`` 整体
#: 平移码率曲线），故等质量表的 rav1e 也按 speed 分档：原生档进 `QUALITY_MAP`，
#: speed 10 档进本表。由 `Accessory/probe/calibrate_equal_quality.py` 标定回填。
#: 值来源：统一锚点 **18/21/24/27/30**，11 个(素材,口径)样本池化（VU 7 素材 6s +
#: VE 4 素材 10s），720p prep，`n_subsample=1`，
#: LOO worst |ΔVMAF| = **7.15**（⚠ 超仓主裁定的 ≤5.9，rav1e 档训练内误差本身就最高，
#: 属编码器特性；详见 :data:`convert_crf.QUALITY_MAP` 的「门禁偏离」说明）。
#: 空 dict ⇒ 未标定，质量模式下 rav1e 回落 ``_active_table``（即等质量表的原生档）。
_EQQUAL_SPEED_OVERRIDE: Dict[str, tuple] = {
    'librav1e': (7.8373, -95.5520, 0, 255),   # 仅当 RAV1E_SPEED > 0 时生效
}

# ── CONSTQP / QP 轴的**等质量**表（D2b，仅本仓）──────────────────────────────
# 本仓走 ``external/*/nvenc_sdk.py`` 的 ctypes 直连 SDK，``to_constqp_qp()`` 需要
# **QP 轴**上的等质量值，而 ``QUALITY_MAP`` 是 CQ/CRF 轴的表。
#   · 软编（libx265/libsvtav1/librav1e）的 ``-qp`` 本身就落在 QP 刻度上 ⇒ 镜像 D2a；
#   · ⚠ NVENC 的 QP 行**需上机标定**（M4，需 NVIDIA 卡）—— 在标定前，
#     ``_QP_MAP_OVERRIDE`` 会先行命中（h264/hevc 基准轴直取、av1 ×3），行为与现状一致。
QUALITY_MAP_QP: Dict[str, tuple] = {
    # 软编行镜像 QUALITY_MAP 的 2026-10-02 标定值（统一锚点 18/21/24/27/30，
    # 11 个(素材,口径)样本，720p prep，n_subsample=1；LOO worst |ΔVMAF| 4.49~5.19
    # —— 精度边界见 convert_crf.QUALITY_MAP 注释）。
    # ⚠ libx265/libvpx-vp9/libaom-av1/libsvtav1 的 QP 轴 = CRF 轴（ffmpeg 直接透传 -qp）。
    # ⚠ TODO(M4): 'h264_nvenc' / 'hevc_nvenc' / 'av1_nvenc' 需 NVIDIA 机上标定。
    'libx265':     (1.0943, -2.4562, 0, 51),
    'libvpx-vp9':  (1.9736, -12.3252, 0, 63),
    'libaom-av1':  (2.2531, -20.5904, 0, 63),
    'libsvtav1':   (2.2391, -16.7060, 0, 63),
    # ⚠ rav1e 分档：native 进 QUALITY_MAP，speed10 进 _EQQUAL_SPEED_OVERRIDE，
    #   本表不重复登记 librav1e（避免与档位语义冲突）。
}


def _active_table(table=None):
    """当前生效的 ``{codec: (a, b, lo, hi)}``。

    ``table`` 非 None ⇒ **一次性覆盖**（无副作用，供 probe 的 ``--quality-mode`` 用）；
    否则取全局模式维护的活动表（``set_quality_mode()`` → ``convert_crf._ACTIVE_MAP``）。
    默认口径为 ``quality`` ⇒ 用 ``QUALITY_MAP``（等质量；未覆盖的编码器回退 ``SIZE_MAP``）。
    """
    return table if table is not None else get_quality_map()


def _active_cq_model(codec: str, table=None):
    """返回该编码器 **CQ/CRF 轴**上的 ``(a, b, lo, hi)``；未知编码器返回 None。

    按当前生效的 ``-speed`` 档选取 rav1e 标定值 —— ``-speed`` 会整体平移码率
    曲线，同一基准轴值在不同 speed 下对应不同的 rav1e qp；**等质量模式**走
    ``_EQQUAL_SPEED_OVERRIDE``，否则走 ``_EQVOL_SPEED_OVERRIDE``。
    其余编码器落到 :func:`_active_table`（默认口径即 ``QUALITY_MAP``，未覆盖回退 ``SIZE_MAP``）。

    注：2026-10-01 由 ``_eqvol_model`` 改名而来 —— 原名只提「等体积」，但本函数
    同时服务等体积/等质量两套表（按 :func:`get_quality_mode` 分流），旧名有误导性。
    **旧名已彻底移除**（无兼容别名）。
    """
    c = str(codec).lower()
    if c == 'librav1e' and RAV1E_SPEED > 0:
        ov = (_EQQUAL_SPEED_OVERRIDE if get_quality_mode() == 'quality'
              else _EQVOL_SPEED_OVERRIDE)
        if c in ov:
            return ov[c]
    return _active_table(table).get(c)

# ── CQ 轴可调偏移 ────────────────────────────────────────────────────────────
# 各硬件编码器 -cq:v（targetQuality / CQ 轴）的微调量，单位与 CQ 同刻度。
#   · 默认**全 0** ⇒ resolve_quality 的输出严格等于 SIZE_MAP（= 设计文档 §4.3）。
#   · SIZE_MAP 是固定线性模型，存在内容相关误差（合成素材偏过配、真实素材偏
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
    """编码器是否用 -crf（软件编码器）。

    排除 librav1e（只有 -qp）与 VAAPI 族（只有 -qp）——它们虽在 SIZE_MAP 里，
    但都不是 -crf 编码器，不能算作"字面量 crf 原生可用"。
    """
    c = str(codec).lower()
    return (c in SIZE_MAP and c not in _CQ_CODECS
            and c not in _QP_ONLY_CODECS and c != 'librav1e')


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
    m = SIZE_MAP.get(key) or SIZE_MAP['h264_nvenc' if kind == 'cq' else 'libx264']
    return int(m[2]), int(m[3])


def _quality_param(codec: str) -> Tuple[str, List[str]]:
    """返回 (质量参数名, 配套参数列表)。"""
    c = str(codec).lower()
    # librav1e：只有 -qp（0~255）。`-speed` 是**显式可选**的（RAV1E_SPEED>0 才下发）：
    # 实测它能提速 4.6×，但同码率下多掉 ~1.9 dB，故默认不启用（见 RAV1E_SPEED_DEFAULT）。
    # tile 实测为 no-op，任何情况下都不下发。
    if c == 'librav1e':
        return '-qp', (['-speed', str(RAV1E_SPEED)] if RAV1E_SPEED > 0 else [])
    if c in _QP_ONLY_CODECS:
        return '-qp', []
    if c in _CQ_CODECS:
        return '-cq:v', ['-b:v', '0']
    if c in _NEEDS_ZERO_BITRATE:
        return '-crf', ['-b:v', '0']
    return '-crf', []


#: 0 号档里**数学无损**的编码器（其余的 0 只是"最高质量档"）
_MATH_LOSSLESS_ZERO = {'libx264', 'libx265'}


def _zero_note(codec: str) -> str:
    """``0`` 号档的语义提示 —— 避免把"最高质量"误读成"逐位无损"。

    只有 libx264 / libx265 的 0 是数学无损；NVENC 族按 NVIDIA 规范没有无损模式
    （VidUtils ``probe/probe_lossless_qp0.sh`` 实测 561/561 帧与源不同）。
    """
    if str(codec).lower() in _MATH_LOSSLESS_ZERO:
        return '，0 = 数学无损档'
    return '，0 = 最高质量档（非逐位无损）'


def _clamp_int(value: int, codec: str) -> int:
    m = SIZE_MAP.get(str(codec).lower())
    if m is None:
        return value
    lo, hi = m[2], m[3]
    return int(max(lo, min(hi, value)))


def _finish(codec: str, value: float, note: str) -> Tuple[str, int, List[str], str]:
    """收尾：取整 → clamp 到编码器量程 → 配参数名。

    注意：value 已是**目标编码器自身刻度**上的值（SIZE_MAP 的 librav1e 行
    直接输出 qp，无需二次换算）。
    """
    c = str(codec).lower()
    param, extra = _quality_param(c)
    return param, _clamp_int(int(round(value)), c), extra, note


# ── CONSTQP 的 QP 轴模型 ─────────────────────────────────────────────────────
#   QP = a_qp × 基准轴 + b_qp     （再夹到 (lo, hi)）
# 为什么不能直接复用 SIZE_MAP：那张表描述的是 **CQ/targetQuality 轴**（-cq:v），
# 而 CONSTQP 的 `-qp` 是"真实量化步长"轴，两者刻度不同：
#   · NVENC H.264/HEVC 的 -cq:v 相对基准轴有 +5 / +7.5 偏移，而 -qp 没有
#     ⇒ 截距清零（26→21、28→20）；
#   · **AV1 的 -qp 是 qindex（0~255）**，与 -cq 的 0~63 完全是两条刻度
#     ⇒ 倍率 4（21 → 84）。原实现拿 CQ 轴值直发，21 落在 0~255 上等于近无损；
#   · librav1e / libsvtav1 / libx265 的"质量参数"本身就落在 QP 刻度上，
#     直接沿用其 SIZE_MAP 行（含各自截距，如 rav1e 的 4·ref−4、svtav1 的 ref+6）。
# ⚠ av1_nvenc 的 QP 尺度在 L40 上实测确认为 3×（非推断的 4×）。
#   VidUtils/probe/verify_nvenc_quality_gpu.py 的 C 组扫 -qp {21,63,84,105}。
#   T4 无 AV1 NVENC，故 T4 生产与 h264/hevc 路径均为恒等，不受影响。
_QP_MAP_OVERRIDE = {
    'h264_nvenc': (1.0, 0.0, 0, 51),
    'hevc_nvenc': (1.0, 0.0, 0, 51),
    'av1_nvenc':  (3.0, 0.0, 0, 255),   # [L40 实测确认：QP 尺度 3×]
    'h264_vaapi': (1.0, 0.0, 0, 52),
    'hevc_vaapi': (1.0, 0.0, 0, 52),
}


def _qp_model(codec: str, table=None):
    """返回该编码器 **CONSTQP / QP 轴**上的 ``(a, b, lo, hi)``；未知编码器返回 None。

    优先级：硬编/VAAPI 的显式覆盖（``_QP_MAP_OVERRIDE``）→ 等质量模式下的
    ``QUALITY_MAP_QP``（D2b，仅本仓）→ :func:`_active_table`。
    ⚠ NVENC 的等质量 QP 行待 **M4**（需 NVIDIA 机）标定；标定前由 ``_QP_MAP_OVERRIDE``
    先行命中 ⇒ 行为与现状逐字一致。
    """
    c = str(codec).lower()
    if c in _QP_MAP_OVERRIDE:
        return _QP_MAP_OVERRIDE[c]
    if get_quality_mode() == 'quality' and c in QUALITY_MAP_QP:
        return QUALITY_MAP_QP[c]
    return _active_table(table).get(c)


def to_constqp_qp(codec: str, value: int, *, table=None) -> int:
    """把 CQ/targetQuality 轴上的值换算为该编码器 CONSTQP 的 QP。

    码率控制模式在运行中从 vbr_hq/qvbr 切到 constqp 时（如 Real-ESRGAN 的
    [FIX-HIGHRES-RC] ≥2160p 自动切换），已算好的值是 targetQuality 刻度；若原样
    落到 CONSTQP 会被当作真实 QP 使用，画质偏松。换算路径：

        value --to_x264_crf--> 基准轴 --(×a_qp +b_qp +CONSTQP_QP_OFFSET)--> QP

    ``a_qp``/``b_qp`` 见 ``_QP_MAP_OVERRIDE``（H.264/HEVC = 基准轴直取；
    AV1 = ×4 的 qindex 尺度；其余沿用 SIZE_MAP 自身的 QP 刻度）。

    未知编码器原样返回（不做猜测）。
    """
    c = str(codec).lower()
    # value 位于 CQ 轴，可能已包含 CQ_OFFSET；先扣除再回溯基准轴，保证与
    # resolve_quality 的偏移语义一致（CQ_OFFSET 默认 0 时此处为恒等）。
    _v = float(value) - (CQ_OFFSET.get(c, 0) if c in _CQ_CODECS else 0)
    ref = to_x264_crf(c, _v, table=table)
    model = _qp_model(c, table=table) if ref is not None else None
    if model is None:
        return int(value)
    a, b, lo, hi = model
    return int(max(lo, min(hi, round(a * ref + b + CONSTQP_QP_OFFSET))))


def _to_target_from_ref(codec: str, ref, table=None):
    """把**基准轴（libx264 CRF）**值换算到目标编码器刻度。

    与 :func:`convert_crf.from_x264_crf` 的唯一差别：对 ``librav1e`` 且
    ``RAV1E_SPEED > 0`` 时改用对应 speed 档的标定值（``-speed`` 会整体平移码率曲线，
    同一基准轴值在不同 speed 档下对应不同的 rav1e qp）。
    其余编码器 ``_active_cq_model`` 直接落到 :func:`_active_table`（默认模式 = ``SIZE_MAP``，
    此时行为与原函数逐字一致）。
    """
    m = _active_cq_model(codec, table=table)
    if m is None or ref is None:
        return None
    a, b, lo, hi = m
    return max(lo, min(hi, a * float(ref) + b))


def resolve_quality(codec: str, *,
                    crf: Optional[int] = None,
                    cq: Optional[int] = None,
                    crf_ref: Optional[int] = None,
                    cq_ref: Optional[int] = None,
                    default_ref: int = DEFAULT_REF,
                    table: Optional[Dict[str, tuple]] = None,
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
        table: 可选的换算表**一次性覆盖**（``{codec: (a, b, lo, hi)}``）。
               None ⇒ 用当前活动表（``set_quality_mode()`` 维护；默认 ``size``）。
               仅对**派生分支**（基准轴 / 跨族 / 默认）生效；字面量同族仍原样下发。

    Returns:
        (参数名, 参数值, 配套参数列表, 人类可读说明)
    """
    c = str(codec).lower()

    # ── 0. VAAPI 族：只认 `-qp`，且 -qp 与 x264 QP 同尺度（不适用 CQ 轴偏移）──
    # 任意质量输入先归一到**基准轴**，再夹到 VAAPI 量程（ffmpeg: -qp 0 to 52）。
    # 必须早于下面的所有分支：VAAPI 不在 _CQ_CODECS 里，若走通用路径会拿
    # from_x264_crf() 的 CQ 轴值（基准 +3/+5）当 QP 下发，偏松。
    if c in _QP_ONLY_CODECS:
        if crf_ref is not None:
            ref, note = float(crf_ref), f'--crf-ref {crf_ref}（libx264 CRF 基准）'
        elif cq_ref is not None:
            ref = float(to_x264_crf('h264_nvenc', cq_ref, table=table))
            note = f'--cq-ref {cq_ref}（h264_nvenc CQ 基准）'
        elif cq is not None:
            ref = float(to_x264_crf('h264_nvenc', cq, table=table))
            note = f'--cq {cq}（h264_nvenc CQ 刻度）'
        elif crf is not None:
            ref, note = float(crf), f'--crf {crf}（libx264 CRF 刻度）'
        else:
            ref, note = float(default_ref), f'默认基准 CRF {default_ref}'
        note += '；VAAPI 只有 -qp（与 x264 QP 同尺度）'
        if ref == 0:
            note += '，0 = 最高质量档（非逐位无损）'
        _lo, _hi = _QP_ONLY_RANGE
        return '-qp', int(max(_lo, min(_hi, round(ref)))), [], note

    # ── 1/2. 基准轴输入（0 = 最高质量意图，直发 0 走各后端 0 号分支）──────
    #    ⚠ 只有 libx265 / libx264 的 0 是**数学无损**；NVENC 的 0 仅"最高质量档"
    #    （见模块 docstring 的分编码器说明），故对 NVENC 显式加注避免误读。
    if crf_ref is not None:
        note = f'--crf-ref {crf_ref}（libx264 CRF 基准）'
        if int(crf_ref) == 0:
            return _finish(c, 0, note + _zero_note(c))
        value = _to_target_from_ref(c, crf_ref, table=table)
    elif cq_ref is not None:
        note = f'--cq-ref {cq_ref}（h264_nvenc CQ 基准）'
        if int(cq_ref) == 0:
            return _finish(c, 0, note + _zero_note(c))
        value = _to_target_from_ref(c, to_x264_crf('h264_nvenc', cq_ref, table=table),
                                    table=table)
    else:
        # ── 3/4. 字面量同族：原样下发 ────────────────────────────────────────
        if cq is not None and supports_cq(c):
            return _finish(c, cq, f'--cq {cq}（原样下发）')
        if crf is not None and supports_crf(c):
            return _finish(c, crf, f'--crf {crf}（原样下发）')

        # ── 5/6. 字面量跨族：按原量纲等效换算 ────────────────────────────────
        if cq is not None:
            value = _to_target_from_ref(c, to_x264_crf('h264_nvenc', cq, table=table),
                                        table=table)
            note = (f'编码器 {c} 不支持 -cq，已将 --cq {cq}'
                    f'（h264_nvenc 量纲）映射为等效值')
        elif crf is not None:
            value = _to_target_from_ref(c, crf, table=table)
            note = (f'编码器 {c} 不使用 -crf 刻度，已将 --crf {crf}'
                    f'（libx264 量纲）映射为等效值')
        # ── 7. 均未给：用全局基准 ────────────────────────────────────────────
        else:
            value = _to_target_from_ref(c, default_ref, table=table)
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
