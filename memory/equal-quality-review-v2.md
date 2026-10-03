---
name: 等质量换算立项 v2 评审结论（双仓文档已修正，代码待办 1 项）
description: 2026-09-30 评审《等质量换算立项》并落地 v2 双仓修正；含三条推翻初稿的结论（AC7 阈值实为 1.5 dB、缺陷可推广到全部编码器、rav1e speed 跨仓口径不一致）与 1 条未关闭的代码项
type: project
---

2026-09-30 对 `Plan/PROMPT_等质量换算立项.md` 做**需求级评审**，产出 v2 修正并
同步落进 `VidUtils/Plan/PROMPT_等质量换算立项.md`（两仓文档均已带「修订记录」表）。

**Why:** 初稿有三处结论经实测被推翻，若不修，后续标定会照着错的口径开工。

**三条推翻初稿的结论（均已写入两仓文档）：**

1. **AC7 阈值不是 −3.0 dB，而是 `TOL_PSNR_DB = 1.5`**（`Accessory/probe/av1_vp9_quality_matrix.py:70`、
   `Accessory/verify/crf_cq_unification_verify.py:193`）。用 1.5 复核初稿的 ΔPSNR 表，PASS/FAIL 列
   **完全吻合**，反证 1.5 才对。
2. **「等体积 ≠ 等质量」不是 rav1e 特例**。同素材同方法实测 `libsvtav1`（既有表值 `(2.145,−21.35)`，
   video-only 码率比）ΔPSNR = **+1.26 / −0.03 / −1.59 / −3.65 / −5.34 dB**（crf 18/21/24/27/30），
   与 rav1e 同形态（只准在默认工作点）⇒ 根因是**线性等体积拟合模型**本身，
   修复范围是 `QUALITY_MAP` **全部行**，不止 rav1e。
3. **跨仓 rav1e speed 口径不一致**：两仓共用 `QUALITY_MAP['librav1e']=(7.0032,−80.993)`（该行注释写明
   是 **native 档**标定），但 **VU 固定下发 `-speed 10`**（`vidcrop_cpu_v2.py`）、**VE 默认 native**。
   speed 10 的等体积解是 qp 77 而非表值 qp 66 ⇒ 标定前必须统一 speed 口径。

**其他关键实测（写入文档）：**
* **PSNR-HVS 没有独立滤镜**，是 **libvmaf feature**（`feature=name=psnr_hvs`，本机 libvmaf 2.3.1）。
* **libvmaf feature 与独立滤镜数值不可比**（libvmaf 的 `psnr` 给分平面 `psnr_y/cb/cr`、`float_ssim` 是自带实现）
  ⇒ 定标口径必须与门禁口径同源。
* K3（裸 `psnr` vs 显式 `[0:v][1:v]psnr` 差 3 dB）**与环境相关**：A 机有复现记录
  （[[ffmpeg-metric-measurement-traps]]），本机（ffmpeg 8.0.1）全长比较逐位相同 ⇒
  **显式标签保持强制**，但每台机器开工前须就地复现确认。
* 本机 ffmpeg 实为 **8.0.1-+vmaf**（文档原写 7.1）；`new4_raw_4k.mp4` 缺失；
  rav1e 实测 **native 0.055× 实时 / speed10 0.177× 实时**（比文档乐观 3~5×）。

**未关闭项（需用户批准后才能动代码）：**
`Video_Enhancement/src/utils/quality_map.py:117` 的注释仍写 `AC7 **FAIL**（阈值 -3.0）`，
应为 **1.5**。**这是本次唯一未修的落地点**——文档已注明「须一并订正代码注释」。

**How to apply:** 承接本项目时以**两仓 v2 文档**为准（勿沿用初稿记忆）；
改任一侧的 `QUALITY_MAP_QUALITY` / `QUALITY_MAP_QUALITY_QP` 必须同步对侧并回跑 VU 的 ⑨ 组
（当前基线 **13/13**，新增独立断言应变 14/14——实施时须写死）。
