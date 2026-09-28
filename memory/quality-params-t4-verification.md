---
name: 质量控制参数方案 T4 实测收口（E9/E10 + 需 L40 的 AV1 清单）
description: 2026-09-28 VE 质量控制参数修复方案在 T4 上的 Gate 0/1/3/4 实测结果、E9/E10 落地、E5 枚举措辞纠正，以及只有 Ada 才能定案的 AV1 待检清单
type: project
---

2026-09-28 在 Tesla T4 上跑完 VE 质量控制参数方案的「T4 可做部分」。
权威记录在 `Plan/Video_Enhancement_质量控制参数修复方案.md` 的 **§6**（实测收口）与 **§7**（需 L40 的 AV1 清单）。

**实测结果**

- Gate 0：`crf_cq_unification_verify.py --quick` = PASS 91 / FAIL 0 / SKIP 11；
  `plan_implementation_gate.py` = 96 项 / 94 通过 / 0 失败 / 2 跳过（记忆基线 95/93/0/2）；
  `pytest Accessory/test` = 24 passed（修 chroma 测试导入前为 2 failed）。
- Gate 1（`--gpu` + 真实素材 `input_videos/word_world_2.mp4`，码率素材以 `new4_raw.mp4` 替代已缺失的 `new5_raw.mp4`）：
  **PASS 99 / FAIL 0 / WARN 3 / SKIP 1**。
  **G7-3 constqp `-qp 21` ΔPSNR +0.06 dB / 1.46× ⇒ `CONSTQP_QP_OFFSET` 保持 0 即为最优**（E9 的"校准"据此结案）。
  G7-1/G7-2 的 WARN 与 2026-09-11 基线逐位相同（−1.93 / −2.69），是既有内容相关偏松。
- Gate 3：`-preset medium` 与 `-preset p4` **逐字节相同**（E5 锚点成立）；
  `medium` 由 p5→p4 的可感知影响很小（中位 4.81s vs 4.86s，同 `-cq` 下体积 +0.46%）⇒ **不回滚 E5**。
- Gate 4：hevc + LA=8（`--codec-ifrnet/-esrgan hevc_nvenc --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8`）
  单分段帧守恒 ✅（frames==packets==253 = 2×127−1，段首无连 IDR、frame_num 回退 0、色度坏帧簇 0）。
- E10 新增 `G7-8` 双跑实测：合成 ΔPSNR **+5.33 dB** / 1.53× vs 真实 **+0.06 dB** / 1.46× ⇒ "合成偏过配"的旧注解成立（已从注释升级为实测）。

**E9/E10 落地要点**

- 判据新增 `G7-6`（AV1 硬编）/ `G7-7`（软件侧）/ `G7-8`（E10 双跑）；可选覆盖项**先探可用性**再决定跑/SKIP。
- G7 编码阶段异常改为**逐项 FAIL**，不再让整组"执行中断"（原先一个 `Unknown encoder` 会把 G7-1..G7-8 全吞成 1 个组级 FAIL）。
- 软件侧编码器：`libsvtav1` 在本机 ffmpeg 不存在 ⇒ 回退 `libvpx-vp9`（详见 [[nvenc-preset-and-encoder-availability]]）。

**需 L40（或任意 Ada 卡）才能继续检测的 AV1 内容**（方案 §7 的 **AC1~AC6**）

核心是 **AC1：`src/utils/quality_map.py` 的 `_QP_MAP_OVERRIDE['av1_nvenc'] = (4.0, 0.0, 0, 255)`
（标有 `[待 L40 复核]`）的 ×4 倍率定案** —— 扫 `-qp {21, 84, 105}`，若 84 落在
`RATE_PASS=(0.65,1.50)` 且 ΔPSNR ≥ −1.5 dB 则摘掉标记；否则改 `a = 落带 qp / 21` 并同步改判据
`G3-7`（现钉 84）与 `G6-7` 的期望值。其余 AC2~AC6（G7-6 转正、Gate 2 B/C 组、
`av1_qsv`/`av1_amf` 量程）见方案 §7 表。
⚠ 例外：**AC3（`G6-7` 命令形状捕获）不需要 AV1 硬件，T4 上实测已 PASS**——它只证明
"函数把 27 换成了 84"，**期望值 84 是否正确仍由 AC1 决定**（原写"G6-7 需 Ada"不实，已纠正）。

**Why:** 用户明确要求把"需 L40 环境继续检测的内容"在方案里写清楚，避免把 T4 已过的结论误当成 AV1 也已定案；随后进一步要求把 §7 写成**可直接执行的测试内容**（命令 + 判据 + 关闭动作）。
**How to apply:** 任何涉及 `av1_nvenc` 的 constqp / `-qp` 结论一律按"**未定案**"对待；
生产 AV1 优先走 `-cq`/VBR 路径（E0 的 0~63 量程有实测支撑）。拿不到 Ada 时**不要摘** `[待 L40 复核]` 标记。

## 附：AC1 runbook 的度量口径坑（2026-09-28 实测发现）

> 通用版已抽到 [[ffmpeg-metric-measurement-traps]]（任何「编码产物 vs 参考」打分都适用）；
> 下面只留与 AC1 直接相关的部分。

AC1 的手工复现脚本必须与判据脚本 `Ctx._metric` **完全同口径**：`-v info` +
**显式 `-lavfi "[0:v][1:v]psnr"`** + 解析 stderr 的 `average:`。

- ⚠ **裸 `-lavfi psnr`（不带 `[0:v][1:v]`）会走出不同结果**：实测同一对文件
  裸形式给 43.40 dB、显式形式给 46.58 dB（差 3 dB，足以让 ±1.5 dB 的判据失效）。
- ⚠ `-v error` 会把 `psnr` 滤镜的汇总行（INFO 级）压掉 ⇒ ΔPSNR 恒为 0.00 的假象。
- T4 上以 `h264_nvenc` 替代 `av1_nvenc` 端到端预验证：`soft=46.580407`、
  `qp21 ratio=1.46x / dPSNR=+0.06 dB` —— 与任务报告里 G7-3 的 1.46× / +0.06 dB **逐位一致**，
  证明 AC1 的度量管线正确、L40 上唯一未知量就是 AV1 的 ×4 倍率本身。
- §7 的全部 bash 块已过 `bash -n`；文档里凡从不同口径摘来的数字都必须用同口径重跑替换
  （本例 `qp26` 的 ΔPSNR 由误写的 −1.94 更正为 **−3.07**）。

**本轮附带修复与范围外移交**

- A7：`Accessory/test/test_chroma_false_positive.py` 的 `_load_chroma_check()` 仍按**搬迁前的旧模块名**导入
  （旧 `verify_segment_bitstream_v5` vs 新 `segment_bitstream_verify_v5`），且没把 `Accessory/verify` 加进
  `sys.path` ⇒ 2 个用例 `ModuleNotFoundError`，是 Gate 0「pytest 全绿」的实际阻塞；已修（新名优先 + 旧名回退 + 补 path）。
  提示：`tests/`→`Accessory/` 那次整体改名可能留下同类"旧模块名"残留，值得顺带排查。
- A6：判据 G7 的组级「执行中断」已改为逐项 FAIL（一个 `Unknown encoder` 曾吞掉 G7-1..G7-8 全部逐项结论）。
- **范围外、未改动**：`VidUtils/verify/verify_quality_mapping.py` ① 组
  `[1] 端到端命令 … → -crf 18` 报 False、脚本 `exit=1`。根因是该判据假定 `cpu_only=True`
  （`--decode cpu --scale-algo libswscale-lanczos`）会连**编码器**一起降级，但这两个开关只强制
  CPU 解码/缩放；本机 `hevc_nvenc` 可用，dry-run 实发 `-c:v hevc_nvenc -rc constqp -qp 18`。
  **属 VidUtils 仓内断言问题、非 VE 回归**，按"仅 VE 单侧"范围约定交其 V 系列会话处理；
  ⑨ 组跨项目真源一致仍 13/13。
- 本次改动（判据脚本、chroma 测试、方案 md、memory）**尚未提交**。
