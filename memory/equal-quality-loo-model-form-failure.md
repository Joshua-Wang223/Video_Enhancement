---
name: 等质量换算 LOO 门禁未达标 —— 根因是表格式而非过拟合（2026-10-01）
description: 第二轮标定 4 素材 LOO worst ΔVMAF 4.18~13.73（门禁 1.0），已排除过拟合与素材池污染，根因是 (a,b,lo,hi) 单行仿射表格式承载不了跨素材等质量关系；含「0 评估点假通过」判据漏洞与 LOO 工具位置
type: project
---

等质量换算表（`QUALITY_MAP`）第二轮标定（4 素材 × 6 档，`--subsample 1`）的
**留一交叉验证（LOO）门禁 ΔVMAF < 1.0 全部不达标**，worst ΔVMAF：

| 编码器 | LOO worst | 主要离群素材 |
|---|---|---|
| libsvtav1 | 13.73 | new1 / word_world_2 |
| libx265 | 6.89 | new1 |
| libaom-av1 | 6.88 | new1 / new5_raw |
| librav1e@10 | 5.05 | new1 / new5_raw |
| libvpx-vp9 | 4.18 | new5_raw / new1 |

**根因（已定性，纯 CPU 零重编码可复现）**：**表格式不足**，不是标定执行错误、
不是过拟合、不是素材池污染。三条排除性证据：

1. **oracle（素材自身拟合）本身就超门禁** —— `libsvtav1` 的 ΔVMAF 达
   4.10（new4_raw）/ 5.61（new5_raw）/ 3.36（word_world_2）。留出素材完全不参与
   拟合时仍不达标 ⇒ 训练内误差就超了，LOO 只是把它暴露出来。
2. **换模型形式救不回** —— 分段（2 段）把 svtav1 从 4~5.6 降到 1.12~2.43 仍不达标；
   仿射/分段/2 点 oracle 三种形式 LOO 全 FAIL（分段 4.2~10.7、2 点 3.8~7.7）。
3. **素材池污染不是主因** —— 剔除 `word_world_2`（720×576 上采样，VMAF 由缩放主导）
   后仍全 FAIL（x265 6.13 / svtav1 13.51 / rav1e@10 4.58）；锚点收窄到 26/22/30/34
   也只降到 5.09 / 7.26 / 3.99。

**归因**：各素材自身 (a, b) 都不可迁移（x265 的 a ∈ [1.031, 1.150]、
b ∈ [−4.7, −0.3]；b_m 区间 [−4.33, −1.67]），且锚点区 x264 VMAF 局部斜率在
crf18–22 仅 −0.11~−0.49（**近乎平坦 ⇒ 参数反解条件数极差**），小参数误差被放大成大
VMAF 误差。达标需改**表格式**（分段/查表）或按内容类型分档出表。

**⚠ 判据漏洞（已修，值得记住）**：诊断中「二次模型 LOO 全 0.000 ✅」是**假通过** ——
二次预测参数落到扫描区间外，`vmaf_at_param` 返回 `None` ⇒ 0 个评估点，而代码把
「无评估点」当成 `worst=0` 报 PASS。凡是「预测值落在扫描区间外」的折叠式
评估，**必须把 0 评估点判 `inf`（模型失效）而非 PASS**。

**Why**：门禁不放水的前提是判据本身正确；这次若不查证就会拿一个 bug 结果
（「二次模型完美达标」）去改生产表格式。

**How to apply**：任何 LOO / 交叉验证脚本，凡用 `interp/vmaf_at_param` 在
有限扫描区间上做折叠回查，都要有「0 评估点 ⇒ 失败」断言。本次已写入
`Accessory/probe/loo_equal_quality.py` 与 `probe/loo_equal_quality.py`。

**产出与状态**：
- LOO 工具从 harness 的 `--loo`（B 侧旧版）**迁为两仓同源独立脚本**：
  `Accessory/probe/loo_equal_quality.py`（VE）/ `probe/loo_equal_quality.py`（VU），
  支持 `--workroot/--tag`（可重复，Stage3 单独 workdir 时须多 `--tag` 合并）/
  `--tiers/--tol/--quiet`，秒级完成（只读 `points.json`，不重编码）。
- harness 两仓已同源（A 侧为基线，marker-walk 定位项目根）：PAVA 保序回归、
  分段拟合诊断、`--selftest`（19 项纯逻辑自测）、`points.json` 逐点断点、
  `librav1e` 按 `-speed` 档 tier 化。
- **表值尚未回填** —— `QUALITY_MAP` 两仓仍是第一轮值，落表格式待用户裁定。
- 门禁现状（2026-10-01）：`plan_implementation_gate` 84 项 75 通过 / **0 失败** / 4 警告 / 5 跳过；
  `crf_cq_unification_verify --quick` PASS=50 **FAIL=3（G4/G9/G10，cv2 缺失，环境性）** / SKIP=24；
  VidUtils `verify_quality_mapping` ⑨ 组 **14/14 全绿**。

**Why（背景）**：立项 M2 验收标准是「留一法 ΔVMAF < 1.0」，此前从未在 4 素材口径下
真正跑过 LOO（第一轮 3 素材 subsample=8 已作废）。这次是首次执行该门禁，
结论是**当前表格式达不到该门禁**，需要仓主在「改表格式」与「放宽门禁」之间裁定。
