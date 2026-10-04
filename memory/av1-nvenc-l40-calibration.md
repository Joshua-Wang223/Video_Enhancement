---
name: av1_nvenc L40 等质量标定（CQ 行 + QP 仿射行落地）
description: 2026-10-04 L40 落 QUALITY_MAP_QP['av1_nvenc']=(7.9338,-97.5136)（替代 ×3）+ 门禁同步 + CQ 行 LOO 结构性失败；含标定执行坑（manifest 路径/symlink/同名）
type: project
---

# av1_nvenc L40 等质量标定（2026-10-04）

**事实（已落地）**
- **CQ 轴**：`QUALITY_MAP['av1_nvenc'] = (1.4573, 1.1022, 0, 63)`（并行会话/VU 已落；两个独立来源逐字一致，非本会话所加）→ crf21 = `-cq:v 32`。
- **QP 轴**：本会话落 `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)`（LOO 3.64）→ crf21→69、crf30→141。**取代** `_QP_MAP_OVERRIDE` 的 ×3。
- **门禁同步**（`Accessory/verify/crf_cq_unification_verify.py`）：G1-2 av1 `-cq:v 32`；G3-7 → CQ32→QP71；G6-7/G6-8 `-qp 71`；G6-9/G6-10 `-cq:v 32`；G3-9（size 口径）仍 63 不变。探针 AC1 结论文案改指 `QUALITY_MAP_QP`。
- **验收**：`crf_cq --quick --no-gpu` 104/0/0/11；`--gpu` 113/0/3；`plan_implementation_gate` 无失败；AC1 探针表值 `-qp 71` 落带内 PASS。

**CQ 行 LOO 结构性失败（不新增，保留既有行）**
- 17 素材 LOO = 6.21（稀疏）/ 6.08（加密），超门禁 5.9；**单素材离群** `anim2d_forest`（6.08 独占，次高 4.01，其余 ≤2.43）。
- 已排除：噪声（重测逐位相同）、采样稀疏（统一加密仅 6.21→6.08）、拐点陡（其斜率比排第 9）。
- 真根因 = **模型形态**：全局仿射 `crf→cq` 给 crf30 → cq45.3，该素材真等质点 cq41.8 → ΔVMAF 6.08（内容相关编码效率偏移）。
- "增加素材"（bootstrap n=4..16，中位 6.30→6.21 **平台**）与"改模型形态"（仿射 6.41 / 二次 **6.80 更差** / 幂 6.18）**均无效**。
- 属 memory `equal-quality-loo-model-form-failure.md` 同类结构性上限。

**跨仓（CR-4）**
- CQ 行两仓同源 ✅；**QP 轴分叉**：VU 用 `_QP_SCALE['av1_nvenc']=3`（仅 ref21 验证），VE 用仿射表。
- 高 ref 差异：×3 残差 crf24 −20 / crf27 −37 / crf30 −51。已写 handoff：`VidUtils/Plan/CR-4_av1_QP轴_handoff_VE_to_VU_20261004.md`。

**执行坑（本轮新踩，复用时必看）**
1. **manifest `slice_path` 是 WSL 绝对路径**（`/mnt/d/...`），batch loader 对绝对路径不回退 ⇒ 全部 `[skip]`。须生成指向 `/workspace/input_videos/eqq_calib/` 的修正 manifest。⚠ 修正时**不要用 `.resolve()`**——会把 symlink 解析成目标名（见 3）。
2. **素材池在仓库外** `/workspace/input_videos/eqq_calib/`（12×6s + 5×10s = 17），不在 repo `input_videos/`（该目录不存在）。方案文档 §5.3/§10 写的相对路径会失败。
3. **`screen_ui_code` 6s/10s 同名不同内容**（6.0s vs 10.0s）⇒ 必须按库内约定给 10s 加 `_10s` 后缀（symlink 即可，勿成改原文件）；否则 batch workdir 撞车 + points key 合并成 16 素材（LOO 虚降 5.69~5.93）。
4. **6s/10s/legacy10s 侧 points 的素材名是原始源名**（`new1.mp4`/`word_world_2.mp4`），而 gpu_l40 侧是**切片名**（`live_kids_play_src1280x720.mp4`）⇒ 两套名字不重合、不互相 merge；落表器按 points key 首字段识别素材（不走 `clip_name_mapping`）。
5. **multi-session**：本机是共享 GPU 主机，曾有并行会话在会话期间改 `convert_crf.py`（落 CQ 行）⇒ 动手前先 `git status`/mtime 核对，勿假设树静止。
6. 落表器 `--sides gpu_l40_{cq,qp}` 单独即可复现 NVENC 行（每条文件自带 libx264 锚点）；加 `6s,10s,legacy10s` 不改变 NVENC 行。

**已知边界（非表错）**：标定切片是 **720p prep**，生产/G7 在全分辨率量测 ⇒ 映射非分辨率不变量（word_world_2 全分辨率 cq32 ΔPSNR −2.14 WARN，同值在 720p 切片 −0.34 PASS）。

**CQ 结构性上限的解法（2026-10-04 收口）**：不是表的问题，是**门禁加权**问题 —— 逐锚点最差
0.94/1.93/2.91/3.25/**6.41(crf30)**，8/8 编码器 worst 都来自 crf30。已按仓主裁定改
`GATE_ANCHORS=[0,27]`（+ crf30 降为监控列）⇒ av1 CQ 判据 LOO **3.13 ✅**。详见
`equal-quality-loo-gate-anchor-range.md`。

**L40-5 冒烟（358.8s 真实素材）**：首轮 PASS=13/FAIL=3。
- S3（两模式）为**脚本 double-count bug**（`Step 1/2`+`Step 2/2` 各校验一次分段）⇒ 已修
  `[FIX-S3-STAGE-DEDUP]`（只取 `Step 2/2` 区域）；用已有日志回放：24→12 条、35852→**17926 = 产物帧数** ✅，管线本身正确。
- S8（constqp）斜率 +112 MB/min 待空闲机复跑确认（峰值 8640 < vbr 9192，疑似趋稳非泄漏）。

