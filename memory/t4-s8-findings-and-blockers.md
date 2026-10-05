---
name: T4 上执行 L40 §9.5-D 的结论 —— S8 判据不可靠 + 三个阻塞项（B1 生产级缺陷 / B2 / B3）
description: 2026-10-04 在 T4 上以 h264/hevc NVENC 代替 AV1 执行 S8 定位：否证「constqp 路径泄漏」但判据本身不可靠（换 OLS 窗口可得 −786~+1434）；产出 B1（h264+LA>0+跨段复用段 2 必崩，生产默认路径）/B2/B3 三个阻塞项与 E1~E5 记录勘误；权威原文在 Plan/PROMPT_L40_AV1等质量标定专项执行方案.md §0.1
type: project
---

**权威原文**：`Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` **§0.1**（含五臂实测表、窗口对照表、
E1~E5 勘误表、§0.1.5 六步推进顺序）。本条只记**结论与纪律**，细节与 file:line 以该文档为准。
**原始采样已存档** `verification_report/s8_20261004_raw/`（五份 `.mem.tsv` + README）⇒ 复算不必再上机。

**Why**：S8（constqp 臂长跑内存斜率 +149.5 MB/min，两次复现）是 L40 上唯一未闭环的遗留项。
AV1 只能跑 Ada ⇒ 走 T4 降本验证（`h264_nvenc`/`hevc_nvenc` 的 constqp 分支对整个 NVENC 族一致）。

**How to apply**：讨论 S8 / 内存斜率 / 跨段 SPS/PPS 之前先读本条，**不要**沿用「constqp 路径泄漏」
或「显存为 0 ⇒ 无 GPU 挂载」这两个已被推翻的旧结论。

## 1. 能收口的：T4 否证了「constqp 路径泄漏」

两条走完全片的 constqp 臂后半程斜率 **h264 −55.6 / hevc +3.1 MB/min**，均落在噪声内
（后半程 sd 1356 MB ⇒ 不确定度约 ±66 MB/min）。
**不能收口的**：h264 的 vbr 对照臂因 B1 崩溃（有效样本仅 12 点、全在启动爬坡段 ⇒ 复算得
+4585.8/+1872.4 的**启动伪影**，不是趋势）⇒ **A/B 对照只在 hevc 上成立**；
S8 原始问题「L40 上 +149.5 是否 AV1 特有」**仍未回答**。

## 2. ⚠ 最重要的方法论产出：S8 判据本身不可靠

同一份 HEVC constqp 数据、只换 OLS 窗口：前 10% **+1434.2** / 后 50% **+3.1** /
**全程 +149.4** / 剔前 40% +36.1 / 后 25% −240.5 / 90–100% 段 **−786.7**。
⇒ 「全程 +149.4」与 L40 记录的「+149.5」几乎相同，而同一信号换窗口可得 −786 ~ +1434
⇒ **L40 那个数很可能是锯齿信号的窗口伪影，而非真泄漏**。
⚠ 锯齿幅度来源已定位：**pinned result pool 随 `batch_size` 线性增长**（bs=24 实测逐段
305→549→794→1038→1190→**1251 MB**）⇒ 冒烟默认 `--batch-size` 已改为 **8**，
**跨 batch_size 的 S8 读数不可比**。
⚠ `--mem-peak-mb` 默认 **12000 已被实测否决并于 2026-10-05 重标定为 16000**：
T4 三条**已确认无泄漏**的臂峰值 12393/13877/13578 MB **全部超出 12000**（斜率 −55.6/+3.1/+35.8 均在噪声内）
⇒ 12000 **只会产假 FAIL、不具判别力**。新值 = 最坏实测 ×1.15，双向复算通过（干净臂 Fail→Pass、合成泄漏仍 Fail）。
峰值已查明为**结构性锯齿**（出现在全程 80~87%、6~10 次 2~3GB 段切换下跌、RSS≈PSS、几乎全在主进程），
**不是泄漏征兆** ⇒ 峰值判据优先级低于斜率，换素材/卡型/bs 必须重标。详见 `memory-leak-attribution-measurement.md`。

**附带收益（bs=8 vs 24，插帧阶段，2 轮交替 A/B 各差 <0.5%）**：墙钟 33.45s → **29.14s（−12.9%）**，
单批 343/363 ms → 90/90 ms，pinned pool 305 → 102 MB。完整两阶段收益**未实测**。

## 3. 三个阻塞项

| ID | 问题 | 根因要点 | 影响面 |
|---|---|---|---|
| **B1** | **h264 + LA>0 + 跨段复用 ⇒ 片段 2 起必崩**（`non-existing PPS 0 referenced` → muxer pipe broken → rc=1） | `ffmpeg_io.py:163` `nvenc_map={'libx264':'h264_nvenc'}` ⇒ config 默认 `libx264` 在 T4 上就升级为 h264_nvenc；`main.py:1155` `_force_new` 只含 `("hevc","av1")` ⇒ h264 复用编码器；`nvenc_sdk.py:2282/2285` `_stream_begin` 每段清 `_cached_sps_pps=None`；`:971` `repeatSPSPPS` bit12 不写 ⇒ 驱动不重吐参数集；`_prepend_param_sets`(`:2297`) 与 `_drain_write`(`:3927`) 两条通道同时失效 | **生产默认路径**（T4 + 默认 config + 多段视频）；回归点 `1e57c0b`(2026-09-18)；`realesrgan_video/nvenc_sdk.py` 逐字同构。**方案**：`_stream_begin` 保留 `_cached_sps_pps`、只重置 `_sps_pps_injected`，与 LA=0 路径 `:2858` 对齐 |
| **B2** | `--rate-mode vbr/cbr` 在 Level 1 **静默落 CONSTQP** 且 LA 未启用 | `nvenc_sdk.py:1055` `else` 兜底写 `rc_ptr[1]=0`；`:1069` LA 门控 `in ('vbr_hq','qvbr')` 排除 vbr；`:634` 清 LA 判据是 `== 'constqp'` ⇒ Python 认为 LA=8 而硬件 CONSTQP+LA=0。三点连带：① 守卫 `('constqp',0)` 不命中 ⇒ **绕过 per-frame completionEvent**，正落在 `nvenc-drain-unsubmitted-slot-segfault.md` 记载的 T4 崩溃组合上；② QP 未过 `to_constqp_qp` 换算 ⇒ 画质偏松（h264 Ready `QP=22` vs vbr `QP=26`）；③ `nvenc_sdk.py:568` AV1 自动降级是**无需用户传参**的第三入口。方案 A = 脚本级禁用 vbr/cbr + 修 `_log_ready` 自相矛盾回显 | Level 1 直通路径（AV1 全部 + HEVC 降级时） |
| **B2 落地（2026-10-05）** | 方案 A **已实施但做了关键偏离**：不能字面「禁用 vbr/cbr」。**AV1 的 `effective_rate_mode` 本身就是 `'vbr'`**（`:1050` 与 `:568` 双处把 `vbr_hq` 自动降级为 `vbr`）⇒  blanket 禁用会把**本专项主体 AV1 一起打死**。改为**按确定性分级**：codec 字面含 nvenc 且非 av1 ⇒ **硬错误拒绝**（必然走 Level 1）；`libx264/libx265` ⇒ **仅告警放行**（可能被 `nvenc_map` 升级，但软编主机上 `vbr` 合法）；`av1_nvenc` ⇒ 放行。另修 `_log_ready`（两侧 `nvenc_sdk.py` 逐字同构）为**按硬件真值回显**，消除 `CONSTQP … la=8` 自相矛盾。9/9 校验矩阵通过。⚠ 仍**未**实现真 vbr 分支（方案 B，需 GPU） | 已落地（校验层 + 日志层，无GPU 需求） |
| **B3 落地（2026-10-05）** | 显存口径已改：`_gpu_mem_by_pid()` 返回 `(表, 状态)`，四态分开（正常/真空/不可测-NSpid 不通/nvsmi 失败），不可测报 `None`；落盘写 `NA`；`peak()` 返回 `Optional`；报告渲染「不可归属（原因）」。四分支注入自测 + 合成泄漏仍检出 | ✅ 已落地 |
| **B3** | 显存 `gpu_tree_mib` **恒为 0（静默假零）** | `nvidia-smi --query-compute-apps` 返回**宿主 pid**（实测 725115/1071006），在容器 `/proc` 下不存在；`/proc/<pid>/status` 的 `NSpid` 只有一层 ⇒ **容器 PID namespace 与驱动侧不通**，`nvenc_sdk.py` 的 pid 查表永远 miss。**⚠ 推翻** memory 里「本容器无 GPU 挂载 ⇒ 显存记 0」的旧归因（GPU 实际满载 7698 MiB/100%）。`0.0` 会被读成「确实没用显存」，**`None` 才能表达「测不到」** | 仅污染 S8 的显存维度；RSS/PSS 口径正确 |
| **B4** 🔴 | **优先级高于 B1/B2/B3**：`rc_ptr[1]` 写入**非法** rateControlMode | `vbr_hq`→32、`qvbr`→64，而权威枚举只有 `CONSTQP=0 / VBR=1 / CBR=2`（curl 实取 `nvEncodeAPI.h` line 270-275）⇒ 32/64 枚举中不存在；官方 CQ 承载方式是 `RC_VBR(1)`+`targetQuality`（line 1605「for VBR mode」）。**「T4 实测 32 被接受」不能证伪** —— 接受 ≠ 语义正确 | **CQ 轴全部标定**（`QUALITY_MAP`/`QUALITY_MAP_QP`）+ 「constqp>qvbr>vbr_hq」性能排名 + 仓库默认档位（config:99/179）均建立在此未定义值上。详见 `nvenc-rc-enum-illegal-vbr-hq.md` 与 Plan §0.2 |

## 4. 记录勘误（E1–E5，必须随引用一起传播）

- **E1** `nvenc-sps-pps-debugging.md:42`「多段视频段 2+ 不再报 `non-existing PPS`」——
  在 **h264 + LA>0 + 跨段复用** 下**已不成立**；成立范围仅 LA=0 或 hevc/av1 每段新建编码器路径。
- **E2** `sps-pps-la-pipe4-startup-corruption.md:19,:92` 同 E1。
- **E3** `ifrnet-multisegment-la-strict-drain.md`（提交 `1e57c0b`）只记「段内 fi→全局 gfi 错位」与
  「HEVC EOS 漏帧」，**未记同提交引入的 SPS/PPS 缓存清空缺陷**（B1，机制不同但同源）。
- **E4** `t4-vbrhq-verification-plan.md:43` 行号 `1044/1056` 已漂移为 **`1055/1069`**。
- **E5** 同上行内容缺三点：QP 未换算、`[FIX-CONSTQP-FRAME-CE]` 被绕过、AV1 自动降级 `:568` 第三入口。

## 5. 推进顺序（§0.1.5 + §0.2.7，有依赖勿打乱）

0. 🔴 **跑 RC 枚举真值探针**（`Accessory/probe/nvenc_rc_enum_truth.py --caps-only`）——
   **最高优先，新增**。若 caps=`0b111` 则 B4 是必修项，「方案A vs 方案B」这个问题无意义。
   需 GPU。见 `nvenc-rc-enum-illegal-vbr-hq.md`
1. ~~修 B3~~ → **✅ 2026-10-05 已落地**（显存报 `None` + 标注不可归属，无需 GPU）
2. 修 B1 + hevc臂负向回归 — 需 GPU
3. 重跑 h264 **`vbr_hq`** 臂（不是 `vbr`）—— 这才第一次真正测到 h264 的 LA>0 路径 — 需 GPU
4. ~~B2 方案 A（无需 GPU）~~ → **✅ 2026-10-05 已落地**（分级拒绝 + 修日志回显）；方案 B 真 vbr 分支仍需 GPU
5. ~~标定 `--mem-peak-mb`~~ → **✅ 2026-10-05 已重标定为 16000**（据 T4 三条干净臂实测 ×1.15）
4b. **B4**：`vbr_hq`→`RC_VBR(1)`，重跑全套标定 — 需 GPU，**须先有步 0 结论**
6. **L40 复跑**同素材 constqp + vbr_hq + av1 三臂同批，判 S8 原始问题 — 需 L40

## 6. B2 落地时自查出的两个自伤（同轮，2026-10-05）

实施分级拒绝后**自查发现并已修复**，记录以防重犯（通用教训见
`feedback-prescribed-fix-may-break-subject.md` 追加两节）：

1. **拦截打断了它自己要验证的实验**：`h264_nvenc + vbr` 正是 S8 A/B 对照臂，
   冒烟脚本 docstring 与 `--rate-modes` 默认值都用它 ⇒ 已加**前置守卫**（秒退）
   并把示例/默认改为 `vbr_hq`；核实影响面仅此一处（方案内 3 处 `constqp,vbr` 均为 av1_nvenc）。
2. **校验与自动改写 config 的代码顺序颠倒**：B2 检查在首轮循环、AV1 降级在更靠后的循环
   ⇒ 两轮读到不一致状态，AV1 靠逃生门侥幸通过 ⇒ 已把降级**提到首轮**，
   矩阵加「请求值→生效值」两列，13/13 通过。


**Why 这样分级（可复用的判据）**：「拒绝某个取值」这类修复必须先问**该取值在别处是否合法**——
本例 `vbr` 对 AV1 是**必需**的合法 RC，对软编主机也是合法的。⇒ 判据不能是「取值非法」，而是
「**该取值在本次要拦的那条路径上**未实现」，据此只硬拒确定项、对模糊项告警。
**How to apply**：见 `feedback-prescribed-fix-may-break-subject.md`。
