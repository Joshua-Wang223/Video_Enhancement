# 立项 Prompt：Real-ESRGAN 侧补齐 `[FIX-LA-REDRAIN]` 二次排空安全网

> ## ✅ 执行状态（2026-09-17 Linux + GPU 完成）
>
> | 项 | 结果 |
> |---|---|
> | 前置条件 1：抽出 `_apply_sps_pps()` | ✅ 已落地（只收敛 3/11 处，8 处语义不同保留内联，详见实测纠正） |
> | 前置条件 2：`_output_slot_idx` 推进条件注释 | ✅ 已落地（5 个站点标 `[FIX-LA-OUTPTR-ESRGAN]`） |
> | **主体：移植 REDRAIN** | ✅ **已完成**（2026-09-17 Linux + T4 实测） |
> | REDRAIN 移植到 `encode_frames_batch_ce_pipeline` | ✅ Phase1 harvest 推进 `_output_slot_idx`、Phase3 drain 推进 `_output_slot_idx`、批末二次排空调用 `_drain_outputs_blocking()` + SPS/PPS 缓存 |
> | 与 IFRNet `_ce_final_drain` 对齐 | ✅ 完全对齐 |
>
> ### ⚠️ 实测纠正：本文档 §3「内联散落约 15 处同模式」的假设**不成立**（已在执行前确认）
>
> 实测该文件共 **11** 个 SPS/PPS 阶梯站点，**只有一个三段式变体**
> （Phase1 harvest / Phase3 drain / encode_frame）与 IFRNet 的 `_apply_sps_pps`
> 语义**逐字等价**。其余 8 处差异分三类：
>
> | 类别 | 差异 |
> |---|---|
> | prev / results 路径（drain、EOS、final-drain） | 缓存分支**未按 IDR 门控** → **非 IDR** 首块也会预注入 muxer |
> | EOS / final-drain 的 prev-chunk 路径 | **只 prepend、不缓存** |
> | 辅助块（无 VCL）路径 ×3 | **只缓存、不注入** |
>
> ⇒ **「统一入口」对那 8 处不是纯重构，而是行为改动**（要统一必须先判定"哪种行为
> 才是对的"）。故本轮只收敛 3 处（零行为改动），其余 8 处保留内联并在站点标
> `[P2.4c-LADDER-ESRGAN] 未收敛：…`；**不要**按本文档的"纯重构+幂等"去整体替换。
>
> ### ✅ 已完成的 REDRAIN 移植（2026-09-17）
>
> `encode_frames_batch_ce_pipeline` (external/realesrgan_video/nvenc_sdk.py)：
> - Phase1 harvest: 成功取回帧后 `self._output_slot_idx += 1`
> - Phase3 drain: 成功取回帧后 `self._output_slot_idx += 1`
> - 批末 REDRAIN: 调用 `_drain_outputs_blocking()` 兜底回收 CE 提前触发导致的遗漏帧，并补 SPS/PPS 缓存（IDR 判定用首 VCL NAL 实测）
> - 与 IFRNet `_ce_final_drain` 完全对齐
>
> ### ✅ 验证结果
>
> - 门禁全量验证：92 PASS / 0 FAIL / 0 WARN / 2 SKIP
> - NVDEC/软解双路 RGB 逐字节一致（含强制 NVDEC 路径）
> - REDRAIN 相关日志：`[FIX-LA-REDRAIN] 二次排空回收 N 帧`
>
> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。
> 立项时间：2026-09-15（**符号与行号已于当日逐条核对**，见 §1）
> **完成时间：2026-09-17**　立项人：门禁强化会话
> 关联记忆：`memory/realesrgan-missing-la-redrain.md`（2026-09-01 首次记录，含设计与风险细节，
> 本文件是**可直接施工的任务书**版本，两者应一起看）

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| IFRNet 侧 BLKRETRY + REDRAIN 双层兜底 | ✅ 已有 | `_ce_final_drain` 内含 `[FIX-LA-REDRAIN]` |
| ESRGAN 侧 BLKRETRY 单层 | ⚠️ 现状 | `encode_frames_batch_ce_pipeline` 只有一层 |
| ESRGAN 侧 `_apply_sps_pps()` 统一入口 | ⬜ **前置条件 1，必须先做** | 当前**没有**该函数，SPS/PPS 注入内联散落约 15 处 |
| ESRGAN `_output_slot_idx` 二次推进评估 | ⬜ **前置条件 2，必须先做** | `_drain_outputs_blocking` 成功时会自增指针 |
| 遥测（让"少帧"能关联到返回码） | ✅ 已有 | `nvenc_sdk.py:1550` 附近的自述注释即为此而写 |
| 本立项（移植 REDRAIN） | ⬜ 待办 | 行为改动，需完整回归 |

---

## 0. 任务

在 `external/realesrgan_video/nvenc_sdk.py` 的 `encode_frames_batch_ce_pipeline()`
段末收尾处，比照 IFRNet `_ce_final_drain()` 补上 **LA>0 门控的第二层排空兜底**，
使「BLKRETRY 也失败」不再等于「帧静默丢失」。

---

## 1. 环境与代码基线（2026-09-15 核对）

- 文件：`external/realesrgan_video/nvenc_sdk.py`（ESRGAN）、
  `external/ifrnet_video/nvenc_sdk.py`（IFRNet，参照实现）

| 符号 | ESRGAN | IFRNet |
|---|---|---|
| `encode_frames_batch_ce_pipeline` | **:2261（373 行）** | :2846 起（123 行） |
| `_ce_final_drain` | **不存在** | **:3039（80 行）** ← REDRAIN 宿主 |
| `[FIX-LA-REDRAIN]` 实现 | **不存在** | :3089（回收循环）、:3117（回收打印） |
| `_drain_outputs_blocking` | :1466 | :1496 |
| `_apply_sps_pps` | **不存在**（仅 :941 注释提及） | :2281 |
| `_extract_sps_pps` / `_has_sps_pps` / `_nal_first_vcl_type` | :1297 / :1338 / :1365 ✅ | 同名齐备 |
| `_lock_bitstream_blocking` | :1801（BLKRETRY 调用点在 :2614） | 同名 |
| `_reset_output_slot_idx` | :1626（:2301 每批调用） | 同名 |
| `_output_slot_idx += 1` | :1600、:1616（`_drain_outputs_blocking` 内）、:1797、:2168、:2174 | 见 `[FIX-LA-OUTPTR]` 系列 |

> ⚠️ 顺带发现（与本立项相关）：ESRGAN 的 `encode_frames_batch_ce_pipeline` 有 **373 行**，
> 而门禁 `P3-1`（上帝函数拆解）**只覆盖 IFRNet 侧同名函数（123 行）**。
> 拆 REDRAIN 时请一并考虑把这 373 行按 IFRNet 的 Phase 结构（harvest/submit/inline/final）
> 拆解，否则它会是下一个"阈值漂移"来源。

---

## 2. 已确认的事实（实测确立，无需重做）

### 2.1 代码自己写明了这处差异

`external/realesrgan_video/nvenc_sdk.py:1550` 起：

> ⚠️ 与 IFRNet 的一处**实质差异**：ESRGAN 侧没有 `[FIX-LA-REDRAIN]` 二次排空安全网
> （IFRNet 在 `_ce_final_drain` 里有，用于兜底 BLKRETRY 也失败的帧）。本侧若 BLKRETRY
> 同样失败，帧即静默丢失，只能靠段级帧数守恒审计发现。故这里的遥测对超分侧**比插帧侧
> 更关键** —— 它是唯一能关联"少帧"与"LockBitstream 返回码"的证据。

### 2.2 触发机制：CE 在**入队**时触发，而非**完成**时

LA 预热期/高延迟时，帧被 Phase 1/3 误标为 `b""` 占位，稍后才真正完成。
IFRNet 的 REDRAIN 按 `_output_slot_idx` 顺序再 blocking 排空一轮，把"已完成却被遗漏"
的帧覆写回去；ESRGAN 缺这一轮。

### 2.3 为什么超分侧更危险

ESRGAN 没有 REDRAIN ⇒ 一旦发生：`results[fi]` 保持 `b""`/`None`
⇒ **帧静默丢失或输出错位**（错位比丢帧更隐蔽：帧数守恒但内容错位）。
且现象（少帧）与原因（返回码）之间**没有可关联证据**（遥测注释就是为此写的）。

### 2.4 IFRNet 参照实现（`:3089` 起，语义四点）

1. **门控** `self._la_depth > 0`（LA=0 无此需求）
2. **IDR 判定用首 VCL NAL 实测**（`_nal_first_vcl_type`：h264 type5 / HEVC IRAP 16~23），
   **不用** `fi == 0` 门控 —— 迟到回收的 IDR 可能出现在任意位置
3. **所有回收帧统一走 `_apply_sps_pps`**（补挂缓存参数集；否则迟到 IDR 无 SPS/PPS → 花屏）
4. 只在 `results[_est_fi]` 仍是 `None`/`b""` 占位时才覆写，**绝不覆盖已有有效数据**

---

## 3. 实施步骤

**前置条件 1（先做，独立可验证）**：在 ESRGAN 侧抽出
`_apply_sps_pps(h264_data, is_idr) -> bytes` 统一入口，把内联散落的约 15 处
（重复模式：`if _pending_idr and self._cached_sps_pps is not None and not self._has_sps_pps(...)` 等）
收敛进去。原语齐备（§1），属**纯重构 + 幂等**，可先单独验证（逐处替换后逐字节比对产物）。

**前置条件 2**：确认 REDRAIN 的二次 `_drain_outputs_blocking()` 不会与
`_reset_output_slot_idx(0)`（:2301，每批开始）造成**指针二次推进**而错位。
IFRNet 靠 `[FIX-LA-OUTPTR]` 系列注释逐站点钉住推进条件，请照做：
在 ESRGAN 侧每个 `_output_slot_idx += 1` 站点补注释说明"何时允许推进"。

**主体**：把 IFRNet `_ce_final_drain`（:3039，80 行）的 REDRAIN 段移植到
ESRGAN `encode_frames_batch_ce_pipeline`（:2261）的段末收尾处，保留四点语义（§2.4）。

**收尾**：加打印 `[FIX-LA-REDRAIN] 二次排空回收 N 帧 (LA=..., pd=...)`，
与 IFRNet 侧日志口径一致（便于两后端日志对拉）。

---

## 4. 验收判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 单元（模板：`tests/test_nvenc_la_frame_conservation.py` 的 `_drain_outputs()` 模式） | LA=8 下 N 帧 `encode_frames_batch` 分块 + 末块 `send_eos=True`，断言 `valid == N / empty == 0 / none == 0` |
| 2 | 触发确认 | 日志出现 `[FIX-LA-REDRAIN] 二次排空回收 N 帧`（≥1 次） |
| 3 | 参数集完整 | 回收帧的 IDR 都带 SPS/PPS（`_has_sps_pps` 为真） |
| 4 | 指针未错位 | LA=0 与 LA=8 各跑一遍，输出帧序正确（不是只看帧数守恒） |
| 5 | 组合回归 | **LA=0 / LA=8 × h264 / hevc** 四组合：帧数守恒 + 全片高频梯度扫描无花屏 |
| 6 | 端到端 | 超分真实素材：解码级验收通过（`verify_segment_bitstream_v4.py`）+ 帧守恒 |
| 7 | 门禁 | `python tests/verify_plan_implementation.py --no-report-file` 无 FAIL |
| 8 | 幂等性 | 未触发 REDRAIN 的批次行为与改动前逐字节一致（LA=0 路径必须完全不变） |

---

## 5. 风险与回滚

- **这是行为改动，不是纯遥测**：动的是帧回收时序，可能改变 GOP 结构与帧序。
  必须带完整四组合回归，不能只跑 LA=8。
- 最高风险 = **指针双重推进**导致输出错位（帧数守恒但内容错位，比丢帧隐蔽）。
- 回归成本高，建议按「前置条件 1 → 前置条件 2 评估 → 主体 → 四组合回归」分四次提交，
  每步都可独立回滚。
- 纯 Python 改动，回滚 = 还原对应提交；建议保留一个开关
  （如 `ESRGAN_LA_REDRAIN=0`）以便现场对照，开关默认开启。
- 前后置 1/2 项**无需 GPU 即可完成与自检**；主体与组合回归需 Linux + GPU。
