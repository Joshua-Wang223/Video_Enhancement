# HEVC/AV1 跨段 NVENC 编码器复用 —— 恢复评估与分阶段执行

## Context

2026-09-01 修复了 `hevc_nvenc + vbr_hq + LA=8` 段级死锁（根因：LA 输出延迟实测 = LA+1 而物理槽仅 LA+1，余量为 0 → 提交↔排空循环依赖 → LockBitstream 在驱动内永久阻塞）。期间在 `external/ifrnet_video/main.py:1103` 加临时禁用：**HEVC/AV1 从第 2 段起强制新建编码器（`_force_new`）**。

本计划评估该禁用可否撤销。依据两份分析：本会话的代码核查（与 `Plan/plan_hevc_cross_segment_reuse_restore.md` 交叉验证），采纳其分阶段审慎方案。

## 评估结论

**技术上可以恢复，但当前维持禁用默认，按"开关 → 修缺陷 → A/B 实测 → 数据决策"推进。**

支持恢复的证据（已核实）：
1. **两条挂死触发链已结构性消除**：
   - 链 A（LA=0，段首 f0 走 `encode_frame()` → `_drain_outputs_blocking()` 锁残留异常槽）→ 已由 `[FIX-F0-ALWAYS-IN-BATCH]` + `[FIX-F0-IN-BATCH-CE]` 关闭，f0 不再走 `encode_frame`
   - 链 B（LA>0，`_ensure_slot_free` 槽余量 0 循环依赖）→ 已由 `[FIX-LA-SLOT-HEADROOM]`（`_required_buffers = la_depth+2`，`nvenc_sdk.py:584`）+ `[FIX-ESF-NO-LOCK-WHEN-EMPTY]`（ready≤0 绝不 Lock，`nvenc_sdk.py:1656`）关闭
2. **复用路径本身有设计支撑**：段首 `_stream_begin(force=True)` 无条件重置（`nvenc_sdk.py:3379`，清 pending + `_output_slot_idx=_frame_idx` 对齐）；`set_muxer_ref` 每段重绑新 muxer（`main.py:1742`）；段首 IDR 重建 DPB（`main.py:1406`）
3. **对照实验**：H.264 一直跨段复用零事故；ESRGAN 侧 `[FIX-SKIP-REOPEN]` 已验证跨段复用可行（插帧侧修复正是照搬其设计）
4. **防御栈与 `_force_new` 互补、不依赖它**：`_drain_outputs_blocking` HEVC poll 模式（2s deadline，`nvenc_sdk.py:1447`）、EOS 排空只锁 pending 槽（`nvenc_sdk.py:2225`）、`strict_eos` fail-fast 门禁

不立即放开的原因：
- **验证盲区**：会话 2 的修复验证（3 段成功、1591 帧守恒、零告警）是在 `_force_new` 生效（每段新编码器）下完成的，**不能证明复用安全**——驱动级 LA/DPB 状态机跨段残留（R4）无软件侧可观测手段
- **收益极小**：每段重建仅毫秒~百毫秒级（禁用态实测 3 段全片 1 分 13.5 秒，占比 <1%）
- **风险不对称**：死锁发生在驱动内部，不可中断、不可超时、不可降级，只能整段失败
- **存在独立缺陷 R1**（见下），恢复前必须先修

## 修改内容（Phase 0 + Phase 1，均为零行为改变 / 零风险）

### Phase 0 —— 硬编码改可回滚开关（`external/ifrnet_video/main.py:1103`）

```python
# [P0-FIX-HEVC-SEGMENT-HANG] HEVC/AV1 默认禁止跨段复用编码器。
# 触发链 A/B 已由 [FIX-F0-ALWAYS-IN-BATCH] / [FIX-LA-SLOT-HEADROOM] /
# [FIX-ESF-NO-LOCK-WHEN-EMPTY] 从源头消除，故禁用降级为保守默认；
# IFRNET_NVENC_CROSS_SEGMENT_REUSE=1 放开（A/B 实测用，见 Plan/plan_hevc_cross_segment_reuse_restore.md）。
_reuse_env = os.environ.get('IFRNET_NVENC_CROSS_SEGMENT_REUSE', '0')
_force_new = (codec in ("hevc", "av1")
              and _reuse_env != '1'
              and not self._is_first_segment())
```

默认 `'0'` = 维持现状（禁用），`'1'` = 放开复用。行为零改变；`main.py` 已 import os（确认）。

### Phase 1 —— 修 R1：`_sps_pps_injected` 段边界重置（`external/ifrnet_video/nvenc_sdk.py`）

**缺陷（已核实）**：`_sps_pps_injected` 只在 `close()`（3136）重置；跨段复用恰好不 close → 段 2 新 muxer 的 SPS/PPS 预注入被 `_drain_write`（3339）/ LA=0 路径（3422）的 `if not self._nvenc._sps_pps_injected` 门控跳过，只能靠 IDR 自带参数集 + `_prepend_param_sets`（1916）兜底。注释（3136"跨段重置换，支持 encoder 复用"）与实现矛盾。**无论是否恢复复用都该修**（H.264 复用路径同样受影响）。

修复：在 `_stream_begin()`（`nvenc_sdk.py:1879`，已是"每段一次"的统一入口）加 `self._sps_pps_injected = False`；把 `close()` 内 3136 行注释改为"关闭即弃"语义，消除矛盾。

## 验证（Phase 2 起需 Linux 生产侧 GPU，Windows 无 GPU）

A/B 实测矩阵（素材：干净 CFR `new5.mp4` + 缺陷 VFR `wws3e02_26s.mp4`，每组合 ≥3 段）：

| # | codec | rate_mode | LA | 开关 | 关注 |
|---|---|---|---|---|---|
| 1 | hevc_nvenc | constqp | 0 | 0（基线） | 耗时/帧数守恒/段级门禁 |
| 2 | hevc_nvenc | constqp | 0 | 1 | + 段 2+ 首帧 IDR 大小、SPS/PPS 存在性 |
| 3 | hevc_nvenc | vbr_hq | 8 | 0（基线） | + slots 数、frame_num 单调 |
| 4 | hevc_nvenc | vbr_hq | 8 | 1 | 重点：段间是否挂死 |
| 5 | h264_nvenc | constqp | 0 | 1 | 对照（现网即复用），验 Phase 0/1 无回归 |

硬指标（任一 FAIL 判该组合不可恢复）：全部段完成无挂死；`tests/verify_segment_bitstream_v4.py --skip-chroma` 段级全绿（帧守恒/单 IDR/frame_num 单调）；全片 `ffmpeg -f null` 零解码错误、总帧数 = Σ(2n_i−1)；段 2+ 首帧 IDR 正常量级（~71KB 正常，~375KB=噪声花屏）；每段 SPS/PPS 可读。

决策门：Δ<1% 或组合 2/4 任一 FAIL → 维持禁用并归档 memory；全 PASS 且 Δ≥1% → 灰度（`IFRNET_NVENC_CROSS_SEGMENT_REUSE=1` 跑 1~2 个生产任务，稳定后翻默认 '1' 并在 config 加注释化说明）；组合 2 PASS 但 4 FAIL → 只放开 LA=0 路径（`_force_new` 追加 `and self._la_depth <= 0`）。

## 记忆同步

验证定论后写 `memory/ifrnet-hevc-cross-segment-reuse-restore.md`，更新两侧 MEMORY.md 索引（canonical `C:\Users\Administrator\.claude\projects\...\memory` ↔ 仓库 `memory/`）。

## 涉及文件

- `external/ifrnet_video/main.py` — `_get_or_create_nvenc_encoder`(1087-1120)：Phase 0 开关
- `external/ifrnet_video/nvenc_sdk.py` — `_stream_begin`(1879)：Phase 1 重置；`close`(3136)：注释
- 历史 `external/IFRNet/process_video_v6_*.py`：勿改（已确认无 `_force_new`，非生产路径）
