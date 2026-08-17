---
name: esrgan-cross-segment-sigsegv-fix-stack
description: ESRGAN 段2 SIGSEGV 历史根因链与修复栈（2026-07-29 架构已演进）
metadata: 
  node_type: memory
  type: project
  status: historical-reference
  superseded_by: esrgan-cross-segment-optimization-complete
  tags: 
    - realesrgan
    - nvenc
    - cross-segment
    - sigsegv
    - reopen
    - historical
  originSessionId: 0567f4ad-2f7a-4f67-a136-73bc484b316b
  modified: 2026-07-29T08:09:19.379Z
---

> **⚠️ 架构已演进（2026-07-29）**：
> - `reopen()` 方法已删除，`FIX-SEG-START-SYNC` 已移除
> - 当前架构：驱动会话跨段持续复用（`FIX-SKIP-REOPEN`），段首 batch 直接走 CE pipeline
> - 配套加固（SEG2-PREFLIGHT / RC-CHECK / SYNC-BEFORE-SUBMIT / EARLY-FLUSH）全部保留
> - 全貌见 [[esrgan-cross-segment-optimization-complete]]

# ESRGAN 跨段 SIGSEGV 历史根因链与修复栈（2026-07-28）

## 症状

段 1 全程正常，段 2 首个编码 batch 后进程 SIGSEGV（core dumped），
S02E05 / S02E06 生产复现。日志最后停在 `[GPU0] NVENC SDK Level 1 编码已激活`
加进度条前几帧（如 24/9003）。外部修复（只动 main.py/nvenc_sdk.py 表层）无效。

## 根因链（三层递进）

1. **EOS 后复用 session**：段末 EOS flush 后复用同一驱动 session，段 2 首 batch
   必崩。`[FIX-CROSS-SEGMENT-SESSION]` 引入 `reopen()`：段边界原地销毁并重建
   session/slots/events/stream（~百毫秒级），Python 对象 / DLL / primary ctx /
   SPS+PPS 缓存保留（真正的跨段复用）。
2. **reopen 后仍崩**：两轮状态清理补丁（context 健康检查、cuCtxSetCurrent 失败
   abort、_slot_pending 清理）均未根治 → 问题不止于驱动会话状态。
   `[DIAG-SEGFAULT]` main.py 启用 faulthandler 实锤：崩在驱动内部
   `nvEncLockBitstream`（`_lock_bitstream_with_retry` ← ce_pipeline Phase 1
   harvest），模式为 **gen=1 新会话从不崩、gen≥2 段边界（EOS/reopen）后首个
   async CE→LockBitstream 循环必崩**。
3. **根治 `[FIX-SEG-START-SYNC]`**：每段第一个 batch 改用同步 `encode_frame`
   逐帧编码（对齐 IFRNet 每段 f0 同步编码的生产验证模式），第二批起恢复
   ce_pipeline 异步流水线。每段开销 ~百毫秒级。S01E07 多分段跑通验证。

## 配套加固（同批落地，均在 external/realesrgan_video/nvenc_sdk.py）

| 标记 | 内容 |
|------|------|
| [FIX-SEG2-PREFLIGHT] | 编码线程启动预检 context/会话句柄：无效状态从不可归因 SIGSEGV 转化为带 gen 代数的 RuntimeError |
| [FIX-RC-CHECK] | ce_pipeline/flush 的 libcuda 与 EOS 调用返回码检查：被污染的 context（如 700 illegal address）不再静默 |
| [FIX-REOPEN-CTX-ORDER] | reopen()/close() 先 push primary ctx 再销毁：原顺序下销毁调用在缺 context 线程上静默失败 → 驱动资源泄漏，旧会话仍挂驱动里 |
| [FIX-SYNC-BEFORE-SUBMIT] | 活跃 NVENCWriter `write_frame_batch`/`_flush_mini_batch` 在 submit 前补 `torch.cuda.current_stream().synchronize()`——submit() docstring 的强制契约，缺失 → 编码线程读到未写完的 NV12（花帧风险）。此前只补在死代码 nvenc_writer.py |
| [EARLY-FLUSH] | `begin_flush()` 移植到活跃 NVENCWriter：**先 flush mini-batch 残留帧再发 SENTINEL**（pipeline.py 存在单帧 write_frame 路径，段末可能残留 1–3 帧；先发 SENTINEL 会让 close() 补交帧被静默丢弃）。pipeline.py 段末提前非阻塞 flush，缩短"拖尾期" |

## 死代码教训（重要）

`nvenc_writer.py` 全项目无 import（见 [[realesrgan-nvenc-module-architecture]]）。
拖尾期方案的 begin_flush 最初落在该死代码上——pipeline.py 的
`hasattr(writer, 'begin_flush')` 对活跃 writer **恒 False，修复从未生效**。
教训：改动前必须先验证目标文件在调用链上（grep import 链确认），
`hasattr` 特性探测会 silently 跳过不生效的补丁。

## 逃生门

`ESRGAN_DISABLE_SDK_NVENC=1` 整体回退 FFmpeg CLI 编码路径（h264_nvenc）。

## 关联

- [[esrgan-reopen-slot-phase-misalignment]] — reopen 时代的 slot 相位错位 bug
- [[esrgan-segment-reuse-frame-idx-reset]] — reopen 前时代的 _frame_idx 重置 bug
- [[esrgan-nvenc-slot-backpressure]] — _ensure_slot_free 背压机制
- [[realesrgan-nvenc-module-architecture]] — 模块架构与跨段复用模式
