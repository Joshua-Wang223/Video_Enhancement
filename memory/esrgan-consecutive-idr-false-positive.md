---
name: esrgan-consecutive-idr-false-positive
description: ESRGAN 超分输出每段连IDR=3 是统计假象（同帧多 slice），对 IFRNet 无错误传导，无需修改 ESRGAN
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-07T04:35:25.262Z
  originSessionId: 422ae849-ffc4-4927-b7f4-8ecbda158a9e
---

## 结论

ESRGAN 超分输出每段的「连IDR=3」是统计假象（同帧多 slice），对 IFRNet 无错误传导，**无需修改 ESRGAN**。

## 证据链

1. **同帧 slice 统计假象**：`esrgan_nal_000.txt` 证实段首 4 个 `(5, fn=0)` 是同一 IDR 帧的 4 个 slice（NAL 索引紧邻 7,8,9,10），帧级换算只有 1 个 IDR 帧 → 连IDR 实际=0。

2. **per-slot IDR 在此无害**：ESRGAN `nvenc_sdk.py` line 1486 的同源 per-slot IDR warmup：
   ```python
   force_idr = force_idr_first and (slot_idx not in self._slots_warmed)
   ```
   但 ESRGAN 的 `_ensure_slot_free`(line 1335) + `_apply_drained_entries`(line 1278) 有完善的背压/相位管理，drain 数据被正确消费，因此 per-slot IDR 不会引发相位错位。

3. **码流健康**：ESRGAN 输出 FN回退=0、frames==packets、pts无异常。

4. **无传导路径**：IFRNet 输入是解码后的 YUV 帧（非 NAL 流），输入 IDR 布局在解码后消失。

5. **段 2/3 正常**：段 2/3 输入正是这些超分输出且处理正常 → 零传导铁证。

## 相关记忆

- [[verify-bitstream-idr-slice-fix]] — 码流验证工具同帧 slice 聚类修复
- [[ifrnet-v6451-seg1-fix-stack]] — IFRNet 段1花屏修复中排除此假阳性
- [[nvenc-stream-drain-backpressure-iron-law]] — ESRGAN 侧正确背压设计（为何 per-slot IDR 在此无害）
