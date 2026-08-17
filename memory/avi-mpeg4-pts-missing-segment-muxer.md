---
name: avi-mpeg4-pts-missing-segment-muxer
description: 老 AVI/mpeg4 缺 PTS 导致 ffmpeg segment muxer 整段输出不切分，-fflags +genpts 修复
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b336732-03a1-423b-8d4d-956c163baacd
  modified: 2026-08-11T08:35:12.700Z
---

# 老 AVI/mpeg4 缺 PTS 导致 segment 分割失败

2026-08-11 修复：`112 Max Bed Time.avi`（8:13，`--segment-duration 100`）日志显示
`分割为 5 段` 但 `成功分割为 1 个有效片段`，且该段时长 = 全片 8:13.793，退化为整体处理。

## 根因

输入是 **MPEG-4 Part 2 (mpeg4) in AVI** 的非标准编码。ffprobe 帧级证据：
- **DTS 全部为 N/A**（14799 帧去重后只剩 1 个值）
- **PTS 只出现在每 3 帧的 B 帧**（`0.0667,B / N/A,B / N/A,P / 0.1668,B / ...`），约 2/3 包 PTS 缺失

ffmpeg AVI demuxer 无法重建该文件的 PTS/DTS → **segment muxer 的切点判断基于包时间戳，
时间戳缺失 → 切点永不触发 → 整段输出 1 个文件**。

**Why:** 用 `-c copy` 分割时不能重新打时间戳，PTS 缺失直接传导到 segment muxer。
**How to apply:** 分割/按时间截取老 AVI/mpeg4 时，若日志出现"分割 N 段但只有 1 个有效片段、
且该段时长=全片"，优先怀疑输入 PTS 缺失，而非关键帧稀疏或代码逻辑。

## 修复

`src/utils/video_utils.py` `split_video_by_time()` 的 ffmpeg 命令在 `-i` **前**加输入选项：

```python
'ffmpeg', '-fflags', '+genpts', '-i', input_video, ...
```

要点：
- `-fflags` 是**输入选项，必须放在 `-i` 之前**（ffmpeg 选项作用域，否则不生效）
- `genpts` 只在 PTS 缺失时生成，对 PTS 完整的视频（h264/mp4）**零影响** → 无回归

## 验证

- 修复后对 `112 Max Bed Time.avi` 正确切出 5 段（101.0/106.5/94.3/102.4/89.6s），全 ~100s
- `test_video.mp4`（h264/mp4）分割结果与修复前逐字节一致（3 段 1:43/1:38/1:39）→ 无回归

## 附带观察（未处理）

test1 日志 `分段 2/2: 1:38.264` 与 test2 的 `分段 2/4: 1:38.264` 完全相同 —— test1/test2
共用 `temp/esrgan_video/esrgan_video_test_video/` 分段目录，分割失败后**残留旧文件被误判为
有效片段**（`split_video_by_time` 验证只查 exists+integrity，不校验是否本次分割产物）。
genpts 修复后 AVI 分割正常，此隐患不易触发，暂未处理。

关联：[[verify-mpeg4-vop-analysis]]（mpeg4 码流 VOP 解析，不同问题但同为 mpeg4 输入坑）
