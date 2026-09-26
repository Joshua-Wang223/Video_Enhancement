---
name: 性能探测「最佳点」的三项判据
description: 用户要求 batch_size 等性能探测的最佳点判据必须同时给出 FPS、GPU 平均利用率、处理时长/视频时长三项
type: feedback
---

做性能优化探测（batch_size / tile_size / 分辨率等参数寻优）时，**最佳点判据不能只看吞吐**，必须同时记录并汇报三项：

1. 视频超分处理的 FPS（主判据，越高越好）
2. GPU 平均利用率（测量窗口内采样均值；已饱和即不再往上探）
3. 处理时长 / 视频时长（realtime factor = 源帧率 / 超分 FPS，越小越好）

**Why:** 2026-09-26 用户在我给出「只看 FPS 拐点」的探测脚本后明确补充："最佳点的判断依据包括视频超分处理的FPS，GPU 平均利用率，视频处理时长/视频时长"。单一吞吐指标会掩盖「GPU 已喂饱但仍在堆 batch 吃显存」和「看似快但实际远慢于实时」两类误判。

**How to apply:** 探测类脚本（如 `Accessory/probe/batch_size_optimizer_probe.py`）的每个测点都要落这三项到结果表与 JSON 报告；排序时以 FPS 为主，FPS 平手（差距 < min_improve）时依次比 GPU 利用率更高 → 显存更低 → batch_size 更小。注意 ③ 与 ① 数学上反相关（③ = 源帧率/①），不要重复计权，只作直观汇报项。
