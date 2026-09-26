# HEVC LA>0 生产就绪（hevc_la_disable 软退役，2026-08-28）

> 依据：`Plan/HEVC_LA完全根治执行方案_20260828.md`（方案定稿）、
> `Plan/codex-session-01a0414c-*.md`（2026-08-27 综合修复落地）、
> `Plan/session-ses_fbdc.md` / `session-ses_fb9a.md`（2026-08-27/28 生产验证）、
> `Plan/session-ses_fb90.md`（2026-08-28 软退役落地与三路对照）。

## 演进时间线（三阶段接力）

1. **2026-08-18/19（根治基础）**：HEVC 空槽/未就绪槽 LockBitstream 永久阻塞根因定论，
   FIX-HEVC-COUNTED（中途有界排空）/ FIX-HEVC-EOS（EOS pending-only）/
   FIX-HEVC-EOS-FLUSH（LA=0 flush 同语义）落地，diagnose_hevc_la 11 变体矩阵全绿，
   六历史版本 backport（见 [[hevc-la-drain-diagnosis]]）。
2. **2026-08-26/27（水彩综合修复，P0-P5 落地 + 生产验证）**：
   - [P1-FIX-H2D-EVENT-SYNC]：pipeline.py `_try_prefetch_next` + ifrnet_utils.py
     `PinnedBufferPool.get_for_frames` 预取槽 event 同步（水彩根治，见
     [[ifrnet-watercolor-tail-defect-investigation]]）；
   - EOS 排空硬化：[P2-FIX-EOS-OUTPUT-ORDER] / [P2-FIX-STRICT-EOS] /
     [P3-FIX-LockBitstream-SizeCap]（垃圾块 size 上限钳制）/
     [P3-FIX-NAL-COMMON]（HEVC 参数集识别切 nal_utils，"Cached SPS+PPS" 33B→101B）；
   - 解码级验收门禁：video_utils `validate_decodable_video` /
     `count_decoded_video_frames`；verify_plan RT-4 解码级帧数守恒 +
     RT-5 解码错误零容忍 + FIX-GATE（F-修复效果 phase）；
   - 生产验证：verify_plan 93 项 0 失败；verify_segment_bitstream_v4 双视频全 PASS；
     报告：`verification_report/水彩花屏修复验证报告_20260828.md`。
3. **2026-08-28（软退役，LA>0 生产开放）**：见下。

## 软退役（FIX-HEVC-LA-SOFT-RETIRED）

- **动机**：processor 层规避路由（hevc + vbr_hq/qvbr + LA>0 → LA=0）与 nvenc_sdk 层
  "不再降级"（FIX-HEVC-COUNTED/EOS 已生产验证）两层语义矛盾，统一以已验证路径为准。
- **改动**：`src/processors/ifrnet_processor_video_optimized.py` +
  `realesrgan_processor_video_optimized.py`（镜像）+ `external/ifrnet_video/main.py`
  透传：命中 `hevc_la_disable=true` 且 hevc + vbr_hq/qvbr + LA>0 时**仅 WARN 不再改写
  LA**（`[DEPRECATED] ... 保留逃生门但不降级，直通编码器` + `[FIX-HEVC-LA-SOFT-RETIRED]`）。
- **配置**：`config/default_config.json` 的 `models.ifrnet.hevc_la_disable` 与
  `models.realesrgan.hevc_la_disable` 默认 `true → false`（方案要求最后翻转；
  回滚 = 显式置 true 或 `NVENC_HEVC_ALLOW_LA=0`）。

## 验收与验证状态

- plan_implementation_gate.py 新增 **[FIX-HEVC-LA-OPEN]** 门禁（config 双侧 false
  + 处理器含 SOFT-RETIRED 标记且无 LA 改写）→ 2026-08-29 复跑 90 项：88 PASS / 0 FAIL /
  0 WARN / 2 SKIP（`F-修复效果` 10 项全 PASS，含 FIX-H2D-SYNC / FIX-HEVC-LA-OPEN /
  FIX-GATE / FIX-EOS-ORDER / FIX-STRICT-EOS / FIX-SIZE-CAP / FIX-NAL-COMMON）。
  （Linux 生产基准；Windows 开发机无 NVIDIA GPU 时实测 86 PASS / 0 FAIL / 2 WARN
  （R5 CUDA、R7 NVENC 环境探测）/ 2 SKIP，属环境差异。）
- diagnose_hevc_la 快速回归（200 帧 LA=8）：counted / eos_probe / free_pool /
  ce_pipeline_fix 全部 conserved=True + ffmpeg decode OK。
- 三路对照生产验证（A: hevc vbr_hq LA=8 / B: hevc vbr_hq LA=0 / C: h264 vbr_hq LA=8）：
  **三路 verify_segment_bitstream_v4 4 检查全绿**（A 路实测产物 hevc 1591 帧，
  frames==packets、段首无连 IDR、frame_num 不回退、无 pts_anomaly）。收口记录见
  [[hevc-la-soft-retired]]。
- 附带确认：ESRGAN ≥1440p 自动 [FIX-HIGHRES-RC] 降级 constqp+LA=0（2560×1440 bs=16
  实测 ~3.8 frame/s，属预期）；TRT cache key 含 codec，H264/HEVC 引擎隔离。

## 同步状态（2026-08-29 已完成）

软退役补丁（两个 processor 的 [FIX-HEVC-LA-SOFT-RETIRED]）+ config 翻转
（`hevc_la_disable: false`）+ `FIX-HEVC-LA-OPEN` 门禁已在两侧仓库落地一致：
`src/processors/ifrnet_processor_video_optimized.py`、`realesrgan_processor_video_optimized.py`、
`config/default_config.json:94,178`、`Accessory/verify/plan_implementation_gate.py:1451`。
（2026-08-28 记录中的"Windows 仓库仍为规避版待同步"状态已于 2026-08-29 消除。）

## 相关记忆

[[hevc-la-drain-diagnosis]]（根因与修复栈）、[[ifrnet-watercolor-tail-defect-investigation]]
（水彩综合修复）、[[verify-bitstream-large-file-parallel]]（v4 并行验收）。
