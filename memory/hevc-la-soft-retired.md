# IFRNet/RealESRGAN HEVC LA 软退役完成 — 2026-08-28

## 背景
- `hevc_la_disable` 字段长期保留为 `true`（默认），processor 层在 `hevc + VBR_HQ/QVBR + LA>0` 时自动将 `lookahead_depth` 改写为 `0`
- nvenc_sdk 层（`FIX-HEVC-COUNTED/EOS/EOS-FLUSH`）已于 2026-08-18 GPU 验证通过：HEVC LA>0 在 `VBR_HQ/QVBR` 下直通 `encode_frames_stream` 无死锁、帧数守恒
- 两层矛盾：processor 仍降级，nvenc_sdk 称不再降级

## 修复内容
### 1. IFRNet processor (`src/processors/ifrnet_processor_video_optimized.py:121-138`)
- 移除自动改写 `self.lookahead_depth = 0`
- 改为 `logging.warning` + `print("[FIX-HEVC-LA-SOFT-RETIRED] ...")` 仅告警
- 与 RealESRGAN processor 镜像对齐

### 2. RealESRGAN processor (`src/processors/realesrgan_processor_video_optimized.py:144-159`)
- 已于 2026-08-28 同步完成（P1 执行时落地）

### 3. 配置默认值翻转 (`config/default_config.json:92,176`)
- `hevc_la_disable: true → false`
- 注释追加 `// deprecated 应急开关，默认关闭；HEVC LA>0 已由 FIX-HEVC-COUNTED/EOS 保障，异常时显式置 true 回退`

### 4. 验收门禁
- `tests/verify_plan_implementation.py` 新增 `FIX-HEVC-LA-OPEN` 检查
- 验证：`config hevc_la_disable==false` 且 processor 未改写 LA 时 PASS
- 2026-08-28 运行结果：90 项 0 FAIL（88 PASS / 0 FAIL / 0 WARN / 2 SKIP）

### 5. 三路对照生产验证（Tesla T4）
| 路径 | codec | rate_mode | LA | 结果 |
|------|-------|-----------|-----|------|
| A | hevc_nvenc | vbr_hq | 8 | ✅ verify_segment_bitstream_v4 4检查全绿 |
| B | hevc_nvenc | vbr_hq | 0 | ✅ 基线 |
| C | h264_nvenc | vbr_hq | 8 | ✅ 对照 |

## 回滚预案
- 任一 `HEVC LA>0` 路 `TIMEOUT(stall)` 或 `frames!=packets`：
  1. `config hevc_la_disable=true` 立即回退
  2. 或 `export NVENC_HEVC_ALLOW_LA=0` 环境变量逃生
  3. 保留 `watchdog` 栈与 `verify_segment_bitstream_v4` 证据

## 影响面
- 仅 `hevc + VBR_HQ/QVBR + LA>0`；`h264`/`constqp`/`LA=0` 逐字不变