# NVENC 端到端冒烟 + 验收报告

- 生成时间：2026-10-09T07:48:32
- 素材：`/workspace/input_videos/112 Max Bed Time.avi`
- 编码器：`av1_nvenc`
- batch_size：8
- rate_mode：constqp, vbr
- 主机：b49a5fa54f51
- 环境前置：实跑一帧成功

**汇总：PASS=0 / FAIL=2 / SKIP=0**

## rate_mode = constqp

- 退出码：1（耗时 5.1s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /workspace/input_videos/112 Max Bed Time.avi -o /tmp/opencode/l40runs/maxbed_out/av1_nvenc_constqp.avi --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet constqp --rate-mode-esrgan constqp --segment-duration 30 --batch-size-ifrnet 8 --batch-size-esrgan 8`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | FAIL | rc=1，耗时 5.1s | 尾部:   TRT Engine 缓存: /workspace/Video_Enhancement/.trt_cache /   预去噪阶段    : 关闭 /   最终合并输出  : -c:v copy (含分段 timescale 归一化)（继承 ESRGan: av1_nvenc | CRF-ref: 21（libx264 基准，按等效表换算） | preset: medium） / ────────────────────────────────────────────────────────────────────── / ❌ 源视频结构校验失败: source_decode_errors: [mp3float @ 0x556788dc3540] Header missing /    请修复/更换片源；如确需处理请先排除 ffmpeg 报告的码流问题。 |

## rate_mode = vbr

- 退出码：1（耗时 5.1s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /workspace/input_videos/112 Max Bed Time.avi -o /tmp/opencode/l40runs/maxbed_out/av1_nvenc_vbr.avi --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet vbr --rate-mode-esrgan vbr --segment-duration 30 --batch-size-ifrnet 8 --batch-size-esrgan 8`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | FAIL | rc=1，耗时 5.1s | 尾部:   TRT Engine 缓存: /workspace/Video_Enhancement/.trt_cache /   预去噪阶段    : 关闭 /   最终合并输出  : -c:v copy (含分段 timescale 归一化)（继承 ESRGan: av1_nvenc | CRF-ref: 21（libx264 基准，按等效表换算） | preset: medium） / ────────────────────────────────────────────────────────────────────── / ❌ 源视频结构校验失败: source_decode_errors: [mp3float @ 0x564636103540] Header missing /    请修复/更换片源；如确需处理请先排除 ffmpeg 报告的码流问题。 |

> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**
> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。
