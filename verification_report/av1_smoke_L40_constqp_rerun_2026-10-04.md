# AV1 NVENC 端到端冒烟 + 验收报告

- 生成时间：2026-10-04T08:47:19
- 素材：`/workspace/input_videos/01 the race to mystery island .fixed.mp4`
- rate_mode：constqp
- 主机：5e35d15b8919
- 环境前置：实跑一帧成功

**汇总：PASS=7 / FAIL=1 / SKIP=0**

## rate_mode = constqp

- 退出码：0（耗时 661.5s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /workspace/input_videos/01 the race to mystery island .fixed.mp4 -o /workspace/input_videos/av1_smoke_out/av1_constqp.mp4 --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet constqp --rate-mode-esrgan constqp --segment-duration 30`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | PASS | rc=0，耗时 661.5s |
| S2 段级解码级门禁（decoded==expected） | PASS | 12 次通过 / 失败 0 次；分段帧数合计 17926 |
| S3 产物可解码帧数 = 各段之和 | PASS | 产物 17926 帧 vs 分段合计 17926 帧；源 8969 帧 |
| S4 validate_decodable_video(count_mode=decode) | PASS | ok=True reason=ok frames=17926 errors=0 path=direct |
| S5 segment_bitstream_verify_v5（帧守恒/IDR/frame_num/pts） | PASS | ✅ 验收通过 (总用时 8.58 s): 帧数守恒 / 段首无连IDR / 帧号单调 / 无 pts_anomaly / 色度正常 |
| S6 QA sidecar 字段完整 | PASS | 缺字段 无；codec_hint=av1_nvenc rate_mode=constqp |
| S7 产物编码器确为 av1 | PASS | codec=av1；音轨=有 |
| S8 无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min） | FAIL | RSS 峰值 8506 MB，斜率 +149.5 MB/min，显存峰值 11346 MiB |

> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**
> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。
