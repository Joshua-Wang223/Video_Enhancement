# NVENC 端到端冒烟 + 验收报告

- 生成时间：2026-10-09T07:21:05
- 素材：`/workspace/input_videos/new5_raw.mp4`
- 编码器：`av1_nvenc`
- batch_size：8
- rate_mode：constqp, vbr
- 主机：b49a5fa54f51
- 环境前置：实跑一帧成功

**汇总：PASS=15 / FAIL=1 / SKIP=0**

## rate_mode = constqp

- 退出码：0（耗时 844.9s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /workspace/input_videos/new5_raw.mp4 -o /tmp/opencode/l40runs/new5_smoke_out/av1_nvenc_constqp.mp4 --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet constqp --rate-mode-esrgan constqp --segment-duration 30 --batch-size-ifrnet 8 --batch-size-esrgan 8`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | PASS | rc=0，耗时 844.9s |
| S2 段级解码级门禁（decoded==expected） | PASS | 1 次通过 / 失败 0 次；分段帧数合计 1605 |
| S3 产物可解码帧数 = 各段之和 | PASS | 产物 1605 帧 vs 分段合计 1605 帧；源 803 帧 |
| S4 validate_decodable_video(count_mode=decode) | PASS | ok=True reason=ok frames=1605 errors=0 path=direct |
| S5 segment_bitstream_verify_v5（帧守恒/IDR/frame_num/pts） | PASS | ✅ 验收通过 (总用时 40.50 s): 帧数守恒 / 段首无连IDR / 帧号单调 / 无 pts_anomaly / 色度正常 |
| S6 QA sidecar 字段完整 | PASS | 缺字段 无；codec_hint=av1_nvenc rate_mode=constqp |
| S7 产物编码器确为 av1 | PASS | codec=av1（期望 av1）；音轨=有 |
| S8 无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min） | FAIL | RSS 峰值 9386 MB，斜率 +712.0 MB/min（主 +396.1 / 子 +316.0 MB/min；PSS +709.1），显存峰值 0 MiB，进程数 1~3，样本 165；明细 /tmp/s8_mem_new5/constqp.mem.tsv |

- 内存采样：165 点，进程数 1~3，RSS 峰值 9385.7 MB，本进程树显存峰值 0.0（驱动侧 pid 与容器 PID namespace 不通（如 19459 在 /proc 下不存在）⇒ 不可归属）
- 斜率（全树 / 主进程 / 子进程 / PSS）：712.0 / 396.1 / 316.0 / 709.1 MB/min
- 逐进程明细：`/tmp/s8_mem_new5/constqp.mem.tsv`

## rate_mode = vbr

- 退出码：0（耗时 240.4s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /workspace/input_videos/new5_raw.mp4 -o /tmp/opencode/l40runs/new5_smoke_out/av1_nvenc_vbr.mp4 --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet vbr --rate-mode-esrgan vbr --segment-duration 30 --batch-size-ifrnet 8 --batch-size-esrgan 8`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | PASS | rc=0，耗时 240.4s |
| S2 段级解码级门禁（decoded==expected） | PASS | 1 次通过 / 失败 0 次；分段帧数合计 1605 |
| S3 产物可解码帧数 = 各段之和 | PASS | 产物 1605 帧 vs 分段合计 1605 帧；源 803 帧 |
| S4 validate_decodable_video(count_mode=decode) | PASS | ok=True reason=ok frames=1605 errors=0 path=direct |
| S5 segment_bitstream_verify_v5（帧守恒/IDR/frame_num/pts） | PASS | ✅ 验收通过 (总用时 38.76 s): 帧数守恒 / 段首无连IDR / 帧号单调 / 无 pts_anomaly / 色度正常 |
| S6 QA sidecar 字段完整 | PASS | 缺字段 无；codec_hint=av1_nvenc rate_mode=vbr |
| S7 产物编码器确为 av1 | PASS | codec=av1（期望 av1）；音轨=有 |
| S8 无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min） | PASS | RSS 峰值 9778 MB，斜率 -928.5 MB/min（主 -231.2 / 子 -697.3 MB/min；PSS -921.6），显存峰值 0 MiB，进程数 1~3，样本 47；明细 /tmp/s8_mem_new5/vbr.mem.tsv |

- 内存采样：47 点，进程数 1~3，RSS 峰值 9778.2 MB，本进程树显存峰值 0.0（驱动侧 pid 与容器 PID namespace 不通（如 114770 在 /proc 下不存在）⇒ 不可归属）
- 斜率（全树 / 主进程 / 子进程 / PSS）：-928.5 / -231.2 / -697.3 / -921.6 MB/min
- 逐进程明细：`/tmp/s8_mem_new5/vbr.mem.tsv`

> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**
> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。
