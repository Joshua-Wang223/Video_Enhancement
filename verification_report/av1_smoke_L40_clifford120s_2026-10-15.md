# NVENC 端到端冒烟 + 验收报告

- 生成时间：2026-10-15T01:56:03
- 素材：`/tmp/clifford_120s.mp4`
- 编码器：`av1_nvenc`
- batch_size：8
- rate_mode：constqp
- 主机：f6dc3cf83c1e
- 环境前置：实跑一帧成功

**汇总：PASS=8 / FAIL=0 / SKIP=0**

## rate_mode = constqp

- 退出码：0（耗时 182.9s）
- 命令：`/root/.pyenv/versions/3.11.1/bin/python3 -u /workspace/Video_Enhancement/src/main_video_optimized.py -c /workspace/Video_Enhancement/config/default_config.json -i /tmp/clifford_120s.mp4 -o /tmp/av1_nvenc_smoke_out/av1_nvenc_constqp.mp4 --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc --rate-mode-ifrnet constqp --rate-mode-esrgan constqp --segment-duration 30 --batch-size-ifrnet 8 --batch-size-esrgan 8`

| 项 | 结论 | 说明 |
|---|:--:|---|
| S1 管线退出码 | PASS | rc=0，耗时 182.9s |
| S2 段级解码级门禁（decoded==expected） | PASS | 4 次通过 / 失败 0 次；分段帧数合计 5996 |
| S4 validate_decodable_video(count_mode=decode) | PASS | ok=True reason=ok frames=5996 errors=0 path=direct |
| S3 产物可解码帧数 = 各段之和 | PASS | 产物 5996 帧 vs 分段合计 5996 帧；源 3000 帧；帧数复用 S4 解码计数 |
| S5 segment_bitstream_verify_v5（帧守恒/IDR/frame_num/pts） | PASS | ✅ 验收通过 (总用时 3.18 s): 帧数守恒 / 段首无连IDR / 帧号单调 / 无 pts_anomaly / 色度正常 |
| S6 QA sidecar 字段完整 | PASS | 缺字段 无；codec_hint=av1_nvenc rate_mode=constqp |
| S7 产物编码器确为 av1 | PASS | codec=av1（期望 av1）；音轨=有 |
| S8 无内存泄漏（后半程 RSS 斜率 ≤ +50 MB/min） | PASS | RSS 峰值 4874 MB，斜率 -109.3 MB/min（主 -93.6 / 子 -15.7 MB/min；PSS -113.5），显存峰值 0 MiB，进程数 1~5，样本 36；明细 Accessory/data/s8_mem_L40_2026-10-15/clifford_120s/constqp.mem.tsv |

- 内存采样：36 点，进程数 1~5，RSS 峰值 4874.4 MB，本进程树显存峰值 0.0（驱动侧 pid 与容器 PID namespace 不通（如 477267 在 /proc 下不存在）⇒ 不可归属）
- 斜率（全树 / 主进程 / 子进程 / PSS）：-109.3 / -93.6 / -15.7 / -113.5 MB/min
- 逐进程明细：`Accessory/data/s8_mem_L40_2026-10-15/clifford_120s/constqp.mem.tsv`

> 色度检查（segment_bitstream_verify_v5 检查 4）在真实素材上是**内容相关假阳性**
> （方案 §8.5：AV1 长片 113 簇，源片段自身 40 簇）⇒ 硬指标一律 `--skip-chroma`。
