# upscale_then_interpolate 自动模式保护 — 2026-08-29

## 问题背景
用户在 `new5.mp4` (1280×720 @ 30fps) 上使用 `-m upscale_then_interpolate` 导致 IFRNet 早期 EOF 失败：

| 运行 | batch_size | 预期帧 | 实际帧 | 缺失率 | 结果 |
|------|-----------|--------|--------|--------|------|
| 第1次 | 12 | 348 | 145 | 58.3% | ❌ 段失败 |
| 第2次 | 6 | 348 | 319 | 8.3% | ❌ 段失败 |

**根因**：`upscale_then_interpolate` 模式下，ESRGAN 先 2× 超分 → 2560×1440 (3.7M 像素)，IFRNet 在 1440p 下插帧。T4 (SM75, 14.6GB) 无法在合理 batch_size 下跟上 NVDEC 供料：
- T2 (TRT 推理) = 1433ms (bs=12) / 686ms (bs=6)
- NVDEC 解码 >> IFRNet 推理 → reader 队列枯竭 → 早期 EOF

对比：`wws3e02_26s` (720p) 成功，因 IFRNet 在 720p 下推理，T2 ≈ 840ms (bs=24)，吞吐匹配。

## 修复方案

### 1. 可配置阈值 (`config/default_config.json:8`)
```json
"max_upscale_then_interpolate_pixels": 3670016  // 2560×1440 = 3.7M
```
- `0` = 禁用自动切换
- 默认 2560×1440 (T4 经验阈值)

### 2. 自动模式选择函数 (`src/main_video_optimized.py:210-270`)
```python
def _select_optimal_mode(config, input_video, mode, quiet=False):
    if mode != "upscale_then_interpolate":
        return mode
    max_pixels = config.get("processing", "max_upscale_then_interpolate_pixels", default=3670016)
    if max_pixels <= 0:
        return mode
    vi = VideoInfo(input_video)
    post_w = vi.width * upscale_factor
    post_h = vi.height * upscale_factor
    if post_w * post_h > max_pixels:
        # 打印强烈警告
        print("⚠️" + "="*68 + "⚠️")
        print(f"  检测到超分后分辨率 {post_w}×{post_h} ({post_pixels/1e6:.1f}M 像素)")
        print(f"  超过阈值 {max_pixels/1e6:.1f}M 像素")
        print(f"  原因: upscale_then_interpolate 会在 {post_w}×{post_h} 下运行 IFRNet 插帧")
        print(f"        T4 显存/算力不足，极易导致早期 EOF / 帧丢失")
        print(f"  动作: 自动切换为 interpolate_then_upscale 模式（先插帧再超分）")
        print(f"        插帧在 {vi.width}×{vi.height} 下进行，像素吞吐降低 {post_pixels/(vi.width*vi.height):.1f}×")
        print("⚠️" + "="*68 + "⚠️")
        return "interpolate_then_upscale"
    return mode
```

### 3. 双重调用保险
- **配置摘要阶段** (`main:2475-2478`)：提前计算并显示生效模式
- **处理入口** (`_process_single:1460-1462`)：再次调用，双重保险

## 效果验证

### 命令行输出
```
⚠️====================================================================⚠️
  检浃到超分后分辨率 2560×1440 (3.7M 像素)
  超过阈值 3.7M 像素 (config: max_upscale_then_interpolate_pixels)
  原因: upscale_then_interpolate 会在 2560×1440 下运行 IFRNet 插帧
        T4 显存/算力不足，极易导致早期 EOF / 帧丢失
  动作: 自动切换为 interpolate_then_upscale 模式（先插帧再超分）
        插帧在 1280×720 下进行，像素吞吐降低 4.0×
⚠️====================================================================⚠️
```

### 配置摘要正确显示
```
处理模式      : interpolate_then_upscale
```

### 验收结果
- 静态验证 `plan_implementation_gate.py`：90项 0 FAIL
- 码流验证 `segment_bitstream_verify_v4.py --skip-chroma`：4项硬指标全绿
  - `frames=1591 packets=1591` ✅
  - `IDR=8 首个IDR@4 其后32NAL内IDR=0 frame_num回退=0` ✅
  - `无 pts_anomaly / 解码错误` ✅

## 使用建议
| 场景 | 建议 |
|------|------|
| 默认使用 | 无需额外参数，默认 `interpolate_then_upscale` 最安全 |
| 显式指定 `-m upscale_then_interpolate` | 自动保护生效，打印警告并切换 |
| 确需 `upscale_then_interpolate` | 设置 `max_upscale_then_interpolate_pixels=0` 禁用自动切换（需自行评估风险） |
| A10/A100 等高端 GPU | 可适当调大阈值，如 `8000000` (≈4K) |

## 关联记忆
- [[hevc-la-soft-retired]] — 同期完成的 HEVC LA 软退役
- [[ifrnet-watercolor-tail-defect-investigation]] — 早期 EOF 根因分析
- [[t2-static-estimation-undershoot]] — T2 静态估算低估导致 Auto-Tune 队列配置偏小