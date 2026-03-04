# 版本历史与升级指南 — Video Enhancement

> 本文档记录各版本的核心改进和升级步骤。

---

## 版本概览

| 版本 | 发布日期 | 核心改进 |
|------|---------|---------|
| v1.0 | 2026-02-04 | 基础功能：IFRNet + Real-ESRGAN，单视频/批量处理 |
| v2.0 | 2026-02-04 | 分段直连流水线，超分断点续传，延迟清理询问 |
| v3.0 | 2026-02 | Real-ESRGAN 直接视频处理，减少中间 PNG 暂存 |
| v5.0 | 2026-02/03 | 全面硬件加速：FP16、torch.compile、CUDA Graph、NVDEC/NVENC、OOM 降级 |

---

## v5.0（当前版本）

### 核心新增功能

**推理加速**
- `use_fp16: true` — FP16 半精度推理，显存减半，速度提升 1.5–2x
- `use_compile: true` — PyTorch 2.0 torch.compile 图编译（提升 10–40%）
- `use_cuda_graph: true`（IFRNet）— CUDA Graph 推理图固化
- `use_tensorrt: false`（可选）— TensorRT 引擎加速，需单独安装

**I/O 加速**
- `use_hwaccel: true` — NVDEC 硬件解码 + NVENC 硬件编码自动探测
- `batch_size` + `prefetch_factor`（Real-ESRGAN）— 批量推理 + 数据预取

**稳定性**
- OOM 自动降级：显存不足时自动减半 `batch_size` 并重试
- `report_json` — 可选性能报告（处理速度、显存峰值等）

**音频**
- `smart_extract_audio()` — 智能音频处理，优先无损 copy

### 主入口更新

```bash
# v5 主入口（推荐）
python src/main_video_v5_single.py -i video.mp4

# 旧 v1 入口（仍可用，但缺少 v5 功能）
python run.py -c config/default_config.json -i video.mp4
```

### 配置文件更新

v5 在 `models.ifrnet` 和 `models.realesrgan` 下新增了以下参数（旧配置文件缺失时使用默认值，向后兼容）：

```json
{
  "models": {
    "ifrnet": {
      "use_fp16": true,
      "use_compile": true,
      "use_cuda_graph": true,
      "use_tensorrt": false,
      "use_hwaccel": true,
      "report_json": null
    },
    "realesrgan": {
      "batch_size": 4,
      "prefetch_factor": 8,
      "use_compile": true,
      "use_tensorrt": false,
      "use_hwaccel": true,
      "report_json": null
    }
  }
}
```

### 性能对比（RTX 3080，1080p，10 分钟视频，2x插帧 + 4x超分）

| 指标 | v1.0 | v2.0 | v5.0 | 改进 |
|------|------|------|------|------|
| 总处理时间 | 87 分钟 | 58 分钟 | 38 分钟 | ⬇️ 56% |
| 磁盘 I/O | 45 GB | 28 GB | 22 GB | ⬇️ 51% |
| 峰值显存 | 10.5 GB | 8.2 GB | 5.8 GB | ⬇️ 45% |

---

## v2.0

### 核心改进

**分段直连流水线（最重要）**

旧流程：`插帧分段 → 合并 → 再分段 → 超分分段 → 合并`  
新流程：`插帧分段 → [直接传递] → 超分分段 → 合并`

省去中间合并和二次分段，节省 33% 时间和 38% I/O。

**超分断点续传**：v1 只有插帧支持断点，v2 补充了超分阶段的断点恢复。

**延迟清理询问**：批量处理时不再每个视频询问，全部完成后统一询问。

### 升级方法

```bash
# 直接使用 v2 主程序（无需替换任何文件）
python src/main_v2.py -c config/default_config.json -i video.mp4

# 注意：不推荐继续使用 v2，请直接升级到 v5
```

---

## v1.0 — 基础版本

初始版本功能：
- IFRNet 视频插帧（2x/4x/8x/16x）
- Real-ESRGAN 视频超分（2x/4x）
- `interpolate_then_upscale` 和 `upscale_then_interpolate` 两种模式
- 插帧阶段断点续传
- 批量处理
- 视频损坏自动修复

---

## 兼容性说明

| 项目 | v1→v2 | v2→v5 |
|------|-------|-------|
| 配置文件 | ✅ 完全兼容 | ✅ 向后兼容（新增参数有默认值） |
| 命令行参数 | ✅ 完全兼容 | ✅ 完全兼容 |
| 输出格式 | ✅ 一致 | ✅ 一致 |
| 断点文件 | ⚠️ 插帧断点可复用，超分需重新处理 | ✅ 可复用 |

---

## 常见升级问题

**Q: 从 v1/v2 升级到 v5，需要重新下载模型吗？**  
A: 不需要。v5 使用相同的模型文件，只是推理方式不同。

**Q: 旧的 `run.py` 还能用吗？**  
A: 可以，但它对接的是 `src/main.py`（v1），不包含 v5 的加速功能。建议直接使用 `src/main_video_v5_single.py`。

**Q: `use_compile: true` 导致首次运行很慢？**  
A: 正常现象。torch.compile 首次运行需 1–5 分钟编译模型，之后运行会直接使用缓存，速度大幅提升。

**Q: 升级后出现 `AttributeError` 或模块找不到？**  
A: 检查 `base_dir` 配置是否正确，确保 `external/IFRNet/` 和 `external/Real-ESRGAN/` 目录存在且包含对应的 v5 脚本。

---

## 回滚方法

如需回滚到旧版本：

```bash
# 回滚到 v2
python src/main_v2.py -c config/default_config.json -i video.mp4

# 回滚到 v1
python run.py -c config/default_config.json -i video.mp4
```

旧版处理器文件均已保留在 `src/processors/` 目录中，可随时切换。
