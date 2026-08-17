---
name: realesrgan-nvenc-module-architecture
description: Real-ESRGAN NVENC SDK 模块的来源文件、导入策略、跨段复用架构
metadata: 
  node_type: memory
  type: project
  tags: 
    - realesrgan
    - nvenc
    - sdk
    - encoder
    - import
    - cross-segment
    - architecture
  originSessionId: b9b77fa5-2bfd-4abb-a9a7-c262321e2a94
---

# Real-ESRGAN NVENC SDK 模块架构

## 源文件关系

| 文件 | 状态 | 内容 |
|------|------|------|
| `nvenc_sdk.py` | ✅ 活跃 | `NVENCEncoder`(L471) + `_NVENCEncodeThread`(L2100) + `FFmpegMuxer` + `NVENCWriter`(L2458) + RGB→NV12 转换 |
| `nvenc_writer.py` | ❌ 死代码 | 薄封装 `NVENCWriter`，从 nvenc_sdk 导入。不再被任何文件引用 |

**重要**：`nvenc_sdk.py` 包含两个 `NVENCWriter` 类定义：
- `nvenc_writer.py` L37 — 旧版，依赖 nvenc_sdk，已无人导入
- `nvenc_sdk.py` L2458 — 当前版，自包含，所有依赖在同一文件内

## main.py 导入策略

```python
# 模块级条件导入（try/except），不阻塞非 NVENC 环境
try:
    from nvenc_sdk import NVENCEncoder, NVENCWriter
    _SDK_NVENC_AVAILABLE = True
except ImportError:
    NVENCEncoder = NVENCWriter = None
    _SDK_NVENC_AVAILABLE = False
```

## Encoder 跨段复用模式

`NVENCEncoder` 的生命周期由 enhancer dict 管理：

```
create_video_enhancer()
  → enhancer['_sdk_nvenc_encoder'] = None   # 占位

run_pipeline_for_video() [segment 1]
  → encoder is None → NVENCEncoder(...)     # 首段创建
  → enhancer['_sdk_nvenc_encoder'] = encoder # 缓存
  → NVENCWriter(encoder, ...)               # 使用 + 编码
  → writer.close() → flush_and_join()       # 排空 LA FIFO，encoder 保持存活

run_pipeline_for_video() [segment 2+]
  → encoder = enhancer.get('_sdk_nvenc_encoder')  # 复用！
  → NVENCWriter(encoder, ...)               # 新建 writer，共享 encoder
  → writer.close() → flush_and_join()       # 再次排空

close_enhancer()
  → encoder.close() → DestroyEncoder        # 最终销毁
```

### LA 跨段安全性

- `NVENCWriter.close()` → `_NVENCEncodeThread.flush_and_join()` → `_loop()` 末尾执行 NVENC EOS flush → LA FIFO 完全排空
- 下一段创建**新的** `_NVENCEncodeThread` + `FFmpegMuxer`
- 共享的 `NVENCEncoder` 的 NVENC session 已排空 → 无 LA 跨段污染

### 条件判断

进入 Level 1 路径需同时满足：
1. `_SDK_NVENC_AVAILABLE` — nvenc_sdk 模块可导入
2. `cuda_available` — GPU 可用
3. `'nvenc' in args.video_codec` — 用户指定了 NVENC codec

任一失败 → `base_writer = None` → 走 FFmpegWriter 兜底。

## 历史

- 旧版（20260701 前）：`main.py` 从 `nvenc_writer` 导入 `NVENCWriter` + 从 `nvenc_sdk` 导入 `NVENCEncoder`（两次导入两个文件）
- 当前版：统一从 `nvenc_sdk` 导入所有类（单文件源），但误删了跨段复用
- 修正版（本次会话）：恢复模块级条件导入 + 跨段复用 + 单文件源

## 相关记忆

- [[realesrgan-video-nvenc-sdk-audit]] — nvenc_sdk.py 的 bug 审计
- [[level1-nvenc-encoding-flow]] — Level 1 NVENC 编码数据流
