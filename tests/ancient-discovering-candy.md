# ESRGAN VBR_HQ+LA=8 分块编码修复 — v3 (第4次测试问题修复)

## Context

v2 分块编码修复成功解决了 OOM（显存从 22.8GB 降至 ~380MB，GPU 显存平坦）。
但第4次测试暴露了三个新问题：

1. **帧丢失**: 9012 预期 → 9004 实际 (丢失 8 帧 = LA_depth)
2. **H.264 解码错误**: `illegal short term buffer state detected`, `Missing reference picture`
3. **跨段 SPS/PPS**: segment 2 开头 `non-existing PPS 0 referenced`

## 问题 1 & 2 根因: 分块间 slot 碰撞导致 `_slot_pending` 覆盖

### 碰撞机制

`encode_frames_batch` 使用轮转 slot 分配: `slot_idx = self._frame_idx % self._slot_count`。

Chunk 1 最后 8 帧 (LA_depth=8) 提交到了 slot 7, 8, 0, 1, 2, 3, 4, 5 (global fi 142-149)。这些帧在 LA 缓冲中尚未产出输出。

Chunk 2 前 8 帧 (global fi 151-158) 使用 slot 7, 8, 0, 1, 2, 3, 4, 5 — **与 chunk 1 的 LA 缓冲帧完全重叠**。

L1334: `self._slot_pending[slot_idx] = (fi, ...)` — 覆盖 chunk 1 的 pending entry。

当 chunk 2 的 drain 循环取回 chunk 1 延迟产出的 H.264 数据时:
- `_pending = self._slot_pending[drain_slot]` → 找到 chunk 2 的 entry
- `_actual_fi = _pending[0]` → chunk 2 的 local fi (如 0-7)
- `results[0-7] = chunk1_delayed_h264` → **chunk 1 的帧数据写入 chunk 2 的 results 数组**

后果:
- Chunk 1 的 8 个 LA 缓冲帧永久丢失
- Chunk 2 的前几个位置被 chunk 1 的错误帧数据污染
- H.264 参考帧链断裂 → 解码错误

### 修复: 全局帧索引 + 跨块延迟输出处理

**核心思路**: `_slot_pending` 存储 GLOBAL frame index 而非 local fi，drain 时计算 local fi = global_fi - chunk_start_global。

**修改文件**: `external/realesrgan_video/nvenc_sdk.py`

**修改 1**: `NVENCEncoder.encode_frames_batch()` — 全局帧索引

在函数开头记录 chunk 起始全局索引:
```python
_chunk_start_global = self._frame_idx
_prev_chunk_outputs: list = []  # 跨块延迟帧收集
```

L1334 改为存储全局 fi:
```python
# Before:
self._slot_pending[slot_idx] = (fi, slot['bs_buf'], force_idr, _ep_status)
# After:
self._slot_pending[slot_idx] = (self._frame_idx, slot['bs_buf'], force_idr, _ep_status)
```

**修改 2**: 三个 drain 路径全部改用全局 fi → local fi 计算

主循环 drain (L1357-1391):
```python
_global_fi, _, _is_idr, _ep_s = _pending
_local_fi = _global_fi - _chunk_start_global
if _local_fi < 0:
    # 来自前一个 chunk 的延迟输出 → 收集到溢出列表
    if _h264_data:
        # 需要注入 SPS/PPS (如果是 IDR)
        if _is_idr and self._cached_sps_pps is not None:
            _h264_data = self._cached_sps_pps + _h264_data
        ...
        _prev_chunk_outputs.append(_h264_data)
elif _local_fi < n_frames:
    results[_local_fi] = _h264_data if _h264_data else (b"" if _ep_s == NV_ENC_ERR_NEED_MORE_INPUT else results[_local_fi])
```

EOS drain (L1396-1441) 和 send_eos=False 最终 drain (L1443-1464) 同样修改。

**修改 3**: 返回 `_prev_chunk_outputs` 给调用方

在 return 前，将 `_prev_chunk_outputs` 存入实例变量供 `_write_la_output` 读取:
```python
self._prev_chunk_outputs = _prev_chunk_outputs
```

**修改 4**: `_NVENCEncodeThread._write_la_output()` — 先写跨块输出

```python
def _write_la_output(self, h264_list, is_final: bool):
    # 先写入前一个 chunk 的延迟帧（已在 slot 碰撞前恢复）
    _prev = getattr(self._nvenc, '_prev_chunk_outputs', None)
    if _prev:
        for h264_data in _prev:
            if h264_data:
                if not self._nvenc._sps_pps_injected:
                    _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                    if _sps:
                        self._writer.write_sps_pps(_sps)
                        self._nvenc._sps_pps_injected = True
                self._writer.write(h264_data)
                self._written += 1
                self._prev_h264 = h264_data
        self._nvenc._prev_chunk_outputs = None

    # 然后正常写入当前 chunk 的输出 (现有逻辑)
    for h264_data in h264_list:
        ...
```

## 问题 3 根因: `_sps_pps_injected` 跨段不重置

`main.py:717-731`: `NVENCEncoder` 首段创建后缓存到 `enhancer['_sdk_nvenc_encoder']`，后续段复用。

`_NVENCWriter.__init__` 每段创建新的 `_NVENCEncodeThread` + `FFmpegMuxer`，但 encoder 的 `_sps_pps_injected` 仍为 `True`（从段 1 继承）。

段 2 的新 muxer（新输出文件）需要 SPS/PPS 注入，但 `_write_la_output` 检查 `_sps_pps_injected` → `True` → 跳过注入 → muxer 收到无 SPS/PPS 的 H.264 流 → `non-existing PPS 0 referenced`。

### 修复: 段开始时重置标志

**最佳位置**: `_NVENCWriter.__init__` 或 `_NVENCEncodeThread._loop()` 开头。

最简洁: 在 `_loop()` 开头重置 `_sps_pps_injected` + 如果已有缓存 SPS/PPS 则预注入 muxer:
```python
def _loop(self):
    # [FIX-SPS-PPS-SEGMENT] 跨段复用 encoder 时必须重置注入标志
    if self._nvenc._cached_sps_pps is not None:
        self._nvenc._sps_pps_injected = False
```

## 修改清单总结

| 修改 | 文件:位置 | 内容 |
|------|----------|------|
| 1 | `nvenc_sdk.py:encode_frames_batch` | `_chunk_start_global` 记录 + `_slot_pending` 存全局 fi |
| 2 | `nvenc_sdk.py:encode_frames_batch` 三处 drain | 全局 fi → local fi 计算 + `_local_fi < 0` 跨块处理 |
| 3 | `nvenc_sdk.py:encode_frames_batch` return 前 | `self._prev_chunk_outputs` 存储 |
| 4 | `nvenc_sdk.py:_write_la_output` | 先写 `_prev_chunk_outputs` 再写当前 results |
| 5 | `nvenc_sdk.py:_loop` 开头 | 重置 `_sps_pps_injected` (段间复用) |

## 验证方法

1. **帧完整性**: `ffprobe -count_packets` → 输出帧数 == 输入帧数 (9012)
2. **H.264 解码**: `ffprobe -count_frames` 无错误
3. **跨段 SPS/PPS**: 段 2+ 无 `non-existing PPS 0 referenced` 错误
4. **显存**: 保持平坦 (v2 已验证)
5. **CONSTQP 回归**: LA=0 路径不受影响
