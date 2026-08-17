# LA 窗口旋转修复 (FIX-LA-WINDOW-ROTATION) —— [已废弃]

- 日期: 2026-08-11
- 状态: **❌ 实机未修复，已整体废弃**。被 [[stream-ts-reassociation-fix]]（FIX-STREAM-TS-REASSOC）替代。
- 影响版本: IFRNet v6.4.3 / v6.4.3.1 / v6.4.4 / v6.4.4.1 / v6.4.5 / v6.4.5.1

## 为什么失败

1. **双重洗牌**：`_drain_write` 先按**错误 fi 标签** `pairs.sort(key=lambda p: p[0])`，再 `_fix_la_window_rotation` 按「9 帧窗口旋转」假说反向旋转 → 双重错位。
2. **真实错位非干净 rotate-by-1**：幸存率 19%、552 次 frame_num 回退远超旋转模型约 11%，合成数据回归无法覆盖真实形态。
3. **根因是 drain 关联，非窗口旋转**：`_drain_outputs_blocking` 的 `est_fi` 纯顺序假设在冷会话 LA 预热期失效 → 数据块贴错 fi，旋转修复作用在错误标签上无意义。

**结论**：该方案的全部代码（`_RotationBitReader` / `_rotation_*` / `_fix_la_window_rotation`）已在 6 版本中移除，替代为 outputTimeStamp 重关联 + writer 流式重排缓冲。保留本文件仅为记录失败教训，避免重蹈覆辙。

## 现象

首段冷会话 LA 编码(VBR_HQ/QVBR + lookahead)输出中,每 `W` 帧窗口
`[Wk..Wk+W-1]`(k>=1)被写成 `[Wk+1..Wk+W-1, Wk]`——窗口首帧被延迟
`W-1` 位到窗口末尾。结果:frame_num 回退(如 fn 9→8),违反 H.264
解码一致性 → 解码器静默丢帧/播放花屏。**仅段 000 出现**;后续段
为正常流(段边界 EOS 重置 LA 状态机)。

## 根因(推论)

LA 冷会话时硬件 lookahead 缓冲窗口与输出回收槽位存在一个窗口周期的
相位偏移,导致首段前若干窗口的输出整体循环移位。窗口周期 W 与
slot 数绑定:

```
W = _slot_count = max(pipeline_depth, LA + 1)
LA=8  → W=9  (GPU 验证)
LA=16 → W=17 (参数化推导,待 GPU 复验)
```

## 修复方案

### 1. H.264 slice/SPS 解析 (`_RotationBitReader` + 4 方法)

NVENCEncoder 内新增(带 `_rotation_` 前缀):
- `_rotation_extract_rbsp()` — 去 emulation prevention (00 00 03 → 00 00)
- `_rotation_ensure_sps_parsed()` — 从缓存 SPS 解析 frame_num 位宽与
  separate_colour_plane_flag(高 profile 需跳过 chroma/scaling_list 字段)
- `_rotation_skip_scaling_list()` — 跳过 8x8/4x4 scaling delta 序列
- `_rotation_parse_frame_num()` — 单帧返回 (is_idr, frame_num)

### 2. 检测 + 修复 (`_fix_la_window_rotation(pairs, final)`)

在 LA drain 写入前调用(先 `pairs.sort(key=lambda p: p[0])` 按数组顺序归一,
调用后**不得再次排序**):

- **检测**: 遍历 items,非 IDR 帧 fn 回退 → 启用修复
  - V2 回绕判定: `_diff = (prev_fn - fn) & ((1<<fn_bits)-1)`,
    `0 < _diff < half_range` 才算旋转;接近 2^bits 视为正常回绕(长段 >256 帧)
- **修复**: 按全段写入位置 `% _rotation_window` 分窗口,`pos>=window`
  且 `pos%window==0` 的窗口把末帧移到首;跨 chunk 不足 window 的残余
  存 `_rotation_carry`,段末块 `final=True` 直接输出残余
- 状态机: `_rotation_detected`(保持)/`_rotation_enabled`/`_rotation_carry`/
  `_rotation_written`,每段在 `_stream_begin` 重置(仅段 000 会产生旋转)

### 3. 参数化窗口周期(2026-08-11 完成,支持 LA=16)

- stream 版本 (v6.4.5.1/4.4.1/4.3.1): `self._rotation_window = self._slot_count`
  (= max(pipeline_depth, LA+1)),LA 配置改变时自动跟随
- batch 版本 (v6.4.5/4.4/4.3): CONSTQP-only(`_la_depth` 恒 0),窗口默认 9,
  `_fix_la_window_rotation` 含 `_la_depth<=0` 零开销快速路径直通,
  永不启用;未来启用 LA 时需手动设置 `_rotation_window`
- 启用日志: `启用 %d 帧窗口修复(fn %d -> %d)` 打印实际窗口大小

## Backport 矩阵

| 版本 | 类型 | 集成方式 |
|------|------|---------|
| v6.4.5.1 | 主战场(stream) | `_drain_write` 接入(sort + fix) |
| v6.4.4.1 | 完整同构(stream) | `_write_pairs(pairs, final)` 两处调用(块 final=False / 段末 final=True) |
| v6.4.3.1 | 完整同构(stream) | 内联写入循环两处(sort+fix,块 / EOS 段末 final=True) |
| v6.4.5/4.4/4.3 | 防御性(batch) | ce_pipeline 写入循环接入;快速路径零开销 |

## 验证

- `tests/verify_rotation_backport.py`(AST 提取 6 版本方法 → exec dummy 类):
  1. 合成旋转流 4960 帧修复(LA=8 窗口9 / LA=16 窗口17,双场景 12 项)翻转残留 0
  2. 集成点检查(_fix 调用 ≥2 处)
  3. 正常流不误伤 + CONSTQP 快速路径直通
- 全部 6 文件 `py_compile` 通过
- 旧验证脚本: `temp/temp2/verify_reorder_fix.py`(早期,输入为真实采集流)

## 待办 / 假设

- [ ] GPU 复验 LA=16 (VBR_HQ/QVBR + `--lookahead-depth 16`),确认窗口周期 17
- [ ] 假设: 硬件只旋转完整窗口,段末残余(< window)不旋转——合成数据按此构造,需真实流确认
- [ ] CONSTQP 下硬件静默禁用 LA(代码显式清零),LA 配置只对 VBR_HQ/QVBR 生效

## 相关

- 检测阈值参照 H.264 frame_num 回绕语义
- [[pipe4-la8-root-cause-fix]] [[nvenc-la-frame-conservation-fix]]
  [[pipeline-depth-slot-rotation-confusion]] [[sps-pps-la-pipe4-startup-corruption]]
