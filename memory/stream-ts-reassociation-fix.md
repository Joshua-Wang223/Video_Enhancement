# NVENC stream-LA drain 关联修复 (FIX-STREAM-TS-REASSOC)

- 日期: 2026-08-11
- 状态: 已实现 + 6 版本 backport + 合成回归通过；**需生产 GPU 复验**（本机 Windows 无 T4）
- 影响版本: IFRNet v6.4.3 / v6.4.3.1 / v6.4.4 / v6.4.4.1 / v6.4.5 / v6.4.5.1
- 前置失败方案: [[la-window-rotation-fix]]（FIX-LA-WINDOW-ROTATION，实机未修复，已整体移除）

## 现象（生产 tt2 实跑，双阶段 VBR_HQ+LA=8）

- IFRNet 段1：writer 自报输出 4961 帧，但容器 packets=4960、ffmpeg 解码仅 957 帧（幸存率 ~19%）。
- analyze_video_pipeline_v3 检出 pts_drops=153 / est_missing=4003 的周期性 pts 跳变（约每 8-9 帧一次）。
- verify_segment_bitstream_v2 报 frames(957)!=packets(4960)、frame_num 回退 552 次。
- **仅段1损坏**：段2/段3 完全正常（frames==packets，FN回退=0）。
- 日志出现 `[FIX-LA-WINDOW-ROTATION] 检测到 LA 窗口旋转，启用 9 帧窗口修复（fn 17 -> 9）` 但输出仍损坏。

## 离线取证（Windows 本机，下载损坏段 NAL dump 解剖）

**码流字节完整、纯顺序错乱**：
- 损坏段1：P-slice 19408/4=4852 + IDR 108 = 4960 = packets，字节完整。
- 错位模式：每 9 帧（=slot_count=LA+1）一个窗口，窗口内「+9 跳跃帧」与「回退帧」固定交替，**周期遍布全段（非仅预热期）**。
- IDR=108 帧级计数其实正常（健康段 104/107）；真异常是 **552 次 frame_num 回退**。
- 段2/3 健康：会话热后 `est_fi` 顺序假设碰巧成立。

## 根因（代码事实）

`_drain_outputs_blocking` 返回 `est_fi = self._output_slot_idx`（line 1822）——即「第 N 次 LockBitstream 取回 = 第 N 个提交帧」的**纯顺序假设**。冷会话 LA=8 预热期（前 8 帧 NEED_MORE_INPUT）硬件输出 buffer 重路由使该假设失效 → 数据块贴错 fi 标签。所有消费点（backpressure / 主 drain / EOS / 末尾 drain）用 `_est_fi % slot_count` 查 slot_pending 取真实 fi，标签错 → 取错帧。

**rotation 修复为何失败**：`_drain_write` 先按错误 fi 标签 sort，再 `_fix_la_window_rotation` 按「9 帧窗口旋转」假说反向旋转 → **双重洗牌**。真实错位非干净 rotate-by-1（幸存率 19%、552 次回退远超旋转模型约 11%），修复后仍损坏。

**仅段1损坏**：段2+ NVENC 会话复用（FIX-SKIP-REOPEN）LA 状态机已热，顺序假设碰巧成立。

## 修复方案

### 1. ts 重关联 [FIX-STREAM-TS-REASSOC]（主）

- `NV_ENC_PIC_PARAMS.inputTimeStamp@24` 已写入全局 `_frame_idx`（跨段单调，符合「严禁重置 inputTimeStamp」铁律）。
- `NV_ENC_LOCK_BITSTREAM.outputTimeStamp@40`（u64）回显对应输入帧的 inputTimeStamp（偏移由已验证锚点 size@36/ptr@56 推定）。
- `_stream_begin` 记录段基线 `_strm_ts_base = _frame_idx`（段首提交前）。
- drain 时读 `_out_ts@40`，`fi = _out_ts - _strm_ts_base` 得真实流内帧号。
- 新增 `_resolve_drained_fi(est_fi, out_ts)`：ts 有效且命中 pending → 用 ts；否则回退 est_fi 并计数 `_diag_ts_fallback` 告警（首 5 次 + 每 50 次采样），**保证 ts 失效时行为不劣于旧路径**。
- `_drain_outputs_blocking` 返回三元组 `(est_fi, out_ts, h264_data)`，全部消费点适配。

### 2. writer 流式重排缓冲 [FIX-STREAM-REORDER-BUFFER]

- `_drain_write` 删除 per-chunk `pairs.sort` + `_fix_la_window_rotation` 调用。
- 改为 `_ReorderBuffer` 等价逻辑：`_reorder_next` + `_reorder_pending` dict，仅输出严格递增前缀，跨 chunk 缓冲残余，`final=True` 段末强制排空 + 帧数守恒校验。
- 缓冲有界：超过 `8*slot_count` 帧仍未闭合 → 按 fi 排序强制排空（防死锁/内存膨胀）。

### 3. 移除 FIX-LA-WINDOW-ROTATION 全部代码

- 删除：`_rotation_extract_rbsp` / `_rotation_ensure_sps_parsed` / `_rotation_skip_scaling_list` / `_rotation_parse_frame_num` / `_fix_la_window_rotation` / init rotation 状态 / `_stream_begin` 重置段 / `_drain_write` 调用点。
- 主文件 v6.4.5.1 **保留 `_RotationBitReader` 类**（备用方向 B：frame_num 码内解析重关联）；历史版本一并删除。

### 4. 关注点4 次要修复 [FIX-BR-CLAMP-DIAG]

- avgBitrate 估算 `_est_br = max(50M, int(w*h*fps*3.0))` 对 720p48 算出 132.6Mbps 失控 → 新增 `_clamp_bitrate()` 钳到 [5M, 50M]，maxBitRate 同步。
- InitializeEncoder code=8 时输出完整参数诊断（rc/avgBR/maxBR/tq/LA/slots/preset/分辨率/fps）。
- preset 降级重试（原已实现）：veryslow→p4，cache 失败对象清空（隐患 B 补强）。

## Backport 矩阵

| 版本 | 类型 | 内容 |
| --- | --- | --- |
| v6.4.5.1（主） | stream | ts 重关联 + 重排缓冲 + 移除 rotation + BR 钳制 |
| v6.4.4.1 / v6.4.3.1 | stream 同构 | 同上（`_write_pairs`/内联写入两处集成点 + 8 处 drain 消费点） |
| v6.4.5 / v6.4.4 / v6.4.3 | batch（CONSTQP-only） | 仅移除 rotation 死代码（无 `_drain_outputs_blocking`，ts 不适用） |

## 验收工具链修复（v2 与 v3 矛盾根因）

- `tests/verify_segment_bitstream_v2.py`：`check_decode_integrity` 加 `-vsync 0 -vf showinfo`，解析逐帧 pts 计算 dup/drop/backward（对齐 v3 `_parse_showinfo_line` 算法），替换 250 行 `'pts_anomaly' in low` 空转（该串是 v3 自定义格式，ffmpeg 从不输出 → v2 [3] 恒空转 OK）。
- 实测：损坏段1 检出 153 条 pts drop（与 v3 pts_drops=153 **精确一致**），健康段2 无异常 → v2 与 v3 不再矛盾。
- `tests/analyze_video_pipeline_v3.py`：汇总层加 frames<<packets 硬失败标记（`[HARD-FAIL] 帧数守恒`）。
- `tests/verify_segment_bitstream_v3.py`：同样移植 showinfo pts 真实解析（消除 268 行空转）。

## 验证

- 合成回归（`tests/verify_rotation_backport.py` 重构）：20 组随机乱序重排缓冲输出严格递增 + 帧数守恒、正常流零误伤、BR 钳制、6 版本 AST 断言（rotation 已移除 + stream 版 drain 含 ts）全通过。
- **需生产 GPU**：`python tests/diagnose_lockbitstream_timestamp.py --la 8` sweep 确认 outputTimeStamp@40 回显偏移；实跑 tt2 同参数命令复验段1 frames==packets、frame_num 无回退、v2/v3 全部 PASS、无花屏卡顿。

## 2026-08-12 生产实跑 V1→V2 修正（temp/test5/output_tt5.txt）

V1 首次实跑暴露 3 个缺陷，已全部修复为 V2：

1. **sweep 偏移 @40 经生产确认正确**：日志 `ts 未命中 pending: ts=9/18/27 ts_base=0`，ts 间隔恒为 slot_count=9，与离线取证的 9 帧窗口错位完全吻合——outputTimeStamp@40 确实回显提交时的 inputTimeStamp。
2. **V1「ts 需命中 pending」是失败设计**：`ts=0` 首帧走 fallback 消费了 slot0 的 pending → 后续 ts=9/18/27 永不命中 → 数据块被 `continue` 丢弃 → 重排缓冲爆 125 帧强制排空乱序。V2 改为**直接信任 ts**（硬件回显物理必然正确），仅两层防护：ts>=ts_base（含 0）+ ts 未见于 `_ts_seen`；ts 不可信时回退链 pending_fi→est_fi。
3. **ts=0 falsy 陷阱**：`if out_ts` 把首帧合法 ts=0 误判为未回显。V2 用 `out_ts >= _ts_base` 判定。
4. **无 VCL 辅助块识别**：NVENC LA 预热期把 SPS/PPS/AUD 作为独立输出块 drain（无 VCL slice，实测 ts=0 重复回显）。新增 `_nal_first_vcl_type`：无 VCL 块缓存参数集后跳过，不占 fi、不进 seen（否则其 ts=0 与首帧合法 ts=0 冲突污染 seen）。
5. **`_drain_write` nonlocal 崩溃**：`UnboundLocalError: _reorder_next`（重排缓冲上限强制排空时重赋值缺 nonlocal 声明）→ V2 在 `_drain_write`/`_write_pairs` 加 `nonlocal _reorder_next`。

**V2 统一消费 helper `_consume_one_drained`**：物理 slot 由 `est_fi % slot_count` 决定（仅用于释放 pending），fi 标签由 ts 重关联决定，数据块**永不丢弃**，is_idr 用 NAL 实测。四个 drain 消费点 + EOS + 末尾 drain 全部统一调用。

**同次实跑确认有效的部分**：BR 钳制生效（1440x960@59.9 钳到 avgBitrate=50000kbps）、preset 初始化成功（veryslow→index=6，无 code=8）。

**backport**：V2 已同步到 v6.4.4.1（`_write_pairs` 重排缓冲 + nonlocal + 7 处消费点）与 v6.4.3.1（内联写入点重排缓冲 + 7 处消费点）。

## 2026-08-12 生产 sweep 确认 V2 + 诊断脚本双射判定修复（temp/test6/output_tt6.txt）

tt6 生产执行 `python tests/diagnose_lockbitstream_timestamp.py --la 8` 与 `--la 16`：

1. **outputTimeStamp@40 回显完整、双射成立**：LA=8 提交 18 帧、drain 71 块，@40 命中 18/18；LA=16 提交 34 帧、drain 127 块，@40 命中 34/34。sweep 明细中 @40 的 u64/u32 唯一命中值集合完整覆盖提交 ts 集合。
2. **「未找到双射偏移」结论是判定误报**：`analyze()` 完全双射三条件含 `n_drain == len(submitted_set)`（drain 块数==提交帧数）。LA 预热期硬件把 SPS/PPS/AUD 作为独立无 VCL 辅助块 drain（ts=0/重复），LA=8 时 71>18、LA=16 时 127>34 → 计数恒等式必然失败，即使 @40 已 100% 回显。
3. **sweep 判定修复（tests/diagnose_lockbitstream_timestamp.py，仅此一文件）**：双射判定基于 VCL 帧子集——records 组装时已用 `_parse_frame_num` 解析 frame_num（辅助块=None），与生产 `_nal_first_vcl_type`「无 VCL 辅助块不占 fi、不进 seen」策略同源：
   - `analyze()`：`vcl_records = [r for r in records if r['frame_num'] is not None]`；`bij = (len(uniq)==len(submitted_set) and uniq==submitted_set)`，移除 n_drain 计数恒等式；多 slice 同帧 ts 重复经 set 去重兼容；u64/u32 两处 sweep 同样处理。
   - `print_report()`：报告头 `(VCL=n 辅助块=m)`；验证明细辅助块行标注 `aux/(aux,skip)` 不参与一致性比对（辅助块 ts=0 是合法值）；顺序统计（est_fi/ts-fi 序列与乱序判定）仅基于 VCL 帧，避免 ts=0 干扰。
   - 退出码语义不变（命中 exit 0 / 未命中 exit 1）。
4. **离线单测通过**：构造 LA=8（18 VCL + 53 aux）与 LA=16（34 VCL + 93 aux）健康场景 → @40 判完全双射且无非 @40 误判；损坏场景（VCL ts 仅覆盖 10/18）→ 判失败。`py_compile` 通过。

**剩余生产验收（诊断部分已通过，剩实跑）**：重跑 `--la 8` / `--la 16` 应判定 @40 完全双射（exit 0）——✅ **已实机确认**；实跑 tt6 同参数命令复验段1 frames==packets、frame_num 无回退、v2/v3 全 PASS、无花屏卡顿；段2/3 正常。

## 2026-08-12 生产实机确认：诊断判定修复生效（tt7 日志）

重跑 `python tests/diagnose_lockbitstream_timestamp.py --la 8` 与 `--la 16`：

1. **@40 双射判定通过（exit 0）**：LA=8 → `[u64] 完全双射偏移: [40]`、`[u32] 完全双射偏移: [40]`，18/18 命中；LA=16 → 同上，34/34 命中。报告头 `(VCL=18 辅助块=53)` / `(VCL=34 辅助块=93)` 正确，辅助块行标注 `aux/(aux,skip)` 生效。
2. **辅助块 ts 轮转细节（细化 tt5「实测 ts=0 重复回显」说法）**：辅助块（独立 SPS/PPS/AUD）的 outputTimeStamp 非恒 0，而是**按物理 slot 轮转回显上次占用该 slot 的帧 ts**——LA=8 时 1009..1017（共 6 轮）、LA=16 时 1017..1033（共 7 轮），全部落在已提交 ts 集合内、与 VCL 帧 ts 重叠。因 `_nal_first_vcl_type` 先判块内有无 VCL slice（无 VCL 即缓存参数集后跳过、不进 ts 解析/seen），两种 ts 表现（0 或轮转回显）均被安全处理——这是「先判 VCL 再读 ts」顺序的必要性佐证（若先读 ts，辅助块 ts 会与 VCL 帧 ts 冲突污染 seen）。
3. **ts 重关联 V2 方向最终闭环**：sweep 双射 + VCL 行 `ts-BASE == frame_num` 逐帧一致（全 ✓）+ est_fi/ts-fi 序列均无乱序。
