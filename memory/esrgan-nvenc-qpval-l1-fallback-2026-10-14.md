# ESRGan 侧 NVENC Level 1 因 `_qp_val` 缺失而静默降级（2026-10-14 T4 实机发现）

**一句话**：`external/realesrgan_video/nvenc_sdk.py` 的 `_build_encoder_config` 只把 `_qp_val` 当**局部变量**、
从未写回 `self._qp_val`，而 `_log_ready` 在 **CQ 档位**（`vbr_hq`/`qvbr`，= config 默认）读的是 `self._qp_val`
⇒ `AttributeError` ⇒ 被 ESRGan 管线的 `except` 捕获 ⇒ **ESRGan 侧所有 NVENC + vbr_hq/qvbr 静默从
Level 1 SDK 直通降级到 Level 2 FFmpeg 子进程**，且只留一行 WARN 级日志。

## 症状（T4 真机，h264_nvenc + vbr_hq + LA=8）

```
[NVENCEncoder] VBR_HQ: crf=26 targetQuality(CRF)=26 avgBitrate=100000kbps
[NVENCEncoder] 9 slots created (0x565168995370..0x56517299a180)
[NVENCEncoder] [FIX-ASYNC-COPY] 专用编码拷贝 stream 已创建
[GPU0] NVENC SDK 初始化失败 ('NVENCEncoder' object has no attribute '_qp_val')，降级到 FFmpeg 管道编码
[NVENCEncoder] Encoder closed
[FFmpegWriter] rate_mode=vbr_hq rc-lookahead=8
```

⚠ 注意最后一行：**FFmpeg 侧照样带 `rc-lookahead=8`** ⇒ 从「码率/LA 参数」表面看不出降级，
只有 SDK 直通独有的 `[NVENCEncoder] Ready:` 行消失才能发现。

## 根因：双仓「逐字同构」被单侧重构打破

| 文件 | `_build_encoder_config` 末尾 | `_log_ready` 读法 |
|---|---|---|
| `external/ifrnet_video/nvenc_sdk.py` | `self._qp_val = _qp_val`（:1121，`[P3.1-SPLIT] 供 _log_ready 使用`） | `self._qp_val` ✅ |
| `external/realesrgan_video/nvenc_sdk.py` | **缺失**（本地变量随函数结束丢弃） | `self._qp_val`（:1271）❌ AttributeError |

`ecd60f8`(2026-10-05, B2) 重构了 `_log_ready`（`_eff_rc`/`_eff_la`/`_tq` 按硬件真值回显），
在 IFRNet 侧补了 `self._qp_val` 赋值，**ESRGan 侧漏补** ⇒ 缺陷引入时间点即该次重构。

## 为什么此前所有验证都没抓到

1. **只在 CQ 档位暴露**：`constqp` 走 `_log_ready` 时不读 `self._qp_val`（:1276 用的是局部 `_qp_val`）
   ⇒ 而 10-04 之前的 T4 证据里 ESRGan 侧跑的档位/路径组合恰好绕开了它。
2. **降级是「成功」路径**：Level 2 FFmpeg 能正常出片，S1~S7 全绿，只有日志级别可区分。
3. **静态门禁抓不到**：`plan_implementation_gate` 101 项全 PASS（桩不执行真实 CQ 档位初始化）。
4. **对照面不匹配**：IFRNet 侧一直是 Level 1（无降级行），只看「有无 NVENC SDK 激活」会误判为全链路正常。

## 修复

`external/realesrgan_video/nvenc_sdk.py::_build_encoder_config` 末尾（LA 门控之后、`return preset_config` 之前）
补一行与 IFRNet 孪生体同构的赋值：

```python
self._qp_val = _qp_val
```

## 验证（T4，2026-10-14）

| 项 | 修复前 | 修复后 |
|---|---|---|
| ESRGan Ready 行 | 缺失（降级到 FFmpegWriter） | `Ready: 2560x1440@60.0fps H.264 VBR_HQ QP=26 preset=medium slots=9 la=8 tq(CRF)=26` ✅ |
| `_qp_val` 属性 | `MISSING`（属性访问抛错） | `26` ✅（constqp/vbr_hq/qvbr/vbr/cbr 五档全解析） |
| 4 段跨段冒烟 | — | S1~S7 全 PASS（S2 段级门禁 4/4、S5 帧数守恒/IDR/frame_num/pts 全过） |

## 教训

1. **「逐字同构」是声明不是保证** —— 改任一侧后必须按字段级清单核对另一侧（本次是 `_qp_val` 这类
   「重构新增的 self.X 回读」最容易漏）。
2. **降级路径必须进验收判据** —— 建议把「**产物日志中无 `降级到 FFmpeg` 行**」显式列为检查项，
   否则 Level 1→Level 2 的静默降级在所有「产物质量类」验收里都是隐形的。
3. **探针自身会假 FAIL** —— 本批 B4 探针因「先记录后覆写」的顺序错误，把驱动**已接受**的 `rc_ptr[1]=32/64`
   判成「驱动拒绝」并 exit 3（若照单全收会误触发全套标定重跑）。详见 [[t4-pending-gpu-verification-2026-10-14]]。

## 相关

- [[t4-pending-gpu-verification-2026-10-14]] — 本批 T4 补测结果（B1/B2/B4/plan_gate/drain/reopen/LA=16）
- [[t4-s8-findings-and-blockers]] — B2（Level 1 vbr/cbr 落 CONSTQP）原文与勘误表
- [[level1-nvenc-encoding-flow]] — 四级降级架构（本缺陷即「静默落到 Level 2」类）