---
name: realesrgan-missing-la-redrain
description: ESRGAN 缺 [FIX-LA-REDRAIN] 二次排空安全网 —— 代码修复完成（独立 _ce_final_drain）；2026-09-18 P1~P4 全部完成（h264/hevc 生产路由帧守恒双校验、P3 端到端 1149 帧解码级通过、P4 新增 hevc 断言 6 passed）；harness hevc 已 fail-fast 收口
type: project
---

# ESRGAN 侧 `[FIX-LA-REDRAIN]` 二次排空安全网

**状态（2026-09-18，修复完成）**：
前置条件 1 ✅ 已落地｜前置条件 2 ✅ 已落地｜**主体 ✅ 已落地**（独立 `_ce_final_drain()` + 批末调用点，等价 IFRNet v6.4.5.1 line 3059-3145）。

## 未落地部分（为什么）

`[FIX-LA-REDRAIN]` 本体**改变 GPU 运行时语义**（动帧回收时序，可能改变 GOP 结构与
帧序），按 `feedback_no_gpu_work_mode.md` 的约定「只给方案不落地」，等 Linux+GPU
轮次连同 LA=0/LA=8 × h264/hevc 四组合回归一起做。

移植要点（照抄 IFRNet `_ce_final_drain()` 的四点语义）：
1. 门控 `self._la_depth > 0`；
2. IDR 判定用**首 VCL NAL 实测**（`_nal_first_vcl_type`），**不用** `fi == 0`；
3. 回收帧统一走 `_apply_sps_pps`；
4. 只在 `results[_est_fi]` 仍是 `None`/`b""` 占位时覆写，绝不覆盖已有有效数据。

## 实测纠正：立项文档的"约 15 处同模式"假设**不成立**

`external/realesrgan_video/nvenc_sdk.py` 实际有 **11** 个 SPS/PPS 内联阶梯站点，
**只有一个三段式变体**（Phase1 harvest `_prev_idr` / Phase3 drain `_pending_idr` /
encode_frame `force_idr`）与 IFRNet 的 `_apply_sps_pps` 语义**逐字等价**。
其余 8 处差异分三类：

| 类别 | 站点 | 差异 |
|---|---|---|
| ① prev / results 路径（drain、EOS、final-drain） | 1684/1704 附近 | 缓存分支**未按 IDR 门控** → **非 IDR** 首块也会预注入 muxer |
| ② EOS / final-drain 的 prev-chunk 路径 | 2161/2218 附近 | **只 prepend、不缓存** |
| ③ 辅助块（无 VCL）路径 | 1651/2146/2201/2527 附近 | **只缓存、不注入** |

⇒ **「统一入口」对那 8 处不是纯重构，而是行为改动**（要统一必须先判定"哪种行为
才是对的"）。故 2026-09-15 只收敛了 3 处，其余 8 处保留内联并在站点处标
`[P2.4c-LADDER-ESRGAN] 未收敛：…`。**不要**照立项文档的"纯重构+幂等"去整体替换。

## 前置条件 1（已落地）

新增 `NVENCEncoder._apply_sps_pps(h264_data, is_idr) -> bytes`（与 IFRNet 同名方法
同义：预挂 + 缓存 + IDR 时预注入 muxer）。**有意不带** IFRNet `_cache_param_sets`
的 SPS/PPS 字节漂移检测 —— 那是 `[P3-FIX-NAL-COMMON]` 的独立改动，属行为变更。

等价性用 `tests/test_esrgan_apply_sps_pps_equivalence.py` 钉住：以**改造前的内联
阶梯**为 oracle，在 120 组合真值表（h264/hevc × 缓存 3 态 × 数据 5 态 × is_idr 2 ×
muxer 2）上比对 返回数据/缓存值/injected/muxer 调用次数/打印次数 —— 全等。

## 前置条件 2（已落地）

5 个 `_output_slot_idx += 1` 站点各补 `[FIX-LA-OUTPTR-ESRGAN]` 注释，写明"何时允许
推进"（size-cap 强制消费 / 正常取回 / 排空超限兜底 / EOS 排空 ×2），
并显式标出「数据指针为空 → 不推进」的反例。纯注释，零行为改动。

## Why

LA 下 CE 在**入队**时触发而非**完成**时，Phase1/3 会把稍后才完成的帧误标成 `b""`
占位。IFRNet 有 BLKRETRY + REDRAIN 两层兜底，ESRGAN 只有一层；一旦发生，
帧静默丢失**或输出错位**（错位更隐蔽：帧数守恒但内容错位），且现象（少帧）与
原因（LockBitstream 返回码）之间没有可关联证据。

## How to apply

要补主体时按「前置条件 2 已就绪 → 移植 IFRNet `_ce_final_drain` 的 REDRAIN 段 →
`_apply_sps_pps` 补挂参数集 → 四组合回归（LA=0/LA=8 × h264/hevc，看**帧序**不只
看帧数）」推进；LA=0 路径行为必须逐字节不变。

**Related**：[[ifrnet-f0-nv12-async-copy-race]]、[[nvenc-la-frame-conservation-fix]]、
[[ifrnet-la-aux-no-clear-test11-fix]]、[[f0-first-frame-loss-ce-pipeline]]、
[[feedback_no_gpu_work_mode]]

---
## 2026-09-18 修复执行记录

- `external/realesrgan_video/nvenc_sdk.py`: 新增独立 `_ce_final_drain()` 方法（line 2726），包含 Phase 3 剩余 pending 排空 + REDRAIN 二次排空安全网，指针推进语义与 IFRNet `_ce_final_drain()` 完全一致。
- 批末调用点：`encode_frames_batch_ce_pipeline` 已替换内联 REDRAIN 为 `self._ce_final_drain(pd, n_frames, results)`（line 2637），无重复推进/漏推进风险。
- 审计：`AUDIT_REPORT_2026-09-18.md` §5 已同步更新为「完整执行」。
- 镜像：A/B 两侧 `memory/` 已同步（`diff -rq` 一致）。
- 敏感凭据规则：执行过程中未读取 `Video_Enhancement_github_token.txt`。

---
## 2026-09-18 §4.2 回归测试执行记录

执行命令：`python tests/test_regression_min.py --behavior-only`
- 总计：40 项
- 通过：38 项（含 C-行为·NAL等价 13 项、SDK契约 5 项、SPS原语 8 项、指纹 5 项、编译 2 项、配置校验 3 项、调用契约 2 项）
- 失败：0 项
- 警告：0 项
- 跳过：2 项（BEH-B0、BEH-H3 — FATAL: `/proc/sys/crypto/fips_enabled` 只读环境差异，与修复无关）
- 结论：**无功能退化，无新增 FAIL**。修复可安全进入生产环境。

---
## 2026-09-18 P1~P4 真实 GPU 执行记录（完成）

环境：Tesla T4 / 驱动 580.65.06 / CUDA 13.0。

### P1 帧数守恒

**(a) harness** `tests/test_nvenc_la_frame_conservation.py`（自带 `MinimalTestEncoder`，**不导入生产 nvenc_sdk**）：

| 组合 | 结果 | 证据 |
|---|---|---|
| h264 + LA=0 | ✅ PASS | 100 in == 100 out，LockBS=100，EOS=0 |
| h264 + LA=8 | ✅ PASS | 100 in == 100 out，LockBS=92，EOS flush=8 |
| h264 输出解码级复核 | ✅ PASS | 两个输出 `ffprobe -count_frames` 均 =100 |
| hevc + LA=0 / LA=8 | ⛔ harness 结构不支持 | 已加 `[FIX-HEVC-UNSUPPORTED]` fail-fast（详见 [[nvenc-hevc-la-harness-wedge]]） |

**(b) 生产路径** `external/realesrgan_video/nvenc_sdk.py`（正确路由：LA>0 → `encode_frames_batch(..., send_eos=True)`；LA=0 → `encode_frames_batch_ce_pipeline`），300 帧，**编码级 + 解码级双校验**：

| 路由 | h264 | hevc |
|---|---|---|
| LA=8 `encode_frames_batch` | 300==300 / decode 300 | 300==300 / decode 300（重复 2×） |
| LA=0 `ce_pipeline` | 300==300 / decode 300 | 300==300 / decode 300 |

⇒ **h264 与 hevc 在各自生产路由上帧数守恒（编码级 + 解码级）。**

### P2 镜像同步
A/B 各 103 文件，`diff -rq` 逐字节一致；B 可写。本次无删除/改名。

### P3 生产端到端冒烟（默认配置，LA=8）
命令：`python src/main_video_optimized.py -c config/default_config.json --batch-mode --input-dir /tmp/p3_in --output-dir /tmp/p3_out --use-tensorrt-ifrnet --use-tensorrt-esrgan`（输入 640x360、575 帧、25s）。
结果：RC=0；IFRNet `575 → 1149`（×2，LA=8）**解码级验收 `decoded=1149 expected=1149`**；ESRGAN 2x → 1280x720；最终输出 `nb_read_frames=1149`、全解码 rc=0 无错误；时长 25.0s 守恒；`qa.json` fixes 含 `EOS_OUTPUT_ORDER`/`DECODABLE_GATE`。**无 `Bitstream parse error`。**（耗时 8m14s，纯冒烟用时长非性能指标）

### P4 回归断言扩展（已落地并通过）
`tests/test_nvenc_sdk_realesrgan.py::TestFrameConservation` 新增 3 项（该文件**导入生产 nvenc_sdk**）：
- `test_frame_conservation_vbr_hq_la8_send_eos_hevc`（帧守恒 + 零空占位）
- `test_frame_conservation_constqp_la0_hevc`
- `test_frame_conservation_ce_pipeline_la0_hevc`（LA=0 生产入口；LA>0 生产不走该入口）
运行 `pytest ... -k frame_conservation` → **6 passed / 0 failed**。

### 修正与遗留
1. **纠正**：早先「宿主 NVENC 引擎被卡死、需重启容器」为**误判** —— 实为容器 ffmpeg 的 SIGTTOU 假象（`ffmpeg < /dev/null` 后 h264/hevc 全 rc=0）。详见 [[nvenc-hevc-la-harness-wedge]] / [[env-ffmpeg-ffprobe-gotchas]]。
2. **ESRGAN HEVC+LA 就绪门控差异**（`_required_buffers=LA+1`、无 `_hevc_ready_count`，IFRNet 为 LA+3）在生产路径 300 帧实测**未触发**问题；仍属潜在风险（IFRNet 注释称 LA+1 margin 会偶发误锁），保留为待评估项。
3. `encode_frames_batch_ce_pipeline` 在 **LA>0 + HEVC** 下会死锁（`_drain_outputs_blocking`），但生产 LA>0 走 `encode_frames_batch`（`_la_mode` 分支，line 3421/3464-3477），**非生产缺陷**；仅提醒勿误用该组合。
4. **P1 的 harness 不代表生产**：它自带最小编码器；REDRAIN 的生产验证由上面 (b) + P4 覆盖。

---
## 2026-09-18 ESRGAN HEVC/LA 加固 + Harness HEVC 支持（完成）

### A. ESRGAN 生产 `external/realesrgan_video/nvenc_sdk.py`（`[FIX-HEVC-READY]`）
移植 IFRNet 已验证机制，**仅对 hevc/av1 生效**（h264 与 LA=0 逐字不变）：
1. `[FIX-LA-SLOT-HEADROOM]`：hevc/av1 槽数 `max(1, la_depth+3)`（原 LA+1 是 mis-lock 根因）；h264 仍 LA+1。
2. 新增 `_hevc_ready_count(limit)` + `_HEVC_DRAIN_MARGIN`（默认 2，env `NVENC_HEVC_DRAIN_MARGIN` 可覆盖/回滚=0）：
   就绪上界 = `(submitted - la - margin) - drained`；h264/LA=0 原样返回 limit。
3. `_drain_outputs_blocking` 入口集中门控（覆盖所有调用点）；per-frame 与 non-EOS final 两处
   旧 `[FIX-HEVC-COUNTED]` 的 margin=0（off-by-one）改为 `_hevc_ready_count`。
4. `_ensure_slot_free` 提交前门控：就绪不足时**不做任何 Lock**（原会 blocking 探测目标槽 → 永久阻塞），
   直接走占位兜底。
5. **EOS 排空顺序修复**（见 [[nvenc-eos-drain-slot-order-tail-corruption]] 的状态纠正）：
   `encode_frames_batch()` + `flush()` 两处由 `sorted(slot keys)` 改为「轮转序 + pending 过滤」，
   新增 `NVENC_EOS_DEBUG=1`。

**GPU 验证（T4，300 帧）**：hevc LA=8 batch ×2、h264 LA=8 batch、hevc/h264 LA=0 ce
全部 `encode 300==300` + `decode 300`；EOS-DEBUG 实测 `drain_slots=[4,5,6,7,8,9,10,0,1,2]`、
`gfi_seq=[290..299]`、`顺序正确=True`；`pytest -k frame_conservation` → 6 passed。

### B. Harness `tests/test_nvenc_la_frame_conservation.py`（`[FIX-HEVC-READY]`）
按同一套生产架构改造，**HEVC 现已可跑**（详见 [[nvenc-hevc-la-harness-wedge]]）：
槽数 `max(la+3,6)`、per-slot FIFO + `_ensure_slot_free` 背压、就绪门控排空（margin 2/4）、
轮转序非阻塞 EOS、size 钳制、close() 只锁有 pending 的槽、修 `total_need_more` 未初始化；
`[FIX-HEVC-UNSUPPORTED]` fail-fast 已移除。

**GPU 验证**：h264 LA=0/LA=8（100 帧）输出与修复前 **md5 逐字节一致**；
hevc LA=0、LA=8（各 300 帧）`300==300` + `ffprobe -count_frames=300` + `ffmpeg -v error` 零报错。

### 收口
- `tests/verify_plan_implementation.py`：**94 项 / 92 PASS / 0 FAIL / 0 WARN / 2 SKIP**（基线一致）。
- 回滚开关：`NVENC_HEVC_DRAIN_MARGIN=0`（去掉余量）；或整体 revert 本轮改动。
- 未做（有意）：ESRGAN EOS 仍用 blocking+pending 门控（已验证），未移植 IFRNet 的
  non-blocking+deadline 变体，以控制影响面。
