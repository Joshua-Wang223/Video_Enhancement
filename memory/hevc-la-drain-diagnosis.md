# HEVC + LA 排空诊断（2026-08-18，验证脚本先行）

## 状态

生产 VBR_HQ + LA=8 + HEVC（NVENC SDK 13.0 / T4）连续三次失败后，用户否决
"HEVC 降级 LA=0"方案，要求参照 `Accessory/probe/nvenc_la_frame_conservation_suite.py`
先写最小化 GPU 验证脚本、跑通后再应用到生产。

## 三次失败记录

1. muxer 参数集独立 AU 改首帧捆绑：`missing picture in access unit with size 86`
   → H.264 的 SPS/PPS 独立预注入与 HEVC parser 不兼容；捆绑修复有效。
2. `_drain_outputs_blocking` 改 doNotWait=1 非阻塞：HEVC 预热后无帧写出
   （编码线程死锁）→ blocking Lock 在辅助块后等"驱动需要更多输入"的帧。
3. 保持非阻塞 + LA+2 槽位：直接 `segmentation fault` → T4 驱动上
   doNotWait=1 对 LA 预热 buffer 段错误；非阻塞方向废弃。
4. HEVC/AV1 强制 LA=0（ce_pipeline）：用户否决（"退而求其次的投降做法"）
   → 必须让 HEVC+LA 真正跑通。
5. 强制 LA=0 路径实跑（output4）：段末 `写线程 30s 未退出` /
   `编码线程 30s 未退出` → LA=0 ce_pipeline 在段末也死锁；根因是 HEVC
   blocking LockBitstream 行为而非 LA 本身。

## output4（LA=0 ce_pipeline 实跑）分析

- T2 跑完 2473/2480 帧（HEVC 编码本身工作），段末 writer/encode 线程
  join 30s 超时。
- 死锁候选（均为无界等待）：
  1. ce_pipeline Phase1/3 的 `cuEventSynchronize`（CE 不触发则永不返回）；
  2. `_lock_bitstream_with_retry` / `_lock_bitstream_blocking(5000)` 的
     blocking LockBitstream（HEVC 驱动对"未就绪"buffer 阻塞不返回 NMI，
     deadline 失效——output2 同根因）；
  3. flush() EOS 排空的逐槽首锁 blocking（同 2）。
- 结论：**HEVC 的"帧未就绪"语义与 H.264 不同**（blocking 不返回 NMI），
  所有依赖"blocking Lock 立即返回 NMI"的设计（LA 流式 / ce_pipeline /
  flush）都有死锁风险；修复方向是"避免对未就绪 buffer 做无界 blocking 等待"。

## 根因分析（当前认知，待 GPU 脚本验证）

H.264 + LA 已验证的排空模式（FIX-AUX-NO-CLEAR / FIFO 记账 / 轮转指针）：
辅助块（VPS/SPS/PPS，无 VCL）不 pop FIFO、不占 fi，指针照常推进；
VCL 块按 `est_fi % slot_count` 查 per-slot FIFO 队首定标签。
tt7 实测：辅助块 outputTimeStamp@40 按物理 slot 轮转回显上次占用帧 ts
（非恒 0），LA=8 时 18 VCL + 53 辅助块。

HEVC 差异假说（待验证）：
- 首帧 VCL 输出比 H.264 晚一个提交周期（LA+2 vs LA+1）→ 阻塞排空死锁；
- 辅助块在轮转中的槽位/位置与 H.264 不同 → FIFO 标签错位；
- 驱动 LA 输出 buffer 内部重路由（pipeline-depth-slot-rotation-confusion）
  在 HEVC 上表现不同。

## 最小化验证脚本

`Accessory/probe/hevc_lookahead_diagnose.py`（参照 nvenc_la_frame_conservation_suite.py 的
MinimalTestEncoder，复用其 SDK 常量/结构体/GUID）：

- 逐 drain 记录 `(slot_idx, est_fi, outputTimeStamp@40, size, NAL 类型, VCL 有无)`
- 每次 blocking LockBitstream 计时（>0.5s 告警）+ watchdog 线程每 2s dump
  主线程栈（ctypes 调用释放 GIL，卡死时可见卡在哪一行）
- 6 种排空策略对比（帧数守恒 = VCL 输出数 == 提交数，含 EOS flush）：
  - `baseline`：生产同构（LA+1 槽，每帧提交后轮转阻塞排空 + FIFO + 辅助块不占帧槽）
  - `delayed`：LA+2 帧内不排空，之后正常排空（验证"首 VCL 晚一个提交周期"假说）
  - `aux_stay`：HEVC 辅助块不推进轮转指针（验证"帧紧跟同一槽"假说）
  - `delayed_aux_stay`：组合
  - `free_pool`：扩容槽位 + 空闲槽池 + ts@40 重关联映射（tt6/tt7 双射验证基础）
  - `ce_pipeline`：LA=0 CE 流水线（生产 ce_pipeline 同构，复现 output4
    段末死锁；Phase1/3 CE 同步 + blocking 取回 + flush EOS 均带耗时诊断）
  - `nonblocking`：修复方向验证——与 baseline 同构但 LockBitstream 用
    doNotWait=1，未就绪帧立即跳过、由后续提交推进后再取回（界定
    output3 segfault 的安全边界）
- `--run-all`：各变体独立子进程 + 超时（防死锁挂死），输出 PASS/FAIL/TIMEOUT
- `--self-test`：无 GPU 自检（NAL 分类 / FIFO / ts 映射 / aux_stay 指针）

生产执行：
```bash
python Accessory/probe/hevc_lookahead_diagnose.py --self-test
python Accessory/probe/hevc_lookahead_diagnose.py --run-all --frames 700 --la-depth 8 --timeout 75
# baseline 单独复现生产（LA+1 槽）：
timeout 60 python Accessory/probe/hevc_lookahead_diagnose.py --variant baseline \
    --frames 700 --la-depth 8 --decode-check
```

## 首轮实跑结果（diagnose_hevc_la.log，2026-08-18）

- `--self-test` 全过；baseline（LA+1 槽）精确复现死锁，watchdog 栈 dump
  卡在 `_drain_rotation → _lock_bs_diag → _LockBS`（blocking LockBitstream），
  与生产 output2 同点。
- **决定性发现 1**：HEVC 没有独立辅助块——首个 drain 块
  `nals=[32,33,34,19]`（VPS/SPS/PPS 与首个 IDR 捆绑），`_nal_first_vcl_type`
  识别为 VCL 后走正常 FIFO 路径。h264 的 FIX-AUX-NO-CLEAR 对 HEVC 不触发。
- **决定性发现 2**：首 9 帧（gfi0-8）在 LA 窗口填满时批量涌出，每提交一帧
  drain 恰好取回一个（ts=fi 全命中）；**gfi9（首个槽位复用帧）在 sub17 后
  未就绪，blocking Lock 永不返回**。HEVC 帧就绪节奏 = 首 9 帧涌出 → 之后
  每帧延迟 LA+1；任何 blocking 等待未就绪帧的排空设计都会死锁。
- 脚本 bug 已修：--run-all 的 TimeoutExpired 分支 bytes/str 拼接崩溃
  （capture_output 未开 text=True），导致 delayed/free_pool/ce_pipeline
  未跑；已修复并新增 nonblocking 变体，待重跑。

## 全变体实跑结果（temp/hevc_la_diag_8/*.log，2026-08-18）

7 变体全部 TIMEOUT，但每个日志都精确显示了挂起点：

| 变体 | 槽 | 挂起点 |
|------|----|--------|
| baseline | 9 | 中途：锁 gfi9（首个槽位复用帧，sub17 后未就绪） |
| delayed | 10 | 中途：gfi0-9 全取回，锁 gfi10 挂起 |
| aux_stay | 10 | 同上（gfi10） |
| delayed_aux_stay | 10 | 同上（gfi10） |
| nonblocking | 9 | drain#1 读到 7.2MB 垃圾（doNotWait=1 SUCCESS+size 竞态）；随后锁 gfi9 连非阻塞也挂起 |
| free_pool | 10 | **中途全程无死锁**（700 提交完 vcl=690），挂在我脚本 EOS 前的末尾排空（已改为 EOS 优先） |
| ce_pipeline | 4 | 700 提交完（4 槽+LA8 单 pending 覆盖导致标签错乱 vcl=688）；挂 EOS flush 首锁 |

**决定性规律（free_pool 与 delayed 数据共同证实）**：
gfi k 的输出在提交 sub(k+LA) 后就绪（LA=8：gfi0@sub8、gfi9@sub17、gfi10@sub18）。
HEVC 驱动对"未就绪 buffer"的 LockBitstream **无论 doNotWait=0/1 都阻塞不返回**；
doNotWait=1 还伴随 SUCCESS+垃圾 size 竞态（7.2MB 误读）→ 非阻塞彻底否决。
free_pool（10 槽）证明"只在帧就绪后取回"的中途排空可行。

## 修复方案（counted 有界排空，2026-08-18 定型）

`_drain_rotation_counted`：每帧提交后最多取回 `submitted - la_depth` 个帧
（已取回 = `_output_slot_idx`），永不锁未就绪帧 → 无死锁；帧数守恒由
EOS 排空兜底。已加入验证脚本（variant=counted）；free_pool 末尾改为
EOS 优先（不再 EOS 前排空未就绪帧）。

待生产验证：counted/free_pool 中途无死锁（预计 PASS），关键看 EOS 排空
（`_flush_eos_diag` blocking 锁）在健康状态下是否安全。若 EOS 也挂起，
说明 HEVC 的 EOS 后仍有未就绪帧，需 EOS 排空也用有界/CE 等待。

生产落地映射：counted 变体 PASS → `encode_frames_stream` 的 per-frame
drain 按 `min(slot_count, submitted-la_depth-drained)` 限界（HEVC 分支，
h264 保持现状）；EOS 排空保持 blocking（若验证安全）。

## 第二轮实跑（diagnose_hevc_la_test2.log + hevc_la_diag_8，2026-08-18）

- **counted（9 槽）中途完美**：700/700 提交，vcl=692，ts=fi 全部命中，
  零挂起 → 有界排空方案中途成立。
- **free_pool（10 槽，EOS-first 修复后）同样中途全通**：vcl=690。
- **两者都挂在 EOS 后 blocking Lock（`_flush_eos_diag` line 519）**：
  健康状态下 EOS 排空也挂起 → EOS 是 HEVC 的最后、也是最核心障碍，
  与中途状态错乱无关（此前 ce_pipeline 的 EOS 挂起结论现在被独立证实）。
- 滞留帧数 = LA 窗口尾帧（counted 剩 8 帧、free_pool 剩 10 帧），
  只能靠 EOS 取回。
- 新增 `eos_probe` 变体：counted 中途 + EOS 后 doNotWait=1 全槽多轮扫描，
  揭示 EOS 后驱动把滞留帧输出到哪些槽、EOS 后非阻塞锁是否安全
  （若安全 → 生产 EOS 排空改非阻塞轮询；若仍挂 → 需 CE 等待方向）。

## 生产修复最终形态（待 EOS 验证）

- 中途：counted 有界排空（HEVC 分支限界 `submitted - la_depth - drained`）。
- EOS：待 eos_probe 定案——非阻塞轮询（若安全）/ CE 等待 / 进一步调查。

## 第五轮实跑（diagnose_hevc_la_test5.log，2026-08-18）——完整通过

```
[eos_probe] 完成：vcl=700 leftover=0 mismatch=0 aux=0
[RESULT] variant=eos_probe submitted=700 vcl=700 conserved=True ordered=True
         ts_hits=8 ts_misses=0 elapsed=0.9s
[decode] ffmpeg decode OK（849833 bytes）
```

HEVC + LA=8 最小验证**完整通过**：counted 有界中途排空 + EOS pending-only
排空（8 个 pending 槽一轮全部取回，跳过空槽），帧数守恒 + ffmpeg 解码 OK。

## 生产补丁已落地（2026-08-18）

`external/ifrnet_video/nvenc_sdk.py` 与 `external/realesrgan_video/nvenc_sdk.py`：
1. **移除 HEVC/AV1 LA=0 钳制**（FIX-HEVC-LA-ROUTE 的降级块与 LA+2 槽位规则），
   HEVC/AV1 + LA 恢复正常走 encode_frames_stream；
2. **FIX-HEVC-COUNTED**：per-frame drain 与 chunk 末尾 drain 对 HEVC/AV1 限界
   `min(_max_drain, max(0, submitted - la_depth - drained))`——只锁已就绪帧；
3. **FIX-HEVC-EOS**：EOS 排空对 HEVC/AV1 只遍历 `_strm_slot_pending`/`_slot_pending`
   的槽（while 条件加 pending 守卫），绝不锁空槽；H.264 保持原 while-True 轮转；
4. **ESRGAN 补齐 `_stream_begin` 对齐**（`_output_slot_idx = _frame_idx` +
   `_strm_ts_base`，对齐 IFRNet），保证限界公式跨段正确。

验证：两文件 py_compile 通过；muxer HEVC 捆绑回归（两侧）通过。

待生产实跑：原命令（--codec-ifrnet libx265 + VBR_HQ + LA=8）应无死锁/崩溃，
ffprobe codec_name=hevc、帧数守恒校验通过；随后补 ESRGAN 侧 libx265 验证。

## 生产实测（2026-08-18 用户确认）——全部通过

原命令（--codec-ifrnet libx265 + VBR_HQ + LA=8）生产实跑无死锁/崩溃，
ffprobe codec_name=hevc、帧数守恒通过。HEVC+LA 修复闭环完成。

## Backport 到 external/IFRNet v6.4.3+（2026-08-18）

六个历史单文件全部移植（H.264 行为零变化）：

| 文件 | 移植内容 |
|------|---------|
| v6.4.5.1 / 4.4.1 / 3.1（stream） | 三 codec 支持（GUID 常量修正 + 运行时匹配 + NAL 三分支 + h264-cfg 门控 + Ready 日志）+ FFmpegMuxer codec 化/首帧捆绑 + FIX-HEVC-COUNTED（per-frame/末尾限界）+ FIX-HEVC-EOS（pending-only）+ 调用链 codec 透传（cache key + Level-1） |
| v6.4.5 / 6.4.4 / 6.4.3（batch） | 三 codec 支持 + FFmpegMuxer codec 化/首帧捆绑 + 调用链透传（CONSTQP-only，无 LA 路径故无 COUNTED/EOS） |

全部 `python -m py_compile` 通过；`_extract_sps_pps` 统一为实例方法 + codec
分支（保留 DIAG-LEVEL）；`_has_sps_pps`/`_nal_first_vcl_type` 由 staticmethod
改为实例方法（调用点 `self.xxx()` 兼容）。

## 第三轮实跑（diagnose_hevc_la_test3.log，2026-08-18）

- counted 中途依旧完美（vcl=692，pending=8）。
- **EOS 发送后，doNotWait=1 的第一个 LockBitstream 也挂起** → 定论：
  HEVC 驱动对「有 pending 但输出未就绪」的 buffer，LockBitstream **无论
  blocking 还是 non-blocking 都阻塞、从不返回 NEED_MORE_INPUT**。
- 推论：中途 drain 能停是因为 counted 预算精确到就绪帧、从未锁未就绪帧；
  EOS 滞留帧（8 帧 LA 尾）未就绪是因为 **EOS 没有触发驱动完成输出**。
- 下一步诊断（已加入 eos_probe）：EOS 帧带 CE + 打印 EncodePicture 返回码
  与 CE 同步耗时（验证 EOS 是否被驱动接受/完成）；EOS 后扫描打印每槽
  status（观察驱动对无输出槽返回什么）；加 `--codec h264` 对照组验证
  脚本 EOS 逻辑本身正确。
- 若 EOS CE 同步也挂 → HEVC 驱动不接受该 EOS 构造（可能需 outputBitstream
  =NULL / 不同 flag），转测 EOS 构造变体。

## 第四轮实跑（diagnose_hevc_la_test4.log，2026-08-18）——根因定论

- HEVC：EOS EncodePicture status=0、CE 同步 rc=0 0.00s（EOS 被接受）；
  EOS 后 doNotWait=1 扫描 **pending 槽 0-6 全部返回尾帧数据**
  （1000-1217B），然后挂在**空槽 slot=7**。
- **h264 对照组完整 PASS**：vcl=700 conserved=True ts_hits=8，
  **空槽返回 SUCCESS size=0 不挂**。
- **最终根因（一句话）**：HEVC 驱动对「当前无输出数据」的槽
  （空槽 / 输出未就绪）的 LockBitstream **永久阻塞、从不返回
  NEED_MORE_INPUT 或 SUCCESS+size=0**；h264 返回 SUCCESS+size=0。
  这统一解释了中途死锁（锁了未就绪帧）、非阻塞死锁（驱动忽略
  doNotWait）、EOS 死锁（扫到空槽）。
- **修复定论**：中途 = counted 有界排空（只锁就绪帧）；EOS = **只锁
  仍有 pending 的槽**（跳过空槽；EOS 后 pending 帧已就绪可直接取回）。
  eos_probe 已改为 pending-only 扫描，待重跑确认 HEVC vcl=700。
- 生产落地：codec-conditional（HEVC/AV1 分支）——encode_frames_stream
  per-frame drain 限界 `submitted - la_depth - drained`；EOS drain 改为
  遍历 `_strm_slot_pending` 的槽（每槽按 pending 条数取回），h264 不变。

## 预期结果 → 生产修复映射

- baseline 死锁、delayed 守恒 → 首 VCL 晚一周期：encode_frames_stream
  预热期延迟首次排空 + LA+2 槽。
- aux_stay 守恒 → 辅助块不占轮转位置：_apply_drained_entries aux 分支
  按 codec 回退指针。
- free_pool 守恒 → 驱动重路由/轮转失配：生产改 free-pool + ts 重关联。
- ce_pipeline 也死锁（output4 已证）→ 根因是 HEVC blocking Lock 行为：
  生产必须消除"对未就绪 buffer 的无界 blocking 等待"——候选：CE 等待
  有界化、中途不排空只留 EOS 排空、或 ts 驱动 + 非阻塞锁（需先确认
  doNotWait=1 的安全边界，output3 的 segfault 仅限 LA 预热 buffer）。
- nonblocking 变体若守恒 → 生产 `_drain_outputs_blocking` 对 HEVC 改
  doNotWait=1 + 由 EOS 兜底帧数守恒（h264 保持阻塞不变）；若 segfault →
  安全边界在"已提交未完成"buffer 上，改用 delayed/free_pool 方案。
- 全部失败 → 继续诊断（dump 完整 drain 序列逐槽分析）。

## 备注

- 非阻塞 LockBitstream（doNotWait=1）在 T4 上对 LA 预热 buffer 段错误，禁用。
- 上一轮生产代码改动（muxer 首帧参数集捆绑 + HEVC/AV1 LA 路由 LA=0 + 逃生门
  NVENC_HEVC_ALLOW_LA=1）保留在代码中；验证脚本跑通后按上表回改。
- 本轮未修改生产 nvenc_sdk.py 的排空逻辑（仍为阻塞 + LA 路由降级）。

## ESRGAN LA=0 flush() 段尾死锁（codec-esrgan-hevc_nvenc_test1.err，2026-08-18）

### 现象

ESRGAN 侧 CLI 请求 `--codec-esrgan hevc_nvenc --rate-mode-esrgan vbr_hq
--lookahead-depth-esrgan 8`，但输出 1536x1152≥1080p 被 main.py 的
`[FIX-HIGHRES-RC]` 降级为 **constqp + LA=0**：

- 段 1：`encode_frames_batch_ce_pipeline` 正常编码 9210 帧（~30fps）；
  段尾 `flush()` 的 EOS 排空按 `_output_slot_idx` **全槽轮转 + 首锁 blocking
  LockBitstream** → HEVC 空槽永久阻塞（根因同 test4 定论）→
  `[NVENC-Enc] ⚠️ 编码线程未在 120s 内退出，可能死锁`（仅告警不中断）。
- 段 2：新建 `_NVENCEncodeThread` 复用同一 encoder session
  （`[FIX-SKIP-REOPEN] gen=1, frame_idx=9210`），但段 1 编码线程仍卡在
  `flush()` 内**持有 `self._lock`** → 段 2 线程在 ce_pipeline 的
  `with self._lock` 处永久阻塞 → writer 队列满 → 流水线冻结在 1%、
  GPU 0%、NV12 tensor 与 CUDA 缓存不释放。

### 与 ifrnet 对比结论

- ifrnet 生产 HEVC 走 LA>0 `encode_frames_stream`（FIX-HEVC-COUNTED /
  FIX-HEVC-EOS），段尾 EOS 在 stream 内完成，**不经过 `flush()`** → 正常
  分段结尾并复用 encoder。
- realesrgan 的 LA>0 `encode_frames_batch` 已移植同款修复；**唯一缺漏是
  LA=0 路径的 `flush()` 仍全槽轮转**（两文件 `flush()` 同构，ifrnet 在
  hevc+LA=0 配置下存在同样潜在死锁，生产路由未触发）。

### 修复（FIX-HEVC-EOS-FLUSH，2026-08-18 已落地）

`external/realesrgan_video/nvenc_sdk.py` 与 `external/ifrnet_video/nvenc_sdk.py`
的 `flush()` EOS 排空增加 codec 守卫（H.264 路径逐字不变）：

1. `_drain_slots = sorted(pending.keys()) if codec in ("hevc","av1") else 原轮转序`；
2. 每槽 while 循环开头 `if hevc and not pending.get(slot): break`（绝不锁空槽）；
3. VCL 帧按 FIFO pop 对应 pending 条目（无 VCL 辅助块不 pop）。

LA=0 段末 pending 为空 → HEVC/AV1 发送 EOS 后直接返回 b""，零次锁槽；
EOS 每段一次 + `_needs_reopen` 簿记不变（FIX-SKIP-REOPEN 语义保持）。

### 诊断脚本扩展（Accessory/probe/hevc_lookahead_diagnose.py）

- `ce_pipeline_fix`：ce_pipeline 同构 + 修复版 flush（`_flush_eos_pending`，
  生产补丁的参考实现）——复现 test1 的 LA=0 段尾路径；
- `multi_segment`：同一 encoder 实例连续 2 段（ce_pipeline + 修复版 flush），
  逐段校验守恒 + 合并 ES decode——覆盖"EOS 排空 → 跨段复用"生产场景；
- 注意：ce_pipeline 流程用局部 pending 列表消费驱动输出，Phase 3 后必须
  `self._slot_pending.clear()`（清 `_submit_frame` 的 4-tuple 记账残留），
  否则修复版 flush 会误判"仍有 pending"而锁已排空槽。

### 验证状态

- 本机（Windows 开发机）：`--self-test` 全过、三文件 `py_compile` 通过。
- 待 Linux 生产机 GPU 实跑：`ce_pipeline_fix`（constqp/vbr_hq + LA=0 +
  4 槽 + decode-check）、`multi_segment`（2 段）、LA>0 回归（counted /
  eos_probe）、h264 回归（baseline）；随后原生产命令 5 段端到端。

## 第六轮实跑（diagnose_hevc_la_test6.log，2026-08-18）——测试模型补全

用户按验证矩阵实跑：ce_pipeline_fix / multi_segment / eos_probe 等全部通过；
**counted（HEVC）与 baseline（H.264）两变体暴露 EOS 排空模型缺陷**：

1. **counted（HEVC）TIMEOUT**：中途 counted 有界排空完美（vcl=692/700），
   但 EOS 仍走 `_flush_eos_diag` 全槽轮转 + 首锁 blocking → 锁空槽挂死
   （watchdog 栈 line 537 `_lock_bs_diag`）。根因：counted 变体只验证了
   中途排空，EOS 排空未同步生产 FIX-HEVC-EOS（pending-only）。
2. **baseline（H.264）conserved=False**：`vcl=699 leftover=1`——EOS 轮转
   排空漏 1 帧。机制：辅助块（SPS/PPS）推进 `_output_slot_idx` 但不 pop
   pending（相位漂移），或尾帧就绪晚于该槽被排空的时刻（该槽先返回 NMI，
   while-True 已 break，之后不再回访）。

### 测试脚本修正（2026-08-18）

`_flush_eos_diag`（baseline/delayed/aux_stay/counted/free_pool 共用）：

- **HEVC/AV1**：开头委托 `_flush_eos_pending`（EOS 后只锁 pending 槽，
  生产 FIX-HEVC-EOS 同构）——counted/free_pool 因此可端到端 PASS；
- **H.264**：保留原轮转排空，末尾新增 `[FIX-EOS-LEFTOVER-SCAN]`——对剩余
  pending 槽做有界多轮 doNotWait=1 扫描（eos_probe 同款已证安全，h264 空槽
  返回 SUCCESS+size=0 不挂），取回轮转漏掉的尾帧；仍取不回则 leftover 照实
  报告（不掩盖真实丢帧）。

生产代码无需改动（生产 flush()/encode_frames_stream 的 FIX-HEVC-EOS-FLUSH /
FIX-HEVC-EOS 即本修正对应的行为）；这是测试模型与生产行为对齐。

## baseline（H.264）EOS 丢帧根因定论（2026-08-18，test6 复跑后）

加入 `[FIX-EOS-LEFTOVER-SCAN]` 后 baseline 复跑**仍 leftover=1**（扫描也取不回）
→ 该帧数据在 EOS 前已永久丢失，非"就绪晚"问题。推演定论：

- 中段驱动重发 SPS/PPS（独立 aux 块，`aux=1`）：`_drain_rotation` 在成功 lock
  时推进 `_output_slot_idx`，但 `_consume_drained` 的 aux 分支不 pop 帧 →
  **相位漂移**：轮转指针超前 1，pending 达 slot_count（LA+1=9）= **槽位全满**；
- 下一次提交复用满槽时，驱动丢弃该槽未读输出（bs_buf 被覆盖/内部队列丢弃），
  该帧在 EOS 前就消失，EOS 轮转 + leftover 扫描均无法取回；
- 生产 H.264 无此问题：`_ensure_slot_free` 提交前强制排空目标槽（带
  直探/prev-fill 兜底），从不同时出现"pending==slot_count 满槽复用"。

### 修复（测试模型对齐生产）

- **aux 不推进轮转指针**（aux_stay 语义推广）：`_consume_drained` 的 aux 分支
  回退 `_output_slot_idx`（原仅 aux_stay/delayed_aux_stay，现含 baseline/
  delayed/counted/eos_probe 全部轮转变体）。pending 恒 ≤ LA（保留 1 个空槽），
  满槽复用不再发生 → 帧数守恒由 EOS pending-only 排空兜底。
- **丢失帧定位诊断**：`_flush_eos_diag` / `_flush_eos_pending` 打印 EOS 前
  `pending_before` gfi 清单、EOS `recovered` 清单、leftover gfi 清单——再失败
  时可精确到帧，不再盲猜。

待 GPU 复跑：baseline（h264）应 `conserved=True` + decode OK，日志可见
`pending_before` 恒为 8 条；若仍 leftover>0，`recovered`/leftover 清单可直接
指出丢失 gfi，再判断驱动行为。

## 第八轮实跑（diagnose_hevc_la_test8.log，2026-08-18）——全矩阵通过

按验证矩阵逐个实跑，**全部 PASS + ffmpeg decode OK**：

| 变体 | 配置 | 结果 |
|------|------|------|
| ce_pipeline_fix | hevc constqp LA=0, 4 槽 | vcl=700 conserved=True |
| ce_pipeline_fix | hevc vbr_hq LA=0, 4 槽 | vcl=700 conserved=True |
| multi_segment | hevc constqp LA=0, 2 段×700 | 每段 700/700，总 1400 conserved=True |
| counted | hevc vbr_hq LA=8 | vcl=700 conserved=True（pending_before=[692..699] 全回收） |
| eos_probe | hevc vbr_hq LA=8 | vcl=700 conserved=True |
| baseline | h264 vbr_hq LA=8 | vcl=700 aux=8 conserved=True（aux 相位漂移修复生效） |
| free_pool | hevc vbr_hq LA=8 | vcl=700 conserved=True |

baseline h264 关键日志：`pending_before=[692,693,...,699] output_idx=692
vcl=692` → `recovered=[692..699]` → `leftover=0`。aux 回退指针后同一 SPS/PPS
块会被重复读 8 次（drain#10-17，aux=8）——驱动在 VCL 就绪前反复返回未消费的
辅助块，属无害计数现象，不影响守恒。

**run-all 默认 LA=8 的误导项已修**：ce_pipeline_fix / multi_segment 是 LA=0
专用变体（FIX-HIGHRES-RC 降级路径），run-all 现在对二者强制 `--la-depth 0`
（multi_segment 追加 `--segments`），不再误报 FAIL(2)/TIMEOUT。baseline/
delayed/aux_stay/nonblocking/ce_pipeline 在 run-all（hevc LA=8）下的 TIMEOUT
为预期行为（旧轮转模型在 HEVC 上的 bug 复现器），free_pool/counted/eos_probe
为通过项。

### 生产代码状态

- realesrgan/ifrnet `flush()` FIX-HEVC-EOS-FLUSH（HEVC/AV1 只锁 pending 槽）
  已落地，H.264 路径零改动；`ce_pipeline_fix` 即该补丁的参考实现，已与生产
  语义对齐并通过 LA=0 全矩阵验证。
- 生产 LA>0 路径（FIX-HEVC-COUNTED + FIX-HEVC-EOS）由 counted/eos_probe 回归
  通过；生产 H.264（_ensure_slot_free 防满槽复用）由 baseline 回归通过。
- 待办：生产端到端实跑（原 test1 命令 5 段，确认段尾无 120s 告警、段 2+ 正常
  复用 encoder、ffprobe HEVC + 帧数守恒）。

## 第九轮实跑（diagnose_hevc_la_test9.log，2026-08-18）——run-all 误报分析与修复

各变体按验证矩阵单独测试全部通过后，跑 `--run-all --frames 700 --la-depth 8
--timeout 60`（全局 hevc+LA=8）又报 baseline/delayed/aux_stay/delayed_aux_stay/
ce_pipeline 五个 `TIMEOUT(deadlock?)`。分析结论：**TIMEOUT 是真实死锁，但属于
“配置不符 + 预期失败未标记”的分析脚本误报**，不是 GPU 行为误判——

1. **run-all 全局一刀切配置**：所有变体都被强制 codec=hevc + LA=8，而各变体
   “单独验证通过”时的配置并不一致——baseline 验证配置是 h264（test8 全矩阵），
   ce_pipeline_fix/multi_segment 是 LA=0 专用（FIX-HIGHRES-RC 降级路径），
   delayed/aux_stay/delayed_aux_stay/ce_pipeline 在 hevc+LA=8 下是旧轮转模型
   “预期死锁复现器”（test9 日志精确卡在 drain#9/#10 与 EOS flush，与 test4
   定论一致），nonblocking 是 doNotWait=1 segfault 复现器。
2. **无期望结果表**：预期复现器死锁与真实回归同列 `TIMEOUT(deadlock?)`，
   run_all 还恒 exit 0，无法当回归门禁。
3. **纯墙钟超时**：test9 机器 PASS 变体耗时约为 test8 两倍（GPU 负载），
   慢速但仍在推进的运行有被误判为死锁的风险（test9 的 TIMEOUT 均提前停在
   drain#9/#10，故本轮确为真死锁，但判定机制不严谨）。

### 脚本修复（Accessory/probe/hevc_lookahead_diagnose.py，2026-08-18）

- `VARIANT_CANONICAL`：run-all 按各变体规范配置运行（baseline→h264/la=8、
  ce_pipeline_fix/multi_segment→hevc/la=0、其余 hevc/la=8），取代全局
  codec/LA 与旧的 LA=0 特判。
- `VARIANT_EXPECTED`：PASS/TIMEOUT/FAIL 期望表；结果与期望不符标记
  `[UNEXPECTED]`，run_all 返回 exit=1（此前恒 0）。
- `--beat` 心跳 + `_run_variant_proc`：子进程 watchdog 每 2s 打印
  `[beat] ... vcl=N`，父进程实时解析；vcl 停滞 `--stall-timeout` 秒才判死锁
  （`TIMEOUT(stall)`，复现器从 60s 提前到 ~20s 终止），持续推进的慢速运行
  不误杀；墙钟 `--timeout` 兜底（`TIMEOUT(wall)`）。
- `--skip-reproducers`：只跑回归 PASS 集（6 变体），输出纯绿矩阵。
- 本机验证：py_compile、`--self-test`、桩 harness（配置下发 / 期望分类 /
  退出码 / 正常-停滞-墙钟三条子进程路径）全部通过；待 Linux GPU 实跑确认。

## 第十轮实跑（diagnose_hevc_la_test10.log，2026-08-18）——修复版 run-all 全通过

Linux GPU 实跑修复版 `--run-all --frames 700 --la-depth 8 --timeout 60`：
**10/11 变体符合期望，1 个 UNEXPECTED → 修正期望表后全绿**。

| 变体 | 结果 | 说明 |
|------|------|------|
| baseline | PASS 7.7s | h264/la=8 规范配置生效，消除 hevc 误报 |
| delayed / aux_stay / delayed_aux_stay | TIMEOUT(stall) ~18.5s | 预期复现器，stall 检测从 60s 提前终止 |
| ce_pipeline | TIMEOUT(stall) 22.6s | 预期复现器（EOS 锁空槽） |
| free_pool / ce_pipeline_fix / multi_segment / counted / eos_probe | PASS 7–12s | 全部守恒 + ES 落盘 |
| nonblocking | TIMEOUT(stall) 27s [UNEXPECTED] | test9 是 segfault(-11)，本轮变为挂起 |

nonblocking 本轮细节：doNotWait=1 读到 **21MB 垃圾块**（SUCCESS+垃圾 size
竞态，首轮 7.2MB 的放大版），`nal_types` 解析耗时 ~8s（watchdog 前三 dump
都落在 line 161），随后 drain#1-9 取回 gfi0-8，锁 gfi9（未就绪）停滞——
与首轮定论“非阻塞也挂起”一致。**期望修正**：nonblocking 的已知失败模式
有两种（segfault -11 / 垃圾误读后挂起），`VARIANT_EXPECTED` 改为
`{"FAIL", "TIMEOUT"}`（集合 = 任一均可）。

### 脚本二次修正（Accessory/probe/hevc_lookahead_diagnose.py，2026-08-18）

- `VARIANT_EXPECTED` 支持集合（`{exp}` 字符串兼容）；nonblocking 期望
  `{"FAIL", "TIMEOUT"}`。
- watchdog beat 模式增加“vcl 停滞即补 dump”（推进后重新武装）：test10
  实测前三 dump 落在 21MB 垃圾解析的 `nal_types`，真正的挂点在之后的
  `_lock_bs_diag`——补 dump 让 hang 证据指向最终卡点。
- 验证：py_compile、`--self-test`、桩 harness（nonblocking 挂起/段错误均
  expected；counted 意外 TIMEOUT 仍 UNEXPECTED→exit=1）全过。

结论：修复版 run-all 的 11 变体矩阵已与各变体单独验证结果一致（6 PASS +
4 预期死锁复现器 + 1 双失败模式复现器），无误报。

## Backport FIX-HEVC-EOS-FLUSH 到 external/IFRNet v6.4.3+ 六历史版本（2026-08-18/19）

用户确认 v6.4.3+ 历史版本同样存在分段尾部 NVENC 线程死锁：段 1 flush() 的 EOS
排空锁空槽永久阻塞并持有 `self._lock` → 段 2 新建编码线程阻塞在
`with self._lock` → writer 队列满 → 冻结 1%、GPU 0%、显存不释放
（HEVC 驱动对无输出数据槽的 LockBitstream 无论 doNotWait=0/1 均永久阻塞，
test4 定论）。

六个历史单文件的 `flush()` EOS 排空补齐 FIX-HEVC-EOS-FLUSH（H.264 路径逐字不变）：

| 文件 | pending 表 | 修复形态 |
|------|-----------|---------|
| v6.4.5.1 | `_strm_slot_pending` | 与模块化修复（external/ifrnet_video/nvenc_sdk.py）逐字一致：`_drain_slots = sorted(pending)`，while 加 pending 守卫，VCL 帧 FIFO pop（辅助块不 pop） |
| v6.4.4.1 / v6.4.3.1 | `_batch_slot_pending` | 同上（stream 系同构，适配各自变量风格） |
| v6.4.5 / v6.4.4 / v6.4.3（batch） | 无（`getattr(self, '_flush_slot_pending', {})` 恒空） | CONSTQP-only、LA=0：ce_pipeline/batch 已同步排空全部提交帧 → EOS 后无滞留帧，HEVC/AV1 零次锁槽；H.264 保持原 `enumerate(self._slots)` 轮转 |

要点：
- v6.4.3/4/5 为 CONSTQP-only（`_la_depth` 恒 0，无 LA/stream 路径），EOS 后
  pending 恒空 → 直接跳过全部槽位即修复，与模块化 LA=0 路径语义一致。
- v6.4.3.1/4.1/5.1 的 LA>0 `encode_frames_stream(send_eos=True)` 已有
  FIX-HEVC-EOS（按 pending 排空）；flush() 补丁覆盖 LA=0 ce_pipeline 路径
  （`_loop` 仅 `if not _la_mode` 调 flush）与 stream 后的残余调用。
- 全部 `python -m py_compile` 通过；模块化参考文件未改动。

生产验证（2026-08-19 用户确认）：六文件 backport 生产实跑通过——HEVC 无
段尾死锁、分段正常复用 encoder、帧数守恒。

## 后续：LA>0 生产就绪（软退役，2026-08-28）

本文件的 counted/EOS 修复栈经水彩综合方案（EOS 排空硬化 + NAL-COMMON +
LockBitstream SizeCap）加固与生产验证后，`hevc_nvenc + VBR_HQ/QVBR + LA=8/16`
生产开放：processor 层 `hevc_la_disable` 软退役（[FIX-HEVC-LA-SOFT-RETIRED]，
命中仅 WARN 不降级）、config 默认翻转 false、verify_plan 新增 FIX-HEVC-LA-OPEN
门禁（2026-08-29：90 项 = 88 PASS / 0 FAIL / 0 WARN / 2 SKIP）、T4 三路对照
（hevc LA=8 / hevc LA=0 / h264 LA=8）verify v4 全绿。
详见 [[hevc-la-open-production]]（时间线）与 [[hevc-la-soft-retired]]（收口记录）。
