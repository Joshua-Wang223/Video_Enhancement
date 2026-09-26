# hevc_nvenc 分段插帧「NVENC 编码器跨段复用」禁用措施 —— 恢复可行性评估与 Plan

> 生成日期：2026-09-02
> 输入：`Plan/history_202609010809.md`、`Plan/history_202609010932.md`
> 状态：**仅分析，未修改任何生产脚本**
> 活跃链路确认：`src/main_video_optimized.py` → `src/processors/ifrnet_processor_video_optimized.py` → `external/ifrnet_video/`（`main.py` / `nvenc_sdk.py` / `pipeline.py`）
> 注意：`external/IFRNet/process_video_v6_*.py` 均为历史影子文件，不是生产路径（会话 2 已确认并修正 `CODEBUDDY.md`）

---

## 一、两份会话高度概括

### 1.1 `history_202609010809.md`（Linux / T4 / 14.6GB，约 7754 行）

**主线问题**：跳过超分直接插帧时「总在最后一段失败」+ 段间挂死 + 段首花屏。

| 阶段 | 内容 |
|---|---|
| 探测 | `wws3e02_26s.mp4`（VFR 有缺陷）超分→插帧成功；单独插帧末段必失败。`new5.mp4`（干净 CFR）多轮复现 |
| 根因 A（VFR） | 源尾部 pts 空洞（delta=1000 vs 正常 500，tbn=1/11988）；`-c copy` 分段把缺陷全落末段；CFR 读帧复制填洞 24→26 帧，而验收门用 `ffprobe -count_frames`=24 → expected=47 vs decoded=51 |
| 修复 A | `external/ifrnet_video/ffmpeg_io.py` `FFmpegFrameReader` 插 `-fps_mode passthrough`（FFmpeg<5 回退 `-vsync 0`）；末段碎片合并 `merge_trailing_fragment()`；源时间轴归一化 `--normalize-source`；分段 timescale 归一化下沉到 `merge_videos_by_codec`（修 FIX-C 覆盖缺口） |
| 根因 B（段首花屏） | f0 的 NV12 kernel 与 NVENC 私有拷贝流之间**无依赖边**（`[FIX-ASYNC-COPY]` NON_BLOCKING）→ 段首帧编码成噪声。硬证据：segment_002 首帧 IDR **374,751 B** vs 干净段 71,737 B（5.2×），SPS 完全一致 → 内容是噪声不是参数集问题 |
| 修复 B | `main.py:1404` `IFRNET_F0_NV12_SYNC` 显式 `torch.cuda.current_stream().synchronize()` |
| 根因 C（段间挂死） | 段首 f0 走 `encode_frame()` → `_drain_outputs_blocking()`，对上一段残留的异常槽 LockBitstream，**T4 驱动单次调用即不返回**（无视 doNotWait）。py-spy 现场：`_drain_outputs_blocking(1385/1398) ← encode_frame(2769) ← _process_segment(main.py:1378)`，GPU 0% / 显存 11183MiB 不回落 |
| 修复 C | `[FIX-F0-ALWAYS-IN-BATCH]` + `[FIX-F0-IN-BATCH-CE]`：f0 一律暂存 `_pending_f0_nv12`，由首个 batch 的 CE pipeline 处理（照搬 realesrgan_video「从不调用 encode_frame」的已验证设计） |
| 根因 D（差 -1.0 帧） | LA=0 下 `encode_frame` 的 drain 遇 `code=8 (INVALID_PARAM)` 直接 break → 该帧写成「仅含参数集」的无效包。补 `[FIX-ENCFRAME-BLKRETRY]` + `[FIX-EMPTY-PREV-FILL]` |

**收尾**：三段验收 697/521/373 全对，全片花屏扫描 SEG0/1/2/FINAL 全干净，segment_002 首帧 IDR 降至 78,330 B，产物统一 `1/90000`，时长差 0.00s。

**被证伪的假设**：
- ❌ constqp vs vbr_hq 是丢帧主因 → 实验 B（vbr_hq+LA=0）同样失败，真因是 LA=0/slots=4
- ❌ `doNotWait=1` 轮询能根治挂死 → 两阶段仍卡在 `lock_bs_fn` 调用内，**T4 驱动无视 doNotWait**
- ❌ `-bsf:v setts` 零成本归零起始偏移 → 产生 `non monotonically increasing dts`，**已明令禁止**
- ❌ new5 三次中断是工具超时 → 实为每次卡在下一分段起始（挂死的回避式绕过）

### 1.2 `history_202609010932.md`（Linux / T4，约 1740 行）

**主线问题**：生产跑 `--codec-ifrnet hevc_nvenc --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8` 时**段 1 永久卡死**，报「写线程 140s 未退出」+ `Output file does not contain any stream`。

| 阶段 | 内容 |
|---|---|
| 误判 | 先按日志定为「写线程死锁」，并基于 AUX 假设落了 2 处改动 → **前提不成立，已回滚** |
| 纠正 | 用户提示活跃链路是 `external/ifrnet_video/` 包；活跃版已有 `P0-FIX-HEVC-DRAIN-HANG` |
| 取证 | 后台 `setsid nohup` 起复现进程 + **py-spy dump --locals**，临时加 `[DIAG-TRACE]`（`IFRNET_NVENC_TRACE=1`）打印 `drain.enter/exit`、`esf.enter`、`pf.drain` |
| 根因 | **物理槽余量为 0**：实测 LA 输出延迟 = `la_depth + 1`，而 `_required_buffers = la_depth + 1` → slot 0 恰在 gfi 0 就绪前一刻被要求复用 → `_ensure_slot_free(0)`「先排空才能提交 / 排空又需要提交」的**逻辑循环依赖**；而 `_ensure_slot_free` 是无条件发起 Lock 的站点 → 驱动内挂死 |
| 证据 | `--locals: _i=9, frame_idx=10, _pending_cnt=10 ⇒ _output_slot_idx=0, _max_drain=2, slot=0`；`esf.enter slot=0 pending=1 frame_idx=9 guard=0,1,2,...N → 挂死` |
| 修复 | `[FIX-LA-SLOT-HEADROOM]` `_required_buffers = la_depth + 2`（LA=8 → slots 9→**10**）；`[FIX-ESF-NO-LOCK-WHEN-EMPTY]` `_ready = _frame_idx - _la_depth - _output_slot_idx`，`_ready <= 0` 时**绝不 Lock**；`[FIX-ESF-PROBE-ADVANCE]` 探测路径按 FIFO 消费量补偿推进 `_output_slot_idx` |
| 回归 | 3 段全通，2 分 48 秒；slots=10；告警 0；帧数 **1591 = 2×797−3**；`ffmpeg -f null` 全片 1591 帧零解码错误 |

**被证伪的假设**：
- ❌ AUX 辅助块推进 `_output_slot_idx` 致记账错位 → HEVC 的 VPS/SPS/PPS 随 IDR 返回（101 B），**无独立辅助块**，日志无「辅助块 #N」
- ❌ 既有 `P0-FIX-HEVC-DRAIN-HANG`（doNotWait=1）能防挂 → 现场 `poll=True` 同样卡在驱动内部，**参数救不了**
- ❌ 写线程是真凶 → 编码线程不抛异常不打日志，`flush_and_join()` 超时告警永不出现

### 1.3 两份会话的合力结论

两条**相互独立**的 HEVC/AV1 挂死触发链，现在都已从源头被切断：

| 触发链 | 触发路径 | 卡点 | 修复 | 现状 |
|---|---|---|---|---|
| **链 A（LA=0）** | 段首 f0 → `encode_frame()` → `_drain_outputs_blocking()` | 锁上一段残留的异常槽 | `[FIX-F0-ALWAYS-IN-BATCH]` + `[FIX-F0-IN-BATCH-CE]` | 已消除，f0 不再走 `encode_frame` |
| **链 B（LA>0）** | `_ensure_slot_free()`（提交前无条件 Lock） | slot 余量为 0 的循环依赖 | `[FIX-LA-SLOT-HEADROOM]` + `[FIX-ESF-NO-LOCK-WHEN-EMPTY]` | 已消除，`_ready<=0` 直接走空帧兜底 |

> **关键旁证**：`external/realesrgan_video/nvenc_sdk.py:3202` `[FIX-SKIP-REOPEN] 跨段复用已验证可行` —— 超分侧已走完「禁用 → 恢复」闭环（删除 `reopen()`，段边界原地复用会话）。插帧侧的修复正是照搬其设计。

---

## 二、禁用措施的代码事实（当前生产态）

**位置**：`external/ifrnet_video/main.py`，`IFRNetVideoProcessor._get_or_create_nvenc_encoder()`

```1094:1120:external/ifrnet_video/main.py
        # [FIX-CODEC-SUPPORT] cache key 加入 codec 维度，防御 h264/hevc/av1 混用复用错误编码器。
        key = (W, H, fps, preset, qp, rate_mode, la_depth, pipeline_depth, codec)
        # [P0-FIX-HEVC-SEGMENT-HANG] HEVC/AV1 禁止跨段复用编码器。
        # 上一段残留的异常槽，会使本段段首首帧 encode_frame()→
        # _drain_outputs_blocking() 对"空/未就绪槽"调用 LockBitstream；T4 驱动
        # 在此状态下单次调用即不返回（无视 doNotWait，py-spy 实测卡在
        # nvenc_sdk 的 lock_bs_fn 调用内），表现为 GPU 0% + 显存高位不回落。
        # 阻塞发生在驱动内部，Python 侧无法中断，唯一可靠规避是让本段使用
        # 全新编码器 —— 这也解释了为何"新会话续跑"从不触发该死锁。
        # H.264 空槽返回 SUCCESS+size=0，无此死锁，保留跨段复用以省初始化开销。
        _force_new = (codec in ("hevc", "av1") and not self._is_first_segment())
        if (self._cached_nvenc_encoder is not None
                and self._cached_nvenc_key == key
                and not _force_new):
            ...
            return self._cached_nvenc_encoder
        if self._cached_nvenc_encoder is not None:
            if not self.quiet and _force_new:
                print(f'   [NVENC] 新建编码器（HEVC/AV1 跨段不复用，规避空槽 LockBitstream 挂死）'
                      f' ({W}x{H}@{fps:.1f}fps codec={codec})', flush=True)
            self._cached_nvenc_encoder.close()
```

**事实清单**：
- 判定条件：`_force_new = (codec in ("hevc","av1")) and not self._is_first_segment()`，`_is_first_segment()` = `self._segment_index <= 1`（`main.py:1082`）
- 命中后：`close()` 旧实例 → 构造全新 `NVENCEncoder`（`main.py:1114-1117`）
- **无配置项、无环境变量开关**，纯硬编码
- 段尾仍是复用语义：`main.py:1490` `[SEGMENT-REUSE] 编码器跨段复用，仅 flush 不 close`（对 H.264 生效）
- 只影响 HEVC/AV1；**H.264 一直跨段复用，长期运行无事故**（这是最重要的对照实验）
- `codec` 值已归一化为 `hevc`/`av1`（`_CODEC_GUID_MAP` 无 `_nvenc` 后缀），判定有效

---

## 三、恢复可行性分析

### 3.1 结论摘要

> **技术上大概率可以恢复，但当前不建议立即恢复。**
> 两条挂死触发链已结构性消除（§1.3），且 H.264 侧的长期复用运行是强对照；但
> ① 收益极小（会话 1 实测禁用状态下 3 段全片仅 1 分 13.5 秒，每段重建是毫秒~百毫秒级，占比 <1%）；
> ② 风险不对称 —— 死在驱动内部，**不可中断、不可超时、不可降级**，只能整段失败；
> ③ 恢复前必须先修一个**独立于本议题的真实缺陷**：`_sps_pps_injected` 只在 `close()` 重置（§3.4 R1）。
>
> 建议定位为「**可恢复、低优先级**」，按 §4 的分阶段 plan 先补缺陷 + 加开关 + A/B 实测，用数据决定是否放开。

### 3.2 分路径可行性

#### 路径 A：LA = 0 → `encode_frames_batch_ce_pipeline`（**当前默认配置** `rate_mode=constqp, lookahead_depth=0`）

**可行性：高（建议优先验证）**

- 段末状态干净：Phase 3 `_ce_final_drain`（`nvenc_sdk.py:2473-2480`）在**每个 batch 内**排空全部 pending slot，`_slot_pending` 是**局部变量**，段间零残留
- `flush()` 对 LA=0 + HEVC 是 no-op：`nvenc_sdk.py:3020-3021` 注释明确「LA=0 路径（ce_pipeline + flush）下 `_strm_slot_pending` 为空 → HEVC/AV1 直接跳过全部槽位，避免锁空槽死锁」
- f0 已进 batch（`main.py:1419` + `nvenc_sdk.py:2391-2397`），**链 A 触发点不存在**
- 段边界 IDR 已就绪：`main.py:1406` `force_idr_f0 = not self._is_first_segment()`
- slot 分配用全局 `_frame_idx % pd`（`nvenc_sdk.py:2423`），跨段连续，与 `_stream_begin` 的 `self._output_slot_idx = self._frame_idx`（`nvenc_sdk.py:1904`）同基准

**唯一不确定**：驱动侧 DPB/LA 状态机在「未 DestroyEncoder 而连续跨段」下的行为，只能实测。

#### 路径 B：LA > 0 → `encode_frames_stream`（`vbr_hq` + `LA=8`，会话 2 的生产配置）

**可行性：中（需更严格验证）**

- 段末 EOS 全槽排空 + `_strm_slot_pending.clear()` + `_strm_active=False`（`nvenc_sdk.py:2347-2348`），并有 `strict_eos` 门禁（`NVENC_STRICT_EOS=1` 默认开，`nvenc_sdk.py:550, 2332-2335`）→ 段末残留**可被检测并 fail-fast**
- `_ensure_slot_free` 已有 `_ready <= 0` 的 Lock 禁令（`nvenc_sdk.py:1655-1660`）→ **链 B 触发点不存在**
- 但 LA 状态机跨段残留的风险高于 LA=0：`_stream_begin` 重置了 `_strm_ts_base` / `_output_slot_idx` / pending（`nvenc_sdk.py:1889-1913`），然而**驱动内部 LA 队列是否随 EOS 完全清空无软件侧可观测手段**
- 会话 2 的修复是在**禁用状态下**（每段新编码器）完成的，3 段通过**不能**证明复用安全

### 3.3 支持恢复的证据

1. **H.264 长期跨段复用零事故** —— 同一份 `_get_or_create_nvenc_encoder` 代码路径，仅 codec 分支不同
2. **ESRGAN 侧 `[FIX-SKIP-REOPEN]` 已生产验证跨段复用可行**（`realesrgan_video/nvenc_sdk.py:3202`），且插帧侧的两个修复正是照搬其设计
3. **两条触发链的根因已被代码级消除**，不是靠时序侥幸
4. **会话 1 的实测数据**：禁用状态下 new5 三阶段 1 分 13.5 秒一次跑完 3 段 —— 说明禁用没有带来吞吐灾难，反过来也说明恢复的收益空间有限

### 3.4 风险清单（恢复前必须逐项处置）

| ID | 风险 | 位置 | 说明 | 处置 |
|---|---|---|---|---|
| **R1** | `_sps_pps_injected` **只在 `close()` 重置**，复用路径下不重置 → 段 2 的新 muxer 不做参数集预注入 | `nvenc_sdk.py:3136`（在 `close()` 内）；`nvenc_sdk.py:1949-1952`、`3422-3426` | 注释写「支持 encoder 复用」，实际恰好落在复用时被跳过的分支里，**注释与实现矛盾**。HEVC 的 `write_sps_pps` 是「缓存进 `_pending_sps_pps` 随首个 VCL 帧捆绑」（`nvenc_sdk.py:3740-3750`），跳过则段 2 首帧只能靠 `_prepend_param_sets`（`nvenc_sdk.py:1916-1924`）兜底。**这是独立于本议题的真实缺陷，无论是否恢复都应修** | 在段边界（`_stream_begin` 或 `_process_segment` 开头）显式重置 `_sps_pps_injected`（并把 `close()` 里那行改为「关闭即弃」语义的普通清理） |
| **R2** | 恢复过程中误引入「段边界 `_frame_idx` 归零」 | — | memory `esrgan-segment-reuse-frame-idx-reset.md`：归零 → `inputTimeStamp` 回退 → 驱动 LA 重排序状态机打乱 → **段 2+ 丢帧 85%** | 铁律：**`_frame_idx` 跨段严格单调，永不重置**；code review 必查 |
| **R3** | LA=0 与 LA>0 对 `_output_slot_idx` 的处理口径不一致 | `nvenc_sdk.py:2415`（每批 `_reset_output_slot_idx(0)`）vs `nvenc_sdk.py:1904`（段边界 `= _frame_idx`） | LA=0 下 `_drain_outputs_blocking` 不被调用（`nvenc_sdk.py:2465` 的 `if self._la_depth > 0`），靠 per-slot CE harvest，指针口径影响小；但恢复后必须复验段 2 首批 | A/B 实测中把「段 2 首批无空帧、无 slot mismatch」列为硬指标 |
| **R4** | 驱动级 DPB / LA 状态机跨段残留 | 驱动内部，不可观测 | LA=8 窗口内帧在 EOS 后是否 100% 回收无软件侧探针 | 用 `Accessory/verify/segment_bitstream_verify_v4.py` 的 frame_num 单调 + 单 IDR 门禁做**段级**验收 |
| **R5** | 死锁不可中断 | ctypes 同步调用卡在驱动内 | `flush_and_join(timeout=120)` 只能判段失败（`nvenc_sdk.py:3486`），无法救活 | 必须有**一行可回滚开关**，禁止无条件放开 |
| **R6** | `doNotWait` 在 T4+HEVC 上**实测无效** | `P0-FIX-HEVC-DRAIN-HANG` | 现有防挂手段是「无效但无害」，不能作为恢复复用的安全依据 | 安全性论证只能建立在「`_ready<=0` 不 Lock」+「slot 余量 +1」上 |
| **R7** | 段级帧数守恒回归 | — | 会话 1 的「差 -1.0 帧」已随 f0 修复消失，但 `output_count` 计数口径（多写 1 个仅含参数集的包）**未根治** | 恢复后重跑段级守恒门禁，frames == packets == 期望值 |
| **R8** | 收益与风险不对称 | — | 每段重建只省一次 `OpenEncodeSessionEx` + slot buffer/event 分配 | 先量化，若 <1% 则维持禁用（见 §4 Phase 2 决策门） |

---

## 四、Plan（分阶段，未执行）

### Phase 0 —— 零风险改动：把硬编码改成可回滚开关（**不改默认行为**）

- `external/ifrnet_video/main.py:1103` 改为：
  ```
  _reuse_env = os.environ.get('IFRNET_NVENC_CROSS_SEGMENT_REUSE', '0')
  _force_new = (codec in ("hevc", "av1")
                and _reuse_env != '1'
                and not self._is_first_segment())
  ```
  默认 `'0'` = 维持现状（禁用），`'1'` = 放开复用。
- 目的：让后续 A/B 实测**不改代码即可双向切换**，且线上出问题可一行回滚。
- 同时把 `main.py:1095-1102` 的注释更新为「触发链 A/B 已修复，禁用降级为保守默认，见 `Plan/plan_hevc_cross_segment_reuse_restore.md`」。

### Phase 1 —— 修 R1（`_sps_pps_injected` 段边界重置）

- 在段边界（推荐 `_stream_begin()`，因为它已是「每段一次」的统一入口）加 `self._sps_pps_injected = False`
- 把 `nvenc_sdk.py:3136` 那行注释改为「close 后对象不再可用，重置仅为语义一致」，消除注释与实现的矛盾
- **这一步无论是否恢复复用都该做** —— 它是 H.264 复用路径下已存在的潜在缺陷，只是被 `_prepend_param_sets` 兜底掩盖了

### Phase 2 —— A/B 实测（量化收益，这是决策的唯一依据）

矩阵（每段 ≥3 段，素材：干净 CFR `new5.mp4` 1280×720 + 缺陷 VFR `wws3e02_26s.mp4`）：

| # | codec | rate_mode | LA | 复用开关 | 关注指标 |
|---|---|---|---|---|---|
| 1 | hevc_nvenc | constqp | 0 | 0（基线） | 基线耗时 / 帧数守恒 / 段级门禁 |
| 2 | hevc_nvenc | constqp | 0 | 1 | 同上 + 段 2+ 首帧 IDR 大小、SPS/PPS 存在性 |
| 3 | hevc_nvenc | vbr_hq | 8 | 0（基线） | + slots 数、frame_num 单调 |
| 4 | hevc_nvenc | vbr_hq | 8 | 1 | 同上，重点看段间是否挂死 |
| 5 | h264_nvenc | constqp | 0 | 1 | 对照（现网即复用），验证 Phase 0/1 无回归 |

硬指标（任一 FAIL 即判该组合不可恢复）：
- 三/多段全部完成，无挂死、无 `flush_and_join` 超时
- 段级 `Accessory/verify/segment_bitstream_verify_v4.py --skip-chroma`：帧守恒 / 单 IDR / frame_num 单调 / pts 全绿
- 全片 `ffmpeg -f null` 零解码错误，总帧数 == 期望（`2n−1` 口径）
- 段 2+ 首帧 IDR 大小落在正常量级（会话 1 判据：~71KB 正常，~375KB = 噪声花屏）
- 每段 SPS/PPS 存在且 extradata 可读（`ffprobe` 能直接读出分辨率）

量化收益：`Δ = (基线耗时 − 复用耗时) / 基线耗时`。

### Phase 3 —— 决策门

| 条件 | 决策 |
|---|---|
| Δ < 1% 或 矩阵 2/4 任一 FAIL | **维持禁用**，把本 Plan 与实测数据归档为 memory，关闭议题 |
| Δ ≥ 1% 且 矩阵 2/4 全 PASS 且 矩阵 5 无回归 | 进入 Phase 4 灰度 |
| 矩阵 2 PASS 但 矩阵 4 FAIL | **只放开 LA=0 路径**（`_force_new` 追加 `and self._la_depth <= 0`），LA>0 继续禁用 |

### Phase 4 —— 灰度落地（仅在 Phase 3 判定放开后）

1. 默认值保持 `'0'`，先在生产以环境变量 `IFRNET_NVENC_CROSS_SEGMENT_REUSE=1` 跑 1~2 个真实任务
2. 保留 `strict_eos=1`（`NVENC_STRICT_EOS`）作为 fail-fast 探针；任一 strict 异常 → 立即回 `'0'`
3. 稳定 ≥3 个生产任务后，再把默认值翻为 `'1'`，并在 `config/default_config.json` 增加注释化说明（不改现有键值语义）
4. 写 memory：`memory/ifrnet-hevc-cross-segment-reuse-restore.md`，更新两侧 `MEMORY.md` 索引（主源 `C:\Users\Administrator\.claude\projects\D--Workspace-Python-Video-Enhancement-Video-Enhancement\memory` → 镜像 `d:\Workspace_Python\Video_Enhancement\Video_Enhancement\memory`）

---

## 五、比「恢复复用」更值得优先做的事（收益更大、风险更低）

按 ROI 排序，均来自两份会话已确认的遗留项：

1. **默认配置处在高危路径**：`config/default_config.json:89-90` 是 `rate_mode=constqp` + `lookahead_depth=0` → slots=4，正是会话 1 反复出问题的组合。会话 2 修复后 `vbr_hq + LA=8`（slots=10）反而更稳。**建议评估是否把默认切到 `vbr_hq + LA=8`**，或至少在配置注释里标注风险。
2. **`config/default_config.json` 配置漂移未查明**：历史 s3 时期是 `vbr_hq`+LA=8、`max_batch_size` 48→36，未找到 OOM 改写 `rate_mode` 的代码。**配置被谁改的仍是未知数**。
3. **ESRGAN 缺 `[FIX-LA-REDRAIN]`**：`memory/realesrgan-missing-la-redrain.md` 标记未修。前置条件：ESRGAN 没有 `_apply_sps_pps()` 统一入口（SPS/PPS 逻辑内联散落约 15 处），需先抽取。
4. **`pipeline.py:1971/1980` 的 `encode_frame()` 降级分支缺流同步**：Level 1 正常时不该命中（`IFRNET_DIAG=1` 实测 new5 3/3 段命中 0 次），但一旦 Level 1 降级就会复现段首噪声花屏。**同类隐患未改**。
5. **`output_count` 计数口径未根治**：多写 1 个「仅含参数集」的包。「差 -1.0 帧」告警消失是症状消失，不是病灶消失。
6. **`doNotWait=1`（`P0-FIX-HEVC-DRAIN-HANG`）注释与实现矛盾待清理**：T4+HEVC 上实测无效但无害，留着会误导后续维护者把它当安全网。

---

## 六、涉及文件清单

**核心（本议题直接涉及）**
- `external/ifrnet_video/main.py` —— `_get_or_create_nvenc_encoder`(1087-1120)、`_is_first_segment`(1082)、f0 入 batch(1406-1420)、段尾复用(1490)
- `external/ifrnet_video/nvenc_sdk.py` —— `_required_buffers`(584)、`_ensure_slot_free`(1636-1719)、`_stream_begin`(1879-1913)、`_apply_sps_pps`(1957-1968)、`encode_frames_stream`(1976-2368)、`encode_frames_batch_ce_pipeline`(2370-2482)、`encode_frame`(2753-2970)、`flush`(2972+)、`close`→`_sps_pps_injected`(3136)、`_NVENCEncodeThread._loop`(3300+)、`write_sps_pps`(3731-3750)

**相关（验证/对照）**
- `external/realesrgan_video/nvenc_sdk.py` —— `[FIX-SKIP-REOPEN]`(2812/3202/3398)、`_stream_begin`(2825)
- `external/ifrnet_video/pipeline.py` —— `_infer_loop` 的 `encode_frame` 降级分支(1971/1980)
- `config/default_config.json` —— ifrnet 段 78-91
- `Accessory/verify/segment_bitstream_verify_v4.py`、`Accessory/probe/hevc_lookahead_diagnose.py`、`Accessory/verify/plan_implementation_gate.py`

**历史（勿改，勿作为生产依据）**
- `external/IFRNet/process_video_v6_*.py` 全系列、`archive/` 下的所有副本

---

## 七、一句话结论

**两条挂死触发链（f0 走 `encode_frame` / `_ensure_slot_free` slot 余量为 0）已分别被 `[FIX-F0-ALWAYS-IN-BATCH]` 与 `[FIX-LA-SLOT-HEADROOM]+[FIX-ESF-NO-LOCK-WHEN-EMPTY]` 从源头消除，HEVC/AV1 跨段复用**技术上可以恢复**；但每段重建的开销占比 <1%，而死锁在驱动内部不可中断，风险收益不对称 —— 建议按「Phase 0 加开关 → Phase 1 修 `_sps_pps_injected` 段边界重置 → Phase 2 A/B 实测 → Phase 3 用数据决策」推进，在拿到实测收益数据前**维持禁用现状**。**
