# Video_Enhancement L40 专项执行方案 —— av1_nvenc 等质量标定（M4·L40 侧，**只针对 AV1**）

> **本方案只做 `av1_nvenc`**。共享方法论 / harness 改动 / 优秀做法 / 验收门禁的
> **权威定义在 T4 母版**：`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md`（下称「T4 方案」）。
> 本文件只写 **AV1 差异 + 执行步骤**，共享部分引用，避免两份实现漂移。
> ⚠ L40 虽是 Ada 卡、h264/hevc NVENC 也可用，但**本仓 h264/hevc 基线在 T4 标定**
> ⇒ 本专项**不重复标 h264/hevc**（重复标会引入第二套值）。
>
> **上位文档**：`Plan/PROMPT_等质量换算立项.md`（§0.0 / §7.1 B 组）
> **姊妹方案**：`Plan/Video_Enhancement_质量控制参数修复方案.md`（§7 AC1~AC7 / §8 L40 收口 / §9.6）
> **总览指南**：`Accessory/docs/EQQ_CALIBRATION_OVERVIEW.md`
>
> 状态（2026-10-04 **执行完毕并落表；S8 的 §9.5-D 项已在 T4 执行完毕**）：§1~§7 全部走完。
> **S8 遗留项已用降本方式（D 项）部分收口** —— T4 上 constqp 两臂均无泄漏，
> 但 **h264 的 vbr 对照臂因一个新发现的生产级缺陷（B1）崩溃**，且 **S8 判据本身被证明不可靠**。
> 完整结论、三个阻塞项与记录勘误见 **§0.1**；推进顺序见 §0.1.5。
> **原始采样已存档** `verification_report/s8_20261004_raw/`（五份 mem.tsv + README），
> 后续修复与复算无需再上机。
>
> **落表结果（已提交 `b64c1d2` / `20b9e81`）**
> - `QUALITY_MAP['av1_nvenc'] = (1.4566, 1.2165, 0, 63)`（VE 规范化池 `points/gpu_l40_cq`，17 素材；
>   crf21 → `-cq:v 32`）。⚠ 原并行会话/VU 值 `(1.4573, 1.1022)` 无法由任何 VE 池复现，已改；
>   **VE↔VU 该行分叉** ⇒ 已发 CR-4 handoff 请 VU 同步（见 §12 / `VidUtils/Plan/CR-4_av1_QP轴_handoff_VE_to_VU_20261004.md`）。
> - `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)`（LOO[0,27] 2.61）。
>   ⚠ 实测为**仿射**，**取代**早期只在 ref21 验证的 `_QP_MAP_OVERRIDE` ×3；QP 期望 ref21 → **70**（非 63/71）。
> - 门禁同步：G3-7 CQ32→QP70 / G6-7,8 `-qp 70` / G6-9,10 `-cq 32` / G1-2 av1 `-cq:v 32`（G3-9 size 仍 63）。
>
> **验收（已提交 `95a3f75`）**
> - `crf_cq --quick --no-gpu` **104/0/0/11**；`--gpu` **113/0/3**；`plan_implementation_gate` **94/0/2**；
> - AC1 探针：表值 **`-qp 70` 落带内 PASS**（1.07× / −1.13 dB）——「×3(63)」结论已被仿射表取代；
> - ✅ 工具口径已统一（2026-10-05，`[E-EXIT-CODE]`）：B 组 QP 扫描点是**诊断探针**、
>   不是通过性判据（21/84/105 本就有意带外，用于反证旧 ×1/×4/×5 假设），故拆出
>   `n_scan_fail` **不计退出码**；`n_fail` 只计 A 组/AC 判据 ⇒ AC1 落带内 PASS 时
>   工具 exit 0，与 §5.4「退出码 0」一致。诊断计数仍打印并在 JSON 的
>   `summary.scan_fail_diagnostic` 保留，不丢信息。
>
> **门禁口径修订（本次新增，影响全部编码器）**
> - `eqq_pool_fit_table` 新增 `GATE_ANCHORS = [0,27]`（生产工作区间）：判据只看 `LOO[0,27]`，
>   全锚点 worst 降级为**监控列**（不计 FAIL）。
> - 依据：全 8 档实测 worst **对每个编码器都来自最高锚点 crf30**（逐锚点跨素材最差 0.94/1.93/2.91/3.25/**6.41**），
>   属门禁边界锚点等权的系统偏差，非换算缺陷。av1 CQ 判据 LOO 由 6.21 → **3.13 ✅**。
> - 前提：生产 `crf_ref` 不用 >27（仓主确认）；该前提改变须恢复全区间或改稳健统计量。
>
> **L40-5 长视频冒烟（358.8s 真实素材）**
> - `[FIX-S3-STAGE-DEDUP]`：两阶段管线各校验一次同批分段致 S3 假 FAIL（脚本 double-count）；
>   修复后复跑 **S3 ✅ 17926 = 产物帧数**；S1/S2/S4/S5/S6/S7 全过。
> - ❌ **S8（constqp）复现 FAIL**：空闲机斜率 **+149.5 MB/min**（vbr −21.4 通过）⇒ 非并发污染，
>   疑 constqp 路径真实增长。`MemWatcher` 为进程树 RSS 求和且未落盘进程数 ⇒ 待增强采样定位。
>   **与本专项换算正确性无关**（未改管线代码），作为**遗留项**单列。
>
> **S8 的无 GPU 准备项（2026-10-04 已完成，静态审阅 + 采样增强）**
> - **静态审阅（排除性结论）**：① AV1 走 **SDK 直通**，ffmpeg 只做 muxer ⇒ CLI 的
>   `-rc-lookahead`/`-b:v 0` 差异**排除**为累积源；② 两次跑的真正差异是
>   **两条不同代码路径**（constqp→LA=0→`encode_frames_batch_ce_pipeline` per-batch；
>   vbr→LA=8→`encode_frames_stream` 分块累积），**不是同一 encoder 的两种 RC 配置**；
>   ③ `_strm_slot_pending`/`_slot_pending`/`_cached_sps_pps`/`results`/`_slots`/`_la_pinned_pool`
>   **均确认有界**；④ AV1/HEVC **每段强制新建编码器** ⇒ 跨段累积结构性排除；
>   ⑤ 头号候选（per-frame CUDA event 在 `cuEventSynchronize` 失败 raise 前跳过销毁）
>   **被 rc=0 证伪**（该 raise 会置编码线程 error ⇒ rc≠0，而实测 S1 rc=0 ⇒ 从未触发）。
>   ⚠ **没有任何一条能正面解释 +150 MB/min** —— 下次上机的价值是用增强采样**直接定位归属**，
>   不是继续静态猜。详见 `memory/av1-nvenc-l40-calibration.md`。
> - **采样增强已落地**（`Accessory/verify/av1_pipeline_smoke.py`）：
>   `[FIX-S8-ATTRIB]` 保留并落盘**进程数 `n`** + **主进程/子进程分组 RSS 与 PSS** +
>   逐进程明细（`--mem-dump-dir` → `<rate_mode>.mem.tsv`，可离线重分析）；
>   显存改 `--query-compute-apps=pid` **按 pid 归属**（整卡值在共享主机上会污染）。
>   `[FIX-S8-CRITERIA]` 判据 = **斜率（口径不变，与 +149.5/−21.4 可比）+ 峰值上界
>   （`--mem-peak-mb` 默认 12000）+ 样本充分性（`--mem-min-samples` 默认 8，不足报 SKIP）**；
>   斜率超阈值时 detail 直接附主/子分组斜率。
> - **CPU 自测已过**：合成泄漏子进程被检出 +1887.9 MB/min 且分组归因精确
>   （`main +1888.0 / child −0.1`）；静止进程组 0.36 MB/min（噪声量级）⇒ 无误报；
>   `plan_implementation_gate` 84 项 **0 失败**。
>   ⚠ 唯一未被 CPU 覆盖的点：`--query-compute-apps` 在本容器无 GPU 时返回空（显存记 0），
>   **显存归属判定须在 L40 复验**（⚠ 该归因已被 T4 实测推翻，见下方 §0.1）。
> - **下次上机的一条命令**（读 detail 即得归属，无需二次跑）：
>   ```bash
>   python3 Accessory/verify/av1_pipeline_smoke.py --src <330s+真实素材> \
>       --rate-modes constqp --mem-interval 5 --mem-dump-dir /tmp/s8_mem < /dev/null
>   ```
>
> **⚠ 上面「显存字段为 0 ⇒ 无 GPU 挂载」的归因已被推翻**（2026-10-04 T4 实测，见 §0.1）：
> 真因是**容器 PID namespace 与驱动侧不通**，与 GPU 是否挂载无关。
>
> **未做（本方案范围外/条件未触发）**：AC5（av1_qsv/amf，需 Intel/AMD）；L40-6（AV1 Level 1 `code=12`，可选）；
> AV1 `-tune/-multipass` 同 T4 走显式 opt-in（AV1 本就 plain `vbr`，无附加项）。

---

## 0.1 §9.5-D 项已在 T4 执行完毕（2026-10-04）——S8 收口 + 三个新阻塞项

> **执行环境**：本容器是 **Tesla T4（sm75）**，非 L40 ⇒ 走 §9.5 的 **D 项（降本验证）**。
> 素材 `01 the race to mystery island.fixed.mp4`（358.76s / 720×576 / 12 段 / 8969 源帧）。
> **脚本改动**：`av1_pipeline_smoke.py` 新增 `--codec`（默认 `av1_nvenc` 行为不变）与
> `--batch-size`（默认 8）。`plan_implementation_gate` **96 项 / 94 通过 / 0 失败**。

### 0.1.1 S8 结论（措辞已收窄，勿越界引用）

| 臂 | codec / rate_mode | 路径 | S8 后半程斜率 | 全程斜率 | 峰值 RSS | 结果 |
|---|---|---|---|---|---|---|
| 1 | h264_nvenc constqp | LA=0 / `ce_pipeline` | **−55.6** | — | 12393 MB | S1~S7 全 PASS |
| 2 | h264_nvenc vbr | （实为 CONSTQP） | — | — | — | **rc=1 @61s，片段 2 崩** |
| 3 | h264_nvenc vbr_hq | LA=8 / `encode_frames_stream` | — | — | — | **rc=1 @62s，片段 2 崩** |
| 4 | hevc_nvenc constqp | LA=0 / `ce_pipeline` | **+3.1** | **+149.4** | 13877 MB | S1~S7 全 PASS |
| 5 | hevc_nvenc vbr_hq | LA=8 / `encode_frames_stream` | **+35.8** | +63.8 | 13578 MB | S1~S7 全 PASS |

> **能收口的**：T4 上**否证了「constqp 路径泄漏」** —— 两条走完全片的 constqp 臂（h264 / hevc，
> 两种跨段复用模式）后半程斜率分别为 −55.6 与 +3.1 MB/min，均落在噪声内
> （后半程 sd 1356 MB ⇒ 不确定度约 ±66 MB/min）。
> **不能收口的**：**h264 的 vbr 对照臂（臂 2/3）因独立缺陷崩溃**（有效样本仅 12 点、
> 全在启动爬坡段 ⇒ 复算斜率会得 +4585.8/+1872.4 的启动伪影，**不是趋势**），
> **A/B 对照只在 hevc 上成立**；且 S8 原始问题（「L40 上 +149.5 是否 AV1 特有」）**仍未回答**。

> **原始采样已存档**：`verification_report/s8_20261004_raw/`（五份 `.mem.tsv` + README
> 含口径定义、三条使用纪律、失效字段说明）⇒ **后续修复/复算不必再花一次 GPU 上机**。

### 0.1.2 ⚠ S8 判据本身不可靠（本轮最重要的方法论产出）

同一份 HEVC constqp 数据、只换 OLS 窗口：

| 窗口 | 斜率 |
|---|---|
| 前 10%（爬坡） | **+1434.2** |
| 后 50%（S8 判据口径） | +3.1 |
| **全程** | **+149.4** |
| 剔除前 40%（稳态） | +36.1 |
| 后 25% | −240.5 |
| 90–100% 段 | **−786.7** |

⇒ **「全程 +149.4」与 L40 记录的「+149.5」几乎相同**，而同一信号换窗口可得 −786 ~ +1434
⇒ **强烈提示 L40 那个数可能是锯齿信号的窗口伪影，而非真泄漏**。
⚠ 根因之一是 **pinned result pool 随 batch_size 线性增长**（实测 bs=24 时逐段
305→549→794→1038→1190→**1251 MB**），这正是锯齿的幅度来源 ⇒ 已把冒烟默认 `--batch-size`
改为 **8**。**跨 batch_size 的 S8 读数不可比。**

**bs=8 vs bs=24 实测**（T4，40s 段，插帧阶段，2 轮交替 A/B，两轮各差 <0.5%）：

| bs | 墙钟（中位） | 单批 ms | pinned pool |
|---|---|---|---|
| 24（config 默认） | 33.45 s | 343 / 363 | 305 MB |
| **8** | **29.14 s（−12.9%）** | **90 / 90** | **102 MB** |

⇒ **双重收益**：吞吐更高 + S8 斜率噪声更小。这只是插帧阶段；超分阶段 pinned 池同样
按 bs 线性，完整两阶段收益应更大（**未实测，待补**）。

⚠ **S8 的 `--mem-peak-mb` 默认 12000 已被实测否决**：T4 两臂峰值 13578~13877 MB
**超上界但斜率正常** ⇒ 峰值上界比斜率判据更容易误报，需按素材/卡型重新标定。
**已于 2026-10-05 重标定为 16000**（= 最坏实测 13877×1.15）；复算确认三条已知无泄漏臂
从「假FAIL」转为 PASS，合成 +120 MB/min 泄漏仍 FAIL。详见 §0.2。

---

## 0.2 ⚠️ B4（**初版断言已撤回，待 GPU 核实**）：`rc_ptr[1]` 的 RC 枚举值

> **状态（2026-10-05 二轮自查）**：本节初版断言「32/64 是非法枚举」，**该断言已撤回**。
> 依据的 `nvEncodeAPI.h` 后来被判定为**按 FFmpeg 需求裁剪的子集**（证据见 §0.2.3），
> 不能作为「完整枚举全集」。**在GPU 核实前，生产代码不动。**
> **B2 分级策略不受影响** —— 其依据是本仓代码内部逻辑，与外部枚举无关。

### 0.2.1 待核实的事实

`external/ifrnet_video/nvenc_sdk.py:_build_encoder_config` 写入 `rc_ptr[1]`：

| 内部 rate_mode | 写入值 | 状态 |
|---|---|---|
| `constqp` | 0 | 恒定 QP，各来源一致 ✅ |
| `vbr_hq` | **32** | **待 GPU 核实** ⚠ |
| `qvbr` | **64** | **待 GPU 核实** ⚠ |

### 0.2.2 现有证据（**不足以定论**）

| 来源 | 说法 | 权重 |
|---|---|---|
| **T4 实机 GPU 直通 ctypes 实测** | 仓库 memory 记载：`rc_ptr[1]=32` 被驱动接受，且 vbr_hq/constqp/qvbr **三者输出互异**（非静默钳制） | ✅ **运行时事实，最高** |
| 本仓 memory `nvenc_ctypes_verified_layouts.md:78` | `0=CONSTQP, 32=VBR_HQ, 64=QVBR` | 记录与代码一致，原始出处未复核 |
| `Accessory/probe/nvenc_vbr_hq_offsets_probe.py:17,532` | 注释写 `CONSTQP(0)/VBR_HQ(4)/QVBR(32)` | ⚠ **与上一条互相矛盾**（4/32 vs 32/64） |
| FFmpeg `nv-codec-headers` 的 `nvEncodeAPI.h` | 只有 `CONSTQP=0/VBR=1/CBR=2` | ❌ **已判定为裁剪子集，不可作证据** |

⚠ 仓库内两处记录**互相矛盾**（4/32 vs 32/64），这本身就说明该数值**缺乏可靠单一真源**，
必须由驱动 runtime 裁决。

### 0.2.3 ⚠ 初版断言为何不成立（裁剪证据）

实测 `FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h`
（`NVENCAPI_MAJOR_VERSION 13`）：

| 检验项 | 结果 | 含义 |
|---|---|---|
| `grep -cE "VBR_HQ\|QVBR"` | **0** | 完整 SDK 头文件**必然**含这两个模式 |
| RC 枚举项数 | 恰为 3（CONSTQP/VBR/CBR） | 与 FFmpeg `nvenc.c` 实际用到的**完全相同** |
| `NV_ENC_PIC_PARAMS_V2` / `NvEncGetEncodeVersion` | 0 命中 | 完整 SDK 应有 |
| `NVENCAPI_SUBMINOR_VERSION` | 0 命中 | 版本宏不完整 |

⇒ 该文件是**按 FFmpeg 需求裁剪的子集**（FFmpeg 只用 CONSTQP/VBR/CBR）。
**「我查到的文件里没有 32/64」≠「枚举里不存在 32/64」** —— 初版把前者当成了后者。

### 0.2.4 仍然成立的部分（**不依赖外部枚举**）

1. **B2 根因链**（本仓代码内部逻辑，已逐行核实，与枚举是否合法无关）：
   `:1005/:1025/:1056` 只对 `vbr_hq`/`qvbr` 写非 0 值；`:1069` LA 门控
   `in ('vbr_hq','qvbr')`；`:634` 只在 `rate_mode=='constqp'` 时清 `la_depth`
   ⇒ `vbr`/`cbr` 落 `else` 兜底（写 0）且 LA 不使能，而 Python 侧 `self._la_depth` 仍为 8。
   **这是「Python 意图」与「代码实际写入」不一致，与枚举语义无关。**
2. **T4 三模式输出互异**：说明驱动确实区别对待这三个值。
   ⚠ 但这只证明「32 与 0/64 行为不同」，**不证明「32 的语义 == VBR_HQ」**。
3. **CQ 由 targetQuality 承载**：NVENC 文档中 CQ 配合 RC 使用（本仓也把 `targetQuality`
   写在 vbr_hq 分支）。若核实结果为「32/64 非 VBR_HQ/QVBR」，则该组合需改为
   `RC_VBR(1) + targetQuality`；若确认 32/64 是厂商扩展的 VBR_HQ/QVBR，则现状正确。
   **⇒ 两种可能都需 GPU 定夺，代码暂不动。**

### 0.2.5 GPU 核实步骤（一条命令，~30 秒）

```bash
cd /workspace/Video_Enhancement
python3 Accessory/probe/nvenc_rc_enum_truth.py --caps-only --verbose < /dev/null
# 退出码：0=全部合法 / 3=发现非法 / 2=环境不成立
```

探针用**官方 caps API** `NvEncGetEncodeCaps(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES)`
问驱动「支持哪些 RC 模式」。官方文档明确该 caps 返回值是 `NV_ENC_PARAMS_RC_MODE`
值的**位掩码** ⇒ 若返回值含 `bit5`/`bit6`，则 32/64 是该驱动**声明支持**的模式，
初版断言彻底推翻、生产代码正确。

### 0.2.6 判读表（**以 runtime 事实为准**）

| caps 观测 | 结论 | 动作 |
|---|---|---|
| 含 `bit5(32)` | 驱动**声明支持** 32 | **推翻初版**；核对是否即 VBR_HQ |
| 含 `bit6(64)` | 驱动声明支持 64 | 同上（可能为 QVBR 或扩展） |
| 只含 `bit0/1/2` | 仅支持 CONSTQP/VBR/CBR | 32/64 为未定义值 ⇒ 评估改 `RC_VBR(1)+targetQuality` |
| caps 查询失败 | 该 API 不可用 | 退回 `--try-values 0,1,2,32,64` 实编 + 读驱动实际行为 |

### 0.2.7 若确认 32/64 非法时的备选（**当前不实施**）

修法：`vbr_hq` 分支改写 `rc_ptr[1]=1`（`NV_ENC_PARAMS_RC_VBR`）+ 保留 `targetQuality`
+ `averageBitRate`。
⚠ 属**改变 GPU 运行时语义**的改动，且需**重跑全套等质量标定**（CQ 轴结论可能变化）
⇒ 按工作约定**只给方案不落地**，等 caps 结论出来再决策。

### 0.2.8 本节教训（方法论）

初版犯的错是**拿裁剪版头文件当权威全集**，并把「我查到的枚举里没有」表述成
「枚举里不存在」。两者差别很大：
- 「我查的文件里没有」—— 只证明**该文件**没有；
- 「枚举里不存在」—— 需要**全集**证据。

⚠ 仓库内已有同类教训（`memory/feedback_static_review_falsifiable.md`：
「静态审阅的『确认』只是假设」）。本次补充的判据：
**引用外部权威源前，先验证该源是否完整** —— 判据是「该源的使用者需要什么」
与「该源包含什么」是否恰好一致（此处：FFmpeg 只需 3 个 RC 模式，头文件就恰含这 3 个
⇒ 裁剪）。这也解释了为何 T4 的 runtime 实测比任何静态查表更可信。

### 0.1.3 三个新阻塞项（独立于 S8，其中 B1 为生产级缺陷）

| ID | 问题 | 根因 | 影响面 |
|---|---|---|---|
| **B1** | **h264 + LA>0 + 跨段复用 ⇒ 片段 2 起必崩**（`non-existing PPS 0 referenced` → muxer pipe broken → rc=1） | `ffmpeg_io.py:163` `nvenc_map={'libx264':'h264_nvenc'}` ⇒ config 默认 `libx264` **在 T4 上就升级为 h264_nvenc**；`main.py:1155` `_force_new` 只含 `("hevc","av1")` ⇒ h264 复用编码器；`nvenc_sdk.py:2282/2285` `_stream_begin` 每段清 `_cached_sps_pps=None`；`nvenc_sdk.py:971` `repeatSPSPPS bit12 不写` ⇒ 驱动不重吐参数集；`_prepend_param_sets`(`:2297`) 与 `_drain_write`(`:3927`) 两条通道同时失效 | **生产默认路径**（T4 + 默认 config + 多段视频）；回归点 `1e57c0b`(2026-09-18)；ESRGAN 侧 `realesrgan_video/nvenc_sdk.py` 逐字同构 |
| **B2** | `--rate-mode vbr/cbr` 在 Level 1 **静默落 CONSTQP** 且 LA 未启用 | `nvenc_sdk.py:1055` `else` 兜底写 `rc_ptr[1]=0`；`:1069` LA 门控 `in ('vbr_hq','qvbr')` 排除 vbr；`:634` 清 LA 判据是 `== 'constqp'` ⇒ Python 认为 LA=8 而硬件 CONSTQP+LA=0 | 额外三点（**既有 memory 未记录**）：① `[FIX-CONSTQP-FRAME-CE]` 守卫 `('constqp',0)` 不命中 ⇒ **绕过 per-frame completionEvent**，正落在 `nvenc-drain-unsubmitted-slot-segfault.md` 记载的 T4 崩溃组合上；② QP 未过 `to_constqp_qp` 换算 ⇒ 画质偏松（实测 h264 Ready `QP=22` vs vbr `QP=26`）；③ `nvenc_sdk.py:568` AV1 自动降级是**无需用户传参**的第三入口 |
| **B3** | 显存 `gpu_tree_mib` **恒为 0（静默假零）** | `nvidia-smi --query-compute-apps` 返回**宿主 pid**（实测 725115 / 1071006），在容器 `/proc` 下**不存在**；`/proc/<pid>/status` 的 `NSpid` 只有一层 ⇒ 容器 PID namespace 与驱动侧不通，`nvenc_sdk.py` 的 pid 查表永远 miss | **仅污染 S8 的显存维度**，RSS/PSS 口径正确。⚠ **推翻** memory 里「本容器无 GPU 挂载⇒显存记 0」的旧归因（GPU 实际满载 7698 MiB/100%）。0.0 会被读成「确实没用显存」，`None` 才能表达「测不到」 |

### 0.1.4 记录勘误（E1–E5）

| # | 需更正条目 | 更正内容 |
|---|---|---|
| E1 | `memory/nvenc-sps-pps-debugging.md:42`「多段视频段 2+ 不再报 `non-existing PPS`」 | 该结论在 **h264 + LA>0 + 跨段复用** 下**已不成立**（本轮臂 2/3 段 2 必现）。成立范围仅 LA=0 路径或 hevc/av1 每段新建编码器路径 |
| E2 | `memory/sps-pps-la-pipe4-startup-corruption.md:19,:92` | 同 E1 |
| E3 | `memory/ifrnet-multisegment-la-strict-drain.md`（提交 `1e57c0b`） | 该条只记「段内 fi→全局 gfi 错位」与「HEVC EOS 漏帧」，**未记同提交引入的 SPS/PPS 缓存清空缺陷**（B1，机制不同但同源） |
| E4 | `memory/t4-vbrhq-verification-plan.md:43` 行号 | `1044/1056` 已漂移为 **`1055/1069`**（`be8e1db` 插入 11 行注释） |
| E5 | `memory/t4-vbrhq-verification-plan.md:43` 内容完整度 | 缺三点：QP 未换算、`[FIX-CONSTQP-FRAME-CE]` 被绕过、AV1 自动降级 `:568` 第三入口 |

### 0.1.5 下一步（顺序有依赖，勿打乱）

> **状态更新（2026-10-05）**：**全部 CPU 待办已完成**（B2-A/B3/mem-peak/M3/M4/L1/L3/L4/L5/L6/E）；
> 步 2/3 仍被 B1 阻塞（需 GPU）。
> 新增 **M3/M4**（探针 `nvenc_rc_enum_truth.py` 存在三处断裂，未修前 §0.2.5 的
> B4 裁决命令**不可执行**，须先修）。逐步骤归属见下表「归属」列。

| 步 | 动作 | 需 GPU | 归属 | 状态 |
|---|---|---|---|---|
| 1 | **修 B3**（显存口径：`None` 而非 `0.0` + 标注「不可归属」） | 否 | CPU | ✅ 已完成（`[FIX-B3-GPU-UNATTRIBUTABLE]`，四分支实测） |
| 5 | 标定 `--mem-peak-mb`（按 T4 实测 13578~13877 重新取值） | 否 | CPU | ✅ 已完成（12000→**16000**，复算：三条干净臂假FAIL→PASS，合成泄漏仍 FAIL） |
| 4 | B2 方案 A（分级禁用 `vbr`/`cbr` + 修 `_log_ready` 自相矛盾回显） | A 否 / B 是 | CPU | ✅ 已完成（分级：nvenc非av1 硬拒 / libx264·libx265 仅告警 / av1 放行；13/13 矩阵） |
| **M3** | **修探针四处断裂**：`--caps-only` 极性反了；`_FUNC_IDX` 缺 `GetEncodeCaps`；`_NvEncCapsParam` 未定义；caps 序号从未校验 | 否 | CPU | ✅ 已完成（键名 `GetEncodeCaps`=index 7；结构按权威头文件 3 字段=256B；序号 **1** 经60 项枚举解析 + 运行时自检） |
| **M4** | **`--try-values` 必须回报实际写入的 `rc_ptr[1]`**，否则 `--try-values 32/64` 实走 constqp(0)、exit 0 是假 PASS | 否 | CPU | ✅ 已完成（monkeypatch 写值点 + **回读校验**，不符即 FAIL） |
| **0** |跑 §0.2.5 **RC 枚举真值探针**（B4 裁决） | 是 | T4 或 L40 | ⛔ 被 M3/M4 阻塞 |
| **L1** | 跳过阶段（`--skip-*`）不参与硬拒 | 否 | CPU | ✅ 已完成（`[L1-SKIP-STAGE]`，用 `getattr` 兼容缺省 Namespace） |
| **L3** | 第二轮循环节名 `esrgan` → `realesggan` | 否 | CPU | ✅ 已完成（`[L3-SECTION-NAME]`；拆成 `_cfg_sect`/`_sfx`，**CLI 后缀仍 esrgan**） |
| **L4** | 统一 codec 键口径 | 否 | CPU | ✅ 已完成（`[L4-CODEC-KEY]`，三处改小写子串；`H264_NVENC`+vbr 现正确拒绝） |
| **L5** | 冒烟 docstring/默认值 `constqp,vbr` → `constqp,vbr_hq` | 否 | CPU | ✅ 已完成 |
| **L6** | T4 专项文档 `§5.2` 的 `--rate-mode-ifrnet vbr` | 否 | CPU | ✅ 已完成（改 `vbr_hq` + 补 LA 判读；另两处 `vbr` 是**陷阱说明**非命令，保留） |
| **E** | `av1_vp9_quality_matrix` 退出码口径 | 否 | CPU | ✅ 已完成（`[E-EXIT-CODE]`，扫描点拆为 `n_scan_fail` 不计退出码） |
| 2 | **修 B1**（`_stream_begin` 保留 `_cached_sps_pps`、只重置 `_sps_pps_injected`，与 LA=0 路径 `:2858` 对齐）+ hevc 臂负向回归 | 是 | T4 | ❌ 待做（**生产级缺陷**，h264 跨段复用 ⇒ 段 2 必崩） |
| 3 | **重跑 S8 的 h264 对照臂**（用 `vbr_hq` 不是 `vbr`）—— 这才第一次真正测到 h264 的 LA>0 路径 | 是 | T4 | ❌ 待做（前置=步 2） |
| 4b | **B4 备选**（仅当步 0 判定 32/64 非法才做）：`vbr_hq` → `RC_VBR(1)` + `targetQuality`，并**重跑全套等质量标定** | 是 | T4/L40 | ⏸ 待步 0 结论 |
| 6 | **L40 复跑**：同素材 constqp + vbr_hq + av1 三臂同批，判 S8 原始问题（+149.5 是否 AV1 特有） | 是（**仅 L40**） | L40 | ❌ 待做（前置=步 2、4） |

> ⚠ **AV1 侧的历史可比性问题（须在 L40 复跑时标注）**：`av1_nvenc` 的 `rate_mode` 被
> 降级为 `'vbr'`（`main_video_optimized.py:1013` / `nvenc_sdk.py:568`）后同样落
> `else` 兜底 ⇒ **AV1 的 `constqp` 与 `vbr` 两臂按构造同路径**。
> ⇒ 2026-10-04 之前那份 av1 vbr 臂数据（斜率 −21.4）**不可与未来任何臂比较**。
> hevc 侧不受影响（其 A/B 两臂恰为 `constqp` + `vbr_hq`，rc=0，天然可用）。


## 0. 一句话范围

在 L40（Ada）上补齐 **`av1_nvenc` 的等质量标定**两条轴，并复验 AV1 端到端能力：

- **CQ 轴**（`-cq:v`，量程 **0~63**，rate control 用 **`vbr`**）→ 落 `QUALITY_MAP['av1_nvenc']`；
- **QP 轴**（`-qp`，量程 **0~255**，与 CQ 非同刻度；L40 实测为**仿射** `(7.9338, −97.5136)`，**非** ×3）→ 落 `QUALITY_MAP_QP['av1_nvenc']`（D2b）；
- 复验 AC1（QP 尺度）/ AC2（G7-6）/ AC7（软编族，已闭环）与 **AV1 长视频冒烟 S1/S2/S3/S8**；
- 可选：查 AV1 Level 1 直通失败的 `GetEncodePresetConfig code=12`（§9.6）。

---

## 1. 环境体检（Gate 0，最先做）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L                                                       # 期望 NVIDIA L40（Ada / sm89）
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.version.cuda,torch.cuda.is_available())"
ffmpeg -hide_banner -encoders | grep -E 'av1_nvenc|av1_qsv|av1_amf'
ffmpeg -hide_banner -version | head -1

# ① 构建里有没有 av1_nvenc（构建层）
ffmpeg -hide_banner -encoders | grep av1_nvenc
# ② 硬件编不编得动：唯一可靠判据是【实跑一帧】（-h encoder=av1_nvenc 在 Turing 上照样打印选项表）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v av1_nvenc -f null - < /dev/null ; echo "rc=$?"
# rc=0 ⇒ 可执行本方案全部项；rc≠0（No capable devices found）⇒ 环境不成立，停止
```

> ⚠ `av1_qsv` 常在构建里但实跑失败（`Error creating a MFX session: -9`）；
> `av1_amf` 通常不在构建。两者**均不在本专项范围**（需 Intel/AMD 硬件，见方案 §8.4 / AC5）。

---

## 2. 待办任务清单（L40 侧，AV1 专项）

| ID | 任务 | 轴 | 产出 | 依据 |
|---|---|---|---|---|
| **L40-1** | `av1_nvenc` CQ 等质量标定（`-rc:v vbr -cq:v`） | `-cq:v` | `QUALITY_MAP['av1_nvenc']` | 立项 §7.1 B |
| **L40-2** | `av1_nvenc` QP 等质量标定（`-rc:v constqp -qp`） | `-qp` | `QUALITY_MAP_QP['av1_nvenc']` | D2b |
| **L40-3** | AC1 复验：QP 尺度 ×3（扫 21 / 63 / 84 / 105） | `-qp` | 报告（已闭环，复验） | 方案 §7 AC1 |
| **L40-4** | AC2：G7-6 `av1_nvenc -cq` 等质量 | `-cq:v` | 报告 | 方案 §7 AC2 / §8.3 |
| **L40-5** | AV1 长视频冒烟 S1/S2/S3/S8（`av1_pipeline_smoke.py` 完整跑批） | — | 报告 | 方案 §9.4 / §8.5 |
| **L40-6** | （可选）AV1 Level 1 `GetEncodePresetConfig code=12` 根因 | — | 结论 | 方案 §9.6 / P3″ |
| **L40-7** | harness / `_qp_model` 共享改动复用 | — | 代码 | T4 方案 §4 |
| **L40-8** | 跨仓 `QUALITY_MAP` 同步 + ⑨ 组 | — | 门禁 | T4 方案 §6 |

> T4 侧清单（h264/hevc）见 T4 方案 §2。**两卡不互替**：T4 无 AV1 NVENC。

---

## 3. 优秀做法吸收（AV1 专属，通用 17 条见 T4 方案 §3）

| # | 做法 | 出处 | 落地 |
|---|---|---|---|
| A1 | 硬件能力只认「实跑一帧」，不认 `-h encoder` | 方案 §7 AC0 / §8.1 | §1 |
| A2 | `av1_nvenc` 的 `-cq` 量程是 **0~63**（不是 51），`-qp` 是 **0~255** | 方案 E0 / §6.11.2 | §4.1 |
| A3 | **AV1 的 `-rc` 只接受 `constqp/vbr/cbr`** ⇒ `vbr_hq/qvbr` 必须降级为 `vbr` | 方案 §8.6-③ | §4.1 / harness 锁定 |
| A4 | AV1 的 `-qp` 与 `-cq` 是两条刻度；L40 实测 QP 轴为**仿射**（×3 仅 ref21 近似，crf≥24 起偏离） | 方案 §7 AC1 | §4.2；已落 `QUALITY_MAP_QP` |
| A5 | AV1 **Level 1 SDK 直通恒失败**（`GetEncodePresetConfig code=12`）⇒ 实际走 ffmpeg CLI | 方案 §8.6 | 标定的 CLI 口径即生产实际口径 |
| A6 | AV1 长视频色度检查（检查 4）**内容相关假阳性** ⇒ 验收加 `--skip-chroma` | 方案 §8.5 | §5.4 |
| A7 | `verify_video_integrity` 的 cv2→ffmpeg 回退（OpenCV 无 AV1 解码） | 方案 §8.6-① / §9.3 | 若冒烟失败先查是否回退未生效 |
| A8 | 判据/探针里**硬编码的期望值**要跟随上游表改动（A15：写死 84） | 方案 §8.2 / A15 | 探针已改 `av1_expected_qp()` 现场推导 |
| A9 | 命令形状断言（G6-8/9/10）+ 反向验证 | 方案 §9.2 | 改 AV1 下发前先跑 |
| A10 | AC7 软编族口径（`libsvtav1` 用 `-preset 8`、`libaom-av1` 用 `-cpu-used 6`） | 报告 §1.4 | 复跑 AC7 时锁定 |

---

## 4. AV1 差异（相对于 T4 方案的共享改动）

> 共享改动（`SWEEP`/`BASE_LOCK`/`QUALITY_FLAG`/`--axis`/`_qp_model` 模式感知/落表器
> `TIERS`+`GATE`+`tag`）**以 T4 方案 §4 为准**。以下是 AV1 的差异点。

### 4.1 AV1 的档位定义（T4 方案 §4.1 中已含，此处强调）

```python
SWEEP['av1_nvenc']        = [12, 18, 23, 27, 31, 36, 41, 47, 54, 63]   # -cq 量程 0~63
BASE_LOCK['av1_nvenc']    = ['-rc:v', 'vbr', '-b:v', '0', '-preset', 'p4']  # CR-2：显式 vbr
QUALITY_FLAG['av1_nvenc'] = '-cq:v'
HW_CODECS 含 av1_nvenc
```

> ℹ **CR-2（rate control）口径**：av1 统一**显式** `-rc:v vbr`（不加 HQ 附加项）
> （VE 生产 writer 的 av1 降级路径本就是 `vbr`；VU 已把 av1 也改为显式 `-rc vbr`）。
> ⚠ 2026-10-04 起 FFmpeg 9.0 **移除 `vbr_hq`/`qvbr`**（`-rc` 只剩 constqp/vbr/cbr）⇒
>   h264/hevc 的 CLI/harness 口径改为**裸 `vbr`**（`-tune`/`-multipass` 改显式 opt-in；VE SDK 侧仍走
>   `RC_VBR_HQ(32)`，实测驱动仍接受）。**AV1（本专项唯一目标）不受影响**——它本就是 plain `vbr`。
> 探针侧 VE 已修（`av1_vp9_quality_matrix.py` 的 `_PROD_RC`）；VU 侧 h264/hevc 待重新同步（T4 方案 §12.3）。

QP 轴由 `--axis qp` 切到 `-rc:v constqp -qp`，量程 `(0, 255)`（T4 方案 §4.2 的量程分支已含 AV1）。

### 4.2 AV1 的 QP 尺度与既有 override（**2026-10-04 结果已出**）

标定前的回退值（`src/utils/quality_map.py`）：

```python
_QP_MAP_OVERRIDE['av1_nvenc'] = (3.0, 0.0, 0, 255)   # ×3：仅 ref21 附近验证过的近似
```

- **等质量（quality）口径**：标定后由 **`QUALITY_MAP_QP['av1_nvenc'] = (7.9338, −97.5136, 0, 255)`
  优先命中**（仿射，**非 ×3**）。实测 ×3 在 crf24/27/30 残差 −20/−37/−51 ⇒ 已取代。
  ref21 → QP **70**（`to_constqp_qp(32)`）；G3-7/G6-7/8 期望随之更新。
- **等体积（size）口径**：保持 `_QP_MAP_OVERRIDE` 的 ×3（G3-9 期望 av1 CQ27→QP63 不变）。
- **两条轴不是同一刻度**：CQ 0~63 / QP 0~255；且 **CQ 行的 `b` 会经 `to_x264_crf` 往返影响 QP 期望**
  （b=1.1022→71、b=1.2165→70）。

### 4.3 AV1 的锚点/轴映射（标定的物理含义）

| 轴 | ffmpeg 参数 | rate control | 生产对应路径 |
|---|---|---|---|
| CQ | `-cq:v <0~63>` | `-rc:v vbr` | ffmpeg CLI writer（Level 2/3）+ SDK vbr 分支 |
| QP | `-qp <0~255>` | `-rc:v constqp` | ffmpeg CLI constqp + SDK Level 1（若未来恢复直通） |

⇒ **两条轴都要标**，`QUALITY_MAP` 与 `QUALITY_MAP_QP` 各一行。

---

## 5. 执行步骤（L40）

### 5.1 基线（改动前后对照）

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null   # PASS=104 / FAIL=0 / SKIP=11（quality 口径，2026-10-04 B1）
python3 Accessory/verify/plan_implementation_gate.py < /dev/null            # FAIL=0
python3 Accessory/probe/calibrate_equal_quality.py --selftest               # 39 项（含 NVENC/axis/跨仓）
python3 Accessory/probe/av1_vp9_quality_matrix.py --selftest 2>/dev/null || true
```

### 5.2 素材（真实切片，覆盖 6 类，≤7 条，AV1 编码慢需控量）

同 T4 方案 §5.2 的 7 条（`--src-is-prep`）。⚠ AV1 **软件**编码慢的是 `libaom-av1`/`librav1e`，
本专项只用 `av1_nvenc`（硬件），编码快；成本主要在 VMAF（`n_subsample=1`，20~45 s/点）。

### 5.3 标定（AV1 两条轴）

```bash
# ── CQ 轴 ──────────────────────────────────────────────────────
for M in live_kids_play tv_bbc_s01e01 cganim_edu_wordworld anim2d_subs_tobot \
         doc_dark_earth screen_ui_code live_texture_frog; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_l40/cq_${M} --tiers av1_nvenc \
      --duration <6|10> --src-is-prep < /dev/null
done
# ── QP 轴 ──────────────────────────────────────────────────────
for M in <同上 7 条>; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_l40/qp_${M} --tiers av1_nvenc --axis qp \
      --duration <6|10> --src-is-prep < /dev/null
done
```

### 5.4 AV1 端到端复验（AC1/AC2/AC4 + 冒烟）

```bash
# ── AC1/AC2/AC4 一条命令（自动判读，口径与判据同源）──────────
python3 Accessory/probe/av1_vp9_quality_matrix.py \
    --src /workspace/input_videos/word_world_2.mp4 \
    --only av1_nvenc \
    --report verification_report/av1_vp9_matrix_L40_$(date +%F).md \
    --json   verification_report/av1_vp9_matrix_L40_$(date +%F).json < /dev/null
# 判据：AC1 表值 -qp 落 RATE_PASS=(0.65,1.50) 且 ΔPSNR ≥ -1.5 dB；退出码 0

# ── G7/G8 GPU 判据（AC2 = G7-6 转正）─────────────────────────
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source /workspace/input_videos/word_world_2.mp4 \
    --bitrate-source /workspace/input_videos/new4_raw.mp4 \
    --report verification_report/crfcq_gpu_L40_$(date +%F).md < /dev/null

# ── AV1 长视频冒烟（P3 完整跑批，S1/S2/S3/S8 需 GPU）─────────
python3 Accessory/verify/av1_pipeline_smoke.py --src <330s真实素材> \
    --rate-modes constqp,vbr --segment-duration 30 \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem \
    --report verification_report/av1_smoke_L40_$(date +%F).md < /dev/null
# 退出码：0 无 FAIL / 1 有 FAIL / 2 环境前置不成立
# 采样判据：S1 退出码 / S2 段级 decoded==expected / S3 产物帧数=各段之和
#          / S4 解码级验证 / S5 segment_bitstream_verify_v5 --skip-chroma
#          / S6 QA sidecar / S7 产物编码器确为 av1
#          / S8 泄漏：后半程 RSS 斜率 ≤ +50 MB/min【口径未变】+ 峰值 ≤ --mem-peak-mb
#             （默认 12000）+ 采样点 ≥ --mem-min-samples（默认 8，不足报 SKIP）
#          S8 判读：detail 里的「主 x / 子 y MB/min」直接给出泄漏归属
#                 （main ⇒ 主进程；ffmpeg ⇒ 读帧器/分段 muxer）；PSS 同步涨=真泄漏
#          逐进程明细：/tmp/s8_mem/<rate_mode>.mem.tsv（可离线重算，不必二次上机）
```

### 5.5 入池 → LOO → 落表候选

```bash
python3 Accessory/probe/eqq_pool_fit_table.py \
    --sides 6s,10s,legacy10s,gpu_l40_cq --axis cq --out /tmp/table_l40.txt < /dev/null
# 判据：av1_nvenc 行 LOO ≤5.9；顺序无关断言通过
# ⚠ 2026-10-09 修正：GPU 点数据目录带轴后缀（gpu_l40_cq / gpu_l40_qp），
#   CQ/QP 不可混池；旧写法 `--sides …,gpu_l40` 匹配不到目录 → 打印
#   「拟合失败（样本 0 < 4）」但退出码仍为 0（静默假通过）
```

### 5.6 （可选）L40-6 · AV1 Level 1 `code=12`

按方案 §9.6 的复现片段（`GetEncodePresetConfigEx` 回退），
⚠ 必须先 `import torch; torch.cuda.init()` 再加载 CUDA 库。
判读：`GPCEx -> 0` 且能 `InitializeEncoder` ⇒ 加回退（两侧 `nvenc_sdk.py`）；
否则记为驱动侧限制并注释「AV1 恒走 Level 2/3」。

---

## 6. 落表与跨仓同步

| 表 | 键 | L40 新增 | 同步 |
|---|---|---|---|
| `QUALITY_MAP`（等质量） | `av1_nvenc` | CQ 轴标定值 `(a, b, 0, 63)` | **两仓逐条相等**（⑨ 组）；VidUtils 侧须同步 |
| `QUALITY_MAP_QP`（QP 轴，仅本仓） | `av1_nvenc` | QP 轴标定值 `(a, b, 0, 255)` | 仅 VE |

⚠ 写入后 `QUALITY_MAP['av1_nvenc']` 的 hi **必须是 63**（不是 51）——`SIZE_MAP` 已如此，
等质量表须保持一致，否则 `crf_ref≥45` 被挤到 51（方案 E0 的老 bug）。

> ℹ **无损语义（2026-10-04 B1/`[FIX-QP-LOSSLESS]`）**：`to_constqp_qp(codec, 0)` 在 **size/quality
> 两口径均返回 0**（对 `value==0` 短路）。故 L40 落 `QUALITY_MAP_QP['av1_nvenc']` 后，AV1 的
> `-qp 0` 仍是 0（无损/最高质档），不被标定表的 `a/b` 外推；G3-4 已锁双口径、G3-9 锁 size 对照。
> 生产无损另由 writer `crf==0` 分支硬编码（G6-18/19）。
> ⚠ 该口径分流**不影响 AV1 的 AC1 结论**（两口径 `-qp` 均 63）。

---

## 7. 验收门禁

| 门 | 命令 | 判据 |
|---|---|---|
| 落表器 | `eqq_pool_fit_table.py --sides …,gpu_l40_cq --axis cq` | av1 行 LOO ≤5.9 + 顺序无关 ✅ |
| AC1/AC2/AC4 探针 | `av1_vp9_quality_matrix.py --only av1_nvenc` | 退出码 0；AC1 表值落带内 |
| GPU 判据 | `crf_cq_unification_verify.py --gpu …` | G7-6 PASS、FAIL=0 |
| AV1 冒烟 | `av1_pipeline_smoke.py` | 退出码 0；S1~S8 无 FAIL |
| 本仓判据/门禁/pytest | 同 T4 方案 §7 | FAIL=0 / 全绿 |
| 跨仓真源 | `VidUtils/verify/verify_quality_mapping.py` | ⑨ 组 14/14 |

---

## 8. 回滚

| 触发 | 动作 |
|---|---|
| av1 表 LOO 超门禁 | 不落表；保留 `gpu_l40_{cq,qp}` points 与报告，标注根因 |
| `QUALITY_MAP_QP['av1_nvenc']` 与 ×3 冲突 | 以实测为准；若证据不足则维持 `_QP_MAP_OVERRIDE` 的 ×3 并回滚新行 |
| 冒烟 S 项 FAIL | 先查 §8.6 三处 AV1 修复是否在位（②③ 命令形状 / ① cv2 回退）；再查编码线程 |
| Level 1 回退无收益 | 保持现状（Level 2/3 功能正确），仅注释说明 |

---

## 9. 风险与坑

| 风险 | 对策 |
|---|---|
| **`-rc:v vbr_hq/qvbr` 对 av1 非法** | 锁定 `vbr`（A3 / §4.1） |
| **AV1 Level 1 恒降级** | 标定走 CLI 口径即生产实际口径（A5）；勿假设 SDK 直通 |
| **色度检查假阳性** | 验收加 `--skip-chroma`（A6） |
| **cv2 无 AV1 解码** | 若产物被判损坏，查 `[FIX-AV1-CV2]` 回退是否生效（A7） |
| **探针硬编码期望值过期** | 已改现场推导 `av1_expected_qp()`（A8），勿再写死 |
| **L40 会话中途 GPU 被回收** | 每步当场复跑；同 T4 方案 P17 |
| **同机并发 NVENC** | `--jobs 1`；先查 `nvidia-smi` 与他人任务 |
| **误把 av1_qsv/amf 当 AV1 NVENC** | §1 已区分；AC5 需 Intel/AMD 硬件，本专项不含 |

---

## 9.5 下次上机待办（T4 / L40 通用，按优先级）

> **⚠ 2026-10-04 更新：D/E 已在 T4 执行完毕，结论见 §0.1；A/B/C 的原方案已被 §0.1.2 取代。**
> 背景：§1~§7 已收口，**唯一遗留是 S8**（constqp 后半程 RSS 斜率 +149.5 MB/min，
> vbr −21.4 通过，两次复现）。定位工具已就绪（见 §0「S8 的无 GPU 准备项」），下机只需跑一条命令读结论。

| # | 待办 | 卡在哪 | 上机怎么做 | 完成判据 |
|---|---|---|---|---|
| **A** | ~~S8 定位~~ → **已由 D 项在 T4 完成**（§0.1） | — | — | ✅ T4 上 constqp 两臂（h264 −55.6 / hevc +3.1）均无泄漏；但 **h264 的 vbr 对照臂因 B1 崩溃**，A/B 仅在 hevc 成立 |
| **B** | ~~显存归属口径复验~~ → **归因已修正为 PID namespace**（§0.1.3 B3） | **结构性不可行**（非「等 L40」） | 容器内 `nvidia-smi --query-compute-apps` 返宿主 pid，`/proc` 下不存在 | ✅ 已定位根因；待做 = 改报 `None` 而非 `0.0`（§0.1.5 步 1） |
| **C** | ~~S8 判据峰值上界校准~~ → **已实测否决默认 12000** | — | T4 实测两臂峰值 **13578~13877 MB 均超上界但斜率正常** | ✅ 已得出「峰值上界比斜率更易误报」，待按素材/卡型重新取值（§0.1.5 步 5） |
| **D** | ~~constqp 路径可在 T4 复现~~ → **已执行（2026-10-04）** | — | T4 `h264_nvenc` + `hevc_nvenc` × `constqp,vbr_hq`，358.76s 素材同批 | ✅ **不复现** ⇒ L40 的 +149.5 疑为**窗口伪影**（同数据全程 +149.4、换窗口 −786~+1434）；**S8 原始问题仍需 L40 回答** |
| **E** | 顺带复核项（非阻塞） | — | ①`av1_vp9_quality_matrix` 退出码口径（§0 遗留的 1 vs 0）；②L40-6（AV1 Level 1 `code=12`）若仍想做 | 有结论或明确记为不做 |
| **F** | **新增**：修 B1（h264 跨段 SPS/PPS，生产级）+ 重跑 h264 vbr 臂 | 需 GPU | 见 §0.1.5 步 2~3 | 12/12 段通过且 `non-existing PPS` 计数 = 0 |
| **G** | **新增**：B2 方案 A（脚本级禁用 `vbr`/`cbr` + 修日志矛盾回显） | 无需 GPU（方案 B 才需） | 见 §0.1.5 步 4 | 传 `vbr` 时 Ready 行不再出现 `CONSTQP` |

**A 的命令**（一条跑完，读 `S8` 的 detail 即可）：

```bash
cd /workspace/Video_Enhancement
python3 Accessory/verify/av1_pipeline_smoke.py --src <≥330s 真实素材> \
    --rate-modes constqp,vbr --segment-duration 30 \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem \
    --report verification_report/av1_smoke_$(date +%F).md < /dev/null
```

**读数判读表**（直接对应 S8 detail 的「主 x / 子 y MB/min」）：

| 观测 | 结论 | 下一步 |
|---|---|---|
| `子` 为正、`主` ≈ 0 | **ffmpeg 子进程累积**（读帧器 / 分段 muxer），主进程无辜 | 查 reader 帧队列与分段 muxer 的 `_stderr_lines` |
| `主` 与 `子` 同为正、PSS 同步涨 | **主进程真泄漏**（不是共享页虚高） | 按 `/tmp/s8_mem/*.mem.tsv` 里 top RSS 的 pid 定位到具体对象 |
| `主` 正、**PSS 不涨** | 多为 CUDA 上下文 / 共享页虚高，非真泄漏 | 降级判据（考虑改用 PSS 作主判据），别急着改管线 |
| 两者都 ≈ 0 但**峰值超上界** | 是峰值台阶而非单调泄漏（如某段累积后释放） | 查段切换的清理路径（`main.py:1155-1157` 每段新建编码器） |

⚠ **只跑 `constqp` 不足以判读**：必须 `constqp,vbr` 同素材对照，才能区分「constqp 特有」
与「长跑本身的时间相关项」。

---

## 10. 复现命令汇总

```bash
cd /workspace/Video_Enhancement
# 0 体检（AV1 实跑一帧）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v av1_nvenc -f null - < /dev/null; echo rc=$?
# 1 标定（CQ 轴，单素材示例）
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src ../input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out temp/eqq_gpu_l40/cq_live_kids_play --tiers av1_nvenc \
    --duration 6 --src-is-prep < /dev/null
# 2 入池落表
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_l40_cq --axis cq < /dev/null
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_l40_qp --axis qp < /dev/null
# 3 AV1 端到端复验
python3 Accessory/probe/av1_vp9_quality_matrix.py --src /workspace/input_videos/word_world_2.mp4 --only av1_nvenc < /dev/null
python3 Accessory/verify/crf_cq_unification_verify.py --gpu --source /workspace/input_videos/word_world_2.mp4 --bitrate-source /workspace/input_videos/new4_raw.mp4 < /dev/null
python3 Accessory/verify/av1_pipeline_smoke.py --src <330s素材> --rate-modes constqp,vbr \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem < /dev/null
# 4 门禁
python3 Accessory/verify/plan_implementation_gate.py < /dev/null
cd /workspace/VidUtils && python3 verify/verify_quality_mapping.py < /dev/null
```

---

## 11. AC 覆盖现状（进入本专项前请核对）

| AC | 内容 | 现状（方案 §8 收口） | 本专项动作 |
|---|---|---|---|
| AC1 | AV1 constqp QP 尺度 | ✅ 已闭环（L40：旧 ×3/63 已在 ref21 验证；**现为仿射表**，表值 `-qp 70` → 1.07× / −1.13 dB 落带内 PASS） | 已复核 |
| AC2 | AV1 `-cq` 等质量（G7-6） | ✅ PASS（+0.16 dB / 1.28×） | 复验（L40-4） |
| AC3 | AV1 constqp 命令形状（G6-7） | ✅ PASS（无需硬件） | — |
| AC4 | AV1 `-cq` B 组 | ✅ PASS（1.14× / −0.29 dB） | 复验 |
| AC5 | av1_qsv / av1_amf 量程 | ⏭️ SKIP（方法已证伪，需 Intel/AMD） | 不在本专项 |
| AC6 | 跨项目 C 组交叉印证 | ✅ 由 AC1 同源覆盖 | VidUtils 侧另跑 |
| AC7 | AV1/VP9 软编族 | ✅ 完成（T4 重编构建） | 不在本专项（无硬件依赖） |

> **但「换算正确 ≠ 管线能跑」**：AV1 端到端能力由 P3 冒烟（§8.5）+ 三处修复（§8.6）保障，
> 本专项的 **L40-5** 是它首次在 GPU 上的完整回归。**S8 是唯一未闭环项**，定位工具已就绪，
> 待办与读数判读表见 **§9.5**。
>
> **2026-10-04 口径变更对 AC 的影响**：门禁口径已迁 **quality**（B1）且 `to_constqp_qp(0)=0` 双口径，
> 但 **AV1 的 `-cq`/`-qp` 数值在两口径相同**（`-qp 63`、`-cq` 表值不变）⇒ **AC1~AC4 的判据与期望值不变**，
> 仅"门禁口径 == 生产默认"这一形式更强。L40 上机前先按 §0「前置已就绪」核对。

---

## 12. VidUtils（VU）对等方案态势（**协同必备**）

> 完整契约见 **T4 方案 §12**；本节只列 AV1 相关的态势与协同点。

- **VU 侧有同构的 L40 方案**：`/workspace/VidUtils/Plan/VidUtils_等质量标定_L40_AV1专项执行方案.md`
  （其任务编号 `A0~A5` ↔ 本专项 `L40-1~L40-5`；VU `G3` ↔ 本专项 `L40-1`）。
- **VU harness 已实现 AV1/NVENC + `--expect-av1`**；VE harness 已移植（+ `--axis` 扩展）。
  两侧都用「实编探测 + fail-fast」，判据同为「rc==0 且产物非空」（T4 的 av1 会「列表里有、实编 -22」）。
- **AV1 的 `-cq` 等质量行写入两仓共享 `QUALITY_MAP`** ⇒ 受 **CR-1（preset 口径）** 与
  **CR-2（rate control 口径）** 约束。**CR-1 已收口：两仓统一 `p4`**（2026-10-04）——
  VE 本就 p4；**VU 已把生产 `DEFAULT_PRESET_GPU` / harness / 探针一并改 p4**（残留 `p5` 均为
  兼容显式 p5 的有意保留，见 T4 方案 §12.5）。
- **CR-2（rate-control 口径）**：av1 `vbr`，**均显式下发 `-rc`**。⚠ 2026-10-04 FFmpeg 9.0 移除
  `vbr_hq`/`qvbr` ⇒ h264/hevc 的 CLI/harness 口径从 `vbr_hq` 改为**裸 `vbr`**（2026-10-04 二次校正）
  （VE 已改；VE SDK 侧仍 `RC_VBR_HQ(32)`，实测驱动仍接受）。**本专项（AV1）不受影响**（本就 plain `vbr`）。
  VU 侧 h264/hevc 需重新同步，否则共享 `QUALITY_MAP` 的 ⑨ 组变红（handoff）。
- **QP 轴（`QUALITY_MAP_QP` / `_QP_SCALE`）**：VE 侧是独立表；VU 侧无表，只有 `_QP_SCALE`。
  本专项落 `QUALITY_MAP_QP['av1_nvenc']` 后，若与 VU 的 `_QP_SCALE=3` 冲突，**通知 VU 同步**（CR-4）。
- **无损契约 handoff（2026-10-04 实测）**：VU 自有 `to_constqp_qp(codec, 0)` **不保证 0**——
  NVENC 靠 `_QP_LIMITS` 夹回 0，但 **`librav1e`→48/52、`libsvtav1`→10/9**（表 b<0 使 ref>0）。
  VU 生产因 `_resolve_quality_params` 的 `[LOSSLESS]` 短路而未触发，但**函数级与「无损=0」不符**。
  VE 已加 `[FIX-QP-LOSSLESS]` 短路（两口径均 0）；**建议 VU 同步加 `if value==0: return 0`**。
- **跨仓态势已双向对称**：VE harness 现与 VU 一样会打印「两表是否相等 / 对侧 harness 是否同版 /
  对侧方案文档」。上机前先看这一行，再决定要不要协调对侧。
- **素材池共用**：17 条切片在 VE `input_videos/eqq_calib/`（仓库外）——L40 机上需先就位（CR-5）。
