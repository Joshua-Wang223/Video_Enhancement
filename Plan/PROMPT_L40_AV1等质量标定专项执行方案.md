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
> 状态（2026-10-06 **全项执行完毕并落表；L40-5 冒烟完成，L40-6 可选项记录结论**）：§1~§7 全部走完。
> **S8 遗留项已在 T4 降本收口 + L40 实测通过** —— T4 上 constqp 两臂均无泄漏，
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
> **验收（已提交 `95a3f75` + 本轮 L40 复测 `plan_gate` 100/0/1）**
> - `crf_cq --quick --no-gpu` **104/0/0/11**；`--gpu` **113/0/3**；`plan_implementation_gate` **94/0/2** → **本轮 100/0/1**；
> - AC1 探针：表值 **`-qp 70` 落带内 PASS**（1.07× / −1.13 dB）——「×3(63)」结论已被仿射表取代；
> - ⚠ 工具口径：`av1_vp9_quality_matrix` 退出码仍为 1（扫描点 21/84/105 属**有意带外**被计入 `n_fail`），
>   与 §5.4「退出码 0」不一致，属文档/工具口径问题，未改判据。
>
> **L40-5 长视频冒烟（358.8s 真实素材，2026-10-06 实测完成）**
> - `[FIX-S3-STAGE-DEDUP]`：两阶段管线各校验一次同批分段致 S3 假 FAIL（脚本 double-count）；
>   修复后复跑 **S3 ✅ 17926 = 产物帧数**；S1/S2/S4/S5/S6/S7 全过。
> - **constqp/vbr 双臂 S8 均通过**：constqp 斜率 +32.7 MB/min（主 +61.9 / 子 -29.2）；vbr 斜率 -2.8 MB/min（主 +8.5 / 子 -11.3），均 ≤ +50 阈值内。
> - 峰值上界自动标定 9618 MB（bs=8）未触发误报。
>
> **L40-6（可选）**：AV1 Level 1 `GetEncodePresetConfig code=12` 根因 —— 环境限制未触发，记录为「驱动侧限制，AV1 恒走 Level 2/3 CLI」。
>
> **未做（本方案范围外/条件未触发）**：AC5（av1_qsv/amf，需 Intel/AMD）。

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

> **2026-10-05 更新：步 1~4 已完成（T4），步 5 的标定数据正在采集，步 6 仍需 L40。**
> 三个阻塞项 B1/B2/B3 的修复与验证结论见 **§0.2**。

| 步 | 动作 | 需 GPU | 前置 | 状态 |
|---|---|---|---|---|
| 1 | **修 B3**（显存口径：`None` 而非 `0.0` + 标注「不可归属」） | 否 | — | ✅ 已完成（`[FIX-B3-GPU-UNATTRIBUTABLE]`） |
| 2 | **修 B1**（按「`_stream_begin` 保留 `_cached_sps_pps`、只重置 `_sps_pps_injected`」，与 LA=0 路径 `:2858` 对齐）+ hevc 臂负向回归 | 是 | 步 1 | ✅ 已完成（`[FIX-B1-SPS-PPS-SESSION-REUSE]`，T4 rc=0 / 5 段全过 / S1~S8 全绿） |
| 3 | **重跑 S8 的 h264 vbr 臂**（用 `vbr_hq` 不是 `vbr`）—— 这才第一次真正测到 h264 的 LA>0 路径 | 是 | 步 2 | 🔄 100 s 素材已跑通（8/0）；358 s 标准素材双臂对照进行中 |
| 4 | B2 方案 A（脚本级禁用 `vbr`/`cbr` + 修 `_log_ready` 自相矛盾回显）；方案 B（真 vbr 分支）另立 A/B 定量 | A 否 / B 是 | 步 3 | ✅ 方案 A 已完成（`[FIX-B2-RC-MODE-REJECT]`，两侧同构）；方案 B 未做 |
| 5 | 标定 `--mem-peak-mb`（按 T4 实测 13578~13877 重新取值，而非沿用 12000 估值） | 否 | — | 🔄 见 §0.2.4 —— 峰值强依赖 `batch_size` 与素材长度，旧值 12000 是 bs=24 下估的 |
| 6 | **L40 复跑**：同素材 constqp + vbr_hq + av1 三臂同批，判 S8 原始问题 | 是（L40） | 步 2、4 | ⏸ 阻塞：本机是 T4 |

---

## 0.2 §0.1.5 步 1~4 执行完毕（2026-10-05，T4）—— B1/B2/B3 三个阻塞项已修

> 步 5（`--mem-peak-mb` 标定）与步 6（L40 复跑）仍未完成；步 5 的采集在跑，步 6 阻塞于硬件。
> 本节记录三个修复的**根因、落地方式、验证判据**，以及一条被推翻的假设。

### 0.2.1 B1（生产级缺陷）已修 —— `[FIX-B1-SPS-PPS-SESSION-REUSE]`

**根因（比 §0.1.3 记的更精确）**：`1e57c0b` 在同一个段边界重置里加了**两个互相冲突**的动作 ——
`_sps_pps_injected = False`（正确：新段有新 muxer，需重新预注入）与
`_cached_sps_pps = None`（**错误**）。后者的注释假设「跨段会拿到不同的参数集」，但该前提**不成立**：

- 本类**实例即会话** —— `_open_encode_session()` 只在 `__init__` 调用一次，**无 reopen 路径**
  （`grep -n '_open_encode_session('` 只有定义处 + `:__init__` 两处命中）；
- 跨段复用的前提是参数严格相等（`main.py::_get_or_create_nvenc_encoder` 的 `key` 含
  W/H/fps/preset/qp/rate_mode/LA/pipeline_depth/codec 九项）⇒ key 不等就**新建对象**，
  缓存本就是 `None`。

而驱动侧 `repeatSPSPPS` bit12 **不写**（`nvenc_sdk.py:971`）⇒ 参数集**只在会话建立后下发一次**，
复用会话时不会为新段重吐。清缓存与「不重吐」叠加 ⇒ 段 2+ 完全没有参数集 ⇒
`non-existing PPS 0 referenced` → muxer pipe broken → rc=1。
**LA=0 路径不崩的原因**：`ce_pipeline` 的段首只重置 `_sps_pps_injected`、**不清缓存**
⇒ 段 2 由 `_prepend_param_sets` 补挂。**LA=0 / LA>0 的这个不对称就是缺陷面。**

**落地**：`external/ifrnet_video/nvenc_sdk.py::_stream_begin` 删掉那行清空；
`external/realesrgan_video/nvenc_sdk.py` 侧本就正确（会话代数守卫 `[FIX-ESRGAN-SPSPPS-REUSE]`），
只修正其 `:3382` 一处**过时注释**（称「`_stream_begin(force=True)` 现在会清除 `_cached_sps_pps`」——
实际是条件清除）。
**残留风险已由两道既有机制覆盖**（未新增代码）：① `_cache_param_sets` 的 `[P3-FIX-NAL-COMMON]`
字节漂移检测（段 2 若真吐不同参数集则告警并切流内值）；② `_prepend_param_sets` 仅在
「IDR 且原生缺参数集」时预挂 ⇒ 不产生 `avcC numSPS/numPPS=2` 重复。

**T4 GPU 验证**（h264_nvenc + `vbr_hq` = LA>0 路径，100 s / 5 段素材）：
**rc=0、S1~S8 全 8 PASS / 0 FAIL**（此前片段 2 即 rc=1）；S2 段级 `decoded==expected` 5/5、
S3 产物 4999 = 分段合计 4999、S4 解码级 ok、S5 码流硬指标全过、**日志零 `non-existing PPS`**。

### 0.2.2 B2 方案 A 已修 —— `[FIX-B2-RC-MODE-REJECT]`

**方案 A = 不实现真 vbr 分支，改为显式失败**。理由：真 vbr 要动 `rc_ptr[1]=1` + LA 门控 +
全链路内部改名 + 跨仓同步，且需 GPU A/B 定量（= 方案 B，另立）；而生产默认是 `vbr_hq`，
`vbr`/`cbr` **无生产使用者** ⇒ 先把静默错误变成显式失败是当前成本最低的正确解。

**落地（两侧同构）**：`nvenc_sdk.__init__` 在 codec 检查之后立刻
`if rate_mode not in ("constqp","vbr_hq","qvbr"): raise ValueError(...)`（消息里写明
「传 vbr/cbr 会**静默**落 CONSTQP」+ 指向 CLI 路径）。⚠ **只在 SDK 直通层生效**，
Level 2/3 的 ffmpeg CLI 路径**正确支持** vbr/cbr（`ffmpeg_io._rc_v_map` 真实下发 `-rc:v vbr`），
不受影响 —— 这也是 AV1 的 `main_video_optimized.py:1031` 自动降级到 `'vbr'` 仍可用的原因
（该 raise 被 `main.py::_setup_level1_nvenc` 的 `except Exception` 接住 → 落到 Level 2/3 CLI，
与改前最终落点一致）。

**连带修**：`_log_ready` 的三段三元（`vbr_hq→VBR_HQ / qvbr→QVBR / else→CONSTQP`）
把任何其它档位都显示成 `CONSTQP`，于是传 `vbr` 时这行打成 `CONSTQP ... la=8` ——
**自相矛盾**（真实硬件 CONSTQP + LA 被静默禁用，`la=8` 只是未被清零的 `_la_depth` 回显）。
改为字典查表 + 无兜底标签；`la=` 仍只在 `_la_depth > 0` 时追加。

**AV1 自动降级目标改为 `constqp`**：`vbr_hq/qvbr` 曾降级到 `'vbr'`，而 SDK 层未实现 vbr ⇒
降级目标必须仍是已实现档位（AV1 原生支持 constqp）。

**脚本侧同步**：`av1_pipeline_smoke.py` 的 `--rate-modes` 默认由 `constqp,vbr` 改为
`constqp,vbr_hq` —— 对照臂必须是唯一真 VBR_HQ+LA>0 路径，不能用已被拒的 `vbr`。

### 0.2.3 B3 已修 —— `[FIX-B3-GPU-UNATTRIBUTABLE]`

`_gpu_mem_by_pid()` 改为返回 `(by_pid, status)`；不可归属时 `gpu_tree_mib = **None**`
（TSV 落盘写 `NA`，报告写「不可归属（原因）」），不再用 `0.0` 冒充。
**新增能力**：`status="pid_ns_mismatch"` 的判据是「`nvidia-smi --query-compute-apps` 有输出
且其中**没有任何** pid 存在于本容器 `/proc`」⇒ 直接把「驱动侧用宿主 pid、容器 PID namespace
不通」这个**根因**显式写进报告，而不是只留一个 0。
`peak()` 相应返回 `Optional[float]`。
三条分支（`ok` / `pid_ns_mismatch` / `empty`）已用合成数据验证。
实测印证：修复后 S8 detail 显示「显存峰值 不可归属（pid_ns_mismatch）」，
而同一跑批整卡 7698 MiB / 100% 利用率 —— **旧读数会被误读成「确实没用显存」**。

### 0.2.4 S8 判据标定（步 5）：峰值上界比斜率更易误报，且强依赖 bs

**待补**：358 s 标准素材、bs=8 的双臂对照正在采集（`--mem-dump-dir /tmp/s8b/mem`）。
已知的三条实测事实（供判读）：

1. **旧默认 12000 是 bs=24 下的估值**：bs=24 / 358 s 存档峰值 12393~13877（两臂**都超上界**
   而斜率正常）；**bs=8 / 100 s 实测峰值仅 7243** ⇒ 峰值∝ `batch_size`（pinned result pool 线性）
   **且**∝ 素材长度（段数累积）。⇒ **单一常数无法跨 bs/素材通用**，
   `--mem-peak-mb` 只能在**固定 bs + 固定素材**下比较。
2. **峰值判据本身弱于斜率**：同一份 hevc constqp 数据，斜率随 OLS 窗口在 −786 ~ +1434 MB/min
   之间跳变（§0.1.2），而峰值只有一个数、方向单一 ⇒ 峰值超界**不区分**「失控增长」与
   「工作集台阶」。
3. **新读数应同时报窗口与噪声底噪**（后半程 sd ≈ 1356 MB ⇒ 不确定度约 ±66 MB/min）。

### 0.2.5 一条被推翻的假设（记录以免重复）

静态审阅阶段（§0 末「无 GPU 准备项」）曾把**头号候选**定为「per-frame CUDA event 在
`cuEventSynchronize` 失败 raise 前跳过销毁」，并以「rc=0 ⇒ 该 raise 从未触发」证伪。
本轮 B1 修复时确认了同一条推理链**在另一处同样有效**：`_stream_begin` 的清空动作之所以
长期没人质疑，正是因为它**不产生任何错误日志**（清空 → 驱动不重吐 → 段 2 才炸），
静态读代码极易判成「合理的防御性重置」。
⇒ **「静默降级」类缺陷只能靠运行时的第二个数据点（同素材跨段）发现**，
代码审查与单点单元测试都覆盖不到。

---

## 0.3 T4 上机执行结果（2026-10-06，T4 / 580.65.06 / bs=8）—— S8 标定 + 峰值校准完成

> **素材**：`01 the race to mystery island.fixed.mp4`（358.76s / 720×576 / 12 段 / 8969 源帧）  
> **配置**：`batch_size=8`（config 默认 24 → `--batch-size 8` 覆盖，双重收益：吞吐↑12.9% + S8 锯齿↓）  
> **口径**：`--rate-modes constqp,vbr_hq`（vbr_hq 为唯一真 VBR+LA>0 路径，vbr 已被 SDK 直通层拒绝）

### 0.3.1 H.264 双臂（constqp / vbr_hq）完整 358s 跑批

| 指标 | constqp | vbr_hq |
|------|---------|--------|
| S1 退出码 | rc=0 (1821s) | rc=0 (2081s) |
| S2 段级守恒 | 12/12 PASS (17926 帧) | 12/12 PASS (17926 帧) |
| S3 产物帧数 | 17926 = 分段合计 | 17926 = 分段合计 |
| S4 解码级验证 | OK (frames=17926) | OK (frames=17926) |
| S5 码流硬指标 | PASS (帧守恒/IDR/frame_num/pts) | PASS (同上) |
| S6 QA sidecar | 完整 | 完整 |
| S7 编码器确认 | h264 | h264 |
| **S8 后半程 RSS 斜率** | **-76.8 MB/min** (主 -76.4) | **+12.6 MB/min** (主 +13.2) |
| **S8 RSS 峰值** | **6730 MB** | **7694 MB** |
| GPU 显存归属 | pid_ns_mismatch (B3 生效) | pid_ns_mismatch (B3 生效) |

✅ **两臂 S1~S8 全绿**。constqp 斜率为负（回收），vbr_hq 斜率 +12.6 落在 +50 阈值内。

### 0.3.2 HEVC 双臂（constqp / vbr_hq）完整 358s 跑批

| 指标 | constqp | vbr_hq |
|------|---------|--------|
| S1 退出码 | rc=0 (1707s) | rc=0 (1684s) |
| S2 段级守恒 | 12/12 PASS (17926 帧) | 12/12 PASS (17926 帧) |
| S3 产物帧数 | 17926 = 分段合计 | 17926 = 分段合计 |
| S4 解码级验证 | OK | OK |
| S5 码流硬指标 | PASS | PASS |
| S6 QA sidecar | 完整 | 完整 |
| S7 编码器确认 | hevc | hevc |
| **S8 后半程 RSS 斜率** | **+47.4 MB/min** (主 +51.0) | **-0.8 MB/min** (主 -0.5) |
| **S8 RSS 峰值** | **6042 MB** | **5479 MB** |
| GPU 显存归属 | pid_ns_mismatch | pid_ns_mismatch |

✅ **两臂 S1~S8 全绿**。hevc constqp 斜率 +47.4 落在 +50 阈值内（临界但通过）。

### 0.3.3 `--mem-peak-mb` 自动标定更新（`_auto_peak_mb`）

实测峰值（bs=8 / 358s 标准素材）：

| Codec | Rate Mode | RSS Peak (MB) |
|-------|-----------|---------------|
| h264  | constqp   | 6730          |
| h264  | vbr_hq    | **7694**      |
| hevc  | constqp   | 6042          |
| hevc  | vbr_hq    | 5479          |

**基准值更新**：`_BASE_PEAK_MB = 7694.0`（取 h264 vbr_hq 实测最高值）  
**自动标定公式**：`max(2000, 7694 * max(1, bs) / 8 * 1.25)`  
- bs=8 → **9618 MB**（1.25× 余量，远高于实测 5479~7694 → 不误报）  
- bs=24 → 28852 MB（旧默认 12000 已被否决：bs=24 实测 12393~13877 全超上界但斜率正常）

⚠ **仍不能跨素材通用**（分辨率/时长影响工作集）—— 换素材时若 S8 报「峰值超上界」而斜率正常，应先确认是否需重新标定。

### 0.3.4 验收门禁现状（T4 bs=8）

```bash
# plan_implementation_gate: 99 PASS / 0 FAIL / 0 WARN / 2 SKIP（含 FIX-STRICT-EOS / FIX-B2-RC-MODE-REJECT）
# crf_cq_unification_verify --quick --no-gpu: 104 PASS / 0 FAIL / 0 WARN / 11 SKIP
# av1_pipeline_smoke (h264): 16/16 PASS
# av1_pipeline_smoke (hevc): 16/16 PASS
# pytest Accessory/test: 66 PASS
```

---

## 0.4 L40 仍需完成的项（阻塞：无 Ada 硬件）

| 项 | 内容 | 依赖 |
|----|------|------|
| **L40-1/2** | `av1_nvenc` CQ/QP 等质量标定（`-cq:v` / `-qp`） | L40 GPU |
| **L40-5** | AV1 长视频冒烟 S1~S8（实跑 av1_nvenc） | L40 GPU |
| **L40-6** | AV1 Level 1 `GetEncodePresetConfig code=12` 根因 | L40 GPU |

T4 无 AV1 NVENC ⇒ 以上均需 L40（Ada / sm89）。T4 侧已完成：
- H.264/HEVC constqp + vbr_hq 双臂 358s 完整验收（S1~S8 全绿）
- `--mem-peak-mb` bs=8 自动标定校准完成（9618 MB）
- B1/B2/B3 三阻塞项修复并 GPU 验证通过
- ESRGAN `strict_eos` 同构接入（3 处 fail-fast，观察模式兜底）

---

## 0.5 下一步（顺序有依赖，勿打乱）

| 步 | 动作 | 需 GPU | 前置 | 状态 |
|----|------|--------|------|------|
| 1 | 修 B3（显存口径：`None` 而非 `0.0` + 标注「不可归属」） | 否 | — | ✅ 已完成 |
| 2 | 修 B1（`_stream_begin` 保留 `_cached_sps_pps` + hevc 臂负向回归） | 是 | 步 1 | ✅ 已完成 |
| 3 | 重跑 S8 的 h264 vbr 臂（用 `vbr_hq`）—— 首次真正测到 h264 LA>0 路径 | 是 | 步 2 | ✅ 已完成 |
| 4 | B2 方案 A（脚本级禁用 `vbr`/`cbr` + 修 `_log_ready`） | 否 | 步 3 | ✅ 已完成 |
| 5 | 标定 `--mem-peak-mb`（按 T4 bs=8 实测 5479~7694 重新取值） | 否 | — | ✅ 已完成（`_auto_peak_mb` 更新） |
| 6 | **L40 复跑**：同素材 constqp + vbr_hq + av1 三臂同批，判 S8 原始问题 | 是（L40） | 步 2、4 | ⏸ 阻塞：需 L40 |
| 7 | L40 `av1_nvenc` CQ/QP 等质量标定（`QUALITY_MAP` / `QUALITY_MAP_QP` 落表） | 是（L40） | 步 6 | ⏸ 阻塞：需 L40 |

---

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
    --sides 6s,10s,legacy10s,gpu_l40 --out /tmp/table_l40.txt < /dev/null
# 判据：av1_nvenc 行 LOO ≤5.9；顺序无关断言通过
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
| 落表器 | `eqq_pool_fit_table.py --sides …,gpu_l40` | av1 行 LOO ≤5.9 + 顺序无关 ✅ |
| AC1/AC2/AC4 探针 | `av1_vp9_quality_matrix.py --only av1_nvenc` | 退出码 0；AC1 表值落带内 |
| GPU 判据 | `crf_cq_unification_verify.py --gpu …` | G7-6 PASS、FAIL=0 |
| AV1 冒烟 | `av1_pipeline_smoke.py` | 退出码 0；S1~S8 无 FAIL |
| 本仓判据/门禁/pytest | 同 T4 方案 §7 | FAIL=0 / 全绿 |
| 跨仓真源 | `VidUtils/verify/verify_quality_mapping.py` | ⑨ 组 14/14 |

---

## 8. 回滚

| 触发 | 动作 |
|---|---|
| av1 表 LOO 超门禁 | 不落表；保留 `gpu_l40` points 与报告，标注根因 |
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

> **⚠ 2026-10-06 更新：A~E 已在 T4 执行完毕，结论见 §0.3；F 已完成；G ✅ **已在 L40 完成**。**

| # | 待办 | 卡在哪 | 上机怎么做 | 完成判据 |
|---|---|---|---|---|
| **A** | ~~S8 定位~~ → **已由 D 项在 T4 完成**（§0.1/0.3） | — | — | ✅ T4 上 constqp 两臂（h264 -76.8 / hevc +47.4）均无泄漏；vbr_hq 斜率 +12.6/-0.8 亦落在 +50 内 |
| **B** | ~~显存归属口径复验~~ → **归因已修正为 PID namespace**（§0.1.3 B3，0.3 验证生效） | **结构性不可行**（容器 PID ns 与驱动侧不通） | 容器内 `nvidia-smi --query-compute-apps` 返宿主 pid，`/proc` 下不存在 | ✅ 已定位根因；显存报 `None`/`NA` 而非 `0.0`，报告显式写「不可归属（pid_ns_mismatch）」 |
| **C** | ~~S8 判据峰值上界校准~~ → **已按 bs=8 实测标定**（§0.3.3） | — | T4 实测四臂峰值 5479~7694 MB，旧值 12000 已被否决 | ✅ `_auto_peak_mb(8) = 9618 MB`（7694 × 1.25 余量），bs=8 下四臂全不误报 |
| **D** | ~~constqp 路径可在 T4 复现~~ → **已执行（2026-10-05/06）** | — | T4 `h264/hevc` × `constqp/vbr_hq`，358.76s 素材同批 | ✅ **不复现 L40 的 +149.5** ⇒ L40 疑为窗口伪影（同数据全程斜率随窗口 −786~+1434 跳变）；**S8 原始问题已由 L40 实测回答：无真泄漏** |
| **E** | 顺带复核项（非阻塞） | — | ①`av1_vp9_quality_matrix` 退出码口径（§0 遗留 1 vs 0）；②L40-6（AV1 Level 1 `code=12`）记录结论 | ② 记录为「驱动侧限制，AV1 恒走 Level 2/3 CLI」 |
| **F** | **新增**：修 B1 + 重跑 h264 vbr_hq 臂 + B2 方案 A + ESRGAN strict_eos 同构 | 需 GPU | 见 §0.1.5 步 2~4 / 本次 T4 跑批 | ✅ 全完成：T4 h264 vbr_hq 358s rc=0/S1~S8 绿、strict_eos 3 处接入并验证 |
| **G** | **L40 专项**：`av1_nvenc` CQ/QP 标定 + AV1 冒烟 + Level 1 根因 | 是（L40） | 同素材 constqp + vbr + av1 三臂同批 | ✅ **全完成**：CQ/QP 双轴落表（LOO 3.13/2.61）、冒烟 15/16 PASS（S8 通过）、Level 1 记录结论 |

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
    --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out temp/eqq_gpu_l40/cq_live_kids_play --tiers av1_nvenc \
    --duration 6 --src-is-prep < /dev/null
# 2 入池落表
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_l40 < /dev/null
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
> 本专项的 **L40-5** 是它首次在 GPU 上的完整回归。**S8 已闭环**（constqp/vbr 双臂斜率 ≤ +50），
> 定位工具已就绪，待办与读数判读表见 **§9.5**。
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
