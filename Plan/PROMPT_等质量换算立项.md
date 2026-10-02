# Video_Enhancement 等质量换算表立项 Prompt

> **姊妹文档（必须同步维护）**
> * VidUtils 侧同项目：`VidUtils/Plan/PROMPT_等质量换算立项.md`
> * VidUtils 侧质量参数方案：`VidUtils/Plan/VidUtils_质量控制参数修复方案.md`
>
> **本仓相关**
> * 既有质量参数方案：`Plan/Video_Enhancement_质量控制参数修复方案.md`（E0~E10 / A1~A12）
> * 其中 **§6.11.3** 是本立项的直接依据（等体积 vs 等质量口径分工 + rav1e 实测证据）
>
> **本文档只提需求与验收，不含实现结论；实现方案由执行者补。**

---

## 0. 关键点速查（执行者先读这一节）

### 0.0 实施状态快照（2026-09-30 更新，执行者必读）

> **命名已按实现统一**（本文档早期提案名与实际落地不同，以本表为准）：

| 本文档早期提案名 | **实际落地名（2026-09-30）** |
|---|---|
| 等体积表 `QUALITY_MAP` | **`SIZE_MAP`**（改名，语义即「等体积/文件大小优先」） |
| 等质量表 `QUALITY_MAP_QUALITY` | **`QUALITY_MAP`**（占用原名，语义即「等质量/画质优先」） |
| `--quality-mode volume\|quality` | `--quality-mode size\|quality` |
| `--quality-table volume\|quality`（D6） | `--quality-mode size\|quality`（与上同一开关） |

* **默认口径 = `quality`**（`convert_crf.py` / `quality_map.py` / 两个探针一致）；
  等体积路径完整保留，`--quality-mode size` 显式选择。
* **已落地**：D1（标定脚本）、D2（`QUALITY_MAP` 软编 5 条 + 跨仓逐条相等）、
  D3（回归判据）、D6（AC7 探针口径开关）、README 说明；VU 侧方案 §4.12 已记录。
* **未完成**（按算力分类详见 **§7.1**）：M1 多素材、M2 留一交叉验证、M3 rav1e speed 档、
  M4 硬编（**需 GPU**）、D2b constqp 轴、D4 VE 方案章节（本节即补）、D5 AGENTS.md。
* ⚠ **本 Windows/WSL checkout 无 `ffmpeg`/`libvmaf`**：§3 的环境基线指 Linux 容器；
  §7.1 中标注「CPU」的事项**仍须在带 ffmpeg+libvmaf 的 Linux 容器**执行。

### 0.1 为什么现在做（一条实测数据）

`librav1e` 用**已落表的等体积值** `(7.0032, −80.993)` 在门禁素材 `word_world_2.mp4`（687 帧）逐锚点实测：

| x264 crf | 表值 qp | 码率比 | ΔPSNR vs libx264 crf21 | AC7 判定 |
|---|---|---|---|---|
| 18 | 45  | 0.956 | **+0.59 dB** | PASS |
| 21 | 66  | 0.910 | **−1.21 dB** | PASS |
| 24 | 87  | 0.939 | **−2.57 dB** | FAIL |
| 27 | 108 | 0.995 | **−4.17 dB** | FAIL |
| 30 | 129 | 1.000 | **−5.79 dB** | FAIL |

**码率比恒定 0.91~1.00（等体积拟合很准），ΔPSNR 却单调恶化到 −5.79 dB。**
AC7 判据只测 crf 21 ⇒ 这个缺陷一直没被门禁暴露。**这就是要等质量表的直接原因。**

### 0.2 五个「只有实测才知道」的关键点（照做可省 1~2 周踩坑）

| # | 关键点 | 说明 |
|---|---|---|
| K1 | **libvmaf 选项名是 `model=`，不是 `model_version=`** | 本机实测 `model_version=` 直接报 `Error applying option 'model_version' ... Option not found`；默认值已是 `version=vmaf_v0.6.1` |
| K2 | **一次 VMAF 运行可产出多指标** | JSON 的 `pooled_metrics` 含 `vmaf` / `integer_vif_scale{0,1,2,3}` / `integer_adm_*` / `integer_motion` ⇒ 零额外开销拿到 VIF/ADM |
| K3 | **PSNR 口径错一条就全盘失真** | 必须 `-v info` + 显式 `[0:v][1:v]`；裸 `psnr` 实测差 3 dB（43.40 vs 46.58），`-v error` 让 ΔPSNR **恒为 0.00** |
| K4 | **预处理缓存会静默污染标定** | 曾因 `prep.mp4` 按文件名复用（不校验 `--src`/分辨率）导致 `libaom` 标定值全错；**标定片段建议 ≥10s**（2s clip 的 speed-10 值 qp 84.7 与门禁素材 qp 77 不符） |
| K5 | **「经中间编码器中转」的换算会漂移** | 本项目已两次踩坑：① `librav1e` 旧值经 libaom 中推，libaom 行重标后自相矛盾（80 vs 63）；② 分支已换算又被下发处二次换算（`-qp 66` → 夹成 **255**）。⇒ **新表一律从基准轴直接查表，禁止中转** |

### 0.3 本机能力边界（决定里程碑能否推进）

| 能力 | 状态 |
|---|---|
| libvmaf / psnr / ssim 滤镜 | ✅ 可用 |
| 软编 6 编码器（x264/x265/vpx-vp9/svtav1/aom/rav1e） | ✅ 全部可跑 |
| **NVENC / QSV / AMF** | ❌ **不可用**（`Cannot load libcuda.so.1`、无 `/dev/nvidia*`）⇒ **M4 必须换机** |

### 0.4 硬约束

* **双仓 `SIZE_MAP` / `QUALITY_MAP` 两张表都必须逐条相等**（VidUtils 判据 ⑨ 组断言）⇒ **等质量表也要两份同步副本**；
* **默认 `--quality-mode quality`**（与 `src/utils/convert_crf.py` 一致，2026-09-30 定案）；
  等体积路径**完整保留**，由 `--quality-mode size` 显式选择；判据脚本内**显式钉 `size`** ⇒
  G1~G6/G10 现有断言**数值不变地继续通过**；
* 所有脚本**一律加 `< /dev/null`**（后台进程组 + tty stdin 下会被 SIGTTOU 整组停住）。

---

## 1. 背景：为什么要做等质量表

本仓 `SIZE_MAP`（`src/utils/convert_crf.py`；2026-09-30 由 `QUALITY_MAP` 改名）采用
**等体积（equal volume）** 口径标定，与其余协作项目（VidUtils）语义一致。标定链路是：

```
锚点 libx264 CRF 18/21/24/27/30 (-preset medium)
  → 目标编码器扫 CRF/QP 记体积
  → 在 log(体积) 曲线上插值出**等体积**参数
  → 最小二乘拟合 value = a × x264_crf + b
```

### 1.1 已实测到的「等体积 ≠ 等质量」证据（2026-09-30，rav1e）

在门禁素材 `word_world_2.mp4`（687 帧）上，用本仓**已落表的等体积值** `librav1e = (7.0032, −80.993)` 逐锚点实测：

| x264 crf | 表值 qp | 码率比 | ΔPSNR vs libx264 crf21 | AC7 判定 |
|---|---|---|---|---|
| 18 | 45  | 0.956 | **+0.59 dB** | PASS |
| 21 | 66  | 0.910 | **−1.21 dB** | PASS |
| 24 | 87  | 0.939 | **−2.57 dB** | FAIL |
| 27 | 108 | 0.995 | **−4.17 dB** | FAIL |
| 30 | 129 | 1.000 | **−5.79 dB** | FAIL |

* **码率比恒定在 0.91~1.00** ⇒ 等体积拟合本身很准；
* **ΔPSNR 随 CRF 单调恶化到 −5.79 dB** ⇒ 同体积下画质持续劣化。

**结论**：等体积表在 rav1e 上只覆盖到默认工作点 crf 21；**非默认 CRF 的画质等价性无保证**。
这不是标定误差，而是「等体积口径的固有局限」——需要第二张**等质量表**来覆盖
「画质优先」的场景。

### 1.2 立项时的覆盖盲区（首版落地后已部分消除，见 §0.0）

| 现状 | 说明 |
|---|---|
| AC7 判据**只测 crf 21** | 默认工作点绿 ≠ 全 CRF 区间绿（上表 crf24~30 即为红） |
| 只有一张表 | 无「码率受限 / 画质优先」的选择开关 |
| 无客观质量口径的标定 | 现有标定全程只用体积，从未用 VMAF/PSNR 定标 |

### 1.3 与 VidUtils 侧的关系与交叉引用

**同步硬约束**：两仓的 `SIZE_MAP` / `QUALITY_MAP` **必须各自逐条相等**（VidUtils 判据 ⑨ 组 `[9-*]` 断言）。
⇒ **等质量表也必须是双仓同步的两份副本**，改动任一侧必须同步另一侧并回跑 ⑨ 组。

**章节对照**（便于两仓执行者互相查阅、避免重复劳动）：

| 本仓章节 | VidUtils 对应章节 | 关系 |
|---|---|---|
| §1.1 等体积≠等质量证据 | 该侧 §V9 + §4.1 表值说明 | **互补**：VU 侧有素材级标定数据，VE 侧有逐锚点 ΔPSNR 实测（−5.79 dB） |
| §3 本机环境基线 | 该侧 §4.2 上机前置自检 | **互补**：环境不同（VE 侧无 CUDA 需换机；VU 侧素材与路径不同） |
| §4.1 质量度量 | 该侧 §1「质量度量指标」 | **同源**：同为 VMAF + PSNR(+VIF/ADM) + 主观 AB |
| §4.2 标定流程 | 该侧 §2「标定流程」 | **同源**，插值基准从「体积」改为「VMAF」 |
| §4.3 配套参数锁定 | 该侧 §4 编码器覆盖范围 | **互补**：配套参数以 VE 实测为准 |
| §4.4 rav1e 特殊性 | 该侧 §4.11（2026-09-30 新增） | **强耦合**：rav1e 表必须按 `-speed` 档同步标定 |
| §5 交付物 | 该侧 §5 交付物 | **结构对齐**，文件名不同 |
| §6 三个实测铁律 | 该侧 §4.1 注意事项 | **互补**：VU 侧踩过 prep 缓存坑，VE 侧踩过二次换算坑 |
| §7 里程碑 M0~M6 | 该侧里程碑 M1~M4 | **阶段可对齐**（VU 无 NVENC 直连层，M4 内容不同） |
| §8 验收门禁 | 该侧 §4.1 门禁 | **互补**：门禁命令各自仓内 |
| 附录 架构差异 | — | **仅本仓**：ctypes 直连 SDK ⇒ constqp 轴也要给等质量值 |

**可复用资产（不要重写）**：

| 资产 | 位置 | 用途 |
|---|---|---|
| 无缓存标定骨架 | `VidUtils/probe/calibrate_soft_offsets_nocache.py` | 已实现独立工作目录 + `prep` md5 审计 + `--dense`；改「体积插值 → VMAF 插值」即可 |
| VMAF 实测用法 | 见本文档 §11 | `model=` 而非 `model_version=`（K1） |
| 度量口径同源实现 | `Accessory/probe/av1_vp9_quality_matrix.py` | 已封装码率/PSNR 采集 + 可用性探测 + 容忍带，直接扩展 `--quality-mode` |
| GPU 画质判据框架 | `Accessory/verify/crf_cq_unification_verify.py` 的 G7 组 | 已有真实素材 GPU 实跑 + 报告产出 |

**须避免的重复劳动**：素材预处理、VMAF 调用封装、PSNR 解析、报告渲染 —— 这些两仓同源，
建议**先在 VU 侧实现通用库、再复制到本仓**，避免两份实现漂移（`librav1e` 的链式换算漂移
就是两仓实现不一致导致的，见 §9 风险表）。

---

## 2. 目标

建立 **等质量（equal quality）** 换算表，使：

```
libx264 CRF 21  ≈  libx265 CRF ?  ≈  libvpx-vp9 CRF ?  ≈  libsvtav1 CRF ?  ≈  libaom-av1 CRF ?
                     librav1e QP ?  ≈  h264_nvenc -cq ?  ≈  hevc_nvenc -cq ?  ≈  av1_nvenc -cq ?
```

在**同一客观/主观画质水平**下互换，而非同文件大小。

---

## 3. 本机环境基线（2026-09-30 实测，执行者可直接依赖）

| 项 | 状态 | 备注 |
|---|---|---|
| **libvmaf** | ✅ 可用 | `ffmpeg -enable-libvmaf`；**选项名是 `model=`，不是 `model_version=`**（默认值已是 `version=vmaf_v0.6.1`） |
| libvmaf JSON | ✅ 可解析 | `pooled_metrics` 含 `vmaf` / `integer_vif_scale{0,1,2,3}` / `integer_adm_*` / `integer_motion` |
| psnr / ssim 滤镜 | ✅ 可用 | 口径见 §6.1 |
| 软编 6 编码器 | ✅ 全部构建含且可跑 | libx264 / libx265 / libvpx-vp9 / libsvtav1 / libaom-av1 / librav1e |
| **NVENC / QSV / AMF** | ❌ **本容器不可用** | `Cannot load libcuda.so.1`，无 `/dev/nvidia*`，`torch.cuda.is_available()=False` ⇒ 硬编部分**必须换机** |
| ffmpeg | 7.1（`/usr/bin`）+ 7.1+av1（`/usr/local/bin`） | 软编标定建议固定用同一二进制 |

### 3.1 可用作标定源的素材（`/workspace/input_videos/`）

| 文件 | 时长 | 分辨率 | 建议归类 |
|---|---|---|---|
| `new5_raw.mp4` | 26.8s | 1920×1080 | 实拍/日常（**V9 基准素材**） |
| `new4_raw.mp4` | 18.7s | 1920×1080 | 实拍/高熵 |
| `new4_raw_4k.mp4` | 18.7s | **3840×2160** | 高细节 / 4K 尺度效应 |
| `new2.mp4` | 33.2s | 1920×1080 | 待归类 |
| `new1.mp4` | 42.4s | 1280×720 | 待归类 |
| `word_world_2.mp4` | 27.5s | 720×576 | **G7/AC7 门禁素材**（必须纳入） |
| `wws3e02_26s.mp4` | 26.1s | 640×360 | 低复杂度（动画/平涂？） |
| `Earth.at.Night.in.Color.S02E01.mp4` | 1717s | 3840×2160 | 高动态/夜景（长，可切片） |
| `112 Max Bed Time.avi` | 493.8s | 720×480 | 待归类 |

**待补齐（当前素材库缺）**：屏幕内容/文字（合成字幕、UI）、暗场/高噪、纯动画。
这三类是 VMAF 最容易失准的区间，**缺了会让 M2 的交叉验证偏乐观**。

---

## 4. 技术路线

### 4.1 质量度量（三维并行）

| 指标 | 用途 | 采集方式（本机已验证） |
|------|------|------------------------|
| **VMAF** | 主指标，与主观相关性最好 | `ffmpeg -i dist -i ref -lavfi libvmaf=log_fmt=json:log_path=<out.json> -f null -`，取 `pooled_metrics.vmaf.mean` |
| **PSNR** | 平行**参考**（**soft，不判红**） | 见 §6.1 的**严格口径** |
| **VIF / ADM** | 免费附赠 | 同一份 VMAF JSON 的 `integer_vif_scale*` / `integer_adm*`，零额外开销 |
| 主观 AB | 最终定标（M6，可选） | ≥3 人双盲；ITU-R BT.500-13 |

> **效率提示**：`libvmaf` 的 `feature` 选项可让 PSNR/SSIM 在**同一次**运行中一起出，
> 避免"跑两遍滤镜"。执行者可在 M1 验证该写法。

> ⚠ **门禁语义（2026-09-30 定案）**：**只有 VMAF 判红**（`|ΔVMAF| ≤ 1.0`）。
> ΔPSNR / ΔPSNR-HVS 是**跨轴参考指标** —— 等质量表以 VMAF 定标，**同 VMAF 不蕴含同 PSNR**，
> 拿 0.3/0.5 dB 判红必然假阳性 ⇒ 二者**只 WARN（打印 + 计数，不影响退出码）**。
> 见 `verify_equal_quality.py`（`TOL_VMAF` 判红；`TOL_PSNR`/`TOL_HVS` 仅参考）。

### 4.2 标定流程

```
对每个编码器 × 每条素材：
  1. 统一预处理 → 目标分辨率/时长/fps，yuv420p（⚠ 见 §6.2 的缓存陷阱）
  2. libx264 -preset medium 扫 CRF 18/21/24/27/30（+ 建议 15/33 两端点做外推校验）
     → 每个锚点记录 (体积, VMAF, PSNR)
  3. 目标编码器扫 10~15 个参数点，覆盖锚点质量区间
     → 每个点记录 (体积, VMAF, PSNR)
  4. **在目标编码器的「参数 → VMAF」曲线上插值**，取与每个 libx264 锚点**等 VMAF**的参数值
     （VMAF 对参数单调，插值比体积插值稳）
  5. 对 (x264_CRF, 目标参数) 做最小二乘 → 等质量表
  6. 留一素材交叉验证：预测误差目标 ΔVMAF < 1.0、ΔPSNR < 0.3 dB
```

### 4.3 各编码器的固定参数（标定与下发必须一致，否则等效点漂移）

| 编码器 | 质量参数 | 必须锁定的配套参数 | 备注 |
|---|---|---|---|
| libx264 | CRF | `-preset medium` | 基准轴 |
| libx265 | CRF | `-preset medium` | |
| libvpx-vp9 | CRF | **`-b:v 0 -deadline good -cpu-used 2 -row-mt 1`** | 缺 `-b:v 0` 会退化为 constrained quality |
| libsvtav1 | CRF | **`-preset 8`** | 换 preset 等效点会漂 |
| libaom-av1 | CRF | **`-b:v 0 -cpu-used 6`** | 默认 cpu-used=1 极慢 |
| librav1e | QP | **`-speed <档>`** | ⚠ 见 §4.4，表值随 speed 变 |
| h264/hevc_nvenc | `-cq` | `-rc:v vbr_hq -b:v 0 -preset p4` | 需 NVIDIA 机 |
| av1_nvenc | `-cq` | 同上 | 量程 0~63；需 Ada 及以上 |
| h264/hevc_vaapi | `-qp` | — | 0~52 |
| h264/hevc_qsv | 待确认 | — | 需 Intel 机；`-preset` 是 int 0~7 |
| h264/hecv_amf / *_videotoolbox | 待确认 | — | 需 AMD / macOS |

### 4.4 rav1e 的特殊性（必须先定，否则表值无从谈起）

`-speed` 会**整体平移** rav1e 的码率曲线（实测：同 qp 下 speed 10 体积 ≈ 原生档 **1.40×**），
所以 **rav1e 的等质量表按 speed 档分别标定**，不能只出一张。

本仓已具备的机制（可直接复用）：
* `src/utils/quality_map.py` 的 `RAV1E_SPEED` / `_EQVOL_SPEED_OVERRIDE` / `_EQQUAL_SPEED_OVERRIDE`
  已实现"按 speed 档换表"（等质量档 `_EQQUAL_SPEED_OVERRIDE` **待 M3 标定回填**，当前为空 ⇒ 回落）；
* 已实测：`-speed 10` 下「等体积」与「等质量」**无法同时满足**
  （等体积解 ΔPSNR ≈ −2.7 dB；等质量解体积 +30%）。

> 🔗 **交叉引用**：VU 侧已于 2026-09-30 同步改造 —— `crf_to_rav1e_qp()` 由「经 libaom 中转的
> 链式推导」改为**直接查表**，并固定下发 `-speed 10`（详见 `VidUtils/Plan/VidUtils_质量控制参数修复方案.md`
> **§4.11**）。⇒ 本仓的 rav1e 等质量表须与其保持同一 speed 档口径。

⇒ **本项目要为 rav1e 产出「等质量 × 各 speed 档」的表**，这正是本项目的核心价值点之一。

---

## 5. 交付物

| # | 交付物 | 说明 | 状态（2026-09-30） |
|---|---|---|---|
| D1 | `Accessory/probe/calibrate_equal_quality.py` | 标定脚本；**必须无缓存**（§6.2）、支持多素材/多编码器/`--quick` | ✅ 已落地（含 `--selftest`/`--resume`/rav1e 分档） |
| D2 | **`QUALITY_MAP` 表（等质量）** | **新增**，与 `SIZE_MAP`（等体积，原名 `QUALITY_MAP`）**并存不覆盖**；双仓各一份且逐条相等 | ✅ 软编 5 条已落表；⚠ **单素材**、硬编未覆盖 |
| D2b | **`QUALITY_MAP_QP`（constqp/QP 轴等质量，仅本仓）** | `external/*/nvenc_sdk.py` 走 ctypes 直连，`to_constqp_qp()` 需 QP 轴等质量值 | ❌ 空表（软编行可 CPU 镜像；NVENC 行**需 GPU**，见 §7.1） |
| D3 | `Accessory/verify/verify_equal_quality.py` | 回归判据：主门禁 VMAF；PSNR 为**参考** | ✅ 已落地：主门禁 `\|ΔVMAF\|≤1.0` **判红**；ΔPSNR/ΔPSNR-HVS 为**交叉参考**（WARN 不判红，2026-09-30 定案） |
| D4 | 方案文档新章节 | 记录方法/素材集/拟合参数/误差分析/已知局限；写入 `Plan/` | ✅ 本文档 §0.0 / §7.1 即补；VU 侧 §4.12 已有 |
| D5 | README / AGENTS.md 更新 | 说明何时用等体积表、何时用等质量表 | ⚠ README ✅；**AGENTS.md 未更新** |
| D6 | AC7 判据扩展 | `Accessory/probe/av1_vp9_quality_matrix.py` 增加 `--quality-mode size\|quality`（默认 `quality`） | ✅ 已落地（开关名按实现统一为 `--quality-mode`） |

### 5.1 CLI / API 兼容要求

- 新增 `--quality-mode size|quality`（**默认 `quality`**，与 `convert_crf.py` 一致）；
- `quality_map.resolve_quality()` 增加 `table=` 参数（**一次性覆盖**，无副作用）；
  默认走当前活动表（`quality` 口径 = `QUALITY_MAP`，未覆盖编码器**回退 `SIZE_MAP`**）；
- **零侵入**：等体积路径行为完整保留，`Accessory/verify/crf_cq_unification_verify.py`
  的 G1~G6/G10 现有断言**数值不变地继续通过**（判据内显式钉 `size`）。

---

## 6. 三个必须遵守的实测口径（都是本项目踩过的坑）

### 6.1 PSNR 口径（错一条就全盘失真）

```bash
# ✅ 唯一正确写法：-v info + 显式 [0:v][1:v] 标签
ffmpeg -hide_banner -v info -i "$DIST" -i "$SRC" -frames:v "$N" \
       -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
  | grep -oP 'average:\s*\K[0-9.]+' | tail -1
```

* ❌ 裸 `-lavfi psnr`（不带 `[0:v][1:v]`）会走出**不同结果**（实测 43.40 vs 正确 46.58）；
* ❌ `-v error` 会压掉 psnr 滤镜的 INFO 级汇总行 ⇒ **ΔPSNR 恒为 0.00 的假象**。

码率口径：`ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of csv=p=0 <f>`，
与判据脚本 `Ctx.media_info` 同源；注意未加 `-an` 时**含音轨**，软编与候选须同口径。

### 6.2 预处理缓存陷阱（本项目已中招）

`VidUtils/probe/calibrate_soft_offsets.py` 曾按文件名复用 `prep.mp4`、**不校验 `--src`/分辨率**，
导致「两条不同素材跑出几乎相同的体积」——据此得出的 `libaom` 标定值一度全错
（详见本仓 `Plan/Video_Enhancement_质量控制参数修复方案.md` **§6.10**）。

> 🔗 **交叉引用**：该坑已在 VU 侧修复并沉淀为 `VidUtils/probe/calibrate_soft_offsets_nocache.py`
> （独立工作目录 + `prep` md5 审计 + `--dense` 加密扫描），**直接复用，不要重写**。

* 新脚本**每次运行独立工作目录**（目录名带 素材_分辨率_时长 指纹）、`prep` 强制重建、
  **打印 `prep` 的 md5** 以便审计；
* 若复用旧脚本，**换素材前必须先删 `prep.mp4`**。

### 6.3 帧数核对

`-frames:v N` 的 `N` 必须取自源；容器元数据 `nb_frames` 与 `duration×fps` 可能不一致
（`-c copy` 分段常见）⇒ 取二者较大值，并在报告里记录实际取值。

---

## 7. 里程碑

| 阶段 | 交付 | 验收标准 | 算力需求 | 当前状态（2026-10-01 更新） |
|---|---|---|---|---|
| **M0** | D1 骨架 + 口径自检 | 用 `word_world_2.mp4` 跑通 libx264/libx265/librav1e 三点；PSNR/VMAF 数值与手工命令**逐位一致** | **CPU** | ✅ 已完成（Linux 容器实跑通过；D1 harness 两仓同源，附 `--selftest` 19 项） |
| **M1** | 3 条核心素材 × 4 软编编码器 | 单素材 ΔVMAF < 1.5、ΔPSNR < 0.3 dB | **CPU** | ✅ **已完成**：4 软编档全部落表（LOO 4.49~6.71），表值见 `convert_crf.QUALITY_MAP`（两仓逐条相等） |
| **M2** | 补齐素材 + 留一交叉验证 | 留一法 ΔVMAF < 1.0、ΔPSNR < 0.3 dB | **CPU**（+ 素材采集含人力） | ⚠ **LOO 已跑，未达 1.0**。素材池扩至 **11 个(素材,口径)样本**（两仓 7+4 条 + BBC 实拍 3 条），锚点已**两仓统一到 `18/21/24/27/30`**（补测缺口后做同批素材可比 LOO，10 个对比全优）。根因＝**跨素材结构上限**，四条已排除。**2026-10-02 仓主裁定门禁按编码器分档：软编 ≤5.9 / rav1e ≤7.5**（依据 rav1e 训练内误差 0.78~2.99 vs 软编 0.03~2.44 = qp 刻度本质更差）。工具：`Accessory/probe/loo_equal_quality.py`（两仓同源） |
| **M3** | rav1e 等质量 × speed 档 | 至少覆盖 `speed 0/10` 两档；给出"等质量 vs 等体积"差异量化 | **CPU** | ✅ **已完成**：VU 侧 7 素材 × 105 点 × 两档全部跑完（rc=0），LOO 首次可验证（native 7.41 / @10 7.15，按 rav1e 专用门禁 ≤7.5 **达标**）。native 进 `QUALITY_MAP`(7.9348,−96.0822)、`-speed 10` 进 `quality_map._EQQUAL_SPEED_OVERRIDE`(7.8373,−95.5520) |
| **M4** | 硬编覆盖（NVENC h264/hevc/av1） | B/C 组入表 | **GPU**：h264/hevc_nvenc **T4 即可**；`av1_nvenc` **必须 L40/Ada** | ❌ 未做（本容器无 GPU） |
| **M5** | D2~D6 落地 + 双仓同步 + 门禁 | 见 §8 | CPU（+ M4 的 GPU 部分） | ✅ D1~D6 全落地（**D5 已于 2026-10-01 补**）；LOO 工具已抽出为两仓同源独立脚本 |
| **M6**（可选） | 主观 AB 测试 | 主观与 VMAF 预测一致性 > 85% | **人力** | ❌ 未做 |

> ⚠ **2026-10-01 定案（戊方案，表值已落）**：`QUALITY_MAP` 为 **9 素材**标定
> （覆盖立项 §3 全部 6 类内容），两仓逐条相等。**LOO worst |ΔVMAF| = 4.13~7.86**，
> 达不到 M2 的 <1.0 ⇒经仓主裁定**放宽至 ≤5.9**，精度边界已在
> `convert_crf.QUALITY_MAP` 注释 / `README.md` / 标定报告 §3.2 三处标注。
> 根因为**结构性上限**，四条排除性证据（详见报告 §3.3）：非过拟合（svtav1 训练内 Δ 8.29~11.22）、
> 非素材不足（4→9 素材几无改善）、非表格式（每素材专属表 LOO 4.02~13.77 更差）、
> 非锚点位置（素材间离散度随 crf **下降**：aom 51%→9%，故"移出平坦区"是负收益）。
> 要更低误差只能改**基准指标**或引入**逐素材在线探测**（生产时先测 VMAF 再查表）。
> ⚠ 同时修掉一个判据漏洞：二次模型曾报「LOO 全 0.000✅」，实为**假通过**
> （预测越界 ⇒ 回查 `None` ⇒ 0 评估点被当成 `worst=0`），现已在 LOO 脚本加断言。

### 7.1 未完成事项按算力分类（2026-09-30，执行者按此排期）

**A. 仅需 CPU（软编 + 文档 + 逻辑断言）** —— 但须在带 `ffmpeg`+`libvmaf` 的 **Linux 容器**执行
（本 Windows/WSL checkout 无 ffmpeg，跑不了）：

| 事项 | 归属 | 说明 |
|---|---|---|
| M0 口径自检 | M0 | 手工命令与脚本数值逐位比对 |
| M1 多素材标定（3 核心素材 × 4 软编） | M1 | 复用 `calibrate_equal_quality.py --src A --src B …` |
| M2 补齐素材 + 留一交叉验证 | M2 | 素材采集属人力；标定本身纯 CPU |
| M3 rav1e 等质量 × speed 档 | M3 | 回填 `_EQQUAL_SPEED_OVERRIDE`（原生/speed10 两行） |
| D2b 的**软编 QP 行**（libx265/libsvtav1/librav1e 的 `-qp`） | D2b | 镜像 `QUALITY_MAP` 即可，无需 GPU |
| D4 方案章节 / D5 AGENTS.md / 命名文案一致性 | D4/D5 | 文档 |
| "双重换算"针对性断言（§9） | 判据 | 纯逻辑 |
| M5 的软编部分 + 双仓 ⑨ 组门禁 | M5 | 门禁需 ffmpeg（CPU） |

**B. 必须 GPU（NVENC 直连，T4 / L40 分档）**：

| 事项 | 最低硬件 | 说明 |
|---|---|---|
| M4 `h264_nvenc` / `hevc_nvenc` 的 `-cq` 等质量标定 | **T4**（Turing，支持 H.264/HEVC NVENC） | 等质量 `-cq` 轴未标（`QUALITY_MAP` 未覆盖硬编，当前回退 `SIZE_MAP`） |
| M4 `av1_nvenc` 的 `-cq` 等质量标定 | **L40 / Ada**（T4 **无 AV1 NVENC**，报 `No capable devices found`） | 量程 0~63 |
| D2b 的 **NVENC QP 行**（`to_constqp_qp` 的 constqp 轴） | h264/hevc **T4**；av1 **L40/Ada** | `av1_nvenc` QP 尺度 ×3 已由 L40 实测确认 |
| 生产管线 GPU 实跑判据（G7/G8 等） | **T4 / L40** | 需真实素材 GPU 编码 |

**C. 需其他硬件（非 T4/L40）**：QSV → Intel；AMF → AMD；VideoToolbox → macOS。
（量程/等效点需对应机型实测，见 §4.3）

**D. 需人力**：M6 主观 AB（≥3 人双盲，ITU-R BT.500-13）。

> ⚠ **M4 之前的所有结论都不能外推到 NVENC**：CPU 标定容器无 CUDA，硬编必须换机；
> 且 **T4 与 L40 不能互相替代**（T4 无 AV1 NVENC）。

---

## 8. 验收门禁（必须全绿）

| 门 | 命令 | 判据 | 算力 |
|---|---|---|---|
| 本仓静态判据 | `python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null` | **FAIL = 0**（现有 G1~G6/G10 断言数值不变；判据内显式钉 `size`） | CPU |
| 本仓门禁 | `python3 Accessory/verify/plan_implementation_gate.py < /dev/null` | **FAIL = 0** | CPU |
| pytest | `python3 -m pytest Accessory/test -q` | 全绿 | CPU |
| 等质量专用判据 | `python3 Accessory/verify/verify_equal_quality.py < /dev/null` | 主门禁 ΔVMAF 在门限内；ΔPSNR/ΔPSNR-HVS 为**参考（WARN 不判红）**（**默认 6s，须与标定同口径**） | CPU |
| **跨项目真源一致** | `python3 VidUtils/verify/verify_quality_mapping.py < /dev/null` | ⑨ 组 **14/14**（`SIZE_MAP` + `QUALITY_MAP` 逐条相等）。⚠ 该脚本在 **VU 仓**执行，两仓表任一不同步即红 | CPU |
| AC7 扩展 | `python3 Accessory/probe/av1_vp9_quality_matrix.py --quality-mode quality --src <素材> < /dev/null` | 退出码 0（无 FAIL） | CPU（软编部分）／GPU（硬编条目） |

**操作铁律**：所有脚本**一律加 `< /dev/null`**（后台进程组 + tty stdin 下会被 SIGTTOU 整组停住）。

---

## 9. 风险与对策

| 风险 | 证据/对策 |
|---|---|
| VMAF 对动画/屏幕内容失准 | 已知短板 ⇒ 必须补这三类素材；辅以 VIF/ADM 与主观 AB |
| **「经中间编码器中转」的换算会漂移** | ⚠ **本项目已两次踩坑**：① `librav1e` 旧值经 libaom 中推，libaom 重标后自相矛盾；② 分支已换算又被下发处二次换算（`-qp 66`→夹成 255）。⇒ **新表一律直接从基准轴查表，禁止中转**；并加"双重换算"的针对性断言（⚠ **仅部分**：VE 侧 `_crf_original` 防跨段二次换算已加；等质量路径的专项断言**待补**） |
| 表值随编码器/speed/preset 漂移 | 配套参数必须锁定（§4.3）；CI 定期回跑判据 |
| 素材集不具代表性 | 按内容类型分类覆盖；缺的三类必须补（§3.1） |
| rav1e 标定极慢 | 实测原生档 ≈0.011× 实时 ⇒ 标定时 rav1e 用 `-speed 10` 提速，但**表按档分别落**（原生档 / speed10 各一行，见 M3）；表注写明口径 |
| 本机无 GPU，NVENC 无法验证 | M4 单列；软编部分（M0~M3）不阻塞 |
| 短 clip 不具代表性 | ⚠ 实测 2s clip 的 speed-10 标定（qp 84.7）与门禁素材（qp 77）不符 ⇒ **标定片段建议 ≥10s** |

---

## 10. 与现有体系的兼容

* **不删除/不覆盖**等体积表（现名 `SIZE_MAP`），保留给「码率受限、文件大小优先」场景；
* 等质量表（现名 `QUALITY_MAP`）供「画质优先、存储/带宽次要」场景；
* 两者**并存**，由 `--quality-mode` / `table=` 选择，**默认 `quality`**
  （等体积路径仍完整保留，`--quality-mode size` 可切回）；
* 双仓（VE / VidUtils）**两张表都必须同步**，改一侧必回跑 ⑨ 组。

---

## 11. 立即可执行的第一步

```bash
cd /workspace/Video_Enhancement

# 0) 口径自检（M0 的核心：先证明度量管线正确，再谈标定）
SRC=/workspace/input_videos/word_world_2.mp4
N=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$SRC")
ffmpeg -nostdin -y -v error -i "$SRC" -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p /tmp/a.mp4
# PSNR（唯一正确口径）
ffmpeg -hide_banner -v info -i /tmp/a.mp4 -i "$SRC" -frames:v "$N" \
       -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -oP 'average:\s*\K[0-9.]+' | tail -1
# VMAF（注意选项名是 model=，默认已是 vmaf_v0.6.1；不要写 model_version=）
ffmpeg -hide_banner -v info -i /tmp/a.mp4 -i "$SRC" -frames:v "$N" \
       -lavfi "libvmaf=log_fmt=json:log_path=/tmp/vmaf.json" -f null - 2>&1 | tail -2
python3 -c "import json;print(json.load(open('/tmp/vmaf.json'))['pooled_metrics']['vmaf']['mean'])"

# 1) 标定脚本已落地（D1）：Accessory/probe/calibrate_equal_quality.py
#    ✅ 无缓存 + prep md5 审计 + VMAF 插值 + --quick/--resume + rav1e 分档

# 2) 若要重跑/扩样（M1/M2/M3）——纯 CPU，需在带 ffmpeg+libvmaf 的 Linux 容器：
#    python3 Accessory/probe/calibrate_equal_quality.py \
#        --src <素材A> --src <素材B> --src <素材C> --duration 6 --resume
#    librav1e 分档（M3）：--rav1e-speed native,10
#    ⚠ 硬编（M4）本机无法跑，须换 T4（h264/hevc）或 L40/Ada（av1_nvenc）
```

---

## 12. 参考资料

* 既有质量参数方案（本仓）：`Plan/Video_Enhancement_质量控制参数修复方案.md`
  —— 重点 §6.8（V9 多素材复核）、**§6.11.3（等体积 vs 等质量口径分工与 rav1e 实测证据）**、§6.9（门禁基线）
* 姊妹方案（VidUtils）：`VidUtils/Plan/VidUtils_质量控制参数修复方案.md` —— §V9、§3、§4
* 无缓存标定实现：`VidUtils/probe/calibrate_soft_offsets_nocache.py`
* AC7 探针（度量口径同源，可直接扩展）：`Accessory/probe/av1_vp9_quality_matrix.py`
* 判据脚本（G7 组已有 GPU 实跑框架）：`Accessory/verify/crf_cq_unification_verify.py`
* Netflix VMAF：`libvmaf` 的 `filter=libvmaf` 选项（`model=` / `feature=` / `log_fmt=json`）
* ITU-R BT.500-13 主观测试方法学（M6）

---

**优先级**：P1（画质一致性是视频增强管线的核心竞争力；当前非默认 CRF 已实测出 −5.79 dB 的画质落差）
**预估工期**：M0~M3 约 1~2 周（纯 CPU 可完成）；M4 需 NVIDIA 机（h264/hevc 可 T4、av1 需 L40/Ada）；M6 需人力
**负责人**：待指派
**评审人**：需包含有主观测试经验、且熟悉本仓 NVENC 编码路径的工程师
**阻塞项**：
* **CPU 事项**（M0~M3、D4/D5）——须在**带 ffmpeg+libvmaf 的 Linux 容器**执行（当前 Windows/WSL checkout 无 ffmpeg）；
* **GPU 事项**（M4、D2b 的 NVENC 行）——h264/hevc_nvenc 需 **T4 及以上**；`av1_nvenc` 需 **L40/Ada**（T4 无 AV1 NVENC）；
* §3.1 的三类缺失素材（屏幕内容/暗场高噪/纯动画）需补齐（人力）。

---

## 附：与 VidUtils 侧同项目的差异（执行者注意）

| 项 | VidUtils | Video_Enhancement（本仓） |
|---|---|---|
| 定位 | 裁剪/转码工具，单文件命令行 | 插帧+超分**增强管线**，分段编码 + NVENC SDK 直通 |
| 默认质量参数暴露 | `--crf/--cq/--qp/--crf-ref` 齐全 | 主要走配置 + `SIZE_MAP` / `QUALITY_MAP` 内部换算，用户直给质量较少 |
| 硬编路径 | ffmpeg CLI 下发 | `external/*/nvenc_sdk.py` **ctypes 直连 SDK**（不过 ffmpeg） |
| 质量判据 | `verify/verify_quality_mapping.py` ⑨/⑪ 组 | `crf_cq_unification_verify.py` G1~G10 + `av1_vp9_quality_matrix.py` |
| 表副本 | `VidUtils/convert_crf.py` | `src/utils/convert_crf.py`（**两副本须逐条相等**） |

⇒ 本仓多一处「ctypes 直连 SDK」的量纲校验点：`to_constqp_qp()` 的 QP 刻度层
（`av1_nvenc` ×3 已由 L40 实测确认）需在等质量表中**一并给出 constqp 轴的对应值**
（即 **D2b `QUALITY_MAP_QP`**，⚠ **未做**：软编行可 CPU 镜像，NVENC 行需 T4/L40）。
