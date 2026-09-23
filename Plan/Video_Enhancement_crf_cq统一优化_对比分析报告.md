# Video_Enhancement 编码质量参数（CRF/CQ）统一优化 —— 对比分析报告与方案设计

- 参照实现：`D:\Workspace_Python\VidUtils`（`convert_crf.py` + `vidcrop_hwaccel.py`）
- 被分析项目：`D:\Workspace_Python\Video_Enhancement\Video_Enhancement`
- 分析日期：2026-09-10
- 分析范围：① 原视频分割编码环节 ② 超分/插帧输出编码环节 ③ 合并编码环节

---

## 〇、结论摘要

| 结论 | 说明 |
|---|---|
| **核心缺陷** | Video_Enhancement **没有任何** crf↔cq 转换逻辑。同一个 `crf` 数字被原样下发给 libx264/-265 的 `-crf` 与 NVENC 的 `-cq:v` / SDK `targetQuality`，**量纲被当作等价**，实际感知质量差 5~10 个 CRF 档位 |
| **默认值三处打架** | config 里 IFRNet=18 / ESRGAN=23 / output=18；CLI help 三处全写 23；摘要打印写 21；底层 argparse 写 23；pipeline 内部写 21。**5 个数，4 种说法** |
| **合并环节硬伤** | `merge_videos_by_codec` 无论 codec 是什么都发 `-crf`（`video_utils.py:3098`）。若 `--output-codec hevc_nvenc`，`-crf` 被 ffmpeg **静默忽略**，编码器退回默认质量——用户以为设了 18，实际是编码器默认值 |
| **自动升级/降级放大问题** | `HardwareCapability.best_encoder()` 会把 libx264 静默升级为 h264_nvenc，crf 语义随之改变却无任何提示 |
| **两侧后端实现已分叉** | IFRNet 与 Real-ESRGAN 的 NVENC `-rc:v` 映射表不同（`constqp` 一个映射成 `vbr`，一个映射成 `constqp`），rate_mode/lookahead 默认值也不同 |
| **建议方案** | 移植 VidUtils 的 `QUALITY_MAP` 单一真源到 `src/utils/quality_map.py`，新增 `-ref` / `--cq-*` 参数族，把三个环节的"取值"统一收敛到一个 `resolve_quality()` 函数 |

---

## 一、参照实现：VidUtils 的做法（值得照抄的部分）

### 1.1 唯一真源映射表

`VidUtils/convert_crf.py:13-51` 定义 `QUALITY_MAP`，以 **libx264 CRF 为统一中间轴**，每个编码器只需一条线性关系：

```
value = a × x264_crf + b      再 clamp 到 [lo, hi]
```

| 编码器 | a | b | 范围 | 含义 |
|---|---|---|---|---|
| `libx264` | 1.0 | 0 | 0–51 | 基准轴 |
| `libx265` | 1.0 | +3.0 | 0–51 | 同画质需高 3 |
| `libvpx-vp9` | 1.98 | −14.46 | 0–63 | 刻度非线性，线性近似 |
| `libaom-av1` | 1.0 | +4.0 | 0–63 | 官方 AV1 CRF 23 ≈ x264 19 |
| `libsvtav1` | 1.0 | +6.0 | 0–63 | 速度换质量，需更高 |
| `librav1e` | 4.0 | −4.0 | 0–255 | 实测标定（非文档推导） |
| `av1_nvenc` | 1.0 | +6.0 | 0–51 | |
| **`h264_nvenc`** | 1.0 | **+5.0** | 0–51 | |
| **`hevc_nvenc`** | 1.0 | **+7.5** | 0–51 | |
| `h264_qsv` / `hevc_qsv` | 1.0 | +3.5 / +4.5 | 1–51 | |
| `h264_amf` / `hevc_amf` | 1.0 | +2.0 / +4.0 | 0–51 | |
| `h264_vaapi` / `hevc_vaapi` | 1.0 | +3.0 / +5.0 | 0–51 | |
| `*_videotoolbox` | −99/51 | +100 / +105 | 1–100 | **a 为负**：q 越大画质越好，语义反向 |

### 1.2 三个 API（`convert_crf.py:81/90/101`）

```python
from_x264_crf(codec, x264_crf)              # 基准 → 目标编码器
to_x264_crf(codec, value)                   # 目标编码器 → 基准
convert_quality(src_codec, src_val, dst)    # 任意互转（经基准轴）
```

### 1.3 参数解析策略（`vidcrop_hwaccel.py:2076 _resolve_quality_params`）

VidUtils 把"字面量"与"基准轴"两类参数**显式分开且互斥**：

| 参数 | 语义 |
|---|---|
| `--crf N` | 字面量，仅对 CPU 编码器生效，原样下发 |
| `--cq N` | 字面量，仅对 GPU 编码器生效，原样下发 |
| `--crf-ref N` | 以 libx264 CRF 为基准，按表换算到**任意**目标编码器 |
| `--cq-ref N` | 以 h264_nvenc CQ 为基准，按表换算到任意目标编码器 |

关键设计点（都值得照搬）：

1. **互斥校验**：`-ref` 与字面量不可同传，否则无法判断用户意图（`vidcrop_hwaccel.py:3091-3099`）。
2. **范围校验分轴**：`--crf/--cq` 允许 0–63（不同编码器量程不同），`-ref` 严格 0–51（基准轴量程）（`:3122-3136`）。
3. **降级时才换算、且明确告知**：只有"用户请求 GPU 编码器但降级为 CPU"这一条路径才做 `cq_to_crf()`，并且打印 `已将 --cq N（h264_nvenc 量纲）映射为 -crf M（等效视觉质量）`（`:2141-2149`）。用户显式给出的同族参数一律原样下发。
4. **特殊编码器兜底**：`librav1e` 不认 `-crf`，换算为 `-qp`（`:2065-2073` + `:2428`）；VP9/VP8 的 `-crf` 必须配 `-b:v 0` 才是纯恒定质量（`:2423-2425`）。
5. **注释里明确记录废弃历史**：旧 `cq_to_crf()` 里的 `+1/+4` 与本表方向相反，已废弃（`convert_crf.py:8-9`）——避免后来者重新引入。

---

## 二、Video_Enhancement 现状分析

### 2.1 环节 ① 原视频分割编码

| 位置 | 行为 |
|---|---|
| `src/utils/video_utils.py:2260-2276` `split_video_by_time` | `-c copy`，**不编码**，按关键帧切分 |
| `src/utils/video_utils.py:2063-2093` `normalize_video_timeline` | 时间轴异常时的**隐式重编码**：`crf: int = 18` 硬编码（`:2066`），命令里写死 `'-crf', str(crf)`（`:2082`） |
| 调用点 | `src/main_video_optimized.py:1417-1460`（检测到时间戳异常时触发） |

**问题 P1-①**：这是全项目唯一一处"分割态"编码，crf=18 硬编码、无 CLI 入口、无 codec 参数、写死 `-crf`。若后续给该环节引入 NVENC，`-crf` 会直接失效。

**说明**：严格说本环节当前"无编码"，所以"分割环节的 crf/cq 转换"目前是**空白而非错误**——但正因为空白，它无法参与统一，必须先把入口建出来。

### 2.2 环节 ② 超分/插帧输出编码（问题最集中）

数据链路：`main_video_optimized.py` → `processors/*_processor_video_optimized.py` → `external/{ifrnet,realesrgan}_video/`

#### (a) 参数来源与默认值现状

| 层 | 位置 | 默认值 |
|---|---|---|
| JSON 配置 | `config_manager.py:61` (ifrnet) | **18** |
| JSON 配置 | `config_manager.py:97` (esrgan) | **23** |
| JSON 配置 | `config_manager.py:109` (output) | **18** |
| 处理器 | `ifrnet_processor_video_optimized.py:118` | fallback **23** |
| 处理器 | `realesrgan_processor_video_optimized.py:144` | fallback **23** |
| 后端 argparse | `ifrnet_video/main.py:2257` | **23** |
| 后端 argparse | `realesrgan_video/main.py:1092` | **23** |
| 后端 pipeline | `ifrnet_video/pipeline.py:1088` | **21** |
| 后端 FFmpegWriter | `ifrnet_video/ffmpeg_io.py:594` | **23** |
| CLI help | `main_video_optimized.py:2158 / 2221 / 2289` | 三处全写 **23** |
| 摘要打印 | `main_video_optimized.py:424 / 447 / 483 / 488` | fallback **21** |

> **问题 P0-②a**：IFRNet 实际跑 18、ESRGAN 实际跑 23。同一条流水线两阶段质量档位天然不一致，而没有任何文档说明这是有意为之。CLI help 与实际生效值不符，用户按 help 理解必然出错。

#### (b) 软编 / 硬编的下发路径（**核心缺陷**）

**IFRNet 后端** `external/ifrnet_video/ffmpeg_io.py:633-707`：

```python
if crf == 0:                       # 无损特例（分 codec 正确映射，这块做得好）
    ...  libx265 → lossless=1 ; nvenc → -rc constqp -qp 0 ; libx264 → -qp 0
elif 'nvenc' in codec:
    _rc_v_map = {'vbr_hq': 'vbr_hq', 'qvbr': 'vbr_hq', 'constqp': 'vbr'}   # ← 疑点
    quality_args = ['-preset', preset,
                    '-rc:v', _rc_v, '-cq:v', str(crf), '-b:v', '0', ...]    # ← crf 原样当 cq
elif codec == 'libx265':
    quality_args = ['-preset', preset, '-crf', str(crf), '-x265-params', ...]
else:                              # libx264
    quality_args = ['-preset', preset, '-crf', str(crf), '-x264-params', ...]
```

**Real-ESRGAN 后端** `external/realesrgan_video/ffmpeg_io.py:809-894`：结构相同，`-cq:v', str(crf)`（`:832`）。

**NVENC SDK Level 1 直通**（不经 ffmpeg CLI）：
- `ifrnet_video/nvenc_sdk.py:962` `_tq = max(1, _qp_val)` → `targetQuality = CRF` 直接映射
- `realesrgan_video/main.py:789` `_sdk_qp = getattr(args, 'crf', 23)`
- `realesrgan_video/nvenc_sdk.py:950 / 964` 同样是 `max(1, _qp_val)`

> **问题 P0-②b（最严重）**：四条路径全部把 `crf` 的**数值**直接塞进 NVENC 的 `cq / targetQuality`。按 VidUtils 的映射表，x264 CRF 18 等效于 `h264_nvenc cq 23` / `hevc_nvenc cq 25.5`。当前实现下：
> - `--codec-ifrnet h264_nvenc --crf-ifrnet 18` → 实际下发 `cq 18`，等效 x264 CRF **13**，码率暴涨约 60–80%
> - `--codec-esrgan hevc_nvenc --crf-esrgan 23` → 下发 `cq 23`，等效 x264 CRF **15.5**，同样严重过保
>
> 反过来说也成立：习惯了 NVENC 数值的人传 `--crf 30`，落到 libx264 就是 CRF 30（糊），落到 NVENC 是 cq 30（等效 x264 25，还行）。**同一个数字的画质含义随 codec 漂移 ±7 档**。

> **问题 P0-②c**：`crf == 0` 的无损特例反而做对了（按 codec 分别映射），这说明"需要按 codec 区分"这件事代码作者是知道的，只是**有损路径没做**。属于遗漏而非设计选择。

#### (c) 两个后端的 NVENC RC 映射已经分叉

| | IFRNet `ffmpeg_io.py:684` | Real-ESRGAN `ffmpeg_io.py:827` |
|---|---|---|
| `constqp` | → `'vbr'` ❌ | → `'constqp'` ✅ |
| `vbr_hq` | → `'vbr_hq'` | → `'vbr_hq'` |
| `qvbr` | → `'vbr_hq'` | → `'vbr_hq'` |

且两处理器默认值也不同：

| | IFRNet processor `:120/:121` | Real-ESRGAN processor `:145/:146` |
|---|---|---|
| `rate_mode` | **`constqp`** | `vbr_hq` |
| `lookahead_depth` | **0** | 8 |

而 CLI help（`main_video_optimized.py:2167 / 2231`）对**两者**都写"默认 vbr_hq"、`:2170/2234` 都写"默认 8" —— IFRNet 侧的两条 help 是错的。

> 叠加影响：`constqp` 下 CRF 数值被当作**真实 QP**，`vbr_hq` 下被当作 **targetQuality**。同为 18，constqp 的 QP=18 与 vbr_hq 的 tq=18 又是两套语义，进一步放大 ②b 的漂移。

#### (d) 编码器自动升级 / 降级（`HardwareCapability.best_encoder`）

- IFRNet：`external/ifrnet_video/ffmpeg_io.py:104`，`nvenc_map = {'libx264':'h264_nvenc','libx265':'hevc_nvenc'}`
- Real-ESRGAN：`external/realesrgan_video/ffmpeg_io.py:233`，同表
- 双向：用户选软编且 NVENC 可用 → **自动升级**；用户选 NVENC 但不可用 → 自动降级

> **问题 P1-②d**：codec 被静默替换，但 crf 数值不变 → 语义被静默改变。用户在一台有 GPU 的机器上得到 h264_nvenc+cq18，在另一台无 GPU 的机器上得到 libx264+crf18，两者画质完全不同，日志里只有一行 probe 提示，没有质量等效换算。

### 2.3 环节 ③ 合并编码

三个合并入口，行为不一致：

| 入口 | 位置 | crf 来源 | 命令构造 |
|---|---|---|---|
| `merge_videos`（旧） | `video_utils.py:2307` | `config['crf']`，默认 **18**（`:2363`） | `'-crf', str(config['crf'])`（`:2404`） |
| `merge_videos_by_codec`（现行） | `video_utils.py:2841` | `params['crf']`，默认 **18**（`:2971`） | `'-crf', crf_str`（`:3098`） |
| `add_audio_to_video` | `video_utils.py:1028` | `config.get('crf', 18)`（`:1055`） | `'-crf', str(crf)`（`:1061`） |
| 最终合并 | `main_video_optimized.py:1862` | `output` section | 同上 |
| 单阶段内部合并 | `ifrnet_processor:368-381` / `realesrgan_processor:243-253` | **也取 `output` section** | 同上 |

> **问题 P0-③a**：`merge_videos_by_codec` **完全不判断 codec**，一律发 `-crf`。
> - `--output-codec h264_nvenc` → ffmpeg 报 "Option crf not found" 或直接忽略，编码器走默认质量，用户设置的 crf 彻底丢失。
> - `--output-codec libvpx-vp9` → `-crf` 无 `-b:v 0` 配套，退化成 constrained quality，不是恒定质量。
> - `--output-codec librav1e` → `-crf` 被静默忽略（VidUtils 已踩过此坑）。
>
> 相比之下**分段编码环节**至少还知道要按 codec 分支（虽然有损分支没换算），合并环节连分支都没有。

> **问题 P1-③b**：分段用 `models.{ifrnet,realesrgan}.crf`（18/23），合并重编码却用 `output.crf`（18）。
> - `--skip-interpolate` 单跑 ESRGAN：分段 crf 23 → 合并重编码 crf 18。**已编码到 23 的内容被 18 重新编码一次**，既浪费码率又无法挽回 23 已经丢掉的细节，属于纯粹的双重损失。
> - `--skip-upscale` 单跑 IFRNet：分段 18 → 合并 18，看似一致，但两阶段模式下最终 `-c:v copy` 不重编码，crf 由第二阶段决定——**同一条命令在不同模式下的质量出口不同**。

> **问题 P2-③c**：`merge_videos_by_codec` 的 `need_reencode` 决策（`:2932-2937`）只看 codec 是否在 `DIRECT_COPY_CODECS`，不看用户是否显式指定 `codec='copy'`；`main_video_optimized.py:1770` 是靠 `{**output_config, "codec": "copy"}` 硬塞进来的，绕过了决策分支。

---

## 三、问题清单

| ID | 严重度 | 环节 | 问题 | 位置 |
|---|---|---|---|---|
| P0-②b | 严重 | ② | crf 数值原样当 NVENC cq，软硬编量纲被当等价，画质漂移 ±7 档 | `ifrnet_video/ffmpeg_io.py:688`、`realesrgan_video/ffmpeg_io.py:832`、`ifrnet_video/nvenc_sdk.py:962`、`realesrgan_video/main.py:789` |
| P0-③a | 严重 | ③ | 合并环节不判断 codec，一律发 `-crf`；NVENC/AV1 下参数静默失效 | `video_utils.py:3098`、`2404`、`1061` |
| P0-②a | 严重 | ② | 默认 crf 三处打架（18/23/18），CLI help 与实际不符 | `config_manager.py:61/97/109`、`main_video_optimized.py:2158/2221/2289/424/447` |
| P0-②c | 严重 | ② | `crf==0` 无损路径按 codec 正确映射，有损路径未做——遗漏而非设计 | `ifrnet_video/ffmpeg_io.py:633-707` |
| P1-① | 高 | ① | 分割态唯一的重编码点 crf=18 硬编码、无 CLI 入口、写死 `-crf` | `video_utils.py:2066/2082` |
| P1-②d | 高 | ② | codec 自动升级/降级时 crf 语义被静默改变，无等效换算与提示 | `ifrnet_video/ffmpeg_io.py:104`、`realesrgan_video/ffmpeg_io.py:233` |
| P1-③b | 高 | ③ | 分段用 `models.*.crf`、合并用 `output.crf`，单阶段路径必然二次编码且质量倒挂 | `ifrnet_processor:368-381`、`realesrgan_processor:243-253` |
| P1-②e | 高 | ② | 两后端 NVENC RC 映射分叉（`constqp`→`vbr` vs `constqp`），rate_mode/LA 默认值也不同 | `ifrnet_video/ffmpeg_io.py:684` vs `realesrgan_video/ffmpeg_io.py:827`；`ifrnet_processor:120/121` vs `realesrgan_processor:145/146` |
| P2-③c | 中 | ③ | `codec='copy'` 未纳入 `need_reencode` 决策，靠上层硬塞绕过 | `video_utils.py:2932-2937` |
| P2-②f | 中 | ② | 无全局质量参数校验入口（仅 `main_video_optimized.py:932-934` 校验了 `models.*.crf` 范围，`output.crf` 无校验） | `main_video_optimized.py:932` |

---

## 四、统一方案设计

### 4.1 新增单一真源模块 `src/utils/quality_map.py`

从 `VidUtils/convert_crf.py` 移植 `QUALITY_MAP`（保留全部条目与注释，含"已废弃历史"说明），并补充 Video_Enhancement 特有内容：

```python
QUALITY_MAP = {                    # codec -> (a, b, lo, hi)   value = a*x264_crf + b
    'libx264':        (1.0,  0.0, 0, 51),
    'libx265':        (1.0,  3.0, 0, 51),
    'h264_nvenc':     (1.0,  5.0, 0, 51),
    'hevc_nvenc':     (1.0,  7.5, 0, 51),
    'av1_nvenc':      (1.0,  6.0, 0, 51),
    'libvpx-vp9':     (1.98, -14.46, 0, 63),
    'libaom-av1':     (1.0,  4.0, 0, 63),
    'libsvtav1':      (1.0,  6.0, 0, 63),
    'librav1e':       (4.0, -4.0, 0, 255),
    ...                            # qsv / amf / vaapi / videotoolbox 同 VidUtils
}

# 参数名与配套参数（VidUtils 分散在命令构造处，这里集中成表）
QUALITY_PARAM = {
    # codec: (主参数名, 需要的配套参数元组)
    'librav1e':    ('-qp',   ()),
    'libvpx-vp9':  ('-crf',  ('-b:v', '0')),
    'libvpx':      ('-crf',  ('-b:v', '0')),
}
# 其余：NVENC/QSV/AMF/VAAPI → '-cq:v' + ('-b:v','0')；libx*/libaom/libsvt → '-crf'

def from_x264_crf(codec, crf) -> Optional[float]
def to_x264_crf(codec, value) -> Optional[float]
def convert_quality(src_codec, src_val, dst_codec) -> Optional[float]

def resolve_quality(codec, *, crf=None, cq=None, crf_ref=None, cq_ref=None,
                    default_ref=21) -> Tuple[str, int, List[str]]:
    """返回 (参数名, 参数值, 配套参数列表)，供 ffmpeg 命令直接 extend。"""
```

**`resolve_quality` 的判定顺序**（与 VidUtils `_resolve_quality_params` 一致）：

1. `crf_ref` given → `from_x264_crf(codec, crf_ref)`
2. `cq_ref` given → 先 `to_x264_crf('h264_nvenc', cq_ref)` 得基准 → `from_x264_crf(codec, ·)`
3. `cq` given 且 codec 支持 cq → 原样下发
4. `crf` given 且 codec 支持 crf → 原样下发
5. `cq` given 但 codec 是软编 → `convert_quality('h264_nvenc', cq, codec)`，并**打印换算提示**
6. `crf` given 但 codec 是硬编 → `convert_quality('libx264', crf, codec)`，并**打印换算提示**
7. 均无 → `from_x264_crf(codec, default_ref)`

第 5/6 步是修复 P1-②d 的关键：**codec 自动升级/降级后，用户给的字面量按原量纲换算到新 codec，并明确告知**。

### 4.2 新增参数矩阵

> 注：需求清单里的 `--cq-ifrnet-crf` 判断为 `--cq-ifrnet-ref` 的笔误（其余三项均为 `-ref` 后缀，且 VidUtils 对应参数为 `--cq-ref`）。下文按 `-ref` 设计。

#### 环节 ② —— IFRNet 分段输出

| 参数 | 类型 | 语义 | 默认 |
|---|---|---|---|
| `--crf-ifrnet N` | int | 字面量，软编 `-crf` 原样下发 | — |
| **`--cq-ifrnet N`** | int | **新增**，字面量，硬编 `-cq:v` 原样下发 | — |
| **`--crf-ifrnet-ref N`** | int | **新增**，libx264 CRF 基准，按表换算 | 21 |
| **`--cq-ifrnet-ref N`** | int | **新增**，h264_nvenc CQ 基准，按表换算 | — |

#### 环节 ② —— Real-ESRGAN 分段输出

| 参数 | 类型 | 语义 | 默认 |
|---|---|---|---|
| `--crf-esrgan N` | int | 字面量 | — |
| **`--cq-esrgan N`** | int | **新增** | — |
| **`--crf-esrgan-ref N`** | int | **新增** | 21 |
| **`--cq-esrgan-ref N`** | int | **新增** | — |

#### 环节 ③ —— 最终合并

| 参数 | 类型 | 语义 | 默认 |
|---|---|---|---|
| `--output-crf N` | int | 字面量 | — |
| **`--output-cq N`** | int | **新增** | — |
| **`--output-crf-ref N`** | int | **新增** | 21 |
| **`--output-cq-ref N`** | int | **新增** | — |

#### 环节 ① —— 分割（补齐入口）

| 参数 | 类型 | 语义 | 默认 |
|---|---|---|---|
| **`--split-codec CODEC`** | str | `normalize_video_timeline` 重编码编码器 | `libx264` |
| **`--split-crf-ref N`** | int | 质量基准 | 21 |
| **`--split-preset PRESET`** | str | 编码预设 | `veryfast` |

#### 互斥与校验（照搬 `vidcrop_hwaccel.py:3091-3136`）

- 同一环节内：`{--crf-*, --cq-*}` 字面量组 与 `{--crf-*-ref, --cq-*-ref}` 基准组 **互斥**，同传即报错退出；字面量组内 `--crf-*` 与 `--cq-*` 可同时给（按实际 codec 二选一，打印选用了哪个）。
- 范围：`--crf-*` / `--cq-*` 允许 0–63；`*-ref` 严格 0–51。
- 校验位置：集中在 `main_video_optimized.py` 现有的 `_validate_*` 区（`main_video_optimized.py:932` 附近），**补上 `output.*` 与新增参数的校验**（修 P2-②f）。

### 4.3 统一后的默认值映射表

统一取 **libx264 CRF 21** 为全局基准（与 VidUtils `DEFAULT_CRF` 一致，也接近当前摘要打印里的 21）：

| 目标编码器 | 环节①分割 | 环节② IFRNet | 环节② ESRGAN | 环节③ 合并 |
|---|---|---|---|---|
| `libx264` | 21 | 21 | 21 | 21 |
| `libx265` | 24 | 24 | 24 | 24 |
| `h264_nvenc` | 26 | 26 | 26 | 26 |
| `hevc_nvenc` | 29 | 29 | 29 | 29 |
| `av1_nvenc` | 27 | 27 | 27 | 27 |
| `libsvtav1` | 27 | 27 | 27 | 27 |

> **变更影响**：当前 IFRNet 实际 crf=18 → 新基准 21（软编下略降码率）；ESRGAN 实际 23 → 21（略升）。硬编侧是**净收益**：`hevc_nvenc` 从"错误地 cq 18/23"变成"正确的 cq 29"，码率显著下降且感知质量对齐预期。
> 若不愿改变现网画质基线，只需把 `DEFAULT_REF` 设为 IFRNet=18 / ESRGAN=23 两个值即可，映射表机制不变——**这正是把基准抽出来的收益：调基线从改 N 处代码变成改一个常量**。

### 4.4 代码落点改造

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/utils/quality_map.py` | **新建**。移植 `QUALITY_MAP` + `resolve_quality()` + `crf_to_rav1e_qp()` + 参数名表 |
| 2 | `src/utils/config_manager.py` | 默认值：`:61` ifrnet crf 18 → 新增 `crf_ref: 21`；`:97` esrgan 23 → `crf_ref: 21`；`:109` output 18 → `crf_ref: 21`。保留 `crf` 字段做兼容（为空则走 `crf_ref`） |
| 3 | `src/utils/video_utils.py` | `merge_videos_by_codec:2994/3095-3099` 用 `resolve_quality()` 替换写死的 `'-crf', crf_str`；`merge_videos:2401-2406` 同改；`normalize_video_timeline:2066/2082` 接 `split_crf_ref`；`add_audio_to_video:1055-1063` 同改 |
| 4 | `src/utils/video_utils.py:2932` | `need_reencode` 决策纳入 `codec == 'copy'`（修 P2-③c） |
| 5 | `src/main_video_optimized.py` | 新增 10 个参数（4.2 表）；`:1026/1112/1197` 处接入 `config.set`；`:932` 区补校验与互斥；`:420-448`、`494-504` 摘要打印改用 resolve 后的真实 `(参数名, 值)` |
| 6 | `src/processors/ifrnet_processor_video_optimized.py` | `:118` 读 `crf_ref`；内部合并（`:368-381`）改用**本阶段** crf_ref 而非 `output.crf`（修 P1-③b） |
| 7 | `src/processors/realesrgan_processor_video_optimized.py` | 同上（`:144`、`:243-253`） |
| 8 | `external/ifrnet_video/ffmpeg_io.py` | `:684` `_rc_v_map['constqp']` 改 `'constqp'`（修 P1-②e）；`:686-707` 的 `-cq:v str(crf)` / `-crf str(crf)` 改为调用 `resolve_quality()` |
| 9 | `external/realesrgan_video/ffmpeg_io.py` | `:830-836` / `:854` / `:874` / `:892` 同样改调 `resolve_quality()` |
| 10 | `external/ifrnet_video/nvenc_sdk.py:944-989` | Level 1 直通：`targetQuality = from_x264_crf(codec, crf_ref)`；`constqp` 模式下 QP 语义保持独立（不套 targetQuality 换算，仅做范围 clamp 到 0–51） |
| 11 | `external/realesrgan_video/main.py:789` / `nvenc_sdk.py:935-980` | 同 #10 |
| 12 | `external/*/ffmpeg_io.py` `best_encoder` | 升级/降级后回传最终 codec，由调用方触发 `resolve_quality` 的第 5/6 步换算并打印（修 P1-②d） |

> **跨镜像同步提醒**：`external/ifrnet_video` 与 `external/realesrgan_video` 是互为镜像的两个包，#8/#10 与 #9/#11 **必须成对修改**（见 CODEBUDDY.md 架构说明）。`nvenc_sdk.py.bak` / `nvenc_sdk_bak.py` 不改动。

### 4.5 兼容性

- 保留 `--crf-ifrnet` / `--crf-esrgan` / `--output-crf` 旧参数，语义**不变**（字面量原样下发）。
- JSON 配置里旧的 `crf: 18` 继续生效；只有显式配置 `crf_ref` 时才走基准轴。建议同时在 `config/default_config.json` 补 `"// crf_ref"` 注释说明优先关系。
- 摘要打印统一改成 `编码器: hevc_nvenc | 质量: -cq:v 29 (基准 CRF 21)`，让换算结果可见。

---

## 五、验收用例

| # | 命令 | 期望 |
|---|---|---|
| 1 | `--codec-ifrnet libx264 --crf-ifrnet-ref 21` | 分段命令含 `-crf 21` |
| 2 | `--codec-ifrnet hevc_nvenc --crf-ifrnet-ref 21` | 分段命令含 `-cq:v 29`，日志有换算说明 |
| 3 | `--codec-esrgan libx265 --crf-esrgan-ref 21` | `-crf 24` |
| 4 | `--output-codec h264_nvenc --output-crf-ref 21` | 合并命令含 `-cq:v 26` + `-b:v 0`，**不再出现 `-crf`** |
| 5 | `--output-codec libvpx-vp9 --output-crf-ref 21` | 合并命令含 `-crf 27` 且**带 `-b:v 0`** |
| 6 | `--crf-ifrnet 18 --crf-ifrnet-ref 21` | 报错退出（互斥） |
| 7 | `--crf-ifrnet-ref 60` | 报错退出（超基准轴量程） |
| 8 | 无 GPU 机器 + `--codec-esrgan h264_nvenc --cq-esrgan 26` | 降级为 libx264，日志打印"已将 --cq 26（h264_nvenc 量纲）映射为 -crf 21" |
| 9 | `--skip-interpolate`（单跑 ESRGAN） | 分段与内部合并使用**同一**质量档，不再出现 23→18 的倒挂 |
| 10 | `--crf-ifrnet 0` | 仍走无损特例：libx265→`lossless=1`、nvenc→`-rc constqp -qp 0`、libx264→`-qp 0`（回归保护） |

---

## 六、建议实施顺序

1. **#1 建 `quality_map.py`**（纯新增，零风险）+ 单元自测换算表。
2. **#8/#9 改两个后端的有损下发路径**（修 P0-②b，收益最大）+ **#4 修 `constqp` 映射**（P1-②e）。
3. **#3 改 `merge_videos_by_codec`**（修 P0-③a；同时覆盖旧 `merge_videos` 与 `add_audio_to_video`）。
4. **#5 加 CLI 参数与校验** + **#2 默认值对齐**（修 P0-②a）。
5. **#6/#7 修单阶段质量倒挂**（P1-③b）+ **#1① 分割入口**（P1-①）。
6. **#10/#11 NVENC SDK Level 1 直通对齐** + **#12 自动升降级换算**（P1-②d）。
