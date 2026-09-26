# 立项 Prompt：读帧器两条解码路径（NVDEC vs 软解）产出的 RGB 不一致

> ## ✅ 执行状态（2026-09-17 Linux + GPU 完成）
>
> **诊断资产已就绪，结论已落地**：`Accessory/probe/reader_rgb_path_diagnose.py`
> （§3 步骤 1~4 的可执行版本，锁步流式比对不落盘，输出 JSON + Markdown 报告模板）。
>
> ### ⚠️ 两条实测纠正（本文档给的复测命令会因此得出错误结论）—— 已验证
>
> 1. **`-hwaccel auto` 的返回码不能判定「NVDEC 可用」。**
>    实测（Windows，**无 NVIDIA GPU**）：`-hwaccel cuda` 正确报
>    `Device creation failed`，而 `-hwaccel auto` **rc=0** 并静默退到 `dxva2`
>    （`Using auto hwaccel type dxva2`）。本文档 §4 的复测命令用的正是 `-hwaccel auto`
>    ⇒ 在无 NVIDIA 卡的机器上会**假阳性**地"复现 NVDEC 现象"。
>    判活必须**点名逐个试**（`cuda` 优先）+ 检查 stderr 失败标记。
> 2. **钉住矩阵/范围不需要 libzimg。**
>    本文档 §2.4 的两个判断（`scale` 的 `in_range`/`in_color_matrix` 是 no-op、
>    该 build 没有 `zscale`）都成立，但据此推出的"必须换带 libzimg 的 build"
>    **是多余前提** —— 把 `-color_range` / `-colorspace` 放在 `-i` **之前**
>    就能真实改变 RGB 输出（实测 sha 变化，非 no-op）。
>    附两坑：**没有 `bt601` 取值**（601 用 `smpte170m`/`bt470bg`）；输入侧选项必须在 `-i` 前。
>    又：本 build 的 `zscale` 只有整数型 `range/primaries/transfer`，
>    **没有** `matrixin=`/`matrixout=`，照抄 §3 步骤 2 会直接 `Option not found`。
>
> ### ✅ Linux + NVDEC 完成（2026-09-17，Tesla T4 / CUDA 13.0）
>
> **对齐方案**：`scale=in_color_matrix=bt601:in_range=tv` 显式加入 IFRNet/ESRGAN 读帧器 `-vf`
>
> **验证结果**：
> - NVDEC (nv12) 与软解 (yuv420p) 两条路径**逐字节一致**（强制 NVDEC/软解双路对照 150 帧、多码率/分辨率验证）
> - reader 忠实性：0 差异（reader vs 同参数参考解码）
> - 诊断脚本 `--sweep` 找到与 NVDEC 逐字节相等的参数组：`in_color_matrix=bt601:in_range=tv`
> - IFRNet/ESRGAN 双读帧器在软解与 NVDEC 强制下均逐字节一致
>
> **关键代码变更**：
> - `external/ifrnet_video/ffmpeg_io.py`: 添加 `scale=in_color_matrix=bt601:in_range=tv` 到 `-vf`
> - `external/realesrgan_video/ffmpeg_io.py`: 使用 `.filter('scale', in_color_matrix='bt601', in_range='tv')`
>
> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。
> 立项时间：2026-09-15　立项人：门禁强化会话
> **完成时间：2026-09-17**　立项人：门禁强化会话
> 数据来源：**2026-09-17 Linux + Tesla T4 实测**

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| 定性：差异发生在哪一层 | ✅ 已定性 | Y/UV 平面**逐字节相同** ⇒ 差异 100% 在 `nv12→rgb24` 转换环节 |
| reader 本身是否忠实 | ✅ 已验证 | 与**同参数**参考解码逐帧字节比对 0/578 不一致 |
| 差异幅度量化 | ✅ 已量化 | 见 §2.2 |
| 元凶（哪条路径用了哪个矩阵/范围） | ⬜ **本立项核心待办** | 本 build 的滤镜无法钉住，见 §2.4 |
| 生产影响评估 | ⬜ 待办 | 与 `READER_HWACCEL`（P2-2 自适应）的交互，见 §2.5 |

---

## 0. 任务

回答并落地一个问题：

> 同一条素材、同一份 `rgb24` 输出契约，**NVDEC 硬解路径**与**CPU 软解路径**
> 为什么产出的 RGB 不同（幅度可达 ±29、约 60% 像素），
> 哪一条才是「正确」的，以及**是否需要**（以及在何处）对齐？

产出物：一份**用可判定的实验**钉住矩阵/范围选择的诊断报告 + 是否对齐的结论；
若结论是「需要对齐」，再给出对齐点（读帧器 `-vf` 参数 / 元数据注入 / 消费端约定）。

---

## 1. 环境与代码基线

- 相关读帧器：
  - `external/ifrnet_video/ffmpeg_io.py::FFmpegFrameReader`（输出 `(H,W,3) uint8` rgb24 对）
  - `external/realesrgan_video/ffmpeg_io.py::FFmpegReader`（输出 `(H,W,3) uint8` rgb24）
- 路径选择器：`src/utils/reader_hwaccel.py`（`decide_reader_hwaccel`，由码率/核数决策）
  —— 环境变量 **`READER_HWACCEL=auto|on|off`**（旧名 `IFRNET_READER_HWACCEL` 兼容）
- 素材（会话中使用的测试片）：
  - `/tmp/clip_sd_25s.mp4`：640×360、25s、**578 帧**（`ffprobe -count_frames`）
  - `/tmp/cm_709tv.mp4`、`/tmp/cm_709pc.mp4`：同内容、**显式**写入
    `color_range=tv|pc` + `color_space=bt709` 的变体（用于验证两条路径对
    `color_range` 的解读是否一致）

---

## 2. 已确认的事实（会话实测确立，接手方应复测确认）

### 2.1 差异不在解码层

把两条路径都导出 `yuv420p` 只比平面：

```
Y  平面不同像素 = 0 (最大差 0)
UV 平面不同像素 = 0 (最大差 0)
```

⇒ **解码完全精确一致**，RGB 的差异 100% 来自 `nv12→rgb24`(swscale) 的矩阵/范围选择。

### 2.2 差异幅度（两条读帧器都复现）

| 比对 | 结果 |
|---|---|
| ESRGAN `FFmpegReader`：NVDEC vs 软解 | 最大逐像素差 **18~29**，约 **60.8%** 像素不同（578 帧全片） |
| IFRNet `FFmpegFrameReader`：NVDEC vs 软解 | **30/30** 帧不同，最大逐像素差 **19** |
| reader vs **同参数**参考解码（同 `-hwaccel auto`） | **0/578 不一致** ⇒ reader 对其自身契约是**忠实**的 |

⇒ 两个 reader 都没有"读错帧"，差异是**路径间**的，不是 reader 缺陷。

### 2.3 源素材没有任何色彩元数据

`color_range` / `color_space` / `color_primaries` / `color_transfer` **全为 `unknown`**。
⇒ 两条路径都在"猜"矩阵与范围，且**猜法不同**。

### 2.4 本 build 的滤镜无法钉住元凶（这是本立项的技术难点）

- 该 ffmpeg **没有** `zscale`（缺 libzimg）、**没有** `colorspace` 滤镜；
- 只有 `scale`(swscale)，但其 `in_range` / `in_color_matrix` / `out_color_matrix`
  对**实际转换无效**：两组参数得到的数字**完全相同**
  （例：NVDEC 输出 vs 软解+`out_color_matrix=bt601` 与 vs 软解默认，都是
  `相同像素=36.10% 最大差=18`）。
- 很可能的解释：`yuv420p→rgb24` 的格式转换被 ffmpeg 用**自动插入的 scaler** 完成，
  用户写的 `scale` 只是在其后做 RGB→RGB（于是那些选项成为 no-op）。
  ⇒ 需用能**真正**控制转换的手段（`format`+显式 `sws_flags`、`-vf
  "scale=in_color_matrix=..."` 的正确写法、或直接换成带 `libzimg` 的 ffmpeg build）。

**观测到的关键钩子**：差异幅度**随声明的 `color_range` 变化**
（`tv` 变体 23.66% 像素相同 vs `pc` 变体 50.38%）⇒ 两条路径对 `color_range`
的**解读不同**，这是最可能的分歧点。

### 2.5 与 `READER_HWACCEL`（P2-2 自适应）的交互 —— 容易被误读成"改坏了"

此前策略是「有 NVDEC 就用」，现在是数据相关的（`bits/px/frame ≤ 0.15 且 ≥8 核 → 软解`）。
因此对低码率素材，两个读帧器的输出像素**会与改动前不同**（最多 ±19~29）
⇒ 改动前后跑同一素材，产物**不是逐字节可比**。

**做 A/B 对照时请先冻结路径**：

```bash
READER_HWACCEL=on  python src/main_video_optimized.py ...   # 强制 NVDEC（与旧行为一致）
READER_HWACCEL=off python src/main_video_optimized.py ...   # 强制软解
```

---

## 3. 建议的实施步骤

1. **先复测**（§4 命令），确认本环境上 §2.2 的三个数字可复现（若用的是另一版 ffmpeg，
   数值可能不同，但"Y/UV 一致 + RGB 不同"的定性应保持）。
2. **换一把能钉住的尺子**：在带 `libzimg` 的 ffmpeg build 上用 `zscale`
   （或 `colorspace` 滤镜）显式指定 `in_range`/`in_color_matrix`，
   逐组合与 NVDEC 输出比对，找出**与 NVDEC 逐字节相等**的那组参数 ⇒ 即 NVDEC 的实际选择。
3. **读 ffmpeg 侧源码/日志**（若可行）：给 NVDEC 路径加 `-loglevel debug` 看
   `swscale` 实际协商到的 `SWS_CS_*`；对比软解路径。
4. **判定"谁对"**：结合源实际内容（比特流真实是 limited 还是 full）判断；
   注意 `unknown` 时 ffmpeg 的默认（据现有立项文档，默认按 **tv** 解释）。
5. **给出结论二选一**：
   - **不强制对齐**（若两条路径都"合理"）：则在文档/日志里**显式声明**"路径影响像素值"，
     并把"跨路径产物不可逐字节比对"写进验收说明（避免后人误判为回归）。
   - **要对齐**：在读帧器侧把范围/矩阵**显式钉死**（`-vf` 或 `-color_range`/`-colorspace`
     参数），使两条路径产出同一结果；再回到
     `Plan/Video_Enhancement_color_range_强制转换_立项Prompt.md` 做 CLI 暴露。

---

## 4. 复测命令（会话中实际用过的最小集）

```bash
# 0) 期望帧数（同时确认两条路径都读满）
ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 /tmp/clip_sd_25s.mp4

# 1) 两条路径各自导出 rgb24，逐帧字节比对 + 幅度量化
#    （注意：ffmpeg 在后台进程组会被 SIGTTOU 停住，务必 < /dev/null 或 stdin=DEVNULL）
ffmpeg -hide_banner -v error -noautorotate -i /tmp/clip_sd_25s.mp4 \
       -an -fps_mode passthrough -f rawvideo -pix_fmt rgb24 - > /tmp/o_soft.rgb </dev/null
ffmpeg -hide_banner -v error -noautorotate -hwaccel auto -i /tmp/clip_sd_25s.mp4 \
       -an -fps_mode passthrough -f rawvideo -pix_fmt rgb24 - > /tmp/o_hw.rgb </dev/null

# 2) 平面级诊断（决定性实验）：都导出 yuv420p 只比 Y / UV
#    期望：Y、UV 均 0 差异 ⇒ 差异在 nv12→rgb24

# 3) 读帧器层复测：用真实 FFmpegFrameReader / FFmpegReader 消费，
#    同时比对「同参数参考解码」，确认 reader 自身忠实（0 不一致）
```

> 会话中踩过的坑：**不要把 `run()` 之类的 shell 函数写在 `&&` 链里跨 subshell 使用**
> （会话里它静默失效导致 NVDEC 输出文件为 0 字节、被误读成"命令失败"）。
> 逐条显式命令最稳。

---

## 5. 验收判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 定性可复现 | Y/UV 平面 0 差异；RGB 有差异（幅度以本环境实测为准） |
| 2 | reader 忠实性 | reader vs 同参数参考解码：0 不一致 |
| 3 | 元凶可判定 | 能给出「NVDEC 实际用的 range/matrix」与「软解实际用的」两组结论，且各自由**逐字节相等**的实验支撑 |
| 4 | 结论明确 | 「对齐」或「声明不对齐」二选一，且写明理由与证据 |
| 5 | 无回归 | 若做对齐：解码级验收 + 帧数守恒通过；`READER_HWACCEL=on/off` 两条路径产物**一致** |
| 6 | 文档同步 | 若结论是"不对齐"，必须把"路径影响像素值"写进验收/对比说明，并更新 memory |

---

## 6. 风险与回滚

- **最大的坑是"把差异当成回归"**：改动前后对比产物时必须先冻结
  `READER_HWACCEL`，否则会把路径切换误判成缺陷（本会话已发生过一次误判）。
- 若做对齐：会改变**低码率素材**的产物像素（因为自适应本来就会切到软解），
  属"有意变更"，需要在验收说明里显式记一笔。
- 本立项前期（1~4 步）是**纯调查，无代码改动**，无风险；
  第 5 步才可能动读帧器参数，回滚 = 去掉显式范围/矩阵参数。
- 第 2 步需要带 `libzimg` 的 ffmpeg build（本容器现有 build 没有，需先解决环境）。
