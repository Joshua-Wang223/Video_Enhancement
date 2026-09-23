# 立项 Prompt：Video_Enhancement 添加 `--color-range` 并在 AI 编码阶段强制真转换

> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。所有路径、行号、数值均已实测核对。
> 立项时间：2026-09-09　立项人：元数据保留改造会话
> 关联已完成项：`VidUtils/vidcrop_hwaccel.py`、`VidUtils/vidcrop_cpu_v2.py` 的
> `--color-range` / `--color-range-convert`（已实现并验证，可直接照抄）

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| VidUtils 两脚本的 `--color-range` + `--color-range-convert` | ✅ 已完成 | 可复用其实现与实测结论，见 §4 |
| Video_Enhancement 侧的 `--color-range` / `--color-range-convert` | ❌ **已撤销，待本立项重做** | 2026-09-09 撤销，只保留 auto 语义 |
| 在 AI 编码阶段做真实值域转换 | ⬜ **本立项核心待办** | 这是唯一能真正转换的位置 |

> ⚠️ **不要走"在最终 merge 层加 `--color-range`"的路**：
> Video_Enhancement 的最终合并几乎总是 `-c:v copy`，加在那里**只能改标签**，
> 会产生"limited 数据 + pc 标签"的错误产物（实测见 §2.2）。已试过并撤销。

---

## 0. 任务

给 `Video_Enhancement` 添加 `--color-range {auto,tv,pc}`，并在 **AI 编码阶段**
（Real-ESRGAN / IFRNet 的 writer，而非最终 merge）做**真实的像素值域转换**，
使产物满足：

- `--color-range pc`：产物**真实存储值**为 0-255，且解码回 RGB 与原片一致
- `--color-range tv`：与当前行为一致（limited 16-235）
- `--color-range auto`（默认，现状）：源有值取源值，`unknown` 取 `tv`

```
python src/main_video_optimized.py -c config/default_config.json \
    -i input.mp4 -o output.mp4 --color-range pc
```

---

## 1. 环境与代码基线

- GPU：NVIDIA Tesla T4，驱动 580.65.06；ffmpeg / ffprobe **6.1.1**
- 本工程：`/workspace/Video_Enhancement`（**非 git 仓库**，无法用 `git diff` 溯源；
  `Video_Enhancement_jxz_*.tar.gz` 是快照，会落后于工作树）
- 参考样本：`/workspace/output_videos/Dora/Season 02/S02E08_Dora…`（768x576、30fps、
  yuv420p、**四项色彩元数据全为 unknown**、实际内容为 limited range）
- 复测命令（约 30 秒）：
  ```bash
  cd /workspace/Video_Enhancement
  python src/main_video_optimized.py -c config/default_config.json \
      -i /tmp/mk/rot90.mp4 -o /tmp/mk/out.mp4 --skip-interpolate --color-range pc
  ```

---

## 2. 已确认的事实（实测确立，无需重做，不得违反）

### 2.1 `unknown` 时 ffmpeg 按 **tv** 解释解码

```
unknown 默认解码 == 按 tv 解释   → True
unknown 默认解码 == 按 pc 解释   → False
```

所以 `auto` 下把 `unknown` 判为 `tv` 是**保持原样**（解码结果与源逐像素一致）；
判为 `pc` 才会改变解码结果。

### 2.2 `-color_range pc` 只改标签，**不改数据**（解码端）

limited 数据被标成 pc 后，解码不再拉伸：

| 分位 | tv 解码 | pc 解码 |
|---|---|---|
| P1（暗） | 0 | **11** |
| P99（亮） | 251 | **233** |

平均绝对偏差 8.61/255，最大 20。表现：发灰、对比度不足。

### 2.3 `-color_range pc` 在**编码端同样只改标签**（这是本立项的关键坑）

AI 编码阶段的输入是 **rgb24（full-range RGB）**。无损编码（`-qp 0`）往返测试：

| 写法 | 产物 pix_fmt/range | RGB 往返偏差 |
|---|---|---|
| `-color_range tv` | yuv444p/unknown | **0.31** |
| `-color_range pc` | yuvj444p/pc | **8.69** ← 错误，数据没转 |
| `-pix_fmt yuvj420p` | yuvj420p/pc | **0.86** ← 真正转换 ✅ |

结论：**要让产物真的是 full range，必须改 `pix_fmt` 为 `yuvj*` 系列（或插 scale 滤镜），
`-color_range` 参数本身做不到。** 本立项曾尝试"只加 `-color_range pc`"，实测让产物
变差（8.69 vs 0.31）后已撤销。

### 2.4 转换只能发生在 AI 编码阶段

链路：`切片 → AI 编码（真重编码） → concat merge（-c:v copy）`。
最终 merge 是 copy，无法改数据。实测两次端到端产物（标记 tv vs 标记 pc+转换）：

```
e2e_rot2 (tv)        存储 Y = [9,249] 均值 120.2
e2e_cr   (pc+转换)    存储 Y = [9,249] 均值 120.2   ← 数据完全相同，只有标签不同
```

（`yuvj420p` 只是 ffprobe 依 range 标记推导的显示值，不代表真重编码。）

---

## 3. 关键代码锚点（行号为 2026-09-09 工作区状态，动手前先复核）

| 位置 | 作用 |
|---|---|
| `external/realesrgan_video/ffmpeg_io.py` `cmd_args = [...]`（约 779-793） | Real-ESRGAN writer 的 ffmpeg 命令；输入 `-f rawvideo -pix_fmt rgb24 -i pipe:` |
| `external/ifrnet_video/ffmpeg_io.py` `class FFmpegWriter`（569+），`cmd = [...]`（约 709-723） | IFRNet writer 的 ffmpeg 命令，同样是 rgb24 输入 |
| `external/ifrnet_video/main.py:1366` `writer = FFmpegWriter(...)` | IFRNet writer 实例化点 |
| `src/processors/realesrgan_processor_video_optimized.py:844` `ns = argparse.Namespace()` | processor 传给后端 `main_optimized(ns)` 的参数包 |
| `src/processors/ifrnet_processor_video_optimized.py:541` `processor = IFRNetVideoProcessor(` | processor 构造后端处 |
| `src/main_video_optimized.py` `_apply_cli_overrides`（966+） | CLI → `config.set("output", ...)`，新增开关在此登记 |
| `src/main_video_optimized.py` `parse_args` 的「处理控制」参数组（约 2105） | 新增 CLI 参数处 |
| `src/utils/video_utils.py` `build_color_args`（约 2542） | 现为固定 auto；**merge 层不要再加覆盖** |
| `VidUtils/vidcrop_hwaccel.py` `build_range_convert_filter` / `_effective_source_range` | 已实现的 YUV→YUV 转换参考实现 |

---

## 4. 可复用的已完成实现（VidUtils）

`vidcrop_hwaccel.py` / `vidcrop_cpu_v2.py` 已有：

- `--color-range {auto,tv,pc}` + `--color-range-convert`（`store_true`）
- `_effective_source_range(meta)`：源有值取源值，unknown 时按 `yuvj*` 判 pc、否则 tv
- `build_range_convert_filter(...)`：不一致时返回
  `scale=w=iw:h=ih:in_range=<src>:out_range=<tgt>`
- 已验证：`auto` 下源 pc→pc、源 tv→tv、源 unknown→tv；强制 pc+转换后存储 Y [0,255]，
  解码 RGB 与原片一致（P1=0、P99=252）

⚠️ 但那是 **YUV→YUV** 场景。AI 编码阶段是 **RGB→YUV**，按 §2.3 应改用
`pix_fmt` 覆盖，两者的转换机制不同，**不要直接照搬 scale 滤镜**。

---

## 5. 实施方向（供参考，最终方案由承接方定，先给方案再动手）

1. **CLI + config**：`main_video_optimized.py` 增加
   `--color-range {auto,tv,pc}`；`_apply_cli_overrides` 里
   `config.set("output", "color_range", value=...)`。
   （是否还需要 `--color-range-convert` 由承接方判断——若 AI 编码阶段总能真转换，
   则可能不需要该开关。）
2. **processor 透传**：
   - Real-ESRGAN：`ns.color_range = <config 值>`（`ns` 在 §3 锚点处构造）
   - IFRNet：`IFRNetVideoProcessor(..., color_range=...)`，并在
     `main.py:1366` 传给 `FFmpegWriter`
3. **writer 落地真转换**（核心）：
   - `tv`：保持现状（或显式 `-pix_fmt yuv420p`）
   - `pc`：`-pix_fmt yuvj420p`（实测往返 0.86）**并**保留 `-color_range pc`
   - `auto`：不动
4. **NVENC 兼容性**：`h264_nvenc` / `hevc_nvenc` 是否接受 `yuvj420p` **未验证**。
   若不接受，需设计替代（例如插 `scale=out_range=pc`，或对 NVENC 路径告警降级）。
   这一步必须实测，不能假设。
5. **10-bit 场景**：`yuvj*` 只有 8-bit。10-bit full range 该用什么 pix_fmt 未验证，
   建议本期只支持 8-bit，10-bit 时告警并回退 auto。

---

## 6. 验收标准

| # | 项 | 通过条件 |
|---|---|---|
| 1 | `--color-range pc` 真实转换 | 产物**真实存储 Y 达到 0-255**（用 `-vf scale=in_range=pc:out_range=pc` 读原始值，勿用默认直读——ffmpeg 默认按 tv 输出 yuv420p，会误判） |
| 2 | 解码一致性 | 产物解码 RGB 与原片一致（P1≈0、P99≈252），不得出现 P1=12 / P99=239 的发灰形态 |
| 3 | `--color-range tv` | 与不指定时行为一致 |
| 4 | `auto`（默认） | 源 pc→pc、源 tv→tv、源 unknown→tv |
| 5 | 短片整段处理 | 不分段（整段处理）路径同样生效 |
| 6 | 回归 | `VidUtils` CPU 48/48 + GPU 36/36 回归脚本全过 |
| 7 | 兼容性 | NVENC 路径要么生效、要么明确告警降级，不得静默产出错误标签 |

---

## 7. 风险与未决问题

- **`yuvj*` 是已废弃的 pix_fmt**，长期看有兼容风险；是否有更好的写法需调研。
- **NVENC 对 `yuvj420p` 的支持未验证**（`ffmpeg -h encoder=h264_nvenc` 列出的
  pixel formats 里未见 `yuvj*`）——这是最大的不确定性。
- **10-bit full range 的 pix_fmt 未验证**。
- 强制 pc 会改变产物编码参数，可能影响下游播放器兼容性，需在文档里说明。
- 本工程流水线偶发挂起（3 线程管道，见 CODEBUDDY.md），端到端验证时若卡住，
  先用定向单元测试验证 writer 命令，再跑完整链路。

---

## 8. 相关文档

- `CODEBUDDY.md`（工程结构、活跃文件表、调试挂起指引）
- `Plan/H264_LA排空放弃缺陷_立项Prompt.md`（立项文档格式参考）
- `memory/`（需与 `/root/.codebuddy/projects/workspace-Video_Enhancement/memory` 双向同步）
- VidUtils 已实现代码：`/workspace/VidUtils/vidcrop_hwaccel.py`、
  `/workspace/VidUtils/vidcrop_cpu_v2.py`（搜索 `META-KEEP` 与 `build_range_convert_filter`）
