---
name: 本机 ffmpeg 的 NVENC preset 命名映射与编码器可用性（实测）
description: ffmpeg 的命名 preset 与 NVENC p-梯并非一一对应（medium≡p4 但 slow≠p5）；本 build 无 libsvtav1/libaom/librav1e；av1_nvenc 在 T4 上实跑失败
type: project
---

2026-09-28 在容器内实测（ffmpeg 8.x + Tesla T4）。两条"不看实测就会写错"的事实：

**1. `-preset` 命名 preset ≠ NVENC p-梯命名**

`h264_nvenc` 的 `-preset` 是 int `0..18`：`default/slow/medium/fast/hp/hq/bd/ll/llhq/llhp/lossless/losslesshp`
是 NVENC **遗留档**，`p1..p7`（索引 12..18）才是现代档梯。
同一条命令（`-rc:v vbr_hq -cq:v 26 -b:v 0`，1080p 真实素材）按输出 md5 去重后的等价关系：

| ffmpeg 命名 | 等价 pN | 备注 |
|---|---|---|
| `default` / `medium` | **`p4`**（逐字节相同） | 唯一可作锚点的等价 |
| `fast` / `hp` | `p1` | |
| `bd` | `p5` | |
| `hq` | `p7` | |
| `slow` | **不落在 p1~p7 梯上** | 遗留 "hq 2 passes"，独有输出 |
| — | `p2`/`p3`/`p6` 各有独立输出 | |

⇒ 方案/注释里"p5=slow / p6=slower / p7=slowest"说的是 **NVIDIA p-梯的命名**，**不是** ffmpeg
命名 preset 的等价关系。VE 的 `_PRESET_P_INDEX`（x264 名 → pN）映射方向正确，但"按 ffmpeg
官方枚举对齐"的措辞不准 —— 应写"按 NVIDIA p-梯命名对齐，并以 `medium ≡ p4` 实测锚定"。

**2. 编码器可用性是两件事：构建里有没有 vs 硬件编不编得动**

- 本机 ffmpeg **构建不含 `libsvtav1` / `libaom-av1` / `librav1e`**；AV1 软编只有 `libvpx-vp9`，
  AV1 硬编有 `av1_nvenc` / `av1_vaapi`。
- **`av1_nvenc` 在 T4 上实跑报 `No capable devices found`**（Turing 无 AV1 NVENC）。
- ⚠ 必须分开判定：构建可用性用 `ffmpeg -encoders`；硬编硬件能力用**实跑一帧**。
  **`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样打印选项表**，不可作依据。
- 判据里把可选编码器"直接纳入"会因 `Unknown encoder` 让整组判据中断 ⇒ 一律先探再跑，不可用记 SKIP。

**Why:** 2026-09-28 这两点分别导致（a）判据 G7 整组因 `Unknown encoder 'libsvtav1'` 中断、
（b）方案里对 preset 档位的断言失真。
**How to apply:** 写涉及 NVENC preset 档位、AV1 软/硬编覆盖的代码、文档或判据前，先按上表核对；
不要假定 ffmpeg 命名 preset 等于 p-梯档位，也不要把 `-h encoder=` 的输出当成硬件可用性证据。
