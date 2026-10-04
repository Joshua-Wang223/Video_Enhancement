# T4 NVENC vbr_hq 移除 · 验证专项

> **触发**：FFmpeg 9.0 移除 `-rc:v vbr_hq`（NVIDIA SDK 10.0+ 重构 RC API）
> **实测环境（2026-10-04）**：Tesla T4 / 驱动 580.65.06 / CUDA 13.0 / FFmpeg 9.0.2 / NVENCAPI 13.0 —— 本专项**已执行完毕**，结果见 §0 / §1.5.5 / §1.6；裁定 **方案 A**
> **FFmpeg 构建特征**：FFmpeg 9.0.2 + libffmpeg-nvenc-dev 12.1.14.0 编译
> **核心问题**：SDK 12.x 仍定义 `NV_ENC_PARAMS_RC_VBR_HQ=32`，但 FFmpeg 9.0 CLI 拒绝 `-rc:v vbr_hq`
> **关键约束**（来自 T4 方案 §12.3 CR-2 rationale）：`nvenc_sdk` 的 `else` 分支遇未知 rate_mode 静默落到 CONSTQP，且 SDK LA 门控只认 `vbr_hq/qvbr` → **SDK 路径不支持 plain `vbr`**
> **目标**：确认 SDK 13.0 对 rc_ptr[1]=32 的接受度 + 验证迁移路径 `-rc:v vbr -tune hq -multipass fullres`，决定 P0 范围

---

## 0. 一句话结论

**实测（2026-10-04，Tesla T4 / 驱动 580.65.06 / FFmpeg 9.0.2 / NVENCAPI 13.0）：裁定 P0 = 方案 A（仅 CLI 映射）。**

- FFmpeg 9.0 CLI 层确实拒绝 `-rc:v vbr_hq`（rc=234，`Unable to parse "rc" option value`）；迁移路径 `-rc:v vbr -tune hq -multipass fullres` rc=0（h264/hevc 均通）。
- nv-codec-headers 13.0/13.1 头文件**已删除** `NV_ENC_PARAMS_RC_VBR_HQ`（`NV_ENC_PARAMS_RC_MODE` 仅剩 CONSTQP/VBR/CBR）——按原判定表应落方案 B。
- **但驱动运行时仍接受 `rc_ptr[1]=32`**：真实 `NVENCEncoder(rate_mode="vbr_hq")` 初始化成功（apiVersion=0xd0=13.0），且行为验证证明 mode 32 被**真实执行而非静默钳制**（见 §1.6.2）。→ 决定性 `[SDK-test]` 通过 → **方案 A**。

> ⚠️ 头文件删除是开源 `nv-codec-headers` 的裁剪，不等于驱动移除。驱动对 32 做了向后兼容。方案 A 保留内部名 `vbr_hq`（SDK 路径仍用 32），仅把 FFmpeg CLI 映射改为 `vbr + -tune hq -multipass fullres`。

---

## 1. Gate 0 执行顺序（T4 上按序执行）

> 顺序不可调换：① 是 ②③ 的前提；④⑤ 依赖 ① 的 h264_nvenc 可用；2.3 是决定性测试。

### ① 环境体检（~3 min）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L
ffmpeg -hide_banner -version | head -1
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
         -c:v $C -f null - < /dev/null >/dev/null 2>&1; echo "$C rc=$?"
done
```

**判定**：h264_nvenc/hevc_nvenc rc=0 才继续；av1_nvenc rc≠0 才是 T4 预期。

### ② CLI 迁移路径确认（~3 min）

```bash
# ④ vbr_hq 在 FFmpeg 9.0 被拒（预期 rc≠0）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v h264_nvenc -rc:v vbr_hq -cq:v 23 -b:v 0 -f null - < /dev/null 2>&1; echo "vbr_hq_cli_rc=$?"

# ⑤ 迁移路径可用（预期 rc=0）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v h264_nvenc -rc:v vbr -tune hq -multipass fullres -cq:v 23 -b:v 0 \
       -f null - < /dev/null 2>&1; echo "vbr_migrate_rc=$?"
```

**判定**：
- ④ rc≠0 → 符合预期（FFmpeg 9.0 确认拒绝）
- ⑤ rc=0 → 迁移路径可行，**P0 至少改 CLI 映射**
- ⑤ rc≠0 → `-tune hq -multipass fullres` 在当前驱动下不可用，需备选（回退 `-tune hq` 无 multipass）

### ③ SDK 头文件确认（~1 min）

```bash
grep -n "NV_ENC_PARAMS_RC_VBR_HQ" $(find / -name "nvEncodeAPI.h" 2>/dev/null | head -1)
# 预期（SDK 12.x）：#define NV_ENC_PARAMS_RC_VBR_HQ  32
# 无输出 → SDK 已彻底移除 VBR_HQ，ctypes 路径必崩
```

### ④ 运行时诊断脚本（~5 min）

```bash
python3 Accessory/probe/nvenc_rc_mode_diagnose.py
```

关注：`NV_ENC_PARAMS_RC_VBR_HQ` 值、`-rc` 选项列表是否含 `vbr_hq`、SDK 版本推断。

### ⑤ ctypes 实编测试（~10 min）—— 决定性测试

```python
"""T4 验证：SDK 13.0 是否接受 rc_ptr[1]=32 (VBR_HQ)"""
import sys
sys.path.insert(0, '/workspace/Video_Enhancement')
from external.ifrnet_video.nvenc_sdk import NVENCEncoder

try:
    enc = NVENCEncoder(width=320, height=240, fps=30,
                       rate_mode="vbr_hq", codec="h264", preset="p4", qp=23)
    print("✅ rc_ptr[1]=32 (VBR_HQ) 被 SDK 接受 — P0 只改 CLI 映射（方案 A）")
    enc.close()
except Exception as e:
    print(f"❌ rc_ptr[1]=32 被 SDK 拒绝: {e}")
    print("→ P0 范围扩大到方案 B（ctypes 加 vbr 分支）")
```

**判定**：
- ✅ 接受 → P0 = 仅改 CLI 映射（方案 A）
- ❌ 拒绝 → P0 = CLI 映射 + ctypes 路径新增 `vbr` 分支（方案 B）

> ⚠ 2.3 必须用真实的 `NVENCEncoder` 类实例化测试（而非裸 ctypes 拼 RC_PARAMS），因为完整的 `_build_encoder_config` 流程中 `_rate_mode` 会影响 LA 启用、targetQuality 写入等。裸 `rc_ptr[1]=32` 写入只能验证枚举值是否被 SDK 接受，不能验证整条初始化路径。

---

## 1.5 T4 机器准备与待办验证事项

> 本节汇总 T4 上机前需准备的命令和待办验证事项，确保上机后一次性跑完所有项。

### 1.5.1 T4 环境准备

```bash
# 上机后首先确认环境
cd /workspace/Video_Enhancement
nvidia-smi -L                                        # 期望 Tesla T4
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.is_available())"
ffmpeg -hide_banner -version | head -1               # 期望 9.0.2

# 确认 libcuda 可用（否则 ctypes 测试无法运行）
ldconfig -p | grep libcuda
# 若 libcuda.so 缺失或为 0 字节桩，需先修复 NVENC 环境（见 memory/project_gpu_container_flaky.md）

# 激活 conda/venv（如需要）
# conda activate <env>  或  source /path/to/venv/bin/activate

# 确认仓库路径
pwd        # 期望 /workspace/Video_Enhancement
ls Accessory/probe/nvenc_vbr_hq_verify.py           # 确认脚本已就位
```

### 1.5.2 nvenc_rc_mode_diagnose.py 调用

```bash
# 运行诊断脚本（需传入 nvEncodeAPI.h 路径）
python3 Accessory/probe/nvenc_rc_mode_diagnose.py $(find / -name "nvEncodeAPI.h" 2>/dev/null | head -1)

# 关注输出：
#   - NV_ENC_PARAMS_RC_VBR_HQ 值是否 = 32
#   - ffmpeg -h encoder=h264_nvenc 中 -rc 选项列表是否含 vbr_hq
#   - SDK 版本推断（驱动版本 → SDK 范围）
```

### 1.5.3 ctypes 实编测试脚本

脚本已置于 `Accessory/probe/nvenc_vbr_hq_verify.py`，运行：

```bash
python3 Accessory/probe/nvenc_vbr_hq_verify.py
```

脚本自动执行：
1. 打印 FFmpeg 版本 + NVENC 编码器列表 + nvenc_rc_mode_diagnose.py 关键输出
2. 实例化 `NVENCEncoder(rate_mode="vbr_hq")` 测试 SDK 是否接受 rc_ptr[1]=32
3. 若拒绝，再测 `rate_mode="vbr"` 验证 vbr 分支可行性

### 1.5.4 T4 标定方案中受 vbr_hq 移除影响的待办项

以下 T4 方案（`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md`）中的待办项需在本专项完成后更新。
**裁定：方案 A**（V12 ctypes 接受 `rc_ptr[1]=32`）——只需改 harness/CLI 映射，生产内部名与 `nvenc_sdk.py` 均不变：

| T4 方案项 | 当前值 | 需更新为 | 依赖 Gate 0 结论 |
|---|---|---|---|
| §4.1 BASE_LOCK h264_nvenc/hevc_nvenc | `-rc:v vbr_hq` | **方案 A**：`-rc:v vbr -tune hq -multipass fullres` | ✅ V12 接受 → 方案 A |
| §4.1 CR-2 路线B 裁定 | "h264/hevc → vbr_hq（与 VE 生产 / SDK Level 1 的 RC_VBR_HQ 一致）" | 重新裁定：理由从"SDK 不支持 vbr"变为"FFmpeg 9.0 移除 vbr_hq"；**VE 生产内部仍用 vbr_hq=32，harness 层映射 vbr** | ✅ V10/V12 |
| §5.6 生产管线冒烟 | `--rate-mode-ifrnet vbr_hq` | **方案 A：保持 `--rate-mode-ifrnet vbr_hq` 不变**（内部名不变；实测 E2E 150→299 帧守恒）。⚠ 不要传 `vbr`——会静默 CONSTQP+LA 失效（§1.6.3） | ✅ V12 |
| §12.3 CR-2 协同契约 | "VE 生产走 ctypes 直连 SDK Level 1，nvenc_sdk 不支持 vbr" | **维持原状**（方案 A 不动 SDK 路径）；仅需补一句"FFmpeg 9.0 CLI 层已移除 vbr_hq，VE 的 ffmpeg_io 回退路径映射为 vbr+tune hq+multipass" | ✅ V12 |
| §12.5 CR-2 复检 | "VU 生产默认 rc = --rc-mode auto = 不发 -rc" | 复检 VU 侧是否也需同步 vbr_hq 移除改动（VU 走 FFmpeg CLI，其 harness `-rc:v vbr_hq` 同样会被 FFmpeg 9.0 拒绝 → 需与 VE 同步改 `vbr -tune hq -multipass fullres`） | ✅ V12 |

### 1.5.5 待办验证清单（T4 上机执行）

| # | 验证项 | 命令/脚本 | 预期结果 | 实际结果（2026-10-04 T4） |
|---|---|---|---|---|
| V1 | nvidia-smi | `nvidia-smi -L` | Tesla T4 | ✅ GPU 0: Tesla T4 |
| V2 | torch CUDA | `python3 -c "import torch;print(torch.cuda.is_available())"` | True | ✅ torch 2.10.0+cu128, True, Tesla T4 |
| V3 | FFmpeg 版本 | `ffmpeg -version \| head -1` | 9.0.2 | ✅ 9.0.2 |
| V4 | NVENC 编码器 | `ffmpeg -encoders \| grep nvenc` | h264/hevc/av1_nvenc 均列出 | ✅ 三者均列出 |
| V5 | h264_nvenc 实跑 | `ffmpeg ... -c:v h264_nvenc -f null -` | rc=0 | ✅ rc=0 |
| V6 | hevc_nvenc 实跑 | `ffmpeg ... -c:v hevc_nvenc -f null -` | rc=0 | ✅ rc=0 |
| V7 | av1_nvenc 实跑 | `ffmpeg ... -c:v av1_nvenc -f null -` | rc≠0 | ✅ rc=187（T4 无 AV1 编码） |
| V8 | vbr_hq CLI 被拒 | `ffmpeg ... -rc:v vbr_hq ...` | rc≠0 | ✅ rc=234 `Unable to parse "rc" option value "vbr_hq"` |
| V9 | vbr 迁移路径 | `ffmpeg ... -rc:v vbr -tune hq -multipass fullres ...` | rc=0 | ✅ rc=0（h264/hevc 均通） |
| V10 | 头文件 VBR_HQ | `grep NV_ENC_PARAMS_RC_VBR_HQ nvEncodeAPI.h` | =32 或无定义 | ⚠️ **无定义**（nv-codec-headers 13.0/13.1 仅 CONSTQP/VBR/CBR） |
| V11 | 诊断脚本 | `python3 nvenc_rc_mode_diagnose.py` | 输出关键信息 | ✅ 确认 VBR_HQ 未在头文件；-rc 仅 constqp/vbr/cbr |
| V12 | ctypes vbr_hq | `python3 nvenc_vbr_hq_verify.py` | 接受/拒绝 | ✅ **接受**（apiVersion=0xd0=13.0，`Ready ... VBR_HQ`） |
| V13 | ctypes vbr（若 V12 拒绝） | 同上脚本自动测 | 接受/拒绝 | —（V12 接受，未触发；另测 `vbr` 落入 `else`→CONSTQP，见 §1.6.3） |
| V14 | 画质对比 ΔVMAF | 标定预跑 | ≤0.3 | ✅ h264 ΔVMAF=−0.048；hevc ΔVMAF=−0.130（真实素材，见 §1.6.1） |
| V15 | LA 联动 | vbr 模式下 LA=8 | 正常启用 | ✅ `-rc-lookahead 8` 输出与 LA=0 不同（生效）；SDK 路径 vbr_hq+LA8 E2E 150→299 帧守恒 |

> **裁定（依 V12）**：方案 A —— P0 只改 CLI 映射；`nvenc_sdk.py` 不动。

---

## 1.6 实测结果（2026-10-04，Tesla T4）

### 1.6.1 §4 画质对比（真实素材，锚点 cq=23）

> ⚠️ 原 §4 的"旧路径"在 FFmpeg 9.0 上无法产出（`vbr_hq` 被拒）。为取得**真实旧基线**，改用系统备份的 **FFmpeg 6.1.1**（`/var/backups/ffmpeg-v9/20261001-015049/usr_bin/ffmpeg`，实测接受 `-rc:v vbr_hq`）编码旧产物；新产物用 FFmpeg 9.0.2 迁移路径。**存在 FFmpeg 版本混淆**（6.1.1 vs 9.0.2），Δ 值仅供量级参考。

素材：`input_vidiow/real_captured/test_video_640_360_real.mp4`（640×360@30，真实拍摄），取前 6 s（183 帧），无损流拷贝作参考。

| codec | 旧 `vbr_hq` bytes | 新 `vbr+th+mp` bytes | ΔPSNR | 旧 VMAF | 新 VMAF | **ΔVMAF** |
|---|---|---|---|---|---|---|
| h264_nvenc | 1,539,271 | 1,467,218 (−4.7%) | −0.090 | 97.679 | 97.631 | **−0.048** |
| hevc_nvenc | 1,562,355 | 1,538,863 (−1.5%) | −0.077 | 97.513 | 97.383 | **−0.130** |

**判据 ΔVMAF ≤ 0.3 → 双 codec PASS。** 新路径体积略小、画质差在噪声量级内。

### 1.6.2 `[SDK-test]` 决定性证据（V12）

- `nvenc_vbr_hq_verify.py`：`NVENCEncoder(rate_mode="vbr_hq")` 构造成功 → SDK/驱动接受 `rc_ptr[1]=32`。
  - ⚠️ 脚本原有两个 bug（`sys.path` 少一层 `dirname` → `No module named 'external'`；`subprocess.run(...).values()` 在 `CompletedProcess` 上不存在）→ 已修复后方可运行。
- **行为验证**（排除"静默钳制为 CONSTQP"）：同输入 30 帧噪声，
  `vbr_hq`=1,076,383 B / `constqp`=2,159,969 B / `qvbr`=1,018,148 B，三者互异且各 30 帧守恒 → mode 32 真实生效。

### 1.6.3 `vbr` 在 SDK 路径的现状（Plan A 依据）

`nvenc_sdk.py:1044` 的 `else` 把未知 rate_mode 映射为 CONSTQP；`:1056` LA 门控 `_rate_mode in ('vbr_hq','qvbr')`。实测 `--rate-mode-ifrnet vbr`：
`[NVENCEncoder] Ready: 640x360@60.0fps HEVC CONSTQP QP=28 ... la=8`（**静默 CONSTQP，且 LA 未真正启用**）。
→ 生产默认内部名始终为 `vbr_hq`，故方案 A 不改内部名即不受影响；但 CLI 已开放 `vbr`/`cbr` 选择，属独立既有隐患（若后续需支持，才走方案 B）。

---

## 2. 判定表（五步结果 → P0 范围）

| 步骤 | 结果 | 含义 | P0 范围 |
|---|---|---|---|
| ① h264_nvenc rc≠0 | 环境不符 | NVENC 不可用 | 停，修环境 |
| ④ vbr_hq_cli rc=0 | FFmpeg 9.0 未移除 | 问题不存在 | 无需改 |
| [CLI-test] vbr_hq_cli rc≠0 + 迁移 rc=0 | CLI 可迁移 | CLI 映射是底线 | 必改 CLI |
| ③ 头文件有 VBR_HQ=32 | SDK 12.x | ctypes 路径可能工作 | [SDK-test] 决定 |
| ③ 头文件无 VBR_HQ | SDK ≥13.0 | ctypes 路径必崩 | 方案 B |
| [SDK-test] ctypes 接受 rc_ptr[1]=32 | SDK 接受 | ctypes 路径安全 | 方案 A |
| [SDK-test] ctypes 拒绝 rc_ptr[1]=32 | SDK 拒绝 | ctypes 路径需迁移 | 方案 B |

---

## 3. 决策树（含 CR-2 约束）

```
① h264_nvenc rc=0? ──否──→ 停止，修 NVENC 环境
    │
   是
    │
  [CLI-test] vbr_hq_cli rc≠0? ──否──→  FFmpeg 9.0 未拒绝 vbr_hq（意外）→ 重新评估
       │    │
      是   [CLI-test] 迁移 rc=0?
       │    ├─否──→ 备选：-tune hq 无 multipass / 降级 FFmpeg
       │    │
      是   是
       │    │
       │    ├─③ 头文件无 VBR_HQ → 方案 B（ctypes 必崩）
       │    │
      是   是
       │    │
       │    └─ [SDK-test] ctypes 接受?
       │        ├─否 → 方案 B（ctypes 加 vbr 分支）
       │        │
      是     是
       │    │
       └─→ 方案 A：P0 = 仅 CLI 映射
```

**标签说明**：
- `[CLI-test]` = Gate 0 步骤②中的 CLI 迁移测试（vbr_hq 被拒 + vbr 迁移可用）
- `[SDK-test]` = Gate 0 步骤⑤中的 ctypes 实编测试（SDK 是否接受 rc_ptr[1]=32）

### 方案 A：仅 CLI 映射（最小改动）

**前提**：[SDK-test] ctypes 实编通过（SDK 仍接受 VBR_HQ=32）

| 文件 | 改动 | 说明 |
|---|---|---|
| `external/ifrnet_video/ffmpeg_io.py:904` | `_NVENC_RC_MAP['vbr_hq']='vbr_hq'` → `'vbr'` | 映射目标改为 vbr |
| `external/realesrgan_video/ffmpeg_io.py:904` | 同上 | 同上 |
| `external/ifrnet_video/ffmpeg_io.py:923` | vbr 路径追加 `-tune hq -multipass fullres` | 构建命令时附加 |
| `external/realesrgan_video/ffmpeg_io.py:936` | 同上 | 同上 |
| `config/default_config.json:97,171` | `"rate_mode": "vbr_hq"` **保留** | 内部名不变（SDK 路径仍用） |
| `external/ifrnet_video/nvenc_sdk.py` | **不动** | SDK ctypes 路径仍用 vbr_hq=32 |
| `external/realesrgan_video/nvenc_sdk.py` | **不动** | 同上 |
| `external/ifrnet_video/main.py:1133` | 默认参数 **不动** | 同上 |

**风险**：
- `-tune hq -multipass fullres` 的画质/码率行为与旧 `vbr_hq` 可能有细微差异 → 需 §3 画质对比验证
- SDK ctypes 路径 `rc_ptr[1]=32` 仍依赖 SDK 13.0 的接受度 → 若后续 SDK 升级移除则需再迁

### 方案 B：CLI + ctypes 双迁移（较大改动）

**前提**：[SDK-test] ctypes 实编拒绝 VBR_HQ=32（SDK 13.0 已移除）

**核心问题**：`nvenc_sdk.py` 的 `_build_encoder_config` 只有三条分支：
```python
if self._rate_mode == 'constqp':   → rc_ptr[1] = 0
elif self._rate_mode == 'vbr_hq':  → rc_ptr[1] = 32
elif self._rate_mode == 'qvbr':    → rc_ptr[1] = 64
else:                              → rc_ptr[1] = 0 (CONSTQP, 静默落地!)
```
`vbr` 会落入 `else` 分支，被当成 CONSTQP 处理，且 LA 也会被静默禁用（LA 启用条件 `self._rate_mode in ('vbr_hq', 'qvbr')`）。

**改动清单**（在方案 A 基础上增加）：

| 文件 | 改动 | 说明 |
|---|---|---|
| `external/ifrnet_video/nvenc_sdk.py` | 新增 `vbr` 分支 | `rc_ptr[1] = 1` (NV_ENC_PARAMS_RC_VBR) + `targetQuality` + `lookahead` |
| `external/realesrgan_video/nvenc_sdk.py` | 同上 | 同上 |
| `external/ifrnet_video/nvenc_sdk.py:4413` | `_NVENC_LEVEL1_RATE_MODE = "vbr_hq"` → `"vbr"` | Level 1 默认模式名 |
| `external/ifrnet_video/main.py:1133` | `rate_mode="vbr_hq"` → `rate_mode="vbr"` | 默认参数 |
| `external/realesrgan_video/main.py:796,829` | `getattr(...,'vbr_hq')` → `'vbr'` | 默认值 |
| `src/utils/config_manager.py:68-69,108-109` | `"vbr_hq"` → `"vbr"` | 配置默认值 |
| `config/default_config.json:97,171` | `"rate_mode": "vbr_hq"` → `"vbr"` | 配置默认值 |
| T4 方案 §4.1 BASE_LOCK | `-rc:v vbr_hq` → `-rc:v vbr -tune hq -multipass fullres` | harness 参数 |
| T4 方案 §12.3 CR-2 | 路线B 重裁定 | "为何选 vbr_hq"的理由已失效 |

**新增 `vbr` 分支的关键细节**：

```python
# nvenc_sdk.py _build_encoder_config 中新增：
elif self._rate_mode == 'vbr':
    # FFmpeg 9.0 迁移：vbr_hq 移除后，vbr + targetQuality 等效于旧 vbr_hq
    # NV_ENC_PARAMS_RC_VBR = 1 (0x1)
    rc_ptr[1] = 1                            # NV_ENC_PARAMS_RC_VBR
    _est_br = _clamp_bitrate(int(width * height * fps * 3.0), width, height)
    rc_ptr[5] = _est_br                      # averageBitRate @offset 20
    rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24
    _tq = max(1, _qp_val)                    # targetQuality = CRF (QP标度)
    _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
    _tq8_ptr[0] = _tq & 0xFF
    # targetQuality: uint8_t at rcParams+88 (需实测确认 vbr 模式下 offset 是否与 vbr_hq 相同)
    print(f"[NVENCEncoder] VBR: crf={_qp_val} targetQuality(CRF)={_tq} avgBitrate={_est_br//1000}kbps", flush=True)

# LA 启用条件同步更新：
if self._la_depth > 0 and self._rate_mode in ('vbr_hq', 'vbr', 'qvbr'):
    # ... 启用 lookahead
```

> ⚠ **`targetQuality` offset 验证**：`vbr` 模式下 targetQuality 的 offset 可能与 `vbr_hq` 不同（SDK 文档中 VBR 的 targetQuality 位置可能与 VBR_HQ 不同）。[SDK-test] 实测时需记录 rcParams 完整布局，确认 `targetQuality@+88` 在 `vbr` 模式下是否仍然正确。

---

## 4. 迁移路径实测（CLI 路径，依赖 ⑤ 通过）

```bash
# 3.1 等质量标定预跑（单素材单点，验证迁移后画质无退化）
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out temp/eqq_t4_verification/cq_live_kids_play \
    --tiers h264_nvenc,hevc_nvenc \
    --duration 6 --src-is-prep < /dev/null

# 3.2 对比：旧 vbr_hq vs 新 vbr-tune-hq（同素材同锚点）
# 旧（预期 rc≠0 失败）：
ffmpeg -i input.mp4 -c:v h264_nvenc -rc:v vbr_hq -cq:v 23 -b:v 0 -f null - < /dev/null
# 新（预期 rc=0）：
ffmpeg -i input.mp4 -c:v h264_nvenc -rc:v vbr -tune hq -multipass fullres -cq:v 23 -b:v 0 -f null - < /dev/null

# 3.3 VMAF 对比（同素材，同锚点 23）
# 旧路径产物 vs 新路径产物 → VMAF 逐帧对比
# 判据：ΔVMAF ≤ 0.3（合成素材已知边界，真实素材以实测为准）
```

---

## 5. LA 联动验证

`vbr_hq` 模式下 LA 由 `self._rate_mode in ('vbr_hq', 'qvbr')` 控制启用。迁移后需确认 LA 仍正常：

```bash
# 5.1 确认 vbr 模式下 LA=8 仍启用
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v h264_nvenc -rc:v vbr -tune hq -multipass fullres \
       -cq:v 23 -b:v 0 -rc-lookahead 8 -f null - < /dev/null 2>&1
# grep 日志确认 "lookahead" 启用

# 5.2 端到端 pipeline（IFRNet，hevc + LA=8）
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --codec-ifrnet hevc_nvenc --codec-esrgan hevc_nvenc \
    --rate-mode-ifrnet vbr --lookahead-depth-ifrnet 8 < /dev/null
# 判据：退出码 0 + 段数正确 + 无超时挂起
```

> ⚠ **LA 启用条件差异**：
> - 方案 A（SDK 仍接受 vbr_hq）：`nvenc_sdk.py` 的 LA 条件 `in ('vbr_hq','qvbr')` 不变，CLI 路径的 LA 由 `-rc-lookahead` 控制 → 两条路径 LA 独立
> - 方案 B（新增 `vbr` 分支）：`nvenc_sdk.py` 的 LA 条件需同步更新为 `in ('vbr_hq','vbr','qvbr')`，否则 SDK 路径的 LA 在 `vbr` 模式下被静默禁用

---

## 6. 诊断脚本更新（实测后）

| 脚本 | 改动 | 时机 |
|---|---|---|
| `nvenc_rc_mode_diagnose.py` | `KNOWN_SDK_RC_VALUES["sdk10"]` 中 `VBR_HQ` 注释为 deprecated | 实测后更新 |
| `nvenc_rate_mode_upgrade_check.sh` | 测试用例 `1a` 期望从 `VBR_HQ.*LA=8` 改为 `VBR.*HQ.*LA=8` 或 `VBR.*tune.*hq` | 实测后更新 |
| `nvenc_targetquality_offset_diagnose.py` | `NV_ENC_PARAMS_RC_VBR_HQ = 32` 注释为 deprecated | 实测后更新 |
| `nvenc_completion_event_matrix_v{1-5}.py` | `RATE_MODE = "vbr_hq"` → 根据实测结果 | 实测后更新 |

---

## 7. 产出与决策点

### 交付物
1. **Gate 0 输出**：①②③④⑤ 的结果
2. **画质对比**：ΔVMAF（合成素材 + 真实素材各一条）
3. **LA 联动验证**：vbr 模式下 LA 是否正常启用

### 决策树

```
① h264_nvenc rc=0? ──否──→ 修 NVENC 环境
    │
   是
    │
② ④ vbr_hq_cli rc≠0 + ⑤ 迁移 rc=0?
        │    └─ ⑤ rc≠0 → 备选：-tune hq 无 multipass / 降级 FFmpeg
        │
       是
        │
       ③ 头文件有 VBR_HQ=32?
        │    └─ 无 → 方案 B（ctypes 必崩）
        │
       是
        │
       ⑤ ctypes 接受?
        ├─ 否 → 方案 B（ctypes 加 vbr 分支）
        │
       是
        │
        └─→ 方案 A：P0 = CLI 映射 only
            后续：③ 画质对比 + ⑤ LA 联动验证 → 进入 T4 标定主线
```

---

## 8. 与 T4 母版方案的联动

本专项结论直接影响 `Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md`：

| 专项结论 | T4 方案改动 |
|---|---|
| 方案 A（仅 CLI） | §4.1 BASE_LOCK h264/hevc `-rc:v vbr_hq` → `-rc:v vbr -tune hq -multipass fullres`；CR-2 路线B 重新裁定（理由从"SDK 不支持 vbr"变为"FFmpeg 9.0 移除 vbr_hq"） |
| 方案 B（双迁移） | §4.1 BASE_LOCK + nvenc_sdk.py `_NVENC_LEVEL1_RATE_MODE` + config 默认值全改 `vbr`；CR-2 路线B 重裁定；nvenc_sdk.py 新增 `vbr` 分支 |
| LA 联动异常 | §4.2 LA 启用条件同步更新 |

L40 方案（`PROMPT_L40_AV1等质量标定专项执行方案.md`）**不受影响**（AV1 早已用 `vbr`）。

---

## 9. 风险与坑

| 风险 | 对策 |
|---|---|
| `-tune hq -multipass fullres` 在 T4 上被驱动拒绝 | Gate 0 ⑤ 先验；reject 则回退 `-tune hq`（无 multipass） |
| `vbr` 模式下 `targetQuality` offset 与 `vbr_hq` 不同 | 2.⑤ 实测时记录 rcParams 布局；方案 B 新增分支时对照验证 |
| 方案 B 的 `vbr` 分支 LA 条件未同步 → LA 被静默禁用 | 代码审查必须检查 `_rate_mode in (...)` 条件是否含 `vbr` |
| 跨段复用缓存 key 中 `rate_mode` 变化导致旧缓存不兼容 | 改 default_config.json 后清 `temp/` |
| 并发会话抢 GPU（共享主机） | `nvidia-smi` 查负载；`--jobs 1` |
| 会话中途 GPU 被回收 | 每步当场复跑；报告写实测时刻 |
| `< /dev/null` 缺失致 SIGTTOU 假挂起 | 全部命令加 |

---

## 10. 方案 A 落地记录（2026-10-04）

按方案 A 落地（仅 CLI token 迁移；SDK ctypes 与内部名不动）：

| 文件 | 改动 |
|---|---|
| `external/ifrnet_video/ffmpeg_io.py` | `[FIX-FFMPEG9-VBRHQ]`：`_rc_v_map` vbr_hq/qvbr→vbr；vbr_hq/qvbr 追加 `-tune hq -multipass fullres`（h264/hevc） |
| `external/realesrgan_video/ffmpeg_io.py` | 同上（`_NVENC_RC_MAP`），与 ifrnet 侧逐字同源（G5-10 校验） |
| `external/ifrnet_video/nvenc_sdk.py` / `external/realesrgan_video/nvenc_sdk.py` | 仅加 `[PLAN-B-CANDIDATE]` 注释（else→CONSTQP、LA 门控），**行为不变** |
| `Accessory/probe/calibrate_equal_quality.py` | BASE_LOCK h264/hevc → `vbr -tune hq -multipass fullres`；selftest 同步；CR-2 理由更新 |
| `Accessory/verify/crf_cq_unification_verify.py` | `enc_nvenc` / `_probe_raw_driver_illegal` / G6-1 / G6-4 命令形状同步 |
| `Accessory/probe/av1_vp9_quality_matrix.py` | `_PROD_RC` h264/hevc → `vbr -tune hq -multipass fullres` |
| `Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md` | §4.1 BASE_LOCK + CR-2 二次更新 |
| `Plan/FFmpeg9.0-移除-vbr_hq-速率控制模式.md` | 补 §七：`qvbr` 一并移除 + 驱动仍接受 SDK 32 + 方案 A/B |
| `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` | CR-2 引用同步（AV1 不受影响） |

**未做（Plan A 边界）**：内部 `rate_mode` 名 / config JSON / `config_manager` / `main.py` 默认参数 / 缓存 key / `nvenc_sdk` ctypes 行为。
**跨仓 handoff**：VU 侧 h264/hevc 的 harness/生产/探针 `-rc:v vbr_hq` 需同步迁移，否则共享 `QUALITY_MAP` 的 ⑨ 组变红（本仓无法代改）。
