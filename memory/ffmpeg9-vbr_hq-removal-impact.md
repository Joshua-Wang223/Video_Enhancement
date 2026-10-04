---
name: FFmpeg 9.0 移除 vbr_hq 速率控制模式的影响与应对
description: FFmpeg 9.0 移除 vbr_hq RC 模式对等质量标定方案和生产代码的影响分析及应对策略（2026-10-03）
type: project
---

## 事实

FFmpeg 9.0 移除 `vbr_hq` 速率控制模式（NVIDIA SDK 10.0+ 已废弃 `NV_ENC_PARAMS_RC_VBR_HQ` mode=32）。迁移路径：`-rc vbr -tune hq -multipass fullres`。

项目有**两条编码路径**依赖 `vbr_hq`：
- **Level 2 FFmpeg CLI**（`ffmpeg_io.py`）：下发 `-rc:v vbr_hq` → FFmpeg 9.0 直接报错 `Specified rc mode is deprecated`
- **Level 1 SDK ctypes**（`nvenc_sdk.py`）：`rc_ptr[1] = 32` → 取决于驱动/SDK版本，可能仍接受但已 deprecated

## 影响范围

| 严重度 | 位置 | 改动 |
|---|---|---|
| P0 | `config/default_config.json:97,171` | `"rate_mode": "vbr_hq"` 默认值 |
| P0 | `external/ifrnet_video/ffmpeg_io.py:904` | `_NVENC_RC_MAP['vbr_hq']='vbr_hq'` |
| P0 | `external/realesrgan_video/ffmpeg_io.py:904` | 同上 |
| P0 | `external/ifrnet_video/main.py:1133` | `rate_mode="vbr_hq"` 默认参数 |
| P0 | `external/realesrgan_video/main.py:796,829` | `getattr(...,'vbr_hq')` |
| P0 | `src/utils/config_manager.py:68-69,108-109` | `"rate_mode": "vbr_hq"` |
| P1 | T4 方案 §4.1 BASE_LOCK / CR-2 | h264/hevc `-rc:v vbr_hq` 与 FFmpeg 9.0 冲突 |
| P1 | `Accessory/probe/calibrate_equal_quality.py:117,121` | 标定脚本引用 `-rc:v vbr_hq` |
| P2 | `nvenc_sdk.py:1002-1005` | `rc_ptr[1]=32` 需实测 SDK 13.0 是否仍接受 |
| P2 | `_NVENC_LEVEL1_RATE_MODE = "vbr_hq"` | Level 1 默认模式名 |

## L40 方案影响

**几乎不受影响**：AV1 早已用 `vbr`（`BASE_LOCK['av1_nvenc'] = ['-rc:v','vbr',...]`），T4 方案 §3 A3 已明确 AV1 必须用 `vbr`。L40 方案只需同步 T4 方案更新后的引用。

## 应对方案（推荐：方案 A —— 保守迁移）

**核心**：内部 `rate_mode` 名称不变（避免大范围重构），只改 FFmpeg CLI 实际下发值。

1. `ffmpeg_io.py`（两侧）：`_NVENC_RC_MAP['vbr_hq']` → `'vbr'`，vbr 路径追加 `-tune hq -multipass fullres`
2. `config/default_config.json`：`"rate_mode": "vbr_hq"` 保留（内部名不变）
3. `nvenc_sdk.py`（两侧）：`rc_ptr[1]=32` 暂不动，先实测 SDK 13.0 是否仍接受
4. `main.py`（两侧）：默认参数暂不动
5. `_NVENC_LEVEL1_RATE_MODE`：暂不改

**关键约束**：VE `nvenc_sdk` 不支持 `vbr`（静默落 CONSTQP + LA 失效）⇒ **否决「VE 改 vbr」**。CTypes 路径必须保持 `vbr_hq` 直到 SDK 层面确认移除。

## 与 T4 方案的关系

T4 方案 CR-2 路线B（h264/hevc → `vbr_hq`）需重新裁定：FFmpeg 9.0 下 `vbr_hq` CLI 不可用，但 ctypes 路径仍可用 `vbr_hq`（SDK 13.0 可能仍支持）。建议方案：
- **生产代码**：ctypes 路径保持 `vbr_hq`（SDK 仍支持），FFmpeg CLI 路径迁移到 `vbr -tune hq -multipass fullres`
- **harness 标定**：`BASE_LOCK` 改用 `-rc:v vbr -tune hq -multipass fullres`（与生产 FFmpeg CLI 路径一致）
- **跨仓同步**：VU 侧 h264/hevc 生产默认 rc=`--rc-mode auto`=不发 `-rc`，VE 保持 ctypes 直连 SDK Level 1 `vbr_hq`，口径差异需注明

## 待办

- [x] **实测 SDK 13.0 `rc_ptr[1]=32` 是否仍接受（T4 机器）→ 2026-10-04 接受**（方案 A 成立；头文件虽删枚举，驱动仍兼容 32）。详见 [[t4-vbrhq-verification-plan]]
- [ ] 更新 T4 方案 CR-2 路线B 裁定（理由改为"FFmpeg 9.0 移除 vbr_hq"）
- [ ] 更新 `calibrate_equal_quality.py` BASE_LOCK
- [x] **GPU 验证：`-rc:v vbr -tune hq -multipass fullres -cq:v N` 画质 vs 旧 `vbr_hq`** → ΔVMAF h264 −0.048 / hevc −0.130（≤0.3 PASS）
- [ ] 更新 `crf_cq_unification_verify.py` G6 测试用例
- [ ] 更新 `plan_implementation_gate.py` 的 `rate_mode == "vbr_hq"` 断言
- [ ] L40 方案同步 T4 方案引用
