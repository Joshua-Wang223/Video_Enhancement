---
name: T4 vbr_hq 移除验证专项
description: FFmpeg 9.0 移除 vbr_hq 后 T4 GPU 验证立项方案（2026-10-03）
type: project
---

FFmpeg 9.0.2（编译用 libffmpeg-nvenc-dev 12.1.14.0）移除 `-rc:v vbr_hq`，但 SDK 12.x 头文件仍定义 `NV_ENC_PARAMS_RC_VBR_HQ=32`。核心判定：FFmpeg CLI 层必炸，ctypes SDK 层（`rc_ptr[1]=32`）需 T4 实测确认。

验证方案见 `Plan/T4_NVENC_vbr_hq移除_验证专项.md`，关键判定点：
- Gate 0：`vbr -tune hq -multipass fullres` 替代路径可用性（预期 rc=0）
- 2.3：ctypes `rc_ptr[1]=32` 的 `InitializeEncoder` 是否被 SDK 13.0 接受
  - 接受 → P0 只改 CLI 映射（方案 A）
  - 拒绝 → P0 需 ctypes 全量迁移（方案 B）
- 3.3：迁移后 ΔVMAF ≤ 0.3
- 4：LA 在 `vbr -tune hq` 下是否正常启用

决策树在方案 §8。结论直接决定 T4 母版 `BASE_LOCK`/`CR-2`/`_NVENC_LEVEL1_RATE_MODE` 改动范围。L40 不受影响（AV1 早已用 `vbr`）。

Why: FFmpeg 9.0 升级后生产环境必现，GPU 验证决定改动范围（CLI-only vs 全量迁移）。
How to apply: T4 上机时按 §1 Gate 0 → 2.3 → 3 → 4 顺序执行，§8 决策树决定 P0 范围。
