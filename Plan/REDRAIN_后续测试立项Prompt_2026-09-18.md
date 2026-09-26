---
name: redrain-follow-up-test-plan
status: 后续立项（修复已执行，待真实 GPU 门禁）
description: REDRAIN 缺口修复（§4.1 完成）后续测试与同步门禁
---

# REDRAIN 缺口后续测试立项（2026-09-18）

## 当前修复状态（已完成 §4.1 + §4.2）
- 代码：`external/realesrgan_video/nvenc_sdk.py` 独立 `_ce_final_drain()` 已实现，批末调用点已替换内联 REDRAIN。
- 语法：`py_compile` 通过，无 FAIL。
- 回归行为验证：`Accessory/verify/test_regression_min.py --behavior-only` → 38 PASS / 0 FAIL / 0 WARN / 2 SKIP（环境差异）。
- 审计：`AUDIT_REPORT_2026-09-18.md` §5 已同步为「完整执行」并记录修复执行。
- 记忆：`memory/realesrgan-missing-la-redrain.md` A 侧已更新（修复状态 + §4.2 结果）；B 侧同步受限（只读文件系统，A 为规范源）。
- 敏感凭据：执行全程未读取 `Video_Enhancement_github_token.txt`。
- 真实 GPU 路径：`Accessory/probe/nvenc_la_frame_conservation_suite.py` 在本容器 SKIP（无 NVIDIA GPU），无 FAIL。

## 后续测试门禁（建议顺序执行）

### [P1-必需] 真实 GPU 路径帧数守恒验证
- 执行环境：真实 NVIDIA GPU 节点（T4 或同级，CUDA 可用）。
- 测试脚本：`Accessory/probe/nvenc_la_frame_conservation_suite.py`（段级帧数守恒 + LA=0 / LA=8 × h264 / hevc 四组合）。
- 验收标准：无 FAIL，帧数守恒（输入帧数 == 输出解码帧数），无静默丢帧。
- 风险控制：若真实 GPU 路径出现 FAIL，立即回滚到备份 `nvenc_sdk.py.bak_*` 并重新触发 `_ce_final_drain` 语义审计。

### [P2-必需] 镜像同步门禁
- 条件：B 侧（`/root/.codebuddy/projects/workspace-Video_Enhancement/memory`）恢复可写。
- 执行：`cp -a "$A/." "$B/"`（A = `/workspace/Video_Enhancement/memory`），同步后执行 `diff -rq "$A" "$B"` 复核一致性。
- 复核重点：`realesrgan-missing-la-redrain.md` 内容一致（修复记录 + §4.2 测试结果）；若有文件删除，先用 `comm -13` / `comm -23` 列差异后人工确认再删。

### [P3-建议] 长视频批量冒烟（生产配置）
- 执行脚本：真实 GPU 节点上 `python src/main_video_optimized.py -c config/default_config.json --batch-mode --use-tensorrt-ifrnet --use-tensorrt-esrgan`（小样本输入）。
- 监控指标：段级帧数守恒、`[NVENC-Enc] [FIX-LA-REDRAIN]` 回收计数、`_output_slot_idx` 无异常推进、无空帧防御误触发。
- 验收标准：输出视频存在、帧数完整、码率统计正常，无 `Bitstream parse error`。

### [P4-可选] 回归测试扩展锁定
- 扩展方向：在 `Accessory/verify/test_regression_min.py` 或独立脚本中新增行为断言，验证 ESRGAN 在 `LA=0` 与 `LA=8` 下的帧数守恒（参考 IFRNet `_drain_outputs()` 验证模式）。
- 执行条件：真实 GPU 路径通过后执行，避免在无 GPU 环境添加无效断言。

## 当前阻塞项与解决建议

| 阻塞项 | 原因 | 建议解决方式 |
|---|---|---|
| 真实 GPU 路径 SKIP | 容器无可用 NVIDIA GPU（`cudaGetDeviceCount` 失败） | 在真实 GPU 节点（Linux 生产基准环境）重新执行 `Accessory/probe/nvenc_la_frame_conservation_suite.py` |
| B 侧镜像同步受限 | `/root/.codebuddy/projects/...` 为只读文件系统 | 当文件系统恢复可写后执行 `cp -a` + `diff -rq` 复核 |

## 责任与时间线
- 执行者：本次已由执行者完成 §4.1 代码修复 + §4.2 行为验证。
- 后续执行建议：真实 GPU 节点测试由环境管理员或生产运维执行，结果反馈至本记忆文件（追加执行记录）。
- 审计闭环标准：P1 真实 GPU 路径无 FAIL + P2 镜像同步通过后，`AUDIT_REPORT_2026-09-18.md` §5 可由「完整执行」进一步标注「已通过真实 GPU 门禁验证」。
