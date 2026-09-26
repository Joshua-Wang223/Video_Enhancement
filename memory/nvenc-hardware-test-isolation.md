---
name: nvenc-hardware-test-isolation
description: 全量 pytest Accessory/ SIGSEGV 的隔离处置：方案 A 逐文件独立进程入口已就绪（run_all_isolated.sh + pytest.ini 的 hw 标记），崩溃定位与方案 B/C 需 Linux+GPU
type: project
---

# NVENC 硬件测试的状态隔离

立项：`Plan/NVENC硬件测试隔离_立项Prompt.md`

## 现象（Linux + Tesla T4 会话实测，本轮未复现——本机无 GPU）

全量 `pytest Accessory/` → `.........`（9 passed）后 **EXIT=139 (SIGSEGV)**；
但把崩溃的类（`TestLAAccumulation`）**单独**跑 → 2 passed。
⇒ 跨测试类的状态污染（NVENC 会话 / CUDA context 未释放），不是某个类的缺陷。

**与读帧器/解码链路无耦合**：`nvenc_sdk.py` 只依赖 stdlib + numpy + torch + ctypes，
不 import `ffmpeg_io`/`pipeline`/`reader_hwaccel`。
⇒ 该 SIGSEGV 不能用来否定解码链路的改动，反之亦然。

另有一个**独立的环境噪声**：pytest 在 capture teardown 抛
`ValueError: I/O operation on closed file`（"no tests ran"），发生在测试执行前后，
与测试内容无关 —— **不要**与 SIGSEGV 混为一谈。

## 本轮已落地（纯 CPU 可做部分）

| 交付 | 说明 |
|---|---|
| `Accessory/run_all_isolated.sh` | **方案 A**：每个测试文件单独起一个 pytest 进程，任一崩溃不影响其余；汇总 PASS / FAIL / **CRASH** / EMPTY / OTHER 五态；支持 `--timeout` / `--file` / 透传 `-m` |
| `pytest.ini` | 注册 `hw` 标记 + `testpaths = tests` + `-p no:cacheprovider` |

**有意不设 `addopts = -m "not hw"`**：把硬件测试默认跳过，会让"跑全量"从
"崩掉"变成"静默少跑 14 个文件"，与 `feedback_keep_strict_criteria_annotate`
（保严格判据 + 标注）冲突。要跳过必须**显式** `pytest -m "not hw"`。
待 Linux+GPU 确认方案 B 无法根治后，再考虑翻转默认。

`run_all_isolated.sh` 的**分类逻辑已用 stub pytest 验证**（正常/断言失败/信号
139/无测试收集/用法错误 5/5 分类正确）；完整验证需装 pytest + GPU。

## 仍需 Linux + GPU

1. 复现全量 SIGSEGV（若已不复现，需重新定性）；
2. 二分定位**最小两类组合**：`pytest Accessory/A.py Accessory/B.py -x --tb=short`
   （SIGSEGV 会把 pytest 一起带走，务必 `-x`）；
3. 方案 B（`conftest.py` 加 autouse 清理 fixture：`torch.cuda.synchronize()` +
   释放 NVENC session + 必要时 `empty_cache()`）或方案 C（`@pytest.mark.hw` + 默认排除）；
4. `bash Accessory/run_all_isolated.sh` 全量跑，确认**不再产生 core dump**
   且 pass 数 ≥ 现有单类之和。`core.*` 已在 `.gitignore` 中。

⚠️ 方案 B 要小心"为了让它不崩而跳过/削弱断言" —— 那是把缺陷藏起来。
隔离措施**不得**把失败的类变成跳过。

**Related**：[[feedback_keep_strict_criteria_annotate]]、[[esrgan-cross-segment-sigsegv-fix-stack]]、
[[nvenc-ctypes-integration]]
