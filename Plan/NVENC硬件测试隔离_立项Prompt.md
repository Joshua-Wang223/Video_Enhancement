# 立项 Prompt：全量 `pytest Accessory/` SIGSEGV —— NVENC 硬件测试的状态隔离

> ## ✅ 执行状态（2026-09-16 Linux + T4 实测完成，重大结论更新）
>
> **原归因（跨测试类状态污染）已被推翻** —— 真实根因是 **NVENC `qp=0`（CRF=0 无损）编码路径的固有段错误**。
>
> | 交付 | 说明 |
> |---|---|
> | `Accessory/probe/nvenc_qp0_segv_repro.py` | **新增**：可复用的 qp=0 段错误复现与定位脚本，三模式（ctor_only / ce_pipeline / batch_direct）+ 对照组（qp=23），每次在独立子进程跑 |
> | `Accessory/run_all_isolated.sh` | 方案 A：每文件独立 pytest 进程（仍有价值：防止其他测试互污染） |
> | `pytest.ini` | 注册 `hw` 标记 + `testpaths = tests` + `-p no:cacheprovider` |
>
> ### ⚠️ 关键修正：SIGSEGV 非测试隔离问题
>
> 2026-09-16 实测（Tesla T4 / 驱动 580.65.06 / CUDA 13.0）证明：
> - `nvenc_sdk_realesrgan_suite.py::TestNVENCEncoder::test_no_empty_frames_constqp_la0` **单独跑也会崩**（3/8）
> - **单测独立循环**也会崩（3/10）
> - 崩溃与测试顺序、前置文件**无关**
> - 真正触发条件：**`NVENCEncoder(qp=0, rate_mode='constqp', la_depth=0)` 编码**（即用户设 `crf=0` 触发的无损路径，见 `memory/crf0-la-depth-cli-ignore-fix.md`）
> - **同一编码器配置、只改 `qp=23` → 0/10 崩** ⇒ 触发条件是 `qp=0`，非 constqp / LA=0 本身
> - `ctor_only`（建会话 + close，不编码） → **0/15 崩** ⇒ 故障在**编码阶段**，非建会话
>
> **结论**：隔离跑（方案 A）**无法根治**，因为单个测试类本身就会崩。该缺陷在 NVENC 驱动/编码器层（qp=0 无损路径）。
>
> ### 已做的验证
>
> * ✅ 真实全量 SIGSEGV 复现、单测隔离复现、二分定位到 `qp=0` 编码阶段
> * ✅ `Accessory/probe/nvenc_qp0_segv_repro.py` 固化复现逻辑（可跨环境复验）
> * ✅ 排除假设：结构体尺寸、函数表索引、LockBitstream size cap、_slot_pending 元组形状、CUDA 张量释放、缓冲区尺寸 —— 均非根因
> * ⬜ 待办：qp=0 路径的深层根因（需驱动级调试或 NVENC SDK 升级）、生产规避策略（文档化 CRF=0 风险 / 回退到 qp>0）
>
> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。
> 立项时间：2026-09-15　立项人：门禁强化会话
> **重大更新：2026-09-16 Linux + T4 实测** 立项人：NVENC 隔离会话
> 数据来源：`Accessory/probe/nvenc_qp0_segv_repro.py` 实测 + `Plan/Conversation-持续跟进6方案计划（GPU部分）-2026-09-17.txt` 调查记录

---

## ✅ 状态总览（动手前必读）—— **2026-09-16 更新**

| 项 | 状态 | 说明 |
|---|---|---|
| 现象：全量 `pytest Accessory/` 硬崩 | ✅ **已定性** | EXIT=139（SIGSEGV），非测试污染，**真实缺陷**：`qp=0` 编码路径 |
| 单独跑崩溃的类 | ✅ **确认崩** | `TestNVENCEncoder::test_no_empty_frames_constqp_la0` 单独跑 3/8 崩 |
| 是否与解码/读帧链路耦合 | ✅ 已定性：**无耦合** | `nvenc_sdk.py` 只依赖 stdlib + numpy + torch + ctypes |
| 非 NVENC 套件 | ✅ 通过 | `test_chroma_false_positive.py` → 2 passed |
| core dump 遗留 | ⚠️ 既有 | 现有 core 文件源于 qp=0 编码缺陷，非测试隔离问题 |
| 本立项（隔离/修复） | 🔄 **重定向** | 原"隔离测试"任务改为：**记录 qp=0 缺陷、提供诊断工具、文档化规避策略** |

---

## 0. 任务

让 `pytest Accessory/` 能够**不崩**地跑完（或明确地按类隔离执行），
使"全量回归"重新成为一个可用的动作，而不是一个会产生 core dump 的雷区。

产出物：**崩溃点定位** + **隔离方案**（三选一，见 §3）+ 可重复的执行入口。

---

## 1. 环境与代码基线

- `Accessory/` 下与 NVENC 直接相关的测试 **14 个**：

  | 文件 | 备注 |
  |---|---|
  | `nvenc_completion_event_matrix.py` + `_v1` ~ `_v5` | 6 个版本并列（历史演进） |
  | `nvenc_comprehensive_matrix.py` | |
  | `nvenc_ipc_worker_probe.py` | 注意 `_worker` 后缀，可能是被 spawn 的子进程脚本 |
  | `nvenc_la_frame_conservation_suite.py` | LA 帧守恒（**本立项最相关**，含 `_drain_outputs()` 模板） |
  | `nvenc_session_pre_torch_probe.py` | |
  | `nvenc_sdk_realesrgan_suite.py` | 会话中崩在它/同族类 |
  | `nvenc_vbr_hq_offsets_probe.py` | |
  | `pipe4_la8_corruption_diff.py` | |
  | `sps_pps_startup_repro.py` | |

- `Accessory/conftest.py` 只有 `collect_ignore_glob`（忽略 `* - Copy*`），**没有任何 fixture/隔离机制**。
- `.gitignore` 已含 `*.core` / `core.*`。

---

## 2. 已确认的事实（2026-09-16 Linux + T4 实测，**推翻原归因**）

### 2.1 崩溃形态 —— 真实缺陷，非测试污染

```
测试：test_no_empty_frames_constqp_la0  (qp=0, constqp, la=0)
→ EXIT=139 (SIGSEGV)，**单独跑也会崩**（3/8），循环单测也崩（3/10）
```

| 模式 | 配置 | 崩溃率 | 关键观测 |
|---|---|---|---|
| `ctor_only` | qp=0, build+close，**不编码** | **0/15** | 故障不在建会话 |
| `ce_pipeline` | qp=0, 生产 LA=0 入口 | **10/15** | 生产入口同样崩 |
| `batch_direct` | qp=0, 非生产入口 | 4/10 | 入口函数非决定因素 |
| **对照** | **qp=23**, ce_pipeline | **0/10** | **qp=0 是唯一触发条件** |

⇒ **故障在编码阶段（qp=0 无损）**，与测试顺序/隔离/前置文件无关。

### 2.2 崩溃现场特征（堆损坏，显现位置随机）

- `faulthandler` 落点游走：`_drain_outputs_blocking` 的 `lock_bs_fn(...)`、GC 阶段、pytest `code.py`
- `gdb` 原生栈：`gc_collect_main → subtract_refs → visit_decref → _PyObject_IS_GC(obj=<坏指针>)` —— **GC 遍历时碰到已损坏对象**
- 崩溃点含 `close()` 后的 `del`/`gc.collect()` 与解释器退出（`atexit`）阶段
- ⇒ **进程内堆/驱动状态被破坏**，只是显现位置随机

### 2.3 排除的假设（避免后续会话重复调查）

| 假设 | 排除依据 |
|---|---|
| `_NvEncPresetConfig` / `_NvEncConfig` 尺寸偏小越界 | 已修正为真实布局（5128/3584），崩溃率**未改善**；驱动实际只写 `presetCfg`（@8 起 3584B，止于 3592 < 5128） |
| `_FUNC_IDX` 索引错位 | 逐条比对 `/usr/include/ffnvcodec/nvEncodeAPI.h` 的 `NV_ENCODE_API_FUNCTION_LIST`，21 条**全部正确** |
| `LockBitstream` 垃圾 size → `from_address` 越界读 | 5 处 `from_address` 站点**全部**有 `_is_legal_bitstream_size` 前置 |
| `_slot_pending` 元组形状不一致 | 4 元组（`encode_frames_batch`）与 5 元组（`ce_pipeline`）**按路径各自自洽** |
| `close()` 后释放 CUDA 张量引发 | 不释放张量/不主动 GC 也照样崩（含 2 次在 `atexit`） |
| 缓冲区尺寸类不匹配 | `NV_ENC_PIC_PARAMS` 3360、`NV_ENC_LOCK_BITSTREAM` 1544、`NV_ENC_CREATE_*` 776、`CUDA_MEMCPY2D` 128 —— 逐个 gcc `sizeof` 实测，**完全一致** |

### 2.4 有用的诊断杠杆

`MALLOC_CHECK_=3 MALLOC_PERTURB_=165 PYTHONMALLOC=malloc` 下实测 **0/5 崩溃**（仅 5 次样本），可作为后续定位堆损坏的入口。

### 2.5 与本文关心的链路无耦合（这点很重要，别在错误的地方找原因）

`external/realesrgan_video/nvenc_sdk.py` 的导入面只有 **stdlib + numpy + torch + ctypes**，
**不 import** `ffmpeg_io` / `pipeline` / `reader_hwaccel`。

⇒ 该 SIGSEGV **不能**用来否定读帧器/解码链路的改动；反之，读帧器改动也不会修好它。

### 2.6 另一个独立的环境噪声（与 qp=0 崩溃无关）

会话中还观察到 pytest 在 **capture teardown** 阶段抛
`ValueError: I/O operation on closed file`（"no tests ran"）。
它发生在测试执行之前/之后的全局 teardown，**与测试内容无关**，属 pytest 版本/环境 artifact。
⇒ 不要把它与 qp=0 SIGSEGV 混为一谈（两者根因不同）。

### 2.7 生产规避策略文档已完成（2026-09-16）

✅ `memory/nvenc-qp0-crash-workaround.md` —— 记录 `crf=0` → 强制 `qp=0` 崩溃风险与建议回退策略（关 `_NVENC_CRF0_FORCE_CONSTQP`、显式 `qp>=1`、启用 LA>0、环境变量兜底）
✅ `Accessory/probe/nvenc_qp0_segv_repro.py` —— 可复用诊断脚本（退出码 1=复现，0=未复现）

---

## 3. 实施步骤 —— **重定向：隔离无法根治，转为缺陷记录与规避**

**原方案 A（隔离跑）无法根治**：单个测试类 `test_no_empty_frames_constqp_la0` 本身就会崩，**不是跨类污染**。

**新方向（三项并行）：**

| 方向 | 交付 | 说明 |
|---|---|---|
| **① 缺陷固化** | `Accessory/probe/nvenc_qp0_segv_repro.py` | ✅ **已完成**：三模式复现 + 对照组，独立子进程，可跨环境复验 |
| **② 生产规避文档** | `memory/nvenc-qp0-crash-workaround.md` | ✅ 已完成（2026-09-16）：记录 `crf=0` → 强制 `qp=0` 崩溃风险，建议生产避免 CRF=0 或回退 `qp>=1` / 启用 LA>0 / 环境变量兜底 |
| **③ 根因深挖（可选）** | 驱动级调试 / NVENC SDK 升级评估 | ⬜ 待排期：需 `cuda-gdb`/`nsight` 或升级驱动/SDK 验证是否修复 |

**方案 A（隔离跑）保留价值**：防止**其他**测试互污染，`Accessory/run_all_isolated.sh` 仍可提供。

**方案 B（清理 fixture）** 降级：仅作通用卫生，不指望根治 qp=0 崩溃。

**方案 C（标记跳过）** 可选：对 `qp=0` 相关测试打 `@pytest.mark.hw_qp0_risk`，CI 可选择跳过。

---

## 4. 判据与验证（更新版）

| # | 判据 | 期望 |
|---|---|---|
| 1 | 复现 | `python Accessory/probe/nvenc_qp0_segv_repro.py --iters 10` 在 T4/CUDA13 环境观测到 `ce_pipeline` 模式 qp=0 崩溃 ≥ 30% |
| 2 | 定位 | 已完成：故障在 **编码阶段**，触发条件 **`qp=0`**，与入口/建会话/测试顺序无关 |
| 3 | 诊断工具可用 | `nvenc_qp0_segv_repro.py` 能在新环境复现/排除该缺陷（退出码 1=复现，0=未复现） |
| 4 | 规避文档存在 | `memory/nvenc-qp0-crash-workaround.md` 记录 CRF=0 风险与建议回退策略 |
| 5 | 无生产误导 | 不把"隔离跑"当根治方案；隔离措施**不得**掩盖 qp=0 真缺陷 |
| 6 | 与门禁不冲突 | `python Accessory/verify/plan_implementation_gate.py --no-report-file` 仍无 FAIL |

**复验命令（标准化）：**
```bash
# 完整复验（约 2-3 分钟）
python Accessory/probe/nvenc_qp0_segv_repro.py --iters 15

# 快速冒烟（约 30 秒）
python Accessory/probe/nvenc_qp0_segv_repro.py --iters 5 --modes ctor_only ce_pipeline

# 对照组验证 qp=23 无崩
python Accessory/probe/nvenc_qp0_segv_repro.py --iters 5 --qp 23
```

---

## 5. 风险与回滚（更新版）

- **原立项不阻塞生产**仍成立：门禁 `plan_implementation_gate.py` 自带的行为阶段已覆盖关键回归。
- **新增风险**：若不文档化 `crf=0` → `qp=0` 崩溃风险，用户在生产开启无损编码会遇到随机段错误（~35-67% 概率）。
- **回滚**：纯 `Accessory/` 层改动（诊断脚本 + 隔离脚本），`git revert` 即可。
- **需要 Linux + GPU（NVENC）** 才能复现与验证；纯 CPU 环境只能跑诊断脚本的逻辑分支（会报环境缺失）。
- **qp=0 根因深挖** 属驱动/SDK 层，可能需 NVIDIA 反馈或升级驱动/SDK 解决；短期以规避为主。
