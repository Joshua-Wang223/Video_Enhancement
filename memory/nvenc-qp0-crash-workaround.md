# NVENC `qp=0` (CRF=0 无损) 段错误 —— 生产规避策略

## 背景

2026-09-16 Linux + Tesla T4（驱动 580.65.06 / CUDA 13.0）实测确认：
`NVENCEncoder(qp=0, rate_mode='constqp', la_depth=0)` 进行编码时，**约 35-67% 概率触发 SIGSEGV**。

触发路径：用户设置 `crf=0`（无损编码）→ 配置层 `_NVENC_CRF0_FORCE_CONSTQP=True`（默认开启）→ 强制走 `constqp` + `qp=0` + `la=0`。

**同一编码器配置、仅把 `qp` 改为 23 → 0/10 崩** ⇒ 唯一触发条件是 `qp=0`。

## 现象

- 崩溃发生在**编码阶段**（`encode_frames_batch_ce_pipeline` 或 `encode_frames_batch`），不在建会话
- `faulthandler` 落点游走：`_drain_outputs_blocking` 的 `lock_bs_fn`、GC 阶段、解释器退出
- `gdb` 原生栈：`gc_collect_main → subtract_refs → visit_decref → _PyObject_IS_GC(obj=<坏指针>)` —— **堆已损坏，显现位置随机**
- 伴随现象：帧丢失（`empty=1`、`none=1`）

## 排除的假设（避免重复调查）

| 假设 | 排除依据 |
|---|---|
| `_NvEncPresetConfig` / `_NvEncConfig` 尺寸偏小越界 | 已修正为真实布局（5128/3584），崩溃率**未改善**；驱动实际只写 `presetCfg`（@8 起 3584B，止于 3592 < 5128） |
| `_FUNC_IDX` 函数表索引错位 | 逐条比对 `/usr/include/ffnvcodec/nvEncodeAPI.h` 的 `NV_ENCODE_API_FUNCTION_LIST`，21 条**全部正确** |
| `LockBitstream` 垃圾 size → `from_address` 越界读 | 5 处 `from_address` 站点**全部**有 `_is_legal_bitstream_size` 前置 |
| `_slot_pending` 元组形状不一致 | 4 元组（`encode_frames_batch`）与 5 元组（`ce_pipeline`）**按路径各自自洽** |
| `close()` 后释放 CUDA 张量引发 | 不释放张量/不主动 GC 也照样崩（含 2 次在 `atexit`） |
| 缓冲区尺寸类不匹配 | `NV_ENC_PIC_PARAMS` 3360、`NV_ENC_LOCK_BITSTREAM` 1544、`NV_ENC_CREATE_*` 776、`CUDA_MEMCPY2D` 128 —— 逐个 gcc `sizeof` 实测，**完全一致** |

## 根因定性

**NVENC 驱动 / SDK 层面的缺陷**（qp=0 无损路径的内部状态管理），而非本项目代码逻辑错误。  
属于「驱动 Bug 类」，短期靠代码层面规避，长期需驱动升级或 NVIDIA 反馈。

## 生产规避策略（立即可用）

### 1. 配置层面：禁用 `crf=0` 强制 CONSTQP

编辑 `config/default_config.json` 或运行时覆盖：

```json
{
  "nvenc": {
    "_NVENC_CRF0_FORCE_CONSTQP": false
  }
}
```

或 CLI：
```bash
python src/main_video_optimized.py ... --nvenc-crf0-force-constqp false
```

效果：`crf=0` 不再强制 `qp=0`，改走 CRF 路径（`qp` 由驱动动态决定），避开缺陷路径。  
**代价**：无损质量不再绝对保证（CRF 0 约等于近无损，非数学无损）。

### 2. 显式指定 `qp>=1`

若必须用 CONSTQP，手动指定 `qp=1`（近无损，体积略增）：

```bash
python src/main_video_optimized.py ... --nvenc-rate-mode constqp --nvenc-qp 1
```

### 3. 启用 Lookahead（LA>0）

实测 `la_depth>0` 时走分块 `encode_frames_batch` 路径，崩溃率**显著更低**（但未归零，仍建议配合上述）：

```bash
python src/main_video_optimized.py ... --nvenc-lookahead 32
```

### 4. 环境变量兜底（驱动层）

```bash
export MALLOC_CHECK_=3 MALLOC_PERTURB_=165 PYTHONMALLOC=malloc
```
实测 5/5 无崩（样本极小，仅作辅助，不能单独依赖）。

## 复现验证工具

```bash
# 完整复验（约 2-3 分钟）
python tests/diagnose_nvenc_qp0_segv.py --iters 15

# 快速冒烟（约 30 秒）
python tests/diagnose_nvenc_qp0_segv.py --iters 5 --modes ctor_only ce_pipeline

# 对照组验证 qp=23 无崩
python tests/diagnose_nvenc_qp0_segv.py --iters 5 --qp 23
```

退出码：`1` = 观测到段错误（缺陷可复现），`0` = 未观测到。

## 相关文件

- `tests/diagnose_nvenc_qp0_segv.py` —— 可复用诊断脚本
- `Plan/NVENC硬件测试隔离_立项Prompt.md` —— 完整调查记录（已修正归因）
- `memory/crf0-la-depth-cli-ignore-fix.md` —— CRF=0 强制 CONSTQP 的配置来源
- `external/realesrgan_video/nvenc_sdk.py:558` — `_NVENC_CRF0_FORCE_CONSTQP` 常量定义处
- `external/realesrgan_video/nvenc_sdk.py:633` — CRF=0 强制转 CONSTQP 的日志点

## 更新记录

- 2026-09-16：首次记录，基于 T4/CUDA13.0 实测定性
- 关联会话记录：`Plan/Conversation-持续跟进6方案计划（GPU部分）-2026-09-17.txt`