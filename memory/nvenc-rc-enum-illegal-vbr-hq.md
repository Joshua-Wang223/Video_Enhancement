---
name: NVENC `rc_ptr[1]` 的 RC 取值 —— 静态断言已撤回，待驱动 caps 裁决
description: 2026-10-05 曾据 FFmpeg 裁剪版头文件断言 vbr_hq=32/qvbr=64「枚举中不存在」，用户以 T4 实机 GPU 直通验证驳回后已撤回；现仅记录仍成立的 B2 根因链（本仓代码内部逻辑，与枚举无关）与 caps 裁决入口
type: project
---

> ⚠⚠ **本条曾断言「32/64 是非法枚举」，该断言已于 2026-10-05 撤回。**
> 依据的 `nvEncodeAPI.h` 后来被判定为 **FFmpeg 按需裁剪的子集**（`VBR_HQ|QVBR`
> 命中 **0** 次，RC 枚举恰为 FFmpeg `nvenc.c` 用到的 3 个）。**在GPU caps 裁决前，
> 不得再引用「枚举里没有 32/64」这一说法。** 完整记录见 Plan §0.2（含撤回说明、
> 裁剪证据、判读表、教训）。

**权威原文**：`Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` **§0.2**
（初版断言 + 撤回 + 四路证据权重 + 判读表）；推进顺序见 **§0.1.5**。

## 仍成立的部分（不依赖外部枚举）

**B2 根因链**（本仓代码内部逻辑，已逐行核实）：

| 位置 | 事实 |
|---|---|
| `nvenc_sdk.py:1005/1025/1056` | 只对 `vbr_hq`/`qvbr` 写非 0 值；`vbr`/`cbr` 落 `:1044` else 兜底（写 0 = CONSTQP） |
| `nvenc_sdk.py:1069` | LA 门控 `in ('vbr_hq','qvbr')` ⇒ `vbr`/`cbr` 的 LA **不使能** |
| `nvenc_sdk.py:634` | 只在 `rate_mode == 'constqp'` 时清 `la_depth` ⇒ `vbr`/`cbr` 下Python 侧 `_la_depth` 仍为 8 |

⇒ **Python 意图与代码实际写入不一致**，与枚举是否合法无关。
　 已落地 `[FIX-LOG-ECHO-LIE]`：Ready 行按生效值回显为
　 `la=8->0(未使能:CONSTQP) [未实现的rate_mode='vbr'→实际CONSTQP]`（纯观测，不改语义）。

## 已确立的 runtime 事实（权重最高）

**T4 实机GPU 直通 ctypes 实测**：`rc_ptr[1]=32` 被驱动接受，且
vbr_hq/constqp/qvbr **三者输出互异**（非静默钳制）。
⚠这只证明「32 与 0/64 行为不同」，**不证明「32 的语义 == VBR_HQ」**——
「驱动接受」与「语义相同」是两个命题。

## 仓库内两处记录互相矛盾（无可靠单一真源）

| 来源 | 声称 |
|---|---|
| `memory/nvenc_ctypes_verified_layouts.md:78` | `32=VBR_HQ, 64=QVBR` |
| `Accessory/probe/nvenc_vbr_hq_offsets_probe.py:17,532` | 注释写 **`4=VBR_HQ, 32=QVBR`** |

⇒ 这种矛盾本身就要求**由驱动 runtime 裁决**，而非任选一记录。

## 裁决入口（GPU 上一条命令）

```bash
python3 Accessory/probe/nvenc_rc_enum_truth.py --caps-only --verbose < /dev/null
```

用**官方 caps API** `NvEncGetEncodeCaps(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES=1)`
问驱动支持哪些 RC 模式（返回位掩码）。
⚠ 该caps 序号**易错**：实测解析权威头文件的 60 项 `NV_ENC_CAPS` 枚举得
`SUPPORTED_RATECONTROL_MODES = 1`（紧跟 `NUM_MAX_BFRAMES` 之后）；
早期版本手抄为 8 会**查到错误字段并得到「不支持 32/64」的假位掩码**，
从而闭环「证实」被撤回的静态推断。探针现已在运行时自检该序号。

## 若caps 确认 32/64 不被支持

官方文档述「Target quality: set rateControlMode to **VBR** with the desired targetQuality」
⇒ 修法为 `vbr_hq` 分支改写 `rc_ptr[1]=1`（`NV_ENC_PARAMS_RC_VBR`）+ 保留 `targetQuality`
+ `averageBitRate`，并**重跑全套等质量标定**。
⚠ 属「会改变 GPU 运行时语义」的改动 ⇒ 按 `feedback_no_gpu_work_mode.md`
**只给方案不落地**。

## 教训

见 `feedback-verify-external-source-completeness.md`（引用外部权威源前先验证它是否完整）
与 `feedback-instrument-must-render-verdict.md`（交付裁决型探针前必须自测它真能产出裁定）。
