---
name: 枚举/常量值要回权威源核对，但先验证「该源是否完整」——「实测被接受」也不等于「语义正确」
description: 2026-10-05 审计 NVENC rateControlMode 32/64 时，仓库三个信息源互相矛盾（代码注释/两个 memory/探针注释）；我据 FFmpeg 裁剪版头文件断言「枚举中不存在」被用户以 T4 实机验证驳回 ⇒ 教训是「核对常量」与「验证权威源本身完整」是两步，且「驱动接受了该值」与「该值含义如你所写」仍是两个不同命题
type: feedback
---

> ⚠ **2026-10-05 勘误**：本条原据 FFmpeg 裁剪版头文件断言 NVENC 32/64「枚举中不存在」，
> 该断言**已撤回**（详见 `nvenc-rc-enum-illegal-vbr-hq.md` 与 Plan §0.2）。
> 本条**仍然有效的部分**是下面「三个信息源互相矛盾」这一事实，
> 以及「驱动接受 ≠ 语义正确」这一区分。

写 ctypes / FFI 层代码时，**任何送往驱动的枚举、魔数、offset都必须回到权威头文件核对**，
不能凭注释、不能凭「实测没报错」。

**Why**：2026-10-05 审计 `nvenc_sdk.py` 的 `rc_ptr[1]`（rateControlMode）时发现三处**互相矛盾**的记录：

| 来源 | 声称 |
|---|---|
| 生产代码 `nvenc_sdk.py:1005/1025` | `vbr_hq`→32、`qvbr`→64 |
| `memory/nvenc_ctypes_verified_layouts.md`（两处） | `0=CONSTQP, 32=VBR_HQ, 64=QVBR` |
| `Accessory/probe/nvenc_vbr_hq_offsets_probe.py:17,532` | `CONSTQP(0)/VBR_HQ(4)/QVBR(32)` |
| **权威头文件**（curl 实取） | **只有 `CONSTQP=0, VBR=1, CBR=2`**；32/64 不存在 |

即仓库里关于同一个枚举有**三种互相冲突的说法**，而生产代码写的 32/64 两个都不是合法值。
若只读代码或只读 memory，会完全错过这个问题。

**关键区分（本条最重要的一点）**：
历史记录里有「T4 实测 driver 13.0 **接受** `rc_ptr[1]=32`，且 vbr_hq/constqp/qvbr 三者
输出互异（非静默钳制）」。这条实测**只能证明驱动没报错**，**不能证明 32 就等于 VBR_HQ**：

> **「驱动接受了该值」与「该值的语义如你所写」是两个不同命题。**

驱动对未定义值可能：①按最接近的合法值处理（行为随驱动版本漂移）；②按位掩码解释
（caps API 返回的正是位掩码，可能命中厂商私有扩展位）；③在未来版本变成错误。
**三者的共同点：都不能作为「枚举合法」的证据。**

**How to apply**：

1. **枚举/魔数一律回权威源核对**：NVIDIA 相关用
   `curl -sL https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h`
   （注意：`cuvid/` 与 `src/nvcuvid/` 路径都是 404，**正确路径是 `include/ffnvcodec/`**；
   用 GitHub API `git/trees/master?recursive=1` 列举可确认）。
   参见 `curl-header-fetch-method.md`。
2. **发现三处以上信息源冲突时，不要试图调和它们** —— 冲突本身就是「有东西错了」的信号，
   直接去找权威源。
3. **落地前先问「这个值的语义我是从哪来的」**：如果答案是「注释」「之前实测没报错」
   「文档里这么写」，都不够。
4. **对外部 API 的「实测通过」要分清证明了什么**：证明「不崩」≠ 证明「语义正确」；
   证明「输出互异」≠ 证明「输出是预期的那个模式产出的」。
5. 顺手更新掉产生错误结论的**下游记忆**（本例连带作废了 RC 性能排名的三臂对比），
   标注作废而不是删除。

**配套**：权威源与仓库注释冲突时的处置、以及「只给方案不落地」的工作约定，
见 `nvenc-rc-enum-illegal-vbr-hq.md` 与 `feedback_no_gpu_work_mode.md`。
