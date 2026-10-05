---
name: 引用外部权威源前先验证它「是否完整」——裁剪版头文件不能当全集
description: 2026-10-05 事故：我据FFmpeg 的 nvEncodeAPI.h 断言 NVENC rateControlMode 32/64「枚举中不存在」，用户以T4 实机验证驳回；该头文件是按 FFmpeg 需求裁剪的子集（VBR_HQ/QVBR 命中 0 次）。含裁剪判据、仓库内互相矛盾的两个记录、以及「我查的文件里没有」≠「不存在」的命题区分
type: feedback
---

**规则**：把外部文件（头文件/规范/文档）当「权威全集」引用前，**先验证它确实是全集**。
「我查到的这份文件里没有 X」只能证明**该文件**没有 X，不能推出「X 不存在」。

**Why**：2026-10-05 我为 NVENC `rc_ptr[1]` 枚举值下结论，curl 取
`FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h`（含 `NVENCAPI_MAJOR_VERSION 13`、
NVIDIA 版权头，看着很权威），它写`NV_ENC_PARAMS_RC_MODE` 只有 CONSTQP=0/VBR=1/CBR=2
⇒ 我据此断言生产代码写入的 `vbr_hq=32` / `qvbr=64` 是**非法枚举**，并写成
"权威依据"。**用户以「生产代码在 T4 真实环境 GPU 直通 ctypes 编码 vbr_hq/qvbr 可是经过验证的，
你确认 32/64 不存在？」驳回** —— 而仓库里T4 实测早就记着「`rc_ptr[1]=32` 被驱动接受、
vbr_hq/constqp/qvbr 三者输出互异」。**我一边写「T4 实测不能证伪」，一边又拿裁剪表下判决，自相矛盾。**

**裁剪判据（可复用）**：怀疑裁剪时看「**该源的使用者需要什么**」与「**该源包含什么**」
是否恰好一致。实测该头文件 `grep -cE "VBR_HQ|QVBR"` = **0**，而 FFmpeg `nvenc.c` 实际只用到
CONSTQP/VBR/CBR 三种 ⇒ 头文件**恰含使用者要用的那几个** = 按需求裁剪的子集。
旁证：缺 `NV_ENC_PIC_PARAMS_V2` / `NvEncGetEncodeVersion` / `NVENCAPI_SUBMINOR_VERSION`，
且**完全没有函数指针表定义块**（`grep nvEncOpenEncodeSession` = 0 命中）。

**How to apply**：

1. 引用前跑两条：`grep -c` 查该源**的完整版必含项**（如"完整 SDK 必含 VBR_HQ/LOSSLESS"）
   + 查**结构完整性**（版本宏齐全、函数表/结构体定义块存在）。
2. 完整版拿不到时（多个镜像同缺 ⇒ 大概率本就裁剪），**降级为「未知」并交给runtime 裁决**，
   典型手段是官方caps API（位掩码）或厂商API 直接问驱动。
3. **仓库内多个来源互相矛盾时，这本身是信号** —— 说明该值缺乏单一真源，应去runtime 定夺，
   而不是挑一个来源当权威。本例矛盾：`nvenc_ctypes_verified_layouts.md:78` 写 32/64，
   而 `Accessory/probe/nvenc_vbr_hq_offsets_probe.py:17,532` 注释写 **4/32**。
4. 与既有教训同族（见 [[feedback_static_review_falsifiable]]「静态审阅的『确认』只是假设」）：
   外部权威源是「假确认」的高发区——**权威长相 ≠ 权威内容**。
5. 撤回要彻底：改完后不仅删结论，还要清理**下游预判**（本例探针把 32/64 硬编码为
   "❌非法"并据此 `return 3` ⇒ 改成「只报告 runtime 观测」，并注入 4 种 caps 掩码
   实测措辞随之变化、退出码不再预判）。

**顺带一条**：判断「驱动接受了某值」与「该值语义如你所写」是**两个不同命题**；
前者是runtime 事实（权重高），后者要靠 quality A/B 或 caps 语义确认。
