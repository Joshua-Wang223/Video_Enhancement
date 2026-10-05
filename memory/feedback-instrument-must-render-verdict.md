---
name: 交付「裁决型探针」前必须自测它真的能产出裁定 ——否则是带假 PASS 的哑弹
description: 2026-10-05 事故：我交付的 NVENC RC 枚举探针 Accessory/probe/nvenc_rc_enum_truth.py 有四处断裂（--caps-only 极性反了使该flag 恰好跳过查询、_FUNC_IDX 缺 GetEncodeCaps、_NvEncCapsParam 结构体根本不存在、_resolve_caps_index 定义但无调用点），且 --try-values 按名映射而非写真值⇒ 32/64 实走 constqp；计划里那条「30 秒裁决」命令 exit 0 却什么也没测。含哑弹探针的四类断点与自测清单
type: feedback
---

**规则**：交付一个用于**下判决**的探针/门禁前，必须在交付前**亲手跑一遍它的每条输出分支**，
确认它 (a) 真能取到目标数据、(b) 结论随观测变化、(c) 失败时不报PASS。
在**当前环境跑不了**（无 GPU）时，至少要用注入/mock 方式把分支走通。

**Why**：2026-10-05 我写`Accessory/probe/nvenc_rc_enum_truth.py` 去裁决
「NVENC `rc_ptr[1]=32/64` 是否合法」，并在 Plan §0.2.5 写下「一条命令，~30 秒出结论」。
用户随后要安排 GPU A/B，我才让workflow 去审，结果 **1 条 finding 以 3/3 全票存活**，
逐条复核确认**探针是哑弹**：

1. **极性反了**：`if not args.caps_only:`包住caps 查询 ⇒ 传了文档里说的 `--caps-only`
   反而**跳过**查询，exit 0 静默通过。
2. **依赖不存在**：`_FUNC_IDX` 里没有 `GetEncodeCaps`（守卫写`"_GetEncodeCaps"`带前导下划线，
   与 dict 键风格永不可能匹配）。
3. **import 了一个不存在的名字**：从 `nvenc_sdk` import `_NvEncCapsParam`，
   而 `external/` 下无此结构体 ⇒ 整个 try 块第一行 ImportError 被 `except` 吞成 `None`。
4. **占位常量当真值用**：`CAPS_SUPPORTED_RATECONTROL_MODES = 0` 是占位，
   而「解析它」的 `_resolve_caps_index()` **定义了却全文件无调用点** ⇒ 即便前三处修好，
   也会拿 `capsToQuery=0`（= `NUM_MAX_BFRAMES`）去查，**返回错误位掩码却不报错**，
   恰好「证实」我原先的静态推断 ⇒ 闭环假结论。
5. **兜底路径同样给假 PASS**：`--try-values` 按 `rm = {0:"constqp",1:"vbr_hq",2:"qvbr"}` **名字**
   映射，`--try-values 32/64` 反而落 `constqp`(写 0) ⇒ 要测的两个值一个都没写，
   exit 0 却标「全部合法」。

**How to apply**：

- **自测清单**（写完探针立刻过）：①每个 `--flag` 的语义与实现**极性**一致？
  ② 引用的每个符号/结构体/索引**当场grep 确认存在**？③ 注入 3~4 种输入，
  断言**输出措辞随观测变化**、退出码不预判？④ 走「取不到数据」的分支时，
  会不会**误报 PASS**？
- **「不可归属/测不到」必须与「零/无」用不同值**（本项目同型教训见
  [[memory-leak-attribution-measurement]]：显存测不到报 `None` 而非 `0.0`）。
- **告警若要发 FAIL/SKIP，先确认那条判据真的被采到数据**——否则它只会稳定产出假 PASS。
- 手抄常量（函数索引、枚举序号、offset）**必须交叉两处以上同源证据**再落值；
  本例 `GetEncodeCaps` 的 index 7 由 `nvenc_struct_dump.c:299` 与
  `nvenc_comprehensive_matrix.py:84` 两处互证，且与 `_FUNC_IDX` 现有 12 项共有条目零冲突。
- 交付时**明说环境前置**：本容器无 GPU 时退出码要能表达「结论未产出」，
  而不是让「没测」长得像「测过了」。

## 2026-10-05 修复中新增的陷阱：判据序号错会「闭环证实」你原有的错结论

比"占位常量"更危险的一类：**手抄的判据序号错了，但探针仍然成功返回**。
本例 `NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES` 我先手抄为 **8**；实测解析权威头文件的
**60 项** `NV_ENC_CAPS` 枚举，正确值是 **1**（紧跟 `NUM_MAX_BFRAMES=0` 之后）。
若只修前四处断裂就上机，`capsToQuery=8`会查到`NV_ENC_CAPS_SUPPORT_UNSUPPORTED`（或别的无关项），
返回一个**看起来合法、实则答非所问**的位掩码 ——若它恰好显示「不支持 32/64」，
就会**闭环「证实」我先前那个已被 T4 实测推翻的静态断言**。错误因此从"没测"升级为"反向制造伪证据"。

⇒ 补充纪律：
1. **判据类常量（枚举序号 / capsToQuery / API 索引）要比"数据类常量"更严**：
   数据常量错了→读出错值；判据常量错了→**返回一个与问题无关但看起来自洽的答案**。
2. 探针必须在运行时**自查判据并把自查结果打进输出**（本例：取不到头文件时显式打印
   「序号未经独立校验」，而不是静默用默认值）。**自检不可用时要说不可用，不能装作已核对。**
3. 判读表里要有一行专门覆盖"探针自身出错"的情形（本例判读表首行就是
   「caps 只含 bit0/1/2 ⇒ 32/64 为未定义值」——这一行的前提是 caps 查对了字段，
   故须与「caps 查询失败」行并列，二者不可合并）。
