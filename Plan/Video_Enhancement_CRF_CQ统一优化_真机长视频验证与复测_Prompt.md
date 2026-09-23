# Video_Enhancement · CRF/CQ 统一优化 —— 真机长视频验证 + 修复 + libvmaf 复测（主提示词）

> 用途：把「**真机长视频基线验证 → 执行修复 → 生产补齐 libvmaf → 复测**」串成一条可复现链路。
> 适用环境：Linux + NVIDIA GPU（NVENC 可用）+ ffmpeg/ffprobe（建议 `--enable-libvmaf`）。
> 关联脚本：`tests/verify_crf_cq_unification.py`

---

## 一、参考产物（`verification_report/`）

| 报告 | 阶段 | 命令要点 | 关键结果 | 总判定 |
|---|---|---|---|---|
| `CRF_CQ统一验证报告_20260911_041308.md` | 修复前基线（真实长视频） | `--gpu --source …/WordWorld_S2/2-01.….avi --bitrate-source …/Prehistoric Planet…/S02E03.mp4` | EMIT 3/6（ESRGAN 恒 SKIP）、G7 4PASS/1SKIP、**G9/SCOPE 5 项全 SKIP**、无 libvmaf | PASS=73 FAIL=0 SKIP=10 |
| `CRF_CQ统一验证报告_20260911_042457.md` | 修复后（合成素材） | `--gpu` | EMIT 6/6、PIPE 8/8、G7-3 WARN、libvmaf 可用 | PASS=85 WARN=1 SKIP=0 |
| `CRF_CQ统一验证报告_20260911_043423.md` | 修复后（真实 `word_world_2`） | `--gpu --source …/word_world_2.mp4 --bitrate-source …/new5_raw.mp4` | G7-1/G7-2 WARN（真实内容欠配 ~2dB）、G7-3 PASS | PASS=84 WARN=2 SKIP=0 |
| `CRF_CQ统一验证报告_20260911_082057.md` | 修复后（真实长视频） | `--gpu --source …/WordWorld_S2/2-01.….avi --bitrate-source …/Operation_Ouch…/S09E02.mp4` | G7-1..4 PASS，**G7 组执行中断 FAIL：`TypeError: float - NoneType`** | PASS=85 FAIL=1 |

> 注：042457/043423/082057 生成于「G9 被并入 G10」的中间版本，故总览里没有 `SCOPE` 组；
> 当前脚本已恢复 **G9（SCOPE，原 5 项闭环核对）** 并新增 `G10-9`（旧版辅助 API 统一），复测时应能看到 `SCOPE` 与 `PIPE` 两个组。

---

## 二、投喂提示词（可整段复制给执行方）

### 阶段 0 ｜基线：用真实长视频 + GPU 验证测试脚本

```text
在 Linux + NVIDIA GPU 机器上，用真实长视频跑 CRF/CQ 统一优化的验证脚本，产出一份基线报告：

python tests/verify_crf_cq_unification.py --gpu \
    --source "/workspace/input_videos/WordWorld_S2/2-01. My Fuzzy Valentine - Love Bug.avi" \
    --bitrate-source "/workspace/input_videos/Prehistoric Planet S021080p/S02E03.mp4"

要求：
1. 确认 NVENC 真机可用（G0-5 PASS）；ffmpeg/ffprobe 在 PATH。
2. 报告落 verification_report/；记录真实素材的分辨率/帧数/时长（本基线为 640x360@24、38932 帧、约 1624s）。
3. 结果基线特征（供后续对照）：EMIT 仅 IFRNet 3 项 PASS、ESRGAN 3 项 SKIP；
   G7 组 4 PASS / 1 SKIP；G9/SCOPE 5 项全 SKIP；ffmpeg 无 libvmaf。
```

### 阶段 1 ｜执行修复（原提示词，逐字）

```text
根据3份报告暴露出的所有FAIL/WARN/SKIP，对照相关脚本代码，分析问题根源并制定修复方案，
修复生产和测试脚本代码中的所有问题。
```

补充约束（与本次实现一致，可按需增删）：

```text
- 全量闭环「环节①分割/归一化」与「环节③最终合并」，但**默认合并必须保持 -c:v copy（无损且快）**，
  仅在显式 --output-codec/质量/preset 时才重编码。
- G7 画质判据改为「有符号 ΔPSNR 单向容差 + 码率保真度」，并在 quality_map 增加 per-codec
  CQ 偏移可调口（默认 0，仍严格等于文档 §4.3）。
- 修复测试夹具 `_FakePopen`（存活期 poll() 必须返回 None），使 ESRGAN 的 G6-4/5/6 由 SKIP 转 PASS。
- 旧版辅助 API（add_audio_to_video / merge_videos / encode_video）一并统一走 resolve_quality。
- 保留 G9 编号（原「范围外/遗留项」→ 闭环核对），另设 G10「环节①/③ 契约」。
- 修复任何在真机复测中暴露的新问题（含测试脚本自身缺陷）。
```

### 阶段 2 ｜生产环境补齐 libvmaf 滤镜能力

```text
给生产环境的 ffmpeg 补齐 libvmaf 滤镜能力，使画质检查能采集 VMAF：
1. 确认/安装 libvmaf（如 ffmpeg 经 conda 安装：conda install -c conda-forge libvmaf；或自编译加 --enable-libvmaf）。
2. 校验：ffmpeg -hide_banner -filters | grep libvmaf
3. 复测报告中 G0-7 应为 PASS，G7-5「VMAF 对齐检查」应给出数值（不再 SKIP）。
```

### 阶段 3 ｜复测：产出 3 份新报告

```text
# 3.1 合成素材（快速回归）
python tests/verify_crf_cq_unification.py --gpu

# 3.2 真实素材（生产片段）
python tests/verify_crf_cq_unification.py --gpu \
    --source /workspace/input_videos/word_world_2.mp4 \
    --bitrate-source /workspace/input_videos/new5_raw.mp4

# 3.3 真实长视频（压力回归；注意 VMAF 长视频截断）
python tests/verify_crf_cq_unification.py --gpu \
    --source "/workspace/input_videos/WordWorld_S2/2-01. My Fuzzy Valentine - Love Bug.avi" \
    --bitrate-source "/workspace/input_videos/Operation_Ouch/Operation_Ouch_S09E02.mp4"
```

---

## 三、验收判据

| 项 | 期望 |
|---|---|
| 总判定 | **FAIL=0**；WARN 允许（真实素材 G7-1/G7-2 欠配 ~2dB 属已知内容相关误差） |
| EMIT | G6-1..G6-6 全部 PASS（ESRGAN 侧由 SKIP → PASS） |
| QUALITY | G7-1/G7-2 = PASS 或 WARN（不得 FAIL）；G7-3 = PASS；G7-4 = PASS；G7-5 = PASS 或 WARN（不得 SKIP，若已补 libvmaf） |
| SCOPE（G9） | G9-1..G9-5 全部 PASS（原 5 项遗留已闭环） |
| PIPE（G10） | G10-1..G10-9 全部 PASS |
| BITRATE | G8-1..G8-5 全部 PASS |
| CLI/STATIC/TABLE/RESOLVE/CONSTQP | 全 PASS |

---

## 四、已知坑（复测前必读）

1. **长视频 VMAF 会超时** → `ctx.vmaf(..., n)` 在数万帧上超出单命令 timeout，返回 `None`。
   已修：`_measure_many` 对 VMAF 截断到前 `VMAF_MAX_FRAMES=300` 帧；G7-5 逐项判空，
   任一编码器 VMAF 为 `None` 时记 `n/a` 且不参与 `worst`（否则 `float - NoneType` 会让整个 G7 组中断——即 082057 的 FAIL）。
2. **ESRGAN 命令捕获**依赖 `_FakePopen` 存活语义；若其 `poll()` 返回值被改回非 `None`，G6-4/5/6 会再次 SKIP。
3. **真实内容 G7-1/G7-2 可能 WARN**：固定线性映射（h264 +5 / hevc +7.5）存在内容相关误差，
   合成素材偏过配、真实素材偏欠配。需要时用 `quality_map.CQ_OFFSET` 按内容微调，**默认保持 0**。
4. **默认合并是 `-c:v copy`**：若在报告里看不到合并的重编码质量参数，属正常；只有显式
   `--output-codec/--output-crf[-ref]/--output-cq[-ref]` 才会重编码。
5. **报告对照**：旧报告无 `SCOPE` 组；当前脚本恢复后，复测报告应同时含 `SCOPE`（G9）与 `PIPE`（G10）。
