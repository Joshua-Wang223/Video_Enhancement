# IFRNet 水彩花屏 + HEVC 尾帧损坏双症状调查（2026-08-26）

## 背景与结论速览

三个方案会话（Plan/session-ses_fd31.md、fd31_updated.md、fcd1.md，2026-08-24 落地的
P0×8+P1+P2+P3+P2.4c+P3.1）执行后，用户发现插帧输出"水彩画色彩润开"花屏。深度调查
（含备份 benchmark 对照、码流解剖、逐帧二分）结论：

- **症状 A（尾帧参考链断裂）= 既有缺陷**：每个 HEVC+LA 分段末尾 5~8 帧不可解码
  （"Could not find ref with POC X" + "Error constructing the frame RPS" +
  "Skipping invalid undecodable NALU: 1"）。备份 bak20260824 的同配置 benchmark
  （v6.4.5.1 hevc vbr_hq la8：1373 包 1368 可解；la16：1373/1360）同样存在 →
  三个方案未引入也未修复。容器帧数守恒（packets 正确）但尾部 AU 解码失败，
  **包级守恒校验（frames==packets）对此盲区**。
- **症状 B（中段水彩单帧）= 方案期间显性化的既有竞态**：孤立单帧、解码干净、
  双场景平滑神经网络混叠（放大图证实无横向撕裂带）。备份同配置文件色度坏帧簇=0，
  当前文件有簇 → 新出现。机制 = H2D 预取 pinned 槽无同步（见下）。

## 症状 A 解剖（决定性证据）

对 word_world seg002（285 包/277 可解）做 Annex-B 解剖：

- 流结构 = `[头部参数集×2 + IDR + 276 TRAIL]` + `[VPS/SPS/PPS + CRA + 23 TRAIL]`。
- **尾部存在"参数集重启 + CRA 组"**（= 新 GOP），组内容 = 本段自己的尾帧
  （frame_0262 ≈ 源帧 131 逐内容匹配）；主输出流在 frame 260 处终止。
- CRA 组单独解码（含自带 VPS/SPS/PPS）：16 帧成功 + **最后 8 帧参考链在编码时已断裂**
  （组内相对 POC 20-27 缺失）→ 不是 muxer/封装问题，是编码产出即坏。
- 组大小不固定：ww2 seg002=24 帧、S01E12 seg001≈4、备份 la8≈17、la16≈17 →
  与触发时机相关，与 chunk(128)/batch(24) 边界均不对齐。
- 备份 la16 文件 IDR=23（多 GOP 重启）；pts drop @frame 1357 (est_missing=10)。
- 逐帧 drain 全程 code=8（INVALID_PARAM）、帧全靠段末 EOS 排空是**既有隐性模式**
  （追加 V 定性），[P0-FIX-RC-TOLERANT] 仅加了遥测。
- CRA 触发点未最终定位（需运行日志）。主嫌疑：`_ensure_slot_free` 的"排空超限→
  空帧占位兜底"路径放弃槽位后继续提交，驱动 LA 链断裂自行重启 GOP（新 IDR/CRA），
  之后提交的帧落入重启组；组内尾帧再次踩同一 EOS 排空缺陷。**验证方法：grep 运行日志
  中"排空超限"/"空帧占位兜底"/"帧数守恒校验失败"/"二次排空回收"出现位置是否与重启点吻合**。

## 症状 B 机制（代码级确认）

`external/ifrnet_video/pipeline.py _try_prefetch_next`（L1135-1197）+
`external/ifrnet_video/ifrnet_utils.py PinnedBufferPool.get_for_frames`（L280-296）：

- 预取用 4 槽 pinned 池轮转（group0=slot0,1 / group1=slot2,3），
  `.to(device, non_blocking=True)` 异步 H2D 在 stream_h2d。
- **`get_for_frames` 的 `dst.copy_(src_f)` 无任何槽位事件/等待** —— CPU 覆写
  group-N pinned 时不等该槽在途 H2D 完成。
- `_infer_batch` 预取分支有 `stream_compute.wait_stream(stream_h2d)`（main.py L724-731），
  只保证 H2D→计算有序，不保护 pinned 宿主缓冲。
- 稳态安全纯靠时序巧合：一轮推理 238ms ≫ H2D(~40ms)+CPU 拷贝。批变小
  （OOM 降批/段尾小批，B-11：降批跨段不复原）或流水线停顿后恢复时迭代时间坍缩，
  竞态窗口打开 → GPU 张量=两对不同时间帧的混合 → IFRNet 忠实混合 → 水彩单帧。
- 此代码三个方案**未改动**（备份即如此），但方案的时序类改动（P2.3 池、P2.4a/d、
  排空遥测打印、muxer 写线程）改变停顿分布使其显性化。
- word_world 的部分水彩（frame_0024 已证实）来自 **ESRGAN 放大分段**（上游已坏，
  IFRNet 忠实继承）；S01E12 的 5 帧（15574/16630/17226/17346/17462）经二分验证
  输入干净 → IFRNet 阶段产生。两后端镜像同构，ESRGAN 侧同类风险见
  [[esrgan-pinned-buffer-pool-race]]。

## 其他发现（潜在雷区）

1. **[P0-FIX-HEVC-HANG] 与 GPU 实测结论冲突**：`_lock_bitstream_blocking` 对 HEVC/AV1
   改 doNotWait=1 轮询，但 diagnose_hevc_la test3/test10 实测 HEVC doNotWait=1
   会永久挂起或返回 SUCCESS+垃圾 size（7.2MB/21MB）。该函数被 `_ensure_slot_free`
   兜底（nvenc_sdk L1553）与 ce-pipeline BLKRETRY 调用。建议回退为 blocking+外部
   watchdog，或至少对返回 size 做上限钳制（≤W*H*4）。
2. **[P2.3-LA-PINNED-REUSE] 池深公式漏算累积**：池深=_q.maxsize+2，未计编码线程
   `_acc_nv12` chunk 累积（最多 chunk_frames/batch 批）。当前生产节奏（T2 238ms/批
   为瓶颈）在途距离 1-2 批打不到，属潜伏缺陷；编码快于推理的场景必炸。
3. **HEVC 参数集识别可疑**：日志 "Cached SPS+PPS: 33 bytes" vs 真实 VPS+SPS+PPS=78B；
   每段头部出现两份参数集块 → `_extract_sps_pps`/`_has_sps_pps`/`_es_has_param_sets`
   的 HEVC 分支需审计（P3.2 nal_utils 已有参考实现，可切换）。
4. verify v4 的检查1 是包级守恒，对"包在但解不出"盲；检查4（色度坏帧簇）能抓水彩帧。

## 修复方案（按优先级）

### Fix-1（症状 B，小改，先做）：预取槽事件同步
pipeline.py `_try_prefetch_next`：`.to()` 后在 stream_h2d 上 `record_event` 记录
每槽最近事件（存 pool 或 runner）；`get_for_frames` 覆写前（或 `_try_prefetch_next`
拿到 dst 后、copy_ 前）`event.synchronize()`。同步路径 frames_to_tensor 同享。
成本：每批一次 event wait（正常已超时，零开销）。

### Fix-2（症状 A 定位）：日志比对 + GPU 诊断脚本扩展
先 grep 生产日志定位重启点与"排空超限"计数是否重合；再扩展
tests/diagnose_hevc_la.py 增加"槽位放弃后继续提交"变体，复现 CRA 重启组，
确认后修复 `_ensure_slot_free` 兜底语义（放弃槽位必须连带终止本段而非继续提交）。

### Fix-3（症状 A 短期规避）：生产切 constqp+LA=0（ce_pipeline，历史验证最充分）
或 h264+vbr_hq+LA=8（h264 排空语义已充分验证）。HEVC+LA 修复前勿用于成片。

### Fix-4（验收门禁升级，防再漏）
段级验收加**解码级守恒**（ffmpeg 解码帧数==期望帧数，替代/并列 packets 计数）+
色度坏帧簇检查（verify v4 检查4 纳入段级判败）；Tier 占位帧（prev 填充）计数
超阈值判败。verify v4 已具备全部能力，只需接线。

### Fix-5：回退 [P0-FIX-HEVC-HANG] 的 doNotWait=1（见上"其他发现 1"）；
[P2.3-LA-PINNED-REUSE] 池深改为 qd+ceil(chunk_frames/batch)+2 或在途计数。

## 关键文件与行号

- pipeline.py L1135-1197（预取）、ifrnet_utils.py L266-296（池）
- nvenc_sdk.py L1530-1580（_ensure_slot_free 兜底）、L1582-1646（doNotWait=1）、
  L2040-2162（EOS 排空）、L2041-2060（EOS fail-fast）
- main.py L3074-3160（编码线程 f0/分块）、pipeline.py L1370（编码线程创建）
- 验证工具：tests/verify_segment_bitstream_v4.py（检查1 包级/检查4 色度簇）
- 解剖脚本：临时 parse_tail.py（Annex-B NAL 分组，可按需再生成）

## 证据索引

- 当前产物：temp/ifrnet_from_segments_{word_world_2,S01E12_Surprise}/processed/
- 备份对照：Video_Enhancement.bak20260824/temp/benchmark_ww2/（hevc vbr_hq la8/la16）
- 用户更正：word_world seg003（h264 21帧）属另一次测试；两次测试均 LA=8；
  temp/word_world_2.mp4 是本次合并输出，原视频在 input_videos/。
