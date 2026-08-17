# verify_segment_bitstream_v3.py mpeg4 VOP 深度分析实机验证结论

2026-08-11 Windows 本地实机验证（`tests/verify_segment_bitstream_v3.py`，v5 多编码扩展）。

## 核心修正：VOP header 真实位序（与 ISO 标准书面顺序不同）

ffmpeg mpeg4 编码器（`mpeg4videoenc.c` 的 `mpeg4_encode_picture_header`）实际写入的
VOP header 位序是：

```
vop_coding_type(2 位: 0=I,1=P,2=B,3=S)     ← 最前面！
→ modulo_time_base(连续读 1 计数，读到 0 结束并消费)
→ marker_bit(1)
→ vop_time_increment(vop_time_increment_bits 位)
→ marker_bit(1)
→ vop_coded(1)
```

**关键教训**：ISO 14496-2 书面顺序是 `modulo_time_base → vop_time_increment → marker → vop_coding_type`，
按此实现会系统性错位——把 vop_coding_type 读到 vop_coding_type 之后的位（实测 ffmpeg 生成
样本解析出规律性 `I P B S` 循环 = ctype 读到 time 低 2 位）。以实际编码器为准，实测 ffmpeg
生成样本 + 真实 XVID 文件（S12E16 14924 帧）均按此顺序 100% 吻合。

`vop_time_increment_bits = ceil(log2(vop_time_increment_resolution))`，VOL header 中
resolution(16 位) 解析顺序: random_accessible_vol(1)+voti(8)+is_object_layer_identifier(1)
[+verid(4)+priority(3)]+aspect_ratio_info(4)[+par 8+8]+vol_control_parameters(1)[+chroma_format(2)
+low_delay(1)+vbv(1)→[bit_rate 15×2+vbv_buf 15×2+vbv_occ 11×2, 各跟 marker]]+shape(2)+marker(1)
+resolution(16)+marker(1)+fixed_vop_rate(1)。

## 连 I-VOP 判定（max_consec_i_vop > 3）

**不能沿用 H.264 的「段首窗口内新 IDR」判据**——mpeg4 正常 GOP 周期性出 I 帧（如每 12 帧
一个），段首 I 后 32 VOP 窗口内必然有新 I 帧。异常特征 = **最长连续 I-VOP 块长度**（相邻
索引差==1 的连续 I 序列）。判定阈值 `max_consec_i_vop > 3`：段首 1-2 个 I 预热帧是 XVID
正常行为（S12E16 VOP#1/#2 双 I），大量连续 I 才是修复工具重编码特征。

## 伪 start code 过滤

MPEG-4 Part 2 无 emulation prevention（不像 H.264 有 0x000003 转义），异常/损坏文件压缩
数据内部可能含伪 `0x000001B6`。过滤仅依赖 VOP header 解析成功（try/except 跳过）。
**前导字节非 0x00 不可作过滤依据**——GOV/user_data payload 可能以 0x00 结尾，紧邻真实
VOP 前导字节恰为 0x00，会误杀；4 字节前缀 0x00000001 由 sc_len 判断处理。伪 start code
导致 VOP 总数 > frames/packets 的差异本身就是验收告警（展示层交叉核对）。

## start code 常量陷阱

解析扫描提取的是**单字节 code**（紧跟 00 00 01 前缀后），常量必须用单字节值：
`_VOP_SC=0xB6`（不是 0x1B6）、`_VOL_SC_MIN/MAX=0x20/0x2F`（不是 0x120/0x12F）。
用完整 4 字节 start code 值比较单字节 code 永远为 False → 分支永不进入。

## 样本实测数据

| 样本 | 结果 |
|------|------|
| ffmpeg 生成含 B 帧 mpeg4 (testsrc 50帧) | VOP=50==frames==packets，类型 `I P B B P B B P B B I...` 与 ffprobe 完全一致，PASS |
| ffmpeg 生成无 B 帧 mpeg4 | `I P P P P...`，PASS |
| S12E16（干净 XVID, 14924 帧） | VOP=14924==frames==packets，I=100，连I块=2（段首预热双 I），回退=0，PASS |
| **01 the race...fixed.avi（用户目标）** | **VOP=22114==frames，packets=22126 多 12 → 12 个垃圾 packet 是容器假象；I-VOP=160 正常、连I块=2、时间回退=0 → 码流健康。验收 FAIL 仅因 frames!=packets 严格判定** |
| 112 Max Bed Time.avi（伪 start code） | VOP=19825 > frames=14799（伪码高估 5026），连I块=1、回退=0 不误报，frames==packets PASS |

## fixed.avi 旧结论修正（重要）

memory 60656380 曾记录「fixed.avi 有 21231 个 I-VOP、21032 连 I-VOP、vop_time regress=1007，
码流被大量重编码为 I 帧」——**该结论基于旧错误 VOP 位序解析（vop_coding_type 读错位置）**。
修正后实测：fixed.avi 码流健康（I-VOP=160 正常分布），唯一异常是容器 12 个垃圾 packet
（frames 22114 vs packets 22126），用户可通过 VOP 总数交叉核对确认。

## 修复文件

`tests/verify_segment_bitstream_v3.py`（v5，基于 v2 的独立扩展）：
- `probe_video_codec()`: ffprobe 读 v:0 codec_name
- `extract_annexb_es(path, codec)`: 按编码分支提取（h264→h264_mp4toannexb / mpeg4 家族→-f m4v / hevc→hevc_mp4toannexb）
- `parse_mpeg4_es()`: VOL resolution + VOP header
- `check_vop_stats()`: I-VOP 计数 / max_consec_i_vop / vop_time_regress
- `_verify_one_video()`: codec 分支编排，非 h264 的 frames!=packets 保持严格判定

## v2 多编码容错（2026-08-11 追加）

`tests/verify_segment_bitstream_v2.py` 也获得多编码容错（与 v3 共用设计）：
- **非 H.264/HEVC（mpeg4/DivX/Xvid/vp9/av1 等）→ 简单提示不判失败**：`[2] 提示: 非
  H.264/HEVC（codec=mpeg4），跳过 NAL 专项检查`，不再因 h264_mp4toannexb bsf 不支持
  而抛出误导性报错崩溃。frames/packets 与 pts 检查照常执行。
- **HEVC 基础 NAL 分析**（v2/v3 一致，新增 `parse_hevc_es` + `check_hevc_stats`）：
  HEVC NAL header 2 字节，`nal_unit_type = (payload[0] >> 1) & 0x3F`；IDR type 19/20
  （IDR_W_RADL / IDR_N_LP）。统计 IDR 帧/连 IDR（多 slice 按 NAL 索引相邻聚类）。
  **HEVC 无 H.264 的 frame_num 概念（用 POC）**，frame_num_regress 恒 0。
- v3 的 hevc 分支从 N/A 升级为同样的基础 NAL 分析 → **v2/v3 对 h264/hevc 输出完全一致**
  （实测 hevc 样本均：`[2] IDR=1 首个IDR@4 其后32NAL内IDR=0 frame_num回退=0` PASS）。
- hevc 提取依赖 `hevc_mp4toannexb` bsf 存在（`_bsf_available` 全局缓存 ffmpeg -bsfs）；
  缺失时 hevc 显示 N/A 不判失败。
- v2 单文件/多文件输出均加 `[codec]` 标识与 `nal_note` 提示行。

v2/v3 实测样本矩阵（2026-08-11）：
| 样本 | v2 | v3 |
|------|----|----|
| h264 (libx264) | IDR=1/连IDR=0/FN回退=0 PASS | 同 v2 PASS |
| hevc (libx265, 50帧) | IDR=1/首个IDR@4/连IDR=0 PASS | 同 v2 PASS |
| mpeg4 含B帧 | 提示跳过 NAL，PASS | VOP=50/I=5/连I块=1/回退=0 PASS |
| fixed.avi（用户原始报错） | codec=mpeg4 + 提示 + frames!=packets FAIL（不再崩溃） | VOP 深度分析 + 交叉核对 FAIL |
