---
name: ffmpeg 画质度量的两个口径陷阱 —— 必须用显式 [0:v][1:v]psnr
description: 裸 `-lavfi psnr` 与显式 `-lavfi "[0:v][1:v]psnr"` 对同一对文件差 3 dB（43.40 vs 46.58）；`-v error` 会压掉 psnr 滤镜 INFO 级汇总行导致 ΔPSNR 恒为 0.00；含与判据脚本同口径的命令模板与「写入文档的数字必须同口径复现」
type: project
---

2026-09-28 在容器内实测（ffmpeg 8.x）。给「编码产物 vs 参考」打分（PSNR / SSIM / VMAF）时有两个
**静默**陷阱 —— 都会让流程看起来"跑通了"，但数字是错的：

**1. 必须用显式输入标签 `[0:v][1:v]`**

同一对文件（libx264 crf21 产物 vs 源）：

| 写法 | PSNR |
|---|---|
| `-lavfi psnr`（裸） | **43.40 dB** ❌ |
| `-lavfi "[0:v][1:v]psnr"`（显式） | **46.58 dB** ✅ |

差 **3 dB** —— 远超 `TOL_PSNR_DB=1.5` 这类容忍带，足以让 PASS 判成 FAIL 或反之。
判据脚本 `Accessory/verify/crf_cq_unification_verify.py` 的 `Ctx._metric` 用的就是**显式标签**形式；
手工复现或新写判据时必须照抄该形式。

**2. `-v error` 会压掉汇总行**

`psnr` 滤镜的整段汇总（`... average:46.58 ...`）打在 **INFO 级**。用 `-v error` 跑 ⇒ 抓不到 `average:`
⇒ 解析返回空串 ⇒ 下游 `awk`/算术把空当 0 ⇒ **ΔPSNR 恒为 0.00**（表现为"完全对齐"，实为没测到）。

**正确命令模板**（镜像 `Ctx._metric`）

```bash
N=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$REF")
psnr_of() {   # $1 = 待测文件；$REF = 参考（源）
  ffmpeg -hide_banner -v info -i "$1" -i "$REF" -frames:v "$N" \
         -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
    | grep -oP 'average:\s*\K[0-9.]+' | tail -1
}
```

替代写法 `psnr=stats_file=<path>` 与 loglevel 无关，但**同样必须配显式标签**才与判据同口径
（实测 stats_file + 裸写法一样给 43.40，是错的）。

**Why:** 2026-09-28 我按裸写法先量出 43.40，据此往方案文档里写了一个 ΔPSNR 值（`qp26 = −1.94 dB`）；
换成显式写法复现后真值是 **−3.07 dB**。只有命令对齐判据口径后，手工数字才能与已出报告里的
`G7-3 1.46× / +0.06 dB` **逐位对上**（`soft=46.580407`、`qp21=46.641246`）。
**How to apply:** 写/改任何"编码产物 vs 参考"的打分命令或判据时：(a) 用显式 `[0:v][1:v]`；
(b) 别用 `-v error`；(c) **往文档里写数字前先用同口径命令重跑一遍复现**，不要从不同口径的历史
输出里拼数字 —— 口径不同的两个数字凑在一起，比没有数字更危险。
