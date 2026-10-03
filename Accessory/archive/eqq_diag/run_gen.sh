#!/bin/bash
# 庚方案：补 6s 口径的 5 条 VE 独有素材。
# 每素材独立 workdir + 独立日志 ⇒ 5 路并行安全（同workdir 并行会互删 prep.mp4）。
set -u
R=/mnt/d/Workspace_Python/Video_Enhancement
IV=/mnt/d/Workspace_Python/input_videos
BC=/tmp/eqq_native_srcs
mkdir -p /tmp/eqq_uni_6s

# 素材清单：VE 独有、且 ≥6.2s（可做 6s 口径）
JOBS=(
  "new1|$IV/new1.mp4"
  "word_world_2|$IV/word_world_2.mp4"
  "bbc_s01e01|$BC/S01E01._The_New_Stall.mp4"
  "bbc_s03e01|$BC/S03E01._James_and_Alice.mp4"
  "bbc_s05e01|$BC/S05E01._Forever_Friends.mp4"
)

TIERS=libx265,libvpx-vp9,libaom-av1,libsvtav1,librav1e,librav1e@10

for j in "${JOBS[@]}"; do
  tag="${j%%|*}"; src="${j#*|}"
  if [ ! -f "$src" ]; then
    echo "[skip] $tag 源缺失: $src" | tee -a /tmp/eqq_uni_6s/_launcher.log
    continue
  fi
  out="/tmp/eqq_uni_6s/$tag"
  mkdir -p "$out"
  nohup python3 /tmp/measure_uni.py --src "$src" --out "$out" --tiers "$TIERS" \
      > "/tmp/eqq_uni_6s/$tag.log" 2>&1 &
  echo "[start] $tag pid=$! src=$(basename "$src")"
  sleep 2   # 错开启动，避免同时抢磁盘（prep 写入）
done
echo "[launcher] 全部启动完毕 $(date '+%H:%M:%S')"
