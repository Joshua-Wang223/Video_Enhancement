#!/bin/bash
# 己方案：补 10s 口径的 5 类素材（重新采集的 10s clip）。
# 与庚（/tmp/eqq_uni_6s）完全隔离：独立 workroot、独立日志、独立素材。
set -u
R=/mnt/d/Workspace_Python/Video_Enhancement
S=/tmp/eqq_uni_10s/src
mkdir -p /tmp/eqq_uni_10s

TAGS=(anim_10s anim_subs_10s dark_10s ui_10s texture_10s)
TIERS=libx265,libvpx-vp9,libaom-av1,libsvtav1,librav1e,librav1e@10

for t in "${TAGS[@]}"; do
  src="$S/$t.mp4"
  if [ ! -f "$src" ]; then
    echo "[skip] $t 源缺失: $src" | tee -a /tmp/eqq_uni_10s/_launcher.log
    continue
  fi
  out="/tmp/eqq_uni_10s/$t"
  mkdir -p "$out"
  nohup python3 /tmp/measure_uni.py --src "$src" --out "$out" --tiers "$TIERS" --duration 10 \
      > "/tmp/eqq_uni_10s/$t.log" 2>&1 &
  echo "[start] $t pid=$!"
  sleep 2
done
echo "[launcher] 己方案 5 路全部启动 $(date '+%H:%M:%S')"
