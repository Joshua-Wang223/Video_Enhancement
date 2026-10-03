#!/usr/bin/env bash
# rav1e **native 档**补标：2 素材 → 5 素材（使 LOO 可验证）
#
# 动机（2026-10-01）：native 档此前只有 new5_raw + word_world_2 两条素材，
#   素材数 < 3 ⇒ LOO 无法验证该行表值。补 3 条实拍影视（BBC Molly and Mack，
#   1920x1080 / 30fps / h264，来自素材库 /mnt/f/English Enlightenment）。
#
# 口径与既有native 档保持一致：**720p / yuv420p / 10s / 锚点 18/22/26/30/34 /
#   n_subsample=1 / rav1e 不下发 -speed**。Stage 3 原workdir 是 1280x720_10s_n2，
#   本轮用 --resume 续跑同一 workdir ⇒ 新点只补缺失的，前 2 素材 20 点不重跑。
#
# 成本：native 单点 ≈ 13 min（实测R2 Stage 3）。新增 3 素材 × (10 扫描 + 5 锚点)
#   但锚点是共享的 libx264，每素材实际需10 个 native 点 + 5 个锚点 = 15 次编码
#   ⇒ 3×15 = 45 点 ≈ 10 小时。⚠ 若只需「能LOO」而非全精度，可先用 --quick 验通路。
set -u
PROJ=/mnt/d/Workspace_Python/Video_Enhancement
IV=/mnt/d/Workspace_Python/input_videos
NAT=/tmp/eqq_native_srcs
cd "$PROJ" || exit 1
H=Accessory/probe/calibrate_equal_quality.py
BASE="--duration 10 --resume --subsample 1 --workroot /tmp/eqq2 --tag 1280x720_10s_n2"

echo "########## [$(date '+%F %H:%M')] native 补标：+3 实拍影视（共 5 素材）##########"
python3 "$H" $BASE \
  --src $IV/new5_raw.mp4 \
  --src $IV/word_world_2.mp4 \
  --src "$NAT/S01E01._The_New_Stall.mp4" \
  --src "$NAT/S03E01._James_and_Alice.mp4" \
  --src "$NAT/S05E01._Forever_Friends.mp4" \
  --codecs librav1e --rav1e-speed native < /dev/null
rc=$?
echo "########## [$(date '+%F %H:%M')] native 补标结束 rc=$rc ##########"
exit $rc
