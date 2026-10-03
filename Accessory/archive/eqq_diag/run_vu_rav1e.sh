#!/usr/bin/env bash
# VU侧 M2 rav1e 标定（7 素材 × native/+speed10）—— 2026-10-01 20:xx 重启
#
# 起因：20:26 的尝试在首个点落盘前被中断（workdir m2_7src_6s_rav1e 只有
#       prep.mp4 + libx264_18.mp4，无 points.json；日志 0 字节 ⇒ 外部 kill）。
#       本脚本按同一workdir/口径重启，--resume 会跳过已完成点。
#
# 口径（与 VU 侧 m2_7src 一致）：720p / yuv420p / **6s** / 锚点 18/21/24/27/30 /
#   n_subsample=1 / 7 素材（覆盖立项 §3 全部 6 类内容）。
#⚠ A 仓 native 补标已 SIGSTOP 暂停（避免抢 CPU）；其数据在 /tmp/eqq2/1280x720_10s_n2。
set -u
cd /mnt/d/Workspace_Python/VidUtils || exit 1
H=probe/calibrate_equal_quality.py
IV=/mnt/d/Workspace_Python/input_videos
M=/mnt/d/Workspace_Python/VidUtils/temp/m2_srcs
WR=/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib

echo "########## [$(date '+%F %H:%M')] VU rav1e 标定启动（7 素材 × native + speed10）##########"
for SPEED in native 10; do
  TAG="m2_7src_6s_rav1e"
  [ "$SPEED" = "10" ] && TAG="m2_7src_6s_rav1e_s10"
  echo "---档位 librav1e --rav1e-speed $SPEED（tag=$TAG）---"
  python3 "$H" \
    --workroot "$WR" --tag "$TAG" \
    --src "$IV/new5_raw.mp4" \
    --src "$IV/new4_raw.mp4" \
    --src "$M/cc_anim_300s.mkv" \
    --src "$M/cc_subs_105s.mp4" \
    --src "$M/earth_dark_80s.mp4" \
    --src "$M/ui_screen_10s.mp4" \
    --src "$M/natgeo_grass_40s.mp4" \
    --duration 6 --resume --subsample 1 \
    --codecs librav1e --rav1e-speed "$SPEED" < /dev/null
  echo "--- rc=$?（speed=$SPEED）---"
done
echo "########## [$(date '+%F %H:%M')] VU rav1e 标定结束 ##########"
