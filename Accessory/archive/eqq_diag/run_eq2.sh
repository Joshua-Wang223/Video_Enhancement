#!/usr/bin/env bash
# 等质量标定 第二轮（修正 subsample 偏置 + 锚点移入判别区）
#   Stage 1: 4 素材 × 4 快编码器（x265/svtav1/vp9/aom）        ≈ 4h
#   Stage 2: 4 素材 × librav1e @speed10                        ≈ 4h
#   Stage 3: **2 素材** × librav1e 原生档（最慢）                ≈ 4.5h
#     为何 Stage 3 只减「素材条数」而不减时长/分辨率：
#       等质量表是**跨编码器**一致性表，rav1e 行若用更短 clip 或更低分辨率拟合，
#       就与其他行落在不同标定口径上（立项 K4：2s clip 的 qp84.7 与门禁素材 qp77 不符）。
#       减少素材条数则口径不变，只是 LOO 折数变少。
#     Stage 3 单独 workdir（tag 带 _n2），聚合时需与 _n4 的 points.json 合并。
set -u
PROJ=/mnt/d/Workspace_Python/Video_Enhancement
IV=/mnt/d/Workspace_Python/input_videos
cd "$PROJ" || exit 1
H=Accessory/probe/calibrate_equal_quality.py
BASE="--duration 10 --resume --subsample 1 --workroot /tmp/eqq2"
COMMON4="$BASE --src $IV/new5_raw.mp4 --src $IV/new4_raw.mp4 --src $IV/new1.mp4 --src $IV/word_world_2.mp4"
COMMON2="$BASE --src $IV/new5_raw.mp4 --src $IV/word_world_2.mp4"

echo "########## [$(date '+%F %H:%M')] R2-Stage1: 4 素材 × 4 快编码器 ##########"
python3 "$H" $COMMON4 --codecs libx265,libsvtav1,libvpx-vp9,libaom-av1 < /dev/null
rc1=$?
echo "########## [$(date '+%F %H:%M')] R2-Stage1 结束 rc=$rc1 ##########"

echo "########## [$(date '+%F %H:%M')] R2-Stage2: 4 素材 × librav1e @speed10 ##########"
python3 "$H" $COMMON4 --codecs librav1e --rav1e-speed 10 < /dev/null
rc2=$?
echo "########## [$(date '+%F %H:%M')] R2-Stage2 结束 rc=$rc2 ##########"

echo "########## [$(date '+%F %H:%M')] R2-Stage3: 2 素材 × librav1e native ##########"
python3 "$H" $COMMON2 --codecs librav1e --rav1e-speed native < /dev/null
rc3=$?
echo "########## [$(date '+%F %H:%M')] R2 全部结束 rc=$rc1/$rc2/$rc3 ##########"
exit $rc3
