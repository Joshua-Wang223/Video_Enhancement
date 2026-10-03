#!/usr/bin/env bash
# 等质量标定：两阶段顺序跑，逐点 checkpoint（--resume 跳过已完成点）。
# 阶段 1：4 个快软编编码器（全部素材）
# 阶段 2：librav1e（speed 10 与 native 两档）
set -u
PROJ=/mnt/d/Workspace_Python/Video_Enhancement
cd "$PROJ" || exit 1
H=Accessory/probe/calibrate_equal_quality.py
COMMON="--duration 10 --resume --subsample 8 --workroot /tmp/eqq_calib"

echo "===== [$(date +%H:%M:%S)] 阶段 1：libx265 / libsvtav1 / libvpx-vp9 / libaom-av1 ====="
python3 "$H" $COMMON --codecs libx265,libsvtav1,libvpx-vp9,libaom-av1 < /dev/null
rc1=$?
echo "===== [$(date +%H:%M:%S)] 阶段 1 结束 rc=$rc1（继续阶段 2，已完成点会被跳过）====="

echo "===== [$(date +%H:%M:%S)] 阶段 2：librav1e（speed 10 → native）====="
python3 "$H" $COMMON --codecs librav1e --rav1e-speed 10,native < /dev/null
rc2=$?
echo "===== [$(date +%H:%M:%S)] 全部结束 rc1=$rc1 rc2=$rc2 ====="
exit $rc2
