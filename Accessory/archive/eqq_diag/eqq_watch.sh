#!/bin/bash
# 庚/己 批次看护：每 10 min 记录一次进度到/root 日志，异常时按resume 语义原样重启。
# ⚠ 重启前必须确认没有另一进程在跑同一 workdir。
LOG=/tmp/eqq_watch.log
D6=/tmp/eqq_uni_6s
while true; do
  ts=$(date '+%F %T')
  line="$ts"
  for t in new1 word_world_2 bbc_s01e01 bbc_s03e01 bbc_s05e01; do
    f=$D6/$t/points.json
    n=0; [ -f "$f" ] && n=$(python3 -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
    running=$(pgrep -f "measure_uni.py .*--out $D6/$t " >/dev/null && echo Y || echo N)
    line="$line  $t=$n/$running"
  done
  echo "$line" >> $LOG
  # 庚 全完成（每条 65 且进程已退）⇒ 启动己
  done6=1
  for t in new1 word_world_2 bbc_s01e01 bbc_s03e01 bbc_s05e01; do
    f=$D6/$t/points.json
    n=0; [ -f "$f" ] && n=$(python3 -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
    [ "$n" -ge 65 ] || done6=0
  done
  if [ "$done6" = 1 ] && ! pgrep -f "measure_uni.py .*--out /tmp/eqq_uni_10s/" >/dev/null; then
    if [ ! -f /tmp/eqq_uni_10s/_launched ]; then
      touch /tmp/eqq_uni_10s/_launched
      echo "$ts 庚全完成 ⇒ 启动己" >> $LOG
      bash /tmp/run_ji.sh >> $LOG 2>&1
    fi
  fi
  sleep 600
done
