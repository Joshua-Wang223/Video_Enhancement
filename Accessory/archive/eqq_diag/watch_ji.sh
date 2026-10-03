#!/bin/bash
# 守护：等己（10s 侧）5 个 workdir 全部 >=65 点。
#判完成一律用 65（不是 80）。
# 异常判据：
#   进程消失 且 键数>=65 => 正常完成，不重启
#   进程消失 且 键数<65  => 原样重启对应 tag（重启前确认无同 workdir 进程）
TAGS=(anim_10s anim_subs_10s dark_10s ui_10s texture_10s)
ROOT=/tmp/eqq_uni_10s
GOAL=65

keys() { [ -f "$ROOT/$1/points.json" ] && python3 -c "import json;print(len(json.load(open('$ROOT/$1/points.json'))))" 2>/dev/null || echo 0; }

declare -A RESTARTED
for round in $(seq 1 400); do
  alldone=1; line=""
  for t in "${TAGS[@]}"; do
    n=$(keys "$t"); line="$line $t=$n"
    if [ "$n" -ge "$GOAL" ]; then continue; fi
    alldone=0
    # 该 tag 未完成
    if ! pgrep -f "measure_uni.py .*--out $ROOT/$t( |$)" > /dev/null 2>&1; then
      if [ "${RESTARTED[$t]:-0}" -lt 3 ]; then
        RESTARTED[$t]=$(( ${RESTARTED[$t]:-0} + 1 ))
        echo "[$(date '+%H:%M:%S')] 重启 $t (keys=$n, 第${RESTARTED[$t]}次)"
        nohup python3 /tmp/measure_uni.py --src "$ROOT/src/$t.mp4" --out "$ROOT/$t" \
          --tiers libx265,libvpx-vp9,libaom-av1,libsvtav1,librav1e,librav1e@10 --duration 10 \
          > "$ROOT/$t.log" 2>&1 &
        sleep 2
      else
        echo "[$(date '+%H:%M:%S')] ⚠ $t 进程消失且 keys=$n <$GOAL，已重启 3 次仍不达标 —— 需人工"
      fi
    fi
  done
  echo "[$(date '+%H:%M:%S')] round $round:$line"
  if [ "$alldone" = "1" ]; then
    echo "[$(date '+%H:%M:%S')] ✅ 己（10s 侧）5/5 全部 >=$GOAL 点 —— 可落地第八版"
    exit 0
  fi
  sleep 120
done
echo "watch 超时退出"
exit 1
