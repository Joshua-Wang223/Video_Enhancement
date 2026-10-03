#!/bin/bash
# 等质量「统一时长」补标任务状态检查
# 用法: bash /tmp/eqq_status.sh [6s|10s|all]
W=${1:-all}
n=0
report() {
  # expect = 每素材真实点数。measure_uni.py 口径：
  #   5 锚点 libx264（keys 去重，6 个 tier 各循环一次但 key 相同只存 5 个）
  #   + 4 软编 × 10 + rav1e × 10 + rav1e@10 × 10 = 5 + 60 = **65**
  # ⚠ 任务简报写的 80 是算术错误（5+40+20=65）
  local root=$1 label=$2 expect=$3
  echo "=========== [$label] $root (expect $expect/material) ==========="
  local total=0
  for d in "$root"/*/points.json; do
    [ -f "$d" ] || continue
    local tag=$(basename "$(dirname "$d")")
    local c=$(python3 - "$d" <<'PY'
import json,sys,collections
p=json.load(open(sys.argv[1]))
c=collections.Counter(k.split('|')[1] for k in p)
print(len(p), ' '.join(f'{k}={v}' for k,v in sorted(c.items())))
PY
)
    local cnt=$(echo "$c" | cut -d' ' -f1)
    total=$((total+cnt))
    local logsz=$(stat -c%s "$root/$tag.log" 2>/dev/null || echo 0)
    local running=$(pgrep -fc "measure_uni.py .*--out $root/$tag" 2>/dev/null || echo 0)
    printf "  %-16s %3d/%s  log=%sB  proc=%s\n" "$tag" "$cnt" "$expect" "$logsz" "$running"
    echo "      $c"
  done
  echo "  ---- TOTAL: $total / $((expect*$(ls -1d $root/*/ 2>/dev/null | wc -l)))"
  n=$((n+1))
}
case "$W" in
  6s)  report /tmp/eqq_uni_6s  "庚/6s" 65 ;;
  10s) report /tmp/eqq_uni_10s "己/10s" 65 ;;
  *)   report /tmp/eqq_uni_6s  "庚/6s" 65
       report /tmp/eqq_uni_10s "己/10s" 65 ;;
esac
echo "=========== 进程 ==========="
ps -eo pid,stat,etime,pcpu,cmd | grep '[m]easure_uni' || echo "  (无 measure_uni 进程)"
echo "=========== 负载 ==========="
uptime
nproc
echo "=========== ffmpeg 子进程数 ==========="
pgrep -c ffmpeg 2>/dev/null || echo 0