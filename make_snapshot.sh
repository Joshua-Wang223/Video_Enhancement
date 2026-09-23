#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# make_snapshot.sh —— 用 `git archive` 生成仓库快照（2026-09-23 起替代 tar 方案）
#
# 为什么改用 git archive
# ----------------------
# 它只打包**已跟踪**文件，于是权重（models_*/.trt_cache）、缓存、日志、会话转储
# 天然进不来 —— 不再需要维护排除清单，也就消除了 tar_excludes.txt 时代的两类事故：
#   - 模式写太窄（如 `/models` 匹配不到）→ 权重**全量进快照**（GB 级）
#   - 模式写太宽（如裸 `models`）        → 把 `external/**/models/` **源码**排掉
#
# 用法
# ----
#   cd /workspace/Video_Enhancement
#   bash make_snapshot.sh                     # → ../Video_Enhancement_snapshot_<ts>.tar.gz
#   OUT=/tmp/snap.tgz bash make_snapshot.sh   # 指定输出路径
#   REF=origin/main bash make_snapshot.sh     # 指定归档 ref（默认 HEAD）
#   FORMAT=zip bash make_snapshot.sh          # 换归档格式
#   DIRTY=allow bash make_snapshot.sh         # 工作区不干净时也继续（见下）
#
# ⚠️ git archive 只含**已提交**内容：未提交改动**不会**进快照。
#    因此工作区不干净时脚本默认**中止**（避免"以为带上了、其实没带"）；
#    确认只想要已提交版本时用 DIRTY=allow。
# =============================================================================

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
REF="${REF:-HEAD}"
FORMAT="${FORMAT:-tar.gz}"
DIRTY="${DIRTY:-abort}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$(dirname "$REPO")/Video_Enhancement_snapshot_$TS.$FORMAT}"

say() { printf '%s\n' "$*"; }
die() { printf '❌ %s\n' "$*" >&2; exit 1; }

cd "$REPO"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO 不是 git 仓库"

# ── 工作区洁净度 ────────────────────────────────────────────────────────────
if [ -n "$(git status --porcelain)" ]; then
  if [ "$DIRTY" = "allow" ]; then
    say "⚠️  工作区不干净（DIRTY=allow → 继续）；以下未提交改动**不会**进快照："
    git status --porcelain | sed -n '1,10p'
  else
    say "❌ 工作区不干净，而 git archive 只含已提交内容 —— 未提交改动会被静默漏掉："
    git status --porcelain | sed -n '1,10p'
    die "请先 commit/push；若确实只要已提交版本，置 DIRTY=allow 重跑"
  fi
fi

# ── 归档 ────────────────────────────────────────────────────────────────────
SHA="$(git rev-parse --short "$REF")"
say "归档 ref : $REF ($SHA)"
say "输出     : $OUT"
git archive --format="$FORMAT" -o "$OUT" "$REF"
say "完成     : $(du -h "$OUT" | cut -f1)"

# ── 哨兵：快照里不应出现权重/缓存（正常必然为 0，非 0 说明有误入库）──────────
case "$OUT" in
  *.tar.gz|*.tgz|*.tar)
    N="$(tar -tzf "$OUT" 2>/dev/null | wc -l | tr -d ' ')"
    BAD="$(tar -tzf "$OUT" 2>/dev/null \
           | grep -cE '(^|/)\.?(models_[^/]*|trt_cache|t2_cache)/|^\.?models/|\.(pth|trt|onnx|safetensors|ckpt)$' || true)"
    if [ "${BAD:-0}" != "0" ]; then
      say "❌ 快照中发现疑似权重/缓存条目 ${BAD} 条 —— 说明有本不该入库的文件被跟踪了："
      tar -tzf "$OUT" | grep -E '(^|/)\.?(models_[^/]*|trt_cache|t2_cache)/|^\.?models/|\.(pth|trt|onnx|safetensors|ckpt)$' | sed -n '1,20p'
      die "请先按 .gitignore 清理并 git rm --cached 相应文件"
    fi
    say "✅ 哨兵通过：快照不含权重/缓存，共 ${N} 个条目"
    ;;
  *)
    say "（非 tar 格式，跳过哨兵检查）"
    ;;
esac
