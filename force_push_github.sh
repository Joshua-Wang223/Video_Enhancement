#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# force_push_github.sh —— 把「本环境（Linux /workspace/Video_Enhancement）」的
# 完整最新代码覆盖推送到 GitHub，用作远程 main 的唯一真源。
#
# 目标远程：https://github.com/Joshua-Wang223/Video_Enhancement.git
# 推送前远程 main 基线：0b0cf12（≈2026-08-30，399 文件）
#
# 设计要点
# ---------
# 1. 默认**保留远程历史**（新提交的父提交 = origin/main），只让「文件内容」全量覆盖；
#    这样既满足"覆盖"，事后也能从远程历史里翻旧版本。要彻底孤儿化请加 --orphan。
# 2. 大目录（models*/ temp/ output/ logs/ gfpgan/ .trt_cache/ .t2_cache/）由
#    .gitignore 挡住；脚本另加「提交树哨兵」二次拦截，绝不把 GB 级文件推上去。
# 3. 推送用 --force-with-lease（远程若被人推过就**拒绝**而不是盲推）。
# 4. 推送前自动打一个本地备份 tag（不会被推送），任何一步失败都可回滚。
# 5. `.gitignore` / `.gitattributes` / `tar_excludes.txt` 属**全仓治理文件**：
#    `.gitignore` 决定"什么能进树"、`tar_excludes.txt` 决定"快照里排掉什么"，
#    一律以 origin 版本为准（本地差异只告警、不采纳），避免任一环境带着旧规则
#    重建索引/打包后覆盖远程。确有需要本地优先时用 IGNORE_LOCAL=1（见第 3 节）。
#    ⚠️ 不要把本脚本自身加进该清单 —— 运行中被改写会让 bash 重读、行为未定义。
#
# 用法
# ----
#   cd /workspace/Video_Enhancement
#   bash force_push_github.sh                    # 正常执行
#   bash force_push_github.sh --orphan           # 单提交覆盖（丢弃远程历史）
#   PUSH=0 bash force_push_github.sh             # 只准备提交、不推送（预演）
#   GH_URL='https://<user>:<token>@github.com/...' bash force_push_github.sh
#   REPO=/path/to/repo bash force_push_github.sh # 指定仓库根
#   ALLOW_BIG=1 bash force_push_github.sh        # 放行大目录哨兵（确有需要时）
#   IGNORE_LOCAL=1 bash force_push_github.sh     # 治理文件(.gitignore/.gitattributes/tar_excludes.txt)用本地版
#
# 回滚（把远程恢复到推送前）
# --------------------------
#   git push --force origin <推送前SHA>:main
# 该 SHA 会打印在结尾的汇总里，同时也存在本地备份 tag backup/pre-force-push-*
# =============================================================================

set -euo pipefail

REPO="${REPO:-/workspace/Video_Enhancement}"
BRANCH="${BRANCH:-main}"
PUSH="${PUSH:-1}"
ALLOW_BIG="${ALLOW_BIG:-0}"
ORPHAN=0
for a in "$@"; do
  case "$a" in
    --orphan) ORPHAN=1 ;;
    --dry-run) PUSH=0 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "❌ 未知参数: $a（支持 --orphan / --dry-run）"; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
head1() { printf '\n=== %s ===\n' "$*"; }
die()  { printf '❌ %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
head1 "0. 仓库根自检"
# -----------------------------------------------------------------------------
[ -d "$REPO" ] || die "目录不存在: $REPO"
cd "$REPO"
for p in run.py \
         src/main_video_optimized.py \
         src/utils/config_manager.py \
         config/default_config.json \
         external/ifrnet_video/main.py \
         external/realesrgan_video/main.py ; do
  [ -e "$p" ] || die "缺少 $p —— $REPO 不是仓库根目录，已中止（未做任何修改）"
done
say "仓库根: $(pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  say "已有 .git：$(git rev-parse --git-dir)  → 走「在本地历史之上提交」路径"
  HAS_GIT=1
else
  say "尚无 .git → 走「新建仓库并以 origin/$BRANCH 为父提交」路径"
  HAS_GIT=0
fi

# -----------------------------------------------------------------------------
head1 "1. 解析远程 URL（优先环境变量，其次已有 origin / 令牌文件）"
# -----------------------------------------------------------------------------
URL="${GH_URL:-}"
scan_token() {  # $1=文件 → 读第一行非空内容作为令牌
  [ -f "$1" ] || return 1
  tr -d '\r\n' < "$1" | sed 's/[[:space:]]*$//' | head -c 200
}
if [ -z "$URL" ] && [ "$HAS_GIT" = "1" ]; then
  URL="$(git remote get-url origin 2>/dev/null || true)"
fi
if [ -z "$URL" ]; then
  for f in Video_Enhancement_github_token.txt ../Video_Enhancement_github_token.txt; do
    if T="$(scan_token "$f" 2>/dev/null)" && [ -n "$T" ]; then
      URL="https://Joshua-Wang223:${T}@github.com/Joshua-Wang223/Video_Enhancement.git"
      say "令牌取自: $f"
      break
    fi
  done
fi
if [ -z "$URL" ] && [ -f ../git_set-url.txt ]; then
  URL="$(grep -oE 'https://[^[:space:]]+@github\.com/[^[:space:]]+\.git' ../git_set-url.txt | head -1 || true)"
  if [ -n "$URL" ]; then say "远程 URL 取自: ../git_set-url.txt"; fi
fi
[ -n "$URL" ] || die "未找到远程 URL。请先执行: export GH_URL='https://<user>:<token>@github.com/Joshua-Wang223/Video_Enhancement.git'"
say "origin = $(printf '%s' "$URL" | sed -E 's#(https?://)[^@]*@#\1***@#')"

# -----------------------------------------------------------------------------
head1 "2. 挂接仓库并取回远程状态"
# -----------------------------------------------------------------------------
if [ "$HAS_GIT" = "0" ]; then
  git init -q
  git symbolic-ref HEAD "refs/heads/$BRANCH"
fi
git remote remove origin 2>/dev/null || true
git remote add origin "$URL"

git fetch -q origin "$BRANCH"
OLD="$(git rev-parse "origin/$BRANCH")"
TAG="backup/pre-force-push-$(date +%Y%m%d-%H%M%S)"
git tag -f "$TAG" "$OLD" >/dev/null
say "远程 $BRANCH（推送前） = $OLD"
say "已打本地备份 tag      = $TAG（不会被推送，用于回滚）"

# 推送阶段禁止交互式凭据提问（令牌已内联在 URL 里），避免脚本卡在提示符上
export GIT_TERMINAL_PROMPT=0

# -----------------------------------------------------------------------------
head1 "3. 治理文件以 origin 为准（.gitignore / .gitattributes / tar_excludes.txt）"
# -----------------------------------------------------------------------------
# [FIX-IGNORE-CANONICAL] 这些文件决定「什么能进树 / 快照里排掉什么」，必须全环境一致。
# 原实现只在本地缺失时才从远程恢复：任一环境带着旧版 .gitignore 跑本脚本，
# 就会用旧规则重建索引并覆盖远程 —— 2026-09-23 实测事故即由此而来
# （另一环境缺 .gitattributes + 未锚定的 models/ → 覆盖后 EOL 锁定丢失、
#  external/IFRNet/models/ 等架构源码被剔除，全新 clone 起不来）。
# 现改为：一律采用 origin/$BRANCH 的版本；本地有差异只告警不采纳。
# 需要本地优先时设 IGNORE_LOCAL=1。
# 注：tar_excludes.txt 不影响本脚本建树，但它是「打快照交给其他环境」的排除清单；
#     旧版（未锚定的 models）会把 external/IFRNet/models/ 等源码目录排掉，2026-09-23
#     已实测复发过一次，故一并纳入治理清单。
for _f in .gitignore .gitattributes tar_excludes.txt; do
  if ! git cat-file -e "$OLD:$_f" 2>/dev/null; then
    say "$_f 在 origin/$BRANCH 中不存在 → 跳过（保持本地现状）"
    continue
  fi
  git show "$OLD:$_f" > "$_f.canonical"
  if [ -f "$_f" ] && ! cmp -s "$_f" "$_f.canonical"; then
    if [ "${IGNORE_LOCAL:-0}" = "1" ]; then
      say "⚠️  $_f 与 origin 不一致；IGNORE_LOCAL=1 → 保留本地版本（$(wc -l < "$_f" | tr -d ' ') 行）"
      rm -f "$_f.canonical"
      continue
    fi
    say "⚠️  $_f 与 origin 不一致 → 覆盖为 origin 版本（本地差异被丢弃；本地优先请设 IGNORE_LOCAL=1）"
  fi
  mv -f "$_f.canonical" "$_f"
  say "$_f ← origin 版本（$(wc -l < "$_f" | tr -d ' ') 行）"
done

# -----------------------------------------------------------------------------
head1 "4. 生成覆盖提交（工作区内容全量进树）"
# -----------------------------------------------------------------------------
# 索引必须**每轮重建**：.gitignore 只作用于「未跟踪」路径，对已在索引里的路径无效。
# 因此上一次中断运行留在索引里的 output/xxx 之类会被后续 add 继续带上（本轮实测踩到）。
# 重建索引前先报出「HEAD 里已跟踪、但被 .gitignore 覆盖」的路径 —— 重建会把它们
# 从提交树里去掉，属于有意的覆盖语义，但必须让人看见，避免误删。
if git rev-parse -q --verify HEAD >/dev/null 2>&1; then
  IGN_TRACKED="$(git ls-tree -r --name-only HEAD \
                 | git check-ignore --no-index --stdin 2>/dev/null || true)"
  if [ -n "$IGN_TRACKED" ]; then
    say "⚠️  以下路径在上一版提交里被跟踪，但当前 .gitignore 判定应忽略（重建索引后会从树里移除）："
    printf '%s\n' "$IGN_TRACKED" | sed -n '1,20p'
    [ "$ALLOW_BIG" = "1" ] || die "如确认要移除它们，请加 ALLOW_BIG=1 重跑"
  fi
fi

git read-tree --empty          # 索引置空 → 提交树 == 工作区内容（逐文件全量覆盖）
git add -A                     # 此时 .gitignore 对所有路径重新生效
TREE="$(git write-tree)"
say "提交树 = $TREE"

if [ "$ORPHAN" = "1" ]; then
  C="$(git -c user.name=Joshua-Wang223 -c user.email=wangjieshu223@gmail.com \
       commit-tree "$TREE" -m "Sync: /workspace latest complete code ($(date +%Y-%m-%d))")"
  say "已按 --orphan 生成**孤儿**提交（远程历史将被彻底替换）"
else
  if git rev-parse -q --verify HEAD >/dev/null 2>&1; then
    PARENT="$(git rev-parse HEAD)"
    [ "$PARENT" = "$OLD" ] || say "父提交取本地 HEAD（$PARENT），远程为 $OLD"
  else
    PARENT="$OLD"
  fi
  C="$(git -c user.name=Joshua-Wang223 -c user.email=wangjieshu223@gmail.com \
       commit-tree "$TREE" -p "$PARENT" \
       -m "Sync: /workspace latest complete code ($(date +%Y-%m-%d))")"
fi
git update-ref "refs/heads/$BRANCH" "$C"
git symbolic-ref HEAD "refs/heads/$BRANCH"
NEW="$C"
say "新提交 = $NEW"

# -----------------------------------------------------------------------------
head1 "5. 提交内容哨兵"
# -----------------------------------------------------------------------------
CHANGED="$(git diff --name-only "$OLD" "$NEW" | wc -l | tr -d ' ')"
say "相对 $OLD 的变更条目数: $CHANGED"
if [ "$CHANGED" = "0" ]; then
  say "提交树与远程 $OLD 完全相同 —— 内容已同步，没有需要推送的差异。"
  say "（若这不是你的预期，说明解压/同步的是旧快照，请先确认工作区确实是「完整最新版」。）"
  exit 0
fi

say "--- 前 30 条变更 ---"
# [FIX-PIPEFAIL-SIGPIPE] 原为 `git diff … | head -30`：变更条目 >30 时 head 提前关闭
# 管道 → git 收到 SIGPIPE(141) → 在 `set -o pipefail` 下**整脚本静默中止**
# （不报错、不推送；且外层常写作 `bash force_push_github.sh | tee log`，
# 退出码被 tee 掩盖成 0，看起来"跑完了"）。改用读满才退出的 sed 避开管道早闭。
# 同类 `printf … | head -20` 一并替换（大目录清单可能超过 20 行，同样会触发）。
git diff --name-status "$OLD" "$NEW" | sed -n '1,30p'

BIG="$(git ls-tree -r --name-only "$NEW" \
        | grep -E '^(models|models_[^/]*|temp|output|logs|gfpgan|\.trt_cache|\.t2_cache)/' || true)"
if [ -n "$BIG" ]; then
  if [ "$ALLOW_BIG" = "1" ]; then
    say "⚠️  大目录已进树，但 ALLOW_BIG=1 → 放行："
    printf '%s\n' "$BIG" | sed -n '1,20p'
  else
    say "❌ 以下大目录被纳入提交（.gitignore 没挡住）："
    printf '%s\n' "$BIG" | sed -n '1,20p'
    die "请先修 .gitignore 后重跑；确有需要可 ALLOW_BIG=1 放行"
  fi
else
  say "✅ 大目录（models*/temp/output/logs/gfpgan/.trt_cache/.t2_cache）未被纳入"
fi

BYTES="$(git ls-tree -r -l "$NEW" | awk '{s += $4} END {printf "%d", s + 0}')"
say "提交树总大小: $((BYTES / 1048576)) MB ($BYTES 字节)"
if [ "$BYTES" -gt 209715200 ]; then
  say "⚠️  超过 200MB，推送可能被 GitHub 拒绝（单文件上限 100MB）"
fi

# -----------------------------------------------------------------------------
head1 "6. 预演 + 推送"
# -----------------------------------------------------------------------------
if [ "$PUSH" = "0" ]; then
  say "PUSH=0（或 --dry-run）→ 停在推送前。要真正推送请重跑不带该参数。"
  say "本地分支 $BRANCH 已指向 $NEW；远程未改动。"
  exit 0
fi

git push --dry-run --force-with-lease="$BRANCH:$OLD" origin "$NEW:refs/heads/$BRANCH"
say "--- 预演通过，开始正式推送 ---"
git push --force-with-lease="$BRANCH:$OLD" origin "$NEW:refs/heads/$BRANCH"

# -----------------------------------------------------------------------------
head1 "7. 复核"
# -----------------------------------------------------------------------------
git ls-remote origin "refs/heads/$BRANCH"
git log --oneline -3 "$NEW"

cat <<EOF

========================= 汇总 =========================
远程 $BRANCH 推送前 : $OLD
远程 $BRANCH 推送后 : $NEW
模式                 : $([ "$ORPHAN" = "1" ] && echo "孤儿提交（历史已替换）" || echo "在 $OLD 之上新增一个提交（历史保留）")
变更条目             : $CHANGED
本地备份 tag         : $TAG

如需回滚远程：
  git push --force origin $OLD:refs/heads/$BRANCH
========================================================
EOF
