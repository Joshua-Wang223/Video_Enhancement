#!/usr/bin/env bash
# ============================================================================
# [FIX-NVENC-TEST-ISOLATE] 逐文件隔离跑 pytest —— 方案 A
#
# 立项：Plan/NVENC硬件测试隔离_立项Prompt.md
#
# 问题：`pytest Accessory/` 全量跑会在 NVENC 编码器测试阶段 SIGSEGV（EXIT=139，
# 实测 `.........` 9 passed 后崩），但把崩溃的类**单独**跑却 2 passed ——
# 属跨测试类的状态污染（NVENC 会话 / CUDA context 未干净释放），不是某个类的缺陷。
# 一次 SIGSEGV 会把 pytest 自己一起带走，于是"跑一半炸掉"，**崩溃点无法归因**。
#
# 本脚本：**每个测试文件单独起一个 pytest 进程**，任一进程崩溃不影响其余；
# 最后汇总 pass / fail / crash / empty 四态，让每一次崩溃都能落到具体文件。
#
# 用法：
#   bash Accessory/run_all_isolated.sh                 # 跑全部 test_*.py
#   bash Accessory/run_all_isolated.sh -m hw           # 只跑标记为硬件测试的
#   bash Accessory/run_all_isolated.sh -m "not hw"     # 显式排除硬件测试
#   bash Accessory/run_all_isolated.sh --timeout 300   # 单文件超时（秒，默认 600）
#   bash Accessory/run_all_isolated.sh --file Accessory/probe/nvenc_la_frame_conservation_suite.py ...
#
# ⚠️ 本脚本**不改变**任何测试的取舍：崩溃记为 CRASH（不是 SKIP），
#    失败记为 FAIL。要跳过硬件测试必须**显式**传 `-m "not hw"` —— 见
#    memory/feedback_keep_strict_criteria_annotate.md 的约定
#    （保严格判据 + 标注，不为消警而放宽/隐藏）。
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

TIMEOUT=600
PYTEST_ARGS=()
FILES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --file)    FILES+=("$2"); shift 2 ;;
        *)         PYTEST_ARGS+=("$1"); shift ;;
    esac
done

if [ "${#FILES[@]}" -eq 0 ]; then
    # ① pytest 真回归：Accessory/test/ 下仍以 test_ 前缀命名（pytest 依赖此前缀收集）
    while IFS= read -r f; do FILES+=("$f"); done < <(
        find Accessory/test -name 'test_*.py' -not -path '*/__pycache__/*' \
             -not -name '*_bak*' -not -name '*.bak*' | sort)
    # ② 独立 harness：原 tests/test_nvenc_* 已改名去掉 test_ 前缀（它们本就不是
    #    pytest 模块，多为 `__test__ = False` 或直接 main()），改用 `python <file>`
    #    执行，避免改名后从隔离跑里静默消失。
    while IFS= read -r f; do FILES+=("$f"); done < <(
        find Accessory/probe -name '*.py' -not -path '*/__pycache__/*' \
             -not -name '*_bak*' -not -name '*.bak*' | sort)
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "找不到 python 解释器"; exit 127; }

# 预检：pytest 不是本项目的运行期依赖（requirements.txt 未声明），
# 缺了就直接说清楚，别让每个文件都伪装成 FAIL。
if ! "$PY" -m pytest --version >/dev/null 2>&1; then
    echo "❌ 当前解释器没有 pytest（$PY -m pytest --version 失败）。"
    echo "   本脚本只是隔离执行入口，不安装依赖；请先："
    echo "       $PY -m pip install pytest"
    echo "   （本机若为无 GPU 开发环境，这一步是可选的——"
    echo "     完整验证需在 Linux + GPU 侧执行。）"
    exit 2
fi

HAVE_TIMEOUT=0
command -v timeout >/dev/null 2>&1 && HAVE_TIMEOUT=1

echo "==============================================================="
echo " 逐文件隔离跑 pytest（方案 A）"
echo " 解释器 : $($PY -V 2>&1)"
echo " 文件数 : ${#FILES[@]}"
echo " 超时   : ${TIMEOUT}s/文件$([ "$HAVE_TIMEOUT" -eq 1 ] || echo '  ⚠️ 无 timeout 命令，单文件超时不生效')"
echo " 附加参数: ${PYTEST_ARGS[*]:-（无）}"
echo "==============================================================="

PASS=(); FAIL=(); CRASH=(); EMPTY=(); OTHER=()
TMPLOG="$(mktemp -d "${TMPDIR:-/tmp}/nvenc_isolate.XXXXXX")"
trap 'rm -rf "$TMPLOG"' EXIT

i=0
for f in "${FILES[@]}"; do
    i=$((i + 1))
    log="$TMPLOG/$(echo "$f" | tr '/' '_').log"
    printf '[%2d/%2d] %-52s ' "$i" "${#FILES[@]}" "$f"
# ⚠️ 必须带 `--capture=no`（= `-s`）：2026-09-16 实测，**开着捕获时 SIGSEGV 会被
#    pytest 的 capture teardown 异常掩盖**——真实 rc=139 被改写成 rc=1 且只打印
#    `ValueError: I/O operation on closed file` / `no tests ran`，
#    于是崩溃被误分类成 FAIL（不是 CRASH），本脚本"每个崩溃都可归因"的能力失效。
#    关掉捕获后 rc 与 `Fatal Python error: Segmentation fault` 都能如实读到。
# 另加 `-p no:cacheprovider`（不写 .pytest_cache）。
# Accessory/probe/ 下的独立 harness 不是 pytest 模块，直接以脚本方式执行；
# 其余（Accessory/test/）走 pytest 逐文件隔离。
case "$f" in
    Accessory/probe/*)
        if [ "$HAVE_TIMEOUT" -eq 1 ]; then
            timeout -k 10 "$TIMEOUT" "$PY" "$f" >"$log" 2>&1
        else
            "$PY" "$f" >"$log" 2>&1
        fi
        ;;
    *)
        if [ "$HAVE_TIMEOUT" -eq 1 ]; then
            timeout -k 10 "$TIMEOUT" "$PY" -m pytest "$f" -x -q --tb=short \
                --capture=no -p no:cacheprovider "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}" >"$log" 2>&1
        else
            "$PY" -m pytest "$f" -x -q --tb=short \
                --capture=no -p no:cacheprovider \
                "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}" >"$log" 2>&1
        fi
        ;;
esac
rc=$?

# 分类优先级：**崩溃特征串优先于返回码** —— 即使 rc 被环境噪声改写，也能归因为 CRASH。
if grep -qE 'Fatal Python error: Segmentation fault|SIGSEGV' "$log" 2>/dev/null; then
    echo "CRASH(SIGSEGV，日志含 Fatal Python error)"
    CRASH+=("$f"); continue
fi

    if [ "$rc" -eq 0 ]; then
        echo "PASS"; PASS+=("$f")
    elif [ "$rc" -eq 5 ]; then
        echo "EMPTY(无测试被收集)"; EMPTY+=("$f")
    elif [ "$rc" -ge 128 ]; then
        sig=$((rc - 128))
        echo "CRASH(signal $sig, rc=$rc)"; CRASH+=("$f")
    elif [ "$rc" -eq 1 ]; then
        echo "FAIL"; FAIL+=("$f")
    elif [ "$rc" -eq 124 ]; then
        echo "CRASH(超时 ${TIMEOUT}s)"; CRASH+=("$f")
    else
        echo "OTHER(rc=$rc)"; OTHER+=("$f")
    fi
done

echo
echo "==============================================================="
echo " 汇总"
echo "---------------------------------------------------------------"
show() {  # $1=标题 $2..=文件
    local title="$1"; shift
    if [ "$#" -eq 0 ]; then
        printf ' %-16s 0\n' "$title"
    else
        printf ' %-16s %d\n' "$title" "$#"
        for x in "$@"; do printf '        · %s\n' "$x"; done
    fi
}
show "PASS"  "${PASS[@]+"${PASS[@]}"}"
show "FAIL"  "${FAIL[@]+"${FAIL[@]}"}"
show "CRASH" "${CRASH[@]+"${CRASH[@]}"}"
show "EMPTY" "${EMPTY[@]+"${EMPTY[@]}"}"
show "OTHER" "${OTHER[@]+"${OTHER[@]}"}"
echo "---------------------------------------------------------------"
echo " 合计 ${#FILES[@]} 个文件"
echo "==============================================================="

# 崩溃/失败不吞掉：任何一种都返回非 0，便于 CI 判定。
if [ "${#FAIL[@]}" -gt 0 ] || [ "${#CRASH[@]}" -gt 0 ] || [ "${#OTHER[@]}" -gt 0 ]; then
    echo
    echo "[提示] CRASH 多为跨类 CUDA context 污染（本立项根因）。"
    echo "       下一步用二分法缩小："
    echo "         pytest Accessory/<A>.py Accessory/<B>.py -x --tb=short"
    echo "       找到必崩的**最小两类组合**后再考虑 conftest 里加"
    echo "       autouse 清理 fixture（方案 B）。"
    exit 1
fi
exit 0
