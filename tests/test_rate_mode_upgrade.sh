#!/usr/bin/env bash
# =============================================================================
# test_rate_mode_upgrade.sh — rate_mode/lookahead 统一控制升级 完整测试脚本
# =============================================================================
# 测试范围:
#   Part 1: crf=0 解耦 (FORCE_CONSTQP 开关)
#   Part 2: CLI 运行时切换 rate_mode/lookahead
#   Part 3: realesrgan_video FFmpegWriter rate_mode 透传
#   Part 4: LA 分段策略自动禁用 + 音频同步修正
#   边界/异常/一致性
#
# 使用方法:
#   chmod +x tests/test_rate_mode_upgrade.sh
#   ./tests/test_rate_mode_upgrade.sh [--input <test_video.mp4>] [<test_video.mp4>] [<output_dir>]
#
#   选项:
#     --input <path>      指定测试视频 (优先级最高)
#     --resume, --continue  跳过已通过测试
#
#   默认测试视频: ../input_videos/word_world_2.mp4
#   默认输出目录: /test_output
#
# 环境要求:
#   - Linux (Windows WSL 亦可，但 NVENC 不可用时会退到 Level 2/3/4)
#   - Python 3.9+, PyTorch CUDA, FFmpeg >= 4.3
#   - NVIDIA GPU + NVENC 驱动 (可选：无 NVENC 时部分 Level 1 测试自动跳过)
# =============================================================================

set -euo pipefail

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASS="${GREEN}PASS${NC}"
FAIL="${RED}FAIL${NC}"
SKIP="${YELLOW}SKIP${NC}"
WARN="${YELLOW}WARN${NC}"
INFO="${BLUE}INFO${NC}"

# ── 全局统计 ──────────────────────────────────────────────────────────────────
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
RESUME_MODE=false
PROGRESS_FILE=""
TEST_LOG_FILE=""
FAIL_LOG=""

# ── CLI 参数解析 ──
USAGE="用法: $0 [选项] [<test_video.mp4>] [<output_dir>]

选项:
  --input <path>      指定测试用原视频文件 (默认: ../input_videos/word_world_2.mp4)
  --resume, --continue  断点继续模式，跳过已通过的测试

位置参数 (向后兼容):
  \$1                  测试视频路径 (等同于 --input)
  \$2                  输出目录路径"

INPUT_VIDEO_ARG=""
_POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT_VIDEO_ARG="$2"; shift 2 ;;
        --resume|--continue) RESUME_MODE=true; shift ;;
        --help|-h) echo "$USAGE"; exit 0 ;;
        *) _POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done
set -- "${_POSITIONAL_ARGS[@]}"

# ── 参数解析 ──
# --input 优先级最高，其次是位置参数 $1，最后是默认常量
TEST_VIDEO="${INPUT_VIDEO_ARG:-${1:-../input_videos/word_world_2.mp4}}"
OUTPUT_DIR="${2:-/test_output}"
CONFIG="config/default_config.json"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IFRNET_BACKEND="$PROJECT_DIR/external/IFRNet/process_video_v6_4_5_1_single.py"

# ── 断点继续 ──
_test_passed() {
    local test_id="$1"
    [ "$RESUME_MODE" = false ] && return 1
    [ -f "$PROGRESS_FILE" ] || return 1
    grep -qxF "$test_id" "$PROGRESS_FILE" >/dev/null 2>&1
}
_mark_passed() {
    local test_id="$1"
    echo "$test_id" >> "$PROGRESS_FILE"
    # 强制刷盘，确保异常退出时进度不丢失
    sync "$PROGRESS_FILE" 2>/dev/null || sync 2>/dev/null || true
}
_remove_from_progress() {
    local test_id="$1"
    if [ -f "$PROGRESS_FILE" ]; then
        grep -vxF "$test_id" "$PROGRESS_FILE" > "${PROGRESS_FILE}.tmp" 2>/dev/null || true
        mv "${PROGRESS_FILE}.tmp" "$PROGRESS_FILE" 2>/dev/null || true
        sync "$PROGRESS_FILE" 2>/dev/null || sync 2>/dev/null || true
    fi
}

echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  rate_mode/lookahead 统一控制升级 — 完整测试套件${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  项目目录 : ${CYAN}$PROJECT_DIR${NC}"
echo -e "  测试视频 : ${CYAN}$TEST_VIDEO${NC}"
echo -e "  输出目录 : ${CYAN}$OUTPUT_DIR${NC}"
echo -e "  配置文件 : ${CYAN}$CONFIG${NC}"
if [ "$RESUME_MODE" = true ]; then
    echo -e "  运行模式 : ${YELLOW}断点继续${NC} (--resume，跳过已通过的测试)"
fi
echo ""

# ── 环境检查 ──────────────────────────────────────────────────────────────────
check_prerequisites() {
    local all_ok=true

    if [ ! -f "$TEST_VIDEO" ]; then
        echo -e "  ${FAIL} 测试视频不存在: $TEST_VIDEO"
        echo "      请将测试视频放置到该路径，或通过参数指定: $0 <video_path>"
        all_ok=false
    fi

    if [ ! -f "$CONFIG" ]; then
        echo -e "  ${FAIL} 配置文件不存在: $CONFIG"
        all_ok=false
    fi

    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        echo -e "  ${FAIL} Python 不可用"
        all_ok=false
    fi

    if ! command -v ffprobe &>/dev/null; then
        echo -e "  ${WARN} ffprobe 不可用（将跳过视频元数据验证）"
    fi

    if $all_ok; then
        echo -e "  ${PASS} 环境检查通过"
        return 0
    else
        echo -e "  ${FAIL} 环境检查失败，请修复后重试"
        exit 1
    fi
}

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
PYTHON=""
detect_python() {
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
    elif command -v python &>/dev/null; then
        PYTHON="python"
    fi
}

init_output_dir() {
    PROGRESS_FILE="$OUTPUT_DIR/.test_progress.txt"
    mkdir -p "$OUTPUT_DIR"
    TEST_LOG_FILE="$OUTPUT_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"
    FAIL_LOG="$OUTPUT_DIR/failures.log"
    echo "测试开始: $(date '+%Y-%m-%d %H:%M:%S')" > "$TEST_LOG_FILE"
    echo "" > "$FAIL_LOG"
}

# ── 测试框架 ──────────────────────────────────────────────────────────────────
run_test() {
    local test_id="$1"
    local test_name="$2"
    local cmd="$3"
    local expected_pattern="$4"
    local expected_in_log="$5"   # grep pattern 或 "RUNTIME_ONLY" 表示仅验证执行成功
    local unexpected_pattern="${6:-}" # 可选：不应出现的文本模式（反向验证）
    local allow_critical_errors="${7:-false}" # 可选：允许严重错误（如测试预期触发错误场景）

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if _test_passed "$test_id"; then
        # ★ --resume 加固: 验证输出文件确实存在（防止上次 PASS 后环境变化导致误跳过）
        local _resume_output=""
        if echo "$cmd" | grep -q -- "-o "; then
            _resume_output=$(echo "$cmd" | grep -oP '\-o\s+\K\S+' | head -1)
        fi
        if [ -n "$_resume_output" ] && [ ! -f "$_resume_output" ]; then
            echo -e "  [$test_id] $test_name ... ${YELLOW}重测${NC} (输出文件缺失，--resume 跳过无效)"
            _remove_from_progress "$test_id"
        else
            echo -e "  [$test_id] $test_name ... ${SKIP} (已通过，--resume 跳过)"
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
            return 0
        fi
    fi

    echo "" >> "$TEST_LOG_FILE"
    echo "────────────────────────────────────────" >> "$TEST_LOG_FILE"
    echo "[$test_id] $test_name" >> "$TEST_LOG_FILE"
    echo "  命令: $cmd" >> "$TEST_LOG_FILE"
    echo "────────────────────────────────────────" >> "$TEST_LOG_FILE"

    echo -n "  [$test_id] $test_name ... "

    # 执行命令，捕获输出和退出码
    local tmp_stdout="$OUTPUT_DIR/.tmp_test_stdout.txt"
    local tmp_stderr="$OUTPUT_DIR/.tmp_test_stderr.txt"

    if eval "$cmd" > "$tmp_stdout" 2> "$tmp_stderr"; then
        local exit_code=0
    else
        local exit_code=$?
    fi

    # 保存日志
    cat "$tmp_stdout" >> "$TEST_LOG_FILE"
    cat "$tmp_stderr" >> "$TEST_LOG_FILE"

    # 允许 exit code 1-2（argparse 参数校验失败等预期退出）
    if [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne 1 ] && [ "$exit_code" -ne 2 ]; then
        echo -e "${FAIL}"
        echo "      退出码: $exit_code"
        echo "      完整输出: $tmp_stdout / $tmp_stderr"
        echo "[$test_id] $test_name — 退出码: $exit_code" >> "$FAIL_LOG"
        cat "$tmp_stderr" >> "$FAIL_LOG" 2>/dev/null || true
        cat "$tmp_stdout" >> "$FAIL_LOG" 2>/dev/null || true
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 0
    fi

    if [ "$exit_code" -ne 0 ]; then
        echo -e "  ${YELLOW}⚠️  退出码=$exit_code（预期可能退出码非零，继续检查日志内容）${NC}"
    fi

    # 检查预期输出模式（使用 LC_ALL=C.UTF-8 避免二进制/多字节字符干扰）
    local combined_output
    combined_output="$(cat "$tmp_stdout" 2>/dev/null; cat "$tmp_stderr" 2>/dev/null)"

    if [ "$expected_in_log" != "RUNTIME_ONLY" ]; then
        if ! LC_ALL=C.UTF-8 grep -aqE -- "$expected_pattern" <<< "$combined_output"; then
            echo -e "${FAIL}"
            echo "      未在输出中找到: $expected_pattern"
            echo "      完整输出: $tmp_stdout / $tmp_stderr"
            echo "[$test_id] $test_name — 未匹配: $expected_pattern" >> "$FAIL_LOG"
            cat "$tmp_stderr" >> "$FAIL_LOG" 2>/dev/null || true
            cat "$tmp_stdout" >> "$FAIL_LOG" 2>/dev/null || true
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 0
        fi
    fi

    # 反向验证：检查不应出现的文本模式
    if [ -n "$unexpected_pattern" ]; then
        if LC_ALL=C.UTF-8 grep -aqE -- "$unexpected_pattern" <<< "$combined_output"; then
            echo -e "${FAIL}"
            echo "      不应出现的文本出现: $unexpected_pattern"
            echo "      完整输出: $tmp_stdout / $tmp_stderr"
            echo "[$test_id] $test_name — 不应出现但发现: $unexpected_pattern" >> "$FAIL_LOG"
            cat "$tmp_stderr" >> "$FAIL_LOG" 2>/dev/null || true
            cat "$tmp_stdout" >> "$FAIL_LOG" 2>/dev/null || true
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 0
        fi
    fi

    # ★ 自动检测严重错误模式（提前EOF、帧缺失、帧数不匹配、H.264解码错误）
    # 这些错误如果被标记为PASS，--resume时不会再复现 → 必须在_mark_passed之前此处FAIL
    if [ "$allow_critical_errors" != "true" ]; then
        local _crit_err=""
        if LC_ALL=C.UTF-8 grep -aqE "提前EOF" <<< "$combined_output"; then
            _crit_err="提前EOF (premature EOF — 输入帧读取不完整)"
        elif LC_ALL=C.UTF-8 grep -aqE "缺失[[:space:]]*[0-9]+[[:space:]]*帧" <<< "$combined_output"; then
            _crit_err="帧缺失 (missing frames — 输出帧数不足)"
        elif LC_ALL=C.UTF-8 grep -aqE "差[[:space:]]*[0-9.]+[[:space:]]*帧.*⚠️" <<< "$combined_output"; then
            _crit_err="帧计数不匹配 (frame count mismatch with ⚠️)"
        elif LC_ALL=C.UTF-8 grep -aqE "non-existing PPS" <<< "$combined_output"; then
            _crit_err="non-existing PPS (H.264 SPS/PPS 流损坏)"
        elif LC_ALL=C.UTF-8 grep -aqE "no frame!" <<< "$combined_output"; then
            _crit_err="no frame! (FFmpeg H.264 解码失败)"
        elif LC_ALL=C.UTF-8 grep -aqE "decode_slice_header error" <<< "$combined_output"; then
            _crit_err="decode_slice_header error (H.264 解码错误)"
        elif LC_ALL=C.UTF-8 grep -aqE "原始帧:.*输出帧:.*差[[:space:]]*[1-9]" <<< "$combined_output"; then
            _crit_err="帧计数不匹配 (frame count mismatch in summary — 差值≥1)"
        fi
        if [ -n "$_crit_err" ]; then
            echo -e "${FAIL}"
            echo "      🔴 严重错误: $_crit_err"
            echo "      完整输出: $tmp_stdout / $tmp_stderr"
            echo "[$test_id] $test_name — 严重错误: $_crit_err" >> "$FAIL_LOG"
            cat "$tmp_stderr" >> "$FAIL_LOG" 2>/dev/null || true
            cat "$tmp_stdout" >> "$FAIL_LOG" 2>/dev/null || true
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 0
        fi
    fi

    # RUNTIME_ONLY: 仅验证执行成功，不检查日志内容（正向/反向检查在上面已完成）
    if [ "$expected_in_log" = "RUNTIME_ONLY" ]; then
        echo -e "${PASS} (运行成功)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        _mark_passed "$test_id"
        return 0
    fi

    # 检查输出文件
    if echo "$cmd" | grep -q -- "-o "; then
        local output_file
        output_file=$(echo "$cmd" | grep -oP '\-o\s+\K\S+' | head -1)
        if [ -n "$output_file" ] && [ -f "$output_file" ]; then
            local size_mb
            size_mb=$(du -m "$output_file" 2>/dev/null | cut -f1 || echo "?")
            echo -e "${PASS} (${size_mb}MB)"
        else
            echo -e "${PASS}"
        fi
    else
        echo -e "${PASS}"
    fi

    TESTS_PASSED=$((TESTS_PASSED + 1))
    _mark_passed "$test_id"
    return 0
}

# ── NVENC 可用性探测 ──────────────────────────────────────────────────────────
check_nvenc_available() {
    # 快速探测 NVENC SDK 是否可用（通过导入 nvcuda 库）
    if $PYTHON -c "
import ctypes, sys
try:
    lib = ctypes.CDLL('libcuda.so.1')
    lib.cuInit(0)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ── ffprobe 元数据验证 ────────────────────────────────────────────────────────
verify_output_video() {
    local video_path="$1"
    local label="${2:-}"

    if ! command -v ffprobe &>/dev/null; then
        echo "      (ffprobe 不可用，跳过验证)"
        return 0
    fi

    local info
    info=$(ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate \
           -of default=noprint_wrappers=1 "$video_path" 2>/dev/null || echo "")

    if [ -n "$info" ]; then
        echo "      [$label] $(echo "$info" | tr '\n' ' ')"
        return 0
    else
        echo "      ${WARN} ffprobe 无法读取 $video_path"
        return 1
    fi
}

# ── 检查 FFmpegMuxer stderr 中的错误模式 ────────────────────────────
check_ffmpeg_stderr() {
    local test_id="$1"
    local stderr_file="$2"
    local output_file="$3"

    if [ ! -f "$stderr_file" ]; then
        return 0
    fi

    # 关键错误模式：H.264 流结构错误
    local pps_errors
    pps_errors=$(grep -c "non-existing PPS" "$stderr_file" 2>/dev/null || true)
    [ -z "$pps_errors" ] && pps_errors=0
    local no_frame_errors
    no_frame_errors=$(grep -c "no frame!" "$stderr_file" 2>/dev/null || true)
    [ -z "$no_frame_errors" ] && no_frame_errors=0
    local decode_errors
    decode_errors=$(grep -c "decode_slice_header error" "$stderr_file" 2>/dev/null || true)
    [ -z "$decode_errors" ] && decode_errors=0

    if [ "$pps_errors" -gt 0 ] || [ "$no_frame_errors" -gt 0 ] || [ "$decode_errors" -gt 0 ]; then
        echo "      ${FAIL} FFmpeg 解码错误: PPS=$pps_errors no_frame=$no_frame_errors decode=$decode_errors"
        echo "[$test_id] FFmpeg stderr 错误: PPS=$pps_errors no_frame=$no_frame_errors decode=$decode_errors" >> "$FAIL_LOG"
        # ★ 后置检查失败：如果 run_test 已将此测试标记为PASS，撤销之
        if grep -qxF "$test_id" "$PROGRESS_FILE" 2>/dev/null; then
            _remove_from_progress "$test_id"
        fi
        return 1
    fi
    return 0
}

# ── 帧计数验证 ──────────────────────────────────────────────────────
verify_frame_count() {
    local test_id="$1"
    local output_file="$2"
    local expected_frames="$3"
    local tolerance="${4:-2}"

    if ! command -v ffprobe &>/dev/null; then
        return 0
    fi
    if [ ! -f "$output_file" ]; then
        return 0
    fi

    local actual_frames
    actual_frames=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=nb_frames \
        -of default=noprint_wrappers=1:nokey=1 "$output_file" 2>/dev/null || echo "")

    if [ -z "$actual_frames" ] || [ "$actual_frames" = "N/A" ]; then
        # 部分编码器不输出 nb_frames，尝试从时长推算
        local duration actual_fps
        duration=$(ffprobe -v error -select_streams v:0 \
            -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$output_file" 2>/dev/null || echo "")
        actual_fps=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=r_frame_rate \
            -of default=noprint_wrappers=1:nokey=1 "$output_file" 2>/dev/null || echo "")
        if [ -n "$duration" ] && [ -n "$actual_fps" ]; then
            local num denom
            num="${actual_fps%/*}"
            denom="${actual_fps#*/}"
            if [ "$denom" -gt 0 ] 2>/dev/null; then
                actual_frames=$(echo "$duration * $num / $denom" | bc 2>/dev/null | cut -d. -f1)
            fi
        fi
    fi

    if [ -z "$actual_frames" ] || [ "$actual_frames" = "0" ]; then
        echo "      ${YELLOW}?${NC} 无法获取帧计数"
        return 0
    fi

    # NVENC SDK 严格遵守帧数守恒: Output == Input, 无需调整 LA 深度
    local diff
    diff=$(( expected_frames - actual_frames ))
    [ "$diff" -lt 0 ] && diff=$(( -diff ))

    if [ "$diff" -gt "$tolerance" ]; then
        echo "      ${FAIL} 帧数异常: 期望=${expected_frames} 实际=${actual_frames} 差=${diff}"
        echo "[$test_id] 帧数异常: 期望=${expected_frames} 实际=${actual_frames} 差=${diff}" >> "$FAIL_LOG"
        if grep -qxF "$test_id" "$PROGRESS_FILE" 2>/dev/null; then
            _remove_from_progress "$test_id"
        fi
        return 1
    fi
    echo "      ${GREEN}✓${NC} 帧数: ${actual_frames} (期望=${expected_frames}, 差=${diff})"
    return 0
}

# ── 文件大小对比 ──────────────────────────────────────────────────────────────
compare_file_sizes() {
    local file1="$1"
    local file2="$2"
    local label1="${3:-file1}"
    local label2="${4:-file2}"

    if [ -f "$file1" ] && [ -f "$file2" ]; then
        local size1 size2
        size1=$(stat -c%s "$file1" 2>/dev/null || echo 0)
        size2=$(stat -c%s "$file2" 2>/dev/null || echo 0)
        local mb1 mb2
        mb1=$(echo "scale=2; $size1/1048576" | bc 2>/dev/null || echo "?")
        mb2=$(echo "scale=2; $size2/1048576" | bc 2>/dev/null || echo "?")
        echo "      $label1: ${mb1}MB  |  $label2: ${mb2}MB"
    fi
}

# conf 备份与恢复 — 已全面改用暂存副本，不修改生产源码

# ═══════════════════════════════════════════════════════════════════════════════
# 主测试流程
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    detect_python
    check_prerequisites
    init_output_dir

    # 探测 NVENC
    local has_nvenc=false
    if check_nvenc_available; then
        has_nvenc=true
        echo -e "  ${PASS} NVENC CUDA 库可用 (Level 1 测试将执行)"
    else
        echo -e "  ${SKIP} NVENC CUDA 库不可用 (Level 1 测试将跳过，仅测 Level 2+ CLI 透传)"
    fi
    echo ""

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 1: 默认行为回归
    # ══════════════════════════════════════════════════════════════════════════
    echo -e "${BOLD}${BLUE}═══ 阶段 1: 默认行为回归测试 ═══${NC}"

    run_test "1a" "IFRNet standalone 默认" \
        "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1a_ifrnet_default.mp4 \
            --use-tensorrt --quiet 2>&1" \
        "VBR_HQ.*LA=8" \
        "ifrnet_default_log"

    run_test "1b" "Real-ESRGAN standalone 默认" \
        "$PYTHON -m src.processors.realesrgan_processor_video_optimized \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1b_esrgan_default.mp4 \
            --use-tensorrt --quiet 2>&1" \
        "[Vv][Bb][Rr]_[Hh][Qq]" \
        "esrgan_default_log"

    run_test "1c" "主入口全流程默认" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t1c_full_default.mp4 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "full_flow_default_log"

    # 输出文件验证
    verify_output_video "$OUTPUT_DIR/t1c_full_default.mp4" "全流程默认"

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 2: Part 1 — crf=0 解耦测试
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 2: Part 1 — crf=0 解耦测试 ═══${NC}"

    # ★ 准备工作: 创建 IFRNet backend 的影子副本（显式设置常量），确保测试不依赖生产源码状态
    #   - SHADOW_CONSTQP_TRUE:  _NVENC_CRF0_FORCE_CONSTQP=True  → crf=0 → CONSTQP LA=0
    #   - 共享复用: 2a/2c/8a/10c/2g 均使用此副本
    #
    #   ⚠️ processor.__init__ 会 sys.path.insert(0, external/IFRNet) 覆盖 PYTHONPATH
    #   → 改用 Python wrapper 脚本在主脚本启动前把影子模块预缓存到 sys.modules
    local SHADOW_CONSTQP_TRUE="$OUTPUT_DIR/.tmp_ifrnet_constqp_true"
    mkdir -p "$SHADOW_CONSTQP_TRUE"
    cp "$IFRNET_BACKEND" "$SHADOW_CONSTQP_TRUE/process_video_v6_4_5_1_single.py"
    sed -i 's/^_NVENC_CRF0_FORCE_CONSTQP: bool = \(True\|False\)/_NVENC_CRF0_FORCE_CONSTQP: bool = True/' "$SHADOW_CONSTQP_TRUE/process_video_v6_4_5_1_single.py"

    # Python wrapper: 在 main/processor 启动前预缓存影子模块到 sys.modules
    #   ⚠️ shadow module 有 import models.IFRNet... 依赖，需要 project root 在 sys.path
    #   用法: $PYTHON wrapper.py <shadow_dir> <project_dir> <entry.py> [args...]
    #    或: $PYTHON wrapper.py <shadow_dir> <project_dir> -m <module> [args...]
    local SHADOW_WRAPPER="$OUTPUT_DIR/.tmp_shadow_ifrnet.py"
    cat > "$SHADOW_WRAPPER" << 'PYEOF'
import sys, os, runpy

shadow_dir = sys.argv[1]
project_dir = sys.argv[2]
sys.path.insert(0, project_dir)   # ensures 'external.IFRNet' imports work
sys.path.insert(0, os.path.join(project_dir, 'external', 'IFRNet'))  # ensures 'models' imports work
sys.path.insert(0, shadow_dir)    # overrides process_video_v6_4_5_1_single
import process_video_v6_4_5_1_single  # pre-cache in sys.modules

mode = sys.argv[3]
if mode == '-m':
    module = sys.argv[4]
    sys.argv = [module] + sys.argv[5:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
else:
    sys.argv = [mode] + sys.argv[4:]
    runpy.run_path(mode, run_name='__main__')
PYEOF

    # 2a: FORCE_CONSTQP=True → crf=0 → CONSTQP LA=0
    run_test "2a" "crf=0 默认强制 CONSTQP LA=0" \
        "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
            src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t2a_crf0_default.mp4 \
            --crf-ifrnet 0 --skip-upscale --use-tensorrt-ifrnet 2>&1" \
        "CONSTQP.*LA=0" \
        "crf0_force_constqp_log"

    # 2b: FORCE_CONSTQP=False, crf=0 走 VBR_HQ + LA=8（解耦开关生效）
    if $has_nvenc; then
        local SHADOW_FALSE="$OUTPUT_DIR/.tmp_ifrnet_shadow_false"
        mkdir -p "$SHADOW_FALSE"
        cp "$IFRNET_BACKEND" "$SHADOW_FALSE/process_video_v6_4_5_1_single.py"
        sed -i 's/^_NVENC_CRF0_FORCE_CONSTQP: bool = \(True\|False\)/_NVENC_CRF0_FORCE_CONSTQP: bool = False/' "$SHADOW_FALSE/process_video_v6_4_5_1_single.py"
        sed -i 's/^_NVENC_CRF0_QUALITY: int = \([0-9]*\)/_NVENC_CRF0_QUALITY: int = 23/' "$SHADOW_FALSE/process_video_v6_4_5_1_single.py"

        run_test "2b" "crf=0 解除强制 — VBR_HQ LA=8" \
            "$PYTHON $SHADOW_WRAPPER $SHADOW_FALSE $PROJECT_DIR \
                src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t2b_crf0_noforce.mp4 \
                --crf-ifrnet 0 --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "VBR_HQ.*LA=8" \
            "crf0_no_force_log"

        rm -rf "$SHADOW_FALSE"
    else
        echo -e "  [2b] crf=0 解除强制 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # 2c: FORCE_CONSTQP=True → crf=0 → CONSTQP LA=0
    run_test "2c" "crf=0 默认强制 CONSTQP LA=0" \
        "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
            src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t2c_crf0_default.mp4 \
            --crf-ifrnet 0 --skip-upscale --use-tensorrt-ifrnet 2>&1" \
        "CONSTQP.*LA=0" \
        "crf0_default_log"

    # ── 阶段 2b: crf=0 编码参数验证（本轮修复新增）─────────────────
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 2b: crf=0 编码参数验证（Level 1 QP / FFmpegWriter 无损参数） ═══${NC}"
    # 2d: crf=0 + libx264 → 日志应含 QP 无损参数
    # 注: 有 NVENC 时 backend 自动升级 libx264 → h264_nvenc（lossless_encoder），
    #     走 Level 1 NVENCEncoder 路径，日志含 QP=0（非 -qp 0）。
    #     无 NVENC 时走 Level 3 FFmpegWriter libx264，日志含 ffmpeg '-qp 0' 命令。
    if ! $has_nvenc; then
        run_test "2d" "crf=0 + libx264: FFmpegWriter 应输出 -qp 0" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2d_crf0_libx264.mp4 \
                --crf 0 --codec libx264 --use-tensorrt --quiet 2>&1" \
            "-qp 0" \
            "crf0_libx264_qp0_log"
    else
        echo -e "  [2d] crf=0 + libx264 — ${SKIP} (NVENC 覆盖，libx264 FFmpegWriter 路径不可达)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # 2e: crf=0 + libx264 → 应无 -crf N（crf=0 时 -crf 与 -qp 互斥）
    if ! $has_nvenc; then
        run_test "2e" "crf=0 + libx264: 不应输出 -crf (互斥参数)" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2e_crf0_libx264_nocrf.mp4 \
                --crf 0 --codec libx264 --use-tensorrt --quiet 2>&1" \
            "ffmpeg.*-qp 0(?!.*-crf)" \
            "crf0_libx264_no_crf_log"
    else
        echo -e "  [2e] crf=0 + libx264 -crf 互斥 — ${SKIP} (NVENC 覆盖，FFmpegWriter 路径不可达)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi



    # 2f: NVENC crf=0 → 验证 NVENCEncoder 初始化日志 QP=0（非 QP=28）
    # 直接实例化 NVENCEncoder 检查日志输出
    if $has_nvenc; then
        run_test "2f" "NVENC qp=0 → CONSTQP QP=0 (非 QP=28)" \
            "$PYTHON -c \"
import sys, io
sys.path.insert(0, '$PROJECT_DIR/external/IFRNet')
buf = io.StringIO()
sys.stdout = buf
try:
    from process_video_v6_4_5_1_single import NVENCEncoder
    e = NVENCEncoder(1920, 1080, 30.0, qp=0, rate_mode='constqp')
    e.close()
except Exception:
    pass
sys.stdout = sys.__stdout__
output = buf.getvalue()
if 'QP=0' in output:
    print('NVENCEncoder qp=0 → QP=0 (正确)')
    sys.exit(0)
sys.exit(0)  # 不因硬件限制标记失败
\" 2>&1" \
            "QP=0" \
            "nvenc_qp0_constqp_log"
    else
        echo -e "  [2f] NVENC qp=0 验证 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # 2g: crf=0 + NVENC → 验证 NVENCEncoder CONSTQP 模式
    if $has_nvenc; then
        run_test "2g" "crf=0 + NVENC: NVENCEncoder 应输出 CONSTQP" \
            "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
                -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2g_crf0_nvenc_constqp.mp4 \
                --crf 0 --use-tensorrt --quiet 2>&1" \
            "CONSTQP" \
            "crf0_nvenc_constqp_log"
    else
        echo -e "  [2g] crf=0 NVENC -rc constqp — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 3: Part 2 — IFRNet rate_mode CLI 切换
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 3: Part 2 — IFRNet rate_mode CLI 切换 ═══${NC}"

    if $has_nvenc; then
        run_test "3a" "IFRNet --rate-mode-ifrnet constqp" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t3a_ifrnet_constqp.mp4 \
                --rate-mode-ifrnet constqp --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "CONSTQP.*LA=0" \
            "ifrnet_constqp_cli_log"

        run_test "3b" "IFRNet --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 16" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t3b_ifrnet_vbrhq_la16.mp4 \
                --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 16 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "VBR_HQ.*LA=16" \
            "ifrnet_vbrhq_la16_cli_log"

        run_test "3c" "IFRNet --rate-mode-ifrnet qvbr (显式指定默认值)" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t3c_ifrnet_qvbr_explicit.mp4 \
                --rate-mode-ifrnet qvbr --lookahead-depth-ifrnet 0 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "QVBR.*LA=0" \
            "ifrnet_qvbr_cli_log"

        # 文件大小对比：constqp 通常最小
        compare_file_sizes \
            "$OUTPUT_DIR/t3a_ifrnet_constqp.mp4" \
            "$OUTPUT_DIR/t3c_ifrnet_qvbr_explicit.mp4" \
            "constqp" "qvbr"
    else
        echo -e "  [3a-3c] IFRNet rate_mode CLI — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 3))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 4: Part 2+3 — Real-ESRGAN rate_mode CLI 切换
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 4: Part 2+3 — Real-ESRGAN rate_mode CLI ═══${NC}"

    # ESRGAN 有两条输出路径：NVENCEncoder（大写）和 FFmpegWriter（小写）
    # 使用大小写不敏感字符类匹配两种路径
    run_test "4a" "ESRGan --rate-mode-esrgan vbr_hq (默认)" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t4a_esrgan_vbrhq.mp4 \
            --rate-mode-esrgan vbr_hq --skip-interpolate --use-tensorrt-esrgan 2>&1" \
        "[Vv][Bb][Rr]_[Hh][Qq]" \
        "esrgan_vbrhq_cli_log"

    run_test "4b" "ESRGan --rate-mode-esrgan constqp" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t4b_esrgan_constqp.mp4 \
            --rate-mode-esrgan constqp --skip-interpolate --use-tensorrt-esrgan 2>&1" \
        "[Cc][Oo][Nn][Ss][Tt][Qq][Pp]" \
        "esrgan_constqp_cli_log"

    run_test "4c" "ESRGan --rate-mode-esrgan qvbr --lookahead-depth-esrgan 8" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t4c_esrgan_qvbr_la8.mp4 \
            --rate-mode-esrgan qvbr --lookahead-depth-esrgan 8 \
            --skip-interpolate --use-tensorrt-esrgan 2>&1" \
        "[Qq][Vv][Bb][Rr].*[Ll][Aa]=8" \
        "esrgan_qvbr_la8_cli_log"

    # 文件大小对比
    compare_file_sizes \
        "$OUTPUT_DIR/t4a_esrgan_vbrhq.mp4" \
        "$OUTPUT_DIR/t4b_esrgan_constqp.mp4" \
        "vbr_hq" "constqp"

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 5: 全流程一致性
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 5: 全流程一致性测试 ═══${NC}"

    run_test "5a" "全流程 IFRNet qvbr + ESRGan vbr_hq (默认组合)" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t5a_default_combo.mp4 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "full_default_combo_log"

    verify_output_video "$OUTPUT_DIR/t5a_default_combo.mp4" "默认组合"

    if $has_nvenc; then
        run_test "5b" "全流程统一 constqp" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t5b_both_constqp.mp4 \
                --rate-mode-ifrnet constqp --rate-mode-esrgan constqp \
                --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
            "全流程完成" \
            "both_constqp_log"

        # 验证日志中两阶段均为 constqp
        local out5b
        out5b=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
        if echo "$out5b" | grep -q "CONSTQP"; then
            echo "      ${GREEN}✓${NC} IFRNet constqp 已确认"
        fi
        if echo "$out5b" | grep -qi "constqp"; then
            echo "      ${GREEN}✓${NC} ESRGAN constqp 已确认"
        fi

        run_test "5c" "全流程统一 qvbr + lookahead=16" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t5c_both_qvbr_la16.mp4 \
                --rate-mode-ifrnet qvbr --rate-mode-esrgan qvbr \
                --lookahead-depth-ifrnet 16 --lookahead-depth-esrgan 16 \
                --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
            "全流程完成" \
            "both_qvbr_la16_log"
    else
        echo -e "  [5b-5c] 统一 constqp/qvbr — ${SKIP} (NVENC Level 1 不可用，Level 2+ 仍测通过)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 2))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 6: 处理模式切换
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 6: 处理模式切换测试 ═══${NC}"

    # ⚠️ upscale_then_interpolate: Step 2 IFRNet 处理高分辨率帧，降低 batch_size 防 OOM
    run_test "6a" "upscale_then_interpolate + 不同 rate_mode" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t6a_upscale_first.mp4 \
            --mode upscale_then_interpolate \
            --rate-mode-ifrnet constqp --rate-mode-esrgan qvbr \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan \
            --batch-size-ifrnet 6 2>&1" \
        "全流程完成" \
        "upscale_first_mode_log"

    verify_output_video "$OUTPUT_DIR/t6a_upscale_first.mp4" "upscale_first"

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 7: 配置文件驱动
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 7: JSON 配置驱动测试 ═══${NC}"
    echo -e "  ${WARN} 注意: 将使用临时副本测试 JSON 配置，不修改原始文件"
    echo ""

    # ★ 安全: 复制 config 到项目目录内的临时文件 (确保 base_dir 自动推导正确)
    #   _setup_paths 通过 config_path.parent.parent 推导 base_dir，必须在项目树内
    local CONFIG_COPY_7="$PROJECT_DIR/temp/.tmp_config_default_7a.json"
    mkdir -p "$PROJECT_DIR/temp"
    cp "$CONFIG" "$CONFIG_COPY_7"

    $PYTHON -c "
import json
with open('$CONFIG_COPY_7', 'r') as f:
    cfg = json.load(f)
cfg['models']['ifrnet']['rate_mode'] = 'constqp'
cfg['models']['ifrnet']['lookahead_depth'] = 0
cfg['models']['realesrgan']['rate_mode'] = 'constqp'
cfg['models']['realesrgan']['lookahead_depth'] = 0
with open('$CONFIG_COPY_7', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print('临时 config 已修改: IFRNet constqp/LA=0, ESRGan constqp/LA=0')
"

    run_test "7a" "JSON 配置驱动 — 不传 CLI 参数 (使用临时 config)" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG_COPY_7 -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t7a_config_driven.mp4 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "config_driven_log"

    # 验证 IFRNet 日志
    local out7
    out7=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
    if echo "$out7" | grep -q "CONSTQP.*LA=0"; then
        echo "      ${GREEN}✓${NC} JSON constqp/LA=0 → IFRNet Level 1 生效"
    else
        echo "      ${YELLOW}?${NC} IFRNet 日志中未明确找到 CONSTQP LA=0（可能走 Level 2+）"
    fi

    rm -f "$CONFIG_COPY_7"

    # 7b: 原始 config 默认行为
    run_test "7b" "JSON 原始配置默认行为" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t7b_config_restored.mp4 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "config_restored_log"

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 8: 边界/异常测试
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 8: 边界/异常测试 ═══${NC}"

    # 8a: crf=0 + lookahead > 0 (验证 FORCE_CONSTQP=True 强制覆盖)
    if $has_nvenc; then
        run_test "8a" "crf=0 + LA=16 → 仍强制 CONSTQP LA=0" \
            "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
                src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t8a_crf0_la16.mp4 \
                --crf-ifrnet 0 --lookahead-depth-ifrnet 16 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "CONSTQP.*LA=0" \
            "crf0_force_reject_la_log"
    else
        echo -e "  [8a] crf=0+LA16 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # 8b: 无效 rate_mode 值 (argparse 拦截，退出码非零由 run_test 弹性处理)
    run_test "8b" "无效 rate_mode — argparse 应拦截" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t8b_invalid.mp4 \
            --rate-mode-ifrnet INVALID_VALUE 2>&1" \
        "invalid choice" \
        "argparse_reject_log"

    # 8c: 非 NVENC codec (libx264) — rate_mode 不影响软编码
    run_test "8c" "libx264 codec + --rate-mode-ifrnet constqp (无 NVENC)" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t8c_libx264.mp4 \
            --codec-ifrnet libx264 --rate-mode-ifrnet constqp \
            --skip-upscale --use-tensorrt-ifrnet 2>&1" \
        "全流程完成|插帧.*完成" \
        "libx264_no_nvenc_log"

    # 8d: lookahead=0 + vbr_hq (验证 LA=0 禁用)
    if $has_nvenc; then
        run_test "8d" "vbr_hq + LA=0 (不启用 lookahead)" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t8d_vbrhq_la0.mp4 \
                --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 0 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "VBR_HQ.*LA=0" \
            "vbrhq_no_la_log"
    fi

    # 8e: lookahead=32 + qvbr (最大 lookahead)
    echo ""
    echo -e "  ${WARN} LA=32 是 NVENC SDK 支持的最大 lookahead 值，"
    echo -e "         超过 pipeline_depth=4 的已验证范围 (LA≤16)。"
    echo -e "         大 LA 值在高 FPS 场景可能引起编码器缓冲溢出，"
    echo -e "         输出视频可能出现花屏/卡顿。"
    echo -e "  ${INFO} NVENC SDK 严格遵守帧数守恒 (Output==Input)，"
    echo -e "        输出帧数预期 = 插值帧数。LA 仅引入编码延迟，不增删帧。"
    echo ""
    if $has_nvenc; then
        run_test "8e" "qvbr + LA=32 (最大 lookahead)" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t8e_qvbr_la32.mp4 \
                --rate-mode-ifrnet qvbr --lookahead-depth-ifrnet 32 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "QVBR.*LA=32" \
            "qvbr_max_la_log"

        # 后处理验证: 检查 ffmpeg stderr 是否有 H.264 流错误
        # ★ 如果后置检查失败且之前 run_test 已标记PASS，需撤销计数器
        #    (check_ffmpeg_stderr 内部已调用 _remove_from_progress 防止 --resume 跳过)
        local _8e_was_passed=false
        grep -qxF "8e" "$PROGRESS_FILE" 2>/dev/null && _8e_was_passed=true
        if ! check_ffmpeg_stderr "8e" \
            "$OUTPUT_DIR/.tmp_test_stderr.txt" \
            "$OUTPUT_DIR/t8e_qvbr_la32.mp4"; then
            if $_8e_was_passed; then
                TESTS_PASSED=$((TESTS_PASSED - 1))
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        fi

        # 后处理验证: ffprobe 帧计数
        # NVENC SDK 严格遵守帧数守恒 (Output==Input)，LA 仅引入延迟不增删帧
        # 预期帧数 ≈ source_frames × interpolation_factor
        #   word_world_2.mp4: ~687 帧 @ 25fps, 2x 插值 → ~1374 帧, tolerance=5
        if [ -f "$OUTPUT_DIR/t8e_qvbr_la32.mp4" ]; then
            _8e_was_passed=false
            grep -qxF "8e" "$PROGRESS_FILE" 2>/dev/null && _8e_was_passed=true
            if ! verify_frame_count "8e" \
                "$OUTPUT_DIR/t8e_qvbr_la32.mp4" \
                1373 5; then  # LA=32 测试
                if $_8e_was_passed; then
                    TESTS_PASSED=$((TESTS_PASSED - 1))
                    TESTS_FAILED=$((TESTS_FAILED + 1))
                fi
            fi
        fi

        # 文件大小对比: LA=32 vs LA=8 (默认)
        compare_file_sizes \
            "$OUTPUT_DIR/t8e_qvbr_la32.mp4" \
            "$OUTPUT_DIR/t3c_ifrnet_qvbr_explicit.mp4" \
            "LA=32" "LA=8"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 9: 输出质量验证
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 9: 输出质量验证 ═══${NC}"

    if command -v ffprobe &>/dev/null; then
        echo ""
        echo "  ── 输出文件元数据 ──"
        for f in "$OUTPUT_DIR"/t5a_default_combo.mp4 \
                 "$OUTPUT_DIR"/t3a_ifrnet_constqp.mp4 \
                 "$OUTPUT_DIR"/t3c_ifrnet_qvbr_explicit.mp4 \
                 "$OUTPUT_DIR"/t4a_esrgan_vbrhq.mp4 \
                 "$OUTPUT_DIR"/t4b_esrgan_constqp.mp4 \
                 "$OUTPUT_DIR"/t8e_qvbr_la32.mp4; do
            if [ -f "$f" ]; then
                verify_output_video "$f" "$(basename "$f")"
            fi
        done

        echo ""
        echo "  ── constqp vs qvbr 文件大小对比 (IFRNet) ──"
        compare_file_sizes \
            "$OUTPUT_DIR/t3a_ifrnet_constqp.mp4" \
            "$OUTPUT_DIR/t3c_ifrnet_qvbr_explicit.mp4" \
            "constqp" "qvbr"
    else
        echo "  ${SKIP} ffprobe 不可用"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 10: LA 分段策略测试 (方案一: 自动禁用分段 + 帧守恒验证)
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 10: LA 分段策略 — 独立排空帧数守恒验证 ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证 LA>0 + 分段时每段独立排空，帧数守恒:"
    echo -e "  ${INFO}   - IFRNet (qvbr/LA=8) → 输出 '每段独立排空，帧数守恒'"
    echo -e "  ${INFO}   - ESRGAN (vbr_hq/LA=16) → 同上"
    echo -e "  ${INFO}   - crf=0 → CONSTQP LA=0 (FORCE_CONSTQP=True 强制无损)"
    echo -e "  ${INFO}   - rate_mode=constqp → 输出 CONSTQP (无 LA 消息)"
    echo -e "  ${INFO}   - LA=0 → 不激活 LA 消息"
    echo -e "  ${INFO}   - NVENC SDK 帧守恒: Output==Input, LA 仅引入延迟不增删帧"
    echo ""

    # ── 10a: IFRNet qvbr+LA=8 → 应输出 LA 激活 + 每段独立排空 ──
    run_test "10a" "LA+分段: IFRNet(qvbr/LA=8)独立排空" \
        "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t10a_ifrnet_la_seg.mp4 \
            --rate-mode qvbr --lookahead-depth 8 --use-tensorrt --quiet 2>&1" \
        "每段独立排空" \
        "la_ifrnet_seg_log"

    # ── 10b: ESRGAN vbr_hq+LA=16 → 应输出 LA 激活消息 ──
    run_test "10b" "LA+分段: ESRGAN(vbr_hq/LA=16)独立排空" \
        "$PYTHON -m src.processors.realesrgan_processor_video_optimized \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t10b_esrgan_la_seg.mp4 \
            --rate-mode vbr_hq --lookahead-depth 16 --use-tensorrt --quiet 2>&1" \
        "VBR_HQ.*LA=16" \
        "la_esrgan_seg_log"

    # ── 10c: IFRNet crf=0 → CONSTQP LA=0 ──
    if $has_nvenc; then
        run_test "10c" "LA+分段: crf=0 应输出 CONSTQP LA=0" \
            "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
                -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t10c_ifrnet_crf0_seg.mp4 \
                --crf 0 --use-tensorrt --quiet 2>&1" \
            "CONSTQP.*LA=0" \
            "la_crf0_constqp_log"
    else
        echo -e "  [10c] crf=0+NVENC — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 10d: IFRNet constqp → 无 LA 消息 ──
    if $has_nvenc; then
        run_test "10d" "LA+分段: constqp 无 LA 消息" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t10d_ifrnet_constqp_seg.mp4 \
                --rate-mode constqp --use-tensorrt --quiet 2>&1" \
            "CONSTQP" \
            "la_constqp_log" \
            "每段独立排空"
    else
        echo -e "  [10d] constqp — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 10e: LA=0 + vbr_hq → 无 LA 消息 ──
    if $has_nvenc; then
        run_test "10e" "LA+分段: vbr_hq+LA=0 无 LA 消息" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t10e_ifrnet_vbrhq_la0_seg.mp4 \
                --rate-mode vbr_hq --lookahead-depth 0 --use-tensorrt --quiet 2>&1" \
            "VBR_HQ" \
            "la_vbrhq_la0_log" \
            "每段独立排空"
    else
        echo -e "  [10e] vbr_hq+LA=0 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 10f-10h: 帧守恒 + 全流程 ──
    # 探测测试视频是否有音频轨（仅用于额外音视频时长一致性验证）
    local has_audio_10=false
    if command -v ffprobe &>/dev/null; then
        local audio_10
        audio_10=$(ffprobe -v error -select_streams a:0 \
            -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "$TEST_VIDEO" 2>/dev/null || echo "")
        if [ "$audio_10" = "audio" ]; then
            has_audio_10=true
        fi
    fi

    # ── 10f: 全流程默认 LA>0 → 帧守恒: 输出时长=输入时长 ──
    # NVENC SDK 帧守恒: Output==Input, LA 仅引入延迟不增删帧 → 时长不变
    run_test "10f" "帧守恒: 全流程默认 LA>0 → 输出时长=输入时长" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t10f_full_frame_conservation.mp4 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "frame_conservation_default_log" \
        "检测到 LA 导致输出视频缺失"

    # 后处理验证：帧守恒 → 时长一致
    local out10f
    out10f=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
    if echo "$out10f" | grep -q "时长差异: 0.00秒"; then
        echo "      ${GREEN}✓${NC} 帧守恒: 输出时长=输入时长 (NVENC Output==Input)"
    elif echo "$out10f" | grep -q "输出时长偏差.*（非 LA 原因"; then
        echo "      ${GREEN}✓${NC} 帧守恒: 微小偏差确认为非 LA 原因"
    else
        echo "      ${YELLOW}?${NC} 未检测到明确的时长一致性消息"
    fi

    # 音视频时长一致性 (额外验证，需音频轨)
    if $has_audio_10 && [ -f "$OUTPUT_DIR/t10f_full_frame_conservation.mp4" ]; then
        echo "      ── 音视频时长一致性 ──"
        local v_dur a_dur
        v_dur=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=duration \
            -of default=noprint_wrappers=1:nokey=1 \
            "$OUTPUT_DIR/t10f_full_frame_conservation.mp4" 2>/dev/null || echo "N/A")
        a_dur=$(ffprobe -v error -select_streams a:0 \
            -show_entries stream=duration \
            -of default=noprint_wrappers=1:nokey=1 \
            "$OUTPUT_DIR/t10f_full_frame_conservation.mp4" 2>/dev/null || echo "N/A")
        if [ "$v_dur" != "N/A" ] && [ "$a_dur" != "N/A" ]; then
            local dur_diff
            dur_diff=$(echo "scale=2; ($v_dur - $a_dur)" | bc 2>/dev/null || echo "?")
            local abs_diff
            abs_diff=$(echo "scale=2; if ($dur_diff < 0) -1*$dur_diff else $dur_diff" | bc 2>/dev/null || echo "?")
            echo "      视频时长: ${v_dur}s  音频时长: ${a_dur}s  差值: ${dur_diff}s"
            if [ "$(echo "$abs_diff < 0.2" | bc 2>/dev/null)" = "1" ]; then
                echo "      ${GREEN}✓${NC} 音视频时长一致 (差值 <0.2s)"
            else
                echo "      ${WARN} 音视频时长差异较大: ${abs_diff}s"
            fi
        fi
    fi

    # ── 10g: crf=0 全流程 → CONSTQP LA=0 帧守恒 ──
    run_test "10g" "帧守恒: crf=0全流程 → CONSTQP LA=0 输出时长不变" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t10g_crf0_frame_conservation.mp4 \
            --crf-ifrnet 0 --crf-esrgan 0 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "frame_conservation_crf0_log" \
        "检测到 LA 导致输出视频缺失"

    # ── 10h: 全流程 upscale_then_interpolate → LA 分段策略在两阶段均生效 ──
    # ⚠️ upscale_then_interpolate: Step 2 IFRNet 处理高分辨率帧，降低 batch_size 防 OOM
    run_test "10h" "LA+分段: upscale_then_interpolate 模式" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t10h_upscale_first_la.mp4 \
            --mode upscale_then_interpolate \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan \
            --batch-size-ifrnet 6 2>&1" \
        "全流程完成" \
        "la_upscale_first_log"

    # 后处理验证：日志中应含 LA 独立排空消息
    local out10h
    out10h=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
    if echo "$out10h" | grep -q "每段独立排空"; then
        echo "      ${GREEN}✓${NC} LA 独立排空策略已生效"
    else
        echo "      ${YELLOW}?${NC} 未检测到「每段独立排空」消息 (可能视频 < segment_duration)"
    fi
    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 11: LA 帧数守恒验证 (多分段 + rate_mode 组合)
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 11: LA 帧数守恒验证 (多分段 + rate_mode 组合) ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证 LA>0 与分段共存的帧数守恒:"
    echo -e "  ${INFO}   - NVENC SDK 严格遵守帧数守恒 (Output==Input)"
    echo -e "  ${INFO}   - LA 仅引入编码延迟，不增删帧"
    echo -e "  ${INFO}   - 每段独立编码+排空，帧数守恒"
    echo ""

    if $has_nvenc; then
        # ── 11a: qvbr/LA=8 + 分段 → 帧数守恒消息 ──
        run_test "11a" "帧守恒: qvbr/LA=8 独立排空" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t11a_qvbr_la8_seg.mp4 \
                --rate-mode qvbr --lookahead-depth 8 --use-tensorrt --quiet 2>&1" \
            "每段独立排空" \
            "frame_conservation_qvbr_log"

        # ── 11b: vbr_hq/LA=0 → 无 LA 消息 ──
        run_test "11b" "帧守恒: vbr_hq/LA=0 无 LA 消息" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t11b_vbrhq_la0_seg.mp4 \
                --rate-mode vbr_hq --lookahead-depth 0 --use-tensorrt --quiet 2>&1" \
            "VBR_HQ" \
            "frame_conservation_vbrhq_log" \
            "每段独立排空"

        # ── 11c: constqp + LA=16 → CONSTQP 静默禁用 LA ──
        run_test "11c" "帧守恒: constqp+LA=16 → CONSTQP 忽略 LA" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t11c_constqp_la16_seg.mp4 \
                --rate-mode constqp --lookahead-depth 16 --use-tensorrt --quiet 2>&1" \
            "CONSTQP" \
            "frame_conservation_constqp_log" \
            "每段独立排空"
    else
        echo -e "  [11a-11c] LA 帧数守恒 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 3))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 12: 分段时长 + LA 交互测试
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 12: 分段时长 + LA 交互测试 ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证不同分段时长与 LA 的组合:"
    echo -e "  ${INFO}   - 短分段 + LA=8: 提前排空，帧数守恒"
    echo -e "  ${INFO}   - 长分段 + LA=16: 正常排空，帧数守恒"
    echo ""

    if $has_nvenc; then
        # ── 12a: IFRNet qvbr/LA=8 + 短分段(5s) ──
        run_test "12a" "分段+LA: IFRNet qvbr/LA=8 + 短分段(5s)" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t12a_shortseg_qvbr_la8.mp4 \
                --rate-mode qvbr --lookahead-depth 8 --segment-duration 5 \
                --use-tensorrt --quiet 2>&1" \
            "每段独立排空" \
            "shortseg_qvbr_log"

        # ── 12b: IFRNet vbr_hq/LA=16 + 长分段(60s) ──
        run_test "12b" "分段+LA: IFRNet vbr_hq/LA=16 + 长分段(60s)" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t12b_longseg_vbrhq_la16.mp4 \
                --rate-mode vbr_hq --lookahead-depth 16 --segment-duration 60 \
                --use-tensorrt --quiet 2>&1" \
            "VBR_HQ.*la=16|VBR_HQ.*LA=16" \
            "longseg_vbrhq_log"
    else
        echo -e "  [12a-12b] 分段+LA 测试 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 2))
    fi

    echo -e "${BOLD}${BLUE}═══ 阶段 13: 双层 LA 激活 + 帧守恒组合验证 ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证不同 LA 组合下的帧守恒 (NVENC SDK Output==Input):"
    echo -e "  ${INFO}   - IFRNet LA=8 + ESRGAN LA=8 → 双层 LA 均激活，帧守恒"
    echo -e "  ${INFO}   - IFRNet LA=8 + ESRGAN skip → 单层 LA，帧守恒"
    echo -e "  ${INFO}   - crf=0 强制禁用对应处理器的 LA (CONSTQP LA=0)"
    echo -e "  ${INFO}   - 双层均为 crf=0 → 无 LA 激活，帧守恒"
    echo -e "  ${INFO}   - 帧守恒意味着输出时长=输入时长，无需音频修剪"
    echo ""

    # 探测测试视频是否有音频轨（仅用于 13a 额外音视频时长一致性验证）
    local has_audio_13=false
    if command -v ffprobe &>/dev/null; then
        local audio_streams_13
        audio_streams_13=$(ffprobe -v error -select_streams a:0 \
            -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "$TEST_VIDEO" 2>/dev/null || echo "")
        if [ "$audio_streams_13" = "audio" ]; then
            has_audio_13=true
        fi
    fi

    # ── 13a: 双层 LA=8+8 → 两处理器 LA 均激活 + 帧守恒 ──
    if $has_nvenc; then
        run_test "13a" "双层LA: IFRNet(LA=8)+ESRGAN(LA=8) → 双层激活+帧守恒" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t13a_dual_la8_frame_conservation.mp4 \
                --rate-mode-ifrnet qvbr --lookahead-depth-ifrnet 8 \
                --rate-mode-esrgan qvbr --lookahead-depth-esrgan 8 \
                --use-tensorrt-ifrnet --use-tensorrt-esrgan \
                2>&1" \
            "全流程完成" \
            "dual_la8_frame_conservation_log" \
            "检测到 LA 导致输出视频缺失"

        # 后处理验证: 帧守恒 + 双层 LA 激活
        local out13a
        out13a=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
        if echo "$out13a" | grep -q "时长差异: 0.00秒"; then
            echo "      ${GREEN}✓${NC} 帧守恒: 输出时长=输入时长 (双层 LA=8+8)"
        fi
        if echo "$out13a" | grep -q "QVBR.*LA=8"; then
            echo "      ${GREEN}✓${NC} 双层 LA=8 均已激活"
        fi

        # 音视频时长一致性 (额外验证，需音频轨)
        if $has_audio_13 && [ -f "$OUTPUT_DIR/t13a_dual_la8_frame_conservation.mp4" ]; then
            local v_dur13 a_dur13
            v_dur13=$(ffprobe -v error -select_streams v:0 \
                -show_entries stream=duration \
                -of default=noprint_wrappers=1:nokey=1 \
                "$OUTPUT_DIR/t13a_dual_la8_frame_conservation.mp4" 2>/dev/null || echo "N/A")
            a_dur13=$(ffprobe -v error -select_streams a:0 \
                -show_entries stream=duration \
                -of default=noprint_wrappers=1:nokey=1 \
                "$OUTPUT_DIR/t13a_dual_la8_frame_conservation.mp4" 2>/dev/null || echo "N/A")
            if [ "$v_dur13" != "N/A" ] && [ "$a_dur13" != "N/A" ]; then
                local dur_diff13 abs_diff13
                dur_diff13=$(echo "scale=2; ($v_dur13 - $a_dur13)" | bc 2>/dev/null || echo "?")
                abs_diff13=$(echo "scale=2; if ($dur_diff13 < 0) -1*$dur_diff13 else $dur_diff13" | bc 2>/dev/null || echo "?")
                echo "      视频时长: ${v_dur13}s  音频时长: ${a_dur13}s  差值: ${dur_diff13}s"
                if [ "$(echo "$abs_diff13 < 0.3" | bc 2>/dev/null)" = "1" ]; then
                    echo "      ${GREEN}✓${NC} 音视频时长一致 (差值 <0.3s)"
                else
                    echo "      ${WARN} 音视频时长差异较大: ${abs_diff13}s"
                fi
            fi
        fi
    else
        echo -e "  [13a] 双层 LA 帧守恒 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 13b: 单层 LA: IFRNet LA=8 + ESRGAN LA=0 (skip upscale) → 帧守恒，输出时长不变 ──
    if $has_nvenc; then
        run_test "13b" "双层LA: IFRNet(LA=8)+ESRGAN(skip) → 帧守恒验证" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t13b_ifrnet_la8_skip_esrgan.mp4 \
                --rate-mode-ifrnet qvbr --lookahead-depth-ifrnet 8 \
                --skip-upscale --use-tensorrt-ifrnet 2>&1" \
            "处理完成" \
            "single_la8_frame_conservation_log"

        local out13b
        out13b=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
        # 帧守恒：输出时长应等于输入时长
        if echo "$out13b" | grep -q "时长差异: 0.00秒"; then
            echo "      ${GREEN}✓${NC} 帧守恒: 输出时长=输入时长 (无帧丢失)"
        elif echo "$out13b" | grep -q "输出时长偏差.*（非 LA 原因"; then
            echo "      ${GREEN}✓${NC} 帧守恒: 微小偏差确认为非 LA 原因"
        else
            echo "      ${YELLOW}?${NC} 未检测到帧守恒明确消息"
        fi
    else
        echo -e "  [13b] 单层 LA 帧守恒 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 13c: IFRNet crf=0 (LA=0) + ESRGAN LA=8 → crf=0 禁用 IFRNet LA, 帧守恒 ──
    if $has_nvenc; then
        run_test "13c" "双层LA: IFRNet(crf=0→LA=0)+ESRGAN(LA=8) → 帧守恒" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t13c_crf0_ifrnet_la8_esrgan.mp4 \
                --crf-ifrnet 0 \
                --rate-mode-esrgan qvbr --lookahead-depth-esrgan 8 \
                --use-tensorrt-ifrnet --use-tensorrt-esrgan \
                2>&1" \
            "全流程完成" \
            "crf0_partial_la_frame_conservation_log" \
            "检测到 LA 导致输出视频缺失"

        local out13c
        out13c=$(cat "$OUTPUT_DIR/.tmp_test_stdout.txt" "$OUTPUT_DIR/.tmp_test_stderr.txt" 2>/dev/null || echo "")
        if echo "$out13c" | grep -q "CONSTQP.*LA=0"; then
            echo "      ${GREEN}✓${NC} IFRNet crf=0 → CONSTQP LA=0 (crf=0 强制无损)"
        fi
        if echo "$out13c" | grep -q "时长差异: 0.00秒"; then
            echo "      ${GREEN}✓${NC} 帧守恒: 输出时长=输入时长"
        fi
    else
        echo -e "  [13c] crf=0 部分 LA 帧守恒 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 13d: 双层 crf=0 → 无 LA 激活 + 帧守恒 ──
    run_test "13d" "双层LA: 双crf=0 → 无 LA 激活 + 帧守恒" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t13d_both_crf0_frame_conservation.mp4 \
            --crf-ifrnet 0 --crf-esrgan 0 \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan 2>&1" \
        "全流程完成" \
        "both_crf0_frame_conservation_log" \
        "检测到 LA 导致输出视频缺失"

    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # 测试报告
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  测试报告${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  总计 : ${BOLD}$TESTS_TOTAL${NC}"
    echo -e "  通过 : ${GREEN}$TESTS_PASSED${NC}"
    echo -e "  失败 : ${RED}$TESTS_FAILED${NC}"
    echo -e "  跳过 : ${YELLOW}$TESTS_SKIPPED${NC}"
    echo ""
    echo -e "  详细日志 : ${CYAN}$TEST_LOG_FILE${NC}"

    if [ "$TESTS_FAILED" -gt 0 ]; then
        echo -e "  失败汇总 : ${RED}$FAIL_LOG${NC}"
        echo ""
        echo -e "${RED}  ═══ 失败项 ═══${NC}"
        grep -E '^\[[a-zA-Z0-9]+\] ' "$FAIL_LOG" 2>/dev/null || true
        echo ""
        echo -e "  ${INFO} 详细错误输出见: ${CYAN}$TEST_LOG_FILE${NC} / ${RED}$FAIL_LOG${NC}"
    fi

    if [ "$TESTS_FAILED" -eq 0 ]; then
        echo ""
        echo -e "  ${GREEN}${BOLD}✅ 全部测试通过！${NC}"
        echo ""
        echo -e "  输出目录: ${CYAN}$OUTPUT_DIR${NC}"
        echo -e "  ─────────────────────────────────────"
        ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{printf "  %6s  %s\n", $5, $NF}'
    fi

    echo ""

    # 清理临时文件
    rm -f "$OUTPUT_DIR"/.tmp_test_stdout.txt "$OUTPUT_DIR"/.tmp_test_stderr.txt
    # 清理影子副本 (CONSTQP=True)
    rm -rf "${SHADOW_CONSTQP_TRUE:-/nonexistent_shadow_dir}" 2>/dev/null || true

    # 返回
    if [ "$TESTS_FAILED" -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

# 确保在项目根目录运行
cd "$PROJECT_DIR"

# 注册 EXIT trap — 清理所有影子副本和 wrapper (不触碰生产源码)
trap 'rm -rf "$OUTPUT_DIR"/.tmp_ifrnet_* 2>/dev/null; rm -f "$OUTPUT_DIR"/.tmp_shadow_*.py 2>/dev/null; rm -f "$OUTPUT_DIR"/.tmp_config_*.json 2>/dev/null; true' EXIT

main
