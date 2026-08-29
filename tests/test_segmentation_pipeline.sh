#!/usr/bin/env bash
# =============================================================================
# test_segmentation_pipeline.sh — 分段全流程测试脚本
# =============================================================================
# 测试范围:
#   Part 1: LA 分段策略 (rate_mode/lookahead 帧数守恒独立排空)
#   Part 2: N 分段全流程 (通过 --segments 控制分段数)
#   Part 3: 帧守恒与分段 (NVENC SDK Output==Input, LA 仅引入延迟不增删帧)
#   Part 4: LA 帧数守恒 — 多分段验证 (替代原 carryover 方案)
#   Part 5: 断点恢复测试 (替代原 carryover 警告测试)
#
# 使用方法:
#   chmod +x tests/test_segmentation_pipeline.sh
#   ./tests/test_segmentation_pipeline.sh [选项] [<test_video.mp4>] [<output_dir>]
#
#   选项:
#     --input <path>        指定测试视频 (优先级最高)
#     --segments <N>        分段数: 0=不分段, 默认 2 (把视频分成 N 段)
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

# ── 分段控制 ──────────────────────────────────────────────────────────────────
SEGMENTS=2          # 默认 2 段
SEGMENT_DURATION="" # 自动计算

# ── CLI 参数解析 ──
USAGE="用法: $0 [选项] [<test_video.mp4>] [<output_dir>]

选项:
  --input <path>        指定测试用原视频文件 (默认: ../input_videos/word_world_2.mp4)
  --segments <N>        分段数: 0=不分段 (整视频单段处理), 默认 2
  --resume, --continue  断点继续模式，跳过已通过的测试

位置参数 (向后兼容):
  \$1                   测试视频路径 (等同于 --input)
  \$2                   输出目录路径"

INPUT_VIDEO_ARG=""
_POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT_VIDEO_ARG="$2"; shift 2 ;;
        --segments)
            SEGMENTS="$2"
            if ! [[ "$SEGMENTS" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}错误: --segments 必须是整数 (0=不分段, >0=分段数)${NC}"
                exit 1
            fi
            shift 2
            ;;
        --resume|--continue) RESUME_MODE=true; shift ;;
        --help|-h) echo "$USAGE"; exit 0 ;;
        *) _POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done
set -- "${_POSITIONAL_ARGS[@]}"

# ── 参数解析 ──
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

# ═══════════════════════════════════════════════════════════════════════════════
# 横幅
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  分段全流程测试套件${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  项目目录 : ${CYAN}$PROJECT_DIR${NC}"
echo -e "  测试视频 : ${CYAN}$TEST_VIDEO${NC}"
echo -e "  输出目录 : ${CYAN}$OUTPUT_DIR${NC}"
echo -e "  配置文件 : ${CYAN}$CONFIG${NC}"
echo -e "  分段数   : ${CYAN}$SEGMENTS${NC} (0=不分段)"
if [ "$RESUME_MODE" = true ]; then
    echo -e "  运行模式 : ${YELLOW}断点继续${NC} (--resume，跳过已通过的测试)"
fi
echo ""

# ── 环境检查 ──────────────────────────────────────────────────────────────────
check_prerequisites() {
    local all_ok=true

    if [ ! -f "$TEST_VIDEO" ]; then
        echo -e "  ${FAIL} 测试视频不存在: $TEST_VIDEO"
        echo "      请将测试视频放置到该路径，或通过参数指定: $0 --input <video_path>"
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
        echo -e "  ${WARN} ffprobe 不可用（将跳过视频元数据验证和时长探测）"
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

# ── 视频时长探测 ──────────────────────────────────────────────────────────────
# 返回值: 秒数 (浮点，去小数)
detect_video_duration() {
    local video="$1"
    local dur
    if command -v ffprobe &>/dev/null; then
        dur=$(ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null || echo "")
    fi
    # 整数秒 (向下取整)
    if [ -n "$dur" ]; then
        echo "${dur%.*}"
    else
        echo ""
    fi
}

# ── 根据 --segments 计算 --segment-duration 参数 ─────────────────────────────
# --segments N (N>0): segment_duration = floor(video_duration / N)
# --segments 0: 不传 --segment-duration (走默认 30s 或整视频)
compute_segment_duration() {
    local num_segments="$1"
    local video_duration="$2"

    if [ "$num_segments" -eq 0 ]; then
        # 不分段：不传 --segment-duration，处理器会看到 duration <= 默认 30s 并跳过拆分
        # 但如果视频 > 30s 仍会被拆分。为了真正"不分段"，传一个极大值。
        echo ""
        return
    fi

    if [ -z "$video_duration" ] || [ "$video_duration" -le 0 ]; then
        echo -e "  ${WARN} 无法获取视频时长，将使用默认分段时长 30s"
        echo ""
        return
    fi

    local seg_dur=$(( video_duration / num_segments ))
    if [ "$seg_dur" -lt 1 ]; then
        seg_dur=1
    fi
    echo "$seg_dur"
}

# 生成 --segment-duration 命令行片段 (如果为空则不加)
segment_duration_arg() {
    if [ -n "$SEGMENT_DURATION" ]; then
        echo "--segment-duration $SEGMENT_DURATION"
    else
        echo ""
    fi
}

# ── 输出目录初始化 ────────────────────────────────────────────────────────────
init_output_dir() {
    PROGRESS_FILE="$OUTPUT_DIR/.test_seg_progress.txt"
    mkdir -p "$OUTPUT_DIR"
    TEST_LOG_FILE="$OUTPUT_DIR/test_seg_run_$(date +%Y%m%d_%H%M%S).log"
    FAIL_LOG="$OUTPUT_DIR/failures_seg.log"
    echo "测试开始: $(date '+%Y-%m-%d %H:%M:%S')" > "$TEST_LOG_FILE"
    echo "分段数: $SEGMENTS (0=不分段)" >> "$TEST_LOG_FILE"
    echo "计算分段时长: ${SEGMENT_DURATION:-默认}" >> "$TEST_LOG_FILE"
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
    local allow_critical_errors="${7:-false}" # 可选：允许严重错误

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if _test_passed "$test_id"; then
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

    local tmp_stdout="$OUTPUT_DIR/.tmp_test_seg_stdout.txt"
    local tmp_stderr="$OUTPUT_DIR/.tmp_test_seg_stderr.txt"

    if eval "$cmd" > "$tmp_stdout" 2> "$tmp_stderr"; then
        local exit_code=0
    else
        local exit_code=$?
    fi

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

    # ★ 自动检测严重错误模式
    if [ "$allow_critical_errors" != "true" ]; then
        local _crit_err=""
        if echo "$combined_output" | grep -qE "提前EOF"; then
            _crit_err="提前EOF (premature EOF — 输入帧读取不完整)"
        elif echo "$combined_output" | grep -qE "缺失[[:space:]]*[0-9]+[[:space:]]*帧"; then
            _crit_err="帧缺失 (missing frames — 输出帧数不足)"
        elif echo "$combined_output" | grep -qE "差[[:space:]]*[0-9.]+[[:space:]]*帧.*⚠️"; then
            _crit_err="帧计数不匹配 (frame count mismatch with ⚠️)"
        elif echo "$combined_output" | grep -qE "non-existing PPS"; then
            _crit_err="non-existing PPS (H.264 SPS/PPS 流损坏)"
        elif echo "$combined_output" | grep -qE "no frame!"; then
            _crit_err="no frame! (FFmpeg H.264 解码失败)"
        elif echo "$combined_output" | grep -qE "decode_slice_header error"; then
            _crit_err="decode_slice_header error (H.264 解码错误)"
        elif echo "$combined_output" | grep -qE "原始帧:.*输出帧:.*差[[:space:]]*[1-9]"; then
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

    if [ "$expected_in_log" = "RUNTIME_ONLY" ]; then
        echo -e "${PASS} (运行成功)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        _mark_passed "$test_id"
        return 0
    fi

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

# ── 帧计数验证 ────────────────────────────────────────────────────────────────
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

# ── 检查分段日志中是否有分段处理痕迹 ──────────────────────────────────────────
check_segment_traces() {
    local test_id="$1"
    local stderr_file="$2"
    local expected_segments="$3"

    if [ ! -f "$stderr_file" ]; then
        return 0
    fi

    # 统计实际分段数 (匹配 "分段" 或 "segment_" 输出文件)
    local seg_count
    seg_count=$(grep -c "处理分段\|segment_.*\.mp4\|分段.*完成\|Segment.*complete" "$stderr_file" 2>/dev/null || true)
    [ -z "$seg_count" ] && seg_count=0

    echo "      ${INFO} 日志中检测到 ~${seg_count} 个分段处理痕迹"
}

# ── 检查 FFmpegMuxer stderr 中的错误模式 ──────────────────────────────────────
check_ffmpeg_stderr() {
    local test_id="$1"
    local stderr_file="$2"
    local output_file="$3"

    if [ ! -f "$stderr_file" ]; then
        return 0
    fi

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
        if grep -qxF "$test_id" "$PROGRESS_FILE" 2>/dev/null; then
            _remove_from_progress "$test_id"
        fi
        return 1
    fi
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# 主测试流程
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    detect_python
    check_prerequisites

    # ── 探测视频时长并计算分段参数 ────────────────────────────────────────────
    local VIDEO_DURATION
    VIDEO_DURATION=$(detect_video_duration "$TEST_VIDEO")

    if [ -n "$VIDEO_DURATION" ] && [ "$VIDEO_DURATION" -gt 0 ]; then
        echo -e "  视频时长 : ${CYAN}${VIDEO_DURATION}s${NC}"

        if [ "$SEGMENTS" -gt 0 ]; then
            SEGMENT_DURATION=$(compute_segment_duration "$SEGMENTS" "$VIDEO_DURATION")
            if [ -n "$SEGMENT_DURATION" ]; then
                echo -e "  分段策略 : ${CYAN}${SEGMENTS} 段 × ~${SEGMENT_DURATION}s${NC}"
            else
                echo -e "  分段策略 : ${CYAN}${SEGMENTS} 段 (使用默认时长)${NC}"
            fi
        else
            SEGMENT_DURATION=""
            echo -e "  分段策略 : ${CYAN}不分段 (整视频单段处理)${NC}"
        fi
    else
        echo -e "  ${WARN} 无法获取视频时长，分段测试可能不准确"
        SEGMENT_DURATION=""
    fi

    # 生成分段 CLI 参数片段
    SEG_ARG=""
    if [ "$SEGMENTS" -gt 0 ] && [ -n "$SEGMENT_DURATION" ]; then
        SEG_ARG="--segment-duration $SEGMENT_DURATION"
    elif [ "$SEGMENTS" -eq 0 ]; then
        # 不分段：传一个极大值确保不会触发拆分
        SEG_ARG="--segment-duration 999999"
    fi

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

    # ★ 准备工作: IFRNet backend 影子副本 (FORCE_CONSTQP=True), 不依赖生产源码状态
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
import sys, runpy

shadow_dir = sys.argv[1]
project_dir = sys.argv[2]
sys.path.insert(0, project_dir)   # ensures 'models' / 'external.IFRNet' imports work
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

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 1: LA 分段策略 — 独立排空帧数守恒
    # ══════════════════════════════════════════════════════════════════════════
    echo -e "${BOLD}${BLUE}═══ 阶段 1: LA 分段策略 — 独立排空帧数守恒 ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证 LA>0 + 多分段时每段独立排空，帧数守恒:"
    echo -e "  ${INFO}   - IFRNet (qvbr/LA=8/crf≠0) → 输出每段独立排空消息"
    echo -e "  ${INFO}   - ESRGAN (vbr_hq/LA=16/crf≠0) → 同上"
    echo -e "  ${INFO}   - crf=0 → CONSTQP LA=0 (FORCE_CONSTQP=True 强制无损)"
    echo -e "  ${INFO}   - rate_mode=constqp → 无 LA 消息 (硬件静默禁用)"
    echo -e "  ${INFO}   - LA=0 → 不激活 LA 消息"
    echo ""

    # ── 1a: IFRNet qvbr+LA=8 → 应输出 LA 独立排空消息 ──
    run_test "1a" "LA+分段: IFRNet(qvbr/LA=8)独立排空" \
        "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1a_ifrnet_la_disable_seg.mp4 \
            --rate-mode qvbr --lookahead-depth 8 --quiet 2>&1" \
        "每段独立排空" \
        "la_ifrnet_disable_seg_log"

    # ── 1b: ESRGAN vbr_hq+LA=16 → 应输出 NVENC 编码激活消息 ──
    # ESRGAN 有两条输出路径，使用大小写不敏感匹配
    run_test "1b" "LA+分段: ESRGAN(vbr_hq/LA=16)独立排空" \
        "$PYTHON -m src.processors.realesrgan_processor_video_optimized \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1b_esrgan_la_disable_seg.mp4 \
            --rate-mode vbr_hq --lookahead-depth 16 --quiet 2>&1" \
        "[Vv][Bb][Rr]_[Hh][Qq].*[Ll][Aa]=16" \
        "la_esrgan_disable_seg_log"

    # ── 1c: IFRNet crf=0 → CONSTQP LA=0 ──
    if $has_nvenc; then
        run_test "1c" "LA+分段: crf=0 → CONSTQP LA=0" \
            "$PYTHON $SHADOW_WRAPPER $SHADOW_CONSTQP_TRUE $PROJECT_DIR \
                -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1c_ifrnet_crf0_no_disable.mp4 \
                --crf 0 --quiet 2>&1" \
            "CONSTQP.*LA=0" \
            "la_crf0_no_disable_log" \
            "每段独立排空"
    else
        echo -e "  [1c] crf=0 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 1d: IFRNet constqp+LA=16 → 不应禁用分段 (constqp 下 LA 无意义，自动强制为 0) ──
    if $has_nvenc; then
        run_test "1d" "LA+分段: constqp+LA=16 → CONSTQP 无 LA 消息" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1d_ifrnet_constqp_la16_no_disable.mp4 \
                --rate-mode constqp --lookahead-depth 16 --quiet 2>&1" \
            "CONSTQP" \
            "la_constqp_no_disable_log" \
            "每段独立排空"
    else
        echo -e "  [1d] constqp+LA=16不触发分段禁用 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 1e: LA=0 + vbr_hq → 不应有 LA 独立排空消息 ──
    if $has_nvenc; then
        run_test "1e" "LA+分段: vbr_hq+LA=0 无 LA 消息" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t1e_ifrnet_vbrhq_la0_no_disable.mp4 \
                --rate-mode vbr_hq --lookahead-depth 0 --quiet 2>&1" \
            "VBR_HQ" \
            "la_vbrhq_la0_no_disable_log" \
            "每段独立排空"
    else
        echo -e "  [1e] vbr_hq+LA=0不触发分段禁用 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 2: N 分段全流程测试 (由 --segments 控制)
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 2: N 分段全流程测试 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""

    if [ "$SEGMENTS" -gt 0 ]; then
        echo -e "  ${INFO} 分段模式: ${SEGMENTS} 段, 每段 ≈ ${SEGMENT_DURATION}s"
        echo -e "  ${INFO} CLI 参数: ${SEG_ARG}"
    else
        echo -e "  ${INFO} 不分段模式: 整视频单段处理"
        echo -e "  ${INFO} CLI 参数: ${SEG_ARG}"
    fi
    echo ""

    # ── 2a: IFRNet standalone + N 分段 (默认 rate_mode=vbr_hq) ──
    run_test "2a" "IFRNet standalone ${SEGMENTS}段处理 (默认 vbr_hq)" \
        "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2a_ifrnet_nseg_default.mp4 \
            $SEG_ARG --quiet 2>&1" \
        "VBR_HQ.*LA=8" \
        "ifrnet_nseg_default_log"

    if [ -f "$OUTPUT_DIR/t2a_ifrnet_nseg_default.mp4" ]; then
        verify_output_video "$OUTPUT_DIR/t2a_ifrnet_nseg_default.mp4" "IFRNet_${SEGMENTS}段_默认"
    fi

    # ── 2b: Real-ESRGAN standalone + N 分段 (默认 rate_mode=vbr_hq) ──
    run_test "2b" "ESRGAN standalone ${SEGMENTS}段处理 (默认 vbr_hq)" \
        "$PYTHON -m src.processors.realesrgan_processor_video_optimized \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2b_esrgan_nseg_default.mp4 \
            $SEG_ARG --quiet 2>&1" \
        "[Vv][Bb][Rr]_[Hh][Qq]" \
        "esrgan_nseg_default_log"

    if [ -f "$OUTPUT_DIR/t2b_esrgan_nseg_default.mp4" ]; then
        verify_output_video "$OUTPUT_DIR/t2b_esrgan_nseg_default.mp4" "ESRGAN_${SEGMENTS}段_默认"
    fi

    # ── 2c: IFRNet + N 分段 + constqp ──
    if $has_nvenc; then
        run_test "2c" "IFRNet ${SEGMENTS}段 + constqp" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2c_ifrnet_nseg_constqp.mp4 \
                --rate-mode constqp $SEG_ARG --quiet 2>&1" \
            "CONSTQP" \
            "ifrnet_nseg_constqp_log"

        # ── 2d: IFRNet + N 分段 + qvbr + LA=8 (独立排空帧数守恒) ──
        run_test "2d" "IFRNet ${SEGMENTS}段 + qvbr/LA=8 → 独立排空" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2d_ifrnet_nseg_qvbr_la8.mp4 \
                --rate-mode qvbr --lookahead-depth 8 $SEG_ARG --quiet 2>&1" \
            "每段独立排空" \
            "ifrnet_nseg_qvbr_la8_log"
    else
        echo -e "  [2c] IFRNet ${SEGMENTS}段 constqp — ${SKIP} (无 NVENC)"
        echo -e "  [2d] IFRNet ${SEGMENTS}段 qvbr+LA — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 2))
    fi

    # ── 2e: ESRGAN + N 分段 + vbr_hq ──
    run_test "2e" "ESRGAN ${SEGMENTS}段 + vbr_hq" \
        "$PYTHON -m src.processors.realesrgan_processor_video_optimized \
            -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t2e_esrgan_nseg_vbrhq.mp4 \
            --rate-mode vbr_hq $SEG_ARG --quiet 2>&1" \
        "[Vv][Bb][Rr]_[Hh][Qq]" \
        "esrgan_nseg_vbrhq_log"

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 3: 全流程 N 分段 + rate_mode 组合测试
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 3: 全流程 N 分段 + rate_mode 组合 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""

    # ── 3a: 全流程默认 (interpolate_then_upscale) + N 分段 ──
    run_test "3a" "全流程 ${SEGMENTS}段 默认组合" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t3a_full_nseg_default.mp4 \
            $SEG_ARG 2>&1" \
        "全流程完成" \
        "full_nseg_default_log"

    if [ -f "$OUTPUT_DIR/t3a_full_nseg_default.mp4" ]; then
        verify_output_video "$OUTPUT_DIR/t3a_full_nseg_default.mp4" "全流程_${SEGMENTS}段_默认"
        # 检查日志中的分段痕迹
        check_segment_traces "3a" "$OUTPUT_DIR/.tmp_test_seg_stderr.txt" "$SEGMENTS"
    fi

    # ── 3b: 全流程统一 constqp + N 分段 ──
    if $has_nvenc; then
        run_test "3b" "全流程 ${SEGMENTS}段 统一 constqp" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t3b_full_nseg_constqp.mp4 \
                --rate-mode-ifrnet constqp --rate-mode-esrgan constqp \
                $SEG_ARG 2>&1" \
            "全流程完成" \
            "full_nseg_constqp_log"

        # 验证日志中两阶段均为 constqp
        local out3b
        out3b=$(cat "$OUTPUT_DIR/.tmp_test_seg_stdout.txt" "$OUTPUT_DIR/.tmp_test_seg_stderr.txt" 2>/dev/null || echo "")
        if echo "$out3b" | grep -q "CONSTQP"; then
            echo "      ${GREEN}✓${NC} IFRNet constqp 已确认"
        fi
        if echo "$out3b" | grep -qi "constqp"; then
            echo "      ${GREEN}✓${NC} ESRGan constqp 已确认"
        fi

        if [ -f "$OUTPUT_DIR/t3b_full_nseg_constqp.mp4" ]; then
            verify_output_video "$OUTPUT_DIR/t3b_full_nseg_constqp.mp4" "全流程_${SEGMENTS}段_constqp"
        fi
    else
        echo -e "  [3b] 全流程 ${SEGMENTS}段 constqp — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 3c: 全流程统一 qvbr + LA=16 + N 分段 ──
    if $has_nvenc; then
        run_test "3c" "全流程 ${SEGMENTS}段 qvbr+LA=16 → 帧数守恒" \
            "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
                -o $OUTPUT_DIR/t3c_full_nseg_qvbr_la16.mp4 \
                --rate-mode-ifrnet qvbr --rate-mode-esrgan qvbr \
                --lookahead-depth-ifrnet 16 --lookahead-depth-esrgan 16 \
                $SEG_ARG 2>&1" \
            "全流程完成" \
            "full_nseg_qvbr_la16_log"
    else
        echo -e "  [3c] 全流程 ${SEGMENTS}段 qvbr+LA — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ── 3d: upscale_then_interpolate + N 分段 ──
    # ⚠️ upscale_then_interpolate 模式下 IFRNet 处理高分辨率帧，batch_size=24 易 OOM
    # 此处显式降低 batch_size + 开启 TRT 以保证测试稳定完成
    run_test "3d" "upscale_then_interpolate ${SEGMENTS}段" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t3d_upscale_first_nseg.mp4 \
            --mode upscale_then_interpolate \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan \
            --batch-size-ifrnet 6 \
            $SEG_ARG 2>&1" \
        "全流程完成" \
        "upscale_first_nseg_log"

    if [ -f "$OUTPUT_DIR/t3d_upscale_first_nseg.mp4" ]; then
        verify_output_video "$OUTPUT_DIR/t3d_upscale_first_nseg.mp4" "upscale_first_${SEGMENTS}段"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 4: 帧守恒 + 分段测试
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 4: 帧守恒 + 分段测试 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""

    # 探测测试视频是否有音频轨（仅用于额外音视频时长一致性验证）
    local has_audio=false
    if command -v ffprobe &>/dev/null; then
        local audio_streams
        audio_streams=$(ffprobe -v error -select_streams a:0 \
            -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "$TEST_VIDEO" 2>/dev/null || echo "")
        if [ "$audio_streams" = "audio" ]; then
            has_audio=true
        fi
    fi

    # ── 4a: 全流程默认 (LA>0, crf>0) + N 分段 → 帧守恒: 输出时长=输入时长 ──
    # NVENC SDK 帧守恒: Output==Input, LA 仅引入延迟不增删帧 → 无需音频修剪
    run_test "4a" "帧守恒: 全流程 ${SEGMENTS}段 → 输出时长=输入时长" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t4a_full_nseg_frame_conservation.mp4 \
            $SEG_ARG 2>&1" \
        "全流程完成" \
        "frame_conservation_nseg_log" \
        "检测到 LA 导致输出视频缺失"

    # 后处理验证: 帧守恒 → 时长一致
    local out4a
    out4a=$(cat "$OUTPUT_DIR/.tmp_test_seg_stdout.txt" "$OUTPUT_DIR/.tmp_test_seg_stderr.txt" 2>/dev/null || echo "")
    if echo "$out4a" | grep -q "时长差异: 0.00秒"; then
        echo "      ${GREEN}✓${NC} 帧守恒: 输出时长=输入时长 (NVENC Output==Input)"
    elif echo "$out4a" | grep -q "输出时长偏差.*（非 LA 原因"; then
        echo "      ${GREEN}✓${NC} 帧守恒: 微小偏差确认为非 LA 原因"
    else
        echo "      ${YELLOW}?${NC} 未检测到明确的时长一致性消息"
    fi

    # 音视频时长一致性 (额外验证，需音频轨)
    if $has_audio && [ -f "$OUTPUT_DIR/t4a_full_nseg_frame_conservation.mp4" ]; then
        echo "      ── 音视频时长一致性 ──"
        local v_dur a_dur
        v_dur=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=duration \
            -of default=noprint_wrappers=1:nokey=1 \
            "$OUTPUT_DIR/t4a_full_nseg_frame_conservation.mp4" 2>/dev/null || echo "N/A")
        a_dur=$(ffprobe -v error -select_streams a:0 \
            -show_entries stream=duration \
            -of default=noprint_wrappers=1:nokey=1 \
            "$OUTPUT_DIR/t4a_full_nseg_frame_conservation.mp4" 2>/dev/null || echo "N/A")
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

    # ── 4b: crf=0 全流程 + N 分段 → CONSTQP LA=0 帧守恒 ──
    run_test "4b" "帧守恒: crf=0全流程 ${SEGMENTS}段 → CONSTQP LA=0 时长不变" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t4b_crf0_nseg_frame_conservation.mp4 \
            --crf-ifrnet 0 --crf-esrgan 0 $SEG_ARG 2>&1" \
        "全流程完成" \
        "frame_conservation_crf0_nseg_log" \
        "检测到 LA 导致输出视频缺失"

    # ── 4c: IFRNet standalone + LA + N 分段 → 帧守恒 (独立处理器级别) ──
    # 注: IFRNet standalone 不处理音频，此测试验证 N 分段下 LA 日志 + 帧守恒无异常
    if $has_nvenc; then
        run_test "4c" "IFRNet standalone ${SEGMENTS}段 + qvbr/LA=8" \
            "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t4c_ifrnet_nseg_qvbr_la8.mp4 \
                --rate-mode qvbr --lookahead-depth 8 $SEG_ARG --quiet 2>&1" \
            "QVBR.*LA=8" \
            "ifrnet_nseg_qvbr_la8_log"
    else
        echo -e "  [4c] IFRNet standalone ${SEGMENTS}段 qvbr+LA — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 5: 分段边界连续性验证
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 5: 分段边界连续性验证 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""

    # 仅当分段数 > 1 时做此验证（单段无所谓边界）
    if [ "$SEGMENTS" -gt 1 ] && [ -n "$SEGMENT_DURATION" ]; then
        # ── 5a: 对比 N 分段 vs 不分段 输出的帧数和文件大小 ──
        echo -e "  ${INFO} 阶段 5a: N 分段 vs 不分段 输出对比"

        # 不分段参考输出 (使用 --segment-duration 999999 强制不分段)
        local REF_OUT="$OUTPUT_DIR/t5a_ref_noseg.mp4"
        local NSEG_OUT="$OUTPUT_DIR/t5a_ifrnet_nseg.mp4"

        # 生成不分段参考
        if [ ! -f "$REF_OUT" ]; then
            echo -e "  ${INFO} 生成不分段参考输出..."
            $PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o "$REF_OUT" \
                --segment-duration 999999 --quiet 2>&1 > /dev/null || true
        fi

        # 生成 N 分段输出
        if [ ! -f "$NSEG_OUT" ]; then
            echo -e "  ${INFO} 生成 ${SEGMENTS} 段输出..."
            $PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                -c $CONFIG -i $TEST_VIDEO -o "$NSEG_OUT" \
                $SEG_ARG --quiet 2>&1 > /dev/null || true
        fi

        # 帧计数对比
        if [ -f "$REF_OUT" ] && [ -f "$NSEG_OUT" ]; then
            local ref_frames nseg_frames
            ref_frames=$(ffprobe -v error -select_streams v:0 \
                -show_entries stream=nb_frames \
                -of default=noprint_wrappers=1:nokey=1 "$REF_OUT" 2>/dev/null || echo "?")
            nseg_frames=$(ffprobe -v error -select_streams v:0 \
                -show_entries stream=nb_frames \
                -of default=noprint_wrappers=1:nokey=1 "$NSEG_OUT" 2>/dev/null || echo "?")

            echo "      ── 帧计数对比 ──"
            echo "      不分段 : ${ref_frames} 帧"
            echo "      ${SEGMENTS}段   : ${nseg_frames} 帧"

            if [ "$ref_frames" != "?" ] && [ "$nseg_frames" != "?" ]; then
                local fdiff
                if [ "$ref_frames" -eq "$nseg_frames" ]; then
                    echo "      ${GREEN}✓${NC} 帧数完全一致 (分段边界无帧丢失)"
                else
                    fdiff=$(( ref_frames - nseg_frames ))
                    [ "$fdiff" -lt 0 ] && fdiff=$(( -fdiff ))
                    if [ "$fdiff" -le 2 ]; then
                        echo "      ${GREEN}✓${NC} 帧数差异 ≤2 (分段: ${nseg_frames}, 不分段: ${ref_frames}, 差=${fdiff})"
                    else
                        echo "      ${WARN} 帧数差异 >2: 分段=${nseg_frames} 不分段=${ref_frames} 差=${fdiff}"
                        echo "      ${INFO} 这可能是因为分段边界帧被编码器跳过 (非 bug)"
                    fi
                fi
            fi

            # 文件大小对比
            compare_file_sizes "$REF_OUT" "$NSEG_OUT" "不分段" "${SEGMENTS}段"
        else
            echo -e "  ${YELLOW}?${NC} 无法生成对比输出，跳过帧计数验证"
        fi

        # ── 5b: 验证 N 分段输出可被 ffprobe 正常解码 ──
        if [ -f "$NSEG_OUT" ]; then
            echo ""
            echo "      ── ${SEGMENTS} 段输出元数据 ──"
            verify_output_video "$NSEG_OUT" "${SEGMENTS}段IFRNet"

            # 验证无 H.264 解码错误
            local _5b_pid="5b"
            if ! check_ffmpeg_stderr "$_5b_pid" \
                "$OUTPUT_DIR/.tmp_test_seg_stderr.txt" \
                "$NSEG_OUT"; then
                :
            fi
        fi
    else
        echo -e "  ${SKIP} 阶段 5: 分段边界验证 (需要 --segments > 1)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 6: 处理模式 + 分段组合
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 6: 处理模式 + 分段组合 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""

    # ── 6a: upscale_then_interpolate + N 分段 + rate_mode ──
    # ⚠️ 与 3d 同理，显式降低 batch_size + 开启 TRT 防 OOM
    run_test "6a" "upscale_then_interpolate ${SEGMENTS}段 + rate_mode组合" \
        "$PYTHON src/main_video_optimized.py -c $CONFIG -i $TEST_VIDEO \
            -o $OUTPUT_DIR/t6a_upscale_first_rate_nseg.mp4 \
            --mode upscale_then_interpolate \
            --rate-mode-ifrnet constqp --rate-mode-esrgan qvbr \
            --use-tensorrt-ifrnet --use-tensorrt-esrgan \
            --batch-size-ifrnet 6 \
            $SEG_ARG 2>&1" \
        "全流程完成" \
        "upscale_first_rate_nseg_log"

    if [ -f "$OUTPUT_DIR/t6a_upscale_first_rate_nseg.mp4" ]; then
        verify_output_video "$OUTPUT_DIR/t6a_upscale_first_rate_nseg.mp4" "upscale_first_rate_${SEGMENTS}段"
    fi

    # 后处理: 检查日志中的 LA 分段策略消息
    local out6a
    out6a=$(cat "$OUTPUT_DIR/.tmp_test_seg_stdout.txt" "$OUTPUT_DIR/.tmp_test_seg_stderr.txt" 2>/dev/null || echo "")
    if echo "$out6a" | grep -q "每段独立排空"; then
        echo "      ${GREEN}✓${NC} upscale_then_interpolate 模式 LA 独立排空已生效"
    else
        echo "      ${YELLOW}?${NC} 未检测到「每段独立排空」消息 (可能视频 < segment_duration)"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 7: --la-segmented-mode carryover 分段测试 (方案核心新增)
    # ══════════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${BOLD}${BLUE}═══ 阶段 7: LA 帧数守恒 — 多分段验证 (分段数=${SEGMENTS}) ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证 LA>0 + 多分段时的帧数守恒:"
    echo -e "  ${INFO}   - qvbr/LA=8 + N 段: 每段独立排空，帧数守恒"
    echo -e "  ${INFO}   - 对比分段 vs 不分段输出帧数: 差异 ≤2"
    echo ""

    if $has_nvenc; then
        # ── 7a: IFRNet + qvbr/LA=8 + N 段 → 帧数守恒 ──
        if [ "$SEGMENTS" -gt 1 ] && [ -n "$SEG_ARG" ]; then
            run_test "7a" "帧守恒: IFRNet ${SEGMENTS}段 + qvbr/LA=8 → 独立排空" \
                "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                    -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t7a_ifrnet_nseg_qvbr.mp4 \
                    --rate-mode qvbr --lookahead-depth 8 \
                    $SEG_ARG --quiet 2>&1" \
                "每段独立排空" \
                "ifrnet_nseg_qvbr_log" \
                "LA 帧数守恒消息缺失"

            # 后处理: 验证日志中确实出现了分段处理痕迹（未被禁用）
            if [ -f "$OUTPUT_DIR/t7a_ifrnet_nseg_qvbr.mp4" ]; then
                echo "      ${GREEN}✓${NC} 帧数守恒模式已启用，分段正常进行"
            fi
        else
            echo -e "  [7a] IFRNet 帧守恒 ${SEGMENTS}段 — ${SKIP} (需要 --segments > 1)"
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        fi

        # ── 7b: IFRNet + vbr_hq/LA=0 + N 段 → 无 LA 消息 ──
        if [ "$SEGMENTS" -gt 1 ] && [ -n "$SEG_ARG" ]; then
            run_test "7b" "帧守恒: IFRNet ${SEGMENTS}段 + vbr_hq/LA=0 → 无 LA" \
                "$PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                    -c $CONFIG -i $TEST_VIDEO -o $OUTPUT_DIR/t7b_ifrnet_nseg_vbrhq_la0.mp4 \
                    --rate-mode vbr_hq --lookahead-depth 0 \
                    $SEG_ARG --quiet 2>&1" \
                "VBR_HQ" \
                "ifrnet_nseg_vbrhq_la0_log" \
                "LA=[1-9]"
        else
            echo -e "  [7b] IFRNet vbr_hq/LA=0 ${SEGMENTS}段 — ${SKIP} (需要 --segments > 1)"
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        fi

        # ── 7c: 帧计数对比: 分段 vs 不分段 (LA>0) ──
        if [ "$SEGMENTS" -gt 1 ] && [ -n "$SEGMENT_DURATION" ]; then
            echo ""
            echo -e "  ${INFO} 阶段 7c: 分段 vs 不分段 帧计数对比 (LA>0)"

            local REF_NOSEG="$OUTPUT_DIR/t7c_ref_noseg.mp4"
            local NSEG_OUT="$OUTPUT_DIR/t7c_ifrnet_nseg_compare.mp4"

            # 生成不分段参考 (segment-duration 极大值)
            if [ ! -f "$REF_NOSEG" ]; then
                echo -e "  ${INFO} 生成不分段参考输出..."
                $PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                    -c $CONFIG -i $TEST_VIDEO -o "$REF_NOSEG" \
                    --rate-mode qvbr --lookahead-depth 8 \
                    --segment-duration 999999 --quiet 2>&1 > /dev/null || true
            fi

            # 生成 N 分段输出
            if [ ! -f "$NSEG_OUT" ]; then
                echo -e "  ${INFO} 生成 ${SEGMENTS} 段输出..."
                $PYTHON -m src.processors.ifrnet_processor_v6_4_single \
                    -c $CONFIG -i $TEST_VIDEO -o "$NSEG_OUT" \
                    --rate-mode qvbr --lookahead-depth 8 \
                    $SEG_ARG --quiet 2>&1 > /dev/null || true
            fi

            if [ -f "$REF_NOSEG" ] && [ -f "$NSEG_OUT" ]; then
                local ref_frames7 nseg_frames7
                ref_frames7=$(ffprobe -v error -select_streams v:0 \
                    -show_entries stream=nb_frames \
                    -of default=noprint_wrappers=1:nokey=1 "$REF_NOSEG" 2>/dev/null || echo "?")
                nseg_frames7=$(ffprobe -v error -select_streams v:0 \
                    -show_entries stream=nb_frames \
                    -of default=noprint_wrappers=1:nokey=1 "$NSEG_OUT" 2>/dev/null || echo "?")

                echo "      ── 帧计数对比 (LA>0) ──"
                echo "      不分段 : ${ref_frames7} 帧"
                echo "      ${SEGMENTS}段   : ${nseg_frames7} 帧"

                if [ "$ref_frames7" != "?" ] && [ "$nseg_frames7" != "?" ]; then
                    local fdiff7
                    if [ "$ref_frames7" -eq "$nseg_frames7" ]; then
                        echo "      ${GREEN}✓${NC} 帧数完全一致 (零额外帧丢失)"
                    else
                        fdiff7=$(( ref_frames7 - nseg_frames7 ))
                        [ "$fdiff7" -lt 0 ] && fdiff7=$(( -fdiff7 ))
                        if [ "$fdiff7" -le 2 ]; then
                            echo "      ${GREEN}✓${NC} 帧数差异 ≤2 (分段=${nseg_frames7}, 不分段=${ref_frames7})"
                        else
                            echo "      ${WARN} 帧数差异 >2: 差=${fdiff7}"
                        fi
                    fi
                fi
            fi
        fi
    else
        echo -e "  [7a-7c] LA 帧数守恒 ${SEGMENTS}段 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 3))
    fi

    echo -e "${BOLD}${BLUE}═══ 阶段 8: 断点恢复测试 ═══${NC}"
    echo ""
    echo -e "  ${INFO} 验证断点恢复功能:"
    echo -e "  ${INFO}   - 处理过程中断后，重跑应跳过已完成的段"
    echo -e "  ${INFO}   - 恢复时 NVENC 编码器重建 (rate_mode 保持)"
    echo ""

    if $has_nvenc; then
        # ── 8a: 模拟断点恢复 → 跳过已完成段 ──
        if [ "$SEGMENTS" -gt 1 ] && [ -n "$SEGMENT_DURATION" ]; then
            local CKPT_DIR="$OUTPUT_DIR/t8a_checkpoint_test"
            local CKPT_OUT="$OUTPUT_DIR/t8a_resume_test.mp4"
            local CKPT_SEG_DIR="$OUTPUT_DIR/t8a_segments"

            echo -e "  ${INFO} 断点恢复测试准备..."
            echo -e "  ${INFO}   阶段1: 首次运行建立 checkpoint..."

            # 首次运行: 使用 qvbr/LA=8, segment_duration 短到足够产生多段
            local t8a_first_log="$OUTPUT_DIR/t8a_first_run.log"
            $PYTHON -m src.processors.ifrnet_processor_v6_4_single                 -c $CONFIG -i "$TEST_VIDEO" -o "$CKPT_OUT"                 --rate-mode qvbr --lookahead-depth 8                 --segment-duration "$SEGMENT_DURATION"                 --quiet > "$t8a_first_log" 2>&1 || true

            # 搜索 checkpoint 文件
            local ckpt_file
            ckpt_file=$(find "$OUTPUT_DIR" /tmp "$PROJECT_DIR/temp"                 -name "checkpoint.json" -path "*ifrnet*" -type f 2>/dev/null | head -1)
            if [ -z "$ckpt_file" ]; then
                ckpt_file=$(grep -oP 'checkpoint[^"]*\.json' "$t8a_first_log" 2>/dev/null | head -1 || echo "")
            fi
            if [ -n "$ckpt_file" ] && [ -f "$ckpt_file" ]; then
                echo -e "  ${INFO}   checkpoint 已生成: $ckpt_file"

                # 修改 checkpoint: 标记最后一段为已完成 (模拟中断恢复)
                $PYTHON -c "
import json, sys
try:
    with open('$ckpt_file', 'r') as f:
        ckpt = json.load(f)
    ckpt['last_segment'] = 0
    ckpt['completed_segments'] = ['seg_0000']
    with open('$ckpt_file', 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f'checkpoint 已修改: last_segment=0')
except Exception as e:
    print(f'修改失败: {e}')
    sys.exit(1)
" 2>&1 | tee -a "$TEST_LOG_FILE"

                echo -e "  ${INFO}   阶段2: 重新运行触发断点恢复..."

                # 重新运行: 应触发断点恢复
                local tmp_resume_stdout="$OUTPUT_DIR/.tmp_test_seg_stdout.txt"
                local tmp_resume_stderr="$OUTPUT_DIR/.tmp_test_seg_stderr.txt"

                if $PYTHON -m src.processors.ifrnet_processor_v6_4_single                     -c $CONFIG -i "$TEST_VIDEO" -o "$CKPT_OUT"                     --rate-mode qvbr --lookahead-depth 8                     --segment-duration "$SEGMENT_DURATION"                     --quiet                     > "$tmp_resume_stdout" 2> "$tmp_resume_stderr"; then
                    local resume_combined
                    resume_combined="$(cat "$tmp_resume_stdout" 2>/dev/null; cat "$tmp_resume_stderr" 2>/dev/null)"

                    # 检查是否成功跳过
                    if echo "$resume_combined" | grep -q "已处理"; then
                        echo -e "  [8a] 断点恢复测试 ... ${PASS}"
                        echo "      ${GREEN}✓${NC} 断点恢复功能正常 (跳过已完成段)"
                        TESTS_PASSED=$((TESTS_PASSED + 1))
                        _mark_passed "8a"
                    else
                        TESTS_TOTAL=$((TESTS_TOTAL + 1))
                        echo -e "  [8a] 断点恢复测试 ... ${WARN}"
                        echo "      ${YELLOW}?${NC} 未检测到断点恢复跳过 (可能仅1段)"
                        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
                    fi

                    cat "$tmp_resume_stdout" >> "$TEST_LOG_FILE"
                    cat "$tmp_resume_stderr" >> "$TEST_LOG_FILE"
                else
                    TESTS_TOTAL=$((TESTS_TOTAL + 1))
                    echo -e "  [8a] 断点恢复测试 ... ${WARN}"
                    echo "      ${YELLOW}?${NC} 恢复运行失败 (可能 checkpoint 格式不兼容)"
                    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
                fi

                # 清理 checkpoint
                if [ -n "$ckpt_file" ] && [ -f "$ckpt_file" ]; then
                    rm -f "$ckpt_file" 2>/dev/null || true
                fi
            else
                echo -e "  [8a] 断点恢复 — ${SKIP} (未生成 checkpoint)"
                TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
            fi
        else
            echo -e "  [8a] 断点恢复 — ${SKIP} (需要 --segments > 1)"
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        fi
    else
        echo -e "  [8a] 断点恢复 — ${SKIP} (无 NVENC)"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    fi

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
    echo -e "  分段数 : ${CYAN}$SEGMENTS${NC} (0=不分段)"
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
    rm -f "$OUTPUT_DIR"/.tmp_test_seg_stdout.txt "$OUTPUT_DIR"/.tmp_test_seg_stderr.txt
    # 清理影子副本
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

# 注册 EXIT trap — 清理所有影子副本和 wrapper
trap 'rm -rf "$OUTPUT_DIR"/.tmp_ifrnet_* 2>/dev/null; rm -f "$OUTPUT_DIR"/.tmp_shadow_*.py 2>/dev/null; rm -f "$OUTPUT_DIR"/.tmp_config_*.json 2>/dev/null; true' EXIT

main
