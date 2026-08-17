#!/bin/bash
# ============================================================================
# test_ltrace_ffmpeg_nvenc.sh
# 目的: 用 ltrace 追踪 FFmpeg h264_nvenc 的 NVENC/CUDA API 调用序列
# 用法: bash test_ltrace_ffmpeg_nvenc.sh
# 输出: /tmp/nvenc_ltrace_<timestamp>/
# ============================================================================

set -e

OUTDIR="/tmp/nvenc_ltrace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

W=1920
H=1080
FPS=30
PIX_FMT="yuv420p"
TEST_YUV="$OUTDIR/test.yuv"

echo "[*] 输出目录: $OUTDIR"
echo "[*] 生成测试 YUV: ${W}x${H} ${PIX_FMT} 1 帧"
ffmpeg -y -v error -f lavfi -i "testsrc=${W}x${H}:r=${FPS}:d=0.1" \
  -pix_fmt "$PIX_FMT" -frames:v 1 -f rawvideo "$TEST_YUV"
echo "[*] test.yuv: $(wc -c < "$TEST_YUV") bytes"

# ── 检查可用工具 ──
HAVE_LTRACE=0; HAVE_STRACE=0; HAVE_GDB=0
command -v ltrace >/dev/null 2>&1 && HAVE_LTRACE=1
command -v strace >/dev/null 2>&1 && HAVE_STRACE=1
command -v gdb    >/dev/null 2>&1 && HAVE_GDB=1

ffmpeg_cmd() {
  # $1: extra ffmpeg args (before -i)
  local extra="$1"
  ffmpeg -y -v error \
    $extra \
    -f rawvideo -pix_fmt "$PIX_FMT" -s "${W}x${H}" -r "$FPS" \
    -i "$TEST_YUV" \
    -c:v h264_nvenc -frames:v 1 \
    -f null /dev/null
}

# ════════════════════════════════════════════════════════════════════════════
# 测试 1: ltrace — 完整调用追踪
# ════════════════════════════════════════════════════════════════════════════
if [ "$HAVE_LTRACE" -eq 1 ]; then
  echo ""
  echo "=== 测试 1a: ltrace 全部 libnvidia-encode + libcuda ==="
  ltrace -o "$OUTDIR/ltrace_full.txt" \
    -e '@libnvidia-encode*' \
    ffmpeg_cmd "" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/ltrace_full.txt") 行 → $OUTDIR/ltrace_full.txt"

  echo ""
  echo "=== 测试 1b: ltrace 仅 libnvidia-encode (含参数) ==="
  ltrace -o "$OUTDIR/ltrace_nvenc.txt" \
    -e '@libnvidia-encode*' \
    ffmpeg_cmd "" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/ltrace_nvenc.txt") 行 → $OUTDIR/ltrace_nvenc.txt"

  echo ""
  echo "=== 测试 1c: ltrace + CUDA hwaccel 上下文 ==="
  ltrace -o "$OUTDIR/ltrace_hwaccel.txt" \
    -e '@libnvidia-encode*' \
    ffmpeg_cmd "-hwaccel cuda -hwaccel_output_format cuda" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/ltrace_hwaccel.txt") 行 → $OUTDIR/ltrace_hwaccel.txt"

  echo ""
  echo "=== 测试 1d: ltrace 追踪 cuCtx / cuDevice / cuInit ==="
  ltrace -o "$OUTDIR/ltrace_cuda.txt" \
    -e '@libcuda*' \
    ffmpeg_cmd "" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/ltrace_cuda.txt") 行 → $OUTDIR/ltrace_cuda.txt"
else
  echo "[!] ltrace 未安装 — 跳过测试 1a-1d"
  echo "    安装: apt install ltrace 或 yum install ltrace"
fi

# ════════════════════════════════════════════════════════════════════════════
# 测试 2: strace — 系统调用层面 (NVENC 最终通过 ioctl 与内核通信)
# ════════════════════════════════════════════════════════════════════════════
if [ "$HAVE_STRACE" -eq 1 ]; then
  echo ""
  echo "=== 测试 2a: strace ioctl 调用 ==="
  strace -e ioctl -o "$OUTDIR/strace_ioctl.txt" \
    ffmpeg_cmd "" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/strace_ioctl.txt") 行 → $OUTDIR/strace_ioctl.txt"

  echo ""
  echo "=== 测试 2b: strace open/openat (看 FFmpeg 打开了哪些 .so) ==="
  strace -e openat,open -o "$OUTDIR/strace_open.txt" \
    ffmpeg_cmd "" 2>&1 | tail -3
  echo "[*] $(wc -l < "$OUTDIR/strace_open.txt") 行 → $OUTDIR/strace_open.txt"
else
  echo "[!] strace 未安装 — 跳过测试 2a-2b"
fi

# ════════════════════════════════════════════════════════════════════════════
# 测试 3: LD_DEBUG — 动态链接器符号绑定
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== 测试 3: LD_DEBUG=bindings (nvidia 相关) ==="
LD_DEBUG=bindings ffmpeg_cmd "" 2>&1 \
  | grep -i -E 'nvenc|nvidia-encode|nvcuda|nvcuvid' \
  > "$OUTDIR/ld_debug.txt" || true
echo "[*] $(wc -l < "$OUTDIR/ld_debug.txt") 行 → $OUTDIR/ld_debug.txt"

# ════════════════════════════════════════════════════════════════════════════
# 测试 4: ldd — 确认 FFmpeg 对 NVENC 的链接方式
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== 测试 4: ldd FFmpeg (nvidia 相关) ==="
FFMPEG_PATH="$(command -v ffmpeg)"
echo "[*] ffmpeg: $FFMPEG_PATH"
ldd "$FFMPEG_PATH" 2>/dev/null | grep -i -E 'nvidia|nvenc|cuda|cuvid' \
  > "$OUTDIR/ldd_ffmpeg.txt" || echo "(无静态链接的 nvidia 库)" > "$OUTDIR/ldd_ffmpeg.txt"
cat "$OUTDIR/ldd_ffmpeg.txt"

# ════════════════════════════════════════════════════════════════════════════
# 测试 5: 检查 libnvidia-encode.so 导出的所有符号
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== 测试 5: libnvidia-encode.so 导出符号 ==="
LIBNVENC=$(ldconfig -p 2>/dev/null | grep libnvidia-encode.so.1 | head -1 | awk '{print $NF}')
if [ -z "$LIBNVENC" ]; then
  # fallback paths
  for p in /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 \
           /usr/lib64/libnvidia-encode.so.1 \
           /usr/lib/libnvidia-encode.so.1; do
    [ -f "$p" ] && LIBNVENC="$p" && break
  done
fi
if [ -n "$LIBNVENC" ] && command -v nm >/dev/null 2>&1; then
  echo "[*] $LIBNVENC"
  nm -D "$LIBNVENC" 2>/dev/null | grep -i ' T \| t ' > "$OUTDIR/nvenc_symbols.txt" || true
  echo "[*] $(wc -l < "$OUTDIR/nvenc_symbols.txt") 个导出符号 → $OUTDIR/nvenc_symbols.txt"
  # 显示前 30 个符号
  head -30 "$OUTDIR/nvenc_symbols.txt"
else
  echo "[!] 找不到 libnvidia-encode.so.1 或 nm 不可用"
fi

# ════════════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════"
echo "  输出目录: $OUTDIR"
echo "  文件列表:"
ls -la "$OUTDIR"
echo ""
echo "  ★ 重点查看:"
echo "    - ltrace_nvenc.txt    (NVENC API 调用序列)"
echo "    - ltrace_cuda.txt     (CUDA 驱动调用)"
echo "    - ltrace_hwaccel.txt  (CUDA hwaccel 路径)"
echo "    - strace_open.txt     (打开了哪些 .so)"
echo "═══════════════════════════════════════════"
