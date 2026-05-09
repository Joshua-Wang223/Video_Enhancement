#!/usr/bin/env bash
# 快速启动（Linux/macOS）：依赖仓库根目录下的 config/default_config.json 与 src/main_video_optimized.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Video Enhancement — 快速检查"
echo "=============================================="

if ! command -v python3 &>/dev/null; then
  echo "❌ 未找到 python3"
  exit 1
fi
echo "✅ Python: $(python3 --version)"

if ! command -v ffmpeg &>/dev/null; then
  echo "❌ 未安装 FFmpeg，请先安装:"
  echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
  echo "   macOS: brew install ffmpeg"
  exit 1
fi
echo "✅ FFmpeg: $(ffmpeg -version | head -n 1)"

CFG="${SCRIPT_DIR}/config/default_config.json"
if [[ ! -f "$CFG" ]]; then
  echo "⚠️  未找到配置文件: $CFG"
else
  echo "✅ 配置: $CFG"
fi

echo ""
echo "📦 检查 PyTorch…"
if ! python3 -c "import torch" 2>/dev/null; then
  echo "⚠️  PyTorch 未安装。请先按 README.md 安装 CUDA 对应 wheel，再执行:"
  echo "   pip install -r ${SCRIPT_DIR}/requirements.txt"
else
  echo "✅ 已检测到 torch"
fi

echo ""
echo "🔍 模型路径（与 README / default_config 约定一致）…"
IFRNET_SMALL="${SCRIPT_DIR}/models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth"
ESRGAN_DEF="${SCRIPT_DIR}/models_RealESRGAN/realesr-general-x4v3.pth"
ok=true
if [[ ! -f "$IFRNET_SMALL" ]]; then
  echo "⚠️  未找到 IFRNet Small: $IFRNET_SMALL"
  ok=false
fi
if [[ ! -f "$ESRGAN_DEF" ]]; then
  echo "⚠️  未找到 Real-ESRGAN 默认权重: $ESRGAN_DEF"
  ok=false
fi
if [[ "$ok" == true ]]; then
  echo "✅ 默认模型文件存在"
fi

echo ""
echo "=============================================="
echo "运行示例（在项目根目录执行）"
echo "=============================================="
echo ""
echo "  # 单文件（输出路径含文件名）"
echo "  python3 src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4"
echo ""
echo "  # 批量：扫描目录 → 输出目录"
echo "  python3 src/main_video_optimized.py -c config/default_config.json \\"
echo "    --batch-mode --input-dir ./videos --output-dir ./enhanced"
echo ""
echo "  # 仅打印配置（不跑流水线）"
echo "  python3 src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4 --dry-run"
echo ""
