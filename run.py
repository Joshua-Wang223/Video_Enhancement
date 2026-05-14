#!/usr/bin/env python3
"""
仓库根目录便捷入口 —— 等价于在 ``src/`` 下执行 ``main_video_optimized.py``。

底层链路：IFRNet（``external/IFRNet/process_video_v6_3_5_single.py``）+
Real-ESRGAN（``external/realesrgan_video``）。

**用法**

- 本脚本将 ``sys.argv`` 原样交给 ``main_video_optimized.main()``；任意 CLI 与主模块一致。
- 默认配置文件：``config/default_config.json``（可用 ``-c`` 覆盖）。未在命令行指定的选项以该 JSON 为准。
- 查看全部参数：::

    python run.py --help

  （Windows 下一般为 ``python run.py --help``）

---------------------------------------------------------------------------
代表性 CLI 示例（在项目根目录执行）
---------------------------------------------------------------------------

**1. 基础 — 先插帧再超分（默认顺序；配置文件可用默认值）**::

    python run.py -i input.mp4 -o output/out.mp4

**2. 显式指定配置**::

    python run.py -c config/default_config.json -i clips/a.mp4 -o output/a_enhanced.mp4

**3. 先超分再插帧（低分辨率真人 / 需尽早放大像素）**::

    python run.py -i in.mp4 -o out.mp4 --mode upscale_then_interpolate \\
        --interpolation-factor 2 --upscale-factor 2

**4. 自定义插帧 / 超分倍数**::

    python run.py -i in.mp4 -o out.mp4 --interpolation-factor 4 --upscale-factor 4

**5. 仅超分（跳过 IFRNet）+ 人脸增强**::

    python run.py -i face.mp4 -o face_4x.mp4 --skip-interpolate \\
        --face-enhance --gfpgan-model 1.4 --face-det-threshold 0.7

**6. 仅插帧（跳过 Real-ESRGAN）**::

    python run.py -i lowfps.mp4 -o smooth.mp4 --skip-upscale --interpolation-factor 4

**7. 动漫素材 — 换动漫专用超分模型**::

    python run.py -i anime.mp4 -o anime_up.mp4 \\
        --esrgan-model realesr-animevideov3 --interpolation-factor 2 --upscale-factor 2

**8. TensorRT（IFRNet + ESRGAN）+ 统一 Engine 缓存目录**::

    python run.py -i in.mp4 -o out.mp4 \\
        --use-tensorrt-ifrnet --use-tensorrt-esrgan \\
        --trt-cache-dir ./.trt_cache

**9. 全流程 TRT（含 GFPGAN TRT 子进程；需开启人脸增强）**::

    python run.py -i in.mp4 -o out.mp4 \\
        --use-tensorrt-ifrnet --use-tensorrt-esrgan \\
        --face-enhance --gfpgan-trt \\
        --trt-cache-dir ./.trt_cache

**10. 预去噪 + 插帧 + 超分**::

    python run.py -i noisy.mp4 -o clean.mp4 \\
        --denoise --denoise-model nafnet --denoise-strength-pre 0.5

**11. 显存紧张 — 分块 + 小批量 + 关闭部分图优化**::

    python run.py -i in.mp4 -o out.mp4 \\
        --tile-size 512 --batch-size-esrgan 2 \\
        --no-cuda-graph-esrgan --no-fp16-esrgan

**12. 批量目录处理**::

    python run.py --batch-mode --input-dir ./videos_raw --output-dir ./videos_out

**13. 分段时长（减轻单段峰值显存）**::

    python run.py -i heavy.mp4 -o out.mp4 --segment-duration 10

**14. 完成后自动删除临时分段**::

    python run.py -i in.mp4 -o out.mp4 --auto-cleanup

**15. Dry-run — 只打印环境与配置，不跑流水线**::

    python run.py -i in.mp4 -o out.mp4 --dry-run

**16. 性能报告（示例路径可按需修改）**::

    python run.py -i in.mp4 -o out.mp4 \\
        --report-ifrnet logs/ifr_perf.json \\
        --report-esrgan logs/esr_perf.json \\
        --report logs/run_summary.json

**17. 调试底层 FFmpeg/SR 日志（关闭静默）**::

    python run.py -i in.mp4 -o out.mp4 --no-quiet-ifrnet --no-quiet-esrgan

---------------------------------------------------------------------------

说明：早期 ``src/main.py`` 等与当前处理器接口不一致，请勿与新流水线混用。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from main_video_optimized import main

if __name__ == "__main__":
    sys.exit(main())
