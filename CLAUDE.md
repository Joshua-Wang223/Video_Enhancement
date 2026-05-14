# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Video Enhancement is a GPU-accelerated video processing pipeline that integrates **IFRNet frame interpolation** (2x–16x frame rate) and **Real-ESRGAN super-resolution** (2x/4x resolution). It runs on NVIDIA GPUs with optional TensorRT, FP16, CUDA Graph, and torch.compile acceleration.

## Main entry point

```bash
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output.mp4
```

**`src/main_video_optimized.py`** is the only active entry point. All other `src/main_video_*.py` files are historical/legacy.

Two processing modes:
- `interpolate_then_upscale` (default) — interpolate at original resolution first, then upscale
- `upscale_then_interpolate` — upscale first, then interpolate on higher-res frames

To skip a stage: `--skip-interpolate` or `--skip-upscale`. Common flags: `--use-tensorrt-ifrnet`, `--use-tensorrt-esrgan`, `--face-enhance`, `--batch-mode`, `--dry-run`.

## Architecture

```
src/main_video_optimized.py          # CLI entry, orchestration, VideoProcessor
  ├── src/processors/ifrnet_processor_v6_1_single.py       # IFRNet processor
  │     └── external/IFRNet/process_video_v6_3_5_single.py #   IFRNet backend (v6.3.5)
  ├── src/processors/realesrgan_processor_video_optimized.py  # Real-ESRGAN processor
  │     └── external/realesrgan_video/main.py            #   Real-ESRGAN backend (v6.4)
  ├── src/utils/config_manager.py   # Config loading, path derivation, CLI override
  └── src/utils/video_utils.py      # Split, merge, audio extraction, VideoInfo
```

**Data flow (interpolate_then_upscale):**
Input → extract audio → split into segments → IFRNet interpolate each segment → pass segment list directly to Real-ESRGAN upscale each segment (no intermediate merge) → merge all segments → mux audio → output

The "direct segment passthrough" optimization skips the intermediate merge+re-split step, saving ~30% I/O.

## Configuration is the single source of truth

**`config/default_config.json`** holds all defaults. When the JSON and README/code comments disagree, the JSON wins. `src/utils/config_manager.py` (`Config` class) loads JSON, derives paths from `base_dir` upward, and applies CLI overrides.

Key config sections: `processing` (mode, factors, segment duration), `paths` (auto-derived from `base_dir`), `models.ifrnet`, `models.realesrgan`, `output`, `temp_files`, `logging`.

The JSON uses `"// key"` convention for documentation comments — these keys are ignored at runtime.

## Which files are current vs. historical

This project has accumulated many versioned scripts. Only these are active:

| Purpose | Current file |
|---------|-------------|
| Main entry | `src/main_video_optimized.py` |
| IFRNet processor | `src/processors/ifrnet_processor_v6_1_single.py` |
| IFRNet backend | `external/IFRNet/process_video_v6_3_5_single.py` |
| Real-ESRGAN processor | `src/processors/realesrgan_processor_video_optimized.py` |
| Real-ESRGAN backend | `external/realesrgan_video/main.py` |
| Config manager | `src/utils/config_manager.py` |
| Video utils | `src/utils/video_utils.py` |

Everything else in `external/IFRNet/process_video_v*.py`, `external/Real-ESRGAN/inference_realesrgan_video_v*.py`, `src/main_video_v*.py`, `src/processors/*_v[1-5]*.py` is historical or reference-only. Files with `_bak` or ` - Copy` suffix are development backups safe to delete.

## IFRNet backend (v6.3.5)

`external/IFRNet/process_video_v6_3_5_single.py` is a ~2500-line single-GPU script. Key internal architecture:

- **Three-stage pipeline**: T1 (Reader: NVDEC decode + frame prep → pair_queue), T2 (GPU: IFRNet model inference), T3 (Writer: FFmpeg H.264 encode from result_queue)
- Uses dual CUDA transfer streams (`stream_h2d` for prefetch, `stream_d2h` for output) with CudaEventPool
- `FFmpegFrameReader` reads via ffmpeg pipe with internal prefetch queue
- `FFmpegWriter` writes via ffmpeg stdin pipe
- GPU monitoring thread samples utilization every 2s
- Adaptive queue tuning post-segment (pair_queue / result_queue sizing)
- TRT engine cached under `.trt_cache/` with GPU SM-architecture in filename

The multi-GPU variant `process_video_v6_3_3.py` (historical) has equivalent code but with multi-GPU distribution.

## Real-ESRGAN backend (v6.4, realesrgan_video)

`external/realesrgan_video/` is a modular subproject with:
- `main.py` — entry, `create_video_enhancer()` / `run_pipeline_for_video()` for multi-segment engine reuse
- `pipeline.py` — 4-level parallel pipeline (read → SR → GFPGAN → write)
- `ffmpeg_io.py` — FFmpeg reader/writer with async prefetch
- `tensorrt_accel.py` — TRT acceleration wrapper
- `gfpgan_subprocess.py` — GFPGAN in isolated subprocess (optional TRT)
- `face_utils.py` — face detection and enhancement
- `config.py` — model paths resolved relative to project root

## TRT engine caching

Both IFRNet and Real-ESRGAN share `trt_cache_dir` (default `base_dir/.trt_cache`). Engine filenames encode model name, batch size, resolution, FP16 mode, and GPU SM architecture — so different configurations produce different caches. Once built, engines are reused across segments and across video runs. Cache rebuilt automatically when GPU SM changes.

## Dependencies

PyTorch must be installed manually first (matching CUDA version), then `pip install -r requirements.txt`. TensorRT components (`tensorrt`, `pycuda`, `onnx`, `onnxruntime-gpu`) are optional. Do NOT call `pycuda.autoinit` — it conflicts with PyTorch's CUDA context.

## OOM handling

Both processors auto-degrade: on CUDA OOM, batch_size is halved and retried, down to 1. The reduced value is persisted to `max_batch_size` in config. For Real-ESRGAN, `tile_size` can also be reduced (e.g., 512 or 256).

## Checkpoint/resume

Each processor maintains `temp/{video_name}_ifrnet/checkpoint.json` and `temp/{video_name}_esrgan/checkpoint.json`. Re-running the same command skips completed segments. Delete the checkpoint file to force re-processing.

## Shell environment

This project runs on Windows (PowerShell). Paths use backslashes. FFmpeg must be in PATH. Python 3.9+ with CUDA-capable PyTorch.
