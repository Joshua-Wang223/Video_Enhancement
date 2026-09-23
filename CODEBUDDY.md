# CODEBUDDY.md

This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

Video Enhancement is a GPU-accelerated video processing pipeline integrating **IFRNet frame interpolation** (2x–16x frame rate) and **Real-ESRGAN super-resolution** (2x/4x). Runs on NVIDIA GPUs with optional TensorRT, FP16, CUDA Graph, and torch.compile.

Two processing modes:
- `interpolate_then_upscale` (default) — interpolate at original resolution, then upscale
- `upscale_then_interpolate` — upscale first, then interpolate on higher-res frames

## Entry Point & Key Commands

**Main entry:**
```bash
cd Video_Enhancement
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output.mp4
```
Also: `python run.py -i input.mp4 -o output.mp4`

**Useful flags:**
- `--skip-interpolate` or `--skip-upscale` — skip one stage
- `--use-tensorrt-ifrnet` / `--use-tensorrt-esrgan` — TensorRT acceleration
- `--face-enhance` — enable GFPGAN face enhancement
- `--batch-mode` — batch process files in input directory
- `--dry-run` — validate config without running
- `--mode upscale_then_interpolate` — swap processing order

**Setup & dependencies:**
```bash
# Install PyTorch manually first (match CUDA version):
pip install "torch>=2.7.0" "torchvision>=0.22.0" --index-url https://download.pytorch.org/whl/cu128
# Then install project deps:
pip install -r requirements.txt
# Optional TensorRT:
# pip install tensorrt pycuda onnx onnxruntime-gpu
```

## Architecture

```
CLI layer (src/main_video_optimized.py)
  ├── IFRNet Processor (src/processors/ifrnet_processor_video_optimized.py)
  │     └── IFRNet Backend (external/ifrnet_video/ — modular package, v6.4.5.1)
  │           ├── main.py        — IFRNetVideoProcessor 入口
  │           ├── pipeline.py    — 3-thread pipeline: T1(NVDEC+prefetch) → T2(GPU inference) → T3(NVENC encode)
  │           ├── nvenc_sdk.py   — NVENC GPU-direct SDK wrapper (encoder + encode thread)
  │           └── tensorrt_accel.py / ffmpeg_io.py / config.py / ifrnet_utils.py
  ├── Real-ESRGAN Processor (src/processors/realesrgan_processor_video_optimized.py)
  │     └── Real-ESRGAN Backend (external/realesrgan_video/main.py + pipeline.py + nvenc_sdk.py)
  │           └── 4-stage pipeline: read → SR inference → GFPGAN(optional) → write
  └── Utils (src/utils/config_manager.py, video_utils.py)
```

Both backends are **modular packages** that mirror each other (`ifrnet_video` / `realesrgan_video`), so a fix on one side usually has a counterpart on the other — check both.

The outer processor layers handle segment management, checkpoint/resume, and OOM degradation. Backend layers handle raw GPU pipeline. The "direct segment passthrough" optimization skips intermediate merge+re-split, saving ~30% I/O.

## Configuration

**`config/default_config.json`** is the single source of truth for defaults (overrides README and code comments). `src/utils/config_manager.py` loads it, derives paths from `base_dir` upward, and applies CLI overrides. JSON uses `"// key"` convention for documentation comments (ignored at runtime).

## Current vs. Historical Files

Only these files are active:

| Purpose | Active File |
|---------|-------------|
| Main entry | `src/main_video_optimized.py` |
| IFRNet processor | `src/processors/ifrnet_processor_video_optimized.py` |
| IFRNet backend | `external/ifrnet_video/main.py` (modular package, v6.4.5.1) |
| Real-ESRGAN processor | `src/processors/realesrgan_processor_video_optimized.py` |
| Real-ESRGAN backend | `external/realesrgan_video/main.py` |
| Config manager | `src/utils/config_manager.py` |
| Video utils | `src/utils/video_utils.py` |

**Historical — no production imports of any of these:**
- `external/IFRNet/process_video_v*.py` (25 files) — pre-split monoliths. `process_video_v6_4_5_1_single.py` (8756 lines) is the exact pre-split source of the `external/ifrnet_video/` package; remaining references to it are doc comments and fix-provenance notes only.
- `src/main_video_v*.py`, `src/processors/*_v[1-5]*.py`

Files with `_bak` or ` - Copy` suffix are safe to delete.

> **⚠️ Traps when locating code (both have cost real debugging time):**
> 1. The `v6.4.5.1` string in logs comes from the **processor layer**, NOT from a backend filename. Do not grep for `process_video_v6_4_5_1_single.py` and assume it is the running code — the active backend is the `external/ifrnet_video/` **package**.
> 2. Always confirm the real call chain (`main_video_optimized.py` → `processors/*_processor_video_optimized.py` → `external/*_video/`) before editing. Patching the historical single file silently does nothing.
> 3. Verify file paths still exist before trusting this table — it has gone stale before.

## OOM Handling

Both processors auto-degrade on CUDA OOM: batch_size is halved and retried down to 1. Persisted to `max_batch_size` in config. For Real-ESRGAN, `tile_size` can also be reduced (e.g., 512 or 256).

## Checkpoint/Resume

Checkpoints live under `temp/{stage}/{prefix}_{video_name}/checkpoint.json`, where `prefix` differs between the first and second pass:

- IFRNet: `temp/ifrnet/ifrnet_source_{video}/checkpoint.json` (second pass: `ifrnet_from_segments_{video}`)
- Real-ESRGAN: `temp/esrgan_video/esrgan_video_{video}/checkpoint.json` (second pass: `esrgan_from_segments_{video}`)

Re-running the same command skips completed segments. Delete the checkpoint file — or the whole `temp/{stage}/{prefix}_{video}/` directory — to force re-processing.

## TRT Engine Caching

Both IFRNet and Real-ESRGAN share `.trt_cache/` under the project root. Engine filenames encode model name, batch size, resolution, FP16 mode, and GPU SM architecture. Engines are reused across segments and video runs; rebuilt automatically when GPU SM changes.

## GPU & Environment Notes

- Do NOT call `pycuda.autoinit` — it conflicts with PyTorch's CUDA context.
- Python 3.9+, NVIDIA GPU with CUDA-capable PyTorch, FFmpeg in PATH.
- For CloudStudio/T4 environments: NVENC GPU direct encoding via ctypes may fail if the `_NvEncOpenEncodeSessionExParams` struct layout doesn't match the driver's expectation — verify field order and version macros against the NVENC SDK.

## Debugging Hangs (3-thread pipeline)

The IFRNet/Real-ESRGAN backends run a 3-thread pipeline (T1 prefetch → T2 inference → T3 encode) plus a writer thread. When it hangs, **symptom logs point at the wrong thread**. Do not reason from log ordering alone.

1. **Attach py-spy first:**
   ```bash
   py-spy dump --pid <PID> --locals
   ```
   - `idle` + `queue.put/get` → the thread is merely *blocked*; find who stopped consuming.
   - `active` inside a ctypes call (e.g. `lock_bs_fn`) → blocked **inside the driver**. No Python-level timeout can break this.
2. **A "writer thread did not exit in Ns" message is a second-hand symptom.** The real hang is usually the encode thread. If it is stuck inside `LockBitstream` rather than raising, `self.error` stays `None` and the `flush_and_join()` timeout warning will **never** print — its absence is not evidence that the encoder is healthy.
3. **HEVC/AV1 + lookahead: `LockBitstream` on a not-ready slot blocks forever in the driver, with both `doNotWait=0` and `doNotWait=1`.** The only reliable protection is a software-side guarantee to never lock a slot that is not ready. Measured LA output latency is `la_depth + 1`, so the physical slot count must be at least `la_depth + 2` — with only `la_depth + 1` slots the margin is zero and you get a submit↔drain circular dependency (see `memory/ifrnet-hevc-la-slot-headroom-deadlock.md`).

## Memory Files Mirror Sync (mandatory, confirmed 2026-09-08)

`memory/` exists in **two mirrored locations**. After editing **either** side, sync the other immediately — never leave them diverged.

| | Path | Role |
|---|---|---|
| A | `/workspace/Video_Enhancement/memory` | canonical, in-repo `memory/` |
| B | `/root/.codebuddy/projects/workspace-Video_Enhancement/memory` | CodeBuddy session side (auto-loaded) |

```bash
A=/workspace/Video_Enhancement/memory
B=/root/.codebuddy/projects/workspace-Video_Enhancement/memory
cp -a "$A/." "$B/"     # A → B, overwrite by name (reverse for B → A)
diff -r "$A" "$B" && echo "✅ in sync"
```

Rules:
- Sync after **every** create/modify under `memory/`, including `MEMORY.md` index entries.
- **`cp -a` does not propagate deletions.** If you delete/rename a file on one side, the stale copy survives on the other. Use `rsync -a --delete "$A/" "$B/"` (or remove it on both sides) whenever a file is deleted or renamed.
  ⚠️ `--delete` is destructive: **always preview first** with
  `rsync -a --delete --dry-run "$A/" "$B/"` and read the to-be-deleted list; only drop `--dry-run`
  after confirming nothing will be lost.
  ⚠️ **`rsync` is NOT installed in this container** (verified 2026-09-08). Dependency-free fallback —
  list the differences, confirm by eye, then delete manually:
  `comm -13 <(cd "$A" && ls -1|sort) <(cd "$B" && ls -1|sort)`  # only in B
  `comm -23 <(cd "$A" && ls -1|sort) <(cd "$B" && ls -1|sort)`  # only in A
- Chinese must be written with a UTF-8-capable direct-write tool (Write/Edit). **Never** write Chinese through a shell pipe/heredoc — a past incident (2026-08-24) corrupted Chinese into literal `?` irreversibly. After writing, spot-check that `0x3F` bytes adjacent to CJK characters is 0.
- Historical mirrors (Windows dev machine, paths dead, archive only):
  `D:\Workspace_Python\Video_Enhancement\Video_Enhancement\memory`,
  `C:\Users\Administrator\.claude\projects\D--Workspace-Python-Video-Enhancement-Video-Enhancement\memory`
