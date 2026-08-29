#!/usr/bin/env python3
"""
IFRNet 版本对比基准测试

对同一视频分别调用 v6.3.5 / v6.4.1–v6.4.5.2 进行插帧测试，
收集耗时、FPS、输出帧数、文件大小、GPU 利用率等指标，输出对比表格。

用法:
    python benchmark_ifrnet_versions.py -i test.mp4 -o benchmark_output/
    python benchmark_ifrnet_versions.py -i test.mp4 -o benchmark_output/ --scale 4 --model IFRNet_L_Vimeo90K
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 脚本在 tests/ 子目录，需回退一层到项目根
IFRNET_DIR = PROJECT_ROOT / "external" / "IFRNet"
MODELS_DIR = PROJECT_ROOT / "models_IFRNet" / "checkpoints"

VERSIONS = [
    "process_video_v6_3_5_single",
    "process_video_v6_4_1_single",
    "process_video_v6_4_2_single",
    "process_video_v6_4_3_single",
    "process_video_v6_4_3_1_single",
    "process_video_v6_4_3_2_single",
    "process_video_v6_4_4_single",
    "process_video_v6_4_4_1_single",
    "process_video_v6_4_4_2_single",
    "process_video_v6_4_5_single",
    "process_video_v6_4_5_1_single",
    "process_video_v6_4_5_2_single",
]

VERSION_LABELS = {
    "process_video_v6_3_5_single": "v6.3.5",
    "process_video_v6_4_1_single": "v6.4.1",
    "process_video_v6_4_2_single": "v6.4.2",
    "process_video_v6_4_3_single": "v6.4.3",
    "process_video_v6_4_3_1_single": "v6.4.3.1",
    "process_video_v6_4_3_2_single": "v6.4.3.2",
    "process_video_v6_4_4_single": "v6.4.4",
    "process_video_v6_4_4_1_single": "v6.4.4.1",
    "process_video_v6_4_4_2_single": "v6.4.4.2",
    "process_video_v6_4_5_single": "v6.4.5",
    "process_video_v6_4_5_1_single": "v6.4.5.1",
    "process_video_v6_4_5_2_single": "v6.4.5.2",
}


def find_model_path(model_name: str) -> str:
    model_file = MODELS_DIR / f"{model_name}.pth"
    if model_file.exists():
        return str(model_file)
    candidates = list(MODELS_DIR.glob(f"{model_name}*.pth"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        f"Model not found: {model_name}.pth in {MODELS_DIR}"
    )


def run_version(version_module: str, input_path: str, output_path: str,
                model_path: str, scale: float, batch_size: int,
                trt_cache_dir: str, codec: str | None = None,
                crf: int = 23, preset: str = "medium") -> dict:
    """Run a single IFRNet version in a subprocess and return timing results."""
    inner_script = f'''
import importlib
import json
import os
import subprocess as _sp
import sys
import threading
import time

_ifrnet_dir = {str(IFRNET_DIR)!r}
if _ifrnet_dir not in sys.path:
    sys.path.insert(0, _ifrnet_dir)

mod = importlib.import_module({version_module!r})

# 自动选择编码器，支持显式指定
# 参照底层 best_encoder()：libx264 → h264_nvenc，libx265 → hevc_nvenc
_codec_explicit = {codec!r}
if _codec_explicit:
    codec = _codec_explicit
else:
    codec = "libx264"
    try:
        codec = mod.HardwareCapability.best_encoder("libx264")
    except Exception:
        pass

proc = mod.IFRNetVideoProcessor(
    model_path={model_path!r},
    device="cuda",
    batch_size={batch_size},
    max_batch_size={batch_size * 2},
    use_fp16=True,
    use_compile=False,
    use_cuda_graph=False,
    use_tensorrt=True,
    trt_cache_dir={trt_cache_dir!r},
    use_hwaccel=True,
    codec=codec,
    crf={crf},
    x264_preset={preset!r},
    keep_audio=False,
    ffmpeg_bin="ffmpeg",
    quiet=True,
)

# ── 后台 GPU 监控（处理期间每秒采样，取平均值 + 峰值显存） ──────
_gpu_samples = []
_gpu_stop = threading.Event()

def _monitor_gpu():
    while not _gpu_stop.is_set():
        try:
            _p = _sp.run([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            if _p.returncode == 0:
                parts = _p.stdout.strip().split(",")
                if len(parts) >= 2:
                    _gpu_samples.append((
                        int(parts[0].strip()),
                        int(parts[1].strip()),
                    ))
        except Exception:
            pass
        _gpu_stop.wait(1.0)

_gpu_thread = threading.Thread(target=_monitor_gpu, daemon=True)
_gpu_thread.start()

t0 = time.perf_counter()

# v6.4.x 新增 total_segments/segment_index 参数（有默认值，向后兼容）
ok = proc.process_video(
    input_path={input_path!r},
    output_path={output_path!r},
    scale={scale},
    preview=False,
    preview_interval=30,
)

elapsed = time.perf_counter() - t0

# 停止 GPU 监控，汇总采样数据
_gpu_stop.set()
_gpu_thread.join(timeout=3)

result = {{
    "version": {version_module!r},
    "label": {VERSION_LABELS.get(version_module, version_module)!r},
    "success": ok,
    "elapsed_sec": round(elapsed, 3),
    "input_path": {input_path!r},
    "output_path": {output_path!r},
    "scale": {scale},
    "codec": getattr(proc, '_last_used_codec', codec),
    "model": {os.path.basename(model_path)!r},
}}

# 诊断数据：pipeline 各阶段帧/批次数（用于帧丢失排查）
_diag_keys = [
    '_diag_reader_pairs', '_diag_infer_batches',
    '_diag_infer_pairs', '_diag_writer_frames',
    '_diag_gpu_stay_batches', '_diag_nvenc_frames', '_diag_empty_h264',
    '_diag_lock_fallback',
    '_diag_active_level',
    '_diag_external_first_frame',
]
for _dk in _diag_keys:
    _dv = getattr(proc, _dk, None)
    if _dv is not None:
        result[_dk] = _dv

# [PHASE4] 额外从 NVENC encoder 收集 fallback 计数器 + 码率控制模式
_nvenc = getattr(proc, '_cached_nvenc_encoder', None)
if _nvenc is not None:
    _lfb = getattr(_nvenc, '_diag_lock_fallback', 0)
    if _lfb > 0:
        result.setdefault('_diag_lock_fallback', 0)
        result['_diag_lock_fallback'] = max(result['_diag_lock_fallback'], _lfb)
    # NVENC 码率控制模式（权威来源，不依赖 ffprobe 推测）
    _rm = getattr(_nvenc, '_rate_mode', None)
    if _rm:
        _label_map = {{"constqp": "CONSTQP", "vbr_hq": "VBR_HQ", "qvbr": "QVBR"}}
        result["output_rc_mode"] = _label_map.get(_rm, _rm.upper())
    else:
        result["output_rc_mode"] = "CONSTQP"
else:
    # ctypes NVENC encoder 引用不可达时的回退检测链
    # 1) 尝试从模块级常量 _NVENC_LEVEL1_RATE_MODE 获取 (v6.4.3.1+)
    _rm = getattr(mod, '_NVENC_LEVEL1_RATE_MODE', None)
    if _rm:
        _label_map = {{"constqp": "CONSTQP", "vbr_hq": "VBR_HQ", "qvbr": "QVBR"}}
        result["output_rc_mode"] = _label_map.get(_rm, _rm.upper())
    else:
        # 2) ffprobe 回退：区分 NVENC 管道 vs 软件编码
        _codec_str = getattr(proc, '_last_used_codec', codec)
        if 'nvenc' in (_codec_str or '').lower():
            result["output_rc_mode"] = "NVENC"
        else:
            result["output_rc_mode"] = "SW"

# GPU 利用率（处理期间平均）和峰值显存
if _gpu_samples:
    avg_util = sum(s[0] for s in _gpu_samples) / len(_gpu_samples)
    peak_mem = max(s[1] for s in _gpu_samples)
    result["gpu_util_pct"] = round(avg_util, 1)
    result["gpu_mem_used_mb"] = peak_mem
    result["gpu_samples"] = len(_gpu_samples)
else:
    # nvidia-smi 不可用时的回退快照
    try:
        _p = _sp.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=10)
        if _p.returncode == 0:
            parts = _p.stdout.strip().split(",")
            if len(parts) >= 1:
                result["gpu_util_pct"] = int(parts[0].strip())
            if len(parts) >= 2:
                result["gpu_mem_used_mb"] = int(parts[1].strip())
    except Exception:
        pass

# 收集输出文件信息
if ok and os.path.exists({output_path!r}):
    try:
        st = os.stat({output_path!r})
        result["output_size_mb"] = round(st.st_size / (1024 * 1024), 2)
        result["output_size_bytes"] = st.st_size
    except OSError:
        pass

    # 用 ffprobe 获取输出帧数（packet 级，可靠的主计数）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=30)
        if _p.returncode == 0 and _p.stdout.strip():
            val = _p.stdout.strip()
            if val != "N/A":
                result["output_frames"] = int(val)
    except Exception:
        pass

    # 容器级帧数（nb_frames 优先，无需解码；不可用时由帧率×时长推算）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=15)
        if _p.returncode == 0 and _p.stdout.strip():
            val = _p.stdout.strip()
            if val not in ("N/A", "", "0"):
                result["output_frames_nf"] = int(val)
    except Exception:
        pass

    # nb_frames 不可用时的回退：r_frame_rate × duration
    if "output_frames_nf" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip():
                parts = _p.stdout.strip().split(",")
                if len(parts) >= 2:
                    from fractions import Fraction as _Frac
                    try:
                        _rfr = float(_Frac(parts[0].strip()))
                        _dur = float(parts[1].strip())
                        if _rfr > 0 and _dur > 0:
                            result["output_frames_nf"] = int(round(_rfr * _dur))
                    except (ValueError, ZeroDivisionError):
                        pass
        except Exception:
            pass

    # 输出视频平均码率（stream 级优先，format 级回退）
    try:
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=15)
        if _p.returncode == 0 and _p.stdout.strip():
            _val = _p.stdout.strip()
            if _val not in ("N/A", "", "0"):
                result["output_bitrate_kbps"] = round(int(_val) / 1000, 1)
    except Exception:
        pass

    if "output_bitrate_kbps" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=bit_rate",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip():
                _val = _p.stdout.strip()
                if _val not in ("N/A", "", "0"):
                    result["output_bitrate_kbps"] = round(int(_val) / 1000, 1)
        except Exception:
            pass

    # 码率控制模式回退：仅当无 NVENC encoder 时尝试 ffprobe
    if "output_rc_mode" not in result:
        try:
            _p = _sp.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate_mode",
                "-of", "csv=p=0", {output_path!r}
            ], capture_output=True, text=True, timeout=15)
            if _p.returncode == 0 and _p.stdout.strip() in ("0", "1", "2"):
                _mode_map = {{"0": "CBR", "1": "VBR", "2": "ABR"}}
                result["output_rc_mode"] = _mode_map[_p.stdout.strip()]
        except Exception:
            pass

# 收集输入帧数
try:
    _p = _sp.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", {input_path!r}
    ], capture_output=True, text=True, timeout=30)
    if _p.returncode == 0 and _p.stdout.strip():
        result["input_frames"] = int(_p.stdout.strip())
except Exception:
    pass

# 计算 FPS
if ok and elapsed > 0 and "output_frames" in result:
    result["fps"] = round(result["output_frames"] / elapsed, 1)

print("__BENCHMARK_RESULT__", json.dumps(result), flush=True)
'''
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.run(
        [sys.executable, "-c", inner_script],
        stdout=subprocess.PIPE,       # 只捕获 stdout（JSON 结果路径）
        stderr=None,                  # stderr 直通终端，使 tqdm 能正常用 \r 刷新
        text=True, timeout=3600,
        env=env, cwd=str(PROJECT_ROOT),
    )

    # 解析输出中的 JSON 结果
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("__BENCHMARK_RESULT__"):
            try:
                return json.loads(line.split("__BENCHMARK_RESULT__", 1)[1].strip())
            except json.JSONDecodeError:
                pass

    # 未找到 JSON 结果
    return {
        "version": version_module,
        "label": VERSION_LABELS.get(version_module, version_module),
        "success": False,
        "error": "no_json_output",
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def _display_width(s: str) -> int:
    """字符串终端显示宽度（CJK 字符计 2，其余计 1）。"""
    w = 0
    for ch in s:
        if '一' <= ch <= '鿿' or '　' <= ch <= '〿' or '＀' <= ch <= '￯':
            w += 2
        else:
            w += 1
    return w


def _pad(s: str, width: int, align: str = '<') -> str:
    """按终端显示宽度填充字符串。"""
    dw = _display_width(s)
    pad = max(0, width - dw)
    if align == '<':
        return s + ' ' * pad
    elif align == '>':
        return ' ' * pad + s
    else:
        left = pad // 2
        return ' ' * left + s + ' ' * (pad - left)


def print_results(results: list[dict], input_path: str, scale: float,
                  batch_size: int, model: str, crf: int = 23, preset: str = "medium"):
    """Print comparison table."""
    H = ["版本", "耗时", "输入帧", "输出帧", "容器帧", "FPS", "文件大小", "码率模式", "平均码率", "GPU%", "显存", "编码器"]
    W = [10, 8, 8, 8, 8, 8, 10, 10, 9, 6, 6, 14]

    print()
    print("=" * 100)
    print("IFRNet 版本基准测试结果")
    print("=" * 100)
    print(f"  输入: {input_path}  |  插帧: {scale}x  |  batch-size: {batch_size}  |  模型: {model}  |  crf: {crf}  |  preset: {preset}")
    print()

    header = "  " + "  ".join(_pad(h, w, '<' if h in ("版本", "编码器") else '>') for h, w in zip(H, W))
    print(header)
    _main_sep_len = sum(W) + 2 * (len(W) - 1)
    print("  " + "-" * _main_sep_len)

    best_fps = 0.0
    best_label = ""

    for r in results:
        label = r.get("label", "?")
        if not r.get("success"):
            err = r.get("error", "unknown")
            print(f"  {label:<10} {'-- FAILED --':>8}  ({err})")
            continue

        elapsed = r.get("elapsed_sec", 0)
        in_frames = r.get("input_frames", "?")
        out_frames = r.get("output_frames", "?")
        container_frames = r.get("output_frames_nf", "?")
        fps = r.get("fps", 0)
        size_mb = r.get("output_size_mb", 0)
        bitrate_kbps = r.get("output_bitrate_kbps")
        rc_mode = r.get("output_rc_mode", "?")
        gpu_util = r.get("gpu_util_pct", "?")
        gpu_mem = r.get("gpu_mem_used_mb", "?")
        codec = r.get("codec", "?")

        elapsed_str = f"{elapsed:.1f}s" if elapsed else "?"
        in_str = str(in_frames) if in_frames else "?"
        out_str = str(out_frames) if out_frames else "?"
        container_str = str(container_frames) if container_frames else "?"
        fps_str = f"{fps:.1f}" if fps else "?"
        size_str = f"{size_mb:.1f} MB" if size_mb else "?"
        rc_mode_str = str(rc_mode) if rc_mode != "?" else "?"
        bitrate_str = f"{bitrate_kbps:.0f}kbps" if bitrate_kbps else "?"
        gpu_util_str = f"{gpu_util}%" if gpu_util != "?" else "?"
        gpu_mem_str = f"{gpu_mem}M" if isinstance(gpu_mem, (int, float)) else "?"

        cols = [label, elapsed_str, in_str, out_str, container_str, fps_str, size_str, rc_mode_str, bitrate_str, gpu_util_str, gpu_mem_str, codec]
        aligns = ['<', '>', '>', '>', '>', '>', '>', '>', '>', '>', '>', '<']
        row = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(cols, W, aligns))
        print(row)

        if isinstance(fps, (int, float)) and fps > best_fps:
            best_fps = fps
            best_label = label

    # 帧数交叉验证提示
    ok_results = [r for r in results if r.get("success")]
    for r in ok_results:
        pkts = r.get("output_frames")
        nf = r.get("output_frames_nf")
        if pkts and nf and pkts != nf:
            label = r.get("label", "?")
            delta = pkts - nf
            print(f"  ⚠️  {label}: nb_read_packets={pkts} ≠ nb_read_frames={nf} (diff={delta:+d})")

    print()
    if best_label:
        print(f"  最快版本: {best_label} ({best_fps:.1f} FPS)")

    # 版本间对比
    if len(ok_results) >= 2:
        fps_values = [r.get("fps", 0) for r in ok_results]
        slowest_fps = min(fps_values)
        if slowest_fps > 0:
            spread_pct = (best_fps - slowest_fps) / slowest_fps * 100
            print(f"  性能差异: {spread_pct:.1f}% (最快 vs 最慢)")
        times = [r.get("elapsed_sec", 0) for r in ok_results]
        print(f"  耗时范围: {min(times):.1f}s – {max(times):.1f}s")

    # 编码器一致性检查
    codecs = set(r.get("codec", "") for r in ok_results)
    if len(codecs) > 1:
        print(f"  ⚠️  编码器不一致: {codecs}")

    # ── 诊断数据: pipeline 帧计数交叉验证 (reader → infer → writer) ──
    _diag_versions = [r for r in ok_results if r.get("_diag_reader_pairs")]
    if _diag_versions:
        print()
        print("  ── Pipeline 帧计数诊断 ──")
        print("  管內期望 = infer对 × scale  (每 pair: T 个插值 + 1 个原始右帧)")
        print("  管內偏移 = writer帧 - 管內期望  (负=丢帧, 正=多帧)")
        print("  注: NVENC SDK 严格遵守帧数守恒, Output == Input；异常偏移需排查编码循环")
        print("  总期望 = 外部首帧 + 管內期望  (首帧 f0 不在 pipeline._written 计数内)")
        print("  文件Δ = ffprobe − 总期望  (0=一致, 负=文件缺帧)\n")

        _pH = ["版本", "reader对", "infer批", "infer对", "writer帧", "管內期望", "管內偏移", "总期望", "ffprobe", "文件Δ"]
        _pW = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        _pA = ['<', '>', '>', '>', '>', '>', '>', '>', '>', '<']
        _dh = "  " + "  ".join(_pad(h, w, a) for h, w, a in zip(_pH, _pW, _pA))
        print(_dh)
        _sep_len = sum(_pW) + 2 * (len(_pW) - 1)
        print("  " + "-" * _sep_len)
        for r in _diag_versions:
            _label = r.get("label", "?")
            _rp = r.get("_diag_reader_pairs", 0)
            _ib = r.get("_diag_infer_batches", 0)
            _ip = r.get("_diag_infer_pairs", 0)
            _wf = r.get("_diag_writer_frames", 0)
            _scale_int = int(r.get("scale", 2.0))
            _out_pkts = r.get("output_frames")

            _pipe_expected = _ip * _scale_int if _ip > 0 else 0
            _pipe_delta = _wf - _pipe_expected if _pipe_expected > 0 else 0

            # 外部首帧：f0 不在 pipeline._written 内，需单独计入总期望
            _ext_first = r.get('_diag_external_first_frame', 1)
            _ext_default = '_diag_external_first_frame' not in r
            _total_expected = _ext_first + _pipe_expected if _pipe_expected > 0 else 0

            # 文件级校验: ffprobe 实际帧数 vs 总期望
            if _out_pkts is not None:
                _file_delta = _out_pkts - _total_expected
                _verify = "✓" if _file_delta == 0 else f"⚠ {_file_delta:+d}"
            else:
                _file_delta = None
                _verify = "?"

            _out_str = str(_out_pkts) if _out_pkts is not None else "?"
            _total_str = f"{_total_expected}" + ("[d]" if _ext_default else "")
            _dcols = [_label, str(_rp), str(_ib), str(_ip), str(_wf),
                      str(_pipe_expected), f"{_pipe_delta:+d}", _total_str, _out_str, _verify]
            _drow = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(_dcols, _pW, _pA))
            print(_drow)

        # ── NVENC 编码精度诊断 (精确数值, 不设阈值) ──
        _gpu_stay = [r for r in _diag_versions if r.get("_diag_gpu_stay_batches")]
        if _gpu_stay:
            print()
            print("  ── NVENC 编码精度诊断 ──")
            print("  编码输出帧 = 管道内产出 H.264 数据的帧总数 (含插值+原始+flush 恢复)")
            print("  管內偏移 = writer帧 − 管內期望 (负=丢帧)")
            print("  注: NVENC SDK 严格遵守帧数守恒, Output == Input；异常偏移需排查编码循环")
            print("  空H264 = encode 返回空且重试仍失败, 若 _prev_h264 可用可补偿")
            print("  文件Δ = ffprobe − 总期望 (负=编码/写入/混流环节帧丢失)\n")

            _nH = ["版本", "GPU批次", "编码输出帧", "空H264", "管內偏移", "Lock降级", "说明"]
            _nW = [10, 10, 10, 10, 10, 10, 30]
            _nA = ['<', '>', '>', '>', '>', '>', '<']
            _dh2 = "  " + "  ".join(_pad(h, w, a) for h, w, a in zip(_nH, _nW, _nA))
            print(_dh2)
            _sep2_len = sum(_nW) + 2 * (len(_nW) - 1)
            print("  " + "-" * _sep2_len)

            # 检测所有版本管內偏移是否一致 (一致=可能 LA 正常行为, 非 bug)
            _all_pipe_deltas = []
            for _r in _gpu_stay:
                __ip = _r.get("_diag_infer_pairs", 0)
                __wf = _r.get("_diag_writer_frames", 0)
                __sc = int(_r.get("scale", 2.0))
                __pe = __ip * __sc
                __pd = __wf - __pe if __pe > 0 else 0
                if __pd < 0:
                    _all_pipe_deltas.append(__pd)
            _all_same_delta = len(_all_pipe_deltas) >= 2 and len(set(_all_pipe_deltas)) == 1

            for r in _gpu_stay:
                _label = r.get("label", "?")
                _gs = r.get("_diag_gpu_stay_batches", 0)
                _nv = r.get("_diag_nvenc_frames", 0)
                _eh = r.get("_diag_empty_h264", 0)
                _lf = r.get("_diag_lock_fallback", 0)
                _scale_int2 = int(r.get("scale", 2.0))
                _ip2 = r.get("_diag_infer_pairs", 0)
                _wf2 = r.get("_diag_writer_frames", 0)
                _pipe_expected2 = _ip2 * _scale_int2
                _pipe_delta2 = _wf2 - _pipe_expected2 if _pipe_expected2 > 0 else 0

                # 构造精确说明
                _notes = []
                if _eh > 0:
                    _notes.append(f"空H264={_eh}")
                if _pipe_delta2 < 0:
                    if _all_same_delta:
                        _notes.append(f"管內缺{abs(_pipe_delta2)}帧(LA可能)")
                    else:
                        _notes.append(f"丢{abs(_pipe_delta2)}帧")
                if _lf > 0:
                    _notes.append(f"降级{_lf}")
                if not _notes:
                    _notes.append("正常")
                _note_str = ", ".join(_notes)

                # 附加诊断: NVENC调用 与 管內期望 是否一致
                _nv_vs_expected = _nv - _pipe_expected2 if _pipe_expected2 > 0 else 0
                if _nv_vs_expected != 0:
                    _note_str += f" (调用Δ{_nv_vs_expected:+d})"

                # 文件级交叉校验: ffprobe vs 总期望（NVENC→Writer→File 全链路）
                _out_pkts2 = r.get("output_frames")
                _ext_first2 = r.get('_diag_external_first_frame', 1)
                _total_expected2 = _ext_first2 + _pipe_expected2 if _pipe_expected2 > 0 else 0
                if _out_pkts2 is not None and _total_expected2 > 0:
                    _file_delta2 = _out_pkts2 - _total_expected2
                    if _file_delta2 != 0:
                        _note_str += f" [文件Δ{_file_delta2:+d}]"

                _ncols = [_label, str(_gs), str(_nv), str(_eh),
                          f"{_pipe_delta2:+d}", str(_lf), _note_str]
                _nrow = "  " + "  ".join(_pad(c, w, a) for c, w, a in zip(_ncols, _nW, _nA))
                print(_nrow)

    # ── 版本间输出帧数一致性 ──
    _out_frames_list = [r.get("output_frames") for r in ok_results if r.get("output_frames")]
    if len(set(_out_frames_list)) > 1:
        print()
        print("  ⚠️  版本间输出帧数不一致 (可能为真实帧丢失 bug):")
        _ref = max(_out_frames_list)
        for r in ok_results:
            _of = r.get("output_frames")
            if _of is not None:
                _label = r.get("label", "?")
                _delta = _of - _ref
                _pct = f" ({_delta:+d}, {_delta/_ref*100:+.1f}%)" if _delta != 0 else " (基准)"
                print(f"      {_label}: {_of} 帧{_pct}")

    # 编码级别诊断：独立于 Pipeline diag，即使缺少 _diag_reader_pairs 也显示
    _level_versions = [r for r in ok_results if r.get("_diag_active_level")]
    if _level_versions:
        _level_map = {1: "NVENC GPU直通", 2: "RingBuf+NVENC", 3: "RingBuf+软编码", 4: "标准路径"}
        print()
        print("  ── 编码级别诊断 ──")
        _dh_lv = "  {:<10} {:>5}  {}".format("版本", "级别", "说明")
        print(_dh_lv)
        print("  " + "-" * 50)
        for r in _level_versions:
            _label = r.get("label", "?")
            _lv = r.get("_diag_active_level", "?")
            _desc = _level_map.get(_lv, f"未知({_lv})")
            _rm = r.get("output_rc_mode", "")
            if _lv == 1 and _rm:
                _desc += f" (rc={_rm})"
            elif _lv != 1 and _rm:
                _desc += f" (回退, 期望rc={_rm})"
            print(f"  {_label:<10} {_lv:>5}  {_desc}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="IFRNet 版本对比基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用版本: {', '.join(VERSION_LABELS.values())}

示例:
    python benchmark_ifrnet_versions.py -i test.mp4 -o benchmark_output/
    python benchmark_ifrnet_versions.py -i test.mp4 -o benchmark_output/ --scale 4
    python benchmark_ifrnet_versions.py -i test.mp4 -o benchmark_output/ --versions v6.3.5 v6.4.3
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="输入视频路径")
    parser.add_argument("-o", "--output-dir", default="benchmark_output",
                        help="中间输出目录（默认: benchmark_output/）")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="插帧倍数（默认: 2）")
    parser.add_argument("--model", default="IFRNet_S_Vimeo90K",
                        choices=["IFRNet_Vimeo90K", "IFRNet_S_Vimeo90K", "IFRNet_L_Vimeo90K"],
                        help="模型名称（默认: IFRNet_S_Vimeo90K）")
    parser.add_argument("--versions", nargs="+",
                        default=["v6.3.5", "v6.4.1", "v6.4.2", "v6.4.3", "v6.4.3.1",
                                 "v6.4.3.2", "v6.4.4", "v6.4.4.1", "v6.4.4.2",
                                 "v6.4.5", "v6.4.5.1", "v6.4.5.2"],
                        help="要测试的版本列表（默认: 全部）")
    parser.add_argument("--cleanup", action="store_true",
                        help="测试完成后删除中间输出文件")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="保留所有版本的输出视频")
    parser.add_argument("--json-report", metavar="PATH",
                        help="保存 JSON 格式详细结果到指定路径")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过 GPU 预热（默认会先用短视频预热）")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="TRT 推理 batch_size（默认 24）")
    parser.add_argument("--warmup-duration", type=int, default=10,
                        help="预热视频时长（秒，默认 10）")
    parser.add_argument("--codec", type=str, default=None,
                        choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                        help="编码器（默认: None=自动选择最佳编码器）。与 --codecs 互斥")
    parser.add_argument("--codecs", nargs="+",
                        help="多编码器轮次测试，如 --codecs libx264 h264_nvenc。与 --codec 互斥")
    parser.add_argument("--crf", type=int, default=23,
                        help="编码质量 / QP 值（默认: 23）。libx264/x265: CRF; NVENC CONSTQP: QP; NVENC VBR_HQ: QP→targetQuality 映射")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="编码速度预设（默认: medium）。x264: x264 预设体系; NVENC: p1-p7 体系（自动映射）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        return 1

    # --codec 与 --codecs 互斥
    if args.codecs and args.codec is not None:
        print("错误: --codec 与 --codecs 互斥，请只指定其中一个")
        return 1

    codec_list = args.codecs if args.codecs else [args.codec]

    try:
        model_path = find_model_path(args.model)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    print(f"模型: {model_path}")
    print(f"输入: {args.input}")
    print(f"插帧: {args.scale}x")
    print(f"batch-size: {args.batch_size}")
    print(f"crf: {args.crf}  |  preset: {args.preset}")
    print(f"版本: {', '.join(args.versions)}")
    print()

    version_map = {v: k for k, v in VERSION_LABELS.items()}
    selected = []
    for v in args.versions:
        if v in version_map:
            selected.append(version_map[v])
        else:
            print(f"警告: 未知版本 '{v}'，已跳过")

    if not selected:
        print("错误: 没有有效的版本")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem

    # ── GPU 预热 ──────────────────────────────────────────────────────────────
    if not args.no_warmup:
        warmup_video = output_dir / f"_warmup_{input_stem}.mp4"
        warmup_output = output_dir / f"_warmup_{input_stem}_output.mp4"
        warmup_version = selected[0]
        warmup_label = VERSION_LABELS[warmup_version]

        print("=" * 60)
        print("GPU 预热阶段")
        print("=" * 60)
        try:
            _p = subprocess.run([
                "ffmpeg", "-y",
                "-i", os.path.abspath(args.input),
                "-t", str(args.warmup_duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(warmup_video),
            ], capture_output=True, text=True, timeout=60)
            if _p.returncode == 0 and warmup_video.exists():
                print(f"  预热视频: {warmup_video} ({args.warmup_duration}s)")
            else:
                # ffmpeg -c copy 失败时回退到重新编码
                _p2 = subprocess.run([
                    "ffmpeg", "-y",
                    "-i", os.path.abspath(args.input),
                    "-t", str(args.warmup_duration),
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-an",
                    str(warmup_video),
                ], capture_output=True, text=True, timeout=120)
                if _p2.returncode != 0 or not warmup_video.exists():
                    print(f"  预热视频生成失败，跳过预热")
                    warmup_video = None

            if warmup_video and warmup_video.exists():
                print(f"  预热版本: {warmup_label}")
                print(f"  预热中...", end=" ", flush=True)
                t_warmup = time.perf_counter()
                _result = run_version(
                    version_module=warmup_version,
                    input_path=str(warmup_video),
                    output_path=str(warmup_output),
                    model_path=model_path,
                    scale=args.scale,
                    batch_size=args.batch_size,
                    trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
                    codec=codec_list[0],
                    crf=args.crf,
                    preset=args.preset,
                )
                t_warmup_elapsed = time.perf_counter() - t_warmup
                if _result.get("success"):
                    print(f"完成 ({t_warmup_elapsed:.1f}s)")
                    print(f"  TRT Engine 已构建/加载，GPU 已进入稳定状态")
                else:
                    print(f"失败，继续基准测试")
        except Exception as e:
            print(f"  预热异常: {e}，跳过")
        finally:
            # 清理预热文件
            for _f in (warmup_video, warmup_output):
                if _f and _f.exists():
                    try:
                        _f.unlink()
                    except OSError:
                        pass

        print("=" * 60)
        print()
        # 预热后短暂冷却，让 GPU 回到基准温度/频率
        time.sleep(3)
    else:
        print("⚠️  已跳过 GPU 预热（--no-warmup）")
        print()

    # ── 正式基准测试 ──────────────────────────────────────────────────────────
    results = []

    for idx, version_module in enumerate(selected):
        label = VERSION_LABELS[version_module]
        output_path = output_dir / f"{input_stem}_{label}_x{args.scale:.0f}.mp4"

        print(f"[{idx+1}/{len(selected)}] 测试 {label} ...\n", end=" ", flush=True)

        try:
            result = run_version(
                version_module=version_module,
                input_path=os.path.abspath(args.input),
                output_path=str(output_path),
                model_path=model_path,
                scale=args.scale,
                batch_size=args.batch_size,
                trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
                codec=codec_list[0],
                crf=args.crf,
                preset=args.preset,
            )
        except subprocess.TimeoutExpired:
            result = {
                "version": version_module,
                "label": label,
                "success": False,
                "error": "timeout",
            }
        except Exception as e:
            result = {
                "version": version_module,
                "label": label,
                "success": False,
                "error": str(e),
            }

        results.append(result)

        if result.get("success"):
            fps = result.get("fps", "?")
            elapsed = result.get("elapsed_sec", "?")
            print(f"OK  ({elapsed}s, {fps} FPS)")
        else:
            err = result.get("error", "unknown")
            print(f"FAILED  ({err})")

        # 版本间有 GPU 内存残留风险，短暂暂停
        if idx < len(selected) - 1:
            time.sleep(2)

    # 汇总
    print_results(results, os.path.abspath(args.input), args.scale,
                  args.batch_size, args.model, args.crf, args.preset)

    # JSON 详细报告
    if args.json_report:
        report = {
            "input": os.path.abspath(args.input),
            "scale": args.scale,
            "model": args.model,
            "model_path": model_path,
            "batch_size": args.batch_size,
            "crf": args.crf,
            "preset": args.preset,
            "results": results,
        }
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON 报告已保存: {args.json_report}")

    # 清理
    if args.cleanup and not args.keep_outputs:
        for r in results:
            out = r.get("output_path", "")
            if out and os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass

    # 全部成功？
    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} 个版本测试失败")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
