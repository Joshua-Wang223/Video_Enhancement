#!/usr/bin/env python3
"""
IFRNet 版本对比基准测试

对同一视频分别调用 v6.3.5 / v6.4.1 / v6.4.2 / v6.4.3 进行插帧测试，
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


PROJECT_ROOT = Path(__file__).resolve().parent
IFRNET_DIR = PROJECT_ROOT / "external" / "IFRNet"
MODELS_DIR = PROJECT_ROOT / "models_IFRNet" / "checkpoints"

VERSIONS = [
    "process_video_v6_3_5_single",
    "process_video_v6_4_1_single",
    "process_video_v6_4_2_single",
    "process_video_v6_4_3_single",
]

VERSION_LABELS = {
    "process_video_v6_3_5_single": "v6.3.5",
    "process_video_v6_4_1_single": "v6.4.1",
    "process_video_v6_4_2_single": "v6.4.2",
    "process_video_v6_4_3_single": "v6.4.3",
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
                model_path: str, scale: float, trt_cache_dir: str) -> dict:
    """Run a single IFRNet version in a subprocess and return timing results."""
    inner_script = f'''
import importlib
import json
import os
import sys
import time

_ifrnet_dir = {str(IFRNET_DIR)!r}
if _ifrnet_dir not in sys.path:
    sys.path.insert(0, _ifrnet_dir)

mod = importlib.import_module({version_module!r})

# 自动选择最佳编码器
codec = "libx264"
try:
    preferred = mod.HardwareCapability.auto_select_codec("libx264")
    if preferred:
        codec = preferred
except Exception:
    pass

print(f"[BENCHMARK] {version_module} codec={{codec}}", flush=True)

proc = mod.IFRNetVideoProcessor(
    model_path={model_path!r},
    device="cuda",
    batch_size=4,
    max_batch_size=16,
    use_fp16=True,
    use_compile=False,
    use_cuda_graph=False,
    use_tensorrt=True,
    trt_cache_dir={trt_cache_dir!r},
    use_hwaccel=True,
    codec=codec,
    crf=23,
    x264_preset="medium",
    keep_audio=False,
    ffmpeg_bin="ffmpeg",
    quiet=True,
)

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

result = {{
    "version": {version_module!r},
    "label": {VERSION_LABELS.get(version_module, version_module)!r},
    "success": ok,
    "elapsed_sec": round(elapsed, 3),
    "input_path": {input_path!r},
    "output_path": {output_path!r},
    "scale": {scale},
    "codec": codec,
    "model": {os.path.basename(model_path)!r},
}}

# 收集输出文件信息
if ok and os.path.exists({output_path!r}):
    try:
        st = os.stat({output_path!r})
        result["output_size_mb"] = round(st.st_size / (1024 * 1024), 2)
        result["output_size_bytes"] = st.st_size
    except OSError:
        pass

    # 用 ffprobe 获取输出帧数
    try:
        import subprocess as _sp
        _p = _sp.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", {output_path!r}
        ], capture_output=True, text=True, timeout=30)
        if _p.returncode == 0 and _p.stdout.strip():
            result["output_frames"] = int(_p.stdout.strip())
    except Exception:
        pass

# 收集输入帧数
try:
    import subprocess as _sp
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

# 收集 GPU 利用率（从 nvidia-smi 快照）
try:
    import subprocess as _sp
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

print("__BENCHMARK_RESULT__", json.dumps(result), flush=True)
'''
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.run(
        [sys.executable, "-c", inner_script],
        capture_output=True, text=True, timeout=3600,
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


def print_results(results: list[dict], input_path: str, scale: float, model: str):
    """Print comparison table."""
    print()
    print("=" * 90)
    print("IFRNet 版本基准测试结果")
    print("=" * 90)
    print(f"  输入: {input_path}  |  插帧: {scale}x  |  模型: {model}")
    print()

    header = f"  {'版本':<10} {'耗时':>8}  {'输入帧':>8}  {'输出帧':>8}  {'FPS':>8}  {'文件大小':>10}  {'GPU':>6}  {'编码器':<14}"
    print(header)
    print("  " + "-" * 86)

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
        fps = r.get("fps", 0)
        size_mb = r.get("output_size_mb", 0)
        gpu = f"{r.get('gpu_util_pct', '?')}%"
        codec = r.get("codec", "?")

        elapsed_str = f"{elapsed:.1f}s" if elapsed else "?"
        in_str = str(in_frames) if in_frames else "?"
        out_str = str(out_frames) if out_frames else "?"
        fps_str = f"{fps:.1f}" if fps else "?"
        size_str = f"{size_mb:.1f} MB" if size_mb else "?"

        print(f"  {label:<10} {elapsed_str:>8}  {in_str:>8}  {out_str:>8}  {fps_str:>8}  {size_str:>10}  {gpu:>6}  {codec:<14}")

        if isinstance(fps, (int, float)) and fps > best_fps:
            best_fps = fps
            best_label = label

    print()
    if best_label:
        print(f"  最快版本: {best_label} ({best_fps:.1f} FPS)")

    # 版本间对比
    ok_results = [r for r in results if r.get("success")]
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

    print("=" * 90)


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
                        default=["v6.3.5", "v6.4.1", "v6.4.2", "v6.4.3"],
                        help="要测试的版本列表（默认: 全部）")
    parser.add_argument("--cleanup", action="store_true",
                        help="测试完成后删除中间输出文件")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="保留所有版本的输出视频")
    parser.add_argument("--json-report", metavar="PATH",
                        help="保存 JSON 格式详细结果到指定路径")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过 GPU 预热（默认会先用短视频预热）")
    parser.add_argument("--warmup-duration", type=int, default=30,
                        help="预热视频时长（秒，默认 30）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        return 1

    try:
        model_path = find_model_path(args.model)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    print(f"模型: {model_path}")
    print(f"输入: {args.input}")
    print(f"插帧: {args.scale}x")
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
                    trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
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

        print(f"[{idx+1}/{len(selected)}] 测试 {label} ...", end=" ", flush=True)

        try:
            result = run_version(
                version_module=version_module,
                input_path=os.path.abspath(args.input),
                output_path=str(output_path),
                model_path=model_path,
                scale=args.scale,
                trt_cache_dir=str(PROJECT_ROOT / ".trt_cache"),
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
    print_results(results, os.path.abspath(args.input), args.scale, args.model)

    # JSON 详细报告
    if args.json_report:
        report = {
            "input": os.path.abspath(args.input),
            "scale": args.scale,
            "model": args.model,
            "model_path": model_path,
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
