#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement v6.4 - 架构优化版 (最终修复版)
基于v6.2代码重构，实现深度模块化分解。
保持原有所有功能及CLI参数，保留原有注释及信息提示不变。

新增实时预览功能：
    --preview            启用实时预览窗口（显示最终输出结果，按 q 键提前退出）
    --preview-interval   预览帧间隔（默认 30 帧刷新一次）
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# 必须在导入任何依赖 realesrgan 的模块之前执行
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# 已经把 Real-ESRGAN 项目拷贝进本项目
# _ext_dir = os.path.dirname(_script_dir)
# _realesrgan_path = os.path.join(_ext_dir, 'Real-ESRGAN')
# if os.path.isdir(_realesrgan_path) and _realesrgan_path not in sys.path:
#     sys.path.insert(0, _realesrgan_path)

from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from config import MODEL_CONFIG, models_RealESRGAN
from realesrgan_utils import get_video_meta_info, _build_upsampler
from ffmpeg_io import FFmpegReader, FFmpegWriter
from tensorrt_accel import TensorRTAccelerator
from gfpgan_subprocess import GFPGANSubprocess
from pipeline import DeepPipelineOptimizer

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

from tqdm import tqdm


class PreviewWriter:
    """
    包装 FFmpegWriter，在写入帧时提供实时预览功能。
    支持按 'q' 键提前退出（通过抛出 KeyboardInterrupt 实现）。
    """
    def __init__(self, writer, preview_enabled=False, preview_interval=30):
        self.writer = writer
        self.preview_enabled = preview_enabled
        self.preview_interval = preview_interval
        self.frame_counter = 0
        self.exit_requested = False
        
        if preview_enabled:
            # 增强的 GUI 可用性检测（子进程隔离，防止 Qt abort 杀死主进程）
            self.preview_enabled = self._check_gui_available_robust()
            if not self.preview_enabled:
                print('[Preview] GUI 环境不可用，预览功能已自动禁用。')

    def _check_gui_available_robust(self) -> bool:
        """
        在独立子进程中测试 GUI 可用性。
        
        Qt xcb 插件连接 X server 失败时会直接调用 C 级别的 abort()，
        Python 的 try/except 无法拦截。通过在子进程中执行 GUI 测试，
        即使 Qt abort 也只终止测试子进程，主进程不受影响。
        """
        try:
            import subprocess as _sp

            # 快速预检：Linux 下无 DISPLAY 直接跳过
            if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
                print('[Preview] 未检测到 DISPLAY 环境变量，GUI 不可用。')
                return False

            test_code = (
                "import sys, os\n"
                "try:\n"
                "    import cv2, numpy as np\n"
                "    cv2.namedWindow('__test__', cv2.WINDOW_NORMAL)\n"
                "    cv2.imshow('__test__', np.zeros((10,10,3), dtype=np.uint8))\n"
                "    cv2.waitKey(1)\n"
                "    cv2.destroyWindow('__test__')\n"
                "    cv2.waitKey(1)\n"
                "    print('GUI_OK')\n"
                "except Exception as e:\n"
                "    print('GUI_FAIL:' + str(e))\n"
                "    sys.exit(1)\n"
            )

            result = _sp.run(
                [sys.executable, '-c', test_code],
                capture_output=True, text=True, timeout=15,
            )

            if result.returncode == 0 and 'GUI_OK' in result.stdout:
                return True

            # 子进程崩溃或返回失败 —— 构造友好提示
            hint = ''
            if result.returncode < 0:
                # 被信号杀死（如 SIGABRT = 6）
                import signal
                try:
                    sig_name = signal.Signals(-result.returncode).name
                except (ValueError, AttributeError):
                    sig_name = str(-result.returncode)
                hint = f'子进程被信号 {sig_name} 终止（Qt/X11 abort）'
            elif 'GUI_FAIL' in result.stdout:
                hint = result.stdout.strip()
            else:
                hint = (result.stderr.strip() or '未知错误')[-300:]

            print(f'[Preview] GUI 检测失败: {hint}')
            return False

        except Exception as e:
            print(f'[Preview] GUI 检测异常: {e}')
            return False

    def _show_preview(self, frame: np.ndarray) -> bool:
        """显示一帧，返回 True 表示用户按 'q' 请求退出"""
        if not self.preview_enabled:
            return False
        try:
            import cv2
            cv2.imshow('RealESRGAN Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True
            return False
        except Exception as e:
            # 捕获 imshow 可能触发的任何异常（包括 Qt 插件错误）
            print(f'[Preview] 显示帧时出错，预览功能已自动禁用: {e}')
            self.preview_enabled = False
            # 尝试清理可能残留的窗口
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            return False

    def write_frame(self, frame: np.ndarray):
        # 先写入原 writer
        self.writer.write_frame(frame)

        # 如果启用了预览且未退出
        if self.preview_enabled and not self.exit_requested:
            self.frame_counter += 1
            if self.frame_counter % self.preview_interval == 0:
                if self._show_preview(frame):
                    self.exit_requested = True
                    print('\n[Preview] 用户按下 q 键，正在安全退出...')
                    raise KeyboardInterrupt  # 抛出中断，由外层 try/except 捕获

    def close(self):
        self.writer.close()
        # 关闭预览窗口
        if self.preview_enabled:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    # 代理其他可能用到的方法/属性
    def __getattr__(self, name):
        return getattr(self.writer, name)


def main_optimized(args):
    """优化版主函数 —— 包含完整的参数标准化与冲突仲裁"""

    # =========================================================================
    # 阶段 -1: 参数标准化、强制覆盖与互斥仲裁
    #          确保任何调用方（CLI 或外部模块）都得到一致的处理结果
    # =========================================================================

    # 0. 硬件加速开关处理
    if getattr(args, 'no_hwaccel', False):
        args.use_hwaccel = False

    # 1. 从 argparse 标志派生出布尔状态（便于仲裁）
    #    注意：args 中可能已有 use_compile / use_cuda_graph 属性，若无则从 no_* 派生
    if not hasattr(args, 'use_compile'):
        args.use_compile = not getattr(args, 'no_compile', False)
    if not hasattr(args, 'use_cuda_graph'):
        args.use_cuda_graph = not getattr(args, 'no_cuda_graph', False)

    # 2. 应用强制覆盖标志（高优先级参数）
    override_msgs = []

    # --no-tensorrt 强制禁用 TensorRT
    if getattr(args, 'no_tensorrt', False):
        if getattr(args, 'use_tensorrt', False):
            override_msgs.append('--no-tensorrt    覆盖了  --use-tensorrt  → TensorRT 已禁用')
        args.use_tensorrt = False

    # --use-compile 强制启用 compile
    if getattr(args, 'use_compile_force', False):
        if getattr(args, 'no_compile', False):
            override_msgs.append('--use-compile    覆盖了  --no-compile   → torch.compile 已启用')
        args.use_compile = True

    # --use-cuda-graph 强制启用 CUDA Graph
    if getattr(args, 'use_cuda_graph_force', False):
        if getattr(args, 'no_cuda_graph', False):
            override_msgs.append('--use-cuda-graph 覆盖了  --no-cuda-graph → CUDA Graph 已启用')
        args.use_cuda_graph = True

    if override_msgs:
        print('[参数覆盖] 以下设置已被高优先级参数调整：')
        for msg in override_msgs:
            print(f'          · {msg}')
        print()

    # 3. 冲突仲裁：TRT > compile > CUDA Graph
    #    优先级：TensorRT 最高，其次是 torch.compile，最后是 CUDA Graph
    if args.use_compile and args.use_tensorrt:
        print('[Warning] --use-tensorrt 与 torch.compile 互斥，TRT 优先，compile 已自动禁用。')
        args.use_compile = False

    if args.use_cuda_graph:
        if args.use_tensorrt:
            print('[Warning] CUDA Graph 与 TRT 互斥，TRT 优先，CUDA Graph 已自动禁用。')
            args.use_cuda_graph = False
        elif args.use_compile:
            # torch.compile 内部已包含 cudagraphs 优化，显式 CUDA Graph 应禁用
            print('[Info] torch.compile 已启用（内含 cudagraphs），CUDA Graph 已自动禁用。')
            args.use_cuda_graph = False

    # 4. 同步 no_compile / no_cuda_graph 标志，保证后续代码读取正确
    args.no_compile = not args.use_compile
    args.no_cuda_graph = not args.use_cuda_graph

    # 5. 处理 --no-adaptive-batch
    if getattr(args, 'no_adaptive_batch', False):
        args.adaptive_batch = False

    # 6. GFPGAN-TRT 启用时自动禁用 FP16（防止溢出噪斑）
    if getattr(args, 'gfpgan_trt', False) and torch.cuda.is_available():
        args.no_fp16 = True
        print("\n[GFPGAN-TRT] 为防止 FP16 溢出噪斑，已自动禁用半精度并启用 FP32 推理。")

    # 8. 自动生成报告路径（如果 --report 未指定具体路径）
    if getattr(args, 'report', None) == '__AUTO__':
        # 基于输出视频路径生成默认报告文件名
        out_base = os.path.splitext(args.output)[0]
        args.report = f"{out_base}_report.json"
        print(f"[自动报告] 使用默认报告路径: {args.report}")

    # 8. 打印版本及特性横幅（仅在 CLI 调用时打印，但放在这里也无妨）
    print("Real-ESRGAN Video Enhancement v6.4 - 架构优化版")
    print("主要优化特性:")
    print("1. 深度流水线架构（4级并行处理）")
    print("2. GPU内存池优化（避免频繁分配释放）")
    print("3. 异步计算模式（多CUDA流并行）")
    print("4. 多级缓冲队列（深度缓冲减少等待）")
    print("5. 优化线程池配置（提高并发效率）")
    print("6. 人脸检测置信度过滤（减少无效 GFPGAN 推理）")
    print("7. 自适应批处理（根据人脸密度动态调整）")
    print("8. SR H2D 预取重叠（利用空闲 GPU 内存总线）")
    if getattr(args, 'preview', False):
        print(f"9. 实时预览已请求（间隔 {getattr(args, 'preview_interval', 30)} 帧，按 q 退出）")
    print()

    # =========================================================================
    # 原有 main_optimized 核心逻辑开始
    # =========================================================================

    print("[优化架构] 修复版: 改进 GFPGAN TRT 就绪判断")
    if args.face_enhance:
        print(f"[优化架构] 人脸检测置信度阈值: {getattr(args, 'face_det_threshold', 0.5)}")
        if getattr(args, 'adaptive_batch', True):
            print(f"[优化架构] 自适应批处理: 已启用")
    print("[优化架构] 阶段 0: 准备环境（不初始化 CUDA）...")

    cuda_available = torch.backends.cuda.is_built() and torch.cuda.device_count() > 0
    if not cuda_available:
        print("[优化架构] CUDA 不可用，使用 CPU 模式")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"[优化架构] CUDA 编译支持: 是")
        print(f"[优化架构] 延迟 CUDA Runtime 初始化直到 GFPGAN 子进程就绪")

    _early_gfpgan_subprocess = None
    gfpgan_ready = False
    use_gfpgan_subprocess = False
    gfpgan_mode = "disabled"

    if args.face_enhance and getattr(args, 'gfpgan_trt', False) and GFPGANer is not None:
        if not cuda_available:
            print("[优化架构] 警告: CUDA 不可用，跳过 GFPGAN TRT")
        else:
            print("[优化架构] 阶段 1: 预启动 GFPGAN 子进程（GPU 干净状态）...")
            _model_paths_early = {
                '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
            }
            _model_url_early = _model_paths_early.get(args.gfpgan_model, _model_paths_early['1.4'])
            _model_dir_early = os.path.join(models_RealESRGAN, 'GFPGAN')
            os.makedirs(_model_dir_early, exist_ok=True)
            _model_filename_early = os.path.basename(_model_url_early)
            _model_path_early = os.path.join(_model_dir_early, _model_filename_early)
            if not os.path.exists(_model_path_early):
                _model_path_early = load_file_from_url(_model_url_early, _model_dir_early, True)

            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

            print('[优化架构] 启动 GFPGAN 子进程...')
            _early_gfpgan_subprocess = GFPGANSubprocess(
                model_path=_model_path_early, device=device,
                gfpgan_weight=args.gfpgan_weight, gfpgan_batch_size=args.gfpgan_batch_size,
                use_fp16=not args.no_fp16, use_trt=True,
                trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                gfpgan_model=args.gfpgan_model,
            )

            print('[优化架构] 等待 GFPGAN 子进程完成初始化...')
            max_wait = 5400
            deadline = time.time() + max_wait
            _poll_interval = 3
            _last_report = time.time()
            _report_every = 60

            while time.time() < deadline:
                if not _early_gfpgan_subprocess.process.is_alive():
                    exitcode = _early_gfpgan_subprocess.process.exitcode
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                    gfpgan_ready = False
                    gfpgan_mode = "failed_subprocess"
                    break
                if _early_gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    time.sleep(0.5)
                    if not _early_gfpgan_subprocess.process.is_alive():
                        _early_gfpgan_subprocess.close()
                        _early_gfpgan_subprocess = None
                        gfpgan_ready = False
                        gfpgan_mode = "failed_trt_warmup"
                        break
                    print('[优化架构] GFPGAN 子进程 signaled ready 且进程稳定')
                    gfpgan_ready = True
                    use_gfpgan_subprocess = True
                    gfpgan_mode = "subprocess_trt"
                    print(f'[优化架构] GFPGAN 子进程验证通过，模式: {gfpgan_mode}')
                    break
                now = time.time()
                if now - _last_report >= _report_every:
                    elapsed = now - (deadline - max_wait)
                    print(f'[优化架构] 等待 GFPGAN 初始化... {elapsed:.0f}s', flush=True)
                    _last_report = now

            if not gfpgan_ready:
                print('[优化架构] GFPGAN 子进程初始化失败，将使用主进程 PyTorch 路径')
                if _early_gfpgan_subprocess:
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                gfpgan_mode = "main_pytorch_fallback"
            else:
                print('[优化架构] GFPGAN 子进程准备就绪')

    print("[优化架构] 阶段 2: 初始化主进程 CUDA 并加载 RealESRGAN...")

    if cuda_available:
        try:
            torch.cuda.init()
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[优化架构] 主进程 CUDA 初始化完成: {device_name} ({device_mem:.1f}GB)")
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"[优化架构] 当前显存: 已分配 {mem_allocated:.2f}GB, 预留 {mem_reserved:.2f}GB")
        except Exception as e:
            print(f"[优化架构] CUDA 初始化失败: {e}")
            cuda_available = False
            device = torch.device('cpu')

    print(f"[优化架构] 加载模型: {args.model_name}")
    use_half = not args.no_fp16
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and hasattr(args, 'denoise_strength'):
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    try:
        upsampler = _build_upsampler(
            args.model_name, dni_weight, args.tile, args.tile_pad,
            args.pre_pad, use_half, device
        )
        print("[优化架构] RealESRGAN模型加载成功")
        _, _netscale, _ = MODEL_CONFIG.get(args.model_name, (None, 4, None))
        args.netscale = _netscale
        if getattr(args, 'use_tensorrt', False) and not getattr(args, 'no_compile', False):
            args.no_compile = True
    except Exception as e:
        print(f"[优化架构] RealESRGAN模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    face_enhancer = None
    if args.face_enhance and GFPGANer is not None:
        if use_gfpgan_subprocess and gfpgan_ready and _early_gfpgan_subprocess is not None:
            print("[优化架构] GFPGAN 由子进程处理，主进程创建 detect helper...")
            try:
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper

                class DummyGFPGANer:
                    def __init__(self, device, upscale):
                        self.device = device
                        self.upscale = upscale
                        self.face_helper = FaceRestoreHelper(
                            upscale_factor=upscale, face_size=512, crop_ratio=(1, 1),
                            det_model='retinaface_resnet50', save_ext='png',
                            use_parse=True, device=device,
                        )
                        self.gfpgan = None
                        self.model_path = None

                face_enhancer = DummyGFPGANer(device, args.outscale)
                print("[优化架构] Detect helper 创建成功（GFPGAN 推理由子进程处理）")
            except Exception as e:
                print(f"[优化架构] Detect helper 创建失败: {e}")
                face_enhancer = None
                use_gfpgan_subprocess = False
                gfpgan_ready = False
        else:
            print("[优化架构] 加载GFPGAN主进程模型...")
            try:
                model_paths = {
                    '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
                }
                model_url = model_paths.get(args.gfpgan_model, model_paths['1.4'])
                model_dir = os.path.join(models_RealESRGAN, 'GFPGAN')
                os.makedirs(model_dir, exist_ok=True)
                model_filename = os.path.basename(model_url)
                model_path = os.path.join(model_dir, model_filename)
                if not os.path.exists(model_path):
                    model_path = load_file_from_url(model_url, model_dir, True)
                face_enhancer = GFPGANer(
                    model_path=model_path, upscale=args.outscale, arch='clean',
                    channel_multiplier=2, bg_upsampler=None, device=device
                )
                if face_enhancer.gfpgan is None:
                    raise RuntimeError("GFPGAN 模型加载失败: gfpgan 网络为 None")
                print("[优化架构] GFPGAN主进程模型加载成功")
                gfpgan_mode = "main_pytorch"
                if getattr(args, 'gfpgan_trt', False):
                    args.gfpgan_trt = False
            except Exception as e:
                print(f"[优化架构] GFPGAN模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                face_enhancer = None
                gfpgan_mode = "disabled"

    print("[优化架构] 阶段 3: 创建视频读写器...")
    reader = FFmpegReader(
        args.input, ffmpeg_bin=getattr(args, 'ffmpeg_bin', 'ffmpeg'),
        prefetch_factor=getattr(args, 'prefetch_factor', 48),
        use_hwaccel=getattr(args, 'use_hwaccel', True),
    )

    out_h = int(reader.height * args.outscale)
    out_w = int(reader.width * args.outscale)
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    base_writer = FFmpegWriter(args, reader.audio, out_h, out_w, args.output, reader.fps)

    # 根据预览选项包装 writer
    preview_enabled = getattr(args, 'preview', False)
    preview_interval = getattr(args, 'preview_interval', 30)
    if preview_enabled:
        writer = PreviewWriter(base_writer, preview_enabled=True, preview_interval=preview_interval)
        if writer.preview_enabled:
            print(f"[优化架构] 实时预览已启用（间隔 {preview_interval} 帧，按 q 退出）")
        else:
            print("[优化架构] 实时预览已请求但 GUI 不可用，将仅写入文件")
        # 同步实际状态到 preview_enabled，供后续 report 使用
        preview_enabled = writer.preview_enabled
    else:
        writer = base_writer

    pbar = tqdm(total=reader.nb_frames, unit='frame', desc='[优化流水线]',
                dynamic_ncols=False, ncols=180,
                bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]\n')

    trt_accel = None
    if getattr(args, 'use_tensorrt', False) and cuda_available:
        # [FIX-TRT-CTX-OOM] 在双步模式（interpolate_then_upscale）下，
        # 前序 IFRNet 步骤可能在 PyTorch 缓存分配器中残留大量显存。
        # 主动清理，为 TRT execution context 腾出空间。
        torch.cuda.empty_cache()
        
        meta = get_video_meta_info(args.input)
        sh = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = getattr(args, 'trt_cache_dir', None) or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.trt_cache')
        print(f'[优化架构] 初始化 SR TensorRT Engine (shape={sh})...')
        try:
            trt_accel = TensorRTAccelerator(
                upsampler.model, device, trt_dir, sh, use_fp16=not args.no_fp16)
            if trt_accel.available:
                print('[优化架构] SR TensorRT Engine 加载成功')
            else:
                trt_accel = None
        except Exception as _te:
            print(f'[优化架构] SR TensorRT 初始化异常: {_te}')
            trt_accel = None

    if _early_gfpgan_subprocess is not None and use_gfpgan_subprocess and gfpgan_ready:
        args._early_gfpgan_subprocess = _early_gfpgan_subprocess
        print(f'[优化架构] GFPGAN 子进程已绑定到流水线（模式: {gfpgan_mode}）')
    else:
        args._early_gfpgan_subprocess = None
        if args.face_enhance:
            print(f'[优化架构] GFPGAN 使用主进程模式: {gfpgan_mode}')

    print("[优化架构] 阶段 4: 启动优化流水线...")
    pipeline = DeepPipelineOptimizer(upsampler, face_enhancer, args, device, trt_accel=trt_accel,
                                     input_h=reader.height, input_w=reader.width)

    start_time = time.time()
    try:
        pipeline.optimize_pipeline(reader, writer, pbar, reader.nb_frames)
    except KeyboardInterrupt:
        print("\n用户中断（可能是预览窗口按 q）")
    except Exception as e:
        print(f"流水线错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[优化架构] ===========================================")
        print("[优化架构] 视频推理完成，正在清理资源...")
        print("[优化架构] ===========================================\n")

        # print("[优化架构] 步骤1/4: 关闭流水线线程...")
        pipeline.close()
        # print("[优化架构] 流水线线程已关闭")

        # print("[优化架构] 步骤2/4: 关闭FFmpeg写入器...")
        writer.close()
        # print("[优化架构] FFmpeg写入器已关闭")

        # print("[优化架构] 步骤3/4: 关闭视频读取器...")
        reader.close()
        # print("[优化架构] 视频读取器已关闭")

        # print("[优化架构] 步骤4/4: 关闭进度条...")
        pbar.close()

        end_time = time.time()
        total_time = end_time - start_time

        # ── 生成性能报告（如果指定了 --report）─────────────────────
        if getattr(args, 'report', None) and pipeline.timing:
            import json
            report_path = args.report
            os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
            elapsed = total_time
            report = {
                'input': args.input,
                'output': args.output,
                'model': args.model_name,
                'outscale': args.outscale,
                'batch_size': args.batch_size,
                'fp16': not args.no_fp16,
                'trt': trt_accel is not None and trt_accel.available,
                'nvdec': getattr(args, 'use_hwaccel', True),
                'nvenc': getattr(args, 'video_codec', 'libx264') in ('h264_nvenc', 'hevc_nvenc'),
                'face_enhance': args.face_enhance,
                'frame_count': reader.nb_frames,
                'elapsed_s': round(elapsed, 2),
                'avg_fps': round(reader.nb_frames / elapsed, 2) if elapsed > 0 else 0,
                'infer_latency_ms': {
                    'mean': round(float(np.mean(pipeline.timing)) * 1000, 2),
                    'p95': round(float(np.percentile(pipeline.timing, 95)) * 1000, 2),
                    'max': round(float(np.max(pipeline.timing)) * 1000, 2),
                },
                'gfpgan_mode': gfpgan_mode,
                'face_filtered': pipeline._face_filtered_total,
                'adaptive_batch': pipeline._enable_adaptive_batch,
                'preview': preview_enabled,
            }
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f'[Info] 性能报告已保存: {report_path}')
        # ──────────────────────────────────────────────────────────

        if pipeline.timing:
            avg_time = np.mean(pipeline.timing) * 1000
            actual_fps = reader.nb_frames / total_time if total_time > 0 else 0
            print(f"\n[性能统计] 总时间: {total_time:.1f}秒 | 平均: {avg_time:.1f}ms | FPS: {actual_fps:.2f}")
            print(f"[性能统计] GFPGAN 模式: {gfpgan_mode}")
            if pipeline._face_filtered_total > 0:
                print(f"[性能统计] 人脸检测: 保留 {pipeline._face_count_total} 个, "
                      f"过滤 {pipeline._face_filtered_total} 个 "
                      f"(阈值={pipeline.face_det_threshold})")
            if pipeline._enable_adaptive_batch:
                print(f"[性能统计] 最终人脸密度EMA: {pipeline._face_density_ema:.2f} 人脸/帧, "
                      f"最终自适应arbs: {pipeline._adaptive_read_batch_size}")


def main():
    """主函数 - 参数解析（仅负责解析命令行，所有预处理已下沉至 main_optimized）"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 架构优化版 v6.4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基础参数
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('-n', '--model-name', type=str, default='realesr-animevideov3', help='模型名称')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='输出缩放比例')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5, help='去噪强度')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件后缀')

    # 优化参数
    parser.add_argument('--batch-size', type=int, default=6, help='批处理大小（优化版本推荐6-8）')
    parser.add_argument('--prefetch-factor', type=int, default=48, help='读帧预取队列深度')

    # 人脸增强参数
    parser.add_argument('--face-enhance', action='store_true', help='启用人脸增强')
    parser.add_argument('--gfpgan-model', type=str, default='1.4', choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本')
    parser.add_argument('--gfpgan-weight', type=float, default=0.5, help='GFPGAN 增强融合权重')
    parser.add_argument('--gfpgan-batch-size', type=int, default=8, help='GFPGAN 单次最多处理人脸数')
    parser.add_argument('--face-det-threshold', type=float, default=0.5,
                        help='人脸检测置信度阈值 [0.0-1.0]，越高过滤越多低质量检测。'
                             '0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸')
    parser.add_argument('--adaptive-batch', action='store_true', default=True,
                        help='启用基于人脸密度的自适应批处理大小（默认开启）')
    parser.add_argument('--no-adaptive-batch', action='store_true', default=False,
                        help='禁用自适应批处理大小（使用固定 --batch-size）')

    # 加速参数
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile', action='store_true', help='禁用 torch.compile')
    parser.add_argument('--use-tensorrt', action='store_true', help='启用 SR TensorRT 加速')
    parser.add_argument('--gfpgan-trt', action='store_true', help='GFPGAN TensorRT 加速')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')

    # 硬件加速参数
    parser.add_argument('--use-hwaccel', action='store_true', default=True, help='启用 NVDEC')
    parser.add_argument('--no-hwaccel', action='store_true', help='禁用 NVDEC')

    # 其他参数
    parser.add_argument('-t', '--tile', type=int, default=0, help='分块大小')
    parser.add_argument('--tile-pad', type=int, default=10, help='分块填充')
    parser.add_argument('--pre-pad', type=int, default=0, help='预填充')
    parser.add_argument('--fps', type=float, default=None, help='输出帧率')

    # 编码参数
    parser.add_argument('--video-codec', type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc'],
                        help='偏好编码器')
    parser.add_argument('--crf', type=int, default=23, help='编码质量')
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                                 'medium', 'slow', 'slower', 'veryslow'],
                        help='libx264/libx265 preset')
    parser.add_argument('--ffmpeg-bin', type=str, default='ffmpeg', help='FFmpeg 二进制路径')

    # 性能报告（支持可选参数自动生成路径）
    parser.add_argument('--report', nargs='?', const='__AUTO__', default=None,
                        help='输出 JSON 性能报告路径。若仅使用 --report 不指定路径，则自动根据输出文件名生成（如 output_report.json）')

    # 高优先级覆盖参数
    parser.add_argument('--no-tensorrt', dest='no_tensorrt', action='store_true', default=False,
                        help='强制禁用 TensorRT（覆盖 --use-tensorrt 或外部 config）')
    parser.add_argument('--use-compile', dest='use_compile_force', action='store_true', default=False,
                        help='强制启用 torch.compile（覆盖 --no-compile 或 config）')
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force', action='store_true', default=False,
                        help='强制启用 CUDA Graph（覆盖 --no-cuda-graph 或 config）')

    # 预览参数
    parser.add_argument('--preview', action='store_true',
                        help='启用实时预览窗口（显示最终输出结果，按 q 键提前退出）')
    parser.add_argument('--preview-interval', type=int, default=30,
                        help='预览帧间隔（每多少帧刷新一次窗口）')

    args = parser.parse_args()

    # 直接调用优化版主函数（所有预处理已在其中完成）
    main_optimized(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()