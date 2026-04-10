#!/usr/bin/env python3
import os
import sys
import numpy as np

# 强制将当前目录加入 sys.path 最前，必须在所有 import 之前执行
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# 必须在导入任何依赖 realesrgan 的模块之前执行
# 已经把 Real-ESRGAN 项目拷贝进本项目
# _ext_dir = os.path.dirname(_script_dir)
# _realesrgan_path = os.path.join(_ext_dir, 'Real-ESRGAN')
# if os.path.isdir(_realesrgan_path) and _realesrgan_path not in sys.path:
#     sys.path.insert(0, _realesrgan_path)

import torch
import time
import os.path as osp
import argparse
from tqdm import tqdm

# 🔑 2. 导入本地模块（避开标准库冲突）
import config
from cli import parse_args
from video_io.reader import FFmpegReader
from video_io.writer import FFmpegWriter
from models.sr_engine import _build_upsampler
from models.face_detect import _make_detect_helper
from subproc.gfpgan_worker import GFPGANSubprocess
from pipeline.optimizer import DeepPipelineOptimizer
from accelerators.trt import TensorRTAccelerator

# 自动定位同级目录下的 Real-ESRGAN 并注入 sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ext_dir    = os.path.dirname(_script_dir)
sys.path.insert(0, _ext_dir)
# _realesrgan_path = os.path.join(_ext_dir, 'Real-ESRGAN')
# if os.path.isdir(_realesrgan_path) and _realesrgan_path not in sys.path:
#     sys.path.insert(0, _realesrgan_path)

def main_optimized(args):
    print("[优化架构] 修复版: 改进 GFPGAN TRT 就绪判断")
    print(f"[优化架构] 人脸检测置信度阈值: {getattr(args, 'face_det_threshold', 0.5)}")
    if getattr(args, 'adaptive_batch', True): print(f"[优化架构] 自适应批处理: 已启用")
    print("[优化架构] 阶段 0: 准备环境（不初始化 CUDA）...")
    cuda_available = torch.backends.cuda.is_built() and torch.cuda.device_count() > 0
    device = torch.device('cuda' if cuda_available else 'cpu')
    if not cuda_available: print("[优化架构] CUDA 不可用，使用 CPU 模式")
    else: print(f"[优化架构] CUDA 编译支持: 是\n[优化架构] 延迟 CUDA Runtime 初始化直到 GFPGAN 子进程就绪")
    _early_gfpgan_subprocess, gfpgan_ready, use_gfpgan_subprocess, gfpgan_mode = None, False, False, "disabled"
    from gfpgan import GFPGANer
    if args.face_enhance and getattr(args, 'gfpgan_trt', False) and GFPGANer is not None:
        if not cuda_available: print("[优化架构] 警告: CUDA 不可用，跳过 GFPGAN TRT")
        else:
            print("[优化架构] 阶段 1: 预启动 GFPGAN 子进程（GPU 干净状态）...")
            _model_paths_early = {'1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'}
            _model_url_early = _model_paths_early.get(args.gfpgan_model, _model_paths_early['1.4'])
            _model_dir_early = osp.join(config.MODELS_DIR, 'GFPGAN'); os.makedirs(_model_dir_early, exist_ok=True)
            from basicsr.utils.download_util import load_file_from_url
            _model_path_early = load_file_from_url(_model_url_early, _model_dir_early, True)
            if torch.cuda.is_initialized(): torch.cuda.empty_cache()
            print('[优化架构] 启动 GFPGAN 子进程...')
            _early_gfpgan_subprocess = GFPGANSubprocess(model_path=_model_path_early, device=device, gfpgan_weight=args.gfpgan_weight, gfpgan_batch_size=args.gfpgan_batch_size, use_fp16=not args.no_fp16, use_trt=True, trt_cache_dir=getattr(args, 'trt_cache_dir', None), gfpgan_model=args.gfpgan_model)
            print('[优化架构] 等待 GFPGAN 子进程完成初始化...')
            max_wait, deadline, _poll_interval, _report_every, _last_report = 5400, time.time()+5400, 3, 60, time.time()
            while time.time() < deadline:
                if not _early_gfpgan_subprocess.process.is_alive(): exitcode = _early_gfpgan_subprocess.process.exitcode; _early_gfpgan_subprocess.close(); _early_gfpgan_subprocess = None; gfpgan_ready, gfpgan_mode = False, "failed_subprocess"; break
                if _early_gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    time.sleep(0.5)
                    if not _early_gfpgan_subprocess.process.is_alive(): _early_gfpgan_subprocess.close(); _early_gfpgan_subprocess = None; gfpgan_ready, gfpgan_mode = False, "failed_trt_warmup"; break
                    print('[优化架构] GFPGAN 子进程 signaled ready 且进程稳定'); gfpgan_ready, use_gfpgan_subprocess, gfpgan_mode = True, True, "subprocess_trt"; print(f'[优化架构] GFPGAN 子进程验证通过，模式: {gfpgan_mode}'); break
                now = time.time()
                if now - _last_report >= _report_every: print(f'[优化架构] 等待 GFPGAN 初始化... {now-(deadline-max_wait):.0f}s', flush=True); _last_report = now
            if not gfpgan_ready:
                print('[优化架构] GFPGAN 子进程初始化失败，将使用主进程 PyTorch 路径')
                if _early_gfpgan_subprocess:
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                    gfpgan_mode = "main_pytorch_fallback"
    print("[优化架构] 阶段 2: 初始化主进程 CUDA 并加载 RealESRGAN...")
    if cuda_available:
        try:
            torch.cuda.init(); print(f"[优化架构] 主进程 CUDA 初始化完成: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)"); torch.cuda.empty_cache()
            print(f"[优化架构] 当前显存: 已分配 {torch.cuda.memory_allocated()/1024**3:.2f}GB, 预留 {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        except Exception as e: print(f"[优化架构] CUDA 初始化失败: {e}"); cuda_available = False; device = torch.device('cpu')
    print(f"[优化架构] 加载模型: {args.model_name}")
    use_half = not args.no_fp16; dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and hasattr(args, 'denoise_strength'): dni_weight = [args.denoise_strength, 1-args.denoise_strength]
    try:
        upsampler = _build_upsampler(args.model_name, dni_weight, args.tile, args.tile_pad, args.pre_pad, use_half, device)
        print("[优化架构] RealESRGAN模型加载成功")
        _, _netscale, _ = config.MODEL_CONFIG.get(args.model_name, (None, 4, None)); args.netscale = _netscale
    except Exception as e: print(f"[优化架构] RealESRGAN模型加载失败: {e}"); import traceback; traceback.print_exc(); return
    face_enhancer = None
    if args.face_enhance and GFPGANer is not None:
        if use_gfpgan_subprocess and gfpgan_ready and _early_gfpgan_subprocess is not None:
            print("[优化架构] GFPGAN 由子进程处理，主进程创建 detect helper...")
            try:
                from realesrgan_video.models.face_detect import _make_detect_helper
                class DummyGFPGANer:
                    def __init__(self, device, upscale): 
                        self.device, self.upscale = device, upscale
                        self.gfpgan = None
                        self.model_path = None
                        # 创建有效的 face_helper 用于人脸贴回
                        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                        self.face_helper = FaceRestoreHelper(
                            upscale_factor=upscale,
                            face_size=512,
                            crop_ratio=(1, 1),
                            det_model='retinaface_resnet50',
                            save_ext='png',
                            use_parse=True,
                            device=device  # 与主进程设备一致
                        )
                face_enhancer = DummyGFPGANer(device, args.outscale)
                print("[优化架构] Detect helper 创建成功（GFPGAN 推理由子进程处理）")
            except Exception as e: print(f"[优化架构] Detect helper 创建失败: {e}"); face_enhancer = None; use_gfpgan_subprocess, gfpgan_ready = False, False
        else:
            print("[优化架构] 加载GFPGAN主进程模型...")
            try:
                model_paths = {'1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'}
                model_url = model_paths.get(args.gfpgan_model, model_paths['1.4'])
                model_dir = osp.join(config.MODELS_DIR, 'GFPGAN'); os.makedirs(model_dir, exist_ok=True)
                model_path = osp.join(model_dir, osp.basename(model_url))
                if not osp.exists(model_path): model_path = load_file_from_url(model_url, model_dir, True)
                face_enhancer = GFPGANer(model_path=model_path, upscale=args.outscale, arch='clean', channel_multiplier=2, bg_upsampler=None, device=device)
                if face_enhancer.gfpgan is None: raise RuntimeError("GFPGAN 模型加载失败: gfpgan 网络为 None")
                print("[优化架构] GFPGAN主进程模型加载成功"); gfpgan_mode = "main_pytorch"
                if getattr(args, 'gfpgan_trt', False): args.gfpgan_trt = False
            except Exception as e: print(f"[优化架构] GFPGAN模型加载失败: {e}"); import traceback; traceback.print_exc(); face_enhancer = None; gfpgan_mode = "disabled"
    print("[优化架构] 阶段 3: 创建视频读写器...")
    reader = FFmpegReader(args.input, ffmpeg_bin=getattr(args, 'ffmpeg_bin', 'ffmpeg'), prefetch_factor=getattr(args, 'prefetch_factor', 48), use_hwaccel=getattr(args, 'use_hwaccel', True))
    out_h, out_w = int(reader.height*args.outscale), int(reader.width*args.outscale)
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = FFmpegWriter(args, reader.audio, out_h, out_w, args.output, reader.fps)
    pbar = tqdm(total=reader.nb_frames, unit='frame', desc='[优化流水线]', dynamic_ncols=False, ncols=180, bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]\n')
    trt_accel = None
    if getattr(args, 'use_tensorrt', False) and cuda_available:
        meta = {'width': reader.width, 'height': reader.height}; sh = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = getattr(args, 'trt_cache_dir', None) or osp.join(config.BASE_DIR, '.trt_cache')
        print(f'[优化架构] 初始化 SR TensorRT Engine (shape={sh})...')
        try:
            trt_accel = TensorRTAccelerator(upsampler.model, device, trt_dir, sh, use_fp16=not args.no_fp16)
            if not trt_accel.available: trt_accel = None
        except Exception as _te: print(f'[优化架构] SR TensorRT 初始化异常: {_te}'); trt_accel = None
    if _early_gfpgan_subprocess is not None and use_gfpgan_subprocess and gfpgan_ready: args._early_gfpgan_subprocess = _early_gfpgan_subprocess; print(f'[优化架构] GFPGAN 子进程已绑定到流水线（模式: {gfpgan_mode}）')
    else: args._early_gfpgan_subprocess = None
    if args.face_enhance: print(f'[优化架构] GFPGAN 使用主进程模式: {gfpgan_mode}')
    print("[优化架构] 阶段 4: 启动优化流水线...")
    pipeline = DeepPipelineOptimizer(upsampler, face_enhancer, args, device, trt_accel=trt_accel, input_h=reader.height, input_w=reader.width)
    start_time = time.time()
    try: pipeline.optimize_pipeline(reader, writer, pbar, reader.nb_frames)
    except KeyboardInterrupt: print("\n用户中断")
    except Exception as e: print(f"流水线错误: {e}"); import traceback; traceback.print_exc()
    finally:
        print("\n[优化架构] ===========================================\n[优化架构] 视频推理完成，正在清理资源...\n[优化架构] ===========================================\n")
        print("[优化架构] 步骤1/4: 关闭流水线线程..."); pipeline.close(); print("[优化架构] 流水线线程已关闭")
        print("[优化架构] 步骤2/4: 关闭FFmpeg写入器..."); writer.close(); print("[优化架构] FFmpeg写入器已关闭")
        print("[优化架构] 步骤3/4: 关闭视频读取器..."); reader.close(); print("[优化架构] 视频读取器已关闭")
        print("[优化架构] 步骤4/4: 关闭进度条..."); pbar.close()
        end_time = time.time(); total_time = end_time - start_time
        if pipeline.timing: print(f"\n[性能统计] 总时间: {total_time:.1f}秒 | 平均: {np.mean(pipeline.timing)*1000:.1f}ms | FPS: {reader.nb_frames/total_time:.2f}")
        print(f"[性能统计] GFPGAN 模式: {gfpgan_mode}")
        if pipeline._face_filtered_total > 0: print(f"[性能统计] 人脸检测: 保留 {pipeline._face_count_total} 个, 过滤 {pipeline._face_filtered_total} 个 (阈值={pipeline.face_det_threshold})")
        if pipeline._enable_adaptive_batch: print(f"[性能统计] 最终人脸密度EMA: {pipeline._face_density_ema:.2f} 人脸/帧, 最终自适应arbs: {pipeline._adaptive_read_batch_size}")

def main():
    args = parse_args()
    print("\nReal-ESRGAN Video Enhancement v6.4 - 架构优化版")
    print("主要优化特性:")
    print("1. 深度流水线架构（4级并行处理）\n2. GPU内存池优化（避免频繁分配释放）\n3. 异步计算模式（多CUDA流并行）\n4. 多级缓冲队列（深度缓冲减少等待）\n5. 优化线程池配置（提高并发效率）")
    print("6. 人脸检测置信度过滤（减少无效 GFPGAN 推理）\n7. 自适应批处理（根据人脸密度动态调整）\n8. SR H2D 预取重叠（利用空闲 GPU 内存总线）\n")
    if args.no_adaptive_batch: args.adaptive_batch = False; print("已禁用自适应批处理大小（使用固定 --batch-size）")
    # 实测 A10 GPU 存在同样的问题，禁用所有 GPU 的 GFPGAN-TRT + FP16 组合
    if getattr(args, 'gfpgan_trt', False) and torch.cuda.is_available():
        args.no_fp16 = True
        print("[GFPGAN-TRT] 为防止 FP16 溢出噪斑，已自动禁用半精度并启用 FP32 推理。\n")
    # CUDA Graph (torch.compile) 与 TensorRT 互斥
    if getattr(args, 'use_tensorrt', False) and not getattr(args, 'no_compile', False):
        args.no_compile = True
        print("[优化架构] TensorRT 已启用，自动禁用 torch.compile (CUDA Graph)")
    main_optimized(args)

if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    main()