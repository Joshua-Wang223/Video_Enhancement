#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - GFPGAN 子进程模块
包含：SharedMemoryDoubleBuffer, GFPGANSubprocess
"""

import os
import sys
import time
import queue
import gc
import threading
import multiprocessing as mp
import multiprocessing.shared_memory as shm
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize as _tv_normalize
from config import models_RealESRGAN

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

# ─────────────────────────────────────────────────────────────────────────────
# SharedMemoryDoubleBuffer（优化3：零拷贝 IPC）
# ─────────────────────────────────────────────────────────────────────────────

class SharedMemoryDoubleBuffer:
    """双缓冲共享内存，用于主进程与 GFPGAN 子进程之间的零拷贝数据传输。"""

    N_SLOTS = 2
    MAX_FACES = 64
    FACE_SHAPE = (512, 512, 3)

    def __init__(self):
        face_bytes = int(np.prod(self.FACE_SHAPE))
        slot_bytes = self.MAX_FACES * face_bytes
        self._input_shms = []
        self._output_shms = []
        for _ in range(self.N_SLOTS):
            self._input_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
            self._output_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
        self._slot_pool = queue.Queue(maxsize=self.N_SLOTS)
        for i in range(self.N_SLOTS):
            self._slot_pool.put(i)

    @property
    def input_names(self) -> List[str]:
        return [s.name for s in self._input_shms]

    @property
    def output_names(self) -> List[str]:
        return [s.name for s in self._output_shms]

    def acquire_slot(self, timeout: float = 30.0) -> int:
        try:
            return self._slot_pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f'无法在 {timeout}s 内获取空闲 slot，'
                f'可能存在 slot 泄漏（未调用 release_slot）')

    def try_acquire_slot(self) -> Optional[int]:
        try:
            return self._slot_pool.get_nowait()
        except queue.Empty:
            return None

    def release_slot(self, slot: int):
        if slot is not None:
            self._slot_pool.put(slot)

    def write_input(self, slot: int, crops: List[np.ndarray]) -> int:
        n = min(len(crops), self.MAX_FACES)
        buf = np.ndarray(
            (self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8,
            buffer=self._input_shms[slot].buf)
        for i in range(n):
            c = crops[i]
            if c.shape == self.FACE_SHAPE:
                buf[i] = c
            else:
                import cv2 as _cv
                buf[i] = _cv.resize(c, (self.FACE_SHAPE[1], self.FACE_SHAPE[0]))
        return n

    def read_output(self, slot: int, n: int) -> List[np.ndarray]:
        buf = np.ndarray(
            (self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8,
            buffer=self._output_shms[slot].buf)
        return [buf[i].copy() for i in range(n)]

    def close(self):
        for s_list in [self._input_shms, self._output_shms]:
            for s in s_list:
                try:
                    s.close()
                except Exception:
                    pass
                try:
                    s.unlink()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# GFPGANSubprocess
# ─────────────────────────────────────────────────────────────────────────────

class GFPGANSubprocess:
    """
    将 GFPGAN 推理移至独立子进程，避免与主进程的 PyTorch 分配器冲突。
    支持 PyTorch FP16 或 TensorRT（取决于参数）。
    """
    def __init__(self, face_enhancer=None, device=None, gfpgan_weight=0.5, gfpgan_batch_size=4,
                 use_fp16=True, use_trt=False, trt_cache_dir=None, gfpgan_model='1.4',
                 model_path=None):
        self.device = device
        self.gfpgan_weight = gfpgan_weight
        self.gfpgan_batch_size = gfpgan_batch_size
        self.use_fp16 = use_fp16
        self.use_trt = use_trt
        self.trt_cache_dir = trt_cache_dir
        self.gfpgan_model = gfpgan_model

        if model_path is not None:
            self.model_path = model_path
        elif face_enhancer is not None:
            try:
                self.model_path = face_enhancer.model_path
            except AttributeError:
                pass

        if not hasattr(self, 'model_path'):
            if face_enhancer is not None:
                try:
                    self.model_path = face_enhancer.model_path
                except AttributeError:
                    pass
            if not hasattr(self, 'model_path'):
                model_paths = {
                    '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
                }
                model_url = model_paths.get(gfpgan_model, model_paths['1.4'])
                model_dir = os.path.join(models_RealESRGAN, 'GFPGAN')
                os.makedirs(model_dir, exist_ok=True)
                model_filename = os.path.basename(model_url)
                self.model_path = os.path.join(model_dir, model_filename)
                if not os.path.exists(self.model_path):
                    print(f'[GFPGANSubprocess] 下载模型: {model_filename}')
                    self.model_path = load_file_from_url(model_url, model_dir, True)

        if face_enhancer is not None:
            self.gfpgan_net = face_enhancer.gfpgan
            self.face_enhancer_upscale = face_enhancer.upscale
        else:
            self.gfpgan_net = None
            self.face_enhancer_upscale = 1

        self._mp_ctx = mp.get_context('spawn')
        self.task_queue = self._mp_ctx.Queue(maxsize=2)
        self.result_queue = self._mp_ctx.Queue(maxsize=2)
        self.ready_event = self._mp_ctx.Event()
        self.process = None

        # 两阶段子进程架构
        if use_trt:
            _sm_tag = ''
            if torch.cuda.is_available():
                _p = torch.cuda.get_device_properties(0)
                import re as _re_coord
                _gpu_slug_coord = _re_coord.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
                _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_coord}'
            tag = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
                   f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag}')
            cache_dir = trt_cache_dir or os.path.join(os.getcwd(), '.trt_cache')
            trt_path = os.path.join(cache_dir, f'{tag}.trt')

            if not os.path.exists(trt_path):
                print(f'[GFPGANSubprocess] Phase 1: 启动独立 Builder 进程构建 TRT Engine...', flush=True)
                print(f'[GFPGANSubprocess] 构建完成后 Builder 自动退出，再启动干净 Inference 进程', flush=True)
                builder = self._mp_ctx.Process(
                    target=GFPGANSubprocess._build_only_worker,
                    args=(self.model_path, gfpgan_model, gfpgan_weight,
                          gfpgan_batch_size, use_fp16, trt_cache_dir),
                    daemon=False,
                )
                builder.start()
                _p1_start = time.time()
                _p1_max = 5400
                _p1_poll = 5
                _p1_reported = False
                _p1_deadline = _p1_start + _p1_max
                while time.time() < _p1_deadline:
                    builder.join(timeout=_p1_poll)
                    if not builder.is_alive():
                        break
                    if not _p1_reported:
                        elapsed = time.time() - _p1_start
                        print(f'[GFPGANSubprocess] Phase 1 编译中... {elapsed:.0f}s（Builder 进程运行中）', flush=True)
                        _p1_reported = True
                _p1_elapsed = time.time() - _p1_start
                if builder.is_alive():
                    builder.terminate()
                    print(f'[GFPGANSubprocess] Builder 超时（>{_p1_max//60}min），TRT 构建失败', flush=True)
                elif os.path.exists(trt_path):
                    print(f'[GFPGANSubprocess] Phase 1 完成，用时 {_p1_elapsed:.0f}s，.trt 已生成，启动 Phase 2 Inference 进程', flush=True)
                else:
                    print(f'[GFPGANSubprocess] Phase 1 失败（用时 {_p1_elapsed:.0f}s，.trt 未生成），Phase 2 将走 PyTorch 路径', flush=True)

        self.shm_buf: Optional[SharedMemoryDoubleBuffer] = None
        try:
            self.shm_buf = SharedMemoryDoubleBuffer()
            print(f'[GFPGANSubprocess] 共享内存双缓冲已创建 '
                  f'(input: {self.shm_buf.input_names}, '
                  f'output: {self.shm_buf.output_names})', flush=True)
        except Exception as _shm_e:
            print(f'[GFPGANSubprocess] 共享内存创建失败，回退 pickle: {_shm_e}',
                  flush=True)
            self.shm_buf = None

        self._start()

    @staticmethod
    def _build_only_worker(model_path, gfpgan_model, gfpgan_weight,
                           gfpgan_batch_size, use_fp16, trt_cache_dir):
        """Phase 1 Builder 进程：仅做 TRT build，完成后立即退出。"""
        import warnings
        warnings.filterwarnings('ignore')
        import os, sys, gc, time, threading
        import os.path as osp
        import torch
        from gfpgan import GFPGANer
        import contextlib, torch.nn.functional as _F

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, _ = _GFPGAN_MODELS[gfpgan_model]
        face_enhancer = GFPGANer(
            model_path=model_path, upscale=1, arch=arch,
            channel_multiplier=channel_multiplier, bg_upsampler=None, device=device,
        )

        _sm_tag_b = ''
        if torch.cuda.is_available():
            _pb = torch.cuda.get_device_properties(0)
            import re as _re_b
            _gpu_slug_b = _re_b.sub(r'[^a-z0-9]', '', _pb.name.lower())[:16]
            _sm_tag_b = f'_sm{_pb.major}{_pb.minor}_{_gpu_slug_b}'
        tag = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
               f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_b}')
        cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache')
        trt_path = osp.join(cache_dir, f'{tag}.trt')
        onnx_path = osp.join(cache_dir, f'{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)

        if osp.exists(trt_path):
            print(f'[Builder] .trt 已存在，跳过构建: {trt_path}', flush=True)
            import os as _os
            _os._exit(0)

        try:
            import tensorrt as trt
        except ImportError as e:
            print(f'[Builder] tensorrt 未安装: {e}', flush=True)
            import os as _os
            _os._exit(1)

        @contextlib.contextmanager
        def _onnx_compat_patch():
            _patches = []
            try:
                import basicsr.ops.fused_act.fused_act as _fa_mod
                _orig = _fa_mod.fused_leaky_relu
                def _compat(inp, bias, negative_slope=0.2, scale=2**0.5):
                    bv = (bias.view(1, -1, 1, 1) if (bias.dim() == 1 and inp.dim() == 4) else bias)
                    return _F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale
                _fa_mod.fused_leaky_relu = _compat
                _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
                try:
                    import basicsr.ops.fused_act as _fa_pkg
                    if hasattr(_fa_pkg, 'fused_leaky_relu'):
                        _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                        _fa_pkg.fused_leaky_relu = _compat
                except Exception:
                    pass
            except Exception:
                pass
            try:
                import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
                if getattr(_ud_mod, '_use_custom_op', False):
                    _patches.append((_ud_mod, '_use_custom_op', True))
                    _ud_mod._use_custom_op = False
            except Exception:
                pass
            try:
                yield
            finally:
                for _obj, _attr, _orig in reversed(_patches):
                    try:
                        setattr(_obj, _attr, _orig)
                    except Exception:
                        pass

        gfpgan_net = face_enhancer.gfpgan.eval()
        dummy_d = torch.randn(1, 3, 512, 512, device=device)
        if use_fp16:
            dummy_d = dummy_d.half()
            gfpgan_net = gfpgan_net.half()
        with torch.no_grad():
            _test = gfpgan_net(dummy_d, return_rgb=True)
        _returns_tuple = isinstance(_test, (tuple, list))
        _w = float(gfpgan_weight)

        class _W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = gfpgan_net

            def forward(self, x):
                out = self.net(x, return_rgb=False)
                if _returns_tuple:
                    out = out[0]
                return out

        wrapper = _W().to(device)

        if use_fp16:
            wrapper = wrapper.half()
        wrapper.eval()

        print(f'[Builder] ONNX 导出 (静态 batch={gfpgan_batch_size})...', flush=True)
        dummy = torch.randn(gfpgan_batch_size, 3, 512, 512, device=device)
        if use_fp16:
            dummy = dummy.half()
        try:
            with _onnx_compat_patch():
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, dummy, onnx_path,
                        input_names=['input'], output_names=['output'],
                        opset_version=18, dynamo=False,
                    )
            print(f'[Builder] ONNX 已导出: {onnx_path}', flush=True)
        except Exception as e:
            print(f'[Builder] ONNX 导出失败: {e}', flush=True)
            import os as _os
            _os._exit(1)
        del wrapper, dummy, dummy_d, face_enhancer
        gc.collect()
        torch.cuda.empty_cache()

        _sm_ok = True
        _gpu_name_b = 'unknown'
        _sm_major = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name_b = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                print(f'[Builder] 警告: {_gpu_name_b} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）', flush=True)

        _sm_minor = 0
        if torch.cuda.is_available():
            _sm_minor = torch.cuda.get_device_properties(0).minor
        _sm_code = _sm_major * 10 + _sm_minor
        _time_hint = {
            75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
            80: '约需 8~15 分钟（A100/A30 SM8.0）',
            86: '约需 10~18 分钟（A10/RTX30系 SM8.6）',
            89: '约需 8~12 分钟（RTX40系 SM8.9）',
            90: '约需 5~10 分钟（H100 SM9.0）',
        }.get(_sm_code, f'约需 10~30 分钟（{_gpu_name_b} SM{_sm_major}.{_sm_minor}）')
        print(f'[Builder] 构建 TRT Engine (B={gfpgan_batch_size}, fp16={use_fp16})...', flush=True)
        print(f'[Builder] {_time_hint}', flush=True)
        try:
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            try:
                flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            except AttributeError:
                flag = 0
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors):
                    print(f'  [Builder] 解析错误: {parser.get_error(i)}', flush=True)
                import os as _os
                _os._exit(1)
            print('[Builder] ONNX 解析完成，开始编译...', flush=True)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
            if use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            _bs = gfpgan_batch_size
            profile.set_shape('input',
                              min=(_bs, 3, 512, 512), opt=(_bs, 3, 512, 512), max=(_bs, 3, 512, 512))
            config.add_optimization_profile(profile)
            _build_start = time.time()
            _build_done = threading.Event()

            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= 300:
                        elapsed = time.time() - _build_start
                        print(f'[Builder] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）', flush=True)
                        _last = time.time()

            _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()
            serialized = builder.build_serialized_network(network, config)
            _build_done.set()
            _build_elapsed = time.time() - _build_start
            del config, profile, parser, network, builder
            gc.collect()
            if serialized is None:
                _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
                _sm_hint = (f'\n[Builder] 提示: {_gpu_name_b} ({_sm_str}) 可能不受此 TRT 版本支持'
                            if _sm_major < 8 else '')
                print(f'[Builder] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}', flush=True)
                import os as _os
                _os._exit(1)
            with open(trt_path, 'wb') as f:
                f.write(serialized)
            del serialized
            print(f'[Builder] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}', flush=True)
        except Exception as e:
            print(f'[Builder] Engine 构建异常: {e}', flush=True)
            import os as _os
            _os._exit(1)
        import os as _os
        _os._exit(0)

    def _start(self):
        _shm_input_names = self.shm_buf.input_names if self.shm_buf else None
        _shm_output_names = self.shm_buf.output_names if self.shm_buf else None
        self.process = self._mp_ctx.Process(target=self._worker, args=(
            self.model_path, self.gfpgan_model, self.gfpgan_weight,
            self.gfpgan_batch_size, self.use_fp16, self.use_trt,
            self.trt_cache_dir, self.task_queue, self.result_queue,
            self.ready_event,
            _shm_input_names, _shm_output_names,
        ), daemon=True)
        self.process.start()

    @staticmethod
    def _worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size,
                use_fp16, use_trt, trt_cache_dir, task_queue, result_queue,
                ready_event=None,
                shm_input_names=None, shm_output_names=None):
        """子进程主函数：加载模型并循环处理任务"""
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore'
        import torch
        import numpy as np
        from gfpgan import GFPGANer
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
        import contextlib
        import time

        cuda_context_dead = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.init()

        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, name = _GFPGAN_MODELS[gfpgan_model]

        if use_trt and torch.cuda.is_available():
            print('[GFPGANSubprocess] FIX-INIT-ORDER: TRT 路径，延迟 GFPGANer GPU 加载', flush=True)
            face_enhancer = GFPGANer(
                model_path=model_path, upscale=1, arch=arch,
                channel_multiplier=channel_multiplier, bg_upsampler=None,
                device=torch.device('cpu'),
            )
            model = face_enhancer.gfpgan
            model.eval()
        else:
            face_enhancer = GFPGANer(
                model_path=model_path, upscale=1, arch=arch,
                channel_multiplier=channel_multiplier, bg_upsampler=None,
                device=device,
            )
            model = face_enhancer.gfpgan
            model.eval()

        gfpgan_trt_accel = None
        if use_trt and torch.cuda.is_available():
            try:
                from typing import Optional, List, Tuple
                import os
                import os.path as osp
                import torch
                import numpy as np
                import contextlib as _ctx
                import torch.nn.functional as _F

                def _get_subprocess_trt_logger():
                    import tensorrt as _trt
                    if not hasattr(_get_subprocess_trt_logger, '_inst'):
                        _get_subprocess_trt_logger._inst = _trt.Logger(_trt.Logger.ERROR)
                    return _get_subprocess_trt_logger._inst

                class GFPGANTRTAccelerator:
                    """GFPGAN TRT 加速（v6.2 验证版，移植入子进程）"""

                    def __init__(self, face_enhancer, device, cache_dir, gfpgan_weight,
                                 max_batch_size, gfpgan_version, use_fp16=True):
                        self.device = device
                        self.use_fp16 = use_fp16
                        self._max_batch_size = max_batch_size
                        self._engine = None
                        self._context = None
                        self._trt_ok = False
                        self._trt_stream = None
                        self._input_name = None
                        self._output_name = None
                        self._use_new_api = False
                        self._cuda_context_dead = False
                        self._warmup_failed = False

                        try:
                            import tensorrt as trt
                            self._trt = trt
                        except ImportError as e:
                            print(f'[GFPGAN-TensorRT] tensorrt 未安装: {e}', flush=True)
                            return

                        _sm_tag_w = ''
                        try:
                            import torch as _t
                            if _t.cuda.is_available():
                                _pw = _t.cuda.get_device_properties(0)
                                import re as _re_w
                                _gpu_slug_w = _re_w.sub(r'[^a-z0-9]', '', _pw.name.lower())[:16]
                                _sm_tag_w = f'_sm{_pw.major}{_pw.minor}_{_gpu_slug_w}'
                        except Exception:
                            pass
                        tag = (f'gfpgan_{gfpgan_version}_w{gfpgan_weight:.3f}'
                               f'_B{max_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_w}')
                        os.makedirs(cache_dir, exist_ok=True)
                        trt_path = osp.join(cache_dir, f'{tag}.trt')
                        onnx_path = osp.join(cache_dir, f'{tag}.onnx')
                        self._trt_path = trt_path
                        if not osp.exists(trt_path):
                            print(f'[GFPGAN-TensorRT] .trt 不存在，跳过构建，走 PyTorch 路径', flush=True)
                            return
                        if osp.exists(trt_path):
                            try:
                                self._load_engine(trt_path)
                            except RuntimeError as _e:
                                print(f'[GFPGAN-TensorRT] 首次加载失败({_e})，重建...', flush=True)
                                wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)
                                if not osp.exists(onnx_path):
                                    self._export_onnx(wrapper, onnx_path, max_batch_size)
                                if osp.exists(onnx_path):
                                    self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                                    if osp.exists(trt_path):
                                        self._load_engine(trt_path)

                    @staticmethod
                    def _build_wrapper(gfpgan_net, weight, device, use_fp16):
                        gfpgan_net = gfpgan_net.eval()
                        actual_device = next(gfpgan_net.parameters()).device
                        dummy = torch.randn(1, 3, 512, 512, device=actual_device)
                        if use_fp16:
                            dummy = dummy.half()
                            gfpgan_net = gfpgan_net.half()
                        with torch.no_grad():
                            _test = gfpgan_net(dummy, return_rgb=True)
                        _returns_tuple = isinstance(_test, (tuple, list))

                        class _W(torch.nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.net = gfpgan_net

                            def forward(self, x):
                                out = self.net(x, return_rgb=False)
                                if _returns_tuple:
                                    out = out[0]
                                return out

                        return _W().to(device)

                    @staticmethod
                    @_ctx.contextmanager
                    def _onnx_compat_patch():
                        _patches = []
                        try:
                            import basicsr.ops.fused_act.fused_act as _fa_mod
                            _orig = _fa_mod.fused_leaky_relu

                            def _compat(inp, bias, negative_slope=0.2, scale=2 ** 0.5):
                                bv = (bias.view(1, -1, 1, 1) if (bias.dim() == 1 and inp.dim() == 4) else bias)
                                return _F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale

                            _fa_mod.fused_leaky_relu = _compat
                            _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
                            try:
                                import basicsr.ops.fused_act as _fa_pkg
                                if hasattr(_fa_pkg, 'fused_leaky_relu'):
                                    _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                                    _fa_pkg.fused_leaky_relu = _compat
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
                            if getattr(_ud_mod, '_use_custom_op', False):
                                _patches.append((_ud_mod, '_use_custom_op', True))
                                _ud_mod._use_custom_op = False
                        except Exception:
                            pass
                        try:
                            yield
                        finally:
                            for _obj, _attr, _orig in reversed(_patches):
                                try:
                                    setattr(_obj, _attr, _orig)
                                except Exception:
                                    pass

                    def _export_onnx(self, wrapper, onnx_path, max_batch_size):
                        wrapper = wrapper.eval()
                        dummy = torch.randn(max_batch_size, 3, 512, 512, device=self.device)
                        if self.use_fp16:
                            dummy = dummy.half()
                            wrapper = wrapper.half()
                        try:
                            with self._onnx_compat_patch():
                                with torch.no_grad():
                                    torch.onnx.export(wrapper, dummy, onnx_path,
                                                      input_names=['input'], output_names=['output'],
                                                      opset_version=18, dynamo=False)
                            return True
                        except Exception as _e:
                            print(f'[GFPGAN-TensorRT] ONNX 导出失败: {_e}', flush=True)
                            return False

                    def _build_engine_dynamic(self, onnx_path, trt_path, max_batch_size, use_fp16):
                        trt = self._trt
                        logger = _get_subprocess_trt_logger()
                        builder = trt.Builder(logger)
                        try:
                            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                        except AttributeError:
                            flag = 0
                        network = builder.create_network(flag)
                        parser = trt.OnnxParser(network, logger)
                        if not parser.parse_from_file(onnx_path):
                            del parser, network, builder
                            return
                        config = builder.create_builder_config()
                        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
                        if use_fp16 and builder.platform_has_fast_fp16:
                            config.set_flag(trt.BuilderFlag.FP16)
                        profile = builder.create_optimization_profile()
                        _bs = max_batch_size
                        profile.set_shape('input', min=(_bs, 3, 512, 512), opt=(_bs, 3, 512, 512), max=(_bs, 3, 512, 512))
                        config.add_optimization_profile(profile)
                        serialized = builder.build_serialized_network(network, config)
                        del config, profile, parser, network, builder
                        import gc
                        gc.collect()
                        if serialized is None:
                            return
                        with open(trt_path, 'wb') as f:
                            f.write(serialized)
                        del serialized

                    def _load_engine(self, trt_path):
                        _cur_sm_tag = ''
                        if torch.cuda.is_available():
                            _pp = torch.cuda.get_device_properties(0)
                            import re as _re
                            _gpu_slug = _re.sub(r'[^a-z0-9]', '', _pp.name.lower())[:16]
                            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{_gpu_slug}'
                        if _cur_sm_tag:
                            import os.path as _osp2
                            _basename = _osp2.basename(trt_path)
                            if _cur_sm_tag not in _basename:
                                try:
                                    os.remove(trt_path)
                                except OSError:
                                    pass
                                raise RuntimeError(f'过期缓存 {_basename} 已删除，需重建')
                        trt = self._trt
                        logger = _get_subprocess_trt_logger()
                        runtime = trt.Runtime(logger)
                        with open(trt_path, 'rb') as f:
                            self._engine = runtime.deserialize_cuda_engine(f.read())
                        del runtime
                        if self._engine is None:
                            try:
                                os.remove(trt_path)
                            except OSError:
                                pass
                            raise RuntimeError('deserialize returned None')
                        self._context = self._engine.create_execution_context()
                        if self._context is None:
                            raise RuntimeError('create_execution_context returned None')
                        try:
                            torch.cuda.synchronize(self.device)
                        except Exception as _ce:
                            self._cuda_context_dead = True
                            return
                        self._use_new_api = hasattr(self._engine, 'num_io_tensors')
                        if self._use_new_api:
                            for i in range(self._engine.num_io_tensors):
                                name = self._engine.get_tensor_name(i)
                                mode = self._engine.get_tensor_mode(name)
                                if mode == trt.TensorIOMode.INPUT:
                                    self._input_name = name
                                elif mode == trt.TensorIOMode.OUTPUT:
                                    self._output_name = name
                            print(f'[GFPGAN-TRT] TRT 10.x | in={self._input_name}, out={self._output_name}', flush=True)
                        else:
                            print('[GFPGAN-TRT] TRT 8.x API', flush=True)
                        self._trt_ok = True
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(self.device)
                            _dt = torch.float16 if self.use_fp16 else torch.float32
                            _B = self._max_batch_size
                            _FULL = (_B, 3, 512, 512)
                            _inp = torch.zeros(_FULL, dtype=_dt, device=self.device)
                            _out = torch.zeros(_FULL, dtype=_dt, device=self.device)
                            _warmup_stream = torch.cuda.Stream(device=self.device)
                            _warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
                            if self._use_new_api:
                                self._context.set_input_shape(self._input_name, _FULL)
                                self._context.set_tensor_address(self._input_name, _inp.data_ptr())
                                self._context.set_tensor_address(self._output_name, _out.data_ptr())
                                self._context.execute_async_v3(stream_handle=_warmup_stream.cuda_stream)
                            else:
                                self._context.set_binding_shape(0, _FULL)
                                self._context.execute_async_v2(
                                    bindings=[_inp.data_ptr(), _out.data_ptr()],
                                    stream_handle=_warmup_stream.cuda_stream)
                            _warmup_stream.synchronize()
                            del _inp, _out, _warmup_stream
                            print('[GFPGAN-TensorRT] Warmup 通过', flush=True)
                        except Exception as _we:
                            print(f'[GFPGAN-TensorRT] Warmup 失败: {_we}', flush=True)
                            self._warmup_failed = True
                            self._cuda_context_dead = True
                            return

                    @property
                    def available(self):
                        return self._trt_ok

                    def infer(self, face_tensor):
                        if self._trt_stream is None:
                            self._trt_stream = torch.cuda.Stream(device=self.device)
                        B = face_tensor.shape[0]
                        max_bs = self._max_batch_size
                        dtype = torch.float16 if self.use_fp16 else torch.float32
                        _FULL = (max_bs, 3, 512, 512)
                        if (not hasattr(self, '_inp_buf')
                                or self._inp_buf.shape != torch.Size(_FULL)
                                or self._inp_buf.dtype != dtype):
                            self._inp_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
                            self._out_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
                        self._inp_buf.zero_()
                        self._inp_buf[:B].copy_(face_tensor[:B])
                        self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
                        if self._use_new_api:
                            self._context.set_input_shape(self._input_name, _FULL)
                            self._context.set_tensor_address(self._input_name, self._inp_buf.data_ptr())
                            self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
                            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
                        else:
                            self._context.set_binding_shape(0, _FULL)
                            self._context.execute_async_v2(
                                bindings=[self._inp_buf.data_ptr(), self._out_buf.data_ptr()],
                                stream_handle=self._trt_stream.cuda_stream)
                        self._trt_stream.synchronize()
                        return self._out_buf[:B].clone()

                gfpgan_trt_accel = GFPGANTRTAccelerator(
                    face_enhancer=face_enhancer, device=device,
                    cache_dir=trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'),
                    gfpgan_weight=gfpgan_weight, max_batch_size=gfpgan_batch_size,
                    gfpgan_version=gfpgan_model, use_fp16=use_fp16)
                if gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] TRT 加速已启用（子进程版本）', flush=True)
                else:
                    print('[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch', flush=True)
                    use_trt = False
            except Exception as e:
                print(f'[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch: {e}', flush=True)
                use_trt = False
                gfpgan_trt_accel = None

        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            print('[GFPGANSubprocess] TRT warmup 失败导致 CUDA context 损坏', flush=True)
            # import time
            time.sleep(0.5)
            import os as _os
            _os._exit(0)

        _model_needs_gpu = False
        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            print('[GFPGANSubprocess] TRT warmup 通过，迁移 GFPGANer 到 GPU...', flush=True)
            _model_needs_gpu = True
        elif use_trt and not (gfpgan_trt_accel is not None
                              and getattr(gfpgan_trt_accel, '_cuda_context_dead', False)):
            print('[GFPGANSubprocess] TRT 失败，迁移 GFPGANer 到 GPU...', flush=True)
            _model_needs_gpu = True

        if _model_needs_gpu:
            face_enhancer.gfpgan = face_enhancer.gfpgan.to(device)
            model = face_enhancer.gfpgan
            if use_fp16:
                face_enhancer.gfpgan = face_enhancer.gfpgan.half()
                model = face_enhancer.gfpgan
            face_enhancer.face_helper = None
            print('[GFPGANSubprocess] GFPGANer 已迁移到 GPU', flush=True)

        fp16_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            if ready_event is not None:
                ready_event.set()
            import os as _os, time
            time.sleep(0.5)
            _os._exit(0)

        if cuda_context_dead:
            pass
        elif use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            print('[GFPGANSubprocess] TRT 成功，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()
        elif use_trt and (gfpgan_trt_accel is None or not gfpgan_trt_accel.available):
            print('[GFPGANSubprocess] TRT 失败但 context 正常，以 PyTorch 模式服务', flush=True)
            if ready_event is not None:
                ready_event.set()
        else:
            print('[GFPGANSubprocess] PyTorch 模式，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()

        # 共享内存 attach
        import multiprocessing.shared_memory as shm
        _shm_inputs = []
        _shm_outputs = []
        _shm_available = False
        _shm_max_faces = 64
        _shm_face_shape = (512, 512, 3)
        if shm_input_names and shm_output_names:
            try:
                for _sname in shm_input_names:
                    _shm_inputs.append(shm.SharedMemory(name=_sname))
                for _sname in shm_output_names:
                    _shm_outputs.append(shm.SharedMemory(name=_sname))
                _shm_available = True
                print(f'[GFPGANSubprocess] 共享内存 attach 成功 '
                      f'({len(_shm_inputs)} input slots + '
                      f'{len(_shm_outputs)} output slots)', flush=True)
            except Exception as _shm_e:
                print(f'[GFPGANSubprocess] 共享内存 attach 失败，回退 pickle: '
                      f'{_shm_e}', flush=True)
                _shm_available = False
                _shm_inputs = []
                _shm_outputs = []
        else:
            print('[GFPGANSubprocess] 未传入共享内存名称，使用 pickle 传输',
                  flush=True)

        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:
                break

            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__validate__':
                _val_id = task[1]
                if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] post-SR warmup（真实显存压力下）...', flush=True)
                    try:
                        torch.cuda.empty_cache()
                        _vdt = torch.float16 if use_fp16 else torch.float32
                        _vdmy = torch.zeros(gfpgan_batch_size, 3, 512, 512, dtype=_vdt, device=device)
                        _vout = gfpgan_trt_accel.infer(_vdmy)
                        del _vdmy, _vout
                        torch.cuda.synchronize(device)
                        print('[GFPGANSubprocess] post-SR warmup 通过，TRT 推理正式启用', flush=True)
                        result_queue.put(('__validate__', _val_id, True), timeout=5.0)
                    except Exception as _ve:
                        _ve_str = str(_ve).lower()
                        _ctx_dead = any(kw in _ve_str for kw in (
                            'illegal memory', 'cudaerrorillegaladdress',
                            'illegal instruction', 'prior launch failure',
                            'acceleratorerror', 'cudaerror'))
                        _is_oom = 'out of memory' in _ve_str
                        if _ctx_dead:
                            if 'out of memory' in _ve_str:
                                gfpgan_trt_accel._trt_ok = False
                                use_trt = False
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                            else:
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                                os._exit(0)
                        else:
                            gfpgan_trt_accel._trt_ok = False
                            use_trt = False
                            result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                else:
                    result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                continue

            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__pause__':
                _pause_duration = task[1]
                torch.cuda.empty_cache()
                time.sleep(_pause_duration)
                torch.cuda.empty_cache()
                continue

            _use_shm_output = False
            _shm_slot_id = -1
            if (isinstance(task, tuple) and len(task) == 3
                    and isinstance(task[1], int) and isinstance(task[2], int)):
                task_id, _n_faces, _shm_slot_id = task
                if _shm_available and _shm_slot_id < len(_shm_inputs):
                    _inp_buf = np.ndarray(
                        (_shm_max_faces, *_shm_face_shape), dtype=np.uint8,
                        buffer=_shm_inputs[_shm_slot_id].buf)
                    crops_np = [_inp_buf[i].copy() for i in range(_n_faces)]
                    _use_shm_output = True
                else:
                    print(f'[GFPGANSubprocess] 共享内存 slot {_shm_slot_id} 不可用，跳过',
                          flush=True)
                    try:
                        result_queue.put((task_id, []), timeout=5.0)
                    except queue.Full:
                        pass
                    continue
            else:
                task_id, crops_np = task

            crops_tensor = []
            for crop in crops_np:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                crops_tensor.append(t)
            if not crops_tensor:
                result_queue.put((task_id, []), timeout=5.0)
                continue

            all_out = []
            sub_bs = gfpgan_batch_size
            i = 0
            while i < len(crops_tensor):
                sub = crops_tensor[i:i + sub_bs]
                sub_batch = torch.stack(sub).to(device)
                try:
                    with torch.no_grad():
                        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                            if use_fp16:
                                sub_batch = sub_batch.half()
                            out = gfpgan_trt_accel.infer(sub_batch)
                            out = out.float() if out.dtype != torch.float32 else out
                            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                            out = torch.clamp(out, min=-1.0, max=1.0)
                            if abs(gfpgan_weight - 1.0) > 1e-6:
                                _sub_b = sub_batch.float() if sub_batch.dtype == torch.float16 else sub_batch
                                out = gfpgan_weight * out + (1.0 - gfpgan_weight) * _sub_b
                            out = torch.clamp(out, min=-1.0, max=1.0)
                        else:
                            with fp16_ctx:
                                out = model(sub_batch, return_rgb=False)
                                if isinstance(out, (tuple, list)):
                                    out = out[0]
                            out = out.float()
                            if abs(gfpgan_weight - 1.0) > 1e-6:
                                out = gfpgan_weight * out + (1.0 - gfpgan_weight) * sub_batch.float()
                    all_out.extend(out.unbind(0))
                    i += len(sub)
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if 'out of memory' in error_str and sub_bs > 1:
                        sub_bs = max(1, sub_bs // 2)
                        torch.cuda.empty_cache()
                        print(f'[GFPGANSubprocess] GFPGAN OOM，sub_bs 降级至 {sub_bs}，重试...', flush=True)
                    elif 'cudaerrorillegaladdress' in error_str or 'illegal memory' in error_str:
                        print(f'[GFPGANSubprocess] CUDA 非法内存访问，切换到 PyTorch 路径: {e}')
                        use_trt = False
                        gfpgan_trt_accel = None
                        torch.cuda.empty_cache()
                    else:
                        all_out.extend([None] * len(sub))
                        i += len(sub)
                        torch.cuda.empty_cache()
                finally:
                    del sub_batch

            restored = []
            for out_t in all_out:
                if out_t is None:
                    restored.append(None)
                else:
                    out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=-1.0)
                    out_t = torch.clamp(out_t, min=-1.0, max=1.0)
                    img = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
                    restored.append(img.astype('uint8'))

            if _use_shm_output and _shm_slot_id >= 0 and _shm_slot_id < len(_shm_outputs):
                _out_buf = np.ndarray(
                    (_shm_max_faces, *_shm_face_shape), dtype=np.uint8,
                    buffer=_shm_outputs[_shm_slot_id].buf)
                _n_valid = 0
                for i, r in enumerate(restored):
                    if r is not None and i < _shm_max_faces:
                        if r.shape == _shm_face_shape:
                            _out_buf[i] = r
                        else:
                            import cv2 as _cv_w
                            _out_buf[i] = _cv_w.resize(
                                r, (_shm_face_shape[1], _shm_face_shape[0]))
                        _n_valid += 1
                    elif i < _shm_max_faces:
                        _out_buf[i] = 0
                try:
                    result_queue.put((task_id, len(restored)), timeout=5.0)
                except queue.Full:
                    pass
            else:
                try:
                    result_queue.put((task_id, restored), timeout=5.0)
                except queue.Full:
                    pass

        for _s in _shm_inputs + _shm_outputs:
            try:
                _s.close()
            except Exception:
                pass

        if gfpgan_trt_accel is not None:
            gfpgan_trt_accel._trt_ok = False
        try:
            del face_enhancer, model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        import os as _os_worker_exit
        _os_worker_exit._exit(0)

    def infer(self, crops_list):
        if not self.process or not self.process.is_alive():
            return [None] * len(crops_list)
        task_id = id(crops_list)
        try:
            self.task_queue.put((task_id, crops_list), timeout=10.0)
        except queue.Full:
            return [None] * len(crops_list)
        while True:
            try:
                res = self.result_queue.get(timeout=60)
            except queue.Empty:
                raise RuntimeError('GFPGAN子进程超时无响应')
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__':
                continue
            res_id, result = res
            if res_id == task_id:
                return result
            self.result_queue.put((res_id, result))

    def post_sr_validate(self) -> bool:
        if not self.process or not self.process.is_alive():
            return False
        val_id = id(self)
        try:
            self.task_queue.put(('__validate__', val_id), timeout=5.0)
        except queue.Full:
            return False
        deadline = time.time() + 180
        while time.time() < deadline:
            if not self.process.is_alive():
                return False
            try:
                res = self.result_queue.get(timeout=5)
            except queue.Empty:
                continue
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__' and res[1] == val_id:
                return res[2]
            self.result_queue.put(res)
        return False

    def pause(self, duration: float = 5.0):
        if not self.process or not self.process.is_alive():
            return
        try:
            self.task_queue.put(('__pause__', duration), timeout=2.0)
        except queue.Full:
            pass

    def close(self):
        if self.process and self.process.is_alive():
            try:
                self.task_queue.put(None, timeout=3)
            except Exception:
                pass
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                print('[GFPGANSubprocess] 子进程未响应 SIGTERM，发送 SIGKILL...', flush=True)
                self.process.kill()
                self.process.join(timeout=5)

        if self.shm_buf is not None:
            self.shm_buf.close()
            self.shm_buf = None

        try:
            self.task_queue.close()
        except Exception:
            pass
        try:
            self.result_queue.close()
        except Exception:
            pass