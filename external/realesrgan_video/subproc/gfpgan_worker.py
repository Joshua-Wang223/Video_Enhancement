#!/usr/bin/env python3
import os, sys, time, gc, queue, threading, warnings
import os.path as osp
import torch
import numpy as np
import multiprocessing as mp
import multiprocessing.shared_memory as shm
from typing import List, Optional, Tuple
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize as _tv_normalize
import contextlib, torch.nn.functional as _F

warnings.filterwarnings('ignore')

class GFPGANSubprocess:
    def __init__(self, face_enhancer=None, device=None, gfpgan_weight=0.5, gfpgan_batch_size=4, use_fp16=True, use_trt=False, trt_cache_dir=None, gfpgan_model='1.4', model_path=None):
        self.device, self.gfpgan_weight, self.gfpgan_batch_size, self.use_fp16, self.use_trt = device, gfpgan_weight, gfpgan_batch_size, use_fp16, use_trt
        self.trt_cache_dir, self.gfpgan_model = trt_cache_dir, gfpgan_model
        if model_path is not None: self.model_path = model_path
        elif face_enhancer is not None:
            try: self.model_path = face_enhancer.model_path
            except: pass
        if not hasattr(self, 'model_path'):
            model_paths = {'1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'}
            model_url = model_paths.get(gfpgan_model, model_paths['1.4'])
            model_dir = osp.join(os.path.dirname(os.path.dirname(__file__)), 'models_RealESRGAN', 'GFPGAN')
            os.makedirs(model_dir, exist_ok=True)
            from basicsr.utils.download_util import load_file_from_url
            self.model_path = osp.join(model_dir, osp.basename(model_url))
            if not osp.exists(self.model_path): print(f'[GFPGANSubprocess] 下载模型: {osp.basename(model_url)}'); self.model_path = load_file_from_url(model_url, model_dir, True)
        if face_enhancer is not None:
            self.gfpgan_net = face_enhancer.gfpgan; self.face_enhancer_upscale = face_enhancer.upscale
        self._mp_ctx = mp.get_context('spawn')
        
        # ── 两阶段子进程架构（FIX-TWO-PHASE）────────────────────────
        if use_trt:
            _sm_tag = ''
            if torch.cuda.is_available():
                _p = torch.cuda.get_device_properties(0)
                import re as _re_coord
                _gpu_slug_coord = _re_coord.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
                _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_coord}'
            tag = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
                   f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag}')
            cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache')
            trt_path = osp.join(cache_dir, f'{tag}.trt')

            if not osp.exists(trt_path):
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
                elif osp.exists(trt_path):
                    print(f'[GFPGANSubprocess] Phase 1 完成，用时 {_p1_elapsed:.0f}s，.trt 已生成，启动 Phase 2 Inference 进程', flush=True)
                else:
                    print(f'[GFPGANSubprocess] Phase 1 失败（用时 {_p1_elapsed:.0f}s，.trt 未生成），Phase 2 将走 PyTorch 路径', flush=True)

        self.task_queue = self._mp_ctx.Queue(maxsize=2); self.result_queue = self._mp_ctx.Queue(maxsize=2); self.ready_event = self._mp_ctx.Event()
        self.process = None
        self.shm_buf: Optional['SharedMemoryDoubleBuffer'] = None
        try:
            from realesrgan_video.utils.shm import SharedMemoryDoubleBuffer
            self.shm_buf = SharedMemoryDoubleBuffer()
            print(f'[GFPGANSubprocess] 共享内存双缓冲已创建 (input: {self.shm_buf.input_names}, output: {self.shm_buf.output_names})', flush=True)
        except Exception as _shm_e: print(f'[GFPGANSubprocess] 共享内存创建失败，回退 pickle: {_shm_e}', flush=True)
        # Phase 2: 始终启动 Inference 进程（干净 CUDA context）
        self._start()

    def _start(self):
        _shm_in = self.shm_buf.input_names if self.shm_buf else None
        _shm_out = self.shm_buf.output_names if self.shm_buf else None
        self.process = self._mp_ctx.Process(target=GFPGANSubprocess._worker, args=(self.model_path, self.gfpgan_model, self.gfpgan_weight, self.gfpgan_batch_size, self.use_fp16, self.use_trt, self.trt_cache_dir, self.task_queue, self.result_queue, self.ready_event, _shm_in, _shm_out), daemon=True)
        self.process.start()

    @staticmethod
    def _build_only_worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size, use_fp16, trt_cache_dir):
        import warnings; warnings.filterwarnings('ignore')
        import os, sys, gc, time, threading, os.path as osp
        import torch; from gfpgan import GFPGANer
        import contextlib, torch.nn.functional as _F
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _GFPGAN_MODELS = {'1.3': ('clean', 2, 'GFPGANv1.3'), '1.4': ('clean', 2, 'GFPGANv1.4'), 'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer')}
        arch, channel_multiplier, _ = _GFPGAN_MODELS[gfpgan_model]
        face_enhancer = GFPGANer(model_path=model_path, upscale=1, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=None, device=device)
        _sm_tag_b = ''; _gpu_name_b = 'unknown'
        if torch.cuda.is_available():
            _pb = torch.cuda.get_device_properties(0)
            import re as _re_b
            _sm_tag_b = f'_sm{_pb.major}{_pb.minor}_{_re_b.sub(r"[^a-z0-9]", "", _pb.name.lower())[:16]}'
            _gpu_name_b = _pb.name
        tag = f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_b}'
        cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'); trt_path = osp.join(cache_dir, f'{tag}.trt'); onnx_path = osp.join(cache_dir, f'{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)
        if osp.exists(trt_path): print(f'[Builder] .trt 已存在，跳过构建: {trt_path}', flush=True); os._exit(0)
        try: import tensorrt as trt
        except ImportError as e: print(f'[Builder] tensorrt 未安装: {e}', flush=True); os._exit(1)
        @contextlib.contextmanager
        def _onnx_compat_patch():
            _patches = []
            try:
                import basicsr.ops.fused_act.fused_act as _fa_mod; _orig = _fa_mod.fused_leaky_relu
                def _compat(inp, bias, negative_slope=0.2, scale=2**0.5): bv = (bias.view(1,-1,1,1) if (bias.dim()==1 and inp.dim()==4) else bias); return _F.leaky_relu(inp+bv, negative_slope=negative_slope)*scale
                _fa_mod.fused_leaky_relu = _compat; _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
            except: pass
            try: yield
            finally:
                for _obj, _attr, _orig in reversed(_patches):
                    try: setattr(_obj, _attr, _orig)
                    except: pass
        gfpgan_net = face_enhancer.gfpgan.eval(); dummy_d = torch.randn(1, 3, 512, 512, device=device)
        if use_fp16: dummy_d = dummy_d.half(); gfpgan_net = gfpgan_net.half()
        with torch.no_grad(): _test = gfpgan_net(dummy_d, return_rgb=True); _returns_tuple = isinstance(_test, (tuple, list))
        class _W(torch.nn.Module):
            def __init__(self): super().__init__(); self.net = gfpgan_net
            def forward(self, x): out = self.net(x, return_rgb=False); return out[0] if _returns_tuple else out
        wrapper = _W().to(device)
        if use_fp16: wrapper = wrapper.half()
        wrapper.eval()
        print(f'[Builder] ONNX 导出 (静态 batch={gfpgan_batch_size})...', flush=True)
        dummy = torch.randn(gfpgan_batch_size, 3, 512, 512, device=device)
        if use_fp16: dummy = dummy.half()
        try:
            with _onnx_compat_patch(), torch.no_grad(): torch.onnx.export(wrapper, dummy, onnx_path, input_names=['input'], output_names=['output'], opset_version=18, dynamo=False)
            print(f'[Builder] ONNX 已导出: {onnx_path}', flush=True)
        except Exception as e: print(f'[Builder] ONNX 导出失败: {e}', flush=True); os._exit(1)
        del wrapper, dummy, dummy_d, face_enhancer; gc.collect(); torch.cuda.empty_cache()
        print(f'[Builder] 构建 TRT Engine (B={gfpgan_batch_size}, fp16={use_fp16})...', flush=True)
        _sm_ok = True
        _gpu_name_b = 'unknown'
        _sm_major = 0
        _sm_minor = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name_b = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                print(f'[Builder] 警告: {_gpu_name_b} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）', flush=True)
        _sm_code = _sm_major * 10 + _sm_minor
        _time_hint = {
            75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
            80: '约需 8~15 分钟（A100/A30 SM8.0）',
            86: '约需 10~18 分钟（A10/RTX30系 SM8.6）',
            89: '约需 8~12 分钟（RTX40系 SM8.9）',
            90: '约需 5~10 分钟（H100 SM9.0）',
        }.get(_sm_code, f'约需 10~30 分钟（{_gpu_name_b} SM{_sm_major}.{_sm_minor}）')
        print(f'[Builder] {_time_hint}', flush=True)
        try:
            logger = trt.Logger(trt.Logger.ERROR); builder = trt.Builder(logger)
            try: flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            except: flag = 0
            network = builder.create_network(flag); parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors): print(f'  [Builder] 解析错误: {parser.get_error(i)}', flush=True); os._exit(1)
            print('[Builder] ONNX 解析完成，开始编译...', flush=True)
            config = builder.create_builder_config(); config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
            if use_fp16 and builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile(); profile.set_shape('input', min=(gfpgan_batch_size,3,512,512), opt=(gfpgan_batch_size,3,512,512), max=(gfpgan_batch_size,3,512,512)); config.add_optimization_profile(profile)
            _build_start = time.time(); _build_done = threading.Event()
            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= 300:
                        elapsed = time.time() - _build_start
                        print(f'[Builder] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）', flush=True)
                        _last = time.time()
            threading.Thread(target=_heartbeat, daemon=True).start()
            serialized = builder.build_serialized_network(network, config); _build_done.set()
            _build_elapsed = time.time() - _build_start
            del config, profile, parser, network, builder; gc.collect()
            if serialized is None:
                _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
                _sm_hint = (f'\n[Builder] 提示: {_gpu_name_b} ({_sm_str}) 可能不受此 TRT 版本支持'
                            if _sm_major < 8 else '')
                print(f'[Builder] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}', flush=True)
                os._exit(1)
            with open(trt_path, 'wb') as f: f.write(serialized); del serialized
            print(f'[Builder] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}', flush=True)
        except Exception as e: print(f'[Builder] Engine 构建异常: {e}', flush=True); os._exit(1)
        os._exit(0)

    @staticmethod
    def _worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size, use_fp16, use_trt, trt_cache_dir, task_queue, result_queue, ready_event=None, shm_input_names=None, shm_output_names=None):
        import warnings; warnings.filterwarnings('ignore')
        import os, torch, numpy as np, queue, time, multiprocessing.shared_memory as shm
        from gfpgan import GFPGANer; from basicsr.utils import img2tensor, tensor2img; from torchvision.transforms.functional import normalize as _tv_normalize
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available(): torch.cuda.init()
        _GFPGAN_MODELS = {'1.3': ('clean', 2, 'GFPGANv1.3'), '1.4': ('clean', 2, 'GFPGANv1.4'), 'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer')}
        arch, channel_multiplier, name = _GFPGAN_MODELS[gfpgan_model]
        face_enhancer = GFPGANer(model_path=model_path, upscale=1, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=None, device=torch.device('cpu'))
        model = face_enhancer.gfpgan; model.eval()
        gfpgan_trt_accel = None
        if use_trt and torch.cuda.is_available():
            print('[GFPGANSubprocess] FIX-INIT-ORDER: TRT 路径，延迟 GFPGANer GPU 加载', flush=True)
            try:
                import os.path as osp
                class GFPGANTRTAccelerator:
                    def __init__(self, face_enhancer, device, cache_dir, gfpgan_weight, max_batch_size, gfpgan_version, use_fp16=True):
                        self.device, self.use_fp16, self._max_batch_size = device, use_fp16, max_batch_size
                        self._engine, self._context, self._trt_ok, self._trt_stream = None, None, False, None
                        self._input_name, self._output_name, self._use_new_api = None, None, False
                        self._cuda_context_dead, self._warmup_failed = False, False
                        try: import tensorrt as trt; self._trt = trt
                        except ImportError as e: print(f'[GFPGAN-TensorRT] tensorrt 未安装: {e}', flush=True); return
                        _sm_tag_w = ''; import re as _re_w
                        if torch.cuda.is_available():
                            _pw = torch.cuda.get_device_properties(0)
                            _sm_tag_w = f'_sm{_pw.major}{_pw.minor}_{_re_w.sub(r"[^a-z0-9]", "", _pw.name.lower())[:16]}'
                        tag = f'gfpgan_{gfpgan_version}_w{gfpgan_weight:.3f}_B{max_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_w}'
                        os.makedirs(cache_dir, exist_ok=True); trt_path = osp.join(cache_dir, f'{tag}.trt'); onnx_path = osp.join(cache_dir, f'{tag}.onnx')
                        self._trt_path = trt_path
                        if not osp.exists(trt_path): print(f'[GFPGAN-TensorRT] .trt 不存在（构建失败或未构建），走 PyTorch 路径', flush=True); return
                        try: self._load_engine(trt_path)
                        except RuntimeError as _e:
                            print(f'[GFPGAN-TensorRT] 首次加载失败({_e})，重建...', flush=True)
                            wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)
                            if not osp.exists(onnx_path): self._export_onnx(wrapper, onnx_path, max_batch_size)
                            if osp.exists(onnx_path): self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                            if osp.exists(trt_path): self._load_engine(trt_path)
                    @staticmethod
                    def _build_wrapper(gfpgan_net, weight, device, use_fp16):
                        gfpgan_net = gfpgan_net.eval()
                        with torch.no_grad(): _test = gfpgan_net(torch.randn(1,3,512,512, device=next(gfpgan_net.parameters()).device), return_rgb=True); _returns_tuple = isinstance(_test, (tuple,list))
                        class _W(torch.nn.Module):
                            def __init__(self): super().__init__(); self.net = gfpgan_net
                            def forward(self, x): out = self.net(x, return_rgb=False); return out[0] if _returns_tuple else out
                        return _W().to(device)
                    def _export_onnx(self, wrapper, onnx_path, max_batch_size):
                        wrapper.eval(); dummy = torch.randn(max_batch_size,3,512,512, device=self.device)
                        if self.use_fp16: dummy = dummy.half(); wrapper = wrapper.half()
                        try:
                            with torch.no_grad(): torch.onnx.export(wrapper, dummy, onnx_path, input_names=['input'], output_names=['output'], opset_version=18, dynamo=False)
                            return True
                        except: return False
                    def _build_engine_dynamic(self, onnx_path, trt_path, max_batch_size, use_fp16):
                        trt = self._trt
                        logger = trt.Logger(trt.Logger.ERROR)
                        builder = trt.Builder(logger)
                        try: flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                        except: flag = 0
                        network = builder.create_network(flag); parser = trt.OnnxParser(network, logger)
                        if not parser.parse_from_file(onnx_path): del parser, network, builder; return
                        config = builder.create_builder_config(); config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4*(1<<30))
                        if use_fp16 and builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16)
                        profile = builder.create_optimization_profile(); profile.set_shape('input', min=(max_batch_size,3,512,512), opt=(max_batch_size,3,512,512), max=(max_batch_size,3,512,512)); config.add_optimization_profile(profile)
                        serialized = builder.build_serialized_network(network, config); del config, profile, parser, network, builder; gc.collect()
                        if serialized is None: return
                        with open(trt_path, 'wb') as f: f.write(serialized); del serialized
                    def _load_engine(self, trt_path):
                        _cur = ''; import re as _re
                        if torch.cuda.is_available():
                            _pp = torch.cuda.get_device_properties(0); _cur = f'_sm{_pp.major}{_pp.minor}_{_re.sub(r"[^a-z0-9]","",_pp.name.lower())[:16]}'
                        if _cur and _cur not in osp.basename(trt_path):
                            try: os.remove(trt_path)
                            except: pass
                            raise RuntimeError('过期缓存已删除')
                        trt = self._trt
                        logger = trt.Logger(trt.Logger.ERROR)
                        runtime = trt.Runtime(logger)
                        with open(trt_path, 'rb') as f: self._engine = runtime.deserialize_cuda_engine(f.read()); del runtime
                        if self._engine is None: raise RuntimeError('deserialize returned None')
                        self._context = self._engine.create_execution_context()
                        if self._context is None: raise RuntimeError('create_execution_context returned None')
                        self._use_new_api = hasattr(self._engine, 'num_io_tensors')
                        if self._use_new_api:
                            for i in range(self._engine.num_io_tensors):
                                name = self._engine.get_tensor_name(i); mode = self._engine.get_tensor_mode(name)
                                if mode == trt.TensorIOMode.INPUT: self._input_name = name
                                elif mode == trt.TensorIOMode.OUTPUT: self._output_name = name
                            print(f'[GFPGAN-TRT] TRT 10.x | in={self._input_name}, out={self._output_name}', flush=True)
                        else: print('[GFPGAN-TRT] TRT 8.x API', flush=True)
                        self._trt_ok = True
                    @property
                    def available(self): return self._trt_ok
                    def infer(self, face_tensor):
                        if self._trt_stream is None: self._trt_stream = torch.cuda.Stream(device=self.device)
                        B, max_bs, dtype = face_tensor.shape[0], self._max_batch_size, torch.float16 if self.use_fp16 else torch.float32
                        _FULL = (max_bs, 3, 512, 512)
                        if not hasattr(self, '_inp_buf') or self._inp_buf.shape != torch.Size(_FULL) or self._inp_buf.dtype != dtype:
                            self._inp_buf = torch.empty(_FULL, dtype=dtype, device=self.device); self._out_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
                        self._inp_buf.zero_(); self._inp_buf[:B].copy_(face_tensor[:B])
                        self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
                        if self._use_new_api:
                            self._context.set_input_shape(self._input_name, _FULL)
                            self._context.set_tensor_address(self._input_name, self._inp_buf.data_ptr())
                            self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
                            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
                        else:
                            self._context.set_binding_shape(0, _FULL)
                            self._context.execute_async_v2(bindings=[self._inp_buf.data_ptr(), self._out_buf.data_ptr()], stream_handle=self._trt_stream.cuda_stream)
                        self._trt_stream.synchronize()
                        return self._out_buf[:B].clone()
                gfpgan_trt_accel = GFPGANTRTAccelerator(face_enhancer=face_enhancer, device=device, cache_dir=trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'), gfpgan_weight=gfpgan_weight, max_batch_size=gfpgan_batch_size, gfpgan_version=gfpgan_model, use_fp16=use_fp16)
                if gfpgan_trt_accel.available: print('[GFPGANSubprocess] TRT 加速已启用（子进程版本）', flush=True)
                else: print('[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch', flush=True); use_trt = False
            except Exception as e: print(f'[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch: {e}', flush=True); use_trt = False; gfpgan_trt_accel = None
        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            print('[GFPGANSubprocess] TRT warmup 失败导致 CUDA context 损坏', flush=True); time.sleep(0.5); os._exit(0)
        _model_needs_gpu = False
        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available: print('[GFPGANSubprocess] TRT warmup 通过，迁移 GFPGANer 到 GPU...', flush=True); _model_needs_gpu = True
        elif use_trt and not (gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False)): print('[GFPGANSubprocess] TRT 失败，迁移 GFPGANer 到 GPU...', flush=True); _model_needs_gpu = True
        if _model_needs_gpu:
            face_enhancer.gfpgan = face_enhancer.gfpgan.to(device); model = face_enhancer.gfpgan
            if use_fp16: face_enhancer.gfpgan = face_enhancer.gfpgan.half(); model = face_enhancer.gfpgan
            face_enhancer.face_helper = None; print('[GFPGANSubprocess] GFPGANer 已迁移到 GPU', flush=True)
        fp16_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()
        if ready_event is not None: ready_event.set()
        _shm_inputs, _shm_outputs, _shm_available = [], [], False
        if shm_input_names and shm_output_names:
            try:
                for _sname in shm_input_names: _shm_inputs.append(shm.SharedMemory(name=_sname))
                for _sname in shm_output_names: _shm_outputs.append(shm.SharedMemory(name=_sname))
                _shm_available = True; print(f'[GFPGANSubprocess] 共享内存 attach 成功 ({len(_shm_inputs)} input slots + {len(_shm_outputs)} output slots)', flush=True)
            except Exception as _shm_e: print(f'[GFPGANSubprocess] 共享内存 attach 失败，回退 pickle: {_shm_e}', flush=True)
        while True:
            try: task = task_queue.get(timeout=1)
            except queue.Empty: continue
            if task is None: break
            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__validate__':
                _val_id = task[1]
                if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] post-SR warmup（真实显存压力下）...', flush=True)
                    try:
                        torch.cuda.empty_cache(); _vdt = torch.float16 if use_fp16 else torch.float32; _vdmy = torch.zeros(gfpgan_batch_size, 3, 512, 512, dtype=_vdt, device=device); _vout = gfpgan_trt_accel.infer(_vdmy); del _vdmy, _vout; torch.cuda.synchronize(device)
                        print('[GFPGANSubprocess] post-SR warmup 通过，TRT 推理正式启用', flush=True); result_queue.put(('__validate__', _val_id, True), timeout=5.0)
                    except Exception as _ve:
                        _ve_str = str(_ve).lower()
                        if any(kw in _ve_str for kw in ('illegal memory', 'cudaerrorillegaladdress', 'illegal instruction', 'prior launch failure', 'acceleratorerror', 'cudaerror')): result_queue.put(('__validate__', _val_id, False), timeout=5.0); os._exit(0)
                        else: result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                else: result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                continue
            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__pause__': torch.cuda.empty_cache(); time.sleep(task[1]); torch.cuda.empty_cache(); continue
            _use_shm_output, _shm_slot_id = False, -1
            if isinstance(task, tuple) and len(task) == 3 and isinstance(task[1], int) and isinstance(task[2], int):
                task_id, _n_faces, _shm_slot_id = task
                if _shm_available and _shm_slot_id < len(_shm_inputs):
                    _inp_buf = np.ndarray((64, 512, 512, 3), dtype=np.uint8, buffer=_shm_inputs[_shm_slot_id].buf)
                    crops_np = [_inp_buf[i].copy() for i in range(_n_faces)]; _use_shm_output = True
                else: print(f'[GFPGANSubprocess] 共享内存 slot {_shm_slot_id} 不可用，跳过', flush=True); result_queue.put((task_id, []), timeout=5.0); continue
            else: task_id, crops_np = task
            crops_tensor = []; 
            for crop in crops_np:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True); _tv_normalize(t, (0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True); crops_tensor.append(t)
            if not crops_tensor: result_queue.put((task_id, []), timeout=5.0); continue
            all_out, sub_bs, i = [], gfpgan_batch_size, 0
            while i < len(crops_tensor):
                sub = crops_tensor[i:i+sub_bs]; sub_batch = torch.stack(sub).to(device)
                try:
                    with torch.no_grad():
                        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                            if use_fp16: sub_batch = sub_batch.half()
                            out = gfpgan_trt_accel.infer(sub_batch).float()
                            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                            out = torch.clamp(out, min=-1.0, max=1.0)
                            if abs(gfpgan_weight - 1.0) > 1e-6: out = gfpgan_weight * out + (1.0 - gfpgan_weight) * sub_batch.float(); out = torch.clamp(out, min=-1.0, max=1.0)
                        else:
                            with fp16_ctx: out = model(sub_batch, return_rgb=False)
                            if isinstance(out, (tuple, list)): out = out[0]; out = out.float()
                            if abs(gfpgan_weight - 1.0) > 1e-6: out = gfpgan_weight * out + (1.0 - gfpgan_weight) * sub_batch.float()
                        all_out.extend(out.unbind(0)); i += len(sub)
                except RuntimeError as e:
                    _estr = str(e).lower()
                    if 'out of memory' in _estr and sub_bs > 1: sub_bs = max(1, sub_bs//2); torch.cuda.empty_cache(); print(f'[GFPGANSubprocess] GFPGAN OOM，sub_bs 降级至 {sub_bs}，重试...', flush=True)
                    elif 'cudaerrorillegaladdress' in _estr or 'illegal memory' in _estr: print(f'[GFPGANSubprocess] CUDA 非法内存访问，切换到 PyTorch 路径: {e}'); use_trt = False; gfpgan_trt_accel = None; torch.cuda.empty_cache()
                    else: all_out.extend([None]*len(sub)); i += len(sub); torch.cuda.empty_cache()
                finally: del sub_batch
            restored = []
            for out_t in all_out:
                if out_t is None: restored.append(None)
                else:
                    out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=-1.0); out_t = torch.clamp(out_t, min=-1.0, max=1.0)
                    restored.append(tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1)).astype('uint8'))
            if _use_shm_output and _shm_slot_id >= 0 and _shm_slot_id < len(_shm_outputs):
                _out_buf = np.ndarray((64, 512, 512, 3), dtype=np.uint8, buffer=_shm_outputs[_shm_slot_id].buf)
                for i, r in enumerate(restored):
                    if r is not None and i < 64:
                        if r.shape == (512, 512, 3): _out_buf[i] = r
                        else:
                            import cv2 as _cv_w; _out_buf[i] = _cv_w.resize(r, (512, 512))
                    elif i < 64: _out_buf[i] = 0
                try: result_queue.put((task_id, len(restored)), timeout=5.0)
                except: pass
            else:
                try: result_queue.put((task_id, restored), timeout=5.0)
                except: pass
        for _s in _shm_inputs + _shm_outputs:
            try: _s.close()
            except: pass
        if gfpgan_trt_accel is not None: gfpgan_trt_accel._trt_ok = False
        try: del face_enhancer, model
        except: pass
        torch.cuda.empty_cache(); os._exit(0)

    def close(self):
        if self.process and self.process.is_alive():
            try: self.task_queue.put(None, timeout=3)
            except: pass
            self.process.join(timeout=15)
            if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=5)
            if self.process.is_alive(): print('[GFPGANSubprocess] 子进程未响应 SIGTERM，发送 SIGKILL...', flush=True); self.process.kill(); self.process.join(timeout=5)
        if self.shm_buf is not None: self.shm_buf.close(); self.shm_buf = None
        try: self.task_queue.close()
        except: pass
        try: self.result_queue.close()
        except: pass