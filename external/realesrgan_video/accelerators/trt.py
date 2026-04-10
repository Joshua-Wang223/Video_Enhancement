#!/usr/bin/env python3
import os
import re
import gc
import threading
import time
import torch
from typing import Tuple, Optional

_TRT_LOGGER = None
def _get_trt_logger():
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        try:
            import tensorrt as _trt_mod; _TRT_LOGGER = _trt_mod.Logger(_trt_mod.Logger.ERROR)
        except ImportError: pass
    return _TRT_LOGGER

class TensorRTAccelerator:
    def __init__(self, model: torch.nn.Module, device: torch.device, cache_dir: str, input_shape: Tuple[int, int, int, int], use_fp16: bool = True):
        self.device, self.input_shape, self.use_fp16 = device, input_shape, use_fp16
        self._engine, self._context, self._trt_ok, self._trt_stream = None, None, False, None
        try: import tensorrt as trt; self._trt = trt
        except ImportError as e: print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}'); return
        _sm_tag = ''
        if torch.cuda.is_available():
            _p = torch.cuda.get_device_properties(0)
            _sm_tag = f'_sm{_p.major}{_p.minor}_{re.sub(r"[^a-z0-9]", "", _p.name.lower())[:16]}'
        B, C, H, W = input_shape
        tag = f'B{B}_C{C}_H{H}_W{W}_fp{"16" if use_fp16 else "32"}{_sm_tag}'
        trt_path = os.path.join(cache_dir, f'realesrgan_{tag}.trt')
        onnx_path = os.path.join(cache_dir, f'realesrgan_{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(trt_path): print(f'[TensorRT] 构建 Engine (shape={input_shape}, tag={tag}) ...'); self._export_onnx(model, onnx_path, input_shape); self._build_engine(onnx_path, trt_path, use_fp16)
        if os.path.exists(trt_path):
            try: self._load_engine(trt_path)
            except RuntimeError as _e:
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not os.path.exists(onnx_path): self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if os.path.exists(trt_path): self._load_engine(trt_path)

    def _export_onnx(self, model, onnx_path, input_shape):
        model.eval(); dummy = torch.randn(*input_shape, device=self.device)
        if self.use_fp16: dummy = dummy.half(); model = model.half()
        with torch.no_grad(): torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'], opset_version=18, dynamic_axes=None)
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)
        try: explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError: explicit_batch_flag = 0
        network = builder.create_network(explicit_batch_flag); parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors): print(f'  [TensorRT] ONNX 解析错误: {parser.get_error(i)}'); return
        config = builder.create_builder_config(); config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        if use_fp16 and builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16)
        
        # 添加 SM 检测和时间预估
        _gpu_name = 'unknown'
        _sm_major = 0
        _sm_minor = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                print(f'[TensorRT] 警告: {_gpu_name} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）')
        _sm_code = _sm_major * 10 + _sm_minor
        _time_hint = {
            75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
            80: '约需 8~15 分钟（A100/A30 SM8.0）',
            86: '约需 10~18 分钟（A10/RTX30系 SM8.6）',
            89: '约需 8~12 分钟（RTX40系 SM8.9）',
            90: '约需 5~10 分钟（H100 SM9.0）',
        }.get(_sm_code, f'约需 10~30 分钟（{_gpu_name} SM{_sm_major}.{_sm_minor}）')
        print(f'[TensorRT] {_time_hint}')
        
        # 添加心跳线程，每300秒报告一次状态
        _build_start = time.time()
        _build_done = threading.Event()
        def _heartbeat():
            _last = time.time()
            while not _build_done.wait(timeout=5):
                if time.time() - _last >= 300:
                    elapsed = time.time() - _build_start
                    print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）')
                    _last = time.time()
        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()
        
        serialized = builder.build_serialized_network(network, config)
        _build_done.set()
        _build_elapsed = time.time() - _build_start
        del config, parser, network, builder; gc.collect()
        if serialized is None:
            _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
            _sm_hint = (f'\n[TensorRT] 提示: {_gpu_name} ({_sm_str}) 可能不受此 TRT 版本支持'
                        if _sm_major < 8 else '')
            print(f'[TensorRT] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}')
            return
        with open(trt_path, 'wb') as f: f.write(serialized); del serialized
        print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')

    def _load_engine(self, trt_path):
        _cur_sm_tag = ''
        if torch.cuda.is_available():
            _pp = torch.cuda.get_device_properties(0)
            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{re.sub(r"[^a-z0-9]", "", _pp.name.lower())[:16]}'
        if _cur_sm_tag and _cur_sm_tag not in os.path.basename(trt_path):
            print(f'[TensorRT] .trt 文件名不含当前 GPU SM tag {_cur_sm_tag}'); print(f'[TensorRT] 删除过期缓存，触发针对当前 GPU 的重建')
            try: os.remove(trt_path)
            except OSError: pass
            raise RuntimeError(f'[TensorRT] 过期缓存已删除，需重建')
        trt = self._trt
        logger = _get_trt_logger()
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f: self._engine = runtime.deserialize_cuda_engine(f.read())
        del runtime
        if self._engine is None: raise RuntimeError('[TensorRT] _load_engine: deserialize returned None')
        self._context = self._engine.create_execution_context()
        self._use_new_api = hasattr(self._engine, 'num_io_tensors')
        self._input_name, self._output_name = None, None
        if self._use_new_api:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i); mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT: self._input_name = name
                elif mode == trt.TensorIOMode.OUTPUT: self._output_name = name
            print(f'[TensorRT] 使用新版 API (TRT 10.x)，输入: {self._input_name}，输出: {self._output_name}')
        else: print('[TensorRT] 使用旧版 API (TRT 8.x)')
        self._trt_ok = True; print('[TensorRT] Engine 加载成功，已启用 TRT 推理')

    @property
    def available(self) -> bool: return self._trt_ok

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        actual_B = input_tensor.shape[0]; engine_B = self.input_shape[0]
        if actual_B < engine_B: input_tensor = torch.cat([input_tensor, input_tensor[-1:].expand(engine_B - actual_B, -1, -1, -1)], dim=0)
        inp = input_tensor.contiguous(); out_dtype = torch.float16 if self.use_fp16 else torch.float32
        if self._trt_stream is None: self._trt_stream = torch.cuda.Stream(device=self.device)
        if self._use_new_api:
            out_shape = tuple(self._engine.get_tensor_shape(self._output_name))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.set_tensor_address(self._input_name, inp.data_ptr())
            self._context.set_tensor_address(self._output_name, out_tensor.data_ptr())
            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        else:
            out_shape = tuple(self._engine.get_binding_shape(1))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.execute_async_v2(bindings=[inp.data_ptr(), out_tensor.data_ptr()], stream_handle=self._trt_stream.cuda_stream)
        self._trt_stream.synchronize()
        return out_tensor[:actual_B] if actual_B < engine_B else out_tensor