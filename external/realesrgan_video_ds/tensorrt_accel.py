#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - TensorRT 加速模块 (SR)
"""

import os
import sys
import gc
import threading
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import models_RealESRGAN

_TRT_LOGGER = None


def _get_trt_logger():
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        try:
            import tensorrt as _trt_mod
            _TRT_LOGGER = _trt_mod.Logger(_trt_mod.Logger.ERROR)
        except ImportError:
            pass
    return _TRT_LOGGER


class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 cache_dir: str, input_shape: Tuple[int, int, int, int],
                 use_fp16: bool = True):
        self.device = device
        self.input_shape = input_shape
        self.use_fp16 = use_fp16
        self._engine = None
        self._context = None
        self._trt_ok = False
        self._trt_stream: Optional[torch.cuda.Stream] = None

        try:
            import tensorrt as trt
            self._trt = trt
        except ImportError as e:
            print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}')
            print('  安装命令: pip install tensorrt onnx onnxruntime-gpu')
            return

        _sm_tag = ''
        if torch.cuda.is_available():
            _p = torch.cuda.get_device_properties(0)
            import re as _re_sr
            _gpu_slug_sr = _re_sr.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
            _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_sr}'

        B, C, H, W = input_shape
        tag = f'B{B}_C{C}_H{H}_W{W}_fp{"16" if use_fp16 else "32"}{_sm_tag}'
        trt_path = os.path.join(cache_dir, f'realesrgan_{tag}.trt')
        onnx_path = os.path.join(cache_dir, f'realesrgan_{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}, tag={tag}) ...')
            self._export_onnx(model, onnx_path, input_shape)
            self._build_engine(onnx_path, trt_path, use_fp16)

        if os.path.exists(trt_path):
            try:
                self._load_engine(trt_path)
            except RuntimeError as _e:
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not os.path.exists(onnx_path):
                    self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if os.path.exists(trt_path):
                    self._load_engine(trt_path)

    def _export_onnx(self, model, onnx_path, input_shape):
        model.eval()
        dummy = torch.randn(*input_shape, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()
            model = model.half()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=18,
                dynamic_axes=None,
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)
        try:
            explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            explicit_batch_flag = 0
        network = builder.create_network(explicit_batch_flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(f'  [TensorRT] ONNX 解析错误: {parser.get_error(i)}')
            del parser, network, builder
            return
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
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
        del config, parser, network, builder
        gc.collect()
        if serialized is None:
            _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
            _sm_hint = (f'\n[TensorRT] 提示: {_gpu_name} ({_sm_str}) 可能不受此 TRT 版本支持'
                        if _sm_major < 8 else '')
            print(f'[TensorRT] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}')
            return
        with open(trt_path, 'wb') as f:
            f.write(serialized)
        del serialized
        print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')

    def _load_engine(self, trt_path):
        _cur_sm_tag = ''
        if torch.cuda.is_available():
            _pp = torch.cuda.get_device_properties(0)
            import re as _re
            _gpu_slug = _re.sub(r'[^a-z0-9]', '', _pp.name.lower())[:16]
            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{_gpu_slug}'
        if _cur_sm_tag:
            _basename = os.path.basename(trt_path)
            if _cur_sm_tag not in _basename:
                print(f'[TensorRT] .trt 文件名不含当前 GPU SM tag {_cur_sm_tag}，'
                      f'可能是旧版本缓存或跨 GPU 遗留文件: {_basename}')
                print(f'[TensorRT] 删除过期缓存，触发针对当前 GPU 的重建')
                try:
                    os.remove(trt_path)
                except OSError:
                    pass
                raise RuntimeError(f'[TensorRT] 过期缓存 {_basename} 已删除，需重建')
        trt = self._trt
        logger = _get_trt_logger()
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        del runtime
        if self._engine is None:
            print(f'[TensorRT] Engine 反序列化失败，删除过期缓存并重新构建: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[TensorRT] _load_engine: deserialize_cuda_engine returned None')

        # [FIX-TRT-CTX-OOM] create_execution_context() 在 GPU 显存不足时
        # 返回 None 而非抛出 Python 异常（与 deserialize_cuda_engine 行为一致）。
        # 典型场景：interpolate_then_upscale 模式下，前序 IFRNet 步骤的
        # PyTorch 缓存分配器残留大量显存，导致 TRT 无法分配 context 所需的
        # 激活内存（通常为数 GB 量级）。
        # 若不检测，后续 infer() 中 self._context.set_tensor_address() /
        # execute_async_v3() 会在 NoneType 上调用 → AttributeError 崩溃。
        self._context = self._engine.create_execution_context()
        if self._context is None:
            print('[TensorRT] ⚠️  create_execution_context() 失败'
                  '（GPU 显存不足），回退 PyTorch 推理路径。')
            print('[TensorRT] 提示: 前序处理步骤可能占用了大量显存。'
                  '可尝试减小 --batch-size 或移除 --use-tensorrt。')
            # 释放已加载的 engine，归还显存
            self._engine = None
            self._trt_ok = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return  # 不抛异常，__init__ 中 self._trt_ok 保持 False，自动走 PyTorch 路径

        # ── 区分 TRT 版本，预先解析 tensor 名称 / binding 信息 ──────────────────
        # TRT 10.x: 使用 num_io_tensors + get_tensor_name + get_tensor_mode
        # TRT 8.x : 使用 num_bindings + get_binding_shape（旧接口）
        self._use_new_api = hasattr(self._engine, 'num_io_tensors')
        self._input_name = None
        self._output_name = None
        trt = self._trt
        if self._use_new_api:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self._output_name = name
            if self._input_name is None or self._output_name is None:
                raise RuntimeError(
                    '[TensorRT] 无法在 Engine 中找到有效输入/输出 tensor')
            print(f'[TensorRT] 使用新版 API (TRT 10.x)，'
                  f'输入: {self._input_name}，输出: {self._output_name}')
        else:
            print('[TensorRT] 使用旧版 API (TRT 8.x)')
        self._trt_ok = True
        print('[TensorRT] Engine 加载成功，已启用 TRT 推理')

    @property
    def available(self) -> bool:
        return self._trt_ok

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        actual_B = input_tensor.shape[0]
        engine_B = self.input_shape[0]
        if actual_B < engine_B:
            pad_cnt = engine_B - actual_B
            pad = input_tensor[-1:].expand(pad_cnt, -1, -1, -1)
            input_tensor = torch.cat([input_tensor, pad], dim=0)
        inp = input_tensor.contiguous()
        out_dtype = torch.float16 if self.use_fp16 else torch.float32
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)
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
            self._context.execute_async_v2(
                bindings=[inp.data_ptr(), out_tensor.data_ptr()],
                stream_handle=self._trt_stream.cuda_stream,
            )
        self._trt_stream.synchronize()
        if actual_B < engine_B:
            out_tensor = out_tensor[:actual_B]
        return out_tensor