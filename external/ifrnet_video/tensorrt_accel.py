#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - TensorRT 加速模块（Engine 构建/缓存 + TRT 推理分支）。
# 镜像 external/realesrgan_video/tensorrt_accel.py 的职责。

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Tuple

import torch


class TensorRTAccelMixin:
    """IFRNet TRT Engine 构建/缓存与 TRT 推理分支（与 IFRNetVideoProcessor 混合）。

    自 v6.4.5.1 单文件 M4: TensorRT 小节逐字提取。
    """

    # ── M4: TensorRT ─────────────────────────────────────────────────────────

    def _build_trt_engine(self, input_shape: Tuple[int, int, int, int], cache_dir: str,
                          _rebuild_attempt: bool = False):
        try:
            import tensorrt as trt
        except ImportError:
            print('[TensorRT] 未安装，跳过 TRT 加速。')
            self.use_tensorrt = False
            return

        os.makedirs(cache_dir, exist_ok=True)
        B, C, H, W = input_shape

        _sm_tag = ''
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            import re as _re_sm
            _gpu_slug = _re_sm.sub(r'[^a-z0-9]', '', _props.name.lower())[:16]
            _sm_tag = f'_sm{_props.major}{_props.minor}_{_gpu_slug}'

        # ✅ 加入模型变体，避免跨模型加载错误 Engine
        tag       = f'{self.model_name}_B{B}_H{H}_W{W}_fp{"16" if self.use_fp16 else "32"}{_sm_tag}'
        trt_path  = os.path.join(cache_dir, f'{tag}.trt')
        onnx_path = os.path.join(cache_dir, f'{tag}.onnx')

        if os.path.exists(trt_path):
            if _sm_tag and _sm_tag not in os.path.basename(trt_path):
                print(f'[TensorRT] 缓存文件缺少当前 GPU 标记 {_sm_tag}，删除并重建: {trt_path}')
                try: os.remove(trt_path)
                except OSError: pass
                if os.path.exists(onnx_path):
                    try: os.remove(onnx_path)
                    except OSError: pass

        if not os.path.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
            dummy0 = torch.randn(*input_shape, device=self.device)
            dummy1 = torch.randn(*input_shape, device=self.device)
            embt   = torch.full((B,), 0.5, dtype=torch.float32,
                                device=self.device).view(B, 1, 1, 1)
            if self.use_fp16:
                dummy0, dummy1, embt = dummy0.half(), dummy1.half(), embt.half()

            _base_model = getattr(self.model, '_orig_mod', self.model)

            class _InferenceWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, img0, img1, embt):
                    return self.m.inference(img0, img1, embt)

            export_model = _InferenceWrapper(_base_model)
            with torch.no_grad():
                torch.onnx.export(
                    export_model, (dummy0, dummy1, embt), onnx_path,
                    input_names=['img0', 'img1', 'embt'],
                    output_names=['output'],
                    opset_version=18,
                    dynamic_axes=None,
                )
            import onnx
            model_proto = onnx.load(onnx_path)
            onnx.save(model_proto, onnx_path,
                      save_as_external_data=False, all_tensors_to_one_file=False)
            print(f'[TensorRT] ONNX 已导出: {onnx_path}')

            if not hasattr(self, '_trt_logger'):
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger  = self._trt_logger
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f'  [TRT ONNX Error] {parser.get_error(i)}')
                    self.use_tensorrt = False
                    return

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
            if self.use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'
            _sm_code  = _props.major * 10 + _props.minor if torch.cuda.is_available() else 0
            _time_hint = {
                75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
                80: '约需 10~20 分钟（A100/A30 SM8.0）',
                86: '约需 5~15 分钟（A10/RTX30系 SM8.6）',
                89: '约需 5~10 分钟（RTX40系 SM8.9）',
                90: '约需 3~8 分钟（H100 SM9.0）',
            }.get(_sm_code, f'约需 5~20 分钟（{_gpu_name}）')
            print(f'[TensorRT] {_time_hint}')

            _build_start = time.time()
            _build_done  = threading.Event()

            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= 300:
                        elapsed = time.time() - _build_start
                        print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行）', flush=True)
                        _last = time.time()

            _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()

            serialized = builder.build_serialized_network(network, config)
            _build_done.set()
            _build_elapsed = time.time() - _build_start
            del config, parser, network, builder
            import gc; gc.collect()

            if serialized is None:
                print('[TensorRT] Engine 构建失败，回退 PyTorch 路径。')
                self.use_tensorrt = False
                return

            with open(trt_path, 'wb') as f:
                f.write(serialized)
            print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')

        # 加载 Engine
        try:
            if not hasattr(self, '_trt_logger'):
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger  = self._trt_logger
            runtime = trt.Runtime(logger)
            with open(trt_path, 'rb') as f:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())

            if self._trt_engine is None:
                if _rebuild_attempt:
                    print('[TensorRT] ⚠️  重建后 Engine 仍反序列化失败，回退 PyTorch。')
                    self.use_tensorrt = False
                    self._trt_ok = False
                    return
                print(f'[TensorRT] Engine 反序列化失败，删除并重建: {trt_path}')
                try: os.remove(trt_path)
                except OSError: pass
                if os.path.exists(onnx_path):
                    try: os.remove(onnx_path)
                    except OSError: pass
                return self._build_trt_engine(input_shape, cache_dir, _rebuild_attempt=True)

            self._trt_context = self._trt_engine.create_execution_context()
            if self._trt_context is None:
                print('[TensorRT] ⚠️  create_execution_context() 失败（显存不足），回退 PyTorch。')
                self._trt_engine  = None
                self.use_tensorrt = False
                self._trt_ok      = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

            n = self._trt_engine.num_io_tensors
            inputs, outputs = [], []
            for i in range(n):
                name = self._trt_engine.get_tensor_name(i)
                mode = self._trt_engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            self._trt_input_names  = inputs
            self._trt_output_names = outputs
            if not self.quiet:
                print(f'[TensorRT] inputs={inputs} outputs={outputs}')
            self._trt_ok = True
            print('[TensorRT] Engine 已激活，TRT 推理就绪。')
        except Exception as e:
            print(f'[TensorRT] Engine 加载失败: {e}，回退 PyTorch。')
            try: os.remove(trt_path)
            except OSError: pass
            self.use_tensorrt = False
            self._trt_ok = False

    def _infer_batch_trt(self, img0_exp, img1_exp, timesteps, B):
        """TRT 推理分支（自 _infer_batch 的 elif 分支逐字提取，末尾补 return）。"""
        import tensorrt as _trt2
        in_names  = getattr(self, '_trt_input_names',  ['img0', 'img1', 'embt'])
        out_names = getattr(self, '_trt_output_names', ['output'])
        engine_BT = self._trt_engine.get_tensor_shape(in_names[0])[0]
        BT        = img0_exp.shape[0]
        out_dtype = torch.float16 if self.use_fp16 else torch.float32
        out_shape = tuple(self._trt_engine.get_tensor_shape(out_names[0]))
        out_buf   = torch.empty(out_shape, dtype=out_dtype, device=self.device)

        _trt_stream_ctx = (torch.cuda.stream(self.stream_compute)
                           if self.stream_compute is not None else nullcontext())
        with _trt_stream_ctx:
            t_vals = timesteps * B
            embt_t = torch.tensor(t_vals, dtype=torch.float32,
                                  device=self.device).view(-1, 1, 1, 1)
            i0 = img0_exp.half().contiguous() if self.use_fp16 else img0_exp.float().contiguous()
            i1 = img1_exp.half().contiguous() if self.use_fp16 else img1_exp.float().contiguous()
            em = embt_t.half().contiguous() if self.use_fp16 else embt_t.contiguous()
            if BT < engine_BT:
                pad_n = engine_BT - BT
                def _pad(t):
                    return torch.cat([t, t[-1:].expand(pad_n, *t.shape[1:])], 0).contiguous()
                i0, i1, em = _pad(i0), _pad(i1), _pad(em)
            ctx = self._trt_context
            for name, buf in zip(in_names, [i0, i1, em]):
                ctx.set_tensor_address(name, buf.data_ptr())
            ctx.set_tensor_address(out_names[0], out_buf.data_ptr())
            _dummy_bufs = []
            for _out_name in out_names[1:]:
                _shape  = tuple(self._trt_engine.get_tensor_shape(_out_name))
                _dtype  = self._trt_engine.get_tensor_dtype(_out_name)
                _tdtype = torch.float16 if _dtype == _trt2.DataType.HALF else torch.float32
                _dummy  = torch.empty(_shape, dtype=_tdtype, device=self.device)
                ctx.set_tensor_address(_out_name, _dummy.data_ptr())
                _dummy_bufs.append(_dummy)
            _trt_stream_handle = (self.stream_compute.cuda_stream
                                  if self.stream_compute is not None
                                  else torch.cuda.current_stream().cuda_stream)
            ctx.execute_async_v3(stream_handle=_trt_stream_handle)

        # [FIX-PREFETCH-TIMING] TRT kernel 已入 stream_compute，立即预取下批
        if self._pipeline_runner is not None:
            self._pipeline_runner._try_prefetch_next()

        # [STREAM-DUAL] 不再 wait default stream；
        # stream_d2h 将在 PINNED-D2H 路径内直接 wait stream_compute。
        # 保持原始类型（FP16），float() 转换移入 stream_d2h。
        result_buf = out_buf[:BT]
        pred_big   = result_buf
        return pred_big
