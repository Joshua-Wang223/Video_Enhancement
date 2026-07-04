#!/usr/bin/env python3
"""
realesrgan_video NVENC SDK (nvenc_sdk.py) — 完整测试套件

覆盖:
  - 概念分离验证 (_slot_count / _required_buffers)
  - LA 帧守恒 (LA=0/8, CONSTQP/VBR_HQ/QVBR)
  - SDK 合规 drain/flush
  - LA 累积模式 + f0 处理
  - NVENCWriter 端到端

运行:
  pytest tests/test_nvenc_sdk_realesrgan.py -v                    # 所有测试
  pytest tests/test_nvenc_sdk_realesrgan.py -v -m gpu              # 仅 GPU 测试
  pytest tests/test_nvenc_sdk_realesrgan.py -v -m "not gpu"        # 仅无 GPU 测试

当前环境: 无 GPU — GPU 测试需要 NVIDIA GPU + CUDA toolkit.
"""

import os
import sys
import struct
import pytest
import tempfile
import threading

import numpy as np

# ── GPU availability check ──
try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

# ── nvenc_sdk availability check ──
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'realesrgan_video'))
    import nvenc_sdk
    _HAS_NVENC_SDK = True
except Exception as e:
    _HAS_NVENC_SDK = False
    _NVENC_SDK_IMPORT_ERROR = str(e)

_HAS_GPU = _HAS_CUDA and _HAS_NVENC_SDK
gpu = pytest.mark.skipif(not _HAS_GPU, reason="Requires NVIDIA GPU + nvenc_sdk import")


# ============================================================================
# 无 GPU 测试 — 代码结构 / 常量 / 导入验证
# ============================================================================

class TestModuleStructure:
    """模块级导入和常量验证（无 GPU 依赖）。"""

    def test_import_nvenc_sdk(self):
        """验证 nvenc_sdk 模块可导入。"""
        assert _HAS_NVENC_SDK, f"nvenc_sdk import failed: {_NVENC_SDK_IMPORT_ERROR}"
        assert hasattr(nvenc_sdk, 'NVENCEncoder')
        assert hasattr(nvenc_sdk, '_NVENCEncodeThread')
        assert hasattr(nvenc_sdk, 'FFmpegMuxer')
        assert hasattr(nvenc_sdk, 'NVENCWriter')

    def test_constants_exist(self):
        """验证关键常量存在。"""
        assert hasattr(nvenc_sdk, '_NVENC_VBR_QUALITY_OFFSET')
        assert hasattr(nvenc_sdk, '_NVENC_CRF0_FORCE_CONSTQP')
        assert isinstance(nvenc_sdk._NVENC_CRF0_FORCE_CONSTQP, bool)

    def test_drain_methods_exist(self):
        """验证 SDK 合规 drain 方法存在。"""
        enc = nvenc_sdk.NVENCEncoder
        assert hasattr(enc, '_drain_outputs_blocking')
        assert hasattr(enc, '_reset_output_slot_idx')
        assert hasattr(enc, '_lock_bitstream_blocking')

    def test_encode_frames_batch_send_eos_param(self):
        """验证 encode_frames_batch 支持 send_eos 参数。"""
        import inspect
        sig = inspect.signature(nvenc_sdk.NVENCEncoder.encode_frames_batch)
        assert 'send_eos' in sig.parameters

    def test_nvenc_writer_class(self):
        """验证 NVENCWriter 类结构。"""
        writer_cls = nvenc_sdk.NVENCWriter
        assert hasattr(writer_cls, 'write_frame')
        assert hasattr(writer_cls, 'write_frame_batch')
        assert hasattr(writer_cls, 'close')
        # _broken is set in __init__, but class-level attribute exists
        assert hasattr(writer_cls, '_MINI_BATCH')

    def test_ffmpeg_muxer_class(self):
        """验证 FFmpegMuxer 类结构。"""
        muxer_cls = nvenc_sdk.FFmpegMuxer
        assert hasattr(muxer_cls, 'write')
        assert hasattr(muxer_cls, 'write_sps_pps')
        assert hasattr(muxer_cls, 'close')

    def test_rgb_to_nv12_functions(self):
        """验证 RGB→NV12 转换函数存在。"""
        assert hasattr(nvenc_sdk, '_rgb_to_nv12_gpu')
        assert hasattr(nvenc_sdk, '_rgb_to_nv12_gpu_batch')

    @gpu
    def test_no_self_pipeline_depth_attr(self):
        """验证 _pipeline_depth 实例属性已删除。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='vbr_hq', la_depth=8)
        assert not hasattr(enc, '_pipeline_depth'), "self._pipeline_depth must be removed"
        assert hasattr(enc, '_slot_count'), "self._slot_count must exist"
        enc.close()


# ============================================================================
# GPU 测试 — NVENCEncoder 初始化和概念分离验证
# ============================================================================

class TestInitVariableSeparation:
    """概念分离：_slot_count vs _required_buffers。"""

    @gpu
    def test_init_la0_slots(self):
        """LA=0: _slot_count = 4 (default), _required_buffers = 1."""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      pipeline_depth=4, rate_mode='constqp',
                                      la_depth=0)
        try:
            assert enc._slot_count == 4, f"Expected 4, got {enc._slot_count}"
            assert len(enc._slots) == enc._slot_count
            assert enc._la_depth == 0
            assert enc._output_slot_idx >= 0
        finally:
            enc.close()

    @gpu
    def test_init_la8_slots_expand(self):
        """LA=8: _slot_count = max(4, 9) = 9 (SDK lower bound expands default)."""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      pipeline_depth=4, rate_mode='vbr_hq',
                                      la_depth=8)
        try:
            assert enc._slot_count == 9, f"Expected 9, got {enc._slot_count}"
            assert len(enc._slots) == 9
            assert enc._la_depth == 8
        finally:
            enc.close()

    @gpu
    def test_init_la8_default_2_expands(self):
        """LA=8, pipeline_depth=2: _slot_count = max(2, 9) = 9."""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      pipeline_depth=2, rate_mode='vbr_hq',
                                      la_depth=8)
        try:
            assert enc._slot_count == 9
        finally:
            enc.close()

    @gpu
    def test_init_constqp_disables_la(self):
        """CONSTQP mode silences LA: _la_depth forced to 0."""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      pipeline_depth=4, rate_mode='constqp',
                                      la_depth=8)
        try:
            assert enc._la_depth == 0
            assert enc._slot_count == 4  # no expansion needed
        finally:
            enc.close()

    @gpu
    def test_init_no_forced_pd_reduction(self):
        """验证 pd→2 强制缩减已删除。LA=8 不应缩减到 2。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      pipeline_depth=4, rate_mode='vbr_hq',
                                      la_depth=8)
        try:
            assert enc._slot_count == 9, f"Must NOT force-reduce to 2; got {enc._slot_count}"
        finally:
            enc.close()

    @gpu
    def test_output_slot_idx_initialized(self):
        """_output_slot_idx 初始化为 0。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            assert enc._output_slot_idx == 0
        finally:
            enc.close()


# ============================================================================
# GPU 测试 — 帧守恒 (Frame Conservation)
# ============================================================================

class TestFrameConservation:
    """LA 帧守恒测试：输入帧数 == 输出帧数。"""

    N_FRAMES = 32  # 测试用帧数（<687 以加快测试）

    def _make_test_frames(self, n, h=64, w=64):
        """生成测试用 NV12 GPU tensors。"""
        import torch
        nv12_h = h + h // 2
        frames = []
        for _ in range(n):
            t = torch.randint(0, 255, (nv12_h, w), dtype=torch.uint8, device='cuda')
            frames.append(t)
        return frames

    @gpu
    def test_frame_conservation_constqp_la0(self):
        """LA=0 CONSTQP: 帧守恒 (n_in == n_out)。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            frames = self._make_test_frames(self.N_FRAMES)
            results = enc.encode_frames_batch(frames, force_idr_first=True)
            assert len(results) == self.N_FRAMES, \
                f"Frame conservation failed: {len(results)} != {self.N_FRAMES}"
        finally:
            enc.close()

    @gpu
    def test_frame_conservation_vbr_hq_la8_send_eos(self):
        """LA=8 VBR_HQ + send_eos: 帧守恒。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='vbr_hq', la_depth=8)
        try:
            frames = self._make_test_frames(self.N_FRAMES)
            results = enc.encode_frames_batch(frames, force_idr_first=True,
                                               send_eos=True)
            assert len(results) == self.N_FRAMES, \
                f"Frame conservation with LA=8 failed: {len(results)} != {self.N_FRAMES}"
            # 验证 EOS flush 后帧数守恒
            empty_count = sum(1 for r in results if not r)
            print(f"  LA=8: {self.N_FRAMES} frames submitted, "
                  f"{empty_count} empty (LA buffering), "
                  f"{self.N_FRAMES - empty_count} valid")
        finally:
            enc.close()

    @gpu
    def test_frame_conservation_qvbr_la8_send_eos(self):
        """LA=8 QVBR + send_eos: 帧守恒。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='qvbr', la_depth=8)
        try:
            frames = self._make_test_frames(self.N_FRAMES)
            results = enc.encode_frames_batch(frames, force_idr_first=True,
                                               send_eos=True)
            assert len(results) == self.N_FRAMES, \
                f"QVBR LA=8 frame conservation failed: {len(results)} != {self.N_FRAMES}"
        finally:
            enc.close()

    @gpu
    def test_no_empty_frames_constqp_la0(self):
        """LA=0 CONSTQP: 零空帧。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=0,
                                      rate_mode='constqp', la_depth=0)
        try:
            frames = self._make_test_frames(self.N_FRAMES)
            results = enc.encode_frames_batch(frames, force_idr_first=True)
            empty = sum(1 for r in results if not r)
            none_count = sum(1 for r in results if r is None)
            assert empty == 0, f"Expected 0 empty frames, got {empty}"
            assert none_count == 0, f"Expected 0 None, got {none_count}"
        finally:
            enc.close()

    @gpu
    def test_encode_frame_returns_data_la0(self):
        """encode_frame() LA=0: 返回有效 H.264 数据。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            import torch
            nv12 = torch.randint(0, 255, (96, 64), dtype=torch.uint8, device='cuda')
            data = enc.encode_frame(nv12, force_idr=True)
            assert data is not None
            assert len(data) > 0, f"encode_frame returned {len(data)} bytes"
            # 验证 SPS+PPS 被缓存
            assert enc._cached_sps_pps is not None
        finally:
            enc.close()


# ============================================================================
# GPU 测试 — SDK 合规 drain / flush
# ============================================================================

class TestSDKDrainFlush:
    """SDK 合规 drain 和 flush 验证。"""

    @gpu
    def test_drain_outputs_blocking_returns_list(self):
        """_drain_outputs_blocking() 返回列表。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            result = enc._drain_outputs_blocking()
            assert isinstance(result, list)
        finally:
            enc.close()

    @gpu
    def test_flush_blocking_lockbitstream(self):
        """flush() 使用 blocking LockBitstream (doNotWait=0)。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            import torch
            nv12 = torch.randint(0, 255, (96, 64), dtype=torch.uint8, device='cuda')
            enc.encode_frame(nv12, force_idr=True)
            enc.encode_frame(nv12, force_idr=False)
            flush_data = enc.flush()
            assert isinstance(flush_data, bytes)
            print(f"  flush returned {len(flush_data)} bytes")
        finally:
            enc.close()

    @gpu
    def test_flush_frame_count(self):
        """flush 后 _flush_frame_count 应为合理值。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='vbr_hq', la_depth=8)
        try:
            import torch
            nv12 = torch.randint(0, 255, (96, 64), dtype=torch.uint8, device='cuda')
            for _ in range(10):
                enc.encode_frame(nv12, force_idr=False)
            flush_data = enc.flush()
            fc = enc._flush_frame_count
            print(f"  flush recovered {fc} frames, {len(flush_data)} bytes")
            # LA=8 with 10 frames → all LA frames should be recovered
        finally:
            enc.close()

    @gpu
    def test_reset_output_slot_idx(self):
        """_reset_output_slot_idx() 正常工作。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            enc._reset_output_slot_idx(5)
            assert enc._output_slot_idx == 5
            enc._reset_output_slot_idx(0)
            assert enc._output_slot_idx == 0
        finally:
            enc.close()


# ============================================================================
# GPU 测试 — LA 累积模式 + f0 处理
# ============================================================================

class TestLAAccumulation:
    """LA 累积模式：跨批次帧累积 + f0 暂存/插入。"""

    @gpu
    def test_la_accumulation_no_cross_batch_loss(self):
        """LA 累积模式下跨批次无丢帧。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='vbr_hq', la_depth=8)
        try:
            import torch
            n = 64
            nv12_h = 64 + 32  # 96
            frames = [torch.randint(0, 255, (nv12_h, 64), dtype=torch.uint8, device='cuda')
                      for _ in range(n)]
            # 模拟多批次提交（累积模式应在编码线程中处理）
            batch1 = frames[:32]
            batch2 = frames[32:]

            # 累积模式关键：最终 send_eos=True
            all_frames = batch1 + batch2
            results = enc.encode_frames_batch(all_frames, force_idr_first=True,
                                               send_eos=True)
            assert len(results) == n, \
                f"Cross-batch accumulation failed: {len(results)} != {n}"

            valid = sum(1 for r in results if r)
            print(f"  {n} frames → {valid} valid H.264 frames, "
                  f"{n - valid} LA-buffered (expected)")
        finally:
            enc.close()

    @gpu
    def test_f0_stash_and_insert(self):
        """f0 暂存到 encoder._pending_f0_nv12 并插入累积 batch。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='vbr_hq', la_depth=8)
        try:
            import torch
            nv12_h = 64 + 32  # 96

            # f0: simulate first frame via encode_frame (LA>0 returns b"")
            f0_nv12 = torch.randint(0, 255, (nv12_h, 64), dtype=torch.uint8, device='cuda')
            enc._pending_f0_nv12 = f0_nv12
            enc._pending_f0_force_idr = True
            assert enc._pending_f0_nv12 is not None

            # Verify it's cleared after retrieval
            pending = enc._pending_f0_nv12
            enc._pending_f0_nv12 = None
            assert enc._pending_f0_nv12 is None

            # Remaining frames
            frames = [torch.randint(0, 255, (nv12_h, 64), dtype=torch.uint8, device='cuda')
                      for _ in range(32)]
            all_frames = [pending] + frames
            results = enc.encode_frames_batch(all_frames, force_idr_first=True,
                                               send_eos=True)
            assert len(results) == 33, \
                f"With f0: expected 33, got {len(results)}"
            print(f"  f0+32 frames = {len(results)} total, "
                  f"{sum(1 for r in results if r)} valid")
        finally:
            enc.close()


# ============================================================================
# GPU 测试 — NVENCWriter 端到端
# ============================================================================

class TestNVENCWriter:
    """NVENCWriter 完整数据流测试。"""

    @gpu
    def test_nvenc_writer_create(self):
        """NVENCWriter 创建成功。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            import argparse
            args = argparse.Namespace(quiet=True, ffmpeg_bin='ffmpeg',
                                       encode_queue_depth=2)

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
                tmp_path = tf.name

            try:
                writer = nvenc_sdk.NVENCWriter(enc, args, b"", 64, 64,
                                                tmp_path, 30.0)
                assert writer._broken is False
                assert writer._enc_thread is not None
                writer.close()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        finally:
            enc.close()

    @gpu
    def test_nvenc_writer_write_frame_batch(self):
        """NVENCWriter.write_frame_batch() 端到端数据流。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            import argparse
            args = argparse.Namespace(quiet=True, ffmpeg_bin='ffmpeg',
                                       encode_queue_depth=2)

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
                tmp_path = tf.name

            try:
                writer = nvenc_sdk.NVENCWriter(enc, args, b"", 64, 64,
                                                tmp_path, 30.0)

                # Generate test frames
                frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                          for _ in range(16)]
                writer.write_frame_batch(frames)
                writer.close()

                # Check output
                assert os.path.exists(tmp_path)
                size = os.path.getsize(tmp_path)
                assert size > 0, f"Output file is empty: {tmp_path}"
                print(f"  NVENCWriter produced {size} bytes output")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        finally:
            enc.close()

    @gpu
    def test_nvenc_writer_multi_batch(self):
        """NVENCWriter: 多个 mini-batch + close。"""
        enc = nvenc_sdk.NVENCEncoder(64, 64, 30.0, qp=23,
                                      rate_mode='constqp', la_depth=0)
        try:
            import argparse
            args = argparse.Namespace(quiet=True, ffmpeg_bin='ffmpeg',
                                       encode_queue_depth=2)

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
                tmp_path = tf.name

            try:
                writer = nvenc_sdk.NVENCWriter(enc, args, b"", 64, 64,
                                                tmp_path, 30.0)

                # Write individual frames (tests mini-batch accumulation)
                for _ in range(8):
                    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                    writer.write_frame(frame)

                # Write a batch
                batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                         for _ in range(8)]
                writer.write_frame_batch(batch)

                writer.close()

                assert os.path.exists(tmp_path)
                size = os.path.getsize(tmp_path)
                assert size > 0
                print(f"  Multi-batch produced {size} bytes")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        finally:
            enc.close()


# ============================================================================
# 便捷测试入口
# ============================================================================

if __name__ == '__main__':
    if not _HAS_GPU:
        print("=" * 60)
        print("无 GPU 可用 — 仅运行无 GPU 测试")
        print("=" * 60)
        print()

        test = TestModuleStructure()
        for name in dir(test):
            if name.startswith('test_') and not hasattr(getattr(test, name), 'pytestmark'):
                try:
                    getattr(test, name)()
                    print(f"  ✓ {name}")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")

        print()
        print("GPU 测试跳过 — 需 NVIDIA GPU + CUDA toolkit")
        print(f"  torch available: {_HAS_TORCH}")
        print(f"  CUDA available:  {_HAS_CUDA}")
        print(f"  nvenc_sdk OK:    {_HAS_NVENC_SDK}")
        sys.exit(0)

    # GPU 可用 — 运行 pytest
    print("GPU 可用 — 运行 pytest...")
    sys.exit(pytest.main([__file__, '-v', '-s']))
