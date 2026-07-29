#!/usr/bin/env python3
"""
NVENCWriter — FFmpegWriter 的 SDK Level 1 替代品

接口兼容 FFmpegWriter:
  - write_frame(frame: np.ndarray) -> bool
  - close()

内部数据流:
  RGB np.ndarray → torch GPU tensor → _rgb_to_nv12_gpu_batch()
  → _NVENCEncodeThread.submit() → NVENCEncoder.encode_frames_batch_ce_pipeline()
  → FFmpegMuxer.write(h264_es) → ffmpeg -f h264 -c:v copy → MP4

跨段复用:
  - NVENCEncoder 实例由外部传入（视频级生命周期）
  - FFmpegMuxer + _NVENCEncodeThread 每段新建（段级生命周期）
"""

import os
import sys
import threading
from typing import List, Optional

import numpy as np
import torch

# 从同目录导入 SDK 模块
from nvenc_sdk import (
    NVENCEncoder,
    _NVENCEncodeThread,
    FFmpegMuxer,
    _rgb_to_nv12_gpu,
    _rgb_to_nv12_gpu_batch,
)


class NVENCWriter:
    """SDK Level 1 NVENC 编码写入器 —— FFmpegWriter 替代品。

    接收 RGB numpy 帧，通过 GPU NV12 转换 + ctypes NVENC SDK 编码，
    最后经 FFmpegMuxer 纯 mux 到 MP4 容器。
    """

    def __init__(
        self,
        args,
        audio: Optional[dict],
        height: int,
        width: int,
        output_path: str,
        fps: float,
        nvenc_encoder: NVENCEncoder,
        audio_src: Optional[str] = None,
    ):
        self.args = args
        self.height = height
        self.width = width
        self.output_path = output_path
        self.fps = fps

        self._running = True
        self._broken = False
        self._error: Optional[Exception] = None
        self._written = 0
        self._quiet = getattr(args, 'quiet', True)

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ── 每段独立的 FFmpegMuxer ──
        self._muxer = FFmpegMuxer(
            output_path=self.output_path,
            fps=fps,
            audio_src=audio_src,
            ffmpeg_bin=getattr(args, 'ffmpeg_bin', 'ffmpeg'),
            quiet=self._quiet,
        )

        # ── 关联 NVENCEncoder 到本段 muxer ──
        nvenc_encoder.set_muxer_ref(self._muxer)
        # 重置跨段标志：SPS+PPS 已缓存（首段），但新 muxer 需要重新注入
        nvenc_encoder._sps_pps_injected = False

        # ── 每段独立的编码线程 ──
        # NVENC SDK 帧数守恒，每段独立编码+排空，LA 状态不可跨会话复用
        self._enc_thread = _NVENCEncodeThread(
            nvenc_encoder, self._muxer,
            encode_queue_depth=4,  # pipeline_depth 默认 4
        )

        if not self._quiet:
            print(f"[NVENCWriter] SDK Level 1 NVENC 编码器已就绪: "
                  f"{width}x{height}@{fps:.1f}fps", flush=True)

    def write_frame(self, frame: np.ndarray) -> bool:
        """写入单帧 RGB numpy (H, W, 3) uint8。"""
        if self._error is not None:
            raise RuntimeError(
                f"[NVENCWriter] 编码线程已崩溃: {self._error}") from self._error
        if self._broken:
            return False

        try:
            # RGB numpy → GPU tensor → NV12
            rgb_gpu = torch.from_numpy(frame).to('cuda', non_blocking=True)
            nv12_gpu = _rgb_to_nv12_gpu(rgb_gpu, input_is_bgr=False)

            # [FIX-SYNC-BEFORE-SUBMIT] nvenc_sdk.py 的 _NVENCEncodeThread
            # docstring 明确要求：T3 Writer 在 submit() 之前必须先
            # torch.cuda.current_stream().synchronize()，确保 NV12 GPU
            # tensor 数据已完全写入 VRAM，否则编码线程的 cuMemcpy2D 可能
            # 读到未写完的数据（静默花帧）。此前这一步在这里被遗漏——
            # 旧版编码线程用的是同步 cuMemcpy2D_v2(legacy null stream)，
            # 它自带的隐式全局同步"意外"替 Writer 补上了这次等待；
            # 迁移到 cuMemcpy2DAsync_v2 + 专用 stream（性能修复）后，
            # 这个"意外保护"不再存在，必须在此显式补回，否则会重新引入
            # PinnedBufferPool 文档描述过的那类"帧内容被更早批次顶替/
            # 花帧"数据竞态。
            torch.cuda.current_stream().synchronize()

            # 提交到编码线程
            self._enc_thread.submit([nv12_gpu], force_idr_first=(self._written == 0))
            self._written += 1
            return True
        except Exception as e:
            self._error = e
            self._broken = True
            raise

    def write_frame_batch(self, frames: List[np.ndarray]) -> bool:
        """批量写入多帧 RGB numpy 列表。每个元素为 (H, W, 3) uint8。"""
        if self._error is not None:
            raise RuntimeError(
                f"[NVENCWriter] 编码线程已崩溃: {self._error}") from self._error
        if self._broken:
            return False
        if not frames:
            return True

        try:
            # 批量 RGB numpy → GPU tensor → NV12
            rgb_batch = torch.from_numpy(np.stack(frames)).to('cuda', non_blocking=True)
            nv12_batch = _rgb_to_nv12_gpu_batch(rgb_batch, input_is_bgr=False)

            # [FIX-LA-D2H-HOST] LA>0 累积模式下将 NV12 主动搬迁到 pinned host
            # 内存。避免编码线程 _acc_nv12 累积 GPU tensor 导致 VRAM 持续占用
            # 与 SR 推理竞争显存带宽。cuMemcpy2D 已支持 host→GPU 路径。
            if getattr(self._enc_thread._nvenc, '_la_depth', 0) > 0:
                try:
                    nv12_host = torch.empty(nv12_batch.shape, dtype=nv12_batch.dtype,
                                             device='cpu', pin_memory=True)
                    nv12_host.copy_(nv12_batch, non_blocking=False)
                    nv12_batch = nv12_host
                except RuntimeError:
                    nv12_batch = nv12_batch.cpu()

            # 拆分为单帧列表提交编码线程
            # [FIX-SYNC-BEFORE-SUBMIT] LA=0（或 pinned host 拷贝失败退化为
            # .cpu() 分页内存）时 nv12_batch 仍是 GPU tensor，同样需要在
            # submit() 前显式同步 current stream（理由同 write_frame()）。
            # LA>0 且 pinned 拷贝成功的分支已经用阻塞的
            # nv12_host.copy_(nv12_batch, non_blocking=False) 间接完成了
            # 这个等待，此处按需跳过，避免重复同步。
            if nv12_batch.is_cuda:
                torch.cuda.current_stream().synchronize()
            nv12_list = [nv12_batch[i] for i in range(nv12_batch.shape[0])]
            self._enc_thread.submit(nv12_list, force_idr_first=(self._written == 0))
            self._written += len(frames)
            return True
        except Exception as e:
            self._error = e
            self._broken = True
            raise

    @property
    def frames_written(self) -> int:
        return self._written

    def begin_flush(self):
        """[EARLY-FLUSH] 段末提前触发 NVENC 编码收尾的公共入口。

        委托 _NVENCEncodeThread.begin_flush() 非阻塞发送 SENTINEL，
        使 GPU EOS drain 与后续 CPU 管线清理并行。
        异常不抛出、不中断 pipeline 清理流程。
        幂等：若已调用则 no-op；close() 中 flush_and_join 自动跳过重复 put。
        """
        if not self._running:
            return
        if self._error is not None:
            return
        try:
            self._enc_thread.begin_flush()
        except Exception as e:
            if self._error is None:
                self._error = e
            print(f"[NVENCWriter] begin_flush 异常: {e}", flush=True)

    def close(self):
        """关闭写入器：flush 编码线程 → 关闭 muxer。"""
        if not self._running:
            return
        self._running = False

        try:
            if self._error is None:
                # flush_and_join 在编码线程内执行 NVENC EOS flush 后
                # join 线程，确保所有已提交帧已写入 muxer
                written, empty = self._enc_thread.flush_and_join()
                if not self._quiet and empty > 0:
                    print(f"[NVENCWriter] 编码完成: {written} 帧写入, "
                          f"{empty} 空帧(已补偿)", flush=True)
        except Exception as e:
            if self._error is None:
                self._error = e
            print(f"[NVENCWriter] 编码线程关闭异常: {e}", flush=True)
        finally:
            self._muxer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
