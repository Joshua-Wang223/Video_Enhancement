#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - 人脸处理模块
包含：人脸检测、GFPGAN推理、人脸贴回辅助函数
"""

import sys
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

from facexlib.utils.face_restoration_helper import FaceRestoreHelper


def _make_detect_helper(face_enhancer, device):
    """
    创建独立的 FaceRestoreHelper 实例，专供后台 detect 线程使用。
    与主线程 face_enhancer.face_helper 互为独立对象，无共享状态，线程安全。
    FIX-DET-CPU: 强制在 CPU 上运行人脸检测。
    """
    upscale_factor = getattr(face_enhancer.face_helper, 'upscale_factor', 1)
    return FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=torch.device('cpu'),  # 强制 CPU
    )


def _detect_faces_batch(frames: List[np.ndarray], helper,
                        det_threshold: float = 0.5) -> Tuple[List[dict], int, int, int]:
    """
    在原始低分辨率帧上检测人脸，返回序列化检测结果。
    FIX-DET-THRESHOLD + FIX-PRECOMPUTE-INV-AFFINE: 在检测阶段预计算逆仿射矩阵。
    """
    face_data = []
    _total_filtered = 0
    for orig_frame in frames:
        helper.clean_all()
        bgr_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
        helper.read_image(bgr_frame)
        try:
            helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
        except TypeError:
            helper.get_face_landmarks_5(
                only_center_face=False, eye_dist_threshold=5)

        # FIX-DET-THRESHOLD: 按置信度阈值过滤低质量人脸检测
        if (det_threshold > 0 and
                hasattr(helper, 'det_faces') and
                helper.det_faces is not None and
                len(helper.det_faces) > 0):
            _before_count = len(helper.det_faces)
            keep_indices = []
            MIN_FACE_AREA = 48 * 48
            for _fi, _face in enumerate(helper.det_faces):
                x1, y1, x2, y2 = _face[:4]
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_FACE_AREA:
                    continue
                _score = float(_face[4]) if len(_face) > 4 else float(_face[-1])
                if _score >= det_threshold:
                    keep_indices.append(_fi)
            _after_count = len(keep_indices)
            if _after_count < _before_count:
                _total_filtered += _before_count - _after_count
                if hasattr(helper, 'all_landmarks_5') and helper.all_landmarks_5:
                    helper.all_landmarks_5 = [helper.all_landmarks_5[_ki]
                                              for _ki in keep_indices]
                helper.det_faces = [helper.det_faces[_ki] for _ki in keep_indices]

        helper.align_warp_face()

        # FIX-PRECOMPUTE-INV-AFFINE: 预计算逆仿射矩阵
        inv_affines = []
        _upscale = getattr(helper, 'upscale_factor', 1)
        for a in helper.affine_matrices:
            inv = cv2.invertAffineTransform(a)
            inv *= _upscale
            inv_affines.append(inv)

        face_data.append({
            'crops': [c.copy() for c in helper.cropped_faces],
            'affines': [a.copy() for a in helper.affine_matrices],
            'inv_affines': inv_affines,
            'orig': bgr_frame,
        })
    _nf = sum(len(fd['crops']) for fd in face_data)
    _fw = sum(1 for fd in face_data if fd['crops'])
    return face_data, _fw, _nf, _total_filtered


def _gfpgan_infer_batch(face_data, face_enhancer, device, fp16_ctx,
                        gfpgan_weight, sub_bs, gfpgan_trt_accel=None,
                        gfpgan_subprocess=None):
    """
    GFPGAN 批量推理 —— 直接对预检测 crops 做网络前向。
    不调用 face_enhancer.enhance(has_aligned=False)，避免覆盖预设数据。
    """
    import contextlib as _ctx
    if not face_data:
        return [], sub_bs

    try:
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
    except ImportError:
        restored_by_frame = []
        for fd in face_data:
            restored_by_frame.append([])
        return restored_by_frame, sub_bs

    _fp16 = fp16_ctx if fp16_ctx is not None else _ctx.nullcontext()

    restored_by_frame = []
    for fd in face_data:
        crops = fd.get('crops', [])
        if not crops:
            restored_by_frame.append([])
            continue

        all_restored = []
        i = 0
        _cur_sub_bs = sub_bs
        while i < len(crops):
            sub_crops = crops[i:i + _cur_sub_bs]
            tensors = []
            for crop in sub_crops:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                tensors.append(t)
            sub_batch = torch.stack(tensors).to(device)
            try:
                with torch.no_grad():
                    with _fp16:
                        out = face_enhancer.gfpgan(sub_batch, return_rgb=True, weight=gfpgan_weight)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                    out = out.float()

                for out_t in out.unbind(0):
                    if out_t is None:
                        all_restored.append(None)
                    else:
                        out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=-1.0)
                        out_t = torch.clamp(out_t, min=-1.0, max=1.0)
                        restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
                        all_restored.append(restored.astype('uint8'))

                i += len(sub_crops)
            except RuntimeError as e:
                _estr = str(e).lower()
                if 'out of memory' in _estr and _cur_sub_bs > 1:
                    _cur_sub_bs = max(1, _cur_sub_bs // 2)
                    sub_bs = _cur_sub_bs
                    torch.cuda.empty_cache()
                else:
                    all_restored.extend([None] * len(sub_crops))
                    i += len(sub_crops)
                    torch.cuda.empty_cache()
            finally:
                del sub_batch

        restored_by_frame.append(all_restored)

    return restored_by_frame, sub_bs


def _paste_faces_batch(face_data, restored_by_frame, sr_results, face_enhancer):
    """人脸贴回函数，使用预计算的逆仿射矩阵。"""
    expected_h, expected_w = sr_results[0].shape[:2]
    final_frames = []

    for fi, (fd, sr_frame) in enumerate(zip(face_data, sr_results)):
        if not restored_by_frame[fi]:
            final_frames.append(sr_frame)
            continue
        try:
            face_enhancer.face_helper.clean_all()
            sr_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
            face_enhancer.face_helper.read_image(fd['orig'])

            _raw = restored_by_frame[fi]
            _affines = fd['affines']
            _crops = fd['crops']
            _inv_affines = fd.get('inv_affines', [])
            _n = min(len(_raw), len(_affines), len(_crops))
            valid_pairs = [(rf, _affines[j], _crops[j],
                            _inv_affines[j] if j < len(_inv_affines) else None)
                           for j, rf in enumerate(_raw[:_n]) if rf is not None]
            if not valid_pairs:
                final_frames.append(sr_frame)
                continue

            valid_restored, valid_affines, valid_crops, valid_inv = zip(*valid_pairs)
            face_enhancer.face_helper.affine_matrices = list(valid_affines)
            face_enhancer.face_helper.cropped_faces = list(valid_crops)
            for rf in valid_restored:
                face_enhancer.face_helper.add_restored_face(rf)

            if all(v is not None for v in valid_inv):
                face_enhancer.face_helper.inverse_affine_matrices = list(valid_inv)
            else:
                face_enhancer.face_helper.get_inverse_affine(None)

            _ret = face_enhancer.face_helper.paste_faces_to_input_image(upsample_img=sr_bgr)

            result_bgr = _ret if _ret is not None else getattr(
                face_enhancer.face_helper, 'output', None)

            if result_bgr is not None:
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            else:
                result = sr_frame

        except Exception as e:
            print(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
            result = sr_frame

        if result.shape[0] != expected_h or result.shape[1] != expected_w:
            print(f'[WARN] face_enhance 帧{fi} 尺寸异常 '
                  f'{result.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
            result = cv2.resize(result, (expected_w, expected_h),
                                interpolation=cv2.INTER_LANCZOS4)
        final_frames.append(result)

    return final_frames