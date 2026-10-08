# T4 侧全流程验证立项

## 背景
CPU 侧 7/7 测试已通过，需在 Tesla T4 (Turing, sm75) 上完成 GPU 相关验收。

## 环境要求
- Tesla T4 (sm75) / 驱动 ≥535 / CUDA 12.x
- FFmpeg 9.0+ with h264_nvenc, hevc_nvenc
- PyTorch CUDA 可用
- 模型权重就位：`models_IFRNet/checkpoints/`, `models_RealESRGAN/`, `models_GFPGAN/`

## 验收范围（T4 独有）

| 测试项 | 脚本 | 关键判据 |
|--------|------|----------|
| plan_gate (完整) | `plan_implementation_gate.py` | 89 项 PASS/0 FAIL（含 BEH-G9 NVENC 段边界） |
| crf_cq --gpu | `crf_cq_unification_verify.py` | G7/G8 画质/码率天花板 FAIL=0 |
| nvenc_vbr_hq_verify | `nvenc_vbr_hq_verify.py` | CLI 拒绝 vbr_hq / SDK 接受 rc_ptr[1]=32 / ΔVMAF≤0.3 |
| nvenc_rc_diagnose | `nvenc_rc_mode_diagnose.py` | RC 枚举/驱动能力/选项表确认 |
| segment_bitstream_verify (hevc+LA=8) | `segment_bitstream_verify_v5.py` | 帧守恒/单 IDR/frame_num 单调/无色度簇 |
| av1_vp9_matrix_cpu | 复用 CPU 结果 | 软编 3 编码器 PASS |
| eqq_pool_fit (含 gpu_t4) | `eqq_pool_fit_table.py` | h264/hevc CQ/QP LOO ≤5.9 |

## 执行命令
```bash
cd /workspace/Video_Enhancement

# 0. 环境体检
nvidia-smi -L
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.is_available())"
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v $C -f null - < /dev/null; echo "$C rc=$?"
done
# 期望: h264=0, hevc=0, av1≠0

# 1. 完整验证（含冒烟）
python3 Accessory/verify/comprehensive_verify.py --env t4 \
    -i input_videos/word_world_2.mp4 -o output.mp4 \
    --source input_videos/word_world_2.mp4 \
    --bitrate-source input_videos/new4_raw.mp4 \
    --smoke --smoke-mode interpolate_then_upscale

# 2. 单独跑 NVENC 专项（如需）
python3 Accessory/probe/nvenc_vbr_hq_verify.py
python3 Accessory/probe/nvenc_rc_mode_diagnose.py $(find / -name nvEncodeAPI.h 2>/dev/null | head -1)

# 3. hevc+LA=8 帧守恒回归
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --codec-ifrnet hevc_nvenc --codec-esrgan hevc_nvenc \
    --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8 < /dev/null
python3 Accessory/verify/segment_bitstream_verify_v5.py /tmp/seg_hevc_la8.mp4 < /dev/null

# 4. 落表器含 GPU 侧
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_t4 < /dev/null
```

## 门禁基线（2026-10-04 T4 实测）
- `plan_implementation_gate`: 96/94/0/2
- `crf_cq --gpu`: PASS=101 / FAIL=0 / WARN=4 / SKIP=1
- `nvenc_vbr_hq_verify`: V12 接受 rc_ptr[1]=32 → 方案 A
- hevc+LA=8: frames=packets=199, 无色度簇

## 产出物
- `verification_report/verification_report_YYYYMMDD_HHMMSS.json/.md`
- `verification_report/CRF_CQ统一验证报告_YYYYMMDD_HHMMSS.md`
- `verification_report/av1_vp9_matrix_T4_YYYYMMDD.md`
- `verification_report/nvenc_vbrhq_migration_YYYYMMDD/`